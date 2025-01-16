# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import dataclass
from typing import Optional

import discord

# from discord.ext import commands
from ....io.base import IOStream
from .comms_platform_agent import (
    BasePlatformConfig,
    CommsPlatformAgent,
    PlatformExecutorAgent,
    ReplyConfig,
)
from .platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

__PLATFORM_NAME__ = "Discord"


# Discord-specific errors
class DiscordChannelError(PlatformError):
    """Raised when there's an error with Discord channels."""

    pass


@dataclass
class DiscordConfig(BasePlatformConfig):
    """Discord-specific configuration."""

    bot_token: str
    channel_name: str
    guild_name: Optional[str] = None

    @property
    def intents(self) -> discord.Intents:
        """Get the required Discord intents required to send a message and check for messages."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        return intents

    def validate_config(self) -> bool:
        """Validate Discord configuration."""
        if not all([self.bot_token, self.guild_name, self.channel_name]):
            raise ValueError("bot_token, guild_name, and channel_name are required")
        return True


class DiscordExecutor(PlatformExecutorAgent):
    """Discord-specific executor agent."""

    def __init__(self, platform_config: DiscordConfig, reply_config: Optional[ReplyConfig] = None):
        super().__init__(platform_config, reply_config)

        # Initialize Discord client
        self.platform_client = discord.Client(intents=platform_config.intents)
        self.guild = None
        self.channel = None
        self._ready = asyncio.Event()
        self._message_queue = asyncio.Queue()
        self.message_futures = {}

        @self.platform_client.event
        async def on_ready():
            await self._initialize_channel()
            self._ready.set()

        @self.platform_client.event
        async def on_message(message):
            if message.author == self.platform_client.user:
                return

            # Check if we're waiting for a reply to this message
            parent_id = str(message.reference.message_id if message.reference else None)
            if parent_id in self.message_futures:
                future = self.message_futures[parent_id]
                if not future.done():
                    future.set_result(message)

            await self._message_queue.put(message)

        # Start client in background
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start Discord client in the background."""
        iostream = IOStream.get_default()

        if not self.platform_client:
            raise PlatformError(
                message=f"{__PLATFORM_NAME__} client not properly initialized", platform_name=__PLATFORM_NAME__
            )

        try:
            # Try to get the existing event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there isn't one, create a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def start_client():
            try:
                await self.platform_client.start(self.platform_config.bot_token)
            except discord.LoginFailure as e:
                iostream.send(f"Failed to login to {__PLATFORM_NAME__}: Invalid token or permissions")
                # Close the client
                await self.platform_client.close()
                raise PlatformAuthenticationError(
                    message="Failed to login with provided token", platform_error=e, platform_name=__PLATFORM_NAME__
                )
            except discord.HTTPException as e:
                iostream.send(f"HTTP Error connecting to {__PLATFORM_NAME__}: {e}")
                await self.platform_client.close()
                raise PlatformConnectionError(
                    message=f"Failed to connect to {__PLATFORM_NAME__}: {str(e)}",
                    platform_error=e,
                    platform_name=__PLATFORM_NAME__,
                )
            except Exception as e:
                iostream.send(f"Unexpected error starting Discord client: {e}")
                await self.platform_client.close()
                raise PlatformError(
                    message=f"Unexpected error: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
                )
            finally:
                self._ready.set()  # Ensure the ready event is set so nothing waits forever

        def done_callback(future):
            try:
                future.result()
            except Exception as e:
                iostream.send(f"{__PLATFORM_NAME__} Background task failed: {e}")
                if hasattr(loop, "is_running") and loop.is_running():
                    loop.stop()

        # Create and run the task
        task = loop.create_task(start_client())
        task.add_done_callback(done_callback)

    async def _initialize_channel(self):
        """Initialize guild and channel objects."""
        try:
            self.guild = discord.utils.get(self.platform_client.guilds, name=self.platform_config.guild_name)
            if not self.guild:
                self._last_init_error = DiscordChannelError(
                    message=f"Could not find guild: {self.platform_config.guild_name}",
                    platform_name=__PLATFORM_NAME__,
                    context={"guild_name": self.platform_config.guild_name},
                )
                raise self._last_init_error

            self.channel = discord.utils.get(self.guild.text_channels, name=self.platform_config.channel_name)
            if not self.channel:
                self._last_init_error = DiscordChannelError(
                    message=f"Could not find channel: {self.platform_config.channel_name}",
                    platform_name=__PLATFORM_NAME__,
                    context={
                        "guild_name": self.platform_config.guild_name,
                        "channel_name": self.platform_config.channel_name,
                    },
                )
                raise self._last_init_error

            # If we got here, initialization was successful
            self._ready.set()

        except Exception as e:
            # Store any other unexpected errors
            if not hasattr(self, "_last_init_error"):
                self._last_init_error = e
            # Make sure we set ready event even on failure
            self._ready.set()
            # Re-raise the exception
            raise

    async def _wait_for_reply(self, msg_id: str) -> str:
        """Wait for reply to specific message."""
        if not self.reply_config:
            return None

        try:
            # Create future for this message
            future = asyncio.Future()
            self.message_futures[msg_id] = future

            # Wait for response with timeout
            timeout = self.reply_config.timeout_minutes * 60
            response = await asyncio.wait_for(future, timeout)

            # Process response
            return response.content

        except asyncio.TimeoutError as e:
            raise PlatformTimeoutError(
                message="No reply received within timeout period", platform_error=e, platform_name=__PLATFORM_NAME__
            )
        finally:
            # Clean up
            self.message_futures.pop(msg_id, None)

    async def _send_to_platform(self, message: str) -> str:
        """Send a message to Discord channel."""
        try:
            if not self.channel:
                await self._ready.wait()  # Wait for client to be ready and then recheck if we have a channel
                if not self.channel or not self.guild:
                    if hasattr(self, "_last_init_error"):
                        raise self._last_init_error
                    raise PlatformError(
                        message="Discord channel not initialized. Check token, guild, and channel name.",
                        platform_name=__PLATFORM_NAME__,
                    )

            # Split message if it exceeds Discord's limit
            if len(message) > 2000:
                chunks = [message[i : i + 1999] for i in range(0, len(message), 1999)]
                for chunk in chunks:
                    await self.channel.send(chunk)
                return "Message sent (split into chunks)"

            await self.channel.send(message)
            return "Message sent"

        except discord.Forbidden as e:
            raise PlatformAuthenticationError(
                message="Bot lacks permission to send messages", platform_error=e, platform_name=__PLATFORM_NAME__
            )
        except discord.HTTPException as e:
            if e.status == 429:  # Rate limit
                raise PlatformRateLimitError(
                    message="Rate limit exceeded",
                    platform_error=e,
                    platform_name=__PLATFORM_NAME__,
                    retry_after=e.retry_after,
                )
            raise PlatformConnectionError(
                message=f"Failed to send message: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )
        except discord.DiscordException as e:
            raise PlatformError(
                message=f"Discord exception: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )
        except PlatformError as e:
            # It's one of our handled errors, just raise it as is
            raise e
        except Exception as e:
            # Other exception, wrap with ours and raise
            raise PlatformError(
                message=f"Unexpected error sending message: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )


class DiscordAgent(CommsPlatformAgent):
    """Agent for Discord communication."""

    def __init__(
        self,
        name: str,
        platform_config: DiscordConfig,
        send_config: dict,
        message_to_send: Optional[
            callable
        ] = None,  # The function to determine the message to send, returns None to indicate do not send a message, otherwise determined automatically
        reply_config: Optional[ReplyConfig] = None,
        auto_reply: str = "Message sent to Discord",
        system_message: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that communicates through Discord. "
                "Remember that Discord uses Markdown for formatting and has a character limit. "
                "Keep messages clear and concise, and consider using appropriate formatting when helpful."
            )

        # Create Discord-specific executor
        discord_executor = DiscordExecutor(platform_config, reply_config)

        super().__init__(
            name=name,
            platform_config=platform_config,
            executor_agent=discord_executor,
            send_config=send_config,
            message_to_send=message_to_send,
            reply_config=reply_config,
            auto_reply=auto_reply,
            system_message=system_message,
            *args,
            **kwargs,
        )

        # Update decision agent's knowledge of Discord specifics
        self.decision_agent.update_system_message(
            self.decision_agent.system_message
            + "\n"
            + "Format guidelines for Discord:\n"
            + "1. Max message length: 2000 characters\n"
            + "2. Supports Markdown formatting\n"
            + "3. Can use ** for bold, * for italic, ``` for code blocks\n"
            + "4. Consider using appropriate emojis when suitable"
        )
