# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import discord

# from discord.ext import commands
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


class DiscordHandler:
    """Handles Discord client operations and connection management."""

    def __init__(self, bot_token: str, guild_name: str, channel_name: str):
        self._bot_token = bot_token
        self._guild_name = guild_name
        self._channel_name = channel_name

        self._client = None
        self._guild = None
        self._channel = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._thread = None
        self._loop = None
        self._error = None  # Track initialisation errors

        # Set up the client with proper intents
        self._setup_client()

    def _setup_client(self):
        """Set up the Discord client with proper intents."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True

        self._client = discord.Client(intents=intents)
        self._setup_events()

    def _setup_events(self):
        """Set up Discord client event handlers."""

        @self._client.event
        async def on_ready():
            try:
                # Validate guild
                self._guild = discord.utils.get(self._client.guilds, name=self._guild_name)
                if not self._guild:
                    self._error = DiscordChannelError(
                        message=f"Could not find guild: {self._guild_name}", platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()  # Set ready to unblock validation
                    return

                # Validate channel
                self._channel = discord.utils.get(self._guild.text_channels, name=self._channel_name)
                if not self._channel:
                    self._error = DiscordChannelError(
                        message=f"Could not find channel: {self._channel_name}", platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()  # Set ready to unblock validation
                    return

                # Signal that we're ready
                self._ready.set()

            except Exception as e:
                self._error = e
                self._ready.set()  # Set ready to unblock validation

    async def validate(self):
        """Wait for validation to complete and return result."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30)
            if self._error:
                # Catch initialisation exceptions, incorrect token, guild, or channel.
                raise PlatformError(
                    message="Error initialising Discord client",
                    platform_error=self._error,
                    platform_name=__PLATFORM_NAME__,
                )
            return True
        except asyncio.TimeoutError:
            raise PlatformTimeoutError(
                message="Timeout waiting for Discord validation", platform_name=__PLATFORM_NAME__
            )

    def start(self):
        """Start the Discord client and wait for validation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._loop = loop
                asyncio.create_task(self._client.start(self._bot_token))
            else:
                self._start_background_thread()
        except RuntimeError:
            self._start_background_thread()

        # Wait for validation in appropriate context
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.validate(), self._loop)
            return future.result(timeout=30)
        else:
            # For non-async context, we need to use a new event loop
            # since the client is running in its own loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.validate())
            finally:
                loop.close()

    def _start_background_thread(self):
        """Start Discord client in a background thread."""

        def run_client():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def start_client():
                try:
                    await self._client.start(self._bot_token)
                except discord.LoginFailure as e:
                    self._error = PlatformAuthenticationError(
                        message="Failed to login with provided token", platform_error=e, platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()  # Set ready to trigger validation check
                except Exception as e:
                    self._error = PlatformError(
                        message=f"Error during client initialization: {str(e)}",
                        platform_error=e,
                        platform_name=__PLATFORM_NAME__,
                    )
                    self._ready.set()

            try:
                self._loop.run_until_complete(start_client())
                self._loop.run_forever()
            except Exception as e:
                if not isinstance(self._error, (PlatformAuthenticationError, PlatformError)):
                    self._error = PlatformError(
                        message=f"Error in client thread: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()
            finally:
                if not self._client.is_closed():
                    self._loop.run_until_complete(self._client.close())
                self._loop.close()

        self._thread = threading.Thread(target=run_client, daemon=True)
        self._thread.start()

        # Give the thread a second to start
        time.sleep(1)

    async def send_message(self, message: str) -> str:
        """Send a message to the Discord channel."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=5)
        except asyncio.TimeoutError:
            raise PlatformTimeoutError(
                message="Timeout waiting for Discord client to be ready", platform_name=__PLATFORM_NAME__
            )

        try:
            if len(message) > 2000:
                chunks = [message[i : i + 1999] for i in range(0, len(message), 1999)]
                for chunk in chunks:
                    await self._channel.send(chunk)
                return "Message sent (split into chunks)"
            else:
                await self._channel.send(message)
                return "Message sent successfully"
        except discord.Forbidden as e:
            raise PlatformAuthenticationError(
                message="Bot lacks permission to send messages", platform_error=e, platform_name=__PLATFORM_NAME__
            )
        except discord.HTTPException as e:
            if e.status == 429:
                raise PlatformRateLimitError(
                    message="Rate limit exceeded",
                    platform_error=e,
                    platform_name=__PLATFORM_NAME__,
                    retry_after=e.retry_after,
                )
            raise PlatformConnectionError(
                message=f"Failed to send message: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    def shutdown(self):
        """Shutdown the Discord client."""
        if self._client and not self._client.is_closed():
            if self._loop and self._loop.is_running():
                self._loop.create_task(self._client.close())
            else:
                asyncio.run(self._client.close())

        if self._thread and self._thread.is_alive():
            self._shutdown.set()
            self._thread.join(timeout=2)


class DiscordExecutor(PlatformExecutorAgent):
    """Discord-specific executor agent."""

    def __init__(self, platform_config: DiscordConfig, reply_config: Optional[ReplyConfig] = None):
        super().__init__(platform_config, reply_config)

        # Create Discord handler
        self._discord = DiscordHandler(
            bot_token=platform_config.bot_token,
            guild_name=platform_config.guild_name,
            channel_name=platform_config.channel_name,
        )

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start Discord client
        self._discord.start()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        """Clean shutdown of the Discord client."""
        self._discord.shutdown()

    def send_to_platform(self, message: str) -> str:
        """Send a message to Discord channel."""
        # Get the event loop
        loop = self._discord._loop

        # If we're in the event loop thread, run directly
        if loop and loop.is_running() and threading.current_thread() == self._discord._thread:
            return loop.run_until_complete(self._discord.send_message(message))

        # Otherwise, run in the appropriate context
        future = asyncio.run_coroutine_threadsafe(self._discord.send_message(message), loop)
        result = future.result(timeout=5)

        # Initiate shutdown after sending message
        self.shutdown()

        return result

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.shutdown()


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
