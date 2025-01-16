# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from dataclasses import dataclass
from typing import Optional

import discord
from discord.ext import commands

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


# Discord-specific errors
class DiscordChannelError(PlatformError):
    """Raised when there's an error with Discord channels."""

    pass


@dataclass
class DiscordConfig(BasePlatformConfig):
    """Discord-specific configuration."""

    bot_token: str
    channel_id: str
    guild_id: Optional[str] = None
    application_id: Optional[str] = None
    intents: discord.Intents = discord.Intents.default()
    command_prefix: str = "!"

    def validate_config(self) -> bool:
        """Validate Discord configuration."""
        if not self.bot_token or not self.channel_id:
            raise ValueError("bot_token and channel_id are required")
        return True


class DiscordExecutor(PlatformExecutorAgent):
    """Discord-specific executor agent."""

    def __init__(self, platform_config: DiscordConfig, reply_config: Optional[ReplyConfig] = None):
        super().__init__(platform_config, reply_config)

        # Initialize Discord bot
        self.bot = commands.Bot(command_prefix=platform_config.command_prefix, intents=platform_config.intents)
        self.bot.executor = self  # Save reference to executor for event handlers
        self.message_futures = {}  # Store futures for reply waiting

        # Set up event handlers
        @self.bot.event
        async def on_ready():
            print(f"Discord bot logged in as {self.bot.user}")

        @self.bot.event
        async def on_message(message):
            # Ignore messages from self
            if message.author == self.bot.user:
                return

            # Check if we're waiting for a reply to this message
            parent_id = str(message.reference.message_id if message.reference else None)
            if parent_id in self.message_futures:
                future = self.message_futures[parent_id]
                if not future.done():
                    future.set_result(message)

        # Start bot in background
        self._start_bot()

    def _start_bot(self):
        """Start the Discord bot in the background."""
        loop = asyncio.get_event_loop()
        loop.create_task(self.bot.start(self.platform_config.bot_token))

    async def _get_channel(self) -> discord.TextChannel:
        """Get the Discord channel.

        Returns:
            discord.TextChannel: The Discord channel object.

        Raises:
            DiscordChannelError: If channel not found or inaccessible.
            PlatformConnectionError: If connection to Discord fails.
            PlatformAuthenticationError: If bot lacks permissions.
        """
        try:
            # First try getting from cache
            channel = self.bot.get_channel(int(self.platform_config.channel_id))

            if not channel:
                # If not in cache, fetch from API
                channel = await self.bot.fetch_channel(int(self.platform_config.channel_id))

            # Verify it's a text channel
            if not isinstance(channel, discord.TextChannel):
                raise DiscordChannelError(
                    message="Channel must be a text channel",
                    platform_name="Discord",
                    context={"channel_id": self.platform_config.channel_id, "channel_type": type(channel).__name__},
                )

            return channel

        except discord.NotFound as e:
            raise DiscordChannelError(
                message=f"Channel {self.platform_config.channel_id} not found",
                platform_error=e,
                platform_name="Discord",
                context={"channel_id": self.platform_config.channel_id},
            )

        except discord.Forbidden as e:
            raise PlatformAuthenticationError(
                message="Bot lacks permission to access channel",
                platform_error=e,
                platform_name="Discord",
                context={"channel_id": self.platform_config.channel_id},
            )

        except discord.HTTPException as e:
            raise PlatformConnectionError(
                message="Failed to connect to Discord",
                platform_error=e,
                platform_name="Discord",
                context={"channel_id": self.platform_config.channel_id},
            )

        except ValueError as e:
            # This would happen if channel_id can't be converted to int
            raise DiscordChannelError(
                message="Invalid channel ID format",
                platform_error=e,
                platform_name="Discord",
                context={"channel_id": self.platform_config.channel_id},
            )

    async def _send_to_platform(self, message: str) -> str:
        """Send message to Discord channel."""
        try:
            channel = await self._get_channel()
            discord_message = await channel.send(message)
            return str(discord_message.id)

        except discord.Forbidden as e:
            raise PlatformAuthenticationError(
                message="Bot lacks permission to send messages",
                platform_error=e,
                platform_name="Discord",
                context={"channel_id": self.platform_config.channel_id},
            )

        except discord.HTTPException as e:
            if e.status == 429:  # Rate limit
                raise PlatformRateLimitError(
                    message="Rate limit exceeded", platform_error=e, platform_name="Discord", retry_after=e.retry_after
                )
            raise PlatformConnectionError(message="Failed to send message", platform_error=e, platform_name="Discord")

        except asyncio.TimeoutError as e:
            raise PlatformTimeoutError(message="Operation timed out", platform_error=e, platform_name="Discord")

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
                message="No reply received within timeout period", platform_error=e, platform_name="Discord"
            )
        finally:
            # Clean up
            self.message_futures.pop(msg_id, None)


class DiscordAgent(CommsPlatformAgent):
    """Agent for Discord communication."""

    def __init__(
        self,
        name: str,
        platform_config: DiscordConfig,
        send_config: dict,
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
            send_config=send_config,
            executor_agent=discord_executor,
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
