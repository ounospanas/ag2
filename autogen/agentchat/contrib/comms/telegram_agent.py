# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import telegram
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
)

from .comms_platform_agent import (
    CommsPlatformAgent,
    PlatformExecutorAgent,
)
from .platform_configs import ReplyMonitorConfig, TelegramConfig
from .platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
)

__PLATFORM_NAME__ = "Telegram"


class TelegramHandler:
    """Handles Telegram client operations using Application pattern."""

    def __init__(self, config: TelegramConfig):
        self._config = config
        self._app: Optional[Application] = None
        self._message_replies: Dict[str, List[dict]] = {}
        self._reply_events: Dict[str, asyncio.Event] = {}

    async def initialize(self) -> None:
        """Initialize the Telegram application."""
        try:
            self._app = (
                ApplicationBuilder()
                .token(self._config.bot_token)
                .connection_pool_size(8)  # Increase connection pool
                .get_updates_connection_pool_size(8)  # Dedicated pool for updates
                .pool_timeout(30.0)  # Longer pool timeout
                .build()
            )

            # Start the application
            await self._app.initialize()
            await self._app.start()

            # Verify chat access
            try:
                await self._app.bot.get_chat(self._config.destination_id)
            except telegram.error.Forbidden:
                raise PlatformConnectionError(
                    message=f"Could not access chat: {self._config.destination_id}", platform_name=__PLATFORM_NAME__
                )

        except telegram.error.InvalidToken as e:
            raise PlatformAuthenticationError(
                message="Invalid bot token", platform_error=e, platform_name=__PLATFORM_NAME__
            )
        except Exception as e:
            raise PlatformError(
                message=f"Error initializing Telegram: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    async def _handle_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming reply messages."""
        # print(f"Received update: {update}")  # Debug print

        if not update.message and not update.channel_post:
            return

        # Handle both regular messages and channel posts
        msg = update.message or update.channel_post
        if not msg or not msg.reply_to_message:
            return

        reply_to_id = str(msg.reply_to_message.message_id)
        # print(f"Found reply to message {reply_to_id}")  # Debug print

        if reply_to_id in self._message_replies:
            reply_data = {
                "content": msg.text or "",
                "author": msg.from_user.username or str(msg.from_user.id) if msg.from_user else "Channel",
                "timestamp": msg.date.isoformat(),
                "id": str(msg.message_id),
            }

            # Handle media attachments
            if msg.photo:
                reply_data["content"] += " (Attachment: photo)"
            elif msg.document:
                reply_data["content"] += f" (Attachment: document - {msg.document.file_name})"
            elif msg.video:
                reply_data["content"] += " (Attachment: video)"
            elif msg.audio:
                reply_data["content"] += " (Attachment: audio)"
            elif msg.voice:
                reply_data["content"] += " (Attachment: voice)"

            # print(f"Adding reply: {reply_data}")  # Debug print
            self._message_replies[reply_to_id].append(reply_data)

            # Set event if we're waiting for this reply
            if reply_to_id in self._reply_events:
                self._reply_events[reply_to_id].set()

    def start(self) -> bool:
        """Start the Telegram client and validate connection."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.initialize())
            return True
        except Exception as e:
            raise PlatformError(
                message=f"Error starting Telegram client: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    async def send_message(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to the Telegram chat."""
        if not self._app:
            await self.initialize()

        try:
            # Split message if it exceeds Telegram's limit (4096 characters)
            if len(message) > 4096:
                chunks = [message[i : i + 4095] for i in range(0, len(message), 4095)]
                first_message = None

                for chunk in chunks:
                    sent_message = await self._app.bot.send_message(
                        chat_id=self._config.destination_id,
                        text=chunk,
                        parse_mode="HTML",
                        reply_to_message_id=first_message.message_id if first_message else None,
                    )
                    if not first_message:
                        first_message = sent_message

                return "Message sent (split into chunks)", str(first_message.message_id)
            else:
                sent_message = await self._app.bot.send_message(
                    chat_id=self._config.destination_id, text=message, parse_mode="HTML"
                )
                return "Message sent successfully", str(sent_message.message_id)

        except Exception as e:
            raise PlatformError(
                message=f"Failed to send message: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    async def wait_for_replies(self, message_id: str, timeout_minutes: int = 5, max_replies: int = 0) -> List[dict]:
        """Wait for replies to a specific message."""
        if not message_id:
            return []

        if not self._app:
            await self.initialize()

        # print(f"Starting to wait for replies to message {message_id}")

        # Initialize tracking for this message
        self._message_replies[message_id] = []
        start_time = datetime.now()
        timeout_seconds = timeout_minutes * 60 if timeout_minutes > 0 else float("inf")

        try:
            # Start polling
            offset = 0
            while (datetime.now() - start_time).total_seconds() < timeout_seconds:
                try:
                    # Get updates with detailed reply info
                    updates = await self._app.bot.get_updates(
                        offset=offset,
                        timeout=1,
                        allowed_updates=["message", "channel_post"],
                        write_timeout=20,
                    )

                    for update in updates:
                        # print(f"Got update: {update.to_dict()}")
                        if update.update_id >= offset:
                            offset = update.update_id + 1

                        msg = None
                        if update.message:
                            msg = update.message
                        elif update.channel_post:
                            msg = update.channel_post

                        if not msg:
                            continue

                        # Check if this is a reply to our message
                        if msg.reply_to_message and str(msg.reply_to_message.message_id) == message_id:
                            # print(f"Found reply to message {message_id}")
                            reply_data = {
                                "content": msg.text or "",
                                "author": msg.from_user.username if msg.from_user else "Channel",
                                "timestamp": msg.date.isoformat(),
                                "id": str(msg.message_id),
                            }

                            # Handle media attachments
                            if msg.photo:
                                reply_data["content"] += " (Attachment: photo)"
                            elif msg.document:
                                reply_data["content"] += f" (Attachment: document - {msg.document.file_name})"
                            elif msg.video:
                                reply_data["content"] += " (Attachment: video)"
                            elif msg.audio:
                                reply_data["content"] += " (Attachment: audio)"
                            elif msg.voice:
                                reply_data["content"] += " (Attachment: voice)"

                            # print(f"Adding reply: {reply_data}")
                            self._message_replies[message_id].append(reply_data)

                            # Check if we've hit max replies
                            if max_replies > 0 and len(self._message_replies[message_id]) >= max_replies:
                                # print(f"Got max replies ({max_replies}), returning")
                                return self._message_replies[message_id]

                except Exception:
                    pass
                    # print(f"Error during polling: {e}")

                await asyncio.sleep(1)  # Small delay between polls

            # print(f"Timeout reached after {timeout_minutes} minutes")

        finally:
            # print(f"Returning {len(self._message_replies.get(message_id, []))} replies")
            return self._message_replies.pop(message_id, [])

    async def cleanup(self):
        """Cleanup monitoring resources without shutting down app."""
        # print("Cleaning up monitoring resources...")
        try:
            self._message_replies.clear()
            self._last_update_id = 0
            # print("Cleanup successful")
        except Exception:
            # print(f"Error during cleanup: {e}")
            pass

    def cleanup_reply_monitoring(self, message_id: str):
        """Clean up reply monitoring for a specific message."""
        if message_id:
            self._message_replies.pop(message_id, None)


class TelegramExecutor(PlatformExecutorAgent):
    """Telegram-specific executor agent."""

    def __init__(self, platform_config: TelegramConfig, reply_monitor_config: Optional[ReplyMonitorConfig] = None):
        super().__init__(platform_config, reply_monitor_config)
        self._telegram = TelegramHandler(platform_config)
        self._telegram.start()

    def send_to_platform(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to Telegram chat."""
        need_new_loop = False
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            need_new_loop = True
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Make sure app is initialized
            if not self._telegram._app:
                loop.run_until_complete(self._telegram.initialize())

            return loop.run_until_complete(self._telegram.send_message(message))
        finally:
            if need_new_loop and loop and not loop.is_closed():
                loop.close()

    def _format_replies(self, replies: List[dict]) -> List[str]:
        """Format replies for display."""
        formatted_replies = []
        for reply in replies:
            timestamp = datetime.fromisoformat(reply["timestamp"])
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M UTC")
            formatted_reply = f"[{formatted_time}] {reply['author']}: {reply['content']}"
            formatted_replies.append(formatted_reply)
        return formatted_replies

    def wait_for_reply(self, msg_id: str) -> List[str]:
        """Wait for reply from Telegram."""
        if not self.reply_monitor_config:
            return []

        need_new_loop = False
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            need_new_loop = True
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            replies = loop.run_until_complete(
                self._telegram.wait_for_replies(
                    msg_id,
                    timeout_minutes=self.reply_monitor_config.timeout_minutes,
                    max_replies=self.reply_monitor_config.max_reply_messages,
                )
            )
            return self._format_replies(replies)
        finally:
            if need_new_loop and not loop.is_closed():
                loop.close()

    def cleanup_monitoring(self, msg_id: str):
        """Clean up reply monitoring for a specific message."""
        if self._telegram and msg_id:
            self._telegram.cleanup_reply_monitoring(msg_id)
            if not self._telegram._message_replies:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self._telegram.cleanup())


class TelegramAgent(CommsPlatformAgent):
    """Agent for Telegram communication."""

    def __init__(
        self,
        name: str,
        platform_config: TelegramConfig,
        send_config: dict,
        message_to_send: Optional[callable] = None,
        reply_monitor_config: Optional[ReplyMonitorConfig] = None,
        auto_reply: str = "Message sent to Telegram",
        system_message: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that communicates through Telegram. "
                "Remember that Telegram uses HTML for formatting and has message length limits. "
                "Keep messages clear and concise, and consider using appropriate formatting when helpful."
            )

        # Create Telegram-specific executor
        telegram_executor = TelegramExecutor(platform_config, reply_monitor_config)

        super().__init__(
            name=name,
            platform_config=platform_config,
            executor_agent=telegram_executor,
            send_config=send_config,
            message_to_send=message_to_send,
            reply_monitor_config=reply_monitor_config,
            auto_reply=auto_reply,
            system_message=system_message,
            *args,
            **kwargs,
        )
