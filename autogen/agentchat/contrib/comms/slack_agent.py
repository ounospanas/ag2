# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import signal
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse

from .comms_platform_agent import (
    CommsPlatformAgent,
    PlatformExecutorAgent,
)
from .platform_configs import ReplyMonitorConfig, SlackConfig
from .platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
    PlatformRateLimitError,
)

__PLATFORM_NAME__ = "Slack"  # Platform name for messages
__TIMEOUT__ = 5  # Timeout in seconds


class SlackHandler:
    """Handles Slack client operations and connection management."""

    def __init__(self, config: SlackConfig):
        self._config = config
        self._web_client = WebClient(token=config.bot_token)
        self._socket_client = (
            None if not config.app_token else SocketModeClient(app_token=config.app_token, web_client=self._web_client)
        )

        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._thread = None
        self._loop = None
        self._error = None
        self._message_replies = {}
        self._reply_events = {}

        # Initialize socket mode if available
        if self._socket_client:
            self._setup_socket_mode()

    def _setup_socket_mode(self):
        """Set up Socket Mode event handlers."""

        @self._socket_client.socket_mode_request_listeners.append
        def handle_message(client: SocketModeClient, req):
            # Acknowledge the request
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)

            if req.payload.get("type") == "message" and "thread_ts" in req.payload:
                thread_ts = req.payload["thread_ts"]
                if thread_ts in self._message_replies:
                    reply_data = {
                        "content": req.payload.get("text", ""),
                        "author": self._get_user_name(req.payload.get("user")),
                        "timestamp": datetime.fromtimestamp(float(req.payload.get("ts", 0))).isoformat(),
                        "id": req.payload.get("ts"),
                    }

                    # Handle attachments
                    if "files" in req.payload:
                        attachments = [
                            f"(Attachment: {f.get('name')}, URL: {f.get('url_private')})" for f in req.payload["files"]
                        ]
                        if attachments:
                            reply_data["content"] += f" {' '.join(attachments)}"

                    self._message_replies[thread_ts].append(reply_data)

                    # Check if we've hit max replies
                    max_replies = self._reply_events[thread_ts].get("max_replies", 0)
                    if max_replies > 0 and len(self._message_replies[thread_ts]) >= max_replies:
                        self._reply_events[thread_ts]["event"].set()

    def _get_user_name(self, user_id: str) -> str:
        """Get user name from user ID."""
        try:
            response = self._web_client.users_info(user=user_id)
            if response["ok"]:
                return response["user"]["real_name"]
        except SlackApiError:
            pass
        return user_id

    async def validate(self):
        """Validate Slack configuration and connection."""
        try:
            # Test API connection
            response = self._web_client.auth_test()
            if not response["ok"]:
                raise PlatformAuthenticationError(
                    message="Failed to authenticate with Slack", platform_name=__PLATFORM_NAME__
                )

            # Verify channel access
            try:
                response = self._web_client.conversations_info(channel=self._config.channel_id)
                if not response["ok"]:
                    raise PlatformError(
                        message=f"Could not access channel: {self._config.channel_id}", platform_name=__PLATFORM_NAME__
                    )
            except SlackApiError as e:
                if e.response["error"] == "channel_not_found":
                    raise PlatformConnectionError(
                        message=f"Channel not found: {self._config.channel_id}", platform_name=__PLATFORM_NAME__
                    )
                raise

            self._ready.set()
            return True

        except SlackApiError as e:
            if e.response["error"] == "invalid_auth":
                raise PlatformAuthenticationError(
                    message="Invalid authentication token", platform_error=e, platform_name=__PLATFORM_NAME__
                )
            elif e.response["error"] == "rate_limited":
                raise PlatformRateLimitError(
                    message="Rate limit exceeded",
                    platform_error=e,
                    platform_name=__PLATFORM_NAME__,
                    retry_after=float(e.response.headers.get("Retry-After", 60)),
                )
            raise PlatformError(message=f"Slack API error: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__)

    def start(self):
        """Start the Slack client and validate connection."""
        try:
            # Start socket mode if available
            if self._socket_client:
                self._socket_client.connect()

            # Validate configuration
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._loop = loop
                return asyncio.run_coroutine_threadsafe(self.validate(), loop)
            else:
                return loop.run_until_complete(self.validate())

        except Exception as e:
            raise PlatformError(
                message=f"Error starting Slack client: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    async def send_message(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to the Slack channel.

        Args:
            message: The message to send

        Returns:
            Tuple[str, Optional[str]]: Status message and message ID if successful
        """
        try:
            # Split message if it exceeds Slack's limit (40k characters)
            if len(message) > 40000:
                chunks = [message[i : i + 39999] for i in range(0, len(message), 39999)]
                first_response = None
                for i, chunk in enumerate(chunks):
                    response = self._web_client.chat_postMessage(
                        channel=self._config.channel_id,
                        text=chunk,
                        thread_ts=first_response["ts"] if first_response else None,
                    )
                    if i == 0:
                        first_response = response
                return "Message sent (split into chunks)", first_response["ts"]
            else:
                response = self._web_client.chat_postMessage(channel=self._config.channel_id, text=message)
                return "Message sent successfully", response["ts"]

        except SlackApiError as e:
            if e.response["error"] == "invalid_auth":
                raise PlatformAuthenticationError(
                    message="Invalid authentication token", platform_error=e, platform_name=__PLATFORM_NAME__
                )
            elif e.response["error"] == "rate_limited":
                raise PlatformRateLimitError(
                    message="Rate limit exceeded",
                    platform_error=e,
                    platform_name=__PLATFORM_NAME__,
                    retry_after=float(e.response.headers.get("Retry-After", 60)),
                )
            raise PlatformError(
                message=f"Failed to send message: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    async def wait_for_replies(self, message_id: str, timeout_minutes: int = 5, max_replies: int = 0) -> List[dict]:
        """Wait for replies to a specific message.

        Args:
            message_id: ID of message to monitor for replies
            timeout_minutes: How long to wait for replies (0 = no timeout)
            max_replies: Maximum number of replies to collect (0 = unlimited)

        Returns:
            List of reply messages with content, author, timestamp, and attachments
        """
        if not message_id:
            return []

        # Initialize reply tracking
        self._message_replies[message_id] = []
        event = asyncio.Event()
        self._reply_events[message_id] = {"event": event, "max_replies": max_replies}

        try:
            # If using Socket Mode, we'll get real-time updates
            if self._socket_client:
                if timeout_minutes > 0:
                    try:
                        await asyncio.wait_for(event.wait(), timeout=timeout_minutes * 60)
                    except asyncio.TimeoutError:
                        if not self._message_replies[message_id]:
                            raise PlatformError(
                                message=f"Timeout waiting for replies after {timeout_minutes} minutes",
                                platform_name=__PLATFORM_NAME__,
                            )
            else:
                # Fall back to polling if Socket Mode isn't available
                start_time = time.time()
                while True:
                    try:
                        response = self._web_client.conversations_replies(
                            channel=self._config.channel_id, ts=message_id
                        )

                        # Process any new replies
                        for msg in response["messages"][1:]:  # Skip the first message (original)
                            if msg["ts"] not in [r["id"] for r in self._message_replies[message_id]]:
                                reply_data = {
                                    "content": msg.get("text", ""),
                                    "author": self._get_user_name(msg.get("user")),
                                    "timestamp": datetime.fromtimestamp(float(msg.get("ts", 0))).isoformat(),
                                    "id": msg.get("ts"),
                                }

                                if "files" in msg:
                                    attachments = [
                                        f"(Attachment: {f.get('name')}, URL: {f.get('url_private')})"
                                        for f in msg["files"]
                                    ]
                                    if attachments:
                                        reply_data["content"] += f" {' '.join(attachments)}"

                                self._message_replies[message_id].append(reply_data)

                        # Check if we've hit max replies
                        if max_replies > 0 and len(self._message_replies[message_id]) >= max_replies:
                            break

                        # Check timeout
                        if timeout_minutes > 0 and (time.time() - start_time) > (timeout_minutes * 60):
                            if not self._message_replies[message_id]:
                                raise PlatformError(
                                    message=f"Timeout waiting for replies after {timeout_minutes} minutes",
                                    platform_name=__PLATFORM_NAME__,
                                )
                            break

                        # Sleep before next poll
                        await asyncio.sleep(1)

                    except SlackApiError as e:
                        if e.response["error"] == "rate_limited":
                            await asyncio.sleep(float(e.response.headers.get("Retry-After", 1)))
                        else:
                            raise

        finally:
            # Cleanup and return replies
            replies = self._message_replies.pop(message_id, [])
            self._reply_events.pop(message_id, None)
            return replies

    def cleanup_reply_monitoring(self, message_id: str):
        """Clean up reply monitoring for a specific message."""
        self._message_replies.pop(message_id, None)
        self._reply_events.pop(message_id, None)

    def shutdown(self):
        """Shutdown the Slack client."""
        if self._socket_client:
            self._socket_client.close()

        if self._thread and self._thread.is_alive():
            self._shutdown.set()
            self._thread.join(timeout=2)


class SlackExecutor(PlatformExecutorAgent):
    """Slack-specific executor agent."""

    def __init__(self, platform_config: SlackConfig, reply_monitor_config: Optional[ReplyMonitorConfig] = None):
        super().__init__(platform_config, reply_monitor_config)

        self._slack = SlackHandler(platform_config)

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start Slack client
        self._slack.start()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        """Clean shutdown of the Slack client."""
        self._slack.shutdown()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.shutdown()

    def send_to_platform(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to Slack channel.

        Args:
            message: The message to send

        Returns:
            Tuple[str, Optional[str]]: Status message and message ID if successful
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._slack.send_message(message))
        finally:
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
        """Wait for reply from Slack.

        Args:
            msg_id: Message ID to monitor for replies

        Returns:
            List of reply messages formatted as strings
        """
        if not self.reply_monitor_config:
            return []

        _ = asyncio.new_event_loop

    def cleanup_monitoring(self, msg_id: str):
        pass


class SlackAgent(CommsPlatformAgent):
    """Agent for Slack communication."""

    def __init__(
        self,
        name: str,
        platform_config: SlackConfig,
        send_config: dict,
        message_to_send: Optional[
            callable
        ] = None,  # The function to determine the message to send, returns None to indicate do not send a message, otherwise determined automatically
        reply_monitor_config: Optional[ReplyMonitorConfig] = None,
        auto_reply: str = "Message sent to Slack",
        system_message: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that communicates through Slack. "
                "Remember that Slack uses Markdown-like formatting and has message length limits. "
                "Keep messages clear and concise, and consider using appropriate formatting when helpful."
            )

        # Create Slack-specific executor
        slack_executor = SlackExecutor(platform_config, reply_monitor_config)

        super().__init__(
            name=name,
            platform_config=platform_config,
            executor_agent=slack_executor,
            send_config=send_config,
            message_to_send=message_to_send,
            reply_monitor_config=reply_monitor_config,
            auto_reply=auto_reply,
            system_message=system_message,
            *args,
            **kwargs,
        )

        # Update decision agent's knowledge of Slack specifics
        self.decision_agent.update_system_message(
            self.decision_agent.system_message
            + "\n"
            + "Format guidelines for Slack:\n"
            + "1. Max message length: 40,000 characters\n"
            + "2. Supports Markdown-like formatting:\n"
            + "   - *text* for italic\n"
            + "   - **text** for bold\n"
            + "   - `code` for inline code\n"
            + "   - ```code block``` for multi-line code\n"
            + "3. Supports message threading for organized discussions\n"
            + "4. Can use :emoji_name: for emoji reactions\n"
            + "5. Supports block quotes with > prefix\n"
            + "6. Can use <!here> or <!channel> for notifications"
        )
