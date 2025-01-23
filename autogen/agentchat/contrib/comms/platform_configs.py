# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class BasePlatformConfig(ABC):
    """Base configuration for all platform configs."""

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the configuration settings.

        Returns:
            bool: True if valid, raises ValueError if invalid
        """
        pass


@dataclass
class ReplyMonitorConfig:
    """Configuration for handling platform replies."""

    timeout_minutes: int = 1
    """How long to wait for replies before timing out."""

    max_reply_messages: int = 1
    """Maximum number of messages to collect before returning."""


# MS MOVE THESE TO THEIR RESPECTIVE CLASSES


@dataclass
class SlackConfig(BasePlatformConfig):
    """Slack-specific configuration.

    To get started:
    1. Create Slack App in your workspace
    2. Add Bot User OAuth token under OAuth & Permissions
    3. Install app to workspace
    4. Add bot to desired channel
    """

    bot_token: str
    """Bot User OAuth Token starting with xoxb-."""

    channel_id: str
    """Channel ID where messages will be sent."""

    signing_secret: str
    """Signing secret for verifying requests from Slack."""

    app_token: Optional[str] = None
    """App-level token starting with xapp- (required for Socket Mode)."""

    def validate_config(self) -> bool:
        if not self.bot_token or not self.bot_token.startswith("xoxb-"):
            raise ValueError("Valid Slack bot_token required (should start with xoxb-)")
        if not self.channel_id:
            raise ValueError("channel_id is required")
        if not self.signing_secret:
            raise ValueError("signing_secret is required")
        return True


@dataclass
class TeamsConfig(BasePlatformConfig):
    """Microsoft Teams configuration using Bot Framework.

    To get started:
    1. Register bot with Azure Bot Service
    2. Create Microsoft App ID and secret
    3. Configure bot channels registration
    """

    app_id: str
    """Microsoft App ID (UUID)."""

    app_password: str
    """Microsoft App Password/Secret."""

    tenant_id: str
    """Microsoft 365 tenant ID."""

    channel_id: Optional[str] = None
    """For specific channel messages."""

    user_id: Optional[str] = None
    """For direct messages."""

    def validate_config(self) -> bool:
        if not self.app_id:
            raise ValueError("app_id is required")
        if not self.app_password:
            raise ValueError("app_password is required")
        if not self.tenant_id:
            raise ValueError("tenant_id is required")
        return True


@dataclass
class WhatsAppConfig(BasePlatformConfig):
    """WhatsApp Business API configuration.

    To get started:
    1. Create Meta Developer Account
    2. Set up WhatsApp Business API
    3. Complete business verification
    4. Get permanent access token
    """

    access_token: str
    """Permanent access token from Meta/Facebook."""

    phone_number_id: str
    """ID of the registered WhatsApp phone number."""

    business_account_id: str
    """WhatsApp Business Account ID."""

    recipient_phone: str
    """Default recipient's phone number with country code."""

    api_version: str = "v17.0"
    """Meta API version."""

    def validate_config(self) -> bool:
        if not self.access_token:
            raise ValueError("access_token is required")
        if not self.phone_number_id:
            raise ValueError("phone_number_id is required")
        if not self.business_account_id:
            raise ValueError("business_account_id is required")
        if not self.recipient_phone:
            raise ValueError("recipient_phone is required")
        return True


@dataclass
class TelegramConfig(BasePlatformConfig):
    """Telegram configuration using Bot API.

    To get started:
    1. Create bot with @BotFather
    2. Get the bot token from BotFather
    3. Get chat_id by:
       - For direct messages: Send message to bot and use getUpdates API
         (/start, then check https://api.telegram.org/bot<YourBOTToken>/getUpdates)
       - For group: Add bot to group and use getUpdates API
         (add bot to group, send a message, then check getUpdates)
    """

    bot_token: str
    """Bot token from BotFather (starts with numbers:ABC...)."""

    chat_id: str
    """Chat or group ID where messages will be sent (can be negative for groups)."""

    def validate_config(self) -> bool:
        if not self.bot_token:
            raise ValueError("bot_token is required")
        if not self.chat_id:
            raise ValueError("chat_id is required")
        return True
