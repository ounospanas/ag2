# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseCommsPlatformConfig(ABC):
    """Base configuration for all communication platform configs."""

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
