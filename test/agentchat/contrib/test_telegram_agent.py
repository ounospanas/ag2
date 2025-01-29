import asyncio
import threading
from typing import Dict, List, Optional, Union
from unittest.mock import MagicMock, AsyncMock

import pytest
import telegram
from autogen import Agent

DEFAULT_TEST_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": "fake-key",
        }
    ]
}

from autogen.agentchat.contrib.comms.telegram_agent import (
    TelegramAgent,
    TelegramConfig,
    TelegramExecutor,
)
from autogen.agentchat.contrib.comms.platform_configs import ReplyMonitorConfig
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

def test_telegram_config_validation():
    config = TelegramConfig(
        bot_token="123456:ABC-DEF",
        destination_id="@testchannel"
    )
    assert config.validate_config() is True

    with pytest.raises(ValueError, match="bot_token is required"):
        TelegramConfig(
            bot_token="",
            destination_id="@testchannel"
        ).validate_config()

    with pytest.raises(ValueError, match="destination_id is required"):
        TelegramConfig(
            bot_token="123456:ABC-DEF",
            destination_id=""
        ).validate_config()


def test_telegram_agent_initialization(mocker, telegram_config):
    mock_handler = mocker.patch(
        "autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler"
    )
    handler = mock_handler.return_value
    handler.start.return_value = True
    handler._loop = asyncio.get_event_loop()
    handler._thread = threading.current_thread()
    handler._message_replies = {}
    handler.silent = False
    
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    assert agent.name == "test_telegram_agent"
    assert isinstance(agent.executor_agent, TelegramExecutor)
    mock_handler.assert_called_once()


def test_telegram_agent_invalid_token(mocker, telegram_config):
    mock_handler = mocker.patch(
        "autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler"
    )
    handler = mock_handler.return_value
    handler.start.return_value = True
    handler.send_message = AsyncMock(side_effect=telegram.error.InvalidToken())
    handler._loop = asyncio.get_event_loop()
    handler._thread = threading.current_thread()
    
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    
    with pytest.raises(PlatformAuthenticationError) as exc_info:
        agent.executor_agent.send_to_platform("test message")
    assert "Invalid bot token" in str(exc_info.value)
    assert exc_info.value.platform_name == "Telegram"
    assert handler.send_message.call_count == 1


def test_telegram_agent_invalid_destination(mocker, telegram_config):
    mock_handler = mocker.patch(
        "autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler"
    )
    handler = mock_handler.return_value
    handler.start.return_value = True
    handler.send_message = AsyncMock(
        side_effect=telegram.error.BadRequest("Chat not found")
    )
    handler._loop = asyncio.get_event_loop()
    handler._thread = threading.current_thread()
    
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    
    with pytest.raises(PlatformConnectionError) as exc_info:
        agent.executor_agent.send_to_platform("test message")
    assert "Chat not found" in str(exc_info.value)
    assert exc_info.value.platform_name == "Telegram"
    assert handler.send_message.call_count == 1


def test_telegram_agent_send_message_sync(mocker, telegram_config):
    mock_handler = mocker.patch(
        "autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler"
    )
    handler = mock_handler.return_value
    handler.start.return_value = True
    handler.send_message = AsyncMock(
        return_value=("Message sent successfully", "123456789")
    )
    handler._loop = asyncio.get_event_loop()
    handler._thread = threading.current_thread()
    
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    
    response = agent.executor_agent.send_to_platform("Hello Telegram!")
    assert response[0] == "Message sent successfully"
    assert response[1] == "123456789"
    handler.send_message.assert_called_once_with("Hello Telegram!")


def test_telegram_agent_send_message_async(mocker, telegram_config):
    mock_handler = mocker.patch(
        "autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler"
    )
    handler = mock_handler.return_value
    handler.start.return_value = True
    handler.send_message = AsyncMock(
        return_value=("Message sent successfully", "123456789")
    )
    handler._loop = asyncio.get_event_loop()
    handler._thread = threading.current_thread()
    
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    
    response = agent.executor_agent.send_to_platform("Send this message to Telegram")
    assert response[0] == "Message sent successfully"
    assert response[1] == "123456789"
    handler.send_message.assert_called_once_with("Send this message to Telegram")
