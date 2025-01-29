import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, List, Optional, Union, Tuple
import asyncio

from autogen import Agent
from autogen.agentchat.contrib.comms.telegram_agent import TelegramAgent, TelegramConfig
from autogen.agentchat.contrib.comms.slack_agent import SlackAgent, SlackConfig
from autogen.agentchat.contrib.comms.discord_agent import DiscordAgent, DiscordConfig
from autogen.agentchat.contrib.comms.platform_configs import ReplyMonitorConfig
from autogen.agentchat.contrib.comms.platform_errors import PlatformError

DEFAULT_TEST_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": "fake-key"
        }
    ]
}

@pytest.fixture
def telegram_config():
    return TelegramConfig(bot_token="123456:ABC-DEF", destination_id="@testchannel")

@pytest.fixture
def slack_config():
    return SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")

@pytest.fixture
def discord_config():
    return DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")

@pytest.mark.asyncio
async def test_telegram_agent_init(telegram_config):
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    assert agent.name == "test_telegram_agent"
    assert isinstance(agent, TelegramAgent)

@pytest.mark.asyncio
async def test_telegram_agent_send_message(telegram_config):
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    agent.executor_agent.send_to_platform = MagicMock(return_value=("Message sent", "123"))
    
    result = agent.send_message("Test message")
    assert result == ("Message sent", "123")
    agent.executor_agent.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_telegram_agent_async_send(telegram_config):
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    agent.executor_agent.send_to_platform = MagicMock(return_value=("Message sent", "123"))
    
    result = await agent.a_send_message("Test message")
    assert result == ("Message sent", "123")
    agent.executor_agent.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_slack_agent_init(slack_config):
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=slack_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    assert agent.name == "test_slack_agent"
    assert isinstance(agent, SlackAgent)

@pytest.mark.asyncio
async def test_slack_agent_send_message(slack_config):
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=slack_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    agent.executor_agent.send_to_platform = MagicMock(return_value=("Message sent", "123"))
    
    result = agent.send_message("Test message")
    assert result == ("Message sent", "123")
    agent.executor_agent.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_slack_agent_async_send(slack_config):
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=slack_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    agent.executor_agent.send_to_platform = MagicMock(return_value=("Message sent", "123"))
    
    result = await agent.a_send_message("Test message")
    assert result == ("Message sent", "123")
    agent.executor_agent.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_discord_agent_init(discord_config):
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=discord_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    assert agent.name == "test_discord_agent"
    assert isinstance(agent, DiscordAgent)

@pytest.mark.asyncio
async def test_discord_agent_send_message(discord_config):
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=discord_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    agent.executor_agent.send_to_platform = MagicMock(return_value=("Message sent", "123"))
    
    result = agent.send_message("Test message")
    assert result == ("Message sent", "123")
    agent.executor_agent.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_discord_agent_async_send(discord_config):
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=discord_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    agent.executor_agent.send_to_platform = MagicMock(return_value=("Message sent", "123"))
    
    result = await agent.a_send_message("Test message")
    assert result == ("Message sent", "123")
    agent.executor_agent.send_to_platform.assert_called_once_with("Test message")
