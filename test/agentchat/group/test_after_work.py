# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.group.after_work import (
    AfterWork,
    AfterWorkSelectionMessage,
    AfterWorkSelectionMessageContextStr,
    AfterWorkSelectionMessageString,
    AfterWorkTarget,
    AfterWorkTargetAgent,
    AfterWorkTargetAgentName,
    AfterWorkTargetOption,
)
from autogen.agentchat.group.context_variables import ContextVariables


class TestAfterWorkTarget:
    def test_base_target_resolve(self) -> None:
        """Test that the base AfterWorkTarget class raises NotImplementedError when resolve is called."""
        target = AfterWorkTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.resolve(None, [], None)
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestAfterWorkTargetOption:
    def test_init(self) -> None:
        """Test initialisation with different options."""
        # Test with valid options
        for option in ["terminate", "revert_to_user", "stay", "group_manager"]:
            target = AfterWorkTargetOption(option=option)
            assert target.option == option

    def test_resolve(self) -> None:
        """Test that resolve returns the option value."""
        option = "terminate"
        target = AfterWorkTargetOption(option=option)
        result = target.resolve(None, [], None)
        assert result == option


class TestAfterWorkTargetAgent:
    def test_init(self) -> None:
        """Test initialisation with a mock agent."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AfterWorkTargetAgent(agent=mock_agent)
        assert target.agent_name == "test_agent"

    def test_resolve(self) -> None:
        """Test that resolve returns the agent object from the groupchat."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        mock_groupchat = MagicMock()
        mock_groupchat.agents = [mock_agent]

        target = AfterWorkTargetAgent(agent=mock_agent)
        result = target.resolve(None, [], mock_groupchat)
        assert result == mock_agent

    def test_resolve_agent_not_found(self) -> None:
        """Test that resolve raises ValueError when agent is not found in groupchat."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        mock_groupchat = MagicMock()
        mock_groupchat.agents = []  # Empty list, so agent won't be found

        target = AfterWorkTargetAgent(agent=mock_agent)
        with pytest.raises(ValueError) as excinfo:
            target.resolve(None, [], mock_groupchat)
        assert "Agent with name 'test_agent' not found in groupchat" in str(excinfo.value)


class TestAfterWorkTargetAgentName:
    def test_init(self) -> None:
        """Test initialisation with an agent name."""
        target = AfterWorkTargetAgentName(agent_name="test_agent")
        assert target.agent_name == "test_agent"

    def test_resolve(self) -> None:
        """Test that resolve returns the agent name."""
        agent_name = "test_agent"
        target = AfterWorkTargetAgentName(agent_name=agent_name)
        result = target.resolve(None, [], None)
        assert result == agent_name


class TestAfterWorkSelectionMessage:
    def test_base_message_get_message(self) -> None:
        """Test that the base AfterWorkSelectionMessage class raises NotImplementedError when get_message is called."""
        message = AfterWorkSelectionMessage()
        with pytest.raises(NotImplementedError) as excinfo:
            message.get_message(None, [])
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestAfterWorkSelectionMessageString:
    def test_init(self) -> None:
        """Test initialisation with a string template."""
        template = "This is a test template"
        message = AfterWorkSelectionMessageString(template=template)
        assert message.template == template

    def test_get_message(self) -> None:
        """Test that get_message returns the template string."""
        template = "This is a test template"
        message = AfterWorkSelectionMessageString(template=template)
        result = message.get_message(None, [])
        assert result == template


class TestAfterWorkSelectionMessageContextStr:
    def test_init(self) -> None:
        """Test initialisation with a context string template."""
        template = "Hello, {name}!"
        message = AfterWorkSelectionMessageContextStr(context_str_template=template)
        assert message.context_str_template == template

    def test_get_message(self) -> None:
        """Test that get_message formats the template with context variables."""
        template = "Hello, {name}!"
        message = AfterWorkSelectionMessageContextStr(context_str_template=template)

        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"name": "World"})

        result = message.get_message(mock_agent, [])
        assert result == "Hello, World!"


class TestAfterWork:
    def test_init(self) -> None:
        """Test initialisation with different parameters."""
        # Test with target arg
        target = AfterWorkTargetOption(option="terminate")
        after_work = AfterWork(target_arg=target)
        assert after_work.target == target
        assert after_work.selection_message is None

        # Test with target and selection message args
        message = AfterWorkSelectionMessageString(template="Test message")
        after_work = AfterWork(target_arg=target, selection_message_arg=message)
        assert after_work.target == target
        assert after_work.selection_message == message

        # Test with data dict
        target = AfterWorkTargetOption(option="stay")
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    @patch("autogen.agentchat.group.after_work.ContextStr")
    def test_with_integrated_components(self, mock_context_str) -> None:
        """Test AfterWork with integrated components."""
        # Configure mock
        mock_context_str_instance = MagicMock()
        mock_context_str_instance.format.return_value = "Hello, Test Agent!"
        mock_context_str.return_value = mock_context_str_instance

        # Create components
        target = AfterWorkTargetOption(option="terminate")
        message = AfterWorkSelectionMessageContextStr(context_str_template="Hello, {name}!")

        # Create AfterWork with components
        after_work = AfterWork(target_arg=target, selection_message_arg=message)

        # Create mock objects for resolve
        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"name": "Test Agent"})
        mock_groupchat = MagicMock()

        # Test resolve
        result = after_work.target.resolve(None, [], mock_groupchat)
        assert result == "terminate"

        # Test get_message
        message_result = after_work.selection_message.get_message(mock_agent, [])
        mock_context_str_instance.format.assert_called_once_with(mock_agent.context_variables)
        assert message_result == "Hello, Test Agent!"
