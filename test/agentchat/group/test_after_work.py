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
)
from autogen.agentchat.group.context_str import ContextStr
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.transition_target import (
    AfterWorkOptionTarget,
    AgentNameTarget,
    AgentTarget,
    NestedChatTarget,
)


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

    def test_init_with_agentlist(self) -> None:
        """Test initialisation with {agentlist} placeholder."""
        template = "Choose an agent from: {agentlist}"
        message = AfterWorkSelectionMessageContextStr(context_str_template=template)

        # Check that the template is properly transformed during initialisation
        assert message.context_str_template == "Choose an agent from: <<agent_list>>"

        # Directly create a mock for ContextStr to test the replacement behavior
        mock_format_result = "Choose an agent from: <<agent_list>>"

        with patch.object(ContextStr, "format", return_value=mock_format_result):
            mock_agent = MagicMock()
            mock_agent.context_variables = ContextVariables(data={})

            result = message.get_message(mock_agent, [])

            # Verify it replaced <<agent_list>> back to {agentlist} for the GroupChat's speaker selection
            assert result == "Choose an agent from: {agentlist}"


class TestAfterWork:
    def test_init(self) -> None:
        """Test initialisation with different parameters."""
        # Test with target
        target = AfterWorkOptionTarget(after_work_option="terminate")
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

        # Test with target and selection message
        message = AfterWorkSelectionMessageString(template="Test message")
        after_work = AfterWork(target=target, selection_message=message)
        assert after_work.target == target
        assert after_work.selection_message == message

    @patch("autogen.agentchat.group.after_work.ContextStr")
    def test_with_integrated_components(self, mock_context_str) -> None:
        """Test AfterWork with integrated components."""
        # Configure mock
        mock_context_str_instance = MagicMock()
        mock_context_str_instance.format.return_value = "Hello, Test Agent!"
        mock_context_str.return_value = mock_context_str_instance

        # Create components
        target = AfterWorkOptionTarget(after_work_option="terminate")
        message = AfterWorkSelectionMessageContextStr(context_str_template="Hello, {name}!")

        # Create AfterWork with components
        after_work = AfterWork(target=target, selection_message=message)

        # Create mock objects for resolve
        mock_agent = MagicMock()
        mock_agent.context_variables = ContextVariables(data={"name": "Test Agent"})

        # Test get_message
        message_result = after_work.selection_message.get_message(mock_agent, [])
        mock_context_str_instance.format.assert_called_once_with(mock_agent.context_variables)
        assert message_result == "Hello, Test Agent!"

    def test_init_with_option_target(self) -> None:
        """Test initialisation with AfterWorkOptionTarget."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    def test_init_with_agent_target(self) -> None:
        """Test initialisation with AgentTarget."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    def test_init_with_agent_name_target(self) -> None:
        """Test initialisation with AgentNameTarget."""
        target = AgentNameTarget(agent_name="test_agent")
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    def test_init_with_nested_chat_target(self) -> None:
        """Test initialisation with NestedChatTarget."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        after_work = AfterWork(target=target)
        assert after_work.target == target
        assert after_work.selection_message is None

    def test_init_with_selection_message(self) -> None:
        """Test initialisation with target and selection message."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        message = AfterWorkSelectionMessageString(template="Test message")
        after_work = AfterWork(target=target, selection_message=message)
        assert after_work.target == target
        assert after_work.selection_message == message

    def test_target_resolution(self) -> None:
        """Test that different target types resolve correctly through AfterWork."""
        # Mock the components needed for resolution
        mock_last_speaker = MagicMock()
        mock_messages = []
        mock_groupchat = MagicMock()
        mock_current_agent = MagicMock(name="current_agent")
        mock_current_agent.name = "current_agent"
        mock_user_agent = MagicMock(name="user_agent")
        mock_user_agent.name = "user_agent"

        # Test with terminate option
        target = AfterWorkOptionTarget(after_work_option="terminate")

        result = target.resolve(mock_last_speaker, mock_messages, mock_groupchat, mock_current_agent, mock_user_agent)

        # Verify SpeakerSelectionResult was created with terminate=True
        assert result.terminate is True
        assert result.agent_name is None
        assert result.speaker_selection_method is None

        # Test with stay option
        target = AfterWorkOptionTarget(after_work_option="stay")

        result = target.resolve(mock_last_speaker, mock_messages, mock_groupchat, mock_current_agent, mock_user_agent)

        # Verify SpeakerSelectionResult was created with the current agent's name
        assert result.terminate is None
        assert result.agent_name == "current_agent"
        assert result.speaker_selection_method is None

        # Test with revert_to_user option
        target = AfterWorkOptionTarget(after_work_option="revert_to_user")

        result = target.resolve(mock_last_speaker, mock_messages, mock_groupchat, mock_current_agent, mock_user_agent)

        # Verify SpeakerSelectionResult was created with the user agent's name
        assert result.terminate is None
        assert result.agent_name == "user_agent"
        assert result.speaker_selection_method is None

        # Test with group_manager option
        target = AfterWorkOptionTarget(after_work_option="group_manager")

        result = target.resolve(mock_last_speaker, mock_messages, mock_groupchat, mock_current_agent, mock_user_agent)

        # Verify SpeakerSelectionResult was created with speaker_selection_method="auto"
        assert result.terminate is None
        assert result.agent_name is None
        assert result.speaker_selection_method == "auto"
