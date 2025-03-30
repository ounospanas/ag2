# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.handoffs import Handoffs
from autogen.agentchat.group.speaker_selection_result import SpeakerSelectionResult
from autogen.agentchat.group.transition_target import (
    __AGENT_WRAPPER_PREFIX__,
    AfterWorkOptionTarget,
    AgentNameTarget,
    AgentTarget,
    NestedChatTarget,
    TransitionTarget,
)


class TestTransitionTarget:
    def test_base_target_can_resolve_for_speaker_selection(self) -> None:
        """Test that the base TransitionTarget's can_resolve_for_speaker_selection returns False."""
        target = TransitionTarget()
        assert target.can_resolve_for_speaker_selection() is False

    def test_base_target_resolve(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError when resolve is called."""
        target = TransitionTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.resolve(None, [], None, None, None)
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_base_target_display_name(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError for display_name."""
        target = TransitionTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.display_name()
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_base_target_normalized_name(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError for normalized_name."""
        target = TransitionTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.normalized_name()
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_base_target_needs_agent_wrapper(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError for needs_agent_wrapper."""
        target = TransitionTarget()
        with pytest.raises(NotImplementedError) as excinfo:
            target.needs_agent_wrapper()
        assert "Requires subclasses to implement" in str(excinfo.value)

    def test_base_target_create_wrapper_agent(self) -> None:
        """Test that the base TransitionTarget class raises NotImplementedError for create_wrapper_agent."""
        target = TransitionTarget()
        mock_agent = MagicMock(spec=ConversableAgent)
        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "Requires subclasses to implement" in str(excinfo.value)


class TestAgentTarget:
    def test_init(self) -> None:
        """Test initialisation with a mock agent."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.agent_name == "test_agent"

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with the agent name."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"

        target = AgentTarget(agent=mock_agent)
        result = target.resolve(None, [], None, None, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name == "test_agent"
        assert result.terminate is None
        assert result.speaker_selection_method is None

    def test_display_name(self) -> None:
        """Test that display_name returns the agent name."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.display_name() == "test_agent"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns the agent name."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.normalized_name() == "test_agent"

    def test_str_representation(self) -> None:
        """Test the string representation of AgentTarget."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert str(target) == "Transfer to test_agent"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        mock_agent = MagicMock(spec=ConversableAgent)
        mock_agent.name = "test_agent"
        target = AgentTarget(agent=mock_agent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "AgentTarget does not require wrapping" in str(excinfo.value)


class TestAgentNameTarget:
    def test_init(self) -> None:
        """Test initialisation with an agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.agent_name == "test_agent"

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with the agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        result = target.resolve(None, [], None, None, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name == "test_agent"
        assert result.terminate is None
        assert result.speaker_selection_method is None

    def test_display_name(self) -> None:
        """Test that display_name returns the agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.display_name() == "test_agent"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns the agent name."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.normalized_name() == "test_agent"

    def test_str_representation(self) -> None:
        """Test the string representation of AgentNameTarget."""
        target = AgentNameTarget(agent_name="test_agent")
        assert str(target) == "Transfer to test_agent"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        target = AgentNameTarget(agent_name="test_agent")
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        target = AgentNameTarget(agent_name="test_agent")
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "AgentNameTarget does not require wrapping" in str(excinfo.value)


class TestNestedChatTarget:
    def test_init(self) -> None:
        """Test initialisation with a nested chat config."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.nested_chat_config == nested_chat_config

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns False."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.can_resolve_for_speaker_selection() is False

    def test_resolve_raises_error(self) -> None:
        """Test that resolve raises NotImplementedError."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)

        with pytest.raises(NotImplementedError) as excinfo:
            target.resolve(None, [], None, None, None)
        assert "NestedChatTarget does not support the resolve method" in str(excinfo.value)

    def test_display_name(self) -> None:
        """Test that display_name returns 'a nested chat'."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.display_name() == "a nested chat"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns 'nested_chat'."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.normalized_name() == "nested_chat"

    def test_str_representation(self) -> None:
        """Test the string representation of NestedChatTarget."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert str(target) == "Transfer to nested chat"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns True."""
        nested_chat_config = {"chat_queue": ["agent1", "agent2"], "use_async": True}
        target = NestedChatTarget(nested_chat_config=nested_chat_config)
        assert target.needs_agent_wrapper() is True

    def test_create_wrapper_agent(self) -> None:
        """Test creating a wrapper agent for a nested chat target."""
        # Set up the nested chat
        sample_agent = ConversableAgent(name="sample_agent")
        sample_agent_two = ConversableAgent(name="sample_agent_two")
        nested_chat_config = {
            "chat_queue": [
                {
                    "recipient": sample_agent,
                    "summary_method": "reflection_with_llm",
                    "summary_prompt": "Summarise the conversation into bullet points.",
                },
                {
                    "recipient": sample_agent_two,
                    "message": "Write a poem about the context.",
                    "max_turns": 1,
                    "summary_method": "last_msg",
                },
            ],
            "use_async": False,
        }
        target = NestedChatTarget(nested_chat_config=nested_chat_config)

        # Set up the parent agent
        parent_agent = ConversableAgent(name="parent_agent")
        parent_agent.handoffs = Handoffs()

        # Call create_wrapper_agent
        index = 2
        result = target.create_wrapper_agent(parent_agent, index)

        assert result.name == f"{__AGENT_WRAPPER_PREFIX__}_nested_{parent_agent.name}_{index + 1}"


class TestAfterWorkOptionTarget:
    def test_init(self) -> None:
        """Test initialisation with different options."""
        # Test with valid options
        for option in ["terminate", "revert_to_user", "stay", "group_manager"]:
            target = AfterWorkOptionTarget(after_work_option=option)
            assert target.after_work_option == option

    def test_can_resolve_for_speaker_selection(self) -> None:
        """Test that can_resolve_for_speaker_selection returns True."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        assert target.can_resolve_for_speaker_selection() is True

    def test_resolve_terminate(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with terminate=True for 'terminate' option."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        result = target.resolve(None, [], None, None, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.terminate is True
        assert result.agent_name is None
        assert result.speaker_selection_method is None

    def test_resolve_stay(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with the current agent for 'stay' option."""
        target = AfterWorkOptionTarget(after_work_option="stay")
        mock_current_agent = MagicMock(spec=ConversableAgent)
        mock_current_agent.name = "current_agent"

        result = target.resolve(None, [], None, mock_current_agent, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name == "current_agent"
        assert result.terminate is None
        assert result.speaker_selection_method is None

    def test_resolve_revert_to_user(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with the user agent for 'revert_to_user' option."""
        target = AfterWorkOptionTarget(after_work_option="revert_to_user")
        mock_user_agent = MagicMock(spec=ConversableAgent)
        mock_user_agent.name = "user_agent"

        result = target.resolve(None, [], None, None, mock_user_agent)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.agent_name == "user_agent"
        assert result.terminate is None
        assert result.speaker_selection_method is None

    def test_resolve_group_manager(self) -> None:
        """Test that resolve returns a SpeakerSelectionResult with 'auto' for 'group_manager' option."""
        target = AfterWorkOptionTarget(after_work_option="group_manager")
        result = target.resolve(None, [], None, None, None)

        assert isinstance(result, SpeakerSelectionResult)
        assert result.speaker_selection_method == "auto"
        assert result.agent_name is None
        assert result.terminate is None

    def test_resolve_unknown_option(self) -> None:
        """Test that resolve raises ValueError for unknown options."""
        # Set an invalid option via a private attribute to bypass any validation in the constructor
        target = AfterWorkOptionTarget(after_work_option="terminate")
        target.after_work_option = "unknown_option"  # type: ignore

        with pytest.raises(ValueError) as excinfo:
            target.resolve(None, [], None, None, None)
        assert "Unknown after work option: unknown_option" in str(excinfo.value)

    def test_display_name(self) -> None:
        """Test that display_name returns the option name with description."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        assert target.display_name() == "After Work option 'terminate'"

    def test_normalized_name(self) -> None:
        """Test that normalized_name returns the formatted option name."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        assert target.normalized_name() == "after_work_option_terminate"

        # Test with an option containing spaces
        target = AfterWorkOptionTarget(after_work_option="revert_to_user")
        assert target.normalized_name() == "after_work_option_revert_to_user"

    def test_str_representation(self) -> None:
        """Test the string representation of AfterWorkOptionTarget."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        assert str(target) == "Transfer option terminate"

    def test_needs_agent_wrapper(self) -> None:
        """Test that needs_agent_wrapper returns False."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        assert target.needs_agent_wrapper() is False

    def test_create_wrapper_agent_raises_error(self) -> None:
        """Test that create_wrapper_agent raises NotImplementedError."""
        target = AfterWorkOptionTarget(after_work_option="terminate")
        mock_agent = MagicMock(spec=ConversableAgent)

        with pytest.raises(NotImplementedError) as excinfo:
            target.create_wrapper_agent(mock_agent, 0)
        assert "AfterWorkOptionTarget does not require wrapping" in str(excinfo.value)
