# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from .speaker_selection_result import SpeakerSelectionResult

if TYPE_CHECKING:
    # Avoid circular import
    from ..conversable_agent import ConversableAgent
    from ..groupchat import GroupChat

__all__ = ["AfterWorkOptionTarget", "AgentNameTarget", "AgentTarget", "NestedChatTarget", "TransitionTarget"]

# Common options for transitions
TransitionOption = Literal["terminate", "revert_to_user", "stay", "group_manager"]


class TransitionTarget(BaseModel):
    """Base class for all transition targets across OnCondition, OnContextCondition, and AfterWork."""

    def resolve(
        self,
        last_speaker: "ConversableAgent",
        messages: list[dict[str, Any]],
        groupchat: "GroupChat",
        current_agent: "ConversableAgent",
        user_agent: "ConversableAgent",
    ) -> SpeakerSelectionResult:
        """Resolve to a speaker selection result (Agent, None for termination, or str for speaker selection method)."""
        raise NotImplementedError("Requires subclasses to implement.")

    def display_name(self) -> str:
        """Get the display name for the target."""
        raise NotImplementedError("Requires subclasses to implement.")

    def normalized_name(self) -> str:
        """Get a normalized name for the target that has no spaces, used for function calling"""
        raise NotImplementedError("Requires subclasses to implement.")

    def can_resolve_for_speaker_selection(self) -> bool:
        """Check if the target can resolve to an option for speaker selection (Agent, 'None' to end, Str for speaker selection method). In the case of a nested chat, this will return False as it should be encapsulated in an agent."""
        return False


class AgentTarget(TransitionTarget):
    """Target that represents a direct agent reference."""

    agent_name: str

    def __init__(self, agent: "ConversableAgent", **data):
        # Store the name from the agent for serialization
        super().__init__(agent_name=agent.name, **data)

    def can_resolve_for_speaker_selection(self) -> bool:
        """Check if the target can resolve for speaker selection."""
        return True

    def resolve(
        self,
        last_speaker: "ConversableAgent",
        messages: list[dict[str, Any]],
        groupchat: "GroupChat",
        current_agent: "ConversableAgent",
        user_agent: "ConversableAgent",
    ) -> SpeakerSelectionResult:
        """Resolve to the actual agent object from the groupchat."""
        return SpeakerSelectionResult(agent_name=self.agent_name)

    def display_name(self) -> str:
        """Get the display name for the target."""
        return f"{self.agent_name}"

    def normalized_name(self) -> str:
        """Get a normalized name for the target that has no spaces, used for function calling"""
        return self.display_name()

    def __str__(self) -> str:
        """String representation for AgentTarget, can be shown as a function call message."""
        return f"Transfer to {self.agent_name}"


class AgentNameTarget(TransitionTarget):
    """Target that represents an agent by name."""

    agent_name: str

    def __init__(self, agent_name: str, **data):
        super().__init__(agent_name=agent_name, **data)

    def can_resolve_for_speaker_selection(self) -> bool:
        """Check if the target can resolve for speaker selection."""
        return True

    def resolve(
        self,
        last_speaker: "ConversableAgent",
        messages: list[dict[str, Any]],
        groupchat: "GroupChat",
        current_agent: "ConversableAgent",
        user_agent: "ConversableAgent",
    ) -> SpeakerSelectionResult:
        """Resolve to the agent name string."""
        return SpeakerSelectionResult(agent_name=self.agent_name)

    def display_name(self) -> str:
        """Get the display name for the target."""
        return f"{self.agent_name}"

    def normalized_name(self) -> str:
        """Get a normalized name for the target that has no spaces, used for function calling"""
        return self.display_name()

    def __str__(self) -> str:
        """String representation for AgentTarget, can be shown as a function call message."""
        return f"Transfer to {self.agent_name}"


class NestedChatTarget(TransitionTarget):
    """Target that represents a nested chat configuration."""

    nested_chat_config: dict[str, Any]

    def __init__(self, nested_chat_config: dict[str, Any], **data):
        super().__init__(nested_chat_config=nested_chat_config, **data)

    def can_resolve_for_speaker_selection(self) -> bool:
        """Check if the target can resolve for speaker selection. For NestedChatTarget the nested chat must be encapsulated into an agent."""
        return False

    def resolve(
        self,
        last_speaker: "ConversableAgent",
        messages: list[dict[str, Any]],
        groupchat: "GroupChat",
        current_agent: "ConversableAgent",
        user_agent: "ConversableAgent",
    ) -> SpeakerSelectionResult:
        """Resolve to the nested chat configuration."""
        raise NotImplementedError(
            "NestedChatTarget does not support the resolve method. An agent should be used to encapsulate this nested chat and then the target changed to an AgentTarget."
        )

    def display_name(self) -> str:
        """Get the display name for the target."""
        return "a nested chat"

    def normalized_name(self) -> str:
        """Get a normalized name for the target that has no spaces, used for function calling"""
        return "nested_chat"

    def __str__(self) -> str:
        """String representation for AgentTarget, can be shown as a function call message."""
        return "Transfer to nested chat"


class AfterWorkOptionTarget(TransitionTarget):
    """Target that represents an AfterWorkOption."""

    after_work_option: TransitionOption

    def __init__(self, after_work_option: TransitionOption, **data):
        super().__init__(after_work_option=after_work_option, **data)

    def can_resolve_for_speaker_selection(self) -> bool:
        """Check if the target can resolve for speaker selection. AfterWorkOptionTarget does not resolve to an agent."""
        return True

    def resolve(
        self,
        last_speaker: "ConversableAgent",
        messages: list[dict[str, Any]],
        groupchat: "GroupChat",
        current_agent: "ConversableAgent",
        user_agent: "ConversableAgent",
    ) -> SpeakerSelectionResult:
        """Resolve to the option value."""
        if self.after_work_option == "terminate":
            return SpeakerSelectionResult(terminate=True)
        elif self.after_work_option == "stay":
            return SpeakerSelectionResult(agent_name=current_agent.name)
        elif self.after_work_option == "revert_to_user":
            return SpeakerSelectionResult(agent_name=user_agent.name)
        elif self.after_work_option == "group_manager":
            return SpeakerSelectionResult(speaker_selection_method="auto")
        else:
            raise ValueError(f"Unknown after work option: {self.after_work_option}")

    def display_name(self) -> str:
        """Get the display name for the target."""
        return f"After Work option '{self.after_work_option}'"

    def normalized_name(self) -> str:
        """Get a normalized name for the target that has no spaces, used for function calling"""
        return f"after_work_option_{self.after_work_option.replace(' ', '_')}"

    def __str__(self) -> str:
        """String representation for AgentTarget, can be shown as a function call message."""
        return f"Transfer option {self.after_work_option}"


# TODO: Consider adding a SequentialChatTarget class
