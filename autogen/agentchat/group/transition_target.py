# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    # Avoid circular import
    from ..conversable_agent import ConversableAgent
    from ..groupchat import GroupChat

__all__ = ["AfterWorkOptionTarget", "AgentNameTarget", "AgentTarget", "NestedChatTarget", "TransitionTarget"]

# Common options for transitions
TransitionOption = Literal["terminate", "revert_to_user", "stay", "group_manager"]


class TransitionTarget(BaseModel):
    """Base class for all transition targets across OnCondition, OnContextCondition, and AfterWork."""

    def resolve(self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat") -> Any:
        """Resolve to a concrete agent, chat configuration, or control option."""
        raise NotImplementedError("Requires subclasses to implement.")

    def display_name(self) -> str:
        """Get the display name for the target."""
        raise NotImplementedError("Requires subclasses to implement.")

    def normalized_name(self) -> str:
        """Get a normalized name for the target that has no spaces, used for function calling"""
        raise NotImplementedError("Requires subclasses to implement.")

class AgentTarget(TransitionTarget):
    """Target that represents a direct agent reference."""

    agent_name: str

    def __init__(self, agent: "ConversableAgent", **data):
        # Store the name from the agent for serialization
        super().__init__(agent_name=agent.name, **data)

    def resolve(
        self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat"
    ) -> "ConversableAgent":
        """Resolve to the actual agent object from the groupchat."""
        for agent in groupchat.agents:
            if agent.name == self.agent_name:
                return agent
        raise ValueError(f"Agent with name '{self.agent_name}' not found in groupchat")

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

    def resolve(
        self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat"
    ) -> "ConversableAgent":
        """Resolve to the agent name string."""
        for agent in groupchat.agents:
            if agent.name == self.agent_name:
                return agent
        raise ValueError(f"Agent with name '{self.agent_name}' not found in groupchat")

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

    def resolve(
        self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat"
    ) -> dict[str, Any]:
        """Resolve to the nested chat configuration."""
        raise NotImplementedError("NestedChatTarget does not support the resolve method. An agent should be used to encapsulate this nested chat and then the target changed to an AgentTarget.")

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

    def resolve(
        self, last_speaker: "ConversableAgent", messages: list[dict[str, Any]], groupchat: "GroupChat"
    ) -> TransitionOption:
        """Resolve to the option value."""
        return self.after_work_option

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
