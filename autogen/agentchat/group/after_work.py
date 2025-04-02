# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional, Type, Union

from pydantic import BaseModel, field_validator

from .context_str import ContextStr
from .transition_target import TransitionTarget

if TYPE_CHECKING:
    # Avoid circular import
    from ..conversable_agent import ConversableAgent

# NOTE: THIS WILL BE REMOVED AND REPLACED BY A TRANSITIONTARGET

__all__ = [
    "AfterWork",
    "AfterWorkSelectionMessage",
    "AfterWorkSelectionMessageContextStr",
    "AfterWorkSelectionMessageString",
]


# AfterWorkSelectionMessage protocol and implementations
class AfterWorkSelectionMessage(BaseModel):
    """Base class for all AfterWork selection message types."""

    def get_message(self, agent: "ConversableAgent") -> str:
        """Get the formatted message."""
        raise NotImplementedError("Requires subclasses to implement.")


class AfterWorkSelectionMessageString(AfterWorkSelectionMessage):
    """Selection message that uses a plain string template."""

    message: str

    def get_message(self, agent: "ConversableAgent") -> str:
        """Get the message string."""
        return self.message


class AfterWorkSelectionMessageContextStr(AfterWorkSelectionMessage):
    """Selection message that uses a ContextStr template."""

    context_str_template: str

    # We will replace {agentlist} with another term and return it later for use with the internal group chat auto speaker selection
    # Otherwise our format will fail
    @field_validator("context_str_template", mode="before")
    def _replace_agentlist_placeholder(cls: Type["AfterWorkSelectionMessageContextStr"], v: Any) -> Union[str, Any]:  # noqa: N805
        """Replace {agentlist} placeholder before validation/assignment."""
        if isinstance(v, str):
            if "{agentlist}" in v:
                return v.replace("{agentlist}", "<<agent_list>>")  # Perform the replacement
            else:
                return v  # If no replacement is needed, return the original value
        return ""

    def get_message(self, agent: "ConversableAgent") -> str:
        """Get the formatted message with context variables substituted."""
        context_str = ContextStr(template=self.context_str_template)
        format_result = context_str.format(agent.context_variables)
        if format_result is None:
            return ""

        return format_result.replace(
            "<<agent_list>>", "{agentlist}"
        )  # Restore agentlist so it can be substituted by the internal group chat auto speaker selection


# AfterWork

# NOTE: This class will be removed and TransitionTargets will be used instead of AfterWork


class AfterWork(BaseModel):
    """Handles the next step in the conversation when an agent doesn't suggest a tool call or a conditional handoff."""

    target: TransitionTarget
    selection_message: Optional[AfterWorkSelectionMessage] = None
