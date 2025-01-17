# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Iterable, Optional, Protocol, Union

from .agent import Agent

if TYPE_CHECKING:
    from ..messages import BaseMessage


class ResponseProtocol(Protocol):
    @property
    def messages(self) -> Iterable["BaseMessage"]:
        """The messages received by the agent."""
        ...

class AsyncResponseProtocol(Protocol):
    @property
    def messages(self) -> Iterable["BaseMessage"]:
        """The messages received by the agent."""
        ...


def initiate_chat(
    agent: Agent, *, message: Union[str, dict[str, Any]], recipient: Optional[Agent], request_reply: bool = False
) -> ResponseProtocol:
    pass

def a_initiate_chat(
    agent: Agent, *, message: Union[str, dict[str, Any]], recipient: Optional[Agent], request_reply: bool = False
) -> ResponseProtocol:
    pass