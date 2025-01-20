# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel
from termcolor import colored

from ..code_utils import content_str
from ..oai.client import OpenAIWrapper
from .base_message import BaseMessage, wrap_message

if TYPE_CHECKING:
    from ..agentchat.agent import Agent
    from ..coding.base import CodeBlock


@wrap_message
class InputRequestMessage(BaseMessage, ABC):
    pass

@wrap_message
class TextInputRequestMessage(InputRequestMessage):
    prompt: str = ""

@wrap_message
class PasswordInputRequestMessage(InputRequestMessage):
    prompt: str = ""

@wrap_message
class SingleChoiceInputRequestMessage(InputRequestMessage):
    prompt: str = ""
    choices: list[str]

@wrap_message
class MultipleChoiceInputRequestMessage(InputRequestMessage):
    prompt: str = ""
    choices: list[str]

@wrap_message
class InputMessageResponse(BaseMessage, ABC):
    request: InputRequestMessage

@wrap_message
class TextInputResponsetMessage(InputMessageResponse):
    text: str
    request: TextInputRequestMessage

@wrap_message
class PasswordInputResponseMessage(InputMessageResponse):
    password: str
    request: PasswordInputRequestMessage

@wrap_message
class SingleChoiceInputResponseMessage(InputMessageResponse):
    choice: str
    request: SingleChoiceInputRequestMessage

@wrap_message
class MultipleChoiceInputResponseMessage(InputMessageResponse):
    choices: list[str]
    request: MultipleChoiceInputRequestMessage