# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel
from termcolor import colored

from ..code_utils import content_str
from ..oai.client import OpenAIWrapper
from .base_message import BaseMessage, wrap_message

if TYPE_CHECKING:
    from ..agentchat.agent import Agent
    from ..coding.base import CodeBlock


__all__ = [
    "ClearAgentsHistoryMessage",
    "ClearConversableAgentHistoryMessage",
    "ConversableAgentUsageSummaryMessage",
    "ConversableAgentUsageSummaryNoCostIncurredMessage",
    "ExecuteCodeBlockMessage",
    "ExecuteFunctionMessage",
    "FunctionCallMessage",
    "FunctionResponseMessage",
    "GenerateCodeExecutionReplyMessage",
    "GroupChatResumeMessage",
    "GroupChatRunChatMessage",
    "PostCarryoverProcessingMessage",
    "SelectSpeakerMessage",
    "SpeakerAttemptFailedMultipleAgentsMessage",
    "SpeakerAttemptFailedNoAgentsMessage",
    "SpeakerAttemptSuccessfullMessage",
    "TerminationAndHumanReplyMessage",
    "TextMessage",
    "ToolCallMessage",
    "ToolResponseMessage",
]

MessageRole = Literal["assistant", "function", "tool"]


class BasePrintReceivedMessage(BaseMessage, ABC):
    """A base class for messages that print received messages.

    Attributes:
        content (Union[str, int, float, bool]): The content of the message.
        sender_name (str): The name of the sender.
        recipient_name (str): The name of the recipient.
    """

    content: Union[str, int, float, bool]
    sender_name: str
    recipient_name: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the sender and recipient names.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print
        f(f"{colored(self.sender_name, 'yellow')} (to {self.recipient_name}):\n", flush=True)


@wrap_message
class FunctionResponseMessage(BasePrintReceivedMessage):
    """A message class representing the response from a function call.

    Attributes:
        name (Optional[str]): The name of the function.
        role (MessageRole): The role of the message, default is "function".
        content (Union[str, int, float, bool]): The content of the message.

    """

    name: Optional[str] = None
    role: MessageRole = "function"
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the function name and content.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print
        super().print(f)

        id = self.name or "No id found"
        func_print = f"***** Response from calling {self.role} ({id}) *****"
        f(colored(func_print, "green"), flush=True)
        f(self.content, flush=True)
        f(colored("*" * len(func_print), "green"), flush=True)

        f("\n", "-" * 80, flush=True, sep="")


class ToolResponse(BaseModel):
    """A model class representing a tool response.

    Attributes:
        tool_call_id (Optional[str]): The ID of the tool call.
        role (MessageRole): The role of the message, default is "tool".
        content (Union[str, int, float, bool]): The content of the message.
    """

    tool_call_id: Optional[str] = None
    role: MessageRole = "tool"
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the tool response details, including the tool call ID and content.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print
        id = self.tool_call_id or "No id found"
        tool_print = f"***** Response from calling {self.role} ({id}) *****"
        f(colored(tool_print, "green"), flush=True)
        f(self.content, flush=True)
        f(colored("*" * len(tool_print), "green"), flush=True)


@wrap_message
class ToolResponseMessage(BasePrintReceivedMessage):
    """A message class representing the response from a tool call.

    Attributes:
        role (MessageRole): The role of the message, default is "tool".
        tool_responses (list[ToolResponse]): A list of tool responses.
        content (Union[str, int, float, bool]): The content of the message.
    """

    role: MessageRole = "tool"
    tool_responses: list[ToolResponse]
    content: Union[str, int, float, bool]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the tool responses and content.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print
        super().print(f)

        for tool_response in self.tool_responses:
            tool_response.print(f)
            f("\n", "-" * 80, flush=True, sep="")


class FunctionCall(BaseModel):
    """A model class representing a function call.

    Attributes:
        name (Optional[str]): The name of the function.
        arguments (Optional[str]): The arguments for the function.
    """

    name: Optional[str] = None
    arguments: Optional[str] = None

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the function call details, including the function name and arguments.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        name = self.name or "(No function name found)"
        arguments = self.arguments or "(No arguments found)"

        func_print = f"***** Suggested function call: {name} *****"
        f(colored(func_print, "green"), flush=True)
        f(
            "Arguments: \n",
            arguments,
            flush=True,
            sep="",
        )
        f(colored("*" * len(func_print), "green"), flush=True)


@wrap_message
class FunctionCallMessage(BasePrintReceivedMessage):
    """A message class representing a function call.

    Attributes:
        content (Optional[Union[str, int, float, bool]]): The content of the message.
        function_call (FunctionCall): The function call details.
    """

    content: Optional[Union[str, int, float, bool]] = None  # type: ignore [assignment]
    function_call: FunctionCall

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the content and function call.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print
        super().print(f)

        if self.content is not None:
            f(self.content, flush=True)

        self.function_call.print(f)

        f("\n", "-" * 80, flush=True, sep="")


class ToolCall(BaseModel):
    """A model class representing a tool call.

    Attributes:
        id (Optional[str]): The ID of the tool call.
        function (FunctionCall): The function call details.
        type (str): The type of the tool call.
    """

    id: Optional[str] = None
    function: FunctionCall
    type: str

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the tool call details, including the ID, function call, and type.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        id = self.id or "No tool call id found"

        name = self.function.name or "(No function name found)"
        arguments = self.function.arguments or "(No arguments found)"

        func_print = f"***** Suggested tool call ({id}): {name} *****"
        f(colored(func_print, "green"), flush=True)
        f(
            "Arguments: \n",
            arguments,
            flush=True,
            sep="",
        )
        f(colored("*" * len(func_print), "green"), flush=True)


@wrap_message
class ToolCallMessage(BasePrintReceivedMessage):
    """A message class representing a tool call.

    Attributes:
        content (Optional[Union[str, int, float, bool]]): The content of the message.
        refusal (Optional[str]): The refusal message.
        role (Optional[MessageRole]): The role of the message.
        audio (Optional[str]): The audio content.
        function_call (FunctionCall): The function call details.
        tool_calls (list[ToolCall]): A list of tool calls.
    """

    content: Optional[Union[str, int, float, bool]] = None  # type: ignore [assignment]
    refusal: Optional[str] = None
    role: Optional[MessageRole] = None
    audio: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: list[ToolCall]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the content, refusal message, function call, and tool calls.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print
        super().print(f)

        if self.content is not None:
            f(self.content, flush=True)

        for tool_call in self.tool_calls:
            tool_call.print(f)

        f("\n", "-" * 80, flush=True, sep="")


@wrap_message
class TextMessage(BasePrintReceivedMessage):
    """A message class representing a simple text message."""

    content: Optional[Union[str, int, float, bool]] = None  # type: ignore [assignment]

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the text content.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print
        super().print(f)

        if self.content is not None:
            f(content_str(self.content), flush=True)  # type: ignore [arg-type]

        f("\n", "-" * 80, flush=True, sep="")


def create_received_message_model(
    *, uuid: Optional[UUID] = None, message: dict[str, Any], sender: "Agent", recipient: "Agent"
) -> Union[FunctionResponseMessage, ToolResponseMessage, FunctionCallMessage, ToolCallMessage, TextMessage]:
    """Creates a received message model based on the message content.

    Args:
        uuid (Optional[UUID]): The UUID of the message.
        message (dict[str, Any]): The message content.
        sender (Agent): The sender of the message.
        recipient (Agent): The recipient of the message.
    """
    # print(f"{message=}")
    # print(f"{sender=}")

    role = message.get("role")
    if role == "function":
        return FunctionResponseMessage(**message, sender_name=sender.name, recipient_name=recipient.name, uuid=uuid)
    if role == "tool":
        return ToolResponseMessage(**message, sender_name=sender.name, recipient_name=recipient.name, uuid=uuid)

    # Role is neither function nor tool

    if message.get("function_call"):
        return FunctionCallMessage(
            **message,
            sender_name=sender.name,
            recipient_name=recipient.name,
            uuid=uuid,
        )

    if message.get("tool_calls"):
        return ToolCallMessage(
            **message,
            sender_name=sender.name,
            recipient_name=recipient.name,
            uuid=uuid,
        )

    # Now message is a simple content message
    content = message.get("content")
    allow_format_str_template = (
        recipient.llm_config.get("allow_format_str_template", False) if recipient.llm_config else False  # type: ignore [attr-defined]
    )
    if content is not None and "context" in message:
        content = OpenAIWrapper.instantiate(
            content,  # type: ignore [arg-type]
            message["context"],
            allow_format_str_template,
        )

    return TextMessage(
        content=content,
        sender_name=sender.name,
        recipient_name=recipient.name,
        uuid=uuid,
    )


@wrap_message
class PostCarryoverProcessingMessage(BaseMessage):
    """A message class for post-carryover processing.

    Attributes:
        carryover (Union[str, list[Union[str, dict[str, Any], Any]]]): The carryover data.
        message (str): The message content.
        verbose (bool): Whether to print verbose output.
        sender_name (str): The name of the sender.
        recipient_name (str): The name of the recipient.
        summary_method (str): The summary method.
        summary_args (Optional[dict[str, Any]]): The summary arguments.
        max_turns (Optional[int]): The maximum number of turns.
    """

    carryover: Union[str, list[Union[str, dict[str, Any], Any]]]
    message: str
    verbose: bool = False

    sender_name: str
    recipient_name: str
    summary_method: str
    summary_args: Optional[dict[str, Any]] = None
    max_turns: Optional[int] = None

    def __init__(self, *, uuid: Optional[UUID] = None, chat_info: dict[str, Any]):
        """Initializes the PostCarryoverProcessingMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            chat_info (dict[str, Any]): The chat information.
        """
        carryover = chat_info.get("carryover", "")
        message = chat_info.get("message")
        verbose = chat_info.get("verbose", False)

        sender_name = chat_info["sender"].name
        recipient_name = chat_info["recipient"].name
        summary_args = chat_info.get("summary_args")
        max_turns = chat_info.get("max_turns")

        # Fix Callable in chat_info
        summary_method = chat_info.get("summary_method", "")
        if callable(summary_method):
            summary_method = summary_method.__name__

        print_message = ""
        if isinstance(message, str):
            print_message = message
        elif callable(message):
            print_message = "Callable: " + message.__name__
        elif isinstance(message, dict):
            print_message = "Dict: " + str(message)
        elif message is None:
            print_message = "None"

        super().__init__(
            uuid=uuid,
            carryover=carryover,
            message=print_message,
            verbose=verbose,
            summary_method=summary_method,
            summary_args=summary_args,
            max_turns=max_turns,
            sender_name=sender_name,
            recipient_name=recipient_name,
        )

    def _process_carryover(self) -> str:
        """Processes the carryover data and returns it as a string.

        Returns:
            str: The processed carryover data as a string.
        """
        if not isinstance(self.carryover, list):
            return self.carryover

        print_carryover = []
        for carryover_item in self.carryover:
            if isinstance(carryover_item, str):
                print_carryover.append(carryover_item)
            elif isinstance(carryover_item, dict) and "content" in carryover_item:
                print_carryover.append(str(carryover_item["content"]))
            else:
                print_carryover.append(str(carryover_item))

        return ("\n").join(print_carryover)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the carryover content.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        print_carryover = self._process_carryover()

        f(colored("\n" + "*" * 80, "blue"), flush=True, sep="")
        f(
            colored(
                "Starting a new chat....",
                "blue",
            ),
            flush=True,
        )
        if self.verbose:
            f(colored("Message:\n" + self.message, "blue"), flush=True)
            f(colored("Carryover:\n" + print_carryover, "blue"), flush=True)
        f(colored("\n" + "*" * 80, "blue"), flush=True, sep="")


@wrap_message
class ClearAgentsHistoryMessage(BaseMessage):
    """A message class for clearing an agent's history.

    Attributes:
        agent_name: The name of the agent whose history is to be cleared.
        nr_messages_to_preserve: The number of messages to preserve.
    """

    agent_name: Optional[str] = None
    nr_messages_to_preserve: Optional[int] = None

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        agent: Optional["Agent"] = None,
        nr_messages_to_preserve: Optional[int] = None,
    ):
        """Initializes the ClearAgentsHistoryMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            agent (Optional[Agent]): The agent whose history is to be cleared.
            nr_messages_to_preserve (Optional[int]): The number of messages to preserve.
        """
        return super().__init__(
            uuid=uuid, agent_name=agent.name if agent else None, nr_messages_to_preserve=nr_messages_to_preserve
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the agent's name and the number of messages to preserve.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        if self.agent_name:
            if self.nr_messages_to_preserve:
                f(f"Clearing history for {self.agent_name} except last {self.nr_messages_to_preserve} messages.")
            else:
                f(f"Clearing history for {self.agent_name}.")
        else:
            if self.nr_messages_to_preserve:
                f(f"Clearing history for all agents except last {self.nr_messages_to_preserve} messages.")
            else:
                f("Clearing history for all agents.")


# todo: break into multiple messages
@wrap_message
class SpeakerAttemptSuccessfullMessage(BaseMessage):
    """A message class representing a successful speaker selection attempt.

    Attributes:
        mentions (dict[str, int]): A dictionary of agent names and their mention counts.
        attempt (int): The current attempt number.
        attempts_left (int): The number of attempts left.
        verbose (Optional[bool]): Whether to print verbose output.
    """

    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        """Initializes the SpeakerAttemptSuccessfullMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            mentions (dict[str, int]): A dictionary of agent names and their mention counts.
            attempt (int): The current attempt number.
            attempts_left (int): The number of attempts left.
            select_speaker_auto_verbose (Optional[bool]): Whether to print verbose output.
        """
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the selected agent name.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        selected_agent_name = next(iter(self.mentions))
        f(
            colored(
                f">>>>>>>> Select speaker attempt {self.attempt} of {self.attempt + self.attempts_left} successfully selected: {selected_agent_name}",
                "green",
            ),
            flush=True,
        )


@wrap_message
class SpeakerAttemptFailedMultipleAgentsMessage(BaseMessage):
    """A message class representing a failed speaker selection attempt due to multiple agent names.

    Attributes:
        mentions (dict[str, int]): A dictionary of agent names and their mention counts.
        attempt (int): The current attempt number.
        attempts_left (int): The number of attempts left.
        verbose (Optional[bool]): Whether to print verbose output.
    """

    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        """Initializes the SpeakerAttemptFailedMultipleAgentsMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            mentions (dict[str, int]): A dictionary of agent names and their mention counts.
            attempt (int): The current attempt number.
            attempts_left (int): The number of attempts left.
            select_speaker_auto_verbose (Optional[bool]): Whether to print verbose output.
        """
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, indicating the failure due to multiple agent names.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(
            colored(
                f">>>>>>>> Select speaker attempt {self.attempt} of {self.attempt + self.attempts_left} failed as it included multiple agent names.",
                "red",
            ),
            flush=True,
        )


@wrap_message
class SpeakerAttemptFailedNoAgentsMessage(BaseMessage):
    """A message class representing a failed speaker selection attempt due to no agent names.

    Attributes:
        mentions (dict[str, int]): A dictionary of agent names and their mention counts.
        attempt (int): The current attempt number.
        attempts_left (int): The number of attempts left.
        verbose (Optional[bool]): Whether to print verbose output.
    """

    mentions: dict[str, int]
    attempt: int
    attempts_left: int
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        mentions: dict[str, int],
        attempt: int,
        attempts_left: int,
        select_speaker_auto_verbose: Optional[bool] = False,
    ):
        """Initializes the SpeakerAttemptFailedNoAgentsMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            mentions (dict[str, int]): A dictionary of agent names and their mention counts.
            attempt (int): The current attempt number.
            attempts_left (int): The number of attempts left.
            select_speaker_auto_verbose (Optional[bool]): Whether to print verbose output.
        """
        super().__init__(
            uuid=uuid,
            mentions=deepcopy(mentions),
            attempt=attempt,
            attempts_left=attempts_left,
            verbose=select_speaker_auto_verbose,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, indicating the failure due to no agent names.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(
            colored(
                f">>>>>>>> Select speaker attempt #{self.attempt} failed as it did not include any agent names.",
                "red",
            ),
            flush=True,
        )


@wrap_message
class GroupChatResumeMessage(BaseMessage):
    """A message class for resuming a group chat.

    Attributes:
        last_speaker_name (str): The name of the last speaker.
        messages (list[dict[str, Any]]): A list of messages in the group chat.
        verbose (Optional[bool]): Whether to print verbose output.
    """

    last_speaker_name: str
    messages: list[dict[str, Any]]
    verbose: Optional[bool] = False

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        last_speaker_name: str,
        messages: list[dict[str, Any]],
        silent: Optional[bool] = False,
    ):
        """Initializes the GroupChatResumeMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            last_speaker_name (str): The name of the last speaker.
            messages (list[dict[str, Any]]): A list of messages in the group chat.
            silent (Optional[bool]): Whether to print verbose output.
        """
        super().__init__(uuid=uuid, last_speaker_name=last_speaker_name, messages=messages, verbose=not silent)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the last speaker and the number of messages.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(
            f"Prepared group chat with {len(self.messages)} messages, the last speaker is",
            colored(self.last_speaker_name, "yellow"),
            flush=True,
        )


@wrap_message
class GroupChatRunChatMessage(BaseMessage):
    """A message class for running a group chat.

    Attributes:
        speaker_name (str): The name of the speaker.
        verbose (Optional[bool]): Whether to print verbose output.
    """

    speaker_name: str
    verbose: Optional[bool] = False

    def __init__(self, *, uuid: Optional[UUID] = None, speaker: "Agent", silent: Optional[bool] = False):
        """Initializes the GroupChatRunChatMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            speaker (Agent): The speaker in the group chat.
            silent (Optional[bool]): Whether to print verbose output.
        """
        super().__init__(uuid=uuid, speaker_name=speaker.name, verbose=not silent)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the speaker's name.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(colored(f"\nNext speaker: {self.speaker_name}\n", "green"), flush=True)


@wrap_message
class TerminationAndHumanReplyMessage(BaseMessage):
    """A message class representing a termination and human reply message.

    Attributes:
        no_human_input_msg (str): The message indicating no human input.
        sender_name (str): The name of the sender.
        recipient_name (str): The name of the recipient.
    """

    no_human_input_msg: str
    sender_name: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        no_human_input_msg: str,
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        """Initializes the TerminationAndHumanReplyMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            no_human_input_msg (str): The message indicating no human input.
            sender (Optional[Agent]): The sender of the message.
            recipient (Agent): The recipient of the message.
        """
        super().__init__(
            uuid=uuid,
            no_human_input_msg=no_human_input_msg,
            sender_name=sender.name if sender else "No sender",
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the no human input message.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(colored(f"\n>>>>>>>> {self.no_human_input_msg}", "red"), flush=True)


@wrap_message
class UsingAutoReplyMessage(BaseMessage):
    """A message class for using an auto reply.

    Attributes:
        human_input_mode (str): The human input mode.
        sender_name (str): The name of the sender.
        recipient_name (str): The name of the recipient.
    """

    human_input_mode: str
    sender_name: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        human_input_mode: str,
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        """Initializes the UsingAutoReplyMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            human_input_mode (str): The human input mode.
            sender (Optional[Agent]): The sender of the message.
            recipient (Agent): The recipient of the message.
        """
        super().__init__(
            uuid=uuid,
            human_input_mode=human_input_mode,
            sender_name=sender.name if sender else "No sender",
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the human input mode.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)


@wrap_message
class ExecuteCodeBlockMessage(BaseMessage):
    """A message class for executing a code block.

    Attributes:
        code (str): The code to be executed.
        language (str): The programming language of the code.
        code_block_count (int): The count of the code block.
        recipient_name (str): The name of the recipient.
    """

    code: str
    language: str
    code_block_count: int
    recipient_name: str

    def __init__(
        self, *, uuid: Optional[UUID] = None, code: str, language: str, code_block_count: int, recipient: "Agent"
    ):
        """Initializes the ExecuteCodeBlockMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            code (str): The code to be executed.
            language (str): The programming language of the code.
            code_block_count (int): The count of the code block.
            recipient (Agent): The recipient of the message.
        """
        super().__init__(
            uuid=uuid, code=code, language=language, code_block_count=code_block_count, recipient_name=recipient.name
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the code block and its language.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(
            colored(
                f"\n>>>>>>>> EXECUTING CODE BLOCK {self.code_block_count} (inferred language is {self.language})...",
                "red",
            ),
            flush=True,
        )


@wrap_message
class ExecuteFunctionMessage(BaseMessage):
    """A message class for executing a function.

    Attributes:
        func_name (str): The name of the function.
        call_id (Optional[str]): The call ID of the function.
        arguments (dict[str, Any]): The arguments for the function.
        recipient_name (str): The name of the recipient.
    """

    func_name: str
    call_id: Optional[str] = None
    arguments: dict[str, Any]
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        func_name: str,
        call_id: Optional[str] = None,
        arguments: dict[str, Any],
        recipient: "Agent",
    ):
        """Initializes the ExecuteFunctionMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            func_name (str): The name of the function.
            call_id (Optional[str]): The call ID of the function.
            arguments (dict[str, Any]): The arguments for the function.
            recipient (Agent): The recipient of the message.
        """
        super().__init__(
            uuid=uuid, func_name=func_name, call_id=call_id, arguments=arguments, recipient_name=recipient.name
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the function name, call ID, and arguments.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(
            colored(
                f"\n>>>>>>>> EXECUTING FUNCTION {self.func_name}...\nCall ID: {self.call_id}\nInput arguments: {self.arguments}",
                "magenta",
            ),
            flush=True,
        )


@wrap_message
class ExecutedFunctionMessage(BaseMessage):
    """A message class representing the execution of a function.

    Attributes:
        func_name (str): The name of the function.
        call_id (Optional[str]): The call ID of the function.
        arguments (dict[str, Any]): The arguments for the function.
        content (str): The content of the function execution.
        recipient_name (str): The name of the recipient.
    """

    func_name: str
    call_id: Optional[str] = None
    arguments: dict[str, Any]
    content: str
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        func_name: str,
        call_id: Optional[str] = None,
        arguments: dict[str, Any],
        content: str,
        recipient: "Agent",
    ):
        """Initializes the ExecutedFunctionMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            func_name (str): The name of the function.
            call_id (Optional[str]): The call ID of the function.
            arguments (dict[str, Any]): The arguments for the function.
            content (str): The content of the function execution.
            recipient (Agent): The recipient of the message.
        """
        super().__init__(
            uuid=uuid,
            func_name=func_name,
            call_id=call_id,
            arguments=arguments,
            content=content,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the function name, call ID, arguments, and content.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(
            colored(
                f"\n>>>>>>>> EXECUTED FUNCTION {self.func_name}...\nCall ID: {self.call_id}\nInput arguments: {self.arguments}\nOutput:\n{self.content}",
                "magenta",
            ),
            flush=True,
        )


@wrap_message
class SelectSpeakerMessage(BaseMessage):
    """A message class for selecting the next speaker.

    Attributes:
        agent_names (Optional[list[str]]): A list of agent names to choose from.
    """

    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, agents: Optional[list["Agent"]] = None):
        """Initializes the SelectSpeakerMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            agents (Optional[list[Agent]]): A list of agents to choose from.
        """
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the list of agent names to choose from.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f("Please select the next speaker from the following list:")
        agent_names = self.agent_names or []
        for i, agent_name in enumerate(agent_names):
            f(f"{i + 1}: {agent_name}")


@wrap_message
class SelectSpeakerTryCountExceededMessage(BaseMessage):
    """A message class for exceeding the speaker selection try count.

    Attributes:
        try_count (int): The number of tries.
        agent_names (Optional[list[str]]): A list of agent names to choose from.
    """

    try_count: int
    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, try_count: int, agents: Optional[list["Agent"]] = None):
        """Initializes the SelectSpeakerTryCountExceededMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            try_count (int): The number of tries.
            agents (Optional[list[Agent]]): A list of agents to choose from.
        """
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, try_count=try_count, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the number of tries and the list of agent names to choose from.

        Args:
            f (Optional[Callable[..., Any]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(f"You have tried {self.try_count} times. The next speaker will be selected automatically.")


@wrap_message
class SelectSpeakerInvalidInputMessage(BaseMessage):
    """A message class for invalid input when selecting the next speaker.

    Attributes:
        agent_names (Optional[list[str]]): A list of agent names to choose from.
    """

    agent_names: Optional[list[str]] = None

    def __init__(self, *, uuid: Optional[UUID] = None, agents: Optional[list["Agent"]] = None):
        """Initializes the SelectSpeakerInvalidInputMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            agents (Optional[list[Agent]]): A list of agents to choose from.
        """
        agent_names = [agent.name for agent in agents] if agents else None
        super().__init__(uuid=uuid, agent_names=agent_names)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints a message indicating invalid input when selecting the next speaker.

        Args:
            f (Optional[Callable[..., Any]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(f"Invalid input. Please enter a number between 1 and {len(self.agent_names or [])}.")


@wrap_message
class ClearConversableAgentHistoryMessage(BaseMessage):
    """A message class for clearing a conversable agent's history.

    Attributes:
        agent_name (str): The name of the agent whose history is to be cleared.
        recipient_name (str): The name of the recipient.
        no_messages_preserved (int): The number of messages to preserve.
    """

    agent_name: str
    recipient_name: str
    no_messages_preserved: int

    def __init__(self, *, uuid: Optional[UUID] = None, agent: "Agent", no_messages_preserved: Optional[int] = None):
        """Initializes the ClearConversableAgentHistoryMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            agent (Agent): The agent whose history is to be cleared.
            no_messages_preserved (Optional[int]): The number of messages to preserve.
        """
        super().__init__(
            uuid=uuid,
            agent_name=agent.name,
            recipient_name=agent.name,
            no_messages_preserved=no_messages_preserved,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the agent's name and the number of messages to preserve.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        for _ in range(self.no_messages_preserved):
            f(
                f"Preserving one more message for {self.agent_name} to not divide history between tool call and "
                f"tool response."
            )


@wrap_message
class ClearConversableAgentHistoryWarningMessage(BaseMessage):
    """A message class for warning about clearing a conversable agent's history.

    Attributes:
        recipient_name (str): The name of the recipient.
    """

    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        """Initializes the ClearConversableAgentHistoryWarningMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            recipient (Agent): The recipient of the message.
        """
        super().__init__(
            uuid=uuid,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints a warning message about clearing a conversable agent's history.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(
            colored(
                "WARNING: `nr_preserved_messages` is ignored when clearing chat history with a specific agent.",
                "yellow",
            ),
            flush=True,
        )


@wrap_message
class GenerateCodeExecutionReplyMessage(BaseMessage):
    """A message class for generating a code execution reply.

    Attributes:
        code_block_languages (list[str]): A list of programming languages for the code blocks.
        sender_name (Optional[str]): The name of the sender.
        recipient_name (str): The name of the recipient.
    """

    code_block_languages: list[str]
    sender_name: Optional[str] = None
    recipient_name: str

    def __init__(
        self,
        *,
        uuid: Optional[UUID] = None,
        code_blocks: list["CodeBlock"],
        sender: Optional["Agent"] = None,
        recipient: "Agent",
    ):
        """Initializes the GenerateCodeExecutionReplyMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            code_blocks (list[CodeBlock]): A list of code blocks.
            sender (Optional[Agent]): The sender of the message.
            recipient (Agent): The recipient of the message.
        """
        code_block_languages = [code_block.language for code_block in code_blocks]

        super().__init__(
            uuid=uuid,
            code_block_languages=code_block_languages,
            sender_name=sender.name if sender else None,
            recipient_name=recipient.name,
        )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints the message details, including the code block languages.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        num_code_blocks = len(self.code_block_languages)
        if num_code_blocks == 1:
            f(
                colored(
                    f"\n>>>>>>>> EXECUTING CODE BLOCK (inferred language is {self.code_block_languages[0]})...",
                    "red",
                ),
                flush=True,
            )
        else:
            f(
                colored(
                    f"\n>>>>>>>> EXECUTING {num_code_blocks} CODE BLOCKS (inferred languages are [{', '.join([x for x in self.code_block_languages])}])...",
                    "red",
                ),
                flush=True,
            )


@wrap_message
class ConversableAgentUsageSummaryNoCostIncurredMessage(BaseMessage):
    """A message class indicating that no cost was incurred from a conversable agent.

    Attributes:
        recipient_name (str):
    """

    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        """Initializes the ConversableAgentUsageSummaryNoCostIncurredMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            recipient (Agent): The recipient of the message.
        """
        super().__init__(uuid=uuid, recipient_name=recipient.name)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints a message indicating no cost incurred from the agent.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(f"No cost incurred from agent '{self.recipient_name}'.")


@wrap_message
class ConversableAgentUsageSummaryMessage(BaseMessage):
    """A message class for summarizing the usage of a conversable agent.

    Attributes:
        recipient_name (str): The name of the recipient.
    """

    recipient_name: str

    def __init__(self, *, uuid: Optional[UUID] = None, recipient: "Agent"):
        """Initializes the ConversableAgentUsageSummaryMessage.

        Args:
            uuid (Optional[UUID]): The UUID of the message.
            recipient (Agent): The recipient of the message.
        """
        super().__init__(uuid=uuid, recipient_name=recipient.name)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        """Prints a message summarizing the usage of the agent.

        Args:
            f (Optional[Callable[..., Any]]): The function to use for printing. Defaults to the built-in print function.
        """
        f = f or print

        f(f"Agent '{self.recipient_name}':")
