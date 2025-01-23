# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import copy
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from ....io.base import IOStream
from ....messages.agent_messages import WaitingForTaskMessage
from ...agent import Agent
from ...conversable_agent import ConversableAgent
from .platform_configs import BasePlatformConfig, ReplyMonitorConfig
from .platform_errors import PlatformError

__NESTED_CHAT_EXECUTOR_PREFIX__ = "Send the message (if needed)"
__DECISION_AGENT_SYSTEM_MESSAGE__ = "You are an AI assistant that determines if there's sufficient need to send a message and, if so, crafts the message."


class PlatformMessageDecision(BaseModel):
    """The structured output format for platform message decisions."""

    should_send: bool = Field(
        description="Whether the message should actually be sent to the platform, if in doubt, do not send.",
    )
    message_to_post: str = Field(description="The message that should be posted to the platform")


class PlatformDecisionAgent(ConversableAgent):
    """Agent responsible for deciding what/whether to send to platform."""

    def __init__(
        self,
        platform_name: str,
        llm_config: Dict[str, Any],
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        *args,
        **kwargs,
    ):
        system_message = (
            f"You are a decision maker for {platform_name} communications. "
            "You evaluate messages and decide what should be sent to the platform. There must be explicit instruction to send a message to the platform. "
            "Consider message appropriateness, platform limitations, and context."
        )

        structured_config_list = copy.deepcopy(llm_config)
        for config in structured_config_list["config_list"]:
            config["response_format"] = PlatformMessageDecision

        super().__init__(
            name=f"{platform_name}DecisionMaker",
            system_message=system_message,
            llm_config=structured_config_list,
            human_input_mode=human_input_mode,
            *args,
            **kwargs,
        )


class PlatformExecutorAgent(ConversableAgent):
    """Agent responsible for executing platform communications."""

    def __init__(self, platform_config: BasePlatformConfig, reply_monitor_config: Optional[ReplyMonitorConfig] = None):
        system_message = (
            "You are a platform communication executor. "
            "You handle sending messages and receiving replies from the platform."
        )

        super().__init__(name=self.__class__.__name__, system_message=system_message)

        self.platform_config = platform_config
        self.reply_monitor_config = reply_monitor_config

    @abstractmethod
    def send_to_platform(self, message: str) -> Tuple[str, Optional[str]]:
        """Send message to platform.

        Args:
            message: The message to send

        Returns:
            Tuple[str, Optional[str]]: Status message and message ID if successful
        """
        pass

    @abstractmethod
    def wait_for_reply(self, msg_id: str) -> List[dict]:
        """Wait for reply from platform.

        Args:
            msg_id: Message ID to monitor for replies

        Returns:
            List of reply messages
        """
        pass

    @abstractmethod
    def cleanup_monitoring(self, msg_id: str):
        """Clean up any monitoring resources used for waiting for the reply

        Args:
            msg_id: Message ID to cleanup monitoring for
        """
        pass


class CommsPlatformAgent(ConversableAgent):
    """Base class for communication platform agents."""

    DEFAULT_SUMMARY_PROMPT = (
        "Analyze the interaction about sending a message to the platform. "
        "If a message was sent, include its content and any reply received. "
        "If no message was sent, explain why."
    )

    def __init__(
        self,
        name: str,
        platform_config: "BasePlatformConfig",
        executor_agent: "PlatformExecutorAgent",
        send_config: Dict[str, Any],
        message_to_send: Optional[
            callable
        ] = None,  # The function to determine the message to send, returns None to indicate do not send a message, otherwise determined automatically
        reply_monitor_config: Optional["ReplyMonitorConfig"] = None,
        auto_reply: str = "Message sent",
        system_message: Optional[str] = None,
        llm_config: Optional[Union[dict, Literal[False]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, system_message=system_message, llm_config=llm_config, *args, **kwargs)

        # self.message_to_send = message_to_send
        # Are we using an LLM to decide and create the message?
        self.message_decision_creation_llm = message_to_send is None
        self.message_to_send = message_to_send

        # Create our specialized agents
        self.executor_agent = executor_agent
        self.decision_agent = PlatformDecisionAgent(self.__class__.__name__.replace("Agent", ""), llm_config=llm_config)

        if message_to_send:
            # If we are using the callable to determine the message to send, update the decision_agent to use their function instead of the LLM
            self.decision_agent.register_reply(
                trigger=[ConversableAgent],
                reply_func=self._process_message_to_send,
                remove_other_reply_funcs=True,
            )

        # Register the reply function on the executor agent that will trigger in the second chat in the nested chat sequence
        self.executor_agent.register_reply(
            trigger=[ConversableAgent],  # , None],
            reply_func=self._executor_reply_function,
            remove_other_reply_funcs=True,
        )

        # Register the nested chat sequence on the current agent to handle the decision and execution and return the result of the process to the parent agent
        self.register_nested_chats(
            [
                {
                    # First: Decision agent decides whether to send a message and crafts the message
                    "sender": self,
                    "recipient": self.decision_agent,
                    "message": self._prepare_decision_message
                    if self.message_decision_creation_llm
                    else "Determine if we need to send a message and what it should be.",
                    "summary_method": "last_msg",
                    "max_turns": 1,
                },
                {
                    # Second: Executor handles platform communication if decision is to send
                    "sender": self,
                    "recipient": self.executor_agent,
                    "message": __NESTED_CHAT_EXECUTOR_PREFIX__,  # Note: We need some text here to keep it in the nested chat queue
                    "max_turns": 1,
                },
            ],
            trigger=[ConversableAgent],
            position=0,
            use_async=False,
        )

        # Reply configuration
        self.reply_monitor_config = reply_monitor_config

    def _process_message_to_send(
        self,
        recipient: ConversableAgent,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Execute the message_to_send and returns our structured output for compatibility with the executor agent's processing."""

        # Get the message to send
        message = self.message_to_send(self, messages)

        # Put into our structured output format so the executor can process it in _executor_reply_function
        decision = PlatformMessageDecision(should_send=message is not None, message_to_post=message or "")

        return True, decision.model_dump_json()

    def _executor_message_validation(
        self, message_content: str
    ) -> Tuple[bool, Optional[PlatformMessageDecision], Optional[str]]:
        """Shared validation logic for both sync and async versions."""
        # Check message prefix
        if not message_content.startswith(f"{__NESTED_CHAT_EXECUTOR_PREFIX__}\nContext: \n"):
            return False, None, "Error, the workflow did not work correctly, message not sent."

        message_content = message_content.replace(f"{__NESTED_CHAT_EXECUTOR_PREFIX__}\nContext: \n", "")

        # Parse decision
        try:
            decision = PlatformMessageDecision.model_validate_json(message_content)
            return True, decision, None
        except ValidationError:
            return (
                False,
                None,
                "Couldn't parse the structured output, ensure you are using a client and model that supports structured output.",
            )

    def _executor_reply_function(
        self,
        recipient: ConversableAgent,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Enhanced executor reply function that handles sending and waiting for replies.

        Handles all platform-specific errors and provides detailed status reporting.
        Returns a tuple of (completed, response_message).
        """
        message_content: str = messages[-1]["content"]

        # Use shared validation
        is_valid, decision, error_msg = self._executor_message_validation(message_content)
        if not is_valid:
            return True, error_msg

        if decision.should_send:
            try:
                # Send message and get tracking ID
                status, msg_id = self.executor_agent.send_to_platform(decision.message_to_post)

                if not msg_id:
                    return True, f"Message sent but unable to track replies: {status}"

                # Initialize response with sent message status
                response = f"Message sent successfully:\n{decision.message_to_post}"

                # Wait for replies if configured
                replies = []
                if self.reply_monitor_config and msg_id:
                    try:
                        iostream = IOStream.get_default()
                        iostream.send(
                            WaitingForTaskMessage(
                                task_details=f"{self.name} has sent the message and is waiting for replies, max {self.reply_monitor_config.timeout_minutes} min(s){(' or ' + str(self.reply_monitor_config.max_reply_messages) + ' replies') if self.reply_monitor_config.max_reply_messages else ''}"
                            )
                        )

                        # Wait for replies...
                        replies = self.executor_agent.wait_for_reply(msg_id)

                        if replies:
                            response += "\n\nReplies received:\n" + "\n".join(replies)
                        else:
                            response += "\n\nNo replies received"

                    except PlatformError as e:
                        response += f"\n\nError while waiting for replies: {str(e)}"

                    except Exception as e:
                        # Non-handled exceptions
                        response += f"\n\nUnexpected error while waiting for replies: {str(e)}"

                    finally:
                        try:
                            self.executor_agent.cleanup_monitoring(msg_id)
                        except Exception as cleanup_error:
                            response += f"\n\nWarning: Error during cleanup: {str(cleanup_error)}"

                return True, response

            # Exception handling for sending messages
            except PlatformError as e:
                return True, f"Error: {str(e)}"

            except Exception as e:
                # Non-handled exceptions
                return True, f"Error: Unexpected error while sending message - {str(e)}"

        return True, "No message was sent based on decision"

    def _prepare_decision_message(self, recipient: Agent, messages: List[Dict[str, Any]], sender: Agent, config) -> str:
        """Prepare message for decision agent based on conversation history."""

        # Compile the list of messages into a single string with the name value and the content value. Ignore anything where the role contains tool'
        # name and/or role may not be present
        message_history = "\n".join(
            [
                f"{m.get('name', 'Unknown')}: {m.get('content', 'No content')}"
                for m in messages
                if "tool" not in m.get("role", "")
            ]
        )

        # Prepare message for decision agent
        return f"Decide whether you should send a message and, if so, what the message should be based on this chat history:\n{message_history}"
