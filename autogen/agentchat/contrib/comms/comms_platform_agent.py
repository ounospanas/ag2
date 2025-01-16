# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from ...agent import Agent
from ...conversable_agent import ConversableAgent
from .platform_configs import BasePlatformConfig, ReplyConfig
from .platform_errors import PlatformError

__NESTED_CHAT_EXECUTOR_PREFIX__ = "Send the message (if needed)"


class PlatformMessageDecision(BaseModel):
    """The structured output format for platform message decisions."""

    message_to_post: str = Field(description="The message that should be posted to the platform")
    should_send: bool = Field(
        description="Whether the message should actually be sent to the platform, if in doubt, do not send.",
    )


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
            "You evaluate messages and decide what should be sent to the platform. "
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

    def __init__(self, platform_config: BasePlatformConfig, reply_config: Optional[ReplyConfig] = None):
        system_message = (
            "You are a platform communication executor. "
            "You handle sending messages and receiving replies from the platform."
        )

        super().__init__(name=self.__class__.__name__, system_message=system_message)

        self.platform_config = platform_config
        self.reply_config = reply_config

        # Register function for actual platform interaction
        self.register_function({"send_to_platform": self._send_to_platform, "wait_for_reply": self._wait_for_reply})

    @abstractmethod
    async def _send_to_platform(self, message: str) -> str:
        """Send message to platform."""
        pass

    @abstractmethod
    async def _wait_for_reply(self, msg_id: str) -> str:
        """Wait for reply from platform."""
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
        reply_config: Optional["ReplyConfig"] = None,
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
                    "message": __NESTED_CHAT_EXECUTOR_PREFIX__,  # As the reply function doesn't need the message but we need text here to keep it in the nested chat queue
                    "max_turns": 1,
                },
            ],
            trigger=[ConversableAgent],
            position=0,
        )

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
        decision = PlatformMessageDecision(message_to_post=message or "", should_send=message is not None)

        return True, decision.model_dump_json()

    def _executor_reply_function(
        self,
        recipient: ConversableAgent,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Process the results from the decision agent and perform an action, or not."""
        message_content: str = messages[-1]["content"]

        # We should have a message that starts with the second nested chat message text, "PLACEHOLDER", and the inbuilt carry over prefix "Context: \n"
        if not message_content.startswith(f"{__NESTED_CHAT_EXECUTOR_PREFIX__}\nContext: \n"):
            return True, "Error, the workflow did not work correctly, message not sent."

        message_content = message_content.replace(f"{__NESTED_CHAT_EXECUTOR_PREFIX__}\nContext: \n", "")

        try:
            decision = PlatformMessageDecision.model_validate_json(message_content)

        except ValidationError:
            return (
                True,
                "Couldn't parse the structured output, ensure you are using a client and model that supports structured output.",
            )

        if decision.should_send:
            # Handle asynchronous functions in here as this chat may not be running async
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def _send_with_error_handling():
                return await self.executor_agent._send_to_platform(decision.message_to_post)

            try:
                # Run the async function in the event loop with proper error handling
                _ = loop.run_until_complete(_send_with_error_handling())
                return True, f"Message sent successfully:\n{decision.message_to_post}"
            except PlatformError as e:
                return True, f"Message not sent due to platform exception: {str(e)}"
            except Exception as e:
                return True, f"Message not sent due to unexpected error: {str(e)}"

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


'''
    def _prepare_executor_message(self, recipient: Agent, messages: List[Dict[str, Any]], sender: Agent, config) -> str:
        """Prepare message for executor based on decision agent's response."""
        if not messages:
            return None

        try:
            # Get last message which should be decision agent's structured response
            decision = PlatformMessageDecision.model_validate_json(messages[-1]["content"])

            # If we aren't sending a message, set the message to none
            if not decision.should_send:
                return None

            # If we should send, create the execution message
            return f'send_to_platform("{decision.message_to_post}")'

        except Exception:
            return "Couldn't convert the structured output, ensure you are using a client and model that supports structured output."
'''
