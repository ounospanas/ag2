# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import copy
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from ...agent import Agent
from ...conversable_agent import ConversableAgent
from .platform_configs import BasePlatformConfig, ReplyConfig


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

        super().__init__(name="PlatformExecutor", system_message=system_message)

        self.platform_config = platform_config
        self.reply_config = reply_config

        # Register function for actual platform interaction
        self.register_function({"send_to_platform": self._send_to_platform, "wait_for_reply": self._wait_for_reply})

    @abstractmethod
    def _send_to_platform(self, message: str) -> str:
        """Send message to platform."""
        pass

    @abstractmethod
    def _wait_for_reply(self, msg_id: str) -> str:
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
        send_config: Dict[str, Any],
        executor_agent: "PlatformExecutorAgent",
        reply_config: Optional["ReplyConfig"] = None,
        auto_reply: str = "Message sent",
        system_message: Optional[str] = None,
        llm_config: Optional[Union[dict, Literal[False]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, system_message=system_message, llm_config=llm_config, *args, **kwargs)

        # Deep copy the llm_config as we'll be adding a response format to it
        # decision_llm_config = copy.deepcopy(self.llm_config)

        # Create our specialized agents
        self.decision_agent = PlatformDecisionAgent(self.__class__.__name__.replace("Agent", ""), llm_config=llm_config)
        self.executor = executor_agent

        # Register the nested chat sequence on the current agent
        self.register_nested_chats(
            [
                {
                    # First: Decision agent evaluates and decides
                    "sender": self,
                    "recipient": self.decision_agent,
                    "message": self._prepare_decision_message,
                    # "summary_method": "last_msg",
                    "max_turns": 1,
                },
                {
                    # Second: Executor handles platform communication if decision is to send
                    "sender": self.decision_agent,
                    "recipient": self.executor,
                    # "message": self._prepare_executor_message,
                    "message": "PLACEHOLDER",  # As the reply function doesn't need the message but we need text here to keep it in the nested chat queue
                    # "summary_method": "last_msg",
                    "max_turns": 1,
                },
            ],
            trigger=[ConversableAgent],  # , None],  # Respond to any agent or direct messages
            # reply_func_from_nested_chats=self._process_nested_chat_results,
            position=0,
        )

        # Register the reply function on the executor agent that will trigger in the second chat in the nested chat sequence
        self.executor.register_reply(
            trigger=[ConversableAgent],  # , None],
            reply_func=self._executor_reply_function,
            remove_other_reply_funcs=True,
        )

    def _executor_reply_function(
        self,
        recipient: ConversableAgent,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Process the results from the decision agent and perform an action, or not."""
        # If no messages, return
        if not messages:
            return True, "No action taken"

        # Extract the relevant information from chat results
        # This might need adjustment based on exact chat_results structure
        return True, "Communication completed"

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
            return "Need to return a string so this chat remains a valid chat to execute."

    '''
    def _process_nested_chat_results(self, chat_queue: List[Dict],
                recipient: ConversableAgent,
                messages: Optional[List[Dict]] = None,
                sender: Optional[Agent] = None,
                config: Optional[Any] = None) -> Tuple[bool, Union[str, Dict, None]]:
        """Process the results from the nested chat sequence."""
        #if not chat_results:
            # return True, "No action taken"

        # Extract the relevant information from chat results
        # This might need adjustment based on exact chat_results structure
        return True, "Communication completed"
    '''
