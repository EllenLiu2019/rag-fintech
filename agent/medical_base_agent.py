import functools
import logging
from typing import Callable
import uuid

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain.messages import SystemMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from agent.tools import extract_ai_message
from agent.graph_state import MedicalState
from agent.entity import MedicalEntity


class BaseMedicalAgent:
    def __init__(
        self,
        configurable_model: _ConfigurableModel,
        prompt: str,
        *,
        tools: dict[str, Callable] | None = None,
        logger: logging.Logger,
        tag: str = "",
    ):

        self._logger = logger
        self._tools = tools or {}
        model = (
            configurable_model.bind_tools(list(self._tools.values()), strict=True)
            if self._tools
            else configurable_model
        )
        self._agent = functools.partial(agent_node, agent=model, prompt=prompt, func=self.post_process, tag=tag)
        self._aagent = functools.partial(async_agent_node, agent=model, prompt=prompt, func=self.post_process, tag=tag)

    @property
    def tools(self) -> dict[str, Callable]:
        return self._tools

    @property
    def agent(self):
        return self._agent

    @property
    def async_agent(self):
        return self._aagent

    def post_process(self, message: AIMessage, state: MedicalState, context: MedicalEntity):
        response = extract_ai_message(message)
        if not response:
            self._logger.warning("No AI message found in agent result")
            return

        try:
            self._post_process(response, state, context)
        except Exception as e:
            self._logger.warning(f"Failed to post process agent output: {e}, ai_message: {message}")

    def _post_process(
        self,
        response: dict,
        state: MedicalState,
        context: MedicalEntity,
    ):
        raise NotImplementedError

    def approval_node(self, state: MedicalState) -> MedicalState:
        raise NotImplementedError


def agent_node(
    state: MedicalState,
    runtime: Runtime[MedicalEntity],
    agent: Runnable,
    prompt: str,
    func: Callable | None = None,
    **kwargs,
) -> MedicalState:

    store: BaseStore = runtime.store

    messages: AIMessage = agent.invoke([SystemMessage(content=prompt)] + state.messages)
    if func and not messages.tool_calls:
        func(messages, state, runtime.context)

    saved_messages = {
        "id": messages.id,
        "content": messages.content,
        "additional_kwargs": messages.additional_kwargs,
        "type": messages.type,
        "usage_metadata": messages.usage_metadata,
        "tool_calls": [
            {
                "name": tool_call.get("name"),
                "args": tool_call.get("args"),
                "type": tool_call.get("type"),
                "id": tool_call.get("id"),
            }
            for tool_call in messages.tool_calls
        ],
        "agent_output_dict": {k: v.to_dict() for k, v in state.agent_output_dict.items()},
    }
    tag = kwargs.get("tag")
    store.put(("medical", tag), str(uuid.uuid4()), saved_messages)

    return MedicalState(
        messages=[messages],
        agent_output_dict=state.agent_output_dict,
        human_decision=state.human_decision,
    )


async def async_agent_node(
    state: MedicalState,
    runtime: Runtime[MedicalEntity],
    agent: Runnable,
    prompt: str,
    func: Callable | None = None,
    **kwargs,
) -> MedicalState:

    store: BaseStore = runtime.store

    message: AIMessage = await agent.ainvoke([SystemMessage(content=prompt)] + state.messages)
    if func and not message.tool_calls:
        func(message, state, runtime.context)

    saved_messages = {
        "id": message.id,
        "content": message.content,
        "additional_kwargs": message.additional_kwargs,
        "type": message.type,
        "usage_metadata": message.usage_metadata,
        "tool_calls": [
            {
                "name": tool_call.get("name"),
                "args": tool_call.get("args"),
                "type": tool_call.get("type"),
                "id": tool_call.get("id"),
            }
            for tool_call in message.tool_calls
        ],
        "agent_output_dict": {k: v.to_dict() for k, v in state.agent_output_dict.items()},
    }
    tag = kwargs.get("tag")
    await store.aput(("medical", tag), str(uuid.uuid4()), saved_messages)

    return MedicalState(
        messages=[message],
        agent_output_dict=state.agent_output_dict,
        human_decision=state.human_decision,
    )
