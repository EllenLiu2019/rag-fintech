import os
import functools
from contextlib import asynccontextmanager
import uuid

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from agent.entity import MedicalEntity
from agent.graph_state import MedicalState, Step, AgentOutput
from common import get_model_registry
from common.config_utils import get_base_config


@asynccontextmanager
async def medical_agent():
    from agent.medical_agents import graph

    yield graph.graph


class GraphComponents:

    def __init__(self):
        rdb_config = get_base_config("postgresql", {})
        self.url = f"postgresql://{rdb_config.get('username')}:{rdb_config.get('password')}@{rdb_config.get('host')}:{rdb_config.get('port')}/{rdb_config.get('database')}?sslmode={rdb_config.get('sslmode')}"
        db_conn = Connection.connect(self.url, autocommit=True, prepare_threshold=0, row_factory=dict_row, connect_timeout=10)
        self.checkpointer = PostgresSaver(db_conn)
        self.store = PostgresStore(db_conn)
        self.async_checkpointer: AsyncPostgresSaver | None = None
        self.async_store: AsyncPostgresStore | None = None

        model_config = get_model_registry().get_chat_model("qa_lite").to_dict()
        self.configurable_model = init_chat_model(
            model=model_config["model_name"],
            model_provider=model_config["provider"],
            base_url=model_config["base_url"],
            api_key=os.getenv(f"{model_config['provider'].upper()}_API_KEY", "EMPTY"),
            temperature=0.0,
        )

    async def async_init(self):
        """Initialize async checkpointer and store with a connection pool.

        Uses AsyncConnectionPool to support concurrent pipeline operations
        from parallel subgraphs.
        """
        pool = AsyncConnectionPool(
            self.url,
            min_size=2,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open(wait=True)
        self.async_checkpointer = AsyncPostgresSaver(pool)
        self.async_store = AsyncPostgresStore(pool)

    def _build_encode_graph(self, *, use_async: bool = False):
        from agent.medical_alignment_agent import MedicalAlignmentAgent
        from agent.medical_encode_agent import MedicalEncodeAgent

        agents = {
            Step.ENCODE_AGENT: MedicalEncodeAgent(self.configurable_model),
            Step.ALIGN_AGENT: MedicalAlignmentAgent(self.configurable_model),
        }
        graph = StateGraph(
            state_schema=MedicalState,
            context_schema=MedicalEntity,
        )
        sub_tools = []
        tool_to_step = {}
        for tool_name, tool in agents[Step.ALIGN_AGENT].tools.items():
            tool_to_step[tool_name] = Step.ALIGN_AGENT
            sub_tools.append(tool)
        tool_node = ToolNode(sub_tools)

        agent_attr = "async_agent" if use_async else "agent"
        tool_router_fn = _async_tool_router if use_async else _tool_router

        graph.add_node(Step.ENCODE_AGENT.value, getattr(agents[Step.ENCODE_AGENT], agent_attr))
        graph.add_node(Step.ALIGN_AGENT.value, getattr(agents[Step.ALIGN_AGENT], agent_attr))
        graph.add_node(Step.TOOL_NODE.value, tool_node)
        graph.add_edge(Step.START.value, Step.ENCODE_AGENT.value)
        graph.add_edge(Step.ENCODE_AGENT.value, Step.ALIGN_AGENT.value)

        end_step: Step = Step.END
        if agents[Step.ALIGN_AGENT].approval_node:
            end_step = Step.APPROVAL
            graph.add_node(Step.APPROVAL.value, agents[Step.ALIGN_AGENT].approval_node)
            graph.add_edge(Step.APPROVAL.value, Step.END.value)

        graph.add_conditional_edges(
            Step.ALIGN_AGENT.value,
            functools.partial(_router, tool_step=Step.TOOL_NODE, agent_step=Step.ALIGN_AGENT, end_step=end_step),
            path_map={Step.TOOL_NODE: Step.TOOL_NODE.value, end_step: end_step.value},
        )

        graph.add_conditional_edges(
            Step.TOOL_NODE.value,
            functools.partial(tool_router_fn, tool_to_step=tool_to_step, invoker=Step.ALIGN_AGENT.value),
            path_map={Step.ALIGN_AGENT: Step.ALIGN_AGENT.value},
        )

        return graph.compile(checkpointer=True)

    def _build_stage_graph(self, *, use_async: bool = False):
        from agent.medical_stage_agent import MedicalStageAgent

        agents = {
            Step.STAGE_AGENT: MedicalStageAgent(self.configurable_model),
        }

        graph = StateGraph(
            state_schema=MedicalState,
            context_schema=MedicalEntity,
        )
        tools = []
        tool_to_step = {}
        for tool_name, tool in agents[Step.STAGE_AGENT].tools.items():
            tool_to_step[tool_name] = Step.STAGE_AGENT
            tools.append(tool)
        tool_node = ToolNode(tools)

        agent_attr = "async_agent" if use_async else "agent"
        tool_router_fn = _async_tool_router if use_async else _tool_router

        graph.add_node(Step.STAGE_AGENT.value, getattr(agents[Step.STAGE_AGENT], agent_attr))
        graph.add_node(Step.TOOL_NODE.value, tool_node)
        graph.add_edge(Step.START.value, Step.STAGE_AGENT.value)

        end_step: Step = Step.END
        if agents[Step.STAGE_AGENT].approval_node:
            end_step = Step.APPROVAL
            graph.add_node(Step.APPROVAL.value, agents[Step.STAGE_AGENT].approval_node)
            graph.add_edge(Step.APPROVAL.value, Step.END.value)

        graph.add_conditional_edges(
            Step.STAGE_AGENT.value,
            functools.partial(_router, tool_step=Step.TOOL_NODE, agent_step=Step.STAGE_AGENT, end_step=end_step),
            path_map={Step.TOOL_NODE: Step.TOOL_NODE.value, end_step: end_step.value},
        )

        graph.add_conditional_edges(
            Step.TOOL_NODE.value,
            functools.partial(tool_router_fn, tool_to_step=tool_to_step, invoker=Step.STAGE_AGENT.value),
            path_map={Step.STAGE_AGENT: Step.STAGE_AGENT.value},
        )

        return graph.compile(checkpointer=True)

    def medical_graph(self):

        graph = StateGraph(
            state_schema=MedicalState,
            context_schema=MedicalEntity,
        )
        graph.add_node("encode_graph", self._build_encode_graph(use_async=False))
        graph.add_node("stage_graph", self._build_stage_graph(use_async=False))
        graph.add_edge(Step.START.value, "encode_graph")
        graph.add_edge(Step.START.value, "stage_graph")
        graph.add_edge("encode_graph", Step.END.value)
        graph.add_edge("stage_graph", Step.END.value)

        return graph.compile(checkpointer=self.checkpointer, store=self.store)

    async def async_medical_graph(self):
        """Async version of medical_graph. Requires async_init() to be called first."""
        if not self.async_checkpointer or not self.async_store:
            await self.async_init()

        graph = StateGraph(
            state_schema=MedicalState,
            context_schema=MedicalEntity,
        )
        graph.add_node("encode_graph", self._build_encode_graph(use_async=True))
        graph.add_node("stage_graph", self._build_stage_graph(use_async=True))
        graph.add_edge(Step.START.value, "encode_graph")
        graph.add_edge(Step.START.value, "stage_graph")
        graph.add_edge("encode_graph", Step.END.value)
        graph.add_edge("stage_graph", Step.END.value)

        return graph.compile(checkpointer=self.async_checkpointer, store=self.async_store)


def _tool_router(
    state: MedicalState,
    tool_to_step: dict[str, Step],
    runtime: Runtime[MedicalEntity],
    invoker: str,
) -> Step:
    store: BaseStore = runtime.store
    last_message = state.last_message()
    if isinstance(last_message, ToolMessage):
        saved_messages = {
            "id": last_message.id,
            "content": last_message.content,
            "name": last_message.name,
            "tool_call_id": last_message.tool_call_id,
            "status": last_message.status,
            "type": last_message.type,
            "agent_output_dict": {k: v.to_dict() for k, v in state.agent_output_dict.items()},
        }
        store.put(("medical", f"{invoker}_tool"), str(uuid.uuid4()), saved_messages)
        return tool_to_step.get(last_message.name)


async def _async_tool_router(
    state: MedicalState,
    tool_to_step: dict[str, Step],
    runtime: Runtime[MedicalEntity],
    invoker: str,
) -> Step:
    store: BaseStore = runtime.store
    last_message = state.last_message()
    if isinstance(last_message, ToolMessage):
        saved_messages = {
            "id": last_message.id,
            "content": last_message.content,
            "name": last_message.name,
            "tool_call_id": last_message.tool_call_id,
            "status": last_message.status,
            "type": last_message.type,
            "agent_output_dict": {k: v.to_dict() for k, v in state.agent_output_dict.items()},
        }
        await store.aput(("medical", f"{invoker}_tool"), str(uuid.uuid4()), saved_messages)
        return tool_to_step.get(last_message.name)


def _router(state: MedicalState, tool_step: Step, agent_step: Step, end_step: Step) -> Step:

    def _valid_tool_call() -> bool:
        if not state.agent_output_dict:
            return False
        agent_output: AgentOutput = state.agent_output_dict.get(agent_step.value, {})
        return agent_output and agent_output.tool_calls

    last_message = state.last_message()
    if last_message.tool_calls:
        return tool_step

    if not _valid_tool_call():
        return tool_step

    return end_step


graph_components = GraphComponents()
