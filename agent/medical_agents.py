from typing import Any, Optional
import asyncio
import json

from langgraph.types import Interrupt, Command
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable
from langgraph.types import StateSnapshot

from agent.entity import MedicalEntity
from agent.tools import search_medical_kb
from common import get_logger
from agent.graph_components import graph_components
from agent.graph_state import MedicalState, HumanDecision

logger = get_logger(__name__)


class MedicalAgents:

    def __init__(self):
        self._async_graph: CompiledStateGraph | None = None

    async def get_async_graph(self) -> CompiledStateGraph:
        """Lazily initialize and return the async graph."""
        if self._async_graph is None:
            self._async_graph = await graph_components.async_medical_graph()
        return self._async_graph

    def _make_config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id, "user_id": "medical_user"}}

    def _get_input(self, context: MedicalEntity) -> MedicalState:
        user_message = HumanMessage(
            content=f"请处理医疗实体:\n {json.dumps(context.to_dict(), ensure_ascii=False, indent=2)}"
        )
        return MedicalState(messages=[user_message])

    @traceable(run_type="chain", name="MedicalAgents.start")
    async def start(self, entity: MedicalEntity, thread_id: str) -> dict[str, Any]:
        """Phase 1: Run graph until interrupt, return intermediate result with interrupt info.

        Args:
            entity: Medical entity to process
            thread_id: Unique thread ID tied to the business context (e.g. claim_id + entity index)

        Returns:
            Graph result dict. If interrupted, contains '__interrupt__' key with pending interrupts.
        """
        await search_medical_kb(entity)

        async_graph = await self.get_async_graph()
        result = await async_graph.ainvoke(
            self._get_input(entity),
            config=self._make_config(thread_id),
            context=entity,
            durability="sync",
        )
        return result

    @traceable(run_type="chain", name="MedicalAgents.resume")
    async def resume(self, decision: HumanDecision, thread_id: str) -> dict[str, Any]:
        """Phase 2: Resume graph execution with human decision.

        Args:
            decision: Human-confirmed ICD code and TNM stage
            thread_id: Same thread ID used in start()

        Returns:
            Final graph result after approval nodes complete.
        """
        async_graph = await self.get_async_graph()
        config = self._make_config(thread_id)

        # Get pending interrupts from the saved checkpoint
        state = await async_graph.aget_state(config, subgraphs=True)
        interrupts = []
        for task in state.tasks:
            interrupts.extend(task.interrupts)

        if not interrupts:
            logger.warning(f"No pending interrupts found for thread_id={thread_id}")
            return {}

        resume_value = {interrupt.id: decision for interrupt in interrupts}
        return await async_graph.ainvoke(Command(resume=resume_value), config=config)

    @staticmethod
    def extract_interrupt_info(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract human-readable interrupt info from graph result for frontend display."""
        interrupts: list[Interrupt] = result.get("__interrupt__", [])
        if not interrupts:
            return []

        return [
            {
                "interrupt_id": interrupt.id,
                "graph_name": interrupt.value.get("graph_name", ""),
                "question": interrupt.value.get("question", ""),
                "agent_output": {
                    k: v.to_dict() if hasattr(v, "to_dict") else v
                    for k, v in interrupt.value.get("agent_output", {}).items()
                },
            }
            for interrupt in interrupts
        ]

    # ------------------------------------------------------------------
    # Time Travel: checkpoint query methods
    # ------------------------------------------------------------------

    async def list_checkpoints(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """List checkpoint metadata for a given thread_id."""
        async_graph = await self.get_async_graph()
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = []
        async for snapshot in async_graph.aget_state_history(config, limit=limit):
            checkpoints.append(
                {
                    "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
                    "step": snapshot.metadata.get("step", -1) if snapshot.metadata else -1,
                    "source": snapshot.metadata.get("source", "") if snapshot.metadata else "",
                    "next": list(snapshot.next),
                    "created_at": snapshot.created_at,
                    "has_interrupt": len(snapshot.interrupts) > 0,
                }
            )
        return checkpoints

    async def get_checkpoint_state(self, thread_id: str, checkpoint_id: Optional[str] = None) -> dict[str, Any]:
        """Get full serialized graph state at a specific checkpoint."""
        async_graph = await self.get_async_graph()
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        snapshot = await async_graph.aget_state(config)
        return self._serialize_snapshot(snapshot)

    @staticmethod
    def _serialize_snapshot(snapshot: StateSnapshot) -> dict[str, Any]:
        """Convert a LangGraph StateSnapshot to a JSON-safe dict."""
        values = snapshot.values if snapshot.values else {}

        # Serialize messages
        serialized_messages = []
        for msg in values.get("messages", []):
            m: dict[str, Any] = {
                "type": getattr(msg, "type", "unknown"),
                "content": msg.content if hasattr(msg, "content") else str(msg),
            }
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                m["tool_calls"] = [{"name": tc.get("name", ""), "args": tc.get("args", {})} for tc in msg.tool_calls]
            if hasattr(msg, "name") and msg.name:
                m["name"] = msg.name
            serialized_messages.append(m)

        # Serialize agent_output_dict
        agent_outputs = {}
        for key, output in values.get("agent_output_dict", {}).items():
            if hasattr(output, "to_dict"):
                agent_outputs[key] = output.to_dict()
            elif isinstance(output, dict):
                agent_outputs[key] = output
            else:
                agent_outputs[key] = str(output)

        # Serialize human_decision
        decision = values.get("human_decision")
        human_decision = None
        if decision is not None:
            human_decision = decision.to_dict() if hasattr(decision, "to_dict") else decision

        return {
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id", ""),
            "step": snapshot.metadata.get("step", -1) if snapshot.metadata else -1,
            "source": snapshot.metadata.get("source", "") if snapshot.metadata else "",
            "next": list(snapshot.next),
            "created_at": snapshot.created_at,
            "messages": serialized_messages,
            "agent_output_dict": agent_outputs,
            "human_decision": human_decision,
        }


graph = MedicalAgents()

if __name__ == "__main__":
    import asyncio
    import json
    import time
    from agent.entity import MedicalEntity

    medical_entity = MedicalEntity(
        patient_age=56,
        term_cn="甲状腺乳头状癌",
        term_en="papillary thyroid carcinoma",
        entity_type="diagnosis",
        attributes={
            "tumor_max_diameter_cm": 1.2,
            "is_lymph_metastasis": False,
        },
        description="甲状腺乳头状癌，肿瘤位置: 右叶下极，肿瘤大小: 1.2 cm × 1.0 cm，被膜侵犯: (-)，脉管侵犯: (-)，神经侵犯: (-)，中央区淋巴结未见癌转移 (0/6).",
    )

    thread_id = "claim_0120164504_f58784_entity_0"
    start = time.time()
    results = asyncio.run(graph.start(medical_entity, thread_id))
    print(f"result gotten in seconds: {time.time() - start}")
    print("-" * 40)
    # for k, v in results.get("agent_output_dict", {}).items():
    #     print(f"{k}: {json.dumps(v.to_dict(), ensure_ascii=False, indent=2)}")
    # v = results.get("human_decision", {})
    # print(f"{json.dumps(v.to_dict(), ensure_ascii=False, indent=2)}")
