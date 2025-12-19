"""
RAG Evaluator for comprehensive evaluation of retrieval and generation quality.
Integrates custom metrics and RAGAS framework.
"""

from typing import List, Dict, Any, Literal, Optional
import json
import os

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from ragas.evaluation import EvaluationResult, evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.run_config import RunConfig

from rag.retrieval.retriever import retriever
from rag.generation.llm_service import llm_service
from common import get_logger
from common import init_root_logger

init_root_logger()
logger = get_logger(__name__)


class DeepSeekWrapper(BaseChatModel):

    def __init__(self, base_model: ChatDeepSeek, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_base_model", base_model)

    @property
    def _llm_type(self) -> str:
        return "deepseek_wrapper"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("n", None)
        return self._base_model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("n", None)
        return await self._base_model._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("n", None)
        return self._base_model._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("n", None)
        return self._base_model._astream(messages, stop=stop, run_manager=run_manager, **kwargs)


class RAGEvaluator:
    """
    Comprehensive RAG evaluation framework.
    """

    ragas_metrics = [faithfulness, answer_relevancy, context_recall]
    run_config = RunConfig(
        max_workers=4,
        timeout=300,
        max_retries=10,
    )

    def __init__(self):
        try:
            self.eval_model = DeepSeekWrapper(
                base_model=ChatDeepSeek(
                    model="deepseek-chat",
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    temperature=1.0,
                )
            )
            self.eval_embedding_model = HuggingFaceEmbeddings(model="BAAI/bge-m3")
        except ImportError as e:
            logger.warning(f"RAGAS not available: {e}. Using custom metrics only.")

    def _evaluate(
        self,
        questions: List[str],
        ground_truth_answers: List[str],
        retrieved_docs: List[List[Dict[str, Any]]],
        generated_answers: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate RAG system performance.

        Args:
            questions: List of input questions
            ground_truth_answers: List of ground truth answers
            retrieved_docs: List of retrieved documents for each question
            generated_answers: List of generated answers

        """
        try:
            # Format contexts for RAGAS
            contexts = []
            for docs in retrieved_docs:
                context = [doc.get("text", "") for doc in docs]
                contexts.append(context)

            # Create RAGAS dataset
            data = {
                "question": questions,
                "ground_truth": ground_truth_answers,
                "contexts": contexts,
                "answer": generated_answers,
            }

            dataset = Dataset.from_dict(data)

            logger.info("Running RAGAS evaluation...")
            result = evaluate(
                dataset,
                metrics=self.ragas_metrics,
                llm=self.eval_model,
                embeddings=self.eval_embedding_model,
                run_config=self.run_config,
            )

            logger.info(f"RAGAS evaluation result: {result}")

            try:
                eval_result_path = os.path.join(os.path.dirname(__file__), "eval_result.csv")
                df = result.to_pandas()
                with open(eval_result_path, "w", encoding="utf-8") as f:
                    f.write(df.to_csv(index=False))
                logger.info(f"Saved evaluation result to: {eval_result_path}")
            except Exception as e:
                logger.warning(f"Failed to save eval_result.csv: {e}", exc_info=True)

            return result

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}

    def __call__(
        self,
        eval_file: str,
    ) -> EvaluationResult:
        """
        Evaluate from a JSON evaluation file.

        Expected format:
        {
            "samples": [
                {
                    "question": "...",
                    "ground_truth_answer": "...",
                    "retrieved_docs": [...],
                    "generated_answer": "...",
                }
            ]
        }
        """
        logger.info(f"Loading evaluation data from: {eval_file}")

        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = []
        ground_truth_answers = []
        retrieved_docs = []
        generated_answers = []

        for sample in data.get("samples", []):
            questions.append(sample["question"])
            ground_truth_answers.append(sample["ground_truth_answer"])
            retrieved_docs.append(sample["retrieved_docs"])
            generated_answers.append(sample["generated_answer"])

        return self._evaluate(
            questions=questions,
            ground_truth_answers=ground_truth_answers,
            retrieved_docs=retrieved_docs,
            generated_answers=generated_answers,
        )

    def generate_answer(
        self,
        eval_file: str,
        kb_id: str = "default_kb",
        top_k: int = 5,
        temperature: float = 1.0,
        mode: Literal["dense", "hybrid"] = "hybrid",
        min_score: float = 0.0,
    ) -> None:

        logger.info(f"Loading evaluation data from: {eval_file}")

        # Read the question\ground truth answer\retrieved docs (filter by doc_id)
        filters = {"doc_id": "7f203234-0620-4a56-a362-005d40d9fa14"}
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data.get("samples", [])
        logger.info(f"Processing {len(samples)} samples...")
        retrieved_columns = ["text", "score"]

        # Process each sample
        for i, sample in enumerate(samples, 1):
            question = sample["question"]
            logger.info(f"[{i}/{len(samples)}] Processing: {question[:50]}...")

            try:
                # Retrieve documents
                search_result = retriever.search(query=question, kb_id=kb_id, top_k=top_k, filters=filters, mode=mode)

                # Update retrieved_docs
                retrieved = search_result.get("results", [])
                filtered_retrieved = [chunk for chunk in retrieved if chunk.get("score") > min_score]
                retrieved_docs = []
                for retrieved_doc in filtered_retrieved:
                    retrieved_doc = {col: retrieved_doc.get(col, "") for col in retrieved_columns}
                    retrieved_docs.append(retrieved_doc)
                sample["retrieved_docs"] = retrieved_docs
                logger.info(f"  Retrieved {len(filtered_retrieved)} validate documents")

                # Generate answer
                query_to_use = search_result.get("query_to_use", question)
                generation_result = llm_service.answer_question(
                    question=query_to_use, context=retrieved, temperature=temperature
                )
                generated = generation_result.get("answer", "")
                sample["generated_answer"] = generated
                logger.info(f"  Generated answer: {generated[:50]}...")

            except Exception as e:
                logger.error(f"  Failed to process sample: {e}", exc_info=True)
                sample["retrieved_docs"] = []
                sample["generated_answer"] = ""

        # Save all results once after processing all samples
        logger.info(f"Saving results to: {eval_file}")
        try:
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully saved {len(samples)} samples to {eval_file}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    print("*****************Rag-fintech Benchmark*****************")
    evaluator = RAGEvaluator()
    eval_file = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    # evaluator.generate_answer(eval_file)
    result = evaluator(eval_file)
