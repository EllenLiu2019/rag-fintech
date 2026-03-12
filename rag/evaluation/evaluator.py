"""
RAG Evaluator for comprehensive evaluation of retrieval and generation quality.
Integrates custom metrics and RAGAS framework.
"""

from typing import List, Dict, Any, Optional
import json
import os
import concurrent.futures
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from ragas.evaluation import EvaluationResult, evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, context_recall, answer_correctness
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

    ragas_metrics = [context_recall, answer_correctness, faithfulness]
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
                    temperature=0.3,
                    max_tokens=8192,
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
        max_workers: int = 5,
        filters: Optional[Dict] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Generate answers for all samples in the evaluation file using thread pool.

        Args:
            eval_file: Path to evaluation JSON file
            max_workers: Maximum number of worker threads (default: 5)
            filters: Optional filters for retrieval (default: doc_id filter)
            timeout: Timeout per task in seconds (default: 300)

        Returns:
            Statistics dictionary with processing results
        """
        logger.info(f"Loading evaluation data from: {eval_file}")

        # Use provided filters or default
        filters = filters or {"doc_id": "a13a9cea-6bff-4875-9772-f4b2645fe82f"}
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data.get("samples", [])
        total_samples = len(samples)

        if total_samples == 0:
            logger.warning("No samples found in evaluation file")
            return {"total": 0, "success": 0, "failed": 0, "timeout": 0}

        logger.info(f"Processing {total_samples} samples with {max_workers} workers")

        # Statistics
        stats = {
            "total": total_samples,
            "success": 0,
            "failed": 0,
            "timeout": 0,
        }

        # Use thread pool for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(
                    self._generate_answer,
                    sample=sample,
                    filters=filters,
                    index=i + 1,
                    total=total_samples,
                ): (i, sample)
                for i, sample in enumerate(samples)
            }

            # Process completed tasks with error handling
            from tqdm import tqdm

            with tqdm(total=total_samples, desc="Generating answers") as pbar:
                for future in concurrent.futures.as_completed(future_to_sample):
                    index, sample = future_to_sample[future]
                    try:
                        future.result(timeout=timeout)
                        stats["success"] += 1
                        pbar.set_postfix({"success": stats["success"], "failed": stats["failed"]})
                    except concurrent.futures.TimeoutError:
                        stats["timeout"] += 1
                        logger.error(f"Sample {index} timed out after {timeout}s")
                        sample["retrieved_docs"] = []
                        sample["generated_answer"] = ""
                    except Exception as e:
                        stats["failed"] += 1
                        logger.error(f"Sample {index} failed: {e}", exc_info=True)
                        # Ensure sample has default values even if _generate_answer partially failed
                        if "retrieved_docs" not in sample:
                            sample["retrieved_docs"] = []
                        if "generated_answer" not in sample:
                            sample["generated_answer"] = ""
                    finally:
                        pbar.update(1)

        # Compute intent accuracy
        intent_total = 0
        intent_correct = 0
        for sample in samples:
            expected = sample.get("type")
            predicted = sample.get("predicted_intent")
            if expected and predicted:
                intent_total += 1
                if expected == predicted:
                    intent_correct += 1
        if intent_total > 0:
            stats["intent_accuracy"] = round(intent_correct / intent_total, 4)
            stats["intent_total"] = intent_total
            stats["intent_correct"] = intent_correct
            logger.info(
                f"Intent accuracy: {intent_correct}/{intent_total} = {stats['intent_accuracy']:.2%}"
            )

        logger.info(f"Saving results to: {eval_file}")
        try:
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully saved {total_samples} samples to {eval_file}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}", exc_info=True)
            raise

        logger.info(f"Processing complete: {stats}")
        return stats

    def _generate_answer(
        self,
        sample: dict,
        filters: dict = None,
        index: int = None,
        total: int = None,
    ) -> None:
        """
        Generate answer for a single sample.

        Args:
            sample: Sample dictionary to process (modified in-place)
            filters: Optional filters for retrieval
            index: Optional sample index for logging
            total: Optional total count for logging
        """
        question = sample["question"]

        if index is not None and total is not None:
            logger.info(f"[{index}/{total}] Processing: {question[:50]}...")

        try:
            search_result = asyncio.run(
                retriever.search(
                    query=question,
                    kb_id="default_kb",
                    top_k=5,
                    filters=filters,
                    mode="hybrid",
                )
            )

            predicted_intent = search_result.get("intent", "fact")
            sample["predicted_intent"] = predicted_intent

            retrieved = search_result.get("results", [])
            filtered_retrieved = [chunk for chunk in retrieved if chunk.get("score", 0.1) > 0.0]
            retrieved_docs = []
            for retrieved_doc in filtered_retrieved:
                retrieved_doc = {col: retrieved_doc.get(col, "") for col in ["text", "score"]}
                retrieved_docs.append(retrieved_doc)
            sample["retrieved_docs"] = retrieved_docs

            if index is not None:
                expected_type = sample.get("type", "unknown")
                match = "OK" if predicted_intent == expected_type else "MISMATCH"
                logger.info(
                    f"  [{index}/{total}] Retrieved {len(filtered_retrieved)} documents, "
                    f"intent: {predicted_intent} vs expected: {expected_type} [{match}]"
                )

            query_to_use = search_result.get("query_to_use", question)
            generation_result = llm_service.answer_question(
                question=query_to_use,
                context=retrieved,
                relevant_foc=search_result.get("relevant_foc", None),
                temperature=0.3,
                is_eval=True,
            )
            generated = generation_result.get("answer", "")
            sample["generated_answer"] = generated

            if index is not None:
                logger.info(f"  [{index}/{total}] Generated answer: {generated[:50]}...")

        except Exception as e:
            logger.error(f"  Failed to process sample: {e}", exc_info=True)
            sample["retrieved_docs"] = []
            sample["generated_answer"] = ""
            raise


if __name__ == "__main__":
    print("*****************Rag-fintech Benchmark*****************")
    evaluator = RAGEvaluator()
    eval_file = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    evaluator.generate_answer(eval_file)
    result = evaluator(eval_file)
