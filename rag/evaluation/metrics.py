"""
Custom metrics for RAG evaluation.
Includes retrieval metrics (MRR, NDCG, Hit Rate) and generation metrics.
"""

from typing import List, Callable, Optional
import numpy as np

from common import get_logger

logger = get_logger(__name__)


class RetrievalMetrics:
    """
    Retrieval evaluation metrics.
    Evaluates how well the retrieval system finds relevant documents.

    Supports two evaluation modes:
    1. ID-based: Exact matching using document IDs (default, faster)
    2. Content-based: Similarity matching using document content (requires similarity function)
    """

    @staticmethod
    def _is_match(
        retrieved_item: str,
        relevant_items: List[str],
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.8,
    ) -> bool:
        """
        Check if retrieved item matches any relevant item.

        Args:
            retrieved_item: Retrieved document ID or content
            relevant_items: List of relevant document IDs or contents
            similarity_fn: Optional function to compute similarity between two strings
            similarity_threshold: Threshold for content-based matching

        Returns:
            True if match found, False otherwise
        """
        if similarity_fn is None:
            # ID-based exact matching
            return retrieved_item in set(relevant_items)
        else:
            # Content-based similarity matching
            for relevant_item in relevant_items:
                similarity = similarity_fn(retrieved_item, relevant_item)
                if similarity >= similarity_threshold:
                    return True
            return False

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.8,
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR measures how high the first relevant document appears in the results.

        Args:
            retrieved_docs: List of retrieved document IDs or contents for each query
            relevant_docs: List of relevant document IDs or contents for each query
            similarity_fn: Optional function for content-based matching.
                          If None, uses exact ID matching.
                          Function signature: (str, str) -> float (0-1)
            similarity_threshold: Threshold for content-based matching (default: 0.8)

        Returns:
            MRR score (0-1, higher is better)
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            for rank, doc_item in enumerate(retrieved, start=1):
                if RetrievalMetrics._is_match(doc_item, relevant, similarity_fn, similarity_threshold):
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    @staticmethod
    def hit_rate_at_k(
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k: int = 5,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.8,
    ) -> float:
        """
        Calculate Hit Rate@K (Recall@K).

        Measures if at least one relevant document appears in top-K results.

        Args:
            retrieved_docs: List of retrieved document IDs or contents for each query
            relevant_docs: List of relevant document IDs or contents for each query
            k: Consider only top-k results
            similarity_fn: Optional function for content-based matching
            similarity_threshold: Threshold for content-based matching

        Returns:
            Hit rate (0-1, higher is better)
        """
        hits = 0

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_k = retrieved[:k]
            # Check if any retrieved doc matches any relevant doc
            found_match = False
            for doc_item in retrieved_k:
                if RetrievalMetrics._is_match(doc_item, relevant, similarity_fn, similarity_threshold):
                    found_match = True
                    break
            if found_match:
                hits += 1

        return hits / len(retrieved_docs) if retrieved_docs else 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k: int = 5,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.8,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).

        Measures ranking quality considering position of relevant documents.

        Args:
            retrieved_docs: List of retrieved document IDs or contents for each query
            relevant_docs: List of relevant document IDs or contents for each query
            k: Consider only top-k results
            similarity_fn: Optional function for content-based matching
            similarity_threshold: Threshold for content-based matching

        Returns:
            NDCG score (0-1, higher is better)
        """
        ndcg_scores = []

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            # DCG: sum of (relevance / log2(rank+1))
            dcg = 0.0
            for rank, doc_item in enumerate(retrieved[:k], start=1):
                if RetrievalMetrics._is_match(doc_item, relevant, similarity_fn, similarity_threshold):
                    dcg += 1.0 / np.log2(rank + 1)

            # IDCG: ideal DCG (all relevant docs at top)
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))

            # Normalize
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    @staticmethod
    def precision_at_k(
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k: int = 5,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.8,
    ) -> float:
        """
        Calculate Precision@K.

        Measures proportion of relevant documents in top-K results.

        Args:
            retrieved_docs: List of retrieved document IDs or contents for each query
            relevant_docs: List of relevant document IDs or contents for each query
            k: Consider only top-k results
            similarity_fn: Optional function for content-based matching
            similarity_threshold: Threshold for content-based matching

        Returns:
            Precision score (0-1, higher is better)
        """
        precisions = []

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_k = retrieved[:k]
            relevant_in_k = sum(
                1
                for doc in retrieved_k
                if RetrievalMetrics._is_match(doc, relevant, similarity_fn, similarity_threshold)
            )
            precision = relevant_in_k / k if k > 0 else 0.0
            precisions.append(precision)

        return float(np.mean(precisions)) if precisions else 0.0

    @staticmethod
    def recall_at_k(
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k: int = 5,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        similarity_threshold: float = 0.8,
    ) -> float:
        """
        Calculate Recall@K.

        Measures proportion of relevant documents found in top-K results.

        Args:
            retrieved_docs: List of retrieved document IDs or contents for each query
            relevant_docs: List of relevant document IDs or contents for each query
            k: Consider only top-k results
            similarity_fn: Optional function for content-based matching
            similarity_threshold: Threshold for content-based matching

        Returns:
            Recall score (0-1, higher is better)
        """
        recalls = []

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            if not relevant:
                continue

            retrieved_k = retrieved[:k]
            # Count how many relevant docs are found in retrieved_k
            relevant_found = sum(
                1
                for rel_doc in relevant
                if any(
                    RetrievalMetrics._is_match(ret_doc, [rel_doc], similarity_fn, similarity_threshold)
                    for ret_doc in retrieved_k
                )
            )
            recall = relevant_found / len(relevant)
            recalls.append(recall)

        return float(np.mean(recalls)) if recalls else 0.0


class GenerationMetrics:
    """
    Generation quality metrics.
    Evaluates the quality of generated answers.
    """

    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """
        Calculate Exact Match score.

        Args:
            predictions: List of predicted answers
            references: List of reference answers

        Returns:
            Exact match score (0-1, higher is better)
        """
        matches = sum(pred.strip().lower() == ref.strip().lower() for pred, ref in zip(predictions, references))
        return matches / len(predictions) if predictions else 0.0

    @staticmethod
    def token_overlap(predictions: List[str], references: List[str]) -> float:
        """
        Calculate token-level overlap (simple F1).

        Args:
            predictions: List of predicted answers
            references: List of reference answers

        Returns:
            Average F1 score (0-1, higher is better)
        """
        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())

            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue

            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref_tokens)

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        return float(np.mean(f1_scores)) if f1_scores else 0.0
