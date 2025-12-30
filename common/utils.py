import json
import numpy as np
from typing import List, Union


def cosine_similarity(a: Union[List[float], np.ndarray, str], b: Union[List[float], np.ndarray, str]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector (list, numpy array, or JSON string)
        b: Second vector (list, numpy array, or JSON string)

    Returns:
        Cosine similarity value between -1 and 1. Returns 0.0 if either vector is zero.

    Raises:
        ValueError: If vectors have different lengths or invalid format
    """
    # Parse JSON strings if needed
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON string for vector a: {e}")

    if isinstance(b, str):
        try:
            b = json.loads(b)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON string for vector b: {e}")

    # Convert to numpy arrays for efficient computation
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Check vector lengths
    if len(a) != len(b):
        raise ValueError(f"Vectors must have the same length. Got {len(a)} and {len(b)}")

    # Calculate norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors (avoid division by zero)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    similarity = dot_product / (norm_a * norm_b)

    # Clamp to [-1, 1] to handle floating point errors
    similarity = np.clip(similarity, -1.0, 1.0)

    # Handle NaN or Inf (shouldn't happen, but safety check)
    if not np.isfinite(similarity):
        return 0.0

    return float(similarity)
