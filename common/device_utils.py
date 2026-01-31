import os
from typing import Literal, Optional
from common import get_logger

logger = get_logger(__name__)

DeviceType = Literal["cpu", "cuda", "mps", "auto"]


def detect_available_devices() -> dict[str, bool]:
    devices = {"cpu": True, "cuda": False, "mps": False}

    try:
        import torch

        devices["cuda"] = torch.cuda.is_available()
        if hasattr(torch.backends, "mps"):
            devices["mps"] = torch.backends.mps.is_available()
    except ImportError:
        logger.debug("PyTorch not available, only CPU will be used")

    return devices


def select_device(
    preferred: Optional[DeviceType] = None,
    operation: Optional[str] = None,
    fallback_to_cpu: bool = True,
) -> str:
    """
    Select appropriate device for an operation.

    Priority:
    1. Environment variable: RAG_DEVICE_{OPERATION} or RAG_DEVICE
    2. Preferred device from config
    3. Auto-detect (GPU > MPS > CPU)
    4. Fallback to CPU

    Args:
        preferred: Preferred device from config ("cpu", "cuda", "mps", "auto")
        operation: Operation name (e.g., "sparse_embedding", "reranker")
        fallback_to_cpu: If True, always fallback to CPU if preferred device unavailable

    Returns:
        Device string: "cpu", "cuda", or "mps"
    """
    # 1. Check environment variable (highest priority)
    env_key = f"RAG_DEVICE_{operation.upper()}" if operation else "RAG_DEVICE"
    env_device = os.getenv(env_key)
    if env_device:
        device = env_device.lower()
        if device in ["cpu", "cuda", "mps"]:
            logger.info(f"Using device from env {env_key}: {device}")
            return device
        elif device == "auto":
            # Continue to auto-detect
            pass
        else:
            logger.warning(f"Invalid device in {env_key}: {env_device}, using auto-detect")

    # 2. Use preferred device from config
    if preferred and preferred != "auto":
        if preferred in ["cpu", "cuda", "mps"]:
            devices = detect_available_devices()
            if devices.get(preferred, False) or (preferred == "cpu"):
                logger.info(f"Using preferred device: {preferred}")
                return preferred
            elif fallback_to_cpu:
                logger.warning(f"Preferred device {preferred} not available, falling back to CPU")
                return "cpu"
            else:
                raise RuntimeError(f"Preferred device {preferred} not available")

    # 3. Auto-detect (GPU > MPS > CPU)
    devices = detect_available_devices()

    if devices["cuda"]:
        logger.info("Auto-detected: CUDA GPU available")
        return "cuda"
    elif devices["mps"]:
        logger.info("Auto-detected: Apple Silicon GPU (MPS) available")
        return "mps"
    else:
        logger.info("Auto-detected: Using CPU (no GPU available)")
        return "cpu"


def get_device_for_operation(
    operation: str,
    config: Optional[dict] = None,
) -> str:
    """
    Get device for a specific operation with config support.

    Args:
        operation: Operation name (e.g., "sparse_embedding", "dense_embedding", "reranker")
        config: Optional config dict with device settings

    Returns:
        Device string
    """
    # Get device config for this operation
    preferred = None
    if config:
        # Check operation-specific config
        operation_config = config.get("devices", {}).get(operation)
        if operation_config:
            preferred = operation_config.get("device", "auto")
        # Check global device config
        elif "device" in config:
            preferred = config["device"]

    return select_device(preferred=preferred, operation=operation)
