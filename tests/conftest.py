"""
Provide shared fixtures and configurations for tests.
"""

import sys
from unittest.mock import MagicMock
from common.log_utils import init_root_logger

init_root_logger()

# Mock external dependencies that may not be available in test environment
# This prevents import errors when running tests
if "pymilvus" not in sys.modules:
    sys.modules["pymilvus"] = MagicMock()

if "voyageai" not in sys.modules:
    sys.modules["voyageai"] = MagicMock()

if "redis" not in sys.modules:
    sys.modules["redis"] = MagicMock()
