from sqlalchemy.orm import DeclarativeBase

from typing import Dict, Any
from sqlalchemy.inspection import inspect as sa_inspect


class Base(DeclarativeBase):
    """Base class for all models with common utility methods"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        return {c.key: getattr(self, c.key) for c in sa_inspect(self).mapper.column_attrs}

    def update_from_dict(self, data: Dict[str, Any]):
        """Update model instance from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
