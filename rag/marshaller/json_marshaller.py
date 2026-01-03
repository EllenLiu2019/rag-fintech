from typing import Any, Dict, List, Optional, Type, TypeVar, Protocol, runtime_checkable, Callable
from datetime import datetime, date, time
from enum import Enum
from dataclasses import is_dataclass, asdict
from uuid import UUID
import json
import inspect

from pydantic import BaseModel

from common import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@runtime_checkable
class Serializable(Protocol):
    def to_dict(self) -> Dict[str, Any]: ...
    def serialize(self) -> Dict[str, Any]: ...


@runtime_checkable
class Deserializable(Protocol):
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any: ...
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Any: ...


class JSONMarshaller:
    """
    support serialization of multiple object types:
    - Pydantic BaseModel
    - Dataclass
    - custom object (with to_dict method)
    - basic types (dict, list, str, int, float, bool)
    - special types (datetime, UUID, Enum)
    - nested structures
    """

    # custom serializer registry
    _custom_serializers: Dict[Type, Callable[[Any], Dict[str, Any]]] = {}
    _custom_deserializers: Dict[Type, Callable[[Dict[str, Any]], Any]] = {}

    @classmethod
    def register_serializer(cls, obj_type: Type, serializer: Callable[[Any], Dict[str, Any]]):
        """
        register custom serializer
        """
        cls._custom_serializers[obj_type] = serializer
        logger.debug(f"Registered custom serializer for {obj_type.__name__}")

    @classmethod
    def register_deserializer(cls, obj_type: Type, deserializer: Callable[[Dict[str, Any]], Any]):
        """
        register custom deserializer
        """
        cls._custom_deserializers[obj_type] = deserializer
        logger.debug(f"Registered custom deserializer for {obj_type.__name__}")

    @classmethod
    def serialize(cls, obj: Any, exclude_none: bool = False) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None

        obj_type = type(obj)
        if obj_type in cls._custom_serializers:
            return cls._custom_serializers[obj_type](obj)

        # Check for custom serialization methods first (before BaseModel default)
        # This allows BaseModel subclasses to override serialization behavior
        # Note: We check hasattr first because isinstance(Serializable) may fail
        # if the method signature doesn't exactly match (e.g., has extra parameters)
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict", None)):
            try:
                # Try calling with exclude_none parameter if method accepts it
                sig = inspect.signature(obj.to_dict)
                if "exclude_none" in sig.parameters:
                    return obj.to_dict(exclude_none=exclude_none)
                else:
                    return obj.to_dict()
            except Exception as e:
                logger.warning(f"Failed to call to_dict() on {obj_type.__name__}: {e}")
        elif hasattr(obj, "serialize") and callable(getattr(obj, "serialize", None)):
            try:
                return obj.serialize()
            except Exception as e:
                logger.warning(f"Failed to call serialize() on {obj_type.__name__}: {e}")

        # Use BaseModel default serialization if no custom method
        if isinstance(obj, BaseModel):
            dumped = obj.model_dump(exclude_none=exclude_none)
            # Recursively serialize the dict to handle datetime objects
            return cls.serialize(dumped, exclude_none=exclude_none)

        if is_dataclass(obj):
            return asdict(obj)

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, dict):
            return {k: cls.serialize(v, exclude_none) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [cls.serialize(item, exclude_none) for item in obj]

        if isinstance(obj, datetime):
            return obj.isoformat()

        if isinstance(obj, date):
            return obj.isoformat()

        if isinstance(obj, time):
            return obj.isoformat()

        if isinstance(obj, UUID):
            return str(obj)

        if isinstance(obj, Enum):
            return obj.value

        if hasattr(obj, "__dict__"):
            try:
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith("_"):  # exclude private attributes
                        result[key] = cls.serialize(value, exclude_none)
                return result
            except Exception as e:
                logger.warning(f"Failed to serialize using __dict__: {e}")

        raise TypeError(
            f"Cannot serialize object of type {obj_type.__name__}. "
            f"Object must be one of: Pydantic BaseModel, dataclass, "
            f"object with to_dict()/serialize() method, or basic type."
        )

    @classmethod
    def deserialize(cls, data: Optional[Dict[str, Any]], target_type: Optional[Type[T]] = None) -> Optional[T]:
        if data is None:
            return None

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        # if target type is specified, use the corresponding deserializer
        if target_type:
            if target_type in cls._custom_deserializers:
                return cls._custom_deserializers[target_type](data)

            # Check for custom deserialization methods first (before BaseModel default)
            # This allows BaseModel subclasses to override deserialization behavior
            if isinstance(target_type, type) and issubclass(target_type, Deserializable):
                if hasattr(target_type, "from_dict"):
                    return target_type.from_dict(data)
                elif hasattr(target_type, "deserialize"):
                    return target_type.deserialize(data)

            # Use BaseModel default deserialization if no custom method
            if issubclass(target_type, BaseModel):
                try:
                    return target_type(**data)
                except Exception as e:
                    logger.error(f"Failed to deserialize as {target_type.__name__}: {e}")
                    raise ValueError(f"Invalid data for {target_type.__name__}: {e}") from e

            # Dataclass
            if is_dataclass(target_type):
                try:
                    return target_type(**data)
                except Exception as e:
                    logger.error(f"Failed to deserialize as {target_type.__name__}: {e}")
                    raise ValueError(f"Invalid data for {target_type.__name__}: {e}") from e

        # if no target type specified and no custom deserializer, return original dict
        return data

    @classmethod
    def to_json(
        cls,
        obj: Any,
        indent: Optional[int] = None,
        ensure_ascii: bool = False,
        exclude_none: bool = False,
    ) -> str:
        """
        serialize object to JSON string
        """
        data = cls.serialize(obj, exclude_none=exclude_none)
        return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=str)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        target_type: Optional[Type[T]] = None,
    ) -> Optional[T]:
        """
        deserialize JSON string to object
        """
        try:
            data = json.loads(json_str)
            return cls.deserialize(data, target_type)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Invalid JSON string: {e}") from e


def serialize(obj: Any, exclude_none: bool = False) -> Optional[Dict[str, Any]]:
    return JSONMarshaller.serialize(obj, exclude_none=exclude_none)


def deserialize(data: Optional[Dict[str, Any]], target_type: Optional[Type[T]] = None) -> Optional[T]:
    return JSONMarshaller.deserialize(data, target_type)


def to_json(
    obj: Any,
    indent: Optional[int] = None,
    ensure_ascii: bool = False,
    exclude_none: bool = False,
) -> str:
    return JSONMarshaller.to_json(
        obj,
        indent=indent,
        ensure_ascii=ensure_ascii,
        exclude_none=exclude_none,
    )


def from_json(json_str: str, target_type: Optional[Type[T]] = None) -> Optional[T]:
    return JSONMarshaller.from_json(
        json_str,
        target_type,
    )


def serialize_batch(objs: List[Any], exclude_none: bool = False) -> List[Dict[str, Any]]:
    """
    Serialize a list of objects to a list of dictionaries.
    None values in the input list are filtered out.

    Args:
        objs: List of objects to serialize
        exclude_none: Whether to exclude None values in serialization

    Returns:
        List of serialized dictionaries (None values filtered out)
    """
    if not objs:
        return []

    # Filter out None values from serialization results
    results = []
    for obj in objs:
        if obj is None:
            continue  # Skip None values in batch
        serialized = serialize(obj, exclude_none)
        if serialized is not None:
            results.append(serialized)
    return results


def deserialize_batch(
    data_list: List[Dict[str, Any]],
    target_type: Optional[Type[T]] = None,
) -> List[T]:
    """
    Deserialize a list of dictionaries to a list of objects.
    None values in the input list are filtered out.

    Args:
        data_list: List of dictionaries to deserialize
        target_type: Target type for deserialization

    Returns:
        List of deserialized objects (None values filtered out)
    """
    if not data_list:
        return []

    # Filter out None values from deserialization results
    results = []
    for data in data_list:
        if data is None:
            continue  # Skip None values in batch
        deserialized = deserialize(data, target_type)
        if deserialized is not None:
            results.append(deserialized)
    return results
