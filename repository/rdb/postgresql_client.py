from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from typing import List, Optional, Type, TypeVar

from repository.rdb.models.models import Base
from common import config
from common.decorator import singleton
import logging

T = TypeVar("T", bound=Base)

logger = logging.getLogger(__name__)


@singleton
class PostgreSQLClient:
    def __init__(self):
        rdb_config = config.RDB
        self.url = f"postgresql+psycopg://{rdb_config.get('username')}:{rdb_config.get('password')}@{rdb_config.get('host')}:{rdb_config.get('port')}/{rdb_config.get('database')}"
        self.engine = create_engine(
            self.url,
            echo=True,
            isolation_level="READ COMMITTED",
            pool_size=int(rdb_config.get("pool_size", 5)),
            max_overflow=int(rdb_config.get("max_overflow", 10)),
            connect_args={"options": "-c search_path=rag_fintech"},
        )
        logger.info(
            f"PostgreSQL client initialized successfully at @{rdb_config.get('host')}/{rdb_config.get('database')}"
        )

        # expire_on_commit=False prevents DetachedInstanceError
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def select(self, model: Type[T]) -> List[T]:
        """Select all records of a model"""
        with self.Session() as session:
            records = session.query(model).all()
            return records

    def select_by_id(self, model: Type[T], id: int) -> Optional[T]:
        """Select a record by ID"""
        with self.Session() as session:
            record = session.get(model, id)
            return record

    def execute_query(self, model: Type[T], name: str) -> List[T]:
        """Execute a query to get IDs by kb_name"""
        query = select(model).where(model.kb_name == name)
        with self.Session() as session:
            result = session.execute(query)
            return result.scalars().all()

    def save(self, model: Base):
        """
        Save or update a model instance.

        Note: Returns a detached instance. Access attributes before session closes
        or use refresh to reload data.
        """
        with self.Session.begin() as session:
            result = session.merge(model)
            session.flush()  # Ensure ID is generated
            # Explicitly load all attributes before session closes
            session.refresh(result)
            # Make the instance detached but with loaded attributes
            session.expunge(result)
            return result

    def save_all(self, models: List[Base]):
        """Save multiple models in a transaction"""
        with self.Session.begin() as session:
            session.add_all(models)

    def delete(self, model: Base):
        """Delete a model instance"""
        with self.Session.begin() as session:
            session.delete(model)

    def update(self, model: Type[T], id: int, **kwargs) -> Optional[T]:
        """Update a model instance by ID"""
        with self.Session.begin() as session:
            record = session.get(model, id)
            if record:
                for key, value in kwargs.items():
                    setattr(record, key, value)
                session.flush()
                session.refresh(record)
                session.expunge(record)
            return record
