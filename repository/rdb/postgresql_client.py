from sqlalchemy import create_engine, select, and_
from sqlalchemy.orm import sessionmaker
from typing import List, Optional, Type, TypeVar, Any

from repository.rdb.models.models import Base
from sqlalchemy.orm import Session

from common.decorator import singleton
from common.config_utils import get_base_config
from common import get_logger

T = TypeVar("T", bound=Base)

logger = get_logger(__name__)


@singleton
class PostgreSQLClient:
    def __init__(self, config: dict = None):
        # Support both DI and standalone usage
        rdb_config = config or get_base_config("postgresql", {})
        self.url = f"postgresql+psycopg://{rdb_config.get('username')}:{rdb_config.get('password')}@{rdb_config.get('host')}:{rdb_config.get('port')}/{rdb_config.get('database')}"
        self.engine = create_engine(
            self.url,
            echo=False,
            isolation_level="READ COMMITTED",
            pool_size=int(rdb_config.get("pool_size", 5)),
            max_overflow=int(rdb_config.get("max_overflow", 10)),
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={
                "options": "-c search_path=rag_fintech",
                "connect_timeout": 10,
            },
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

    def select_by_id(self, model: Type[T], id: Any) -> Optional[T]:
        """Select a record by id"""
        with self.Session() as session:
            record = session.get(model, id)
            return record

    def select_by_kwargs(self, model: Type[T], **kwargs):
        """Select a record by keyword arguments"""
        if not kwargs:
            raise ValueError("At least one keyword argument is required")

        # Build WHERE conditions from kwargs
        conditions = [getattr(model, key) == value for key, value in kwargs.items()]
        statement = select(model).where(and_(*conditions))

        with self.Session() as session:
            result = session.execute(statement).scalars().first()
            return result

    def select_all_by_kwargs(self, model: Type[T], **kwargs) -> List[T]:
        """Select all records matching keyword arguments"""
        if not kwargs:
            raise ValueError("At least one keyword argument is required")

        conditions = [getattr(model, key) == value for key, value in kwargs.items()]
        statement = select(model).where(and_(*conditions))

        with self.Session() as session:
            return session.execute(statement).scalars().all()

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
            session.flush()
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
                session.refresh(record)
                session.expunge(record)
            return record

    def begin_transaction(self):
        """Create and return a session for manual transaction management.
        Caller is responsible for committing, rolling back, and closing the session.
        """
        return Session(self.engine, autobegin=False)

    def update_many_by_kwargs(self, session: Session, model: Type[T], filter_kwargs: dict, update_kwargs: dict) -> int:
        """Update all records matching filter_kwargs with update_kwargs"""
        count = session.query(model).filter_by(**filter_kwargs).update(update_kwargs)
        return count

    def save_with_session(self, session: Session, model: Base):
        result = session.merge(model)
        session.flush()
        session.refresh(result)
        session.expunge(result)
        return result


def _create_postgresql_client() -> PostgreSQLClient:
    return PostgreSQLClient()


rdb_client = _create_postgresql_client()
