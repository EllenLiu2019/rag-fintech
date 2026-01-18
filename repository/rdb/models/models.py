from typing import Optional, List
from datetime import datetime

from sqlalchemy import BigInteger, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

from repository.rdb.models.base import Base


class LLM(Base):
    """LLM configuration table"""

    __tablename__ = "llm"
    __table_args__ = {"schema": "rag_fintech"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(60), nullable=False)
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    model_type: Mapped[str] = mapped_column(String(60), nullable=False)
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    max_tokens_context: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class KnowledgeBase(Base):
    """Knowledge base configuration table"""

    __tablename__ = "knowledgebase"
    __table_args__ = {"schema": "rag_fintech"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kb_name: Mapped[str] = mapped_column(String(60), nullable=False, unique=True, index=True)
    embed_llm_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("rag_fintech.llm.id"), nullable=False)
    doc_num: Mapped[int] = mapped_column(Integer, default=0)
    chunk_num: Mapped[int] = mapped_column(Integer, default=0)
    token_num: Mapped[int] = mapped_column(Integer, default=0)
    update_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    documents: Mapped[List["Document"]] = relationship("Document", back_populates="knowledgebase")


class Document(Base):
    """Document metadata table"""

    __tablename__ = "document"
    __table_args__ = {"schema": "rag_fintech"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[Optional[str]] = mapped_column(String(60), nullable=False, unique=True, index=True)
    doc_type: Mapped[str] = mapped_column(String(60), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_status: Mapped[str] = mapped_column(String(60), nullable=False, default="uploaded")
    file_location: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    doc_location: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_num: Mapped[int] = mapped_column(Integer, default=0)
    token_num: Mapped[int] = mapped_column(Integer, default=0)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    upload_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    kb_name: Mapped[str] = mapped_column(String(60), ForeignKey("rag_fintech.knowledgebase.kb_name"), nullable=False)
    business_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    clause_forest: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    knowledgebase: Mapped["KnowledgeBase"] = relationship(
        "KnowledgeBase", back_populates="documents", foreign_keys=[kb_name]
    )
