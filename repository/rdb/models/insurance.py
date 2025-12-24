from typing import Optional
from datetime import date, datetime
from decimal import Decimal

from repository.rdb.models.base import Base
from sqlalchemy import BigInteger, String, DateTime, Float, Date, Numeric
from sqlalchemy.orm import Mapped, mapped_column


class Policy(Base):

    __tablename__ = "policy"
    __table_args__ = {"schema": "rag_fintech"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    policy_number: Mapped[str] = mapped_column(String(50), nullable=False)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)
    source_file: Mapped[str] = mapped_column(String(200), nullable=False)
    extraction_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(1), nullable=False, default="A")


class PolicyHolder(Base):
    __tablename__ = "policy_holder"
    __table_args__ = {"schema": "rag_fintech"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    id_number: Mapped[str] = mapped_column(String(100), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(1), nullable=False, default="A")

    policy_number: Mapped[str] = mapped_column(String(50), nullable=False)


class Insured(Base):
    __tablename__ = "insured"
    __table_args__ = {"schema": "rag_fintech"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    id_number: Mapped[str] = mapped_column(String(100), nullable=False)
    relationship_to_holder: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(1), nullable=False, default="A")

    policy_number: Mapped[str] = mapped_column(String(50), nullable=False)


class Coverage(Base):

    __tablename__ = "coverage"
    __table_args__ = {"schema": "rag_fintech"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cvg_name: Mapped[str] = mapped_column(String(200), nullable=False)
    cvg_type: Mapped[str] = mapped_column(String(200), nullable=False)
    cvg_amt: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(1), nullable=False, default="A")

    policy_number: Mapped[str] = mapped_column(String(50), nullable=False)


class CvgPremium(Base):

    __tablename__ = "cvg_premium"
    __table_args__ = {"schema": "rag_fintech"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cvg_name: Mapped[str] = mapped_column(String(200), nullable=False)
    cvg_premium: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(1), nullable=False, default="A")

    policy_number: Mapped[str] = mapped_column(String(50), nullable=False)
