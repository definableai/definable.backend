from datetime import datetime
import uuid
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Enum as SQLEnum, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.database import Base
from enum import Enum

class TransactionType(str, Enum):
    CREDIT_PURCHASE = "credit_purchase"
    CREDIT_USAGE = "credit_usage"
    REFUND = "refund"
    DISCOUNT = "discount"

class TransactionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class BillableService(Base):
    __tablename__ = "billable_services"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    functions = relationship("BillableFunction", back_populates="service")

class BillableFunction(Base):
    __tablename__ = "billable_functions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_id = Column(UUID(as_uuid=True), ForeignKey("billable_services.id"))
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)  # Full path to function
    version = Column(String, nullable=False, default="1.0.0")
    pricing_type = Column(String, nullable=False)  # 'per_token', 'fixed', 'custom'
    base_price_credits = Column(Integer, nullable=False)  # Base price in credits
    pricing_config = Column(JSON, nullable=True)  # Additional pricing configuration
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    service = relationship("BillableService", back_populates="functions")
    versions = relationship("FunctionVersion", back_populates="function")

class FunctionVersion(Base):
    __tablename__ = "function_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    function_id = Column(UUID(as_uuid=True), ForeignKey("billable_functions.id"))
    version = Column(String, nullable=False)
    path = Column(String, nullable=False)
    base_price_credits = Column(Integer, nullable=False)
    pricing_config = Column(JSON, nullable=True)
    valid_from = Column(DateTime, nullable=False)
    valid_to = Column(DateTime, nullable=True)
    
    function = relationship("BillableFunction", back_populates="versions")

class BillingTransaction(Base):
    __tablename__ = "billing_transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    parent_transaction_id = Column(UUID(as_uuid=True), ForeignKey("billing_transactions.id"), nullable=True)
    function_id = Column(UUID(as_uuid=True), ForeignKey("billable_functions.id"))
    function_version = Column(String, nullable=False)
    type = Column(SQLEnum(TransactionType), nullable=False)
    status = Column(SQLEnum(TransactionStatus), nullable=False)
    credits_used = Column(Integer, nullable=False)
    metadata = Column(JSON, nullable=True)  # Store additional transaction data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    children = relationship("BillingTransaction", 
                          backref=relationship("parent", remote_side=[id]))

class CreditBalance(Base):
    __tablename__ = "credit_balances"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True)
    balance = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    type = Column(SQLEnum(TransactionType), nullable=False)
    status = Column(SQLEnum(TransactionStatus), nullable=False)
    amount_usd = Column(Float, nullable=False)
    credits = Column(Integer, nullable=False)
    stripe_payment_intent_id = Column(String, nullable=True)
    stripe_customer_id = Column(String, nullable=True)  # Store Stripe customer ID only
    description = Column(String, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LLMModelPricing(Base):
    __tablename__ = "llm_model_pricing"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, unique=True, nullable=False)
    credits_per_token = Column(Integer, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
