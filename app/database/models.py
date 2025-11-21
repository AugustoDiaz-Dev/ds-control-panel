"""
SQLAlchemy database models
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def utcnow():
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class Experiment(Base):
    """MLflow experiment tracking"""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String, index=True, nullable=False)
    run_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    # Experiment metadata
    status = Column(String, default="RUNNING")
    start_time = Column(DateTime, default=utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    
    # Hyperparameters (stored as JSON)
    hyperparameters = Column(JSON, nullable=True)
    
    # Metrics (stored as JSON)
    metrics = Column(JSON, nullable=True)
    
    # Tags (stored as JSON)
    tags = Column(JSON, nullable=True)
    
    # Model info
    model_name = Column(String, nullable=True)
    model_type = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="experiments")


class ModelMetadata(Base):
    """Model metadata and versioning"""
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True, nullable=False)
    model_type = Column(String, nullable=False)  # e.g., 'random_forest', 'xgboost'
    version = Column(String, nullable=False, default="1.0.0")
    
    # Model file path
    model_path = Column(String, nullable=False)
    
    # Training metadata
    training_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    training_duration = Column(Float, nullable=True)  # seconds
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc = Column(Float, nullable=True)
    
    # Hyperparameters
    hyperparameters = Column(JSON, nullable=True)
    
    # Experiment reference
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    run_id = Column(String, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_production = Column(Boolean, default=False)
    
    # Additional metadata
    additional_metadata = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)


class Prediction(Base):
    """Prediction history and logging"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    model_name = Column(String, index=True, nullable=False)
    
    # Input features (stored as JSON array)
    features = Column(JSON, nullable=False)
    
    # Prediction results
    prediction = Column(Integer, nullable=False)
    probability_class_0 = Column(Float, nullable=True)
    probability_class_1 = Column(Float, nullable=True)
    
    # Metadata
    prediction_time = Column(Float, nullable=True)  # milliseconds
    model_version = Column(String, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="predictions")

