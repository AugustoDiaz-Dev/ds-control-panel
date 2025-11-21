"""
Database configuration and connection management
"""
import os
from typing import Generator
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from app.monitoring import get_logger

logger = get_logger(__name__)

# Database configuration from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "dscontrol")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dscontrol123")
DB_NAME = os.getenv("DB_NAME", "ds_control_panel")

# Alternative: Use DATABASE_URL if provided
DATABASE_URL = os.getenv("DATABASE_URL")

# Connection pool settings
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))


def get_db_url() -> str:
    """
    Get database URL from environment or construct from components
    
    Returns:
        Database connection URL
    """
    if DATABASE_URL:
        return DATABASE_URL
    
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine() -> Engine:
    """
    Create and return SQLAlchemy engine with connection pooling
    
    Returns:
        SQLAlchemy engine instance
    """
    db_url = get_db_url()
    
    engine = create_engine(
        db_url,
        poolclass=QueuePool,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT,
        pool_recycle=POOL_RECYCLE,
        pool_pre_ping=True,  # Verify connections before using
        echo=False,  # Set to True for SQL query logging
    )
    
    return engine


# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_session_local() -> sessionmaker:
    """
    Get session factory for creating database sessions
    
    Returns:
        Session factory
    """
    return SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database: create all tables
    """
    from app.database.models import Base
    
    engine = get_engine()
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        raise


def check_db_connection() -> bool:
    """
    Check if database connection is available
    
    Returns:
        True if connection is available, False otherwise
    """
    try:
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False

