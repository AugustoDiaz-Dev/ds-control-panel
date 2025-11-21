"""
Database repository for user operations
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime, timezone
from app.database.models import User as DBUser
from app.auth.models import User, UserInDB, UserCreate, UserUpdate
from app.auth.security import get_password_hash
from app.monitoring import get_logger

logger = get_logger(__name__)


class UserRepository:
    """Repository for user database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        try:
            db_user = self.db.query(DBUser).filter(DBUser.id == user_id).first()
            if db_user:
                return self._db_to_pydantic(db_user)
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {str(e)}", exc_info=True)
            return None
    
    def get_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        try:
            db_user = self.db.query(DBUser).filter(DBUser.username == username).first()
            if db_user:
                return self._db_to_pydantic(db_user)
            return None
        except Exception as e:
            logger.error(f"Error getting user by username {username}: {str(e)}", exc_info=True)
            return None
    
    def get_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email"""
        try:
            db_user = self.db.query(DBUser).filter(DBUser.email == email).first()
            if db_user:
                return self._db_to_pydantic(db_user)
            return None
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {str(e)}", exc_info=True)
            return None
    
    def create(self, user_create: UserCreate) -> UserInDB:
        """Create a new user"""
        try:
            # Check if user already exists
            if self.get_by_username(user_create.username):
                raise ValueError(f"User {user_create.username} already exists")
            
            if self.get_by_email(user_create.email):
                raise ValueError(f"Email {user_create.email} already exists")
            
            user_id = user_create.username  # Use username as ID
            now = datetime.now(timezone.utc)
            
            db_user = DBUser(
                id=user_id,
                username=user_create.username,
                email=user_create.email,
                full_name=user_create.full_name,
                hashed_password=get_password_hash(user_create.password),
                is_active=user_create.is_active if hasattr(user_create, 'is_active') else True,
                is_admin=user_create.is_admin if hasattr(user_create, 'is_admin') else False,
                created_at=now,
                updated_at=now
            )
            
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)
            
            logger.info(f"Created user: {user_create.username}")
            return self._db_to_pydantic(db_user)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating user: {str(e)}", exc_info=True)
            raise
    
    def update(self, user_id: str, user_update: UserUpdate) -> Optional[UserInDB]:
        """Update user"""
        try:
            db_user = self.db.query(DBUser).filter(DBUser.id == user_id).first()
            if not db_user:
                return None
            
            # Update fields
            if user_update.email is not None:
                # Check if email is already taken by another user
                existing = self.get_by_email(user_update.email)
                if existing and existing.id != user_id:
                    raise ValueError(f"Email {user_update.email} already exists")
                db_user.email = user_update.email
            
            if user_update.full_name is not None:
                db_user.full_name = user_update.full_name
            
            if user_update.password is not None:
                db_user.hashed_password = get_password_hash(user_update.password)
            
            if hasattr(user_update, 'is_active') and user_update.is_active is not None:
                db_user.is_active = user_update.is_active
            
            if hasattr(user_update, 'is_admin') and user_update.is_admin is not None:
                db_user.is_admin = user_update.is_admin
            
            db_user.updated_at = datetime.now(timezone.utc)
            
            self.db.commit()
            self.db.refresh(db_user)
            
            logger.info(f"Updated user: {user_id}")
            return self._db_to_pydantic(db_user)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user {user_id}: {str(e)}", exc_info=True)
            raise
    
    def delete(self, user_id: str) -> bool:
        """Delete user"""
        try:
            db_user = self.db.query(DBUser).filter(DBUser.id == user_id).first()
            if not db_user:
                return False
            
            self.db.delete(db_user)
            self.db.commit()
            
            logger.info(f"Deleted user: {user_id}")
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting user {user_id}: {str(e)}", exc_info=True)
            return False
    
    def list_all(self) -> List[User]:
        """List all users"""
        try:
            db_users = self.db.query(DBUser).all()
            return [User(**self._db_to_pydantic(db_user).model_dump(exclude={"hashed_password"})) for db_user in db_users]
        except Exception as e:
            logger.error(f"Error listing users: {str(e)}", exc_info=True)
            return []
    
    def _db_to_pydantic(self, db_user: DBUser) -> UserInDB:
        """Convert database user to Pydantic model"""
        return UserInDB(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
            full_name=db_user.full_name,
            hashed_password=db_user.hashed_password,
            is_active=db_user.is_active,
            is_admin=db_user.is_admin,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at
        )

