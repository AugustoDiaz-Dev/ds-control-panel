"""
User management system
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from app.auth.models import User, UserInDB, UserCreate, UserUpdate
from app.auth.security import get_password_hash, verify_password

# Storage file path
BASE_DIR = Path(__file__).parent.parent.parent
USERS_FILE = BASE_DIR / "users.json"


class UserManager:
    """Manages user storage and operations"""
    
    def __init__(self, users_file: Path = USERS_FILE):
        self.users_file = users_file
        self._users: Dict[str, UserInDB] = {}
        self._load_users()
        self._ensure_default_user()
    
    def _load_users(self):
        """Load users from file"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_id, user_data in data.items():
                        # Convert datetime strings back to datetime objects
                        user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                        user_data['updated_at'] = datetime.fromisoformat(user_data['updated_at'])
                        self._users[user_id] = UserInDB(**user_data)
            except Exception as e:
                print(f"Error loading users: {e}")
                self._users = {}
        else:
            self._users = {}
    
    def _save_users(self):
        """Save users to file"""
        try:
            # Convert to JSON-serializable format
            data = {}
            for user_id, user in self._users.items():
                user_dict = user.model_dump()
                # Convert datetime to ISO format string
                user_dict['created_at'] = user.created_at.isoformat()
                user_dict['updated_at'] = user.updated_at.isoformat()
                data[user_id] = user_dict
            
            # Ensure directory exists
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def _ensure_default_user(self):
        """Create a default admin user if no users exist"""
        if not self._users:
            default_user = UserCreate(
                username="admin",
                email="admin@example.com",
                full_name="Administrator",
                password="admin123",  # Should be changed in production
                is_active=True,
                is_admin=True
            )
            self.create_user(default_user)
            print("Created default admin user: admin/admin123")
    
    def create_user(self, user_create: UserCreate) -> UserInDB:
        """Create a new user"""
        if self.get_user_by_username(user_create.username):
            raise ValueError(f"User {user_create.username} already exists")
        
        user_id = user_create.username  # Use username as ID for simplicity
        now = datetime.utcnow()
        
        user_in_db = UserInDB(
            id=user_id,
            username=user_create.username,
            email=user_create.email,
            full_name=user_create.full_name,
            hashed_password=get_password_hash(user_create.password),
            is_active=user_create.is_active,
            is_admin=user_create.is_admin,
            created_at=now,
            updated_at=now
        )
        
        self._users[user_id] = user_in_db
        self._save_users()
        return user_in_db
    
    def get_user(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username"""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        return user
    
    def update_user(self, user_id: str, user_update: UserUpdate) -> Optional[UserInDB]:
        """Update a user"""
        user = self.get_user(user_id)
        if not user:
            return None
        
        update_data = user_update.model_dump(exclude_unset=True)
        
        if "password" in update_data:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        
        for key, value in update_data.items():
            setattr(user, key, value)
        
        user.updated_at = datetime.utcnow()
        self._save_users()
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        if user_id in self._users:
            del self._users[user_id]
            self._save_users()
            return True
        return False
    
    def list_users(self) -> List[User]:
        """List all users (without passwords)"""
        return [User(**user.model_dump(exclude={"hashed_password"})) for user in self._users.values()]


# Global user manager instance
user_manager = UserManager()

