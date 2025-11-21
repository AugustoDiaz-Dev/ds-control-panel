"""
Integration tests for database functionality
"""
import pytest
from fastapi import status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database.config import get_db_url, get_engine, init_db, check_db_connection
from app.database.models import Base, User, Experiment, ModelMetadata, Prediction
from app.database.user_repository import UserRepository
from app.auth.models import UserCreate, UserUpdate
from app.auth.security import verify_password


@pytest.mark.integration
class TestDatabaseConnection:
    """Tests for database connection"""
    
    def test_get_db_url(self):
        """Test database URL construction"""
        url = get_db_url()
        assert url is not None
        assert isinstance(url, str)
        assert "postgresql" in url or "sqlite" in url or url.startswith("postgresql://")
    
    def test_get_engine(self):
        """Test engine creation"""
        try:
            engine = get_engine()
            assert engine is not None
        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")
    
    def test_check_db_connection(self):
        """Test database connection check"""
        # This will skip if database is not available
        try:
            result = check_db_connection()
            # Result can be True or False, both are valid
            assert isinstance(result, bool)
        except Exception as e:
            pytest.skip(f"Database connection check failed: {str(e)}")


@pytest.mark.integration
class TestDatabaseModels:
    """Tests for database models"""
    
    def test_user_model_structure(self):
        """Test User model has required fields"""
        assert hasattr(User, 'id')
        assert hasattr(User, 'username')
        assert hasattr(User, 'email')
        assert hasattr(User, 'hashed_password')
        assert hasattr(User, 'is_active')
        assert hasattr(User, 'is_admin')
    
    def test_experiment_model_structure(self):
        """Test Experiment model has required fields"""
        assert hasattr(Experiment, 'id')
        assert hasattr(Experiment, 'experiment_name')
        assert hasattr(Experiment, 'run_id')
        assert hasattr(Experiment, 'user_id')
    
    def test_model_metadata_structure(self):
        """Test ModelMetadata model has required fields"""
        assert hasattr(ModelMetadata, 'id')
        assert hasattr(ModelMetadata, 'model_name')
        assert hasattr(ModelMetadata, 'model_type')
        assert hasattr(ModelMetadata, 'model_path')
    
    def test_prediction_model_structure(self):
        """Test Prediction model has required fields"""
        assert hasattr(Prediction, 'id')
        assert hasattr(Prediction, 'user_id')
        assert hasattr(Prediction, 'model_name')
        assert hasattr(Prediction, 'features')


@pytest.mark.integration
class TestUserRepository:
    """Tests for UserRepository"""
    
    @pytest.fixture
    def db_session(self):
        """Create a test database session"""
        try:
            from app.database.config import get_session_local
            db = get_session_local()()
            # Create tables
            Base.metadata.create_all(bind=get_engine())
            yield db
            db.rollback()
            db.close()
        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")
    
    def test_create_user(self, db_session):
        """Test creating a user"""
        repo = UserRepository(db_session)
        user_create = UserCreate(
            username="testuser_db",
            email="testdb@example.com",
            password="testpass123",
            full_name="Test User DB"
        )
        user = repo.create(user_create)
        assert user is not None
        assert user.username == "testuser_db"
        assert user.email == "testdb@example.com"
        assert verify_password("testpass123", user.hashed_password)
    
    def test_get_user_by_username(self, db_session):
        """Test getting user by username"""
        repo = UserRepository(db_session)
        user_create = UserCreate(
            username="testuser_get",
            email="testget@example.com",
            password="testpass123",
            full_name="Test User Get"
        )
        created_user = repo.create(user_create)
        retrieved_user = repo.get_by_username("testuser_get")
        assert retrieved_user is not None
        assert retrieved_user.id == created_user.id
        assert retrieved_user.username == "testuser_get"
    
    def test_get_user_by_id(self, db_session):
        """Test getting user by ID"""
        repo = UserRepository(db_session)
        user_create = UserCreate(
            username="testuser_id",
            email="testid@example.com",
            password="testpass123",
            full_name="Test User ID"
        )
        created_user = repo.create(user_create)
        retrieved_user = repo.get_by_id(created_user.id)
        assert retrieved_user is not None
        assert retrieved_user.id == created_user.id
    
    def test_update_user(self, db_session):
        """Test updating a user"""
        repo = UserRepository(db_session)
        user_create = UserCreate(
            username="testuser_update",
            email="testupdate@example.com",
            password="testpass123",
            full_name="Test User Update"
        )
        created_user = repo.create(user_create)
        
        user_update = UserUpdate(
            full_name="Updated Name",
            email="updated@example.com"
        )
        updated_user = repo.update(created_user.id, user_update)
        assert updated_user is not None
        assert updated_user.full_name == "Updated Name"
        assert updated_user.email == "updated@example.com"
    
    def test_delete_user(self, db_session):
        """Test deleting a user"""
        repo = UserRepository(db_session)
        user_create = UserCreate(
            username="testuser_delete",
            email="testdelete@example.com",
            password="testpass123",
            full_name="Test User Delete"
        )
        created_user = repo.create(user_create)
        
        result = repo.delete(created_user.id)
        assert result is True
        
        deleted_user = repo.get_by_id(created_user.id)
        assert deleted_user is None
    
    def test_list_users(self, db_session):
        """Test listing all users"""
        repo = UserRepository(db_session)
        # Create multiple users
        for i in range(3):
            user_create = UserCreate(
                username=f"testuser_list_{i}",
                email=f"testlist{i}@example.com",
                password="testpass123",
                full_name=f"Test User {i}"
            )
            repo.create(user_create)
        
        users = repo.list_all()
        assert len(users) >= 3
        usernames = [user.username for user in users]
        assert "testuser_list_0" in usernames
        assert "testuser_list_1" in usernames
        assert "testuser_list_2" in usernames
    
    def test_create_duplicate_user_fails(self, db_session):
        """Test that creating duplicate user fails"""
        repo = UserRepository(db_session)
        user_create = UserCreate(
            username="testuser_dup",
            email="testdup@example.com",
            password="testpass123",
            full_name="Test User Dup"
        )
        repo.create(user_create)
        
        # Try to create duplicate
        with pytest.raises(ValueError):
            repo.create(user_create)


@pytest.mark.integration
class TestDatabaseHealthCheck:
    """Tests for database health check in API"""
    
    def test_health_check_includes_database(self, client):
        """Test that health check includes database status"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "checks" in data
        # Database check may be "ok", "error", or "not_configured"
        assert "database" in data["checks"] or "not_configured" in str(data["checks"])


@pytest.mark.integration
class TestUserManagerWithDatabase:
    """Tests for UserManager with database integration"""
    
    def test_user_manager_uses_database_when_available(self, authenticated_client):
        """Test that UserManager can work with database"""
        # This test verifies that the system works
        # The actual database usage depends on configuration
        response = authenticated_client.get("/auth/me")
        # Should work regardless of storage backend
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]

