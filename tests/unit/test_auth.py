"""
Unit tests for authentication
"""
import pytest
from app.controllers.api import app
from app.auth.user_manager import user_manager
from app.auth.models import UserCreate
from app.auth.security import create_access_token, verify_token, get_password_hash, verify_password

# Use the client fixture from conftest instead of creating a global one
# This will be injected via pytest fixtures


class TestPasswordHashing:
    """Test password hashing functions"""
    
    def test_hash_password(self):
        """Test password hashing"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password(self):
        """Test password verification"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False


class TestJWTTokens:
    """Test JWT token functions"""
    
    def test_create_token(self):
        """Test token creation"""
        data = {"sub": "testuser"}
        token = create_access_token(data)
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token(self):
        """Test token verification"""
        data = {"sub": "testuser"}
        token = create_access_token(data)
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "testuser"
    
    def test_verify_invalid_token(self):
        """Test verification of invalid token"""
        invalid_token = "invalid.token.here"
        payload = verify_token(invalid_token)
        assert payload is None


class TestUserManagement:
    """Test user management"""
    
    def test_create_user(self):
        """Test user creation"""
        user_create = UserCreate(
            username="testuser",
            email="test@example.com",
            password="testpass123",
            full_name="Test User"
        )
        user = user_manager.create_user(user_create)
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.hashed_password != "testpass123"
        
        # Cleanup
        user_manager.delete_user(user.id)
    
    def test_authenticate_user(self):
        """Test user authentication"""
        user_create = UserCreate(
            username="authtest",
            password="testpass123"
        )
        user_manager.create_user(user_create)
        
        # Test correct credentials
        authenticated = user_manager.authenticate_user("authtest", "testpass123")
        assert authenticated is not None
        assert authenticated.username == "authtest"
        
        # Test wrong password
        authenticated = user_manager.authenticate_user("authtest", "wrongpass")
        assert authenticated is None
        
        # Test wrong username
        authenticated = user_manager.authenticate_user("wronguser", "testpass123")
        assert authenticated is None
        
        # Cleanup
        user_manager.delete_user("authtest")


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    def test_login_success(self, client):
        """Test successful login"""
        # Create a test user
        user_create = UserCreate(
            username="logintest",
            password="testpass123"
        )
        user_manager.create_user(user_create)
        
        # Login
        response = client.post(
            "/auth/login",
            data={"username": "logintest", "password": "testpass123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        
        # Cleanup
        user_manager.delete_user("logintest")
    
    def test_login_failure(self, client):
        """Test failed login"""
        response = client.post(
            "/auth/login",
            data={"username": "nonexistent", "password": "wrongpass"}
        )
        assert response.status_code == 401
    
    def test_register(self, client):
        """Test user registration"""
        response = client.post(
            "/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "newpass123",
                "full_name": "New User"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser"
        assert "hashed_password" not in data
        
        # Cleanup
        user_manager.delete_user("newuser")
    
    def test_register_duplicate(self, client):
        """Test duplicate user registration"""
        user_create = UserCreate(
            username="duplicate",
            password="testpass123"
        )
        user_manager.create_user(user_create)
        
        response = client.post(
            "/auth/register",
            json={
                "username": "duplicate",
                "password": "testpass123"
            }
        )
        assert response.status_code == 400
        
        # Cleanup
        user_manager.delete_user("duplicate")
    
    def test_get_current_user(self, client):
        """Test getting current user info"""
        # Create user and login
        user_create = UserCreate(
            username="currentuser",
            password="testpass123"
        )
        user_manager.create_user(user_create)
        
        # Login to get token
        login_response = client.post(
            "/auth/login",
            data={"username": "currentuser", "password": "testpass123"}
        )
        token = login_response.json()["access_token"]
        
        # Get current user
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "currentuser"
        
        # Cleanup
        user_manager.delete_user("currentuser")
    
    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without token"""
        response = client.get("/models")
        assert response.status_code == 403  # Forbidden - no token provided
    
    def test_protected_endpoint_with_token(self, client):
        """Test accessing protected endpoint with valid token"""
        # Create user and login
        user_create = UserCreate(
            username="protectedtest",
            password="testpass123"
        )
        user_manager.create_user(user_create)
        
        # Login to get token
        login_response = client.post(
            "/auth/login",
            data={"username": "protectedtest", "password": "testpass123"}
        )
        token = login_response.json()["access_token"]
        
        # Access protected endpoint
        response = client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        # Cleanup
        user_manager.delete_user("protectedtest")
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token"""
        response = client.get(
            "/models",
            headers={"Authorization": "Bearer invalid_token_here"}
        )
        assert response.status_code == 401  # Unauthorized


class TestAdminEndpoints:
    """Test admin-only endpoints"""
    
    def test_list_users_as_admin(self, client):
        """Test listing users as admin"""
        # Create admin user
        user_create = UserCreate(
            username="admintest",
            password="testpass123",
            is_admin=True
        )
        user_manager.create_user(user_create)
        
        # Login as admin
        login_response = client.post(
            "/auth/login",
            data={"username": "admintest", "password": "testpass123"}
        )
        token = login_response.json()["access_token"]
        
        # List users
        response = client.get(
            "/auth/users",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        
        # Cleanup
        user_manager.delete_user("admintest")
    
    def test_list_users_as_non_admin(self, client):
        """Test listing users as non-admin (should fail)"""
        # Create regular user
        user_create = UserCreate(
            username="regulartest",
            password="testpass123",
            is_admin=False
        )
        user_manager.create_user(user_create)
        
        # Login as regular user
        login_response = client.post(
            "/auth/login",
            data={"username": "regulartest", "password": "testpass123"}
        )
        token = login_response.json()["access_token"]
        
        # Try to list users (should fail)
        response = client.get(
            "/auth/users",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403  # Forbidden
        
        # Cleanup
        user_manager.delete_user("regulartest")

