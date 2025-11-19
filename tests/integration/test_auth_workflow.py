"""
Integration tests for authentication workflows
"""
import pytest
from fastapi import status

@pytest.mark.integration
class TestAuthenticationWorkflow:
    """Test complete authentication workflows"""
    
    def test_register_login_workflow(self, client):
        """Test complete workflow: register -> login -> access protected endpoint"""
        # Step 1: Register a new user
        register_response = client.post(
            "/auth/register",
            json={
                "username": "workflowuser",
                "email": "workflow@example.com",
                "password": "workflowpass123",
                "full_name": "Workflow User"
            }
        )
        assert register_response.status_code == status.HTTP_200_OK
        user_data = register_response.json()
        assert user_data["username"] == "workflowuser"
        assert "hashed_password" not in user_data
        
        # Step 2: Login to get token
        login_response = client.post(
            "/auth/login",
            data={"username": "workflowuser", "password": "workflowpass123"}
        )
        assert login_response.status_code == status.HTTP_200_OK
        token_data = login_response.json()
        assert "access_token" in token_data
        token = token_data["access_token"]
        
        # Step 3: Access protected endpoint with token
        models_response = client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert models_response.status_code == status.HTTP_200_OK
        
        # Step 4: Get current user info
        me_response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert me_response.status_code == status.HTTP_200_OK
        me_data = me_response.json()
        assert me_data["username"] == "workflowuser"
        
        # Cleanup
        from app.auth.user_manager import user_manager
        try:
            user_manager.delete_user("workflowuser")
        except Exception:
            pass
    
    def test_full_ml_workflow_with_auth(self, authenticated_client, sample_training_params, sample_features):
        """Test complete ML workflow with authentication"""
        # Step 1: Train models (requires auth)
        train_response = authenticated_client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        
        # Step 2: Get metrics (requires auth)
        metrics_response = authenticated_client.get("/metrics")
        assert metrics_response.status_code == status.HTTP_200_OK
        
        # Step 3: Make prediction (requires auth)
        predict_response = authenticated_client.post("/predict", json={"features": sample_features})
        assert predict_response.status_code == status.HTTP_200_OK
        
        # Step 4: Get visualizations (requires auth)
        viz_response = authenticated_client.get("/visualizations/model-comparison")
        assert viz_response.status_code == status.HTTP_200_OK
    
    def test_admin_workflow(self, admin_client, client):
        """Test admin user workflow"""
        # Step 1: Admin can list users
        users_response = admin_client.get("/auth/users")
        assert users_response.status_code == status.HTTP_200_OK
        users_data = users_response.json()
        assert isinstance(users_data, list)
        
        # Step 2: Admin can access all endpoints
        models_response = admin_client.get("/models")
        assert models_response.status_code == status.HTTP_200_OK
        
        # Step 3: Create a regular user
        register_response = client.post(
            "/auth/register",
            json={
                "username": "regularuser",
                "password": "regularpass123"
            }
        )
        assert register_response.status_code == status.HTTP_200_OK
        
        # Step 4: Admin can update user
        update_response = admin_client.put(
            "/auth/users/regularuser",
            json={"full_name": "Updated Name"}
        )
        assert update_response.status_code == status.HTTP_200_OK
        
        # Step 5: Admin can delete user
        delete_response = admin_client.delete("/auth/users/regularuser")
        assert delete_response.status_code == status.HTTP_200_OK
        
        # Verify user is deleted
        verify_response = admin_client.get("/auth/users")
        users_after = verify_response.json()
        usernames = [u["username"] for u in users_after]
        assert "regularuser" not in usernames
    
    def test_token_expiration_handling(self, client, test_user):
        """Test handling of expired or invalid tokens"""
        # Step 1: Get valid token
        login_response = client.post(
            "/auth/login",
            data={"username": "testuser", "password": "testpass123"}
        )
        assert login_response.status_code == status.HTTP_200_OK
        token = login_response.json()["access_token"]
        
        # Step 2: Use valid token
        response = client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == status.HTTP_200_OK
        
        # Step 3: Try with invalid token
        invalid_response = client.get(
            "/models",
            headers={"Authorization": "Bearer invalid_token_here"}
        )
        assert invalid_response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # Step 4: Try without token
        no_token_response = client.get("/models")
        assert no_token_response.status_code == status.HTTP_403_FORBIDDEN

@pytest.mark.integration
class TestSecurityWorkflow:
    """Test security-related workflows"""
    
    def test_unauthorized_access_attempts(self, client):
        """Test that unauthorized access is properly blocked"""
        # Try to access protected endpoint without auth
        response = client.post("/train", params={"n_estimators": 10})
        assert response.status_code == status.HTTP_403_FORBIDDEN
        
        # Try with invalid token
        response = client.get(
            "/models",
            headers={"Authorization": "Bearer fake_token_12345"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # Try with malformed token
        response = client.get(
            "/models",
            headers={"Authorization": "InvalidFormat token"}
        )
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
    
    def test_regular_user_cannot_access_admin_endpoints(self, authenticated_client):
        """Test that regular users cannot access admin-only endpoints"""
        # Regular user tries to list users
        response = authenticated_client.get("/auth/users")
        assert response.status_code == status.HTTP_403_FORBIDDEN
        
        # Regular user tries to update user
        response = authenticated_client.put(
            "/auth/users/testuser",
            json={"full_name": "Hacked Name"}
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN
        
        # Regular user tries to delete user
        response = authenticated_client.delete("/auth/users/testuser")
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_inactive_user_cannot_access(self, client):
        """Test that inactive users cannot access endpoints"""
        from app.auth.user_manager import user_manager
        from app.auth.models import UserCreate
        
        # Create inactive user
        inactive_user = UserCreate(
            username="inactiveuser",
            password="inactivepass123",
            is_active=False
        )
        user_manager.create_user(inactive_user)
        
        # Try to login
        login_response = client.post(
            "/auth/login",
            data={"username": "inactiveuser", "password": "inactivepass123"}
        )
        # Should fail because user is inactive
        assert login_response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # Cleanup
        try:
            user_manager.delete_user("inactiveuser")
        except Exception:
            pass

