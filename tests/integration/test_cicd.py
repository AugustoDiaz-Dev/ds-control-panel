"""
CI/CD specific integration tests
These tests are designed to run quickly and verify critical functionality
"""
import pytest
from fastapi import status

@pytest.mark.integration
@pytest.mark.cicd
class TestCICDCriticalPaths:
    """Critical path tests for CI/CD pipeline"""
    
    def test_public_endpoints_accessible(self, client):
        """Test that public endpoints are accessible without auth"""
        # Root endpoint
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert "endpoints" in response.json()
        
        # Login endpoint
        response = client.post(
            "/auth/login",
            data={"username": "admin", "password": "admin123"}
        )
        # Should work (default admin user exists)
        assert response.status_code == status.HTTP_200_OK
    
    def test_authentication_flow(self, client):
        """Test basic authentication flow"""
        # Login with default admin
        login_response = client.post(
            "/auth/login",
            data={"username": "admin", "password": "admin123"}
        )
        assert login_response.status_code == status.HTTP_200_OK
        token = login_response.json()["access_token"]
        
        # Use token to access protected endpoint
        response = client.get(
            "/models",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == status.HTTP_200_OK
    
    def test_protected_endpoints_require_auth(self, client):
        """Test that protected endpoints require authentication"""
        endpoints = [
            ("GET", "/models"),
            ("GET", "/metrics"),
            ("POST", "/train"),
            ("POST", "/predict"),
        ]
        
        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json={} if endpoint == "/predict" else {})
            
            assert response.status_code == status.HTTP_403_FORBIDDEN, \
                f"{method} {endpoint} should require authentication"
    
    def test_basic_ml_workflow(self, authenticated_client, sample_training_params, sample_features):
        """Test basic ML workflow works with authentication"""
        # Train
        train_response = authenticated_client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        
        # Get metrics
        metrics_response = authenticated_client.get("/metrics")
        assert metrics_response.status_code == status.HTTP_200_OK
        
        # Predict
        predict_response = authenticated_client.post("/predict", json={"features": sample_features})
        assert predict_response.status_code == status.HTTP_200_OK
        assert "prediction" in predict_response.json()

@pytest.mark.integration
@pytest.mark.cicd
class TestCICDDataValidation:
    """Data validation tests for CI/CD"""
    
    def test_invalid_login_credentials(self, client):
        """Test invalid login credentials are rejected"""
        response = client.post(
            "/auth/login",
            data={"username": "nonexistent", "password": "wrongpass"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_duplicate_user_registration(self, client):
        """Test that duplicate user registration fails"""
        # Register first time
        response1 = client.post(
            "/auth/register",
            json={
                "username": "duplicate",
                "password": "pass123"
            }
        )
        assert response1.status_code == status.HTTP_200_OK
        
        # Try to register again
        response2 = client.post(
            "/auth/register",
            json={
                "username": "duplicate",
                "password": "pass123"
            }
        )
        assert response2.status_code == status.HTTP_400_BAD_REQUEST
        
        # Cleanup
        from app.auth.user_manager import user_manager
        try:
            user_manager.delete_user("duplicate")
        except Exception:
            pass
    
    def test_invalid_token_format(self, client):
        """Test that invalid token formats are rejected"""
        # Missing Bearer prefix
        response = client.get(
            "/models",
            headers={"Authorization": "invalid_token"}
        )
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]
        
        # Empty token
        response = client.get(
            "/models",
            headers={"Authorization": "Bearer "}
        )
        assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN]

@pytest.mark.integration
@pytest.mark.cicd
class TestCICDErrorHandling:
    """Error handling tests for CI/CD"""
    
    def test_missing_required_fields(self, client):
        """Test that missing required fields are handled"""
        # Register without username
        response = client.post(
            "/auth/register",
            json={"password": "pass123"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Register without password
        response = client.post(
            "/auth/register",
            json={"username": "testuser"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_json_payload(self, client):
        """Test that invalid JSON payloads are handled"""
        # Invalid JSON in register
        response = client.post(
            "/auth/register",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

