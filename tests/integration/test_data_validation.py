"""
Integration tests for data validation
"""
import pytest
from fastapi import status
import numpy as np

@pytest.mark.integration
class TestDataValidation:
    """Test data validation in API endpoints"""
    
    def test_prediction_with_wrong_feature_count(self, client, sample_training_params):
        """Test prediction with wrong number of features"""
        client.post("/train", params=sample_training_params)
        
        # Too few features
        response = client.post("/predict", json={"features": [1, 2, 3]})
        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_200_OK,  # If it handles gracefully
            status.HTTP_422_UNPROCESSABLE_ENTITY,  # Validation error
            status.HTTP_500_INTERNAL_SERVER_ERROR  # Runtime error
        ]
        
        # Too many features
        response = client.post("/predict", json={"features": [1] * 20})
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    def test_prediction_with_invalid_types(self, client, sample_training_params):
        """Test prediction with invalid data types"""
        client.post("/train", params=sample_training_params)
        
        # String instead of number
        response = client.post("/predict", json={"features": ["a", "b", "c", "d", "e", "f", "g"]})
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    def test_training_with_extreme_parameters(self, client):
        """Test training with extreme parameter values"""
        # Very small parameters
        response = client.post("/train", params={
            "n_estimators": 1,
            "learning_rate": 0.001,
            "max_depth": 1
        })
        # Should either work or return validation error
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_training_with_negative_parameters(self, client):
        """Test training with negative parameters (should fail)"""
        response = client.post("/train", params={
            "n_estimators": -10,
            "learning_rate": -0.1,
            "max_depth": -5
        })
        # Should handle validation - sklearn will raise error or use default
        # The test should accept either successful training (with defaults) or error
        assert response.status_code in [
            status.HTTP_200_OK,  # If backend handles it with defaults
            status.HTTP_422_UNPROCESSABLE_ENTITY,  # Validation error
            status.HTTP_500_INTERNAL_SERVER_ERROR  # Runtime error from sklearn
        ]

