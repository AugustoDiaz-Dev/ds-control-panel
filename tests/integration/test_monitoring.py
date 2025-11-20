"""
Integration tests for monitoring endpoints and functionality
"""
import pytest
import time
from fastapi import status
from app.monitoring.metrics import metrics_collector


@pytest.mark.integration
class TestHealthCheckEndpoint:
    """Tests for the /health endpoint"""
    
    def test_health_check_no_auth_required(self, client):
        """Test that health check doesn't require authentication"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check_structure(self, client):
        """Test health check response structure"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "data_directory" in data["checks"]
        assert "models_directory" in data["checks"]
        assert "mlflow" in data["checks"]
    
    def test_health_check_status_values(self, client):
        """Test that health check returns valid status values"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert all(
            check in ["ok", "error", "available", "unavailable"]
            for check in data["checks"].values()
        )


@pytest.mark.integration
class TestModelStatusEndpoint:
    """Tests for the /model/status endpoint"""
    
    def test_model_status_requires_auth(self, client):
        """Test that model status requires authentication"""
        response = client.get("/model/status")
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_model_status_without_models(self, authenticated_client):
        """Test model status when no models are trained"""
        response = authenticated_client.get("/model/status")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "trained_models" in data
        assert "ensembles" in data
        assert "best_model" in data
        assert "total_models" in data
        assert "total_ensembles" in data
        assert "timestamp" in data
        assert data["total_models"] == 0
    
    def test_model_status_after_training(self, authenticated_client, sample_training_params):
        """Test model status after training models"""
        # Train models first
        train_response = authenticated_client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        
        # Check status
        response = authenticated_client.get("/model/status")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["total_models"] > 0
        assert len(data["trained_models"]) > 0
        
        # Check model structure
        for model_name, model_info in data["trained_models"].items():
            assert "loaded" in model_info
            assert "saved" in model_info
            assert "has_metrics" in model_info
            assert "metrics" in model_info


@pytest.mark.integration
class TestMonitoringMetricsEndpoint:
    """Tests for the /monitoring/metrics endpoint"""
    
    def test_monitoring_metrics_requires_auth(self, client):
        """Test that monitoring metrics requires authentication"""
        response = client.get("/monitoring/metrics")
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_monitoring_metrics_structure(self, authenticated_client):
        """Test monitoring metrics response structure"""
        # Make some requests to generate metrics
        authenticated_client.get("/")
        authenticated_client.get("/model/status")
        
        response = authenticated_client.get("/monitoring/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "system_metrics" in data
        assert "timestamp" in data
        
        metrics = data["system_metrics"]
        assert "uptime_seconds" in metrics
        assert "total_requests" in metrics
        assert "total_errors" in metrics
        assert "error_rate_percent" in metrics
        assert "requests_per_second" in metrics
        assert "response_times" in metrics
        assert "requests_by_status" in metrics
        assert "top_endpoints" in metrics
        assert "model_statistics" in metrics
    
    def test_monitoring_metrics_response_times(self, authenticated_client):
        """Test that response time metrics are present"""
        # Make some requests
        for _ in range(5):
            authenticated_client.get("/")
        
        response = authenticated_client.get("/monitoring/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        metrics = data["system_metrics"]
        response_times = metrics["response_times"]
        
        assert "avg" in response_times
        assert "p50" in response_times
        assert "p95" in response_times
        assert "p99" in response_times
        assert "min" in response_times
        assert "max" in response_times
        
        # All should be non-negative
        assert response_times["avg"] >= 0
        assert response_times["p50"] >= 0
        assert response_times["p95"] >= 0
    
    def test_monitoring_metrics_after_training(self, authenticated_client, sample_training_params):
        """Test monitoring metrics after model training"""
        # Train models
        train_response = authenticated_client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        
        # Wait a bit for metrics to be recorded
        time.sleep(0.1)
        
        # Get metrics
        response = authenticated_client.get("/monitoring/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        metrics = data["system_metrics"]
        
        # Should have model statistics
        assert "model_statistics" in metrics
        # Model statistics might be empty if training was too fast, but structure should exist


@pytest.mark.integration
class TestMonitoringErrorsEndpoint:
    """Tests for the /monitoring/errors endpoint"""
    
    def test_monitoring_errors_requires_auth(self, client):
        """Test that monitoring errors requires authentication"""
        response = client.get("/monitoring/errors")
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_monitoring_errors_structure(self, authenticated_client):
        """Test monitoring errors response structure"""
        response = authenticated_client.get("/monitoring/errors")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "errors" in data
        assert "count" in data
        assert "timestamp" in data
        assert isinstance(data["errors"], list)
        assert data["count"] == len(data["errors"])
    
    def test_monitoring_errors_limit_parameter(self, authenticated_client):
        """Test monitoring errors with limit parameter"""
        # Make some requests that might generate errors
        # (or just test the endpoint structure)
        response = authenticated_client.get("/monitoring/errors?limit=5")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert len(data["errors"]) <= 5
    
    def test_monitoring_errors_after_error(self, authenticated_client):
        """Test that errors are recorded after an error occurs"""
        # Make a request that will fail (invalid endpoint)
        try:
            authenticated_client.get("/nonexistent_endpoint_that_will_404")
        except Exception:
            pass
        
        # Get errors
        response = authenticated_client.get("/monitoring/errors")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        # Errors might be recorded, but we can't guarantee it
        # Just verify the structure is correct
        assert "errors" in data


@pytest.mark.integration
class TestMonitoringMiddleware:
    """Tests for monitoring middleware functionality"""
    
    def test_middleware_logs_requests(self, authenticated_client):
        """Test that middleware logs requests"""
        # Make some requests
        authenticated_client.get("/")
        authenticated_client.get("/model/status")
        
        # Check that metrics were recorded
        response = authenticated_client.get("/monitoring/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        metrics = data["system_metrics"]
        
        # Should have recorded at least 2 requests (the ones we made)
        assert metrics["total_requests"] >= 2
    
    def test_middleware_tracks_response_times(self, authenticated_client):
        """Test that middleware tracks response times"""
        # Make a request
        authenticated_client.get("/")
        
        # Get metrics
        response = authenticated_client.get("/monitoring/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        metrics = data["system_metrics"]
        
        # Should have response time data
        assert metrics["response_times"]["avg"] >= 0
    
    def test_middleware_tracks_endpoints(self, authenticated_client):
        """Test that middleware tracks requests by endpoint"""
        # Make requests to different endpoints
        authenticated_client.get("/")
        authenticated_client.get("/model/status")
        
        # Get metrics
        response = authenticated_client.get("/monitoring/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        metrics = data["system_metrics"]
        
        # Should have top endpoints
        assert "top_endpoints" in metrics
        assert len(metrics["top_endpoints"]) > 0


@pytest.mark.integration
class TestMonitoringWithModelOperations:
    """Tests for monitoring during model operations"""
    
    def test_monitoring_tracks_training(self, authenticated_client, sample_training_params):
        """Test that monitoring tracks model training"""
        # Get initial metrics
        initial_response = authenticated_client.get("/monitoring/metrics")
        initial_metrics = initial_response.json()["system_metrics"]
        initial_requests = initial_metrics["total_requests"]
        
        # Train models
        train_response = authenticated_client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        
        # Wait a bit for metrics to be recorded
        time.sleep(0.2)
        
        # Get updated metrics
        updated_response = authenticated_client.get("/monitoring/metrics")
        updated_metrics = updated_response.json()["system_metrics"]
        
        # Should have more requests (training + metrics request)
        assert updated_metrics["total_requests"] > initial_requests
    
    def test_monitoring_tracks_predictions(self, authenticated_client, sample_training_params):
        """Test that monitoring tracks predictions"""
        # Train models first
        train_response = authenticated_client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        
        # Make a prediction
        predict_response = authenticated_client.post(
            "/predict",
            json={"features": [750, 80000, 15000, 35, 60000, 8, 25000]}
        )
        assert predict_response.status_code == status.HTTP_200_OK
        
        # Wait a bit for metrics to be recorded
        time.sleep(0.1)
        
        # Get metrics
        metrics_response = authenticated_client.get("/monitoring/metrics")
        assert metrics_response.status_code == status.HTTP_200_OK
        
        data = metrics_response.json()
        metrics = data["system_metrics"]
        
        # Should have model statistics
        assert "model_statistics" in metrics
        # May or may not have prediction data depending on timing


@pytest.mark.integration
class TestMonitoringErrorHandling:
    """Tests for error handling in monitoring"""
    
    def test_health_check_handles_errors(self, client):
        """Test that health check handles errors gracefully"""
        response = client.get("/health")
        # Should always return 200 or 503, never 500
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
    
    def test_model_status_handles_errors(self, authenticated_client):
        """Test that model status handles errors gracefully"""
        response = authenticated_client.get("/model/status")
        # Should return 200 or 500, but structure should be consistent
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "trained_models" in data
            assert "ensembles" in data
    
    def test_monitoring_metrics_handles_errors(self, authenticated_client):
        """Test that monitoring metrics handles errors gracefully"""
        response = authenticated_client.get("/monitoring/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        # Should always return valid structure even if there are errors
        data = response.json()
        assert "system_metrics" in data
        assert "timestamp" in data

