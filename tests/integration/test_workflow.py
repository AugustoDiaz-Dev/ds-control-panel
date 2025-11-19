"""
Integration tests for complete workflows
"""
import pytest
from fastapi import status

@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete training and prediction workflow"""
    
    def test_train_predict_workflow(self, authenticated_client, sample_training_params, sample_features):
        """Test complete workflow: train -> predict -> metrics"""
        # Step 1: Train models
        train_response = authenticated_client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        train_data = train_response.json()
        assert train_data["best_model"] is not None
        
        # Step 2: Get metrics
        metrics_response = authenticated_client.get("/metrics")
        assert metrics_response.status_code == status.HTTP_200_OK
        metrics_data = metrics_response.json()
        assert len(metrics_data["metrics"]) > 0
        
        # Step 3: Make prediction
        predict_response = authenticated_client.post("/predict", json={"features": sample_features})
        assert predict_response.status_code == status.HTTP_200_OK
        predict_data = predict_response.json()
        assert "prediction" in predict_data
        
        # Step 4: Verify model used matches best model
        assert predict_data["model_used"] == train_data["best_model"]
    
    def test_train_optimize_workflow(self, authenticated_client, sample_training_params):
        """Test workflow: train -> optimize -> compare"""
        # Step 1: Train initial models
        train_response = authenticated_client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        
        # Step 2: Get initial metrics
        initial_metrics = authenticated_client.get("/metrics").json()
        initial_auc = initial_metrics["metrics"].get("random_forest", {}).get("auc", 0)
        
        # Step 3: Optimize
        optimize_response = authenticated_client.post(
            "/optimize",
            params={"model_name": "random_forest", "n_trials": 5}
        )
        
        if optimize_response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            pytest.skip("Optuna not available")
        
        assert optimize_response.status_code == status.HTTP_200_OK
        optimize_data = optimize_response.json()
        assert optimize_data["best_metrics"]["auc"] >= 0
    
    def test_visualization_workflow(self, authenticated_client, sample_training_params):
        """Test complete visualization workflow"""
        # Step 1: Train models
        authenticated_client.post("/train", params=sample_training_params)
        
        # Step 2: Get all visualizations
        viz_response = authenticated_client.get("/visualizations/all/random_forest")
        assert viz_response.status_code == status.HTTP_200_OK
        viz_data = viz_response.json()
        
        # Verify all visualization data is present
        assert "metrics" in viz_data
        assert "confusion_matrix" in viz_data
        assert "roc_curve" in viz_data
        assert "feature_importance" in viz_data
        
        # Step 3: Verify individual endpoints work
        cm_response = authenticated_client.get("/visualizations/confusion-matrix/random_forest")
        assert cm_response.status_code == status.HTTP_200_OK
        
        roc_response = authenticated_client.get("/visualizations/roc-curve/random_forest")
        assert roc_response.status_code == status.HTTP_200_OK
        
        fi_response = authenticated_client.get("/visualizations/feature-importance/random_forest")
        assert fi_response.status_code == status.HTTP_200_OK

@pytest.mark.integration
class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_models_are_saved(self, authenticated_client, sample_training_params):
        """Test that models are saved after training"""
        # Train models
        authenticated_client.post("/train", params=sample_training_params)
        
        # Check saved models
        models_response = authenticated_client.get("/models")
        assert models_response.status_code == status.HTTP_200_OK
        models_data = models_response.json()
        assert len(models_data["saved_models"]) > 0

@pytest.mark.integration
class TestMultipleTrainingRuns:
    """Test multiple training runs"""
    
    def test_multiple_training_runs(self, authenticated_client):
        """Test that multiple training runs work correctly"""
        params1 = {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 3}
        params2 = {"n_estimators": 15, "learning_rate": 0.15, "max_depth": 4}
        
        # First training run
        response1 = authenticated_client.post("/train", params=params1)
        assert response1.status_code == status.HTTP_200_OK
        
        # Second training run
        response2 = authenticated_client.post("/train", params=params2)
        assert response2.status_code == status.HTTP_200_OK
        
        # Both should have metrics
        metrics = authenticated_client.get("/metrics").json()
        assert len(metrics["metrics"]) > 0

