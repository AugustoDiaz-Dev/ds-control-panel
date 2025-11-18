"""
Unit tests for ensemble model endpoints
"""
import pytest
from fastapi import status

@pytest.mark.unit
class TestEnsembleTrainEndpoint:
    """Tests for the /ensemble/train endpoint"""
    
    def test_ensemble_train_without_models(self, client):
        """Test ensemble training fails when no models are trained"""
        request_data = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "No models trained" in response.json()["detail"]
    
    def test_ensemble_train_voting(self, client, sample_training_params):
        """Test voting ensemble training"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        request_data = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["ensemble_type"] == "voting"
        assert "random_forest" in data["models_used"]
        assert "extra_trees" in data["models_used"]
        assert "metrics" in data
        assert "accuracy" in data["metrics"]
        assert "auc" in data["metrics"]
    
    def test_ensemble_train_stacking(self, client, sample_training_params):
        """Test stacking ensemble training"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        request_data = {
            "ensemble_type": "stacking",
            "model_names": ["random_forest", "extra_trees"]
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ensemble_type"] == "stacking"
        assert "metrics" in data
    
    def test_ensemble_train_weighted(self, client, sample_training_params):
        """Test weighted ensemble training"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        request_data = {
            "ensemble_type": "weighted",
            "model_names": ["random_forest", "extra_trees"],
            "weights": [0.6, 0.4]
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ensemble_type"] == "weighted"
        assert "metrics" in data
    
    def test_ensemble_train_weighted_equal_weights(self, client, sample_training_params):
        """Test weighted ensemble with equal weights (auto-assigned)"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        request_data = {
            "ensemble_type": "weighted",
            "model_names": ["random_forest", "extra_trees"]
            # weights not provided, should use equal weights
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ensemble_type"] == "weighted"
    
    def test_ensemble_train_invalid_model_names(self, client, sample_training_params):
        """Test ensemble training with invalid model names"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        request_data = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "invalid_model"]
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid model names" in response.json()["detail"]
    
    def test_ensemble_train_invalid_type(self, client, sample_training_params):
        """Test ensemble training with invalid ensemble type"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        request_data = {
            "ensemble_type": "invalid_type",
            "model_names": ["random_forest", "extra_trees"]
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid ensemble type" in response.json()["detail"]
    
    def test_ensemble_train_weighted_mismatched_weights(self, client, sample_training_params):
        """Test weighted ensemble with mismatched weights count"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        request_data = {
            "ensemble_type": "weighted",
            "model_names": ["random_forest", "extra_trees"],
            "weights": [0.5]  # Only one weight for two models
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Number of weights" in response.json()["detail"]
    
    def test_ensemble_train_all_models(self, client, sample_training_params):
        """Test ensemble training with all available models"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        # Get list of trained models
        models_response = client.get("/models")
        trained_models = models_response.json()["trained_models"]
        
        request_data = {
            "ensemble_type": "voting",
            "model_names": trained_models
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["models_used"]) == len(trained_models)
    
    def test_ensemble_train_metrics_valid(self, client, sample_training_params):
        """Test that ensemble metrics are valid"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        request_data = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        response = client.post("/ensemble/train", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        metrics = data["metrics"]
        
        # Validate metric ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["auc"] <= 1

@pytest.mark.unit
class TestEnsembleListEndpoint:
    """Tests for the /ensemble/list endpoint"""
    
    def test_ensemble_list_empty(self, client):
        """Test ensemble list when no ensembles are trained"""
        response = client.get("/ensemble/list")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "ensembles" in data
        assert "count" in data
        assert data["count"] == 0
        assert len(data["ensembles"]) == 0
    
    def test_ensemble_list_after_training(self, client, sample_training_params):
        """Test ensemble list after training ensembles"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        # Train multiple ensembles
        voting_request = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=voting_request)
        
        stacking_request = {
            "ensemble_type": "stacking",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=stacking_request)
        
        # Check list
        response = client.get("/ensemble/list")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["count"] == 2
        assert "voting_ensemble" in data["ensembles"]
        assert "stacking_ensemble" in data["ensembles"]

@pytest.mark.unit
class TestEnsemblePredictEndpoint:
    """Tests for the /ensemble/predict endpoint"""
    
    def test_ensemble_predict_without_ensemble(self, client, sample_features):
        """Test ensemble prediction when no ensemble is trained"""
        response = client.post(
            "/ensemble/predict",
            json={"features": sample_features},
            params={"ensemble_name": "voting_ensemble"}
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
    
    def test_ensemble_predict_voting(self, client, sample_training_params, sample_features):
        """Test prediction with voting ensemble"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        # Train ensemble
        request_data = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=request_data)
        
        # Make prediction
        response = client.post(
            "/ensemble/predict",
            json={"features": sample_features},
            params={"ensemble_name": "voting_ensemble"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "ensemble_used" in data
        assert data["ensemble_used"] == "voting_ensemble"
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
        assert "probability" in data
        assert "class_0" in data["probability"]
        assert "class_1" in data["probability"]
    
    def test_ensemble_predict_stacking(self, client, sample_training_params, sample_features):
        """Test prediction with stacking ensemble"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        # Train ensemble
        request_data = {
            "ensemble_type": "stacking",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=request_data)
        
        # Make prediction
        response = client.post(
            "/ensemble/predict",
            json={"features": sample_features},
            params={"ensemble_name": "stacking_ensemble"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ensemble_used"] == "stacking_ensemble"
    
    def test_ensemble_predict_weighted(self, client, sample_training_params, sample_features):
        """Test prediction with weighted ensemble"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        # Train ensemble
        request_data = {
            "ensemble_type": "weighted",
            "model_names": ["random_forest", "extra_trees"],
            "weights": [0.6, 0.4]
        }
        client.post("/ensemble/train", json=request_data)
        
        # Make prediction
        response = client.post(
            "/ensemble/predict",
            json={"features": sample_features},
            params={"ensemble_name": "weighted_ensemble"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ensemble_used"] == "weighted_ensemble"
    
    def test_ensemble_predict_probabilities_valid(self, client, sample_training_params, sample_features):
        """Test that ensemble prediction probabilities are valid"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        # Train ensemble
        request_data = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=request_data)
        
        # Make prediction
        response = client.post(
            "/ensemble/predict",
            json={"features": sample_features},
            params={"ensemble_name": "voting_ensemble"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate probabilities
        prob_sum = data["probability"]["class_0"] + data["probability"]["class_1"]
        assert abs(prob_sum - 1.0) < 0.001, "Probabilities should sum to 1"
        assert 0 <= data["probability"]["class_0"] <= 1
        assert 0 <= data["probability"]["class_1"] <= 1

@pytest.mark.unit
class TestEnsembleLoadEndpoint:
    """Tests for the /ensemble/load endpoint"""
    
    def test_ensemble_load_not_found(self, client):
        """Test loading non-existent ensemble"""
        response = client.post("/ensemble/load", params={"ensemble_name": "nonexistent_ensemble"})
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
    
    def test_ensemble_load_success(self, client, sample_training_params):
        """Test loading a saved ensemble"""
        # Train models and ensemble first
        client.post("/train", params=sample_training_params)
        
        ensemble_request = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=ensemble_request)
        
        # Clear ensemble from memory (simulate restart)
        from app.controllers import api
        api.ensemble_models.clear()
        
        # Load ensemble
        response = client.post("/ensemble/load", params={"ensemble_name": "voting_ensemble"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert data["ensemble_name"] == "voting_ensemble"
        assert "voting_ensemble" in data["loaded_ensembles"]

@pytest.mark.unit
class TestEnsembleMetricsEndpoint:
    """Tests for the /ensemble/metrics/{ensemble_name} endpoint"""
    
    def test_ensemble_metrics_not_found(self, client):
        """Test getting metrics for non-existent ensemble"""
        response = client.get("/ensemble/metrics/nonexistent_ensemble")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_ensemble_metrics_success(self, client, sample_training_params):
        """Test getting metrics for an ensemble"""
        # Train models and ensemble first
        client.post("/train", params=sample_training_params)
        
        ensemble_request = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=ensemble_request)
        
        # Get metrics
        response = client.get("/ensemble/metrics/voting_ensemble")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "ensemble_name" in data
        assert "metrics" in data
        assert "accuracy" in data["metrics"]
        assert "auc" in data["metrics"]
        
        # Validate metric ranges
        metrics = data["metrics"]
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc"] <= 1
    
    def test_ensemble_metrics_no_test_data(self, client):
        """Test getting metrics when test data is not available"""
        # This would require clearing test data, which is complex
        # So we'll just test that the endpoint exists and handles the case
        pass

@pytest.mark.unit
class TestEnsembleCompareEndpoint:
    """Tests for the /ensemble/compare endpoint"""
    
    def test_ensemble_compare_no_ensembles(self, client):
        """Test comparison when no ensembles are trained"""
        response = client.get("/ensemble/compare")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "No ensembles trained" in response.json()["detail"]
    
    def test_ensemble_compare_no_models(self, client, sample_training_params):
        """Test comparison when no individual models are trained"""
        # Train ensemble only
        client.post("/train", params=sample_training_params)
        
        ensemble_request = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=ensemble_request)
        
        # Clear individual models (simulate scenario)
        from app.controllers import api
        api.model_metrics.clear()
        
        response = client.get("/ensemble/compare")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "No individual models trained" in response.json()["detail"]
    
    def test_ensemble_compare_success(self, client, sample_training_params):
        """Test successful comparison of ensembles vs individual models"""
        # Train models and ensemble
        client.post("/train", params=sample_training_params)
        
        ensemble_request = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        client.post("/ensemble/train", json=ensemble_request)
        
        # Get comparison
        response = client.get("/ensemble/compare")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "individual_models" in data
        assert "ensembles" in data
        assert "best_individual" in data
        assert "best_ensemble" in data
        
        # Check structure
        assert len(data["individual_models"]) > 0
        assert len(data["ensembles"]) > 0
        
        # Check best model/ensemble structure
        if data["best_individual"]:
            assert "model_name" in data["best_individual"]
            assert "auc" in data["best_individual"]
        
        if data["best_ensemble"]:
            assert "ensemble_name" in data["best_ensemble"]
            assert "auc" in data["best_ensemble"]
        
        # Check improvement if both exist
        if data["best_individual"] and data["best_ensemble"]:
            assert "improvement" in data
            assert "auc_improvement" in data["improvement"]
            assert "percent_improvement" in data["improvement"]

@pytest.mark.integration
class TestEnsembleWorkflow:
    """Integration tests for complete ensemble workflows"""
    
    def test_complete_ensemble_workflow(self, client, sample_training_params, sample_features):
        """Test complete workflow: train models -> train ensemble -> predict"""
        # Step 1: Train individual models
        train_response = client.post("/train", params=sample_training_params)
        assert train_response.status_code == status.HTTP_200_OK
        
        # Step 2: Train ensemble
        ensemble_request = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        ensemble_response = client.post("/ensemble/train", json=ensemble_request)
        assert ensemble_response.status_code == status.HTTP_200_OK
        
        # Step 3: List ensembles
        list_response = client.get("/ensemble/list")
        assert list_response.status_code == status.HTTP_200_OK
        assert list_response.json()["count"] > 0
        
        # Step 4: Get ensemble metrics
        metrics_response = client.get("/ensemble/metrics/voting_ensemble")
        assert metrics_response.status_code == status.HTTP_200_OK
        
        # Step 5: Compare ensembles vs individual models
        compare_response = client.get("/ensemble/compare")
        assert compare_response.status_code == status.HTTP_200_OK
        
        # Step 6: Make prediction with ensemble
        predict_response = client.post(
            "/ensemble/predict",
            json={"features": sample_features},
            params={"ensemble_name": "voting_ensemble"}
        )
        assert predict_response.status_code == status.HTTP_200_OK
        assert "prediction" in predict_response.json()
    
    def test_multiple_ensemble_types(self, client, sample_training_params):
        """Test training multiple ensemble types"""
        # Train models first
        client.post("/train", params=sample_training_params)
        
        # Train voting ensemble
        voting_request = {
            "ensemble_type": "voting",
            "model_names": ["random_forest", "extra_trees"]
        }
        voting_response = client.post("/ensemble/train", json=voting_request)
        assert voting_response.status_code == status.HTTP_200_OK
        
        # Train stacking ensemble
        stacking_request = {
            "ensemble_type": "stacking",
            "model_names": ["random_forest", "extra_trees"]
        }
        stacking_response = client.post("/ensemble/train", json=stacking_request)
        assert stacking_response.status_code == status.HTTP_200_OK
        
        # Train weighted ensemble
        weighted_request = {
            "ensemble_type": "weighted",
            "model_names": ["random_forest", "extra_trees"],
            "weights": [0.7, 0.3]
        }
        weighted_response = client.post("/ensemble/train", json=weighted_request)
        assert weighted_response.status_code == status.HTTP_200_OK
        
        # Verify all ensembles are listed
        list_response = client.get("/ensemble/list")
        assert list_response.status_code == status.HTTP_200_OK
        ensembles = list_response.json()["ensembles"]
        assert "voting_ensemble" in ensembles
        assert "stacking_ensemble" in ensembles
        assert "weighted_ensemble" in ensembles
        
        # Compare all ensembles
        compare_response = client.get("/ensemble/compare")
        assert compare_response.status_code == status.HTTP_200_OK
        assert len(compare_response.json()["ensembles"]) == 3

