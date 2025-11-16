"""
Model validation and quality tests
"""
import pytest
from fastapi import status
import numpy as np

@pytest.mark.model
class TestModelMetrics:
    """Test model metric validation"""
    
    def test_metrics_are_valid(self, client, sample_training_params):
        """Test that all metrics are within valid ranges"""
        client.post("/train", params=sample_training_params)
        
        response = client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        for model_name, metrics in data["metrics"].items():
            # All metrics should be between 0 and 1
            assert 0 <= metrics["accuracy"] <= 1, f"{model_name} accuracy out of range"
            assert 0 <= metrics["precision"] <= 1, f"{model_name} precision out of range"
            assert 0 <= metrics["recall"] <= 1, f"{model_name} recall out of range"
            assert 0 <= metrics["f1_score"] <= 1, f"{model_name} f1_score out of range"
            assert 0 <= metrics["auc"] <= 1, f"{model_name} AUC out of range"
            
            # Metrics should be reasonable (not all zeros or ones)
            assert metrics["accuracy"] > 0, f"{model_name} accuracy is zero"
            assert metrics["auc"] > 0, f"{model_name} AUC is zero"
    
    def test_best_model_has_highest_auc(self, client, sample_training_params):
        """Test that best model has the highest AUC"""
        client.post("/train", params=sample_training_params)
        
        response = client.get("/metrics")
        data = response.json()
        
        if len(data["metrics"]) > 1:
            best_model = data["best_model"]
            best_auc = data["metrics"][best_model]["auc"]
            
            for model_name, metrics in data["metrics"].items():
                assert metrics["auc"] <= best_auc + 0.001, \
                    f"Model {model_name} has higher AUC than best model {best_model}"

@pytest.mark.model
class TestModelPredictions:
    """Test model prediction quality"""
    
    def test_predictions_are_binary(self, client, sample_training_params, sample_features):
        """Test that predictions are binary (0 or 1)"""
        client.post("/train", params=sample_training_params)
        
        # Test multiple predictions
        for _ in range(5):
            # Generate random features
            features = np.random.rand(7).tolist()
            response = client.post("/predict", json={"features": features})
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["prediction"] in [0, 1]
    
    def test_probabilities_sum_to_one(self, client, sample_training_params, sample_features):
        """Test that prediction probabilities sum to 1"""
        client.post("/train", params=sample_training_params)
        
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        prob_sum = data["probability"]["class_0"] + data["probability"]["class_1"]
        assert abs(prob_sum - 1.0) < 0.001, "Probabilities should sum to 1"
    
    def test_probabilities_are_valid(self, client, sample_training_params, sample_features):
        """Test that probabilities are between 0 and 1"""
        client.post("/train", params=sample_training_params)
        
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert 0 <= data["probability"]["class_0"] <= 1
        assert 0 <= data["probability"]["class_1"] <= 1

@pytest.mark.model
class TestModelConsistency:
    """Test model consistency across predictions"""
    
    def test_same_input_same_output(self, client, sample_training_params, sample_features):
        """Test that same input gives same output"""
        client.post("/train", params=sample_training_params)
        
        # Make same prediction twice
        response1 = client.post("/predict", json={"features": sample_features})
        response2 = client.post("/predict", json={"features": sample_features})
        
        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["prediction"] == data2["prediction"]
        assert abs(data1["probability"]["class_0"] - data2["probability"]["class_0"]) < 0.001

@pytest.mark.model
class TestConfusionMatrix:
    """Test confusion matrix validity"""
    
    def test_confusion_matrix_sum(self, client, sample_training_params):
        """Test that confusion matrix values sum correctly"""
        client.post("/train", params=sample_training_params)
        
        response = client.get("/visualizations/confusion-matrix/random_forest")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        cm = data["confusion_matrix"]
        total = sum(sum(row) for row in cm)
        assert total > 0, "Confusion matrix should have non-zero sum"
        
        # All values should be non-negative integers
        for row in cm:
            for val in row:
                assert val >= 0
                assert isinstance(val, (int, float))

@pytest.mark.model
class TestROCCurve:
    """Test ROC curve validity"""
    
    def test_roc_curve_validity(self, client, sample_training_params):
        """Test that ROC curve data is valid"""
        client.post("/train", params=sample_training_params)
        
        response = client.get("/visualizations/roc-curve/random_forest")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        fpr = data["fpr"]
        tpr = data["tpr"]
        
        # FPR and TPR should have same length
        assert len(fpr) == len(tpr)
        
        # FPR and TPR should start at 0 and end at 1
        assert abs(fpr[0] - 0.0) < 0.01, "FPR should start near 0"
        assert abs(tpr[0] - 0.0) < 0.01, "TPR should start near 0"
        assert abs(fpr[-1] - 1.0) < 0.01, "FPR should end near 1"
        assert abs(tpr[-1] - 1.0) < 0.01, "TPR should end near 1"
        
        # All values should be between 0 and 1
        for val in fpr:
            assert 0 <= val <= 1
        for val in tpr:
            assert 0 <= val <= 1
        
        # AUC should be reasonable
        assert 0.5 <= data["auc"] <= 1.0, "AUC should be between 0.5 and 1.0"

@pytest.mark.model
class TestFeatureImportance:
    """Test feature importance validity"""
    
    def test_feature_importance_validity(self, client, sample_training_params):
        """Test that feature importance is valid"""
        client.post("/train", params=sample_training_params)
        
        response = client.get("/visualizations/feature-importance/random_forest")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        features = data["features"]
        importances = data["importances"]
        
        # Should have same number of features and importances
        assert len(features) == len(importances)
        assert len(features) > 0
        
        # All importances should be non-negative
        for imp in importances:
            assert imp >= 0
        
        # At least some features should have non-zero importance
        assert any(imp > 0 for imp in importances), "At least one feature should have importance > 0"

