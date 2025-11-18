"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add app to path
BASE_DIR = Path(__file__).parent.parent
APP_DIR = BASE_DIR / "app"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from app.controllers.api import app

@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test to avoid test interference"""
    # Import here to avoid circular imports
    from app.controllers import api
    
    # Save original state
    original_trained_models = dict(api.trained_models)
    original_model_metrics = dict(api.model_metrics)
    original_X_test = api.X_test_global
    original_y_test = api.y_test_global
    original_X_train = api.X_train_global
    original_y_train = api.y_train_global
    original_feature_names = api.feature_names
    original_best_model = api.best_model_name
    original_ensembles = dict(api.ensemble_models)
    original_shap_cache = dict(api.shap_values_cache)
    
    # Clear state before test
    api.trained_models.clear()
    api.model_metrics.clear()
    api.X_test_global = None
    api.y_test_global = None
    api.X_train_global = None
    api.y_train_global = None
    api.feature_names = None
    api.best_model_name = None
    api.ensemble_models.clear()
    api.shap_values_cache.clear()
    
    yield
    
    # Restore state after test
    api.trained_models.clear()
    api.trained_models.update(original_trained_models)
    api.model_metrics.clear()
    api.model_metrics.update(original_model_metrics)
    api.X_test_global = original_X_test
    api.y_test_global = original_y_test
    api.X_train_global = original_X_train
    api.y_train_global = original_y_train
    api.feature_names = original_feature_names
    api.best_model_name = original_best_model
    api.ensemble_models.clear()
    api.ensemble_models.update(original_ensembles)
    api.shap_values_cache.clear()
    api.shap_values_cache.update(original_shap_cache)

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    # TestClient should work with app as positional argument
    # This is the standard way for FastAPI
    return TestClient(app)

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_features():
    """Sample feature vector for testing predictions"""
    return [750, 80000, 15000, 35, 60000, 8, 25000]

@pytest.fixture
def sample_training_params():
    """Sample training parameters"""
    return {
        "n_estimators": 10,  # Small for faster tests
        "learning_rate": 0.1,
        "max_depth": 3,
        "include_catboost": False
    }

