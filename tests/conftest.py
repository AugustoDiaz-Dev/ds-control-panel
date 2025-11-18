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

from app.controllers.api import app

# Handle TestClient compatibility with different httpx/starlette versions
# Try to import TestClient from fastapi first, fallback to starlette
try:
    from fastapi.testclient import TestClient as FastAPITestClient
    TestClient = FastAPITestClient
except (ImportError, TypeError):
    try:
        from starlette.testclient import TestClient as StarletteTestClient
        TestClient = StarletteTestClient
    except (ImportError, TypeError):
        # Last resort: use httpx directly
        import httpx
        from httpx import ASGITransport
        class TestClient:
            def __init__(self, app):
                transport = ASGITransport(app=app)
                self._client = httpx.Client(transport=transport, base_url="http://testserver", follow_redirects=True)
            def __getattr__(self, name):
                return getattr(self._client, name)

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
    # Try different ways to create TestClient based on version compatibility
    try:
        # Method 1: Try with keyword argument (newer versions)
        return TestClient(app=app)
    except (TypeError, ValueError):
        try:
            # Method 2: Try with positional argument (older versions)
            return TestClient(app)
        except (TypeError, ValueError, AttributeError):
            # Method 3: Fallback to httpx directly for httpx 0.28+
            import httpx
            try:
                from httpx import ASGITransport
                transport = ASGITransport(app=app)
                return httpx.Client(transport=transport, base_url="http://testserver", follow_redirects=True)
            except Exception:
                # Last resort: create a simple wrapper
                raise RuntimeError("Could not create TestClient. Please check httpx and starlette versions.")

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

