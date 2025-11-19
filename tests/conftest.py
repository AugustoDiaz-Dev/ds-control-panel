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
# Use starlette TestClient directly as it's more compatible
try:
    from starlette.testclient import TestClient
except ImportError:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        raise ImportError("Could not import TestClient. Please install starlette or fastapi.")

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
    # Starlette TestClient should work with both positional and keyword args
    try:
        return TestClient(app)
    except (TypeError, ValueError, AttributeError):
        try:
            return TestClient(app=app)
        except Exception as e:
            raise RuntimeError(f"Could not create TestClient: {e}. Please check starlette version.")

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

# ==================== Authentication Fixtures ====================

@pytest.fixture(autouse=True)
def reset_user_storage():
    """Reset user storage before each test to avoid test interference"""
    from app.auth.user_manager import user_manager
    import json
    from pathlib import Path
    
    # Save original users file if it exists
    BASE_DIR = Path(__file__).parent.parent
    users_file = BASE_DIR / "users.json"
    original_users = {}
    
    if users_file.exists():
        try:
            with open(users_file, 'r') as f:
                original_users = json.load(f)
        except Exception:
            pass
    
    # Clear users before test
    user_manager._users.clear()
    user_manager._ensure_default_user()
    
    yield
    
    # Restore users after test
    user_manager._users.clear()
    if original_users:
        try:
            with open(users_file, 'w') as f:
                json.dump(original_users, f, indent=2)
            user_manager._load_users()
        except Exception:
            pass
    else:
        # If no original users, ensure default user exists
        user_manager._ensure_default_user()

@pytest.fixture
def test_user():
    """Create a test user for authentication"""
    from app.auth.user_manager import user_manager
    from app.auth.models import UserCreate
    
    user_create = UserCreate(
        username="testuser",
        email="test@example.com",
        password="testpass123",
        full_name="Test User",
        is_active=True,
        is_admin=False
    )
    
    user = user_manager.create_user(user_create)
    yield user
    
    # Cleanup
    try:
        user_manager.delete_user(user.id)
    except Exception:
        pass

@pytest.fixture
def test_admin_user():
    """Create a test admin user for authentication"""
    from app.auth.user_manager import user_manager
    from app.auth.models import UserCreate
    
    user_create = UserCreate(
        username="testadmin",
        email="admin@example.com",
        password="adminpass123",
        full_name="Test Admin",
        is_active=True,
        is_admin=True
    )
    
    user = user_manager.create_user(user_create)
    yield user
    
    # Cleanup
    try:
        user_manager.delete_user(user.id)
    except Exception:
        pass

@pytest.fixture
def auth_token(client, test_user):
    """Get authentication token for test user"""
    response = client.post(
        "/auth/login",
        data={"username": "testuser", "password": "testpass123"}
    )
    assert response.status_code == 200
    token_data = response.json()
    return token_data["access_token"]

@pytest.fixture
def admin_token(client, test_admin_user):
    """Get authentication token for test admin user"""
    response = client.post(
        "/auth/login",
        data={"username": "testadmin", "password": "adminpass123"}
    )
    assert response.status_code == 200
    token_data = response.json()
    return token_data["access_token"]

@pytest.fixture
def authenticated_client(client, auth_token):
    """Create an authenticated test client"""
    # Create a new client with auth headers
    class AuthenticatedClient:
        def __init__(self, base_client, token):
            self._client = base_client
            self._token = token
            self._headers = {"Authorization": f"Bearer {token}"}
        
        def get(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.get(url, headers=headers, **kwargs)
        
        def post(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.post(url, headers=headers, **kwargs)
        
        def put(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.put(url, headers=headers, **kwargs)
        
        def delete(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.delete(url, headers=headers, **kwargs)
        
        def patch(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.patch(url, headers=headers, **kwargs)
    
    return AuthenticatedClient(client, auth_token)

@pytest.fixture
def admin_client(client, admin_token):
    """Create an authenticated admin test client"""
    class AuthenticatedClient:
        def __init__(self, base_client, token):
            self._client = base_client
            self._token = token
            self._headers = {"Authorization": f"Bearer {token}"}
        
        def get(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.get(url, headers=headers, **kwargs)
        
        def post(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.post(url, headers=headers, **kwargs)
        
        def put(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.put(url, headers=headers, **kwargs)
        
        def delete(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.delete(url, headers=headers, **kwargs)
        
        def patch(self, url, **kwargs):
            headers = kwargs.pop("headers", {})
            headers.update(self._headers)
            return self._client.patch(url, headers=headers, **kwargs)
    
    return AuthenticatedClient(client, admin_token)

