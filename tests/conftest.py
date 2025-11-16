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

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
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

