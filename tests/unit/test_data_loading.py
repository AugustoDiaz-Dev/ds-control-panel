"""
Unit tests for data loading and preparation
"""
import pytest
import sys
from pathlib import Path

# Add app to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.controllers.api import load_or_generate_data, prepare_data

@pytest.mark.unit
class TestDataLoading:
    """Test data loading functions"""
    
    def test_load_or_generate_data(self):
        """Test that data can be loaded or generated"""
        df = load_or_generate_data()
        assert df is not None
        assert len(df) > 0
        assert 'target' in df.columns
    
    def test_prepare_data(self):
        """Test data preparation"""
        df = load_or_generate_data()
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Check shapes match
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # Check target is binary
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

