"""
Unit tests for monitoring module
"""
import pytest
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import time

from app.monitoring.logger import setup_logging, get_logger
from app.monitoring.metrics import MetricsCollector
from app.monitoring.middleware import LoggingMiddleware, MetricsMiddleware


@pytest.mark.unit
class TestLogger:
    """Tests for logging configuration"""
    
    def test_setup_logging(self):
        """Test that logging setup works correctly"""
        # Use a temporary directory for logs
        with tempfile.TemporaryDirectory() as temp_dir:
            # This test verifies setup_logging doesn't crash
            # Note: In actual usage, logs go to BASE_DIR/logs
            setup_logging(log_level="INFO")
            logger = get_logger(__name__)
            assert logger is not None
            assert isinstance(logger, logging.Logger)
    
    def test_get_logger(self):
        """Test getting a logger instance"""
        logger = get_logger("test_module")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_logger_levels(self):
        """Test that logger respects log levels"""
        setup_logging(log_level="ERROR")
        logger = get_logger("test")
        
        # Should not crash
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


@pytest.mark.unit
class TestMetricsCollector:
    """Tests for metrics collection"""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization"""
        collector = MetricsCollector()
        assert collector.total_requests == 0
        assert collector.total_errors == 0
        assert collector.start_time is not None
    
    def test_record_request(self):
        """Test recording a request"""
        collector = MetricsCollector()
        
        collector.record_request(
            endpoint="/test",
            method="GET",
            status_code=200,
            response_time=0.1
        )
        
        assert collector.total_requests == 1
        assert collector.total_errors == 0
        assert len(collector.response_times) == 1
        assert collector.response_times[0] == 0.1
    
    def test_record_error_request(self):
        """Test recording an error request"""
        collector = MetricsCollector()
        
        collector.record_request(
            endpoint="/test",
            method="GET",
            status_code=500,
            response_time=0.2,
            error="Internal server error"
        )
        
        assert collector.total_requests == 1
        assert collector.total_errors == 1
        assert len(collector.recent_errors) == 1
        assert collector.recent_errors[0]["status_code"] == 500
    
    def test_record_model_training(self):
        """Test recording model training metrics"""
        collector = MetricsCollector()
        
        collector.record_model_training("random_forest", 5.5)
        
        assert collector.model_training_count["random_forest"] == 1
        assert len(collector.model_training_times["random_forest"]) == 1
        assert collector.model_training_times["random_forest"][0] == 5.5
    
    def test_record_model_prediction(self):
        """Test recording model prediction metrics"""
        collector = MetricsCollector()
        
        collector.record_model_prediction("random_forest", 0.05)
        
        assert collector.model_prediction_count["random_forest"] == 1
        assert len(collector.model_prediction_times["random_forest"]) == 1
        assert collector.model_prediction_times["random_forest"][0] == 0.05
    
    def test_get_stats(self):
        """Test getting statistics"""
        collector = MetricsCollector()
        
        # Record some requests
        for i in range(10):
            collector.record_request(
                endpoint=f"/test{i}",
                method="GET",
                status_code=200,
                response_time=0.1 * (i + 1)
            )
        
        stats = collector.get_stats()
        
        assert stats["total_requests"] == 10
        assert stats["total_errors"] == 0
        assert "uptime_seconds" in stats
        assert "response_times" in stats
        assert "avg" in stats["response_times"]
        assert "p50" in stats["response_times"]
        assert "p95" in stats["response_times"]
        assert "p99" in stats["response_times"]
    
    def test_get_stats_with_errors(self):
        """Test getting statistics with errors"""
        collector = MetricsCollector()
        
        # Record some requests with errors
        collector.record_request("/test", "GET", 200, 0.1)
        collector.record_request("/test", "GET", 500, 0.2, "Error")
        
        stats = collector.get_stats()
        
        assert stats["total_requests"] == 2
        assert stats["total_errors"] == 1
        assert stats["error_rate_percent"] == 50.0
    
    def test_get_recent_errors(self):
        """Test getting recent errors"""
        collector = MetricsCollector()
        
        # Record multiple errors
        for i in range(5):
            collector.record_request(
                endpoint="/test",
                method="GET",
                status_code=500,
                response_time=0.1,
                error=f"Error {i}"
            )
        
        errors = collector.get_recent_errors(limit=3)
        assert len(errors) == 3
        assert errors[-1]["error"] == "Error 4"
    
    def test_reset_metrics(self):
        """Test resetting metrics"""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_request("/test", "GET", 200, 0.1)
        collector.record_model_training("rf", 5.0)
        
        # Reset
        collector.reset()
        
        assert collector.total_requests == 0
        assert collector.total_errors == 0
        assert len(collector.model_training_count) == 0
    
    def test_response_time_percentiles(self):
        """Test response time percentile calculations"""
        collector = MetricsCollector()
        
        # Record requests with varying response times
        response_times = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        for rt in response_times:
            collector.record_request("/test", "GET", 200, rt)
        
        stats = collector.get_stats()
        response_stats = stats["response_times"]
        
        assert response_stats["min"] > 0
        assert response_stats["max"] > 0
        assert response_stats["avg"] > 0
        assert response_stats["p50"] > 0
        assert response_stats["p95"] > 0
        assert response_stats["p99"] > 0
    
    def test_model_statistics(self):
        """Test model statistics in get_stats"""
        collector = MetricsCollector()
        
        # Record training and prediction metrics
        collector.record_model_training("random_forest", 5.0)
        collector.record_model_training("random_forest", 6.0)
        collector.record_model_prediction("random_forest", 0.1)
        collector.record_model_prediction("random_forest", 0.2)
        
        stats = collector.get_stats()
        model_stats = stats["model_statistics"]
        
        assert "random_forest" in model_stats
        assert model_stats["random_forest"]["training_count"] == 2
        assert model_stats["random_forest"]["prediction_count"] == 2
        assert abs(model_stats["random_forest"]["avg_training_time"] - 5.5) < 0.01
        assert abs(model_stats["random_forest"]["avg_prediction_time"] - 0.15) < 0.01
    
    def test_max_history_size(self):
        """Test that history is limited by max_history_size"""
        collector = MetricsCollector(max_history_size=10)
        
        # Record more requests than max_history_size
        for i in range(20):
            collector.record_request("/test", "GET", 200, 0.1)
        
        # Should only keep last 10
        assert len(collector.response_times) == 10


@pytest.mark.unit
class TestMiddleware:
    """Tests for middleware classes"""
    
    def test_logging_middleware_exists(self):
        """Test that LoggingMiddleware class exists"""
        assert LoggingMiddleware is not None
        assert hasattr(LoggingMiddleware, 'dispatch')
    
    def test_metrics_middleware_exists(self):
        """Test that MetricsMiddleware class exists"""
        assert MetricsMiddleware is not None
        assert hasattr(MetricsMiddleware, 'dispatch')

