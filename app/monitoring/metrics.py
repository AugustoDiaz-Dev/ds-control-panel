"""
Metrics collection for monitoring API performance and usage
"""
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from threading import Lock


class MetricsCollector:
    """
    Thread-safe metrics collector for API monitoring
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Initialize metrics collector
        
        Args:
            max_history_size: Maximum number of requests to keep in history
        """
        self._lock = Lock()
        self.max_history_size = max_history_size
        
        # Request counters
        self.total_requests = 0
        self.total_errors = 0
        self.requests_by_endpoint = defaultdict(int)
        self.errors_by_endpoint = defaultdict(int)
        self.requests_by_status = defaultdict(int)
        
        # Response time tracking
        self.response_times = deque(maxlen=max_history_size)
        self.response_times_by_endpoint = defaultdict(lambda: deque(maxlen=max_history_size))
        
        # Error tracking
        self.recent_errors = deque(maxlen=100)
        
        # Timestamps
        self.start_time = datetime.now()
        self.last_request_time: Optional[datetime] = None
        
        # Model-specific metrics
        self.model_training_count = defaultdict(int)
        self.model_prediction_count = defaultdict(int)
        self.model_training_times = defaultdict(list)
        self.model_prediction_times = defaultdict(list)
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        error: Optional[str] = None
    ) -> None:
        """
        Record a request metric
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            status_code: HTTP status code
            response_time: Response time in seconds
            error: Error message if any
        """
        with self._lock:
            self.total_requests += 1
            self.last_request_time = datetime.now()
            
            endpoint_key = f"{method} {endpoint}"
            self.requests_by_endpoint[endpoint_key] += 1
            self.requests_by_status[status_code] += 1
            
            self.response_times.append(response_time)
            self.response_times_by_endpoint[endpoint_key].append(response_time)
            
            if status_code >= 400:
                self.total_errors += 1
                self.errors_by_endpoint[endpoint_key] += 1
                
                if error:
                    self.recent_errors.append({
                        "timestamp": datetime.now().isoformat(),
                        "endpoint": endpoint_key,
                        "status_code": status_code,
                        "error": error
                    })
    
    def record_model_training(self, model_name: str, duration: float) -> None:
        """
        Record model training metric
        
        Args:
            model_name: Name of the model
            duration: Training duration in seconds
        """
        with self._lock:
            self.model_training_count[model_name] += 1
            self.model_training_times[model_name].append(duration)
            # Keep only last 100 training times per model
            if len(self.model_training_times[model_name]) > 100:
                self.model_training_times[model_name] = self.model_training_times[model_name][-100:]
    
    def record_model_prediction(self, model_name: str, duration: float) -> None:
        """
        Record model prediction metric
        
        Args:
            model_name: Name of the model
            duration: Prediction duration in seconds
        """
        with self._lock:
            self.model_prediction_count[model_name] += 1
            self.model_prediction_times[model_name].append(duration)
            # Keep only last 1000 prediction times per model
            if len(self.model_prediction_times[model_name]) > 1000:
                self.model_prediction_times[model_name] = self.model_prediction_times[model_name][-1000:]
    
    def get_stats(self) -> Dict:
        """
        Get current statistics
        
        Returns:
            Dictionary with current metrics
        """
        with self._lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate response time statistics
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                sorted_times = sorted(self.response_times)
                p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
                p95 = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
                p99 = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
                max_response_time = max(self.response_times)
                min_response_time = min(self.response_times)
            else:
                avg_response_time = p50 = p95 = p99 = max_response_time = min_response_time = 0
            
            # Calculate error rate
            error_rate = (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0
            
            # Calculate requests per second
            requests_per_second = self.total_requests / uptime if uptime > 0 else 0
            
            # Get top endpoints
            top_endpoints = dict(
                sorted(
                    self.requests_by_endpoint.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            )
            
            # Get model statistics
            model_stats = {}
            for model_name in set(list(self.model_training_count.keys()) + list(self.model_prediction_count.keys())):
                training_times = self.model_training_times.get(model_name, [])
                prediction_times = self.model_prediction_times.get(model_name, [])
                
                model_stats[model_name] = {
                    "training_count": self.model_training_count.get(model_name, 0),
                    "prediction_count": self.model_prediction_count.get(model_name, 0),
                    "avg_training_time": sum(training_times) / len(training_times) if training_times else 0,
                    "avg_prediction_time": sum(prediction_times) / len(prediction_times) if prediction_times else 0,
                }
            
            return {
                "uptime_seconds": uptime,
                "uptime_formatted": str(timedelta(seconds=int(uptime))),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate_percent": round(error_rate, 2),
                "requests_per_second": round(requests_per_second, 2),
                "response_times": {
                    "avg": round(avg_response_time * 1000, 2),  # Convert to ms
                    "p50": round(p50 * 1000, 2),
                    "p95": round(p95 * 1000, 2),
                    "p99": round(p99 * 1000, 2),
                    "min": round(min_response_time * 1000, 2),
                    "max": round(max_response_time * 1000, 2),
                },
                "requests_by_status": dict(self.requests_by_status),
                "top_endpoints": top_endpoints,
                "model_statistics": model_stats,
                "recent_errors_count": len(self.recent_errors),
                "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            }
    
    def get_recent_errors(self, limit: int = 20) -> List[Dict]:
        """
        Get recent errors
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error dictionaries
        """
        with self._lock:
            return list(self.recent_errors)[-limit:]
    
    def reset(self) -> None:
        """Reset all metrics (use with caution)"""
        with self._lock:
            self.total_requests = 0
            self.total_errors = 0
            self.requests_by_endpoint.clear()
            self.errors_by_endpoint.clear()
            self.requests_by_status.clear()
            self.response_times.clear()
            self.response_times_by_endpoint.clear()
            self.recent_errors.clear()
            self.model_training_count.clear()
            self.model_prediction_count.clear()
            self.model_training_times.clear()
            self.model_prediction_times.clear()
            self.start_time = datetime.now()
            self.last_request_time = None


# Global metrics collector instance
metrics_collector = MetricsCollector()

