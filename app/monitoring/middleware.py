"""
Middleware for request/response logging and metrics collection
"""
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.monitoring.logger import get_logger
from app.monitoring.metrics import metrics_collector

logger = get_logger(__name__)
access_logger = get_logger("access")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all HTTP requests and responses
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details"""
        start_time = time.time()
        
        # Log request
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        
        logger.debug(
            f"Request: {method} {path} | "
            f"IP: {client_ip} | "
            f"Query: {query_params}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            status_code = response.status_code
            access_logger.info(
                f"{method} {path} | "
                f"Status: {status_code} | "
                f"Time: {process_time:.3f}s | "
                f"IP: {client_ip}"
            )
            
            # Record metrics
            error_msg = None
            if status_code >= 400:
                error_msg = f"HTTP {status_code}"
            
            metrics_collector.record_request(
                endpoint=path,
                method=method,
                status_code=status_code,
                response_time=process_time,
                error=error_msg
            )
            
            # Add response time header
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(
                f"Request failed: {method} {path} | "
                f"Error: {error_msg} | "
                f"Time: {process_time:.3f}s | "
                f"IP: {client_ip}",
                exc_info=True
            )
            
            # Record error metric
            metrics_collector.record_request(
                endpoint=path,
                method=method,
                status_code=500,
                response_time=process_time,
                error=error_msg
            )
            
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting detailed metrics
    This is a lightweight version that can be used alongside LoggingMiddleware
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Metrics are already recorded in LoggingMiddleware
            # This middleware can be used for additional metric collection if needed
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Error in metrics middleware: {str(e)}", exc_info=True)
            raise

