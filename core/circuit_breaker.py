"""Circuit breaker implementation for external API calls."""

import time
import logging
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = 0  # Normal operation
    OPEN = 1    # Failing, reject requests
    HALF_OPEN = 2  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for external API calls."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit name for logging
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        
        logger.info(
            f"Circuit breaker initialized | name={name} | "
            f"threshold={failure_threshold} | timeout={recovery_timeout}s"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: When circuit is open
        """
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                logger.warning(f"Circuit breaker {self.name} is OPEN, rejecting call")
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Service unavailable, will retry after {self.recovery_timeout}s"
                )
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker {self.name} recovery successful, closing circuit")
                self._on_success()
            
            return result
            
        except self.expected_exception as e:
            # Failure - increment counter
            self._on_failure()
            logger.error(
                f"Circuit breaker {self.name} caught exception | "
                f"failures={self.failure_count}/{self.failure_threshold} | error={e}"
            )
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.error(
                    f"Circuit breaker {self.name} threshold reached "
                    f"({self.failure_count} failures), opening circuit"
                )
                self.state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker."""
        logger.info(f"Circuit breaker {self.name} manually reset")
        self._on_success()
    
    def get_state(self) -> str:
        """Get current state as string."""
        return self.state.name


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
):
    """
    Decorator for circuit breaker protection.
    
    Usage:
        @circuit_breaker("external_api", failure_threshold=3, recovery_timeout=30)
        def call_external_api():
            ...
    """
    breaker = CircuitBreaker(name, failure_threshold, recovery_timeout)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker for manual control
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator
