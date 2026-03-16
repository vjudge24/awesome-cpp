"""Tests for resilience patterns (retry, circuit breaker)."""

from unittest.mock import MagicMock

import pytest

from src.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    get_breaker,
    with_retry,
)


class TestRetry:
    def test_success_no_retry(self):
        fn = MagicMock(return_value=42)

        @with_retry(max_attempts=3, base_delay=0.001)
        def call():
            return fn()

        assert call() == 42
        assert fn.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        fn = MagicMock(side_effect=[ValueError("fail"), ValueError("fail"), 42])

        @with_retry(max_attempts=3, base_delay=0.001)
        def call():
            return fn()

        assert call() == 42
        assert fn.call_count == 3

    def test_exhausts_retries(self):
        fn = MagicMock(side_effect=ValueError("always fail"))

        @with_retry(max_attempts=2, base_delay=0.001)
        def call():
            return fn()

        with pytest.raises(ValueError, match="always fail"):
            call()
        assert fn.call_count == 2

    def test_only_retries_specified_exceptions(self):
        fn = MagicMock(side_effect=TypeError("wrong type"))

        @with_retry(max_attempts=3, base_delay=0.001, retryable_exceptions=(ValueError,))
        def call():
            return fn()

        with pytest.raises(TypeError):
            call()
        assert fn.call_count == 1  # no retry for TypeError


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        import time
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_call_rejects_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
        cb.record_failure()
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: 42)

    def test_call_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        import time
        time.sleep(0.02)
        # Now HALF_OPEN, a successful call should close it
        result = cb.call(lambda: "ok")
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_call_failure_reopens_from_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        import time
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert cb.state == CircuitState.OPEN


class TestGetBreaker:
    def test_returns_same_instance(self):
        # Use unique names to avoid interference between tests
        b1 = get_breaker("test_same_instance")
        b2 = get_breaker("test_same_instance")
        assert b1 is b2

    def test_different_names_different_instances(self):
        b1 = get_breaker("test_a_unique")
        b2 = get_breaker("test_b_unique")
        assert b1 is not b2
