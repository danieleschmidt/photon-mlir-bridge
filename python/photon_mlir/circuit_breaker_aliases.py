"""
Circuit breaker aliases for backward compatibility.
"""

from .circuit_breaker import ThermalCircuitBreaker, PhaseCoherenceCircuitBreaker, CircuitState

# Aliases for backward compatibility
CircuitBreaker = ThermalCircuitBreaker