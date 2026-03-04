"""
Simulation module for HT-HGNN v2.0

Provides cascade simulation, stress testing, and scenario construction
tools for analyzing supply chain hypergraph resilience.
"""

from .cascade_engine import CascadeEngine, CascadeResult
from .stress_tester import StressTester, StressTestResult
from .scenario_builder import ScenarioBuilder, Scenario

__all__ = [
    'CascadeEngine',
    'CascadeResult',
    'StressTester',
    'StressTestResult',
    'ScenarioBuilder',
    'Scenario',
]
