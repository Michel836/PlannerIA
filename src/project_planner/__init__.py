"""
PlannerIA - AI-Powered Project Planning System

A comprehensive multi-agent system for generating project plans from text briefs.
Includes RAG, validation, optimization, dashboard, and API capabilities.
"""

__version__ = "0.1.0"
__author__ = "PlannerIA Team"
__description__ = "AI-powered project planning system with multi-agent architecture"

# Core components
from .agents.validator import PlanValidator
from .core.optimizer import ProjectOptimizer

__all__ = [
    "PlanValidator",
    "ProjectOptimizer",
]
