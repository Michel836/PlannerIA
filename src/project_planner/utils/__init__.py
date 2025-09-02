"""
Utilities package for PlannerIA
"""

from .error_handler import (
    ErrorHandler, 
    PlannerIAError, 
    LLMConnectionError, 
    ValidationError, 
    ProcessingError,
    handle_llm_error,
    handle_validation_error, 
    handle_processing_error,
    handle_export_error,
    safe_execute
)

__all__ = [
    'ErrorHandler',
    'PlannerIAError', 
    'LLMConnectionError', 
    'ValidationError', 
    'ProcessingError',
    'handle_llm_error',
    'handle_validation_error', 
    'handle_processing_error',
    'handle_export_error',
    'safe_execute'
]