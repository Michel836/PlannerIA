"""
PlannerIA Dash Modules Package
Universal module system for AI-powered project management
"""

from .communication_bus import get_communication_bus, apply_to_project, export_data, share_insights
from .ai_dashboard_module import get_ai_dashboard_module

__all__ = [
    'get_communication_bus',
    'apply_to_project', 
    'export_data',
    'share_insights',
    'get_ai_dashboard_module'
]