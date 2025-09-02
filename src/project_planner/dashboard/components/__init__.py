"""
Module de composants pour le dashboard Streamlit PlannerIA
"""

# Imports des composants utilisés par app.py
from .gantt import render_gantt_chart
from .kpi import render_kpi_section
from .logs import render_logs_section
from .risks import render_risk_analysis_dashboard
from .whatif import render_whatif_simulator

# Exports
__all__ = [
    'render_gantt_chart',
    'render_kpi_section',
    'render_logs_section',
    'render_risk_analysis_dashboard',
    'render_whatif_simulator'
]