"""
Logs Component - Display system logs, run history, and user feedback (FIXED ENCODING)
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go


def render_logs_section(plan_data: Dict[str, Any], current_run_id: str):
    """Render the logs and history dashboard section"""
    
    st.header("📋 Logs & History")
    
    # Create tabs for different log views
    tab1, tab2, tab3, tab4 = st.tabs(["🏃 Run History", "📊 System Logs", "💬 User Feedback", "📈 Analytics"])
    
    with tab1:
        render_run_history(current_run_id)
    
    with tab2:
        render_system_logs()
    
    with tab3:
        render_user_feedback()
    
    with tab4:
        render_usage_analytics()


def render_run_history(current_run_id: str):
    """Render project run history"""
    
    st.subheader("🏃 Project Run History")
    
    # Load all available runs
    runs_data = load_all_runs()
    
    if not runs_data:
        st.info("No project runs found")
        return
    
    # Filter and search controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("🔍 **Search runs**")
        search_query = st.text_input("", placeholder="Search by title or brief...", label_visibility="collapsed")
    
    with col2:
        st.markdown("📅 **Time Period**")
        time_filter = st.selectbox("", ["All Time", "Last 30 days", "Last 7 days", "Today"], label_visibility="collapsed")
    
    with col3:
        st.markdown("📊 **Status**")
        status_filter = st.selectbox("", ["All", "Completed", "Failed", "In Progress"], label_visibility="collapsed")
    
    # Display run statistics
    st.markdown("---")
    st.markdown("### 📊 Run Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    
    total_runs = len(runs_data)
    completed_runs = sum(1 for run in runs_data if run.get('status') == 'completed')
    failed_runs = sum(1 for run in runs_data if run.get('status') == 'failed')
    avg_duration = sum(run.get('duration', 0) for run in runs_data) / total_runs if total_runs > 0 else 0
    avg_cost = sum(run.get('cost', 0) for run in runs_data) / total_runs if total_runs > 0 else 0
    
    with stat_col1:
        st.metric("**Total Runs**", total_runs)
    
    with stat_col2:
        st.metric("**Completed**", completed_runs)
    
    with stat_col3:
        st.metric("**Failed**", failed_runs)
    
    with stat_col4:
        st.metric("**Avg Duration**", f"{avg_duration:.1f} days")
    
    with stat_col5:
        st.metric("**Avg Cost**", f"${avg_cost:,.0f}")
    
    # Display detailed run information
    st.markdown("---")
    st.markdown("### 📋 Run Details")
    st.markdown("🔍 **Detailed Run Information**")
    
    # Filter runs based on search and filters
    filtered_runs = filter_runs(runs_data, search_query, time_filter, status_filter)
    
    for run in filtered_runs:
        with st.expander(f"📁 {run.get('name', 'Unknown Project')} - {run.get('id', 'N/A')[:8]}..."):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {run.get('status', 'Unknown')}")
                st.write(f"**Duration:** {run.get('duration', 0)} days")
            
            with col2:
                st.write(f"**Cost:** ${run.get('cost', 0):,}")
                st.write(f"**Tasks:** {run.get('task_count', 0)}")
            
            with col3:
                st.write(f"**Created:** {run.get('created_at', 'Unknown')}")
                st.write(f"**Completed:** {run.get('completed_at', 'N/A')}")
            
            if st.button(f"View Details", key=f"view_{run.get('id')}"):
                st.session_state.selected_run_id = run.get('id')
                st.success("✅ Action terminée")


def render_system_logs():
    """Render system logs section"""
    st.subheader("📊 System Logs")
    st.info("System logs functionality - To be implemented")


def render_user_feedback():
    """Render user feedback section"""
    st.subheader("💬 User Feedback")
    st.info("User feedback functionality - To be implemented")


def render_usage_analytics():
    """Render usage analytics"""
    st.subheader("📈 Usage Analytics")
    st.info("Usage analytics functionality - To be implemented")


def load_all_runs() -> List[Dict[str, Any]]:
    """Load all available project runs"""
    # Mock data for demonstration
    return [
        {
            'id': '1c3e0146...',
            'name': 'Application E-commerce avec Paiement',
            'status': 'completed',
            'duration': 28,
            'cost': 32000,
            'task_count': 45,
            'created_at': '2024-01-15',
            'completed_at': '2024-02-12'
        },
        {
            'id': '0e306a97...',
            'name': 'Application E-commerce avec Paiement',
            'status': 'completed',
            'duration': 22,
            'cost': 18500,
            'task_count': 38,
            'created_at': '2024-01-10',
            'completed_at': '2024-02-01'
        },
        # Add more mock data...
    ] * 5  # Simulate 10+ runs


def filter_runs(runs: List[Dict], search: str, time_filter: str, status_filter: str) -> List[Dict]:
    """Filter runs based on search criteria"""
    filtered = runs
    
    if search:
        filtered = [run for run in filtered if search.lower() in run.get('name', '').lower()]
    
    if status_filter != "All":
        filtered = [run for run in filtered if run.get('status') == status_filter.lower()]
    
    # Add time filtering logic here if needed
    
    return filtered[:10]  # Limit to first 10 for display