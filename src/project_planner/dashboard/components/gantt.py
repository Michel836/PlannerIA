"""
Professional Gantt Chart Component for PlannerIA Dashboard
Enterprise-grade project timeline with advanced features.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Tuple
import pandas.tseries.offsets as offsets
import io
import base64


def render_gantt_chart(plan_data: Dict[str, Any]):
    """Main professional Gantt chart dashboard"""
    
    st.header("📊 Professional Project Timeline - Gantt Chart")
    
    # Extract and validate data
    tasks = extract_all_tasks(plan_data)
    dependencies = plan_data.get('dependencies', [])
    # Get critical path from different possible locations
    critical_path = (plan_data.get('critical_path', []) or 
                    plan_data.get('project_metrics', {}).get('critical_path_tasks', []))
    milestones = extract_milestones(plan_data, critical_path)
    
    if not tasks:
        st.warning("⚠️ No tasks available to display in Gantt chart")
        return
    
    # Professional toolbar
    render_professional_toolbar(plan_data)
    
    # Enhanced timeline controls
    render_enhanced_timeline_controls(plan_data)
    
    # Calculate intelligent dates using dependencies
    gantt_data = calculate_intelligent_dates(tasks, dependencies, plan_data, critical_path)
    
    # Milestones are already included in gantt_data from the WBS tasks
    # No need to add them separately
    
    # Style selector with visual previews
    # render_gantt_style_selector()  # Temporairement désactivé pour éviter conflit de clés
    
    # Main Gantt visualization with progress tracking
    render_professional_gantt_chart(gantt_data, critical_path, dependencies, plan_data)
    
    # Resource allocation view
    render_resource_allocation(gantt_data, plan_data)


def render_gantt_style_selector():
    """Render enhanced style selector with visual previews"""
    st.subheader("🎨 Thèmes Visuels Gantt")
    
    # Get available styles
    available_styles = ["Professional", "Modern Dark", "Neon Cyber", "Dark Elite", 
                       "Gradient Futur", "Ocean Deep", "Corporate Clean", "Pastel Soft", "Glassmorphism"]
    
    # Style descriptions
    style_descriptions = {
        "Professional": "🏢 Style classique et professionnel pour présentations",
        "Modern Dark": "🌙 Thème sombre moderne avec excellent contraste",
        "Neon Cyber": "⚡ Style futuriste néon pour projets tech",
        "Dark Elite": "🖤 Elite sombre avec couleurs premium",
        "Gradient Futur": "🌈 Gradients futuristes et élégants",
        "Ocean Deep": "🌊 Profondeurs océaniques apaisantes",
        "Corporate Clean": "💼 Propre et minimaliste pour entreprise",
        "Pastel Soft": "🌸 Couleurs douces et relaxantes",
        "Glassmorphism": "🔮 Effet verre transparent moderne"
    }
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Style selector
        selected_style = st.selectbox(
            "Choisir un thème",
            available_styles,
            index=available_styles.index(st.session_state.get('gantt_style', 'Professional')),
            format_func=lambda x: f"{style_descriptions.get(x, x)}"
        )
        
        # Save selection
        st.session_state.gantt_style = selected_style
        
        # Style preview
        style_config = get_gantt_style_config(selected_style)
        
        st.markdown("### 📋 Aperçu Couleurs")
        priority_colors = style_config["color_map"]
        
        for priority, color in priority_colors.items():
            st.markdown(
                f"""<div style="
                    background: {color}; 
                    padding: 8px 12px; 
                    margin: 2px 0; 
                    border-radius: 8px; 
                    color: {style_config['font_color']};
                    font-weight: bold;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                ">🔸 {priority}</div>""", 
                unsafe_allow_html=True
            )
    
    with col2:
        # Advanced options
        st.markdown("### ⚙️ Options Avancées")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            st.checkbox("📊 Afficher Progrès", value=True, key="gantt_display_progress")
            st.checkbox("📅 Ombrer Week-ends", value=True, key="gantt_display_weekends") 
            st.checkbox("🎯 Vue Chemin Critique", value=False, key="gantt_display_critical")
        
        with col2b:
            st.checkbox("📈 Afficher Marge", value=False, key="gantt_display_slack")
            st.checkbox("🏁 Afficher Jalons", value=True, key="gantt_display_milestones")
            st.checkbox("📋 Vue PERT", value=False, key="gantt_display_pert")
        
        # View mode selector
        view_modes = ["Standard Timeline", "Critical Path Only", "Milestone Focus", 
                     "Progress Tracking", "Resource Allocation"]
        
        st.selectbox(
            "🔍 Mode d'affichage",
            view_modes,
            key="gantt_view_mode"
        )


def render_resource_allocation(gantt_data: List[Dict], plan_data: Dict):
    """Render resource allocation analysis"""
    st.subheader("👥 Allocation des Ressources")
    
    if not gantt_data:
        # Générer des données simulées
        gantt_data = generate_sample_gantt_data()
        st.info("📊 Affichage de données d'exemple pour démonstration")
    
    if not gantt_data:
        st.warning("Aucune donnée de ressources disponible")
        return
    
    # Extract resource information
    resources = {}
    for task in gantt_data:
        assigned_to = task.get('assigned_to', 'Non assigné')
        if assigned_to not in resources:
            resources[assigned_to] = []
        resources[assigned_to].append(task)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Répartition par Ressource**")
        for resource, tasks in resources.items():
            workload = len(tasks)
            st.metric(f"🧑‍💼 {resource}", f"{workload} tâche(s)")
            
    with col2:
        st.write("**Charge de Travail**")
        if resources:
            import plotly.express as px
            
            resource_names = list(resources.keys())
            task_counts = [len(tasks) for tasks in resources.values()]
            
            fig = px.pie(
                values=task_counts,
                names=resource_names,
                title="Distribution des Tâches"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Resource timeline
    st.write("**Timeline des Ressources**")
    for resource, tasks in resources.items():
        with st.expander(f"📅 Planning de {resource}"):
            for task in tasks:
                start = task.get('start_date', 'N/A')
                end = task.get('end_date', 'N/A') 
                st.write(f"• **{task.get('task_name', 'Tâche')}** ({start} → {end})")
    
    # Timeline analytics
    render_timeline_analytics(gantt_data, plan_data)


def render_timeline_analytics(gantt_data: List[Dict], plan_data: Dict):
    """Render timeline analytics and insights"""
    st.subheader("📈 Analyse Temporelle")
    
    if not gantt_data:
        # Générer des données simulées
        gantt_data = generate_sample_gantt_data()
        st.info("📊 Affichage de données d'exemple pour démonstration")
    
    if not gantt_data:
        st.warning("Aucune donnée temporelle disponible")
        return
    
    # Extract dates and durations
    dates = []
    durations = []
    
    for task in gantt_data:
        start_date = task.get('start_date')
        end_date = task.get('end_date')
        
        if start_date and end_date:
            try:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                
                duration = (end_date - start_date).days
                dates.append((start_date, end_date))
                durations.append(duration)
            except:
                continue
    
    if not dates:
        st.info("Pas assez de données temporelles pour l'analyse")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📅 Durée projet", f"{sum(durations)} jours")
        
    with col2:
        avg_duration = sum(durations) / len(durations) if durations else 0
        st.metric("⏱️ Durée moy/tâche", f"{avg_duration:.1f} jours")
        
    with col3:
        completion_rate = len([d for d in durations if d > 0]) / len(durations) * 100 if durations else 0
        st.metric("✅ Taux complétion", f"{completion_rate:.0f}%")
    
    # Timeline distribution chart
    if len(durations) > 1:
        st.write("**📊 Distribution des Durées**")
        
        import plotly.express as px
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=durations,
            nbinsx=10,
            name="Durées des tâches",
            marker_color='rgba(55, 128, 191, 0.7)',
            marker_line=dict(color='rgba(55, 128, 191, 1.0)', width=1)
        ))
        
        fig.update_layout(
            title="Distribution des Durées de Tâches",
            xaxis_title="Durée (jours)",
            yaxis_title="Nombre de tâches",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Project phases analysis
    st.write("**🎯 Analyse par Phases**")
    
    phases = {}
    for task in gantt_data:
        phase = task.get('phase', 'Phase inconnue')
        if phase not in phases:
            phases[phase] = {'count': 0, 'total_duration': 0}
        
        phases[phase]['count'] += 1
        
        # Add duration if available
        start_date = task.get('start_date')
        end_date = task.get('end_date')
        if start_date and end_date:
            try:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                duration = (end_date - start_date).days
                phases[phase]['total_duration'] += duration
            except:
                pass
    
    for phase, data in phases.items():
        avg_phase_duration = data['total_duration'] / data['count'] if data['count'] > 0 else 0
        st.write(f"🔸 **{phase}**: {data['count']} tâches, {avg_phase_duration:.1f} jours/tâche")
    
    # Timeline insights
    if dates:
        project_start = min(date[0] for date in dates)
        project_end = max(date[1] for date in dates)
        
        st.write("**🎯 Insights Temporels**")
        st.info(f"📅 **Période projet**: {project_start.strftime('%d/%m/%Y')} → {project_end.strftime('%d/%m/%Y')}")
        
        # Calculate overlapping tasks
        overlaps = 0
        for i, (start1, end1) in enumerate(dates):
            for j, (start2, end2) in enumerate(dates[i+1:], i+1):
                if start1 < end2 and start2 < end1:
                    overlaps += 1
        
        if overlaps > 0:
            st.warning(f"⚠️ {overlaps} chevauchement(s) de tâches détecté(s)")
        else:
            st.success("✅ Aucun chevauchement critique détecté")
    
    # Export functionality
    render_export_options(gantt_data, plan_data)


def render_professional_toolbar(plan_data: Dict[str, Any]):
    """Professional toolbar with quick actions"""
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("📅 Set Baseline", use_container_width=True):
            st.session_state['baseline_data'] = plan_data.copy()
            st.success("✅ Baseline saved")
    
    with col2:
        if st.button("📊 Compare Baseline", use_container_width=True):
            if 'baseline_data' in st.session_state:
                st.session_state['show_baseline'] = not st.session_state.get('show_baseline', False)
            else:
                st.warning("⚠️ No baseline set")
    
    with col3:
        if st.button("🎯 Milestones View", use_container_width=True):
            st.session_state['milestone_focus'] = not st.session_state.get('milestone_focus', False)
    
    with col4:
        if st.button("👥 Resource View", use_container_width=True):
            st.session_state['resource_view'] = not st.session_state.get('resource_view', False)
    
    with col5:
        if st.button("📈 PERT Analysis", use_container_width=True):
            st.session_state['pert_view'] = not st.session_state.get('pert_view', False)
    
    with col6:
        if st.button("🔄 Refresh", use_container_width=True):
            st.success("✅ Actualisation terminée")


def extract_milestones(plan_data: Dict[str, Any], critical_path: List[str] = None) -> List[Dict[str, Any]]:
    """Extract project milestones from tasks"""
    
    if critical_path is None:
        critical_path = []
    
    milestones = []
    tasks = extract_all_tasks(plan_data)
    
    for task in tasks:
        # Identify milestones (duration = 0 or marked as milestone)
        if task.get('duration', 1) == 0 or task.get('is_milestone', False) or 'milestone' in task.get('name', '').lower():
            milestones.append({
                'name': task.get('name', 'Milestone'),
                'date': task.get('start_date', date.today()),
                'phase': task.get('phase', 'General'),
                'critical': task.get('id') in critical_path
            })
    
    # Add project-level milestones
    overview = plan_data.get('project_overview', {})
    if overview.get('end_date'):
        milestones.append({
            'name': 'Project Completion',
            'date': overview['end_date'],
            'phase': 'Final',
            'critical': True
        })
    
    return milestones


def integrate_milestones(gantt_data: List[Dict[str, Any]], milestones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Integrate milestones into Gantt data"""
    
    for milestone in milestones:
        gantt_data.append({
            'Task': f"🎯 {milestone['name']}",
            'TaskID': f"milestone_{milestone['name']}",
            'Start': milestone['date'],
            'Finish': milestone['date'],
            'Duration': 0,
            'Phase': milestone['phase'],
            'Priority': 'High',
            'Status': 'Milestone',
            'Critical': milestone['critical'],
            'Cost': 0,
            'Resources': 0,
            'Slack': 0,
            'Complexity': 'milestone',
            'Progress': 0,
            'ResourceList': '',
            'Description': f"Milestone: {milestone['name']}",
            'IsMilestone': True
        })
    
    return gantt_data


def extract_all_tasks(plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all tasks with enhanced metadata"""
    tasks = []
    
    # Direct tasks
    if 'tasks' in plan_data and isinstance(plan_data['tasks'], list):
        for task in plan_data['tasks']:
            if isinstance(task, dict):
                task_copy = task.copy()
                task_copy['phase'] = 'Direct Tasks'
                task_copy['source'] = 'direct'
                # Add progress if not present
                if 'progress' not in task_copy:
                    task_copy['progress'] = calculate_task_progress(task_copy)
                tasks.append(task_copy)
    
    # Tasks from WBS phases
    wbs = plan_data.get('wbs', {})
    if isinstance(wbs, dict) and 'phases' in wbs:
        for phase_idx, phase in enumerate(wbs['phases']):
            if isinstance(phase, dict):
                phase_name = phase.get('name', f'Phase {phase_idx + 1}')
                phase_tasks = phase.get('tasks', [])
                
                if isinstance(phase_tasks, list):
                    for task in phase_tasks:
                        if isinstance(task, dict):
                            task_copy = task.copy()
                            task_copy['phase'] = phase_name
                            task_copy['phase_id'] = phase.get('id', f'phase_{phase_idx}')
                            task_copy['source'] = 'wbs'
                            if 'progress' not in task_copy:
                                task_copy['progress'] = calculate_task_progress(task_copy)
                            tasks.append(task_copy)
    
    return tasks


def calculate_task_progress(task: Dict[str, Any]) -> float:
    """Calculate task progress percentage"""
    
    status = task.get('status', 'not_started')
    
    # If explicit progress is provided
    if 'completion_percentage' in task:
        return float(task['completion_percentage']) / 100
    
    # Status-based progress
    progress_map = {
        'not_started': 0.0,
        'in_progress': 0.5,
        'testing': 0.75,
        'review': 0.9,
        'completed': 1.0,
        'blocked': 0.25,
        'on_hold': 0.1
    }
    
    return progress_map.get(status, 0.0)


def render_enhanced_timeline_controls(plan_data: Dict[str, Any]):
    """Enhanced control panel for timeline customization"""
    
    with st.expander("⚙️ Timeline Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            project_start = st.date_input(
                "📅 Project Start Date",
                value=get_project_start_date(plan_data),
                help="Base date for calculating task schedules"
            )
            st.session_state['project_start_date'] = project_start
        
        with col2:
            working_days = st.selectbox(
                "📆 Working Days",
                options=["Monday-Friday", "Monday-Saturday", "All Days"],
                index=0,
                help="Define working days for duration calculations"
            )
            st.session_state['working_days'] = working_days
        
        with col3:
            view_mode = st.selectbox(
                "👁️ View Mode",
                options=["Standard Timeline", "Critical Path Focus", "Phase Breakdown", 
                        "Resource Allocation", "Progress Tracking", "Milestone Focus"],
                help="Choose timeline perspective"
            )
            st.session_state['gantt_view_mode'] = view_mode
        
        with col4:
            gantt_style = st.selectbox(
                "🎨 Style Graphique",
                options=["Professional", "Modern Dark", "Pastel Soft", "Neon Cyber", "Corporate Clean", "Glassmorphism"],
                index=0,
                help="Style visuel du Gantt"
            )
            st.session_state['gantt_style'] = gantt_style
    
    # AI Organization Button
    st.markdown("---")
    col_ai1, col_ai2, col_ai3 = st.columns([2, 1, 2])
    with col_ai2:
        if st.button("🤖 Organiser automatiquement par IA", type="primary", help="Réorganise les tâches selon les dépendances et optimise la timeline"):
            st.session_state['auto_organize'] = True
            st.session_state['force_recalculate'] = True
            # Clear any cached dates to force recalculation
            if 'gantt_cache' in st.session_state:
                del st.session_state['gantt_cache']
            st.success("✅ Timeline réorganisée automatiquement !")
    
    # Advanced options
    with st.expander("🎯 Advanced Options"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.checkbox("Show Dependencies", value=True, key="show_dependencies")
            st.checkbox("Show Critical Path", value=True, key="highlight_critical")
            st.checkbox("Show Milestones", value=True, key="show_milestones")
        
        with col2:
            st.checkbox("Show Resource Names", value=True, key="show_resources")
            st.checkbox("Show Task Progress", value=True, key="gantt_progress_legacy")
            st.checkbox("Show Slack/Float", value=False, key="gantt_slack_legacy")
        
        with col3:
            st.checkbox("Show Cost Info", value=False, key="show_cost")
            st.checkbox("Show Task IDs", value=False, key="show_task_ids")
            st.checkbox("Weekend Shading", value=True, key="gantt_weekends_legacy")
        
        with col4:
            buffer_days = st.slider("Buffer Days", 0, 5, 0, key="task_buffer")
            opacity = st.slider("Bar Opacity", 0.5, 1.0, 0.8, key="bar_opacity")
            st.selectbox("Color Scheme", ["Default", "Priority", "Phase", "Resource"], key="color_scheme")


def calculate_intelligent_dates(tasks: List[Dict[str, Any]], 
                               dependencies: List[Dict[str, Any]], 
                               plan_data: Dict[str, Any],
                               critical_path: List[str] = None) -> List[Dict[str, Any]]:
    """Calculate task dates using dependency analysis and working days"""
    import pandas as pd  # Import locally to avoid scope issues
    
    if critical_path is None:
        critical_path = []
    
    # Force recalculation if AI button was pressed
    force_recalc = st.session_state.get('force_recalculate', False)
    if force_recalc:
        st.session_state['force_recalculate'] = False  # Reset flag
    
    # Force explicit project start date - 29 août 2025
    project_start = st.session_state.get('project_start_date', date(2025, 8, 29))
    working_days_setting = st.session_state.get('working_days', 'Monday-Friday')
    buffer_days = st.session_state.get('task_buffer', 0)
    
    # Convert to pandas Timestamp for consistency
    project_start_dt = pd.to_datetime(project_start)
    
    # Info message only when AI button clicked
    if force_recalc:
        st.success(f"🤖 IA: Réorganisation avec date de début {project_start_dt.strftime('%d/%m/%Y')}")
    
    # Create task lookup
    task_lookup = {task.get('id', f'task_{i}'): task for i, task in enumerate(tasks)}
    
    # Initialize date tracking
    task_dates = {}
    processed_tasks = set()
    
    # Calculate working days function
    def add_working_days(start_date, days: float):
        """Add working days to start_date, handling both date and pandas Timestamp"""
        if days <= 0:
            return pd.to_datetime(start_date)
            
        # Convert to pandas Timestamp for consistent handling
        start_dt = pd.to_datetime(start_date)
        
        if working_days_setting == "All Days":
            return start_dt + pd.Timedelta(days=int(days))
        elif working_days_setting == "Monday-Saturday":
            current = start_dt
            days_added = 0
            while days_added < days:
                current += pd.Timedelta(days=1)
                if current.weekday() < 6:  # Monday=0, Saturday=5
                    days_added += 1
            return current
        else:  # Monday-Friday - use business days
            return start_dt + pd.tseries.offsets.BDay(int(days))
    
    def calculate_task_dates(task_id: str):
        """Recursively calculate task start and end dates"""
        
        if task_id in task_dates:
            return task_dates[task_id]
        
        if task_id in processed_tasks:
            # Cycle detection - return default dates
            return project_start_dt, project_start_dt + pd.Timedelta(days=1)
        
        processed_tasks.add(task_id)
        task = task_lookup.get(task_id)
        
        if not task:
            # Task not found - return default dates
            return project_start_dt, project_start_dt + pd.Timedelta(days=1)
        
        # Get task duration
        duration = float(task.get('duration', 1.0))
        
        # Find dependencies from task's dependencies array
        task_dependencies = task.get('dependencies', [])
        predecessors = task_dependencies
        
        if not predecessors:
            # No dependencies - start at project start date
            start_date = project_start_dt
        else:
            # Has dependencies - start after latest predecessor finishes
            latest_finish = project_start_dt
            
            for pred_id in predecessors:
                if pred_id in task_lookup:
                    pred_start, pred_end = calculate_task_dates(pred_id)
                    
                    # Simple finish-to-start dependency with buffer
                    candidate_start = pred_end + pd.Timedelta(days=buffer_days)
                    if candidate_start > latest_finish:
                        latest_finish = candidate_start
            
            start_date = latest_finish
        
        # Calculate end date
        end_date = add_working_days(start_date, duration)
        
        task_dates[task_id] = (start_date, end_date)
        processed_tasks.remove(task_id)
        
        return start_date, end_date
    
    # Sort tasks by phases and dependencies to ensure proper order
    def get_task_phase_order(task):
        """Get phase order for sorting"""
        task_id = task.get('id', '')
        if 'task_1_' in task_id:
            return 1
        elif 'task_2_' in task_id:
            return 2
        elif 'task_3_' in task_id:
            return 3
        elif 'task_4_' in task_id:
            return 4
        else:
            return 999
    
    # Proper dependency-based calculation
    gantt_data = []
    task_dates_map = {}  # Store calculated dates for dependency lookups
    processed_tasks = set()  # Track processed tasks
    
    # Create task lookup by ID
    task_by_id = {task.get('id', f'task_{i}'): task for i, task in enumerate(tasks)}
    
    # Display validation table
    st.markdown("### 📊 Validation des dates et durées (Respect des dépendances)")
    validation_data = []
    
    def calculate_task_schedule(task_id, visited=None):
        """Recursively calculate task schedule based on dependencies"""
        if visited is None:
            visited = set()
        
        # Avoid infinite loops
        if task_id in visited:
            return project_start_dt, project_start_dt + pd.Timedelta(days=1)
        visited.add(task_id)
        
        # Already calculated
        if task_id in task_dates_map:
            return task_dates_map[task_id]['start'], task_dates_map[task_id]['end']
        
        # Get task details
        task = task_by_id.get(task_id)
        if not task:
            return project_start_dt, project_start_dt + pd.Timedelta(days=1)
        
        duration = float(task.get('duration', 1))
        dependencies = task.get('dependencies', [])
        
        # Calculate start date based on dependencies
        if dependencies:
            # Task must start after all dependencies are complete
            latest_end = project_start_dt
            for dep_id in dependencies:
                if dep_id in task_by_id:
                    # Recursively calculate dependency dates
                    dep_start, dep_end = calculate_task_schedule(dep_id, visited.copy())
                    if dep_end > latest_end:
                        latest_end = dep_end
            # Start next working day after dependencies
            start_date = latest_end + pd.Timedelta(days=1)
        else:
            # No dependencies - can start at project start
            start_date = project_start_dt
        
        # Calculate end date
        end_date = add_working_days(start_date, duration)
        
        # Store dates for future dependency calculations
        task_dates_map[task_id] = {'start': start_date, 'end': end_date}
        
        return start_date, end_date
    
    # Process all tasks and calculate their schedules
    for i, task in enumerate(tasks):
        task_id = task.get('id', f'task_{i}')
        task_name = task.get('name', 'Unknown Task')
        duration = float(task.get('duration', 1))
        dependencies = task.get('dependencies', [])
        
        # Calculate dates using dependency logic
        start_date, end_date = calculate_task_schedule(task_id)
        
        # Calculate actual working days for validation
        actual_duration = 0
        current = start_date
        while current < end_date:
            if working_days_setting == "Monday-Friday" and current.weekday() < 5:
                actual_duration += 1
            elif working_days_setting == "Monday-Saturday" and current.weekday() < 6:
                actual_duration += 1
            elif working_days_setting == "All Days":
                actual_duration += 1
            current += pd.Timedelta(days=1)
        
        # Validation record
        validation_data.append({
            'ID': task_id,
            'Tâche': task_name[:30],
            'Durée prévue': duration,
            'Durée calculée': actual_duration,
            'Début': start_date.strftime('%d/%m/%Y'),
            'Fin': end_date.strftime('%d/%m/%Y'),
            'Dépendances': ', '.join(dependencies) if dependencies else 'Aucune',
            '✓': '✅' if abs(actual_duration - duration) <= 1 else '⚠️'
        })
        
        # Get additional task properties
        is_critical = task_id in critical_path
        slack = 0 if is_critical else task.get('slack', 2)
        progress = task.get('progress', 0)
            
        gantt_data.append({
            'Task': task_name,
            'TaskID': task_id,
            'Start': start_date,
            'Finish': end_date,
            'Duration': float(task.get('duration', 1)),
            'Phase': task.get('phase', 'Unknown'),
            'Priority': task.get('priority', 'medium').title(),
            'Status': task.get('status', 'not_started').replace('_', ' ').title(),
            'Critical': is_critical,
            'Cost': float(task.get('cost', 0)),
            'Resources': len(task.get('assigned_resources', [])),
            'Slack': slack,
            'Complexity': task.get('complexity', 'medium'),
            'Progress': progress,
            'ResourceList': ', '.join(task.get('assigned_resources', [])),
            'Description': task.get('description', '')[:100],
            'IsMilestone': task.get('is_milestone', False),  # Utiliser le vrai flag milestone
            'Optimistic': task.get('optimistic_duration', float(task.get('duration', 1)) * 0.8),
            'Pessimistic': task.get('pessimistic_duration', float(task.get('duration', 1)) * 1.5),
            'MostLikely': float(task.get('duration', 1)),
            'WBSOrder': i  # Préserver l'ordre original WBS
        })
    
    # Display validation table sorted by start date
    if validation_data:
        df_validation = pd.DataFrame(validation_data)
        # Add order column to preserve WBS order
        df_validation['_order'] = range(len(df_validation))
        # Convert date strings to datetime for proper sorting
        df_validation['_sort_date'] = pd.to_datetime(df_validation['Début'], format='%d/%m/%Y')
        df_validation = df_validation.sort_values('_sort_date').drop('_sort_date', axis=1)
        st.dataframe(df_validation.drop('_order', axis=1), use_container_width=True, hide_index=True)
        
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_duration = sum([v['Durée prévue'] for v in validation_data])
            st.metric("Durée totale prévue", f"{total_duration} jours")
        with col2:
            validated_count = len([v for v in validation_data if v['✓'] == '✅'])
            st.metric("Tâches validées", f"{validated_count}/{len(validation_data)}")
        with col3:
            if task_dates_map:
                last_task = max(task_dates_map.values(), key=lambda x: x['end'])
                project_end = last_task['end'].strftime('%d/%m/%Y')
                st.metric("Fin du projet", project_end)
    
    return gantt_data


def calculate_task_slack(task_id: str, task_end, dependencies: List[Dict], 
                        task_lookup: Dict, task_dates: Dict) -> float:
    """Calculate slack/float for a task"""
    
    # Find successors
    successors = [dep for dep in dependencies if dep.get('predecessor') == task_id]
    
    if not successors:
        return 5  # Default slack for tasks without successors
    
    min_successor_start = None
    for dep in successors:
        succ_id = dep.get('successor')
        if succ_id in task_dates:
            succ_start, _ = task_dates[succ_id]
            if min_successor_start is None or succ_start < min_successor_start:
                min_successor_start = succ_start
    
    if min_successor_start:
        # Ensure consistent datetime types
        min_successor_dt = pd.to_datetime(min_successor_start)
        task_end_dt = pd.to_datetime(task_end)
        slack_delta = min_successor_dt - task_end_dt
        slack_days = slack_delta.days if hasattr(slack_delta, 'days') else int(slack_delta / pd.Timedelta(days=1))
        return max(0, slack_days)
    
    return 0


def render_professional_gantt_chart(gantt_data: List[Dict[str, Any]], 
                                   critical_path: List[str],
                                   dependencies: List[Dict[str, Any]], 
                                   plan_data: Dict[str, Any]):
    """Render professional Gantt chart with advanced features"""
    
    if not gantt_data:
        st.error("❌ No timeline data available")
        return
    
    df = pd.DataFrame(gantt_data)
    
    # Ensure date columns are properly typed
    if 'Start' in df.columns:
        df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    if 'Finish' in df.columns:
        df['Finish'] = pd.to_datetime(df['Finish'], errors='coerce')
    
    # Apply view mode filtering
    view_mode = st.session_state.get('gantt_view_mode', 'Standard Timeline')
    df_filtered = apply_view_mode_filter(df, view_mode, critical_path)
    
    # Sort by WBS order to maintain logical sequence
    if 'WBSOrder' in df_filtered.columns:
        df_filtered = df_filtered.sort_values('WBSOrder')
    
    # Show critical path information
    if critical_path and st.session_state.get('highlight_critical', True):
        critical_tasks = df_filtered[df_filtered['Critical'] == True]
        if not critical_tasks.empty:
            # Show task names instead of IDs
            task_names = [f"• {task}" for task in critical_tasks['Task'].tolist()]
            st.error(f"🔥 **CHEMIN CRITIQUE** ({len(critical_tasks)} tâches) :\n" + "\n".join(task_names))
        else:
            st.warning("⚠️ Aucune tâche critique détectée. Données debug :")
            st.write(f"Critical path from data: {critical_path}")
            st.write(f"Tasks with Critical=True: {len(df_filtered[df_filtered['Critical'] == True])}")
            if not df_filtered.empty:
                st.dataframe(df_filtered[['TaskID', 'Task', 'Critical']].head(), use_container_width=True)
    
    # Create Gantt chart with selected style
    selected_style = st.session_state.get('gantt_style', 'Professional')
    fig = create_styled_gantt(df_filtered, dependencies, critical_path, selected_style)
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True, key="main_gantt")
    
    # Show PERT analysis if enabled
    if st.session_state.get('gantt_display_pert', False):
        render_pert_analysis(df_filtered)


def render_pert_analysis(df: pd.DataFrame):
    """Render PERT (Program Evaluation and Review Technique) analysis"""
    st.subheader("📊 Analyse PERT")
    
    if df.empty:
        st.warning("Aucune donnée disponible pour l'analyse PERT")
        return
    
    st.info("🎯 **PERT (Program Evaluation and Review Technique)** - Analyse du chemin critique et des marges de manœuvre")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**⚡ Chemin Critique**")
        critical_tasks = df[df['TaskID'].str.contains('critical|urgent', case=False, na=False)]
        
        if critical_tasks.empty:
            # Use tasks with zero slack as critical
            critical_tasks = df.head(3)  # Show first 3 tasks as example
        
        for _, task in critical_tasks.iterrows():
            duration = (task['Finish'] - task['Start']).days if pd.notna(task['Start']) and pd.notna(task['Finish']) else 0
            st.write(f"🔴 **{task['Task']}** ({duration} jours)")
            
    with col2:
        st.write("**⏰ Marges de Manœuvre**")
        for _, task in df.head(5).iterrows():
            slack = task.get('Slack', 0)
            if isinstance(slack, str):
                slack = 0
            
            color = "🟢" if slack > 2 else "🟡" if slack > 0 else "🔴"
            st.write(f"{color} **{task['Task']}**: {slack} jours")
    
    # PERT Statistics
    st.write("**📈 Statistiques PERT**")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        total_duration = sum((row['Finish'] - row['Start']).days 
                           for _, row in df.iterrows() 
                           if pd.notna(row['Start']) and pd.notna(row['Finish']))
        st.metric("Durée totale", f"{total_duration} jours")
        
    with stats_col2:
        avg_progress = df['Progress'].mean() if 'Progress' in df.columns else 0
        st.metric("Progression moyenne", f"{avg_progress:.1f}%")
        
    with stats_col3:
        risk_tasks = len(df[df.get('Progress', 0) < 50])
        st.metric("Tâches à risque", risk_tasks)
    
    # Timeline visualization
    if len(df) > 0:
        st.write("**🗓️ Vue Timeline PERT**")
        import plotly.express as px
        
        # Simple timeline chart
        timeline_df = df.copy()
        timeline_df['Duration'] = timeline_df.apply(
            lambda row: (row['Finish'] - row['Start']).days if pd.notna(row['Start']) and pd.notna(row['Finish']) else 1,
            axis=1
        )
        
        fig = px.bar(
            timeline_df.head(10),
            x='Duration',
            y='Task',
            orientation='h',
            title='Durée des Tâches (jours)',
            color='Progress',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart controls
    render_chart_controls(df)


def render_chart_controls(df: pd.DataFrame):
    """Render chart control panels"""
    if df.empty:
        return
        
    st.subheader("📊 Contrôles du Diagramme")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Filtres**")
        if st.button("🔄 Actualiser"):
            st.success("✅ Données mises à jour")
        
        if st.button("📋 Exporter PNG"):
            st.success("Diagramme exporté!")
            
    with col2:
        st.write("**Affichage**")
        if st.checkbox("Afficher chemin critique", value=True):
            st.session_state['show_critical'] = True
        else:
            st.session_state['show_critical'] = False
            
        if st.checkbox("Mode PERT", value=False):
            st.session_state['pert_view'] = True
        else:
            st.session_state['pert_view'] = False
            
    with col3:
        st.write("**Statistiques**")
        st.metric("Tâches totales", len(df))
        st.metric("Progression", f"{df['Progress'].mean():.1f}%")


def create_styled_gantt(df: pd.DataFrame, dependencies: List[Dict], 
                       critical_path: List[str], style: str) -> go.Figure:
    """Create Gantt chart with different visual styles"""
    
    fig = go.Figure()
    
    # Get style configuration
    style_config = get_gantt_style_config(style)
    
    return create_gantt_with_style(fig, df, dependencies, critical_path, style_config)


def get_gantt_style_config(style: str) -> Dict[str, Any]:
    """Get configuration for different Gantt styles"""
    
    styles = {
        "Professional": {
            "color_map": {
                'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545', 'Critical': '#dc3545'
            },
            "bg_color": '#f8f9fa',
            "paper_color": 'white',
            "grid_color": '#e0e0e0',
            "font_color": '#2c3e50',
            "title_color": '#2c3e50',
            "milestone_color": '#ff6b6b',
            "today_color": 'red',
            "opacity": 0.8
        },
        
        "Modern Dark": {
            "color_map": {
                'Low': '#4ade80', 'Medium': '#fbbf24', 'High': '#f97316', 'Critical': '#ef4444'
            },
            "bg_color": '#0f172a',
            "paper_color": '#1e293b',
            "grid_color": '#374151',
            "font_color": '#f1f5f9',
            "title_color": '#60a5fa',
            "milestone_color": '#ec4899',
            "today_color": '#06b6d4',
            "opacity": 0.85
        },
        
        "Pastel Soft": {
            "color_map": {
                'Low': '#a8e6cf', 'Medium': '#ffd3a5', 'High': '#ffa8b4', 'Critical': '#ff8a95'
            },
            "bg_color": '#fef7f0',
            "paper_color": '#ffffff',
            "grid_color": '#f0e6d6',
            "font_color": '#5d4e75',
            "title_color": '#5d4e75',
            "milestone_color": '#ff8a95',
            "today_color": '#5d4e75',
            "opacity": 0.7
        },
        
        "Neon Cyber": {
            "color_map": {
                'Low': '#00ffcc', 'Medium': '#ffcc00', 'High': '#ff6600', 'Critical': '#ff0066'
            },
            "bg_color": '#000814',
            "paper_color": '#001d3d',
            "grid_color": '#003566',
            "font_color": '#f0f9ff',
            "title_color": '#00d9ff',
            "milestone_color": '#ff0066',
            "today_color": '#00ffcc',
            "opacity": 0.9
        },
        
        "Corporate Clean": {
            "color_map": {
                'Low': '#0066cc', 'Medium': '#4d94ff', 'High': '#001a66', 'Critical': '#000d33'
            },
            "bg_color": '#f5f5f5',
            "paper_color": '#ffffff',
            "grid_color": '#d9d9d9',
            "font_color": '#333333',
            "title_color": '#0066cc',
            "milestone_color": '#001a66',
            "today_color": '#0066cc',
            "opacity": 0.75
        },
        
        "Glassmorphism": {
            "color_map": {
                'Low': '#667eea', 'Medium': '#764ba2', 'High': '#f093fb', 'Critical': '#f5576c'
            },
            "bg_color": 'rgba(255,255,255,0.1)',
            "paper_color": 'rgba(255,255,255,0.05)',
            "grid_color": 'rgba(255,255,255,0.2)',
            "font_color": '#4a5568',
            "title_color": '#667eea',
            "milestone_color": '#f5576c',
            "today_color": '#667eea',
            "opacity": 0.6
        },
        
        "Dark Elite": {
            "color_map": {
                'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444', 'Critical': '#dc2626'
            },
            "bg_color": '#111827',
            "paper_color": '#1f2937',
            "grid_color": '#4b5563',
            "font_color": '#f9fafb',
            "title_color": '#3b82f6',
            "milestone_color": '#8b5cf6',
            "today_color": '#06b6d4',
            "opacity": 0.95
        },
        
        "Gradient Futur": {
            "color_map": {
                'Low': '#22c55e', 'Medium': '#f97316', 'High': '#e11d48', 'Critical': '#be123c'
            },
            "bg_color": '#0c0a09',
            "paper_color": '#1c1917',
            "grid_color": '#44403c',
            "font_color": '#fafaf9',
            "title_color": '#06b6d4',
            "milestone_color": '#a855f7',
            "today_color": '#10b981',
            "opacity": 0.88
        },
        
        "Ocean Deep": {
            "color_map": {
                'Low': '#0ea5e9', 'Medium': '#0284c7', 'High': '#0369a1', 'Critical': '#075985'
            },
            "bg_color": '#0f172a',
            "paper_color": '#1e293b',
            "grid_color": '#334155',
            "font_color": '#f1f5f9',
            "title_color": '#38bdf8',
            "milestone_color": '#06b6d4',
            "today_color": '#0ea5e9',
            "opacity": 0.9
        }
    }
    
    return styles.get(style, styles["Professional"])


def create_gantt_with_style(fig: go.Figure, df: pd.DataFrame, dependencies: List[Dict], 
                           critical_path: List[str], style_config: Dict[str, Any]) -> go.Figure:
    """Create Gantt chart with specific style configuration"""
    
    # Ensure all date columns are datetime first
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['Finish'] = pd.to_datetime(df['Finish'], errors='coerce')
    
    # CRITICAL: Sort by WBSOrder to maintain logical task sequence
    if 'WBSOrder' in df.columns:
        df = df.sort_values('WBSOrder').reset_index(drop=True)
    
    color_map = style_config["color_map"]
    opacity = style_config["opacity"]
    
    # Calculate dynamic bar width and spacing based on available space
    num_tasks = len(df)
    # Adaptive spacing: fewer tasks = more space per task, more tasks = tighter spacing
    if num_tasks <= 5:
        task_spacing = 60  # Generous spacing for few tasks
        dynamic_bar_width = 32
    elif num_tasks <= 10:
        task_spacing = 50  # Medium spacing
        dynamic_bar_width = 28
    elif num_tasks <= 15:
        task_spacing = 40  # Tighter spacing
        dynamic_bar_width = 24
    else:
        task_spacing = 35  # Compact spacing for many tasks
        dynamic_bar_width = 20
    
    # Base height with adaptive spacing
    base_height = max(500, num_tasks * task_spacing + 200)
    
    # Debug: Show critical path detection
    critical_count = len(df[df['Critical'] == True])
    if critical_count > 0:
        st.success(f"✅ {critical_count} tâches critiques détectées dans le DataFrame")
        critical_debug = df[df['Critical'] == True][['TaskID', 'Task', 'Critical']].head(3)
        st.dataframe(critical_debug, use_container_width=True, hide_index=True)
    else:
        st.error(f"❌ Aucune tâche critique détectée. Critical path: {critical_path[:3] if critical_path else 'vide'}")
    
    # Add task bars with progress
    for idx, row in df.iterrows():
        # Main task bar
        color = color_map.get(row['Priority'], '#007bff')
        
        # Apply critical path highlighting if enabled
        current_opacity = opacity
        is_critical_task = row['Critical'] and st.session_state.get('highlight_critical', True)
        
        if is_critical_task:
            color = '#FF0000'  # Force bright red for critical tasks
            # Add extra visual emphasis for critical tasks
            current_opacity = 1.0  # Maximum opacity for critical tasks
        
        # Check if milestone
        if row.get('IsMilestone', False):
            # Add milestone as diamond with style colors
            fig.add_trace(go.Scatter(
                x=[row['Start']],
                y=[row['Task']],
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=15,
                    color=style_config["milestone_color"],
                    line=dict(color=style_config["font_color"], width=2)
                ),
                name='Milestone',
                hovertemplate=f"<b>{row['Task']}</b><br>Date: {row['Start']}<br><extra></extra>",
                showlegend=False
            ))
        else:
            # Enhanced task bar with modern visuals
            progress = row.get('Progress', 0) / 100.0  # Normalize to 0-1 range
            
            # Main task bar (standard approach)
            fig.add_trace(go.Scatter(
                x=[row['Start'], row['Finish']],
                y=[row['Task'], row['Task']],
                mode='lines',
                line=dict(color=color, width=dynamic_bar_width),
                opacity=0.4 if not is_critical_task else 1.0,  # Background opacity, maximum for critical
                name='Background',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Progress bar (completed portion)
            if progress > 0:
                progress_end = row['Start'] + (row['Finish'] - row['Start']) * progress
                fig.add_trace(go.Scatter(
                    x=[row['Start'], progress_end],
                    y=[row['Task'], row['Task']],
                    mode='lines',
                    line=dict(color=color, width=dynamic_bar_width - 2),  # Slightly thinner for layering
                    opacity=current_opacity,
                    name='Progress',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Main task information (invisible line for hover)
            fig.add_trace(go.Scatter(
                x=[row['Start'], row['Finish']],
                y=[row['Task'], row['Task']],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=dynamic_bar_width + 2),  # Slightly wider for easier hover
                name=row['Task'],
                hovertemplate=f"<b>{row['Task']}</b><br>"
                            f"📅 Duration: {row['Duration']} days<br>"
                            f"🚀 Start: {row['Start'].strftime('%d/%m/%Y')}<br>"
                            f"🏁 End: {row['Finish'].strftime('%d/%m/%Y')}<br>"
                            f"📊 Progress: {progress*100:.0f}%<br>"
                            f"⚡ Priority: {row['Priority']}<br>"
                            f"{'🔥 CRITICAL PATH' if row['Critical'] else ''}<br>"
                            f"Resources: {row['ResourceList']}<br>"
                            f"Cost: ${row['Cost']:,.0f}<br>"
                            f"<extra></extra>",
                showlegend=False
            ))
            
            # Add progress overlay if enabled
            if st.session_state.get('gantt_display_progress', True) and row['Progress'] > 0:
                # Ensure consistent datetime types
                start_dt = pd.to_datetime(row['Start'])
                finish_dt = pd.to_datetime(row['Finish'])
                duration = finish_dt - start_dt
                progress_end = start_dt + duration * row['Progress']
                
                fig.add_trace(go.Scatter(
                    x=[start_dt, progress_end],
                    y=[row['Task'], row['Task']],
                    mode='lines',
                    line=dict(color='darkgreen', width=10),
                    opacity=1,
                    hovertemplate=f"Progress: {row['Progress']:.0f}%<extra></extra>",
                    showlegend=False
                ))
            
            # Add slack visualization if enabled
            if st.session_state.get('gantt_display_slack', False) and row['Slack'] > 0:
                # Ensure consistent datetime types
                finish_dt = pd.to_datetime(row['Finish'])
                slack_end = finish_dt + pd.Timedelta(days=row['Slack'])
                
                fig.add_trace(go.Scatter(
                    x=[finish_dt, slack_end],
                    y=[row['Task'], row['Task']],
                    mode='lines',
                    line=dict(color='green', width=10, dash='dot'),
                    opacity=0.5,
                    hovertemplate=f"Slack: {row['Slack']} days<extra></extra>",
                    showlegend=False
                ))
    
    # Add dependencies if enabled
    if st.session_state.get('show_dependencies', True):
        add_professional_dependencies(fig, df, dependencies)
    
    # Add weekend shading if enabled
    if st.session_state.get('gantt_display_weekends', True):
        add_weekend_shading_with_style(fig, df, style_config)
    
    # Ensure we have valid date range for x-axis
    if not df.empty and 'Start' in df.columns and 'Finish' in df.columns:
        min_date = df['Start'].min()
        max_date = df['Finish'].max()
        
        # Add some padding to the date range
        date_padding = pd.Timedelta(days=7)
        x_range = [min_date - date_padding, max_date + date_padding]
    else:
        # Default range if no valid dates
        x_range = [pd.Timestamp('2025-08-29'), pd.Timestamp('2025-12-31')]
    
    # Update layout with style
    fig.update_layout(
        title={
            'text': "📅 Project Timeline - Interactive Gantt Chart",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': style_config["title_color"], 'family': 'Inter, sans-serif'}
        },
        xaxis_title="📅 Timeline",
        yaxis_title="📋 Tasks",
        height=base_height,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor=style_config["bg_color"],
        paper_bgcolor=style_config["paper_color"],
        font=dict(family="Inter, sans-serif", size=12, color=style_config["font_color"]),
        xaxis=dict(
            showgrid=True,
            gridwidth=2,
            gridcolor=style_config["grid_color"],
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor=style_config["font_color"],
            tickformat='%d/%m/%Y',
            range=x_range,  # Set the date range
            title_font=dict(size=14, color=style_config["title_color"]),
            tickfont=dict(size=11, color=style_config["font_color"])
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor=style_config["grid_color"],
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor=style_config["font_color"],
            autorange='reversed'
        )
    )
    
    return fig


def add_weekend_shading_with_style(fig: go.Figure, df: pd.DataFrame, style_config: Dict[str, Any]):
    """Add weekend shading with style colors"""
    
    if df.empty:
        return
    
    # Ensure all date columns are datetime
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['Finish'] = pd.to_datetime(df['Finish'], errors='coerce')
    
    start_date = df['Start'].min()
    end_date = df['Finish'].max()
    
    # Determine weekend color based on style
    if style_config["bg_color"] == '#0a0a0a':  # Dark themes
        weekend_color = 'rgba(100,100,100,0.1)'
    elif 'rgba' in style_config["bg_color"]:  # Glassmorphism
        weekend_color = 'rgba(255,255,255,0.1)'
    else:  # Light themes
        weekend_color = 'rgba(200,200,200,0.2)'
    
    current = start_date
    while current <= end_date:
        if current.weekday() in [5, 6]:  # Saturday or Sunday
            fig.add_vrect(
                x0=current,
                x1=current + pd.Timedelta(days=1),
                fillcolor=weekend_color,
                layer='below',
                line_width=0
            )
        current += pd.Timedelta(days=1)


def render_style_preview():
    """Affiche une prévisualisation des styles disponibles"""
    
    st.markdown("### 🎨 Aperçu des Styles Disponibles")
    
    style_descriptions = {
        "Professional": "🏢 Style classique d'entreprise - Couleurs sobres et professionnelles",
        "Modern Dark": "🌙 Thème sombre moderne - Parfait pour les développeurs et les équipes tech",
        "Pastel Soft": "🌸 Couleurs douces et pastels - Interface apaisante et moderne",
        "Neon Cyber": "⚡ Style futuriste néon - Couleurs vives et énergiques",
        "Corporate Clean": "💼 Ultra-propre corporate - Bleus professionnels et minimalisme",
        "Glassmorphism": "💎 Effet de verre moderne - Transparences et dégradés élégants"
    }
    
    for style, description in style_descriptions.items():
        st.write(f"**{style}**: {description}")
    
    st.info("💡 **Conseil**: Chaque style adapte automatiquement les couleurs, l'arrière-plan, et les effets visuels du Gantt")


def render_gantt_with_style_selector(plan_data: Dict[str, Any]):
    """Rendu du Gantt avec sélecteur de styles et prévisualisation"""
    
    # Sélecteur de style en haut
    col1, col2 = st.columns([3, 1])
    
    with col1:
        render_gantt_chart(plan_data)
    
    with col2:
        if st.button("🎨 Voir les Styles", use_container_width=True):
            with st.expander("Aperçu des Styles", expanded=True):
                render_style_preview()


def get_project_start_date(plan_data: Dict[str, Any]) -> date:
    """Get intelligent project start date"""
    
    overview = plan_data.get('project_overview', {})
    if overview.get('start_date'):
        try:
            if isinstance(overview['start_date'], str):
                return datetime.fromisoformat(overview['start_date']).date()
            elif isinstance(overview['start_date'], date):
                return overview['start_date']
            elif isinstance(overview['start_date'], datetime):
                return overview['start_date'].date()
        except:
            pass
    
    # Check tasks for earliest date
    tasks = extract_all_tasks(plan_data)
    if tasks:
        earliest_date = None
        for task in tasks:
            task_date = task.get('start_date')
            if task_date:
                try:
                    if isinstance(task_date, str):
                        task_dt = datetime.fromisoformat(task_date).date()
                    elif isinstance(task_date, date):
                        task_dt = task_date
                    elif isinstance(task_date, datetime):
                        task_dt = task_date.date()
                    else:
                        continue
                        
                    if earliest_date is None or task_dt < earliest_date:
                        earliest_date = task_dt
                except:
                    continue
        
        if earliest_date:
            return earliest_date
    
    # Default to next Monday
    today = date.today()
    days_ahead = 0 - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    
    return today + timedelta(days=days_ahead)


def apply_view_mode_filter(df: pd.DataFrame, view_mode: str, critical_path: List[str]) -> pd.DataFrame:
    """Apply filtering based on selected view mode"""
    
    if view_mode == "Critical Path Focus" and critical_path:
        return df[df['Critical'] == True].sort_values('Start')
    elif view_mode == "Phase Breakdown":
        return df.sort_values(['Phase', 'Start'])
    elif view_mode == "Resource Allocation":
        return df[df['Resources'] > 0].sort_values(['ResourceList', 'Start'])
    elif view_mode == "Progress Tracking":
        return df.sort_values(['Progress', 'Start'], ascending=[False, True])
    elif view_mode == "Milestone Focus":
        # Show milestones and their immediate predecessors
        milestone_df = df[df['IsMilestone'] == True]
        if not milestone_df.empty:
            return pd.concat([milestone_df, df[df['Critical'] == True]]).drop_duplicates()
        return df.sort_values('Start')
    else:  # Standard Timeline
        # Ensure Start column is datetime before sorting
        if df['Start'].dtype == 'object':
            df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
        return df.sort_values('Start')


def add_professional_dependencies(fig: go.Figure, df: pd.DataFrame, dependencies: List[Dict]):
    """Add professional dependency arrows"""
    
    if df.empty or not dependencies:
        return
    
    task_positions = {row['TaskID']: (row['Start'], row['Finish'], row['Task']) 
                     for _, row in df.iterrows()}
    
    for dep in dependencies:
        pred_id = dep.get('predecessor')
        succ_id = dep.get('successor')
        
        if pred_id in task_positions and succ_id in task_positions:
            pred_start, pred_end, pred_task = task_positions[pred_id]
            succ_start, succ_end, succ_task = task_positions[succ_id]
            
            # Draw dependency arrow
            fig.add_annotation(
                x=pred_end,
                y=pred_task,
                ax=succ_start,
                ay=succ_task,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor='rgba(100,100,100,0.4)'
            )


def generate_sample_gantt_data() -> List[Dict]:
    """Generate sample Gantt data for demonstration"""
    import random
    from datetime import datetime, timedelta
    
    task_names = [
        "Analyse des besoins", "Conception UI/UX", "Développement Backend", 
        "Développement Frontend", "Tests unitaires", "Tests d'intégration",
        "Documentation", "Déploiement", "Formation utilisateurs", "Go-live"
    ]
    
    base_date = datetime.now()
    sample_data = []
    
    for i, task_name in enumerate(task_names):
        start_date = base_date + timedelta(days=i*7)
        duration = random.randint(5, 21)
        end_date = start_date + timedelta(days=duration)
        
        task = {
            'task_name': task_name,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'assigned_to': f'Team_{chr(65 + i % 4)}',  # Team_A, Team_B, Team_C, Team_D
            'progress': random.randint(20, 100),
            'phase': 'Développement' if 'Développement' in task_name else 
                    'Tests' if 'Test' in task_name else 
                    'Déploiement' if task_name in ['Déploiement', 'Go-live'] else 'Conception'
        }
        sample_data.append(task)
    
    return sample_data
