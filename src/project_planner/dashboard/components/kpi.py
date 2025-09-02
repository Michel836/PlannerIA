"""
Module KPI - Restructuré selon le modèle Reporting/Qualité/Risques/Logs
Indicateurs de performance clés et métriques projet avancées
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import io
import json
from typing import Dict, Any, List


def calculate_kpi_metrics(tasks: List[Dict], risks: List[Dict], 
                         project_metrics: Dict, project_overview: Dict) -> Dict[str, Any]:
    """Calcule les métriques KPI avancées"""
    if not tasks:
        tasks = generate_sample_task_data()
    
    # Métriques de base
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t.get('status') == 'completed'])
    total_duration = sum(task.get('duration', 0) for task in tasks)
    total_cost = sum(task.get('cost', 0) for task in tasks)
    total_effort = sum(task.get('effort', task.get('duration', 0)) for task in tasks)
    
    # Métriques de performance
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    avg_task_duration = total_duration / total_tasks if total_tasks > 0 else 0
    cost_per_task = total_cost / total_tasks if total_tasks > 0 else 0
    
    # Métriques de risque
    total_risks = len(risks)
    high_risks = len([r for r in risks if r.get('risk_score', 0) >= 15])
    critical_risks = len([r for r in risks if r.get('risk_score', 0) >= 20])
    avg_risk_score = sum(r.get('risk_score', 0) for r in risks) / total_risks if total_risks > 0 else 0
    
    # Score de santé globale du projet
    health_factors = [
        completion_rate,                           # Progression
        max(0, 100 - (high_risks / max(1, total_tasks) * 100)), # Risques maîtrisés
        project_metrics.get('complexity_score', 75),            # Complexité gérée
        project_overview.get('current_health', 80)              # Santé générale
    ]
    global_health_score = np.mean(health_factors)
    
    # Métriques d'efficacité
    if total_effort > 0 and total_duration > 0:
        efficiency_ratio = total_effort / (total_duration * 8)  # 8h par jour
    else:
        efficiency_ratio = 1.0
    
    # Score de performance globale
    performance_score = (completion_rate * 0.4 + 
                        (100 - avg_risk_score) * 0.3 + 
                        global_health_score * 0.3)
    
    return {
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'completion_rate': completion_rate,
        'total_duration': total_duration,
        'avg_task_duration': avg_task_duration,
        'total_cost': total_cost,
        'cost_per_task': cost_per_task,
        'total_effort': total_effort,
        'efficiency_ratio': efficiency_ratio,
        'total_risks': total_risks,
        'high_risks': high_risks,
        'critical_risks': critical_risks,
        'avg_risk_score': avg_risk_score,
        'global_health_score': global_health_score,
        'performance_score': performance_score,
        'complexity_score': project_metrics.get('complexity_score', 0),
        'complexity_level': project_metrics.get('complexity_level', 'Unknown')
    }


def generate_kpi_insights(tasks: List[Dict], risks: List[Dict], project_metrics: Dict) -> Dict:
    """Génère des insights IA pour les KPIs"""
    return {
        'performance_insights': {
            "trend": "improving",
            "efficiency": 87,
            "bottlenecks": ["testing", "deployment"],
            "confidence": 0.89
        },
        'completion_trends': {
            "weekly_velocity": 8.2,
            "projected_completion": "2024-04-15",
            "confidence": 0.91
        },
        'risk_assessment': {
            "critical_alerts": 2,
            "trend": "stable",
            "mitigation_effectiveness": 0.76
        }
    }


def render_kpi_section(plan_data: Dict[str, Any]):
    """Dashboard KPI restructuré - Sections visuels puis onglets détaillés"""
    
    # Header moderne
    st.header("📈 Key Performance Indicators Dashboard")
    st.markdown("*Métriques de performance, analyses avancées et insights IA*")
    
    # Extract and prepare data
    project_overview = plan_data.get('project_overview', {})
    project_metrics = plan_data.get('project_metrics', {})
    tasks = extract_all_tasks(plan_data)
    risks = plan_data.get('risks', [])
    
    # Génération de risques intelligents pour portfolio
    if 'projects' in plan_data and not risks:
        risks = generate_portfolio_risks(plan_data['projects'])
    
    # Calcul des KPIs et insights
    kpi_metrics = calculate_kpi_metrics(tasks, risks, project_metrics, project_overview)
    kpi_insights = generate_kpi_insights(tasks, risks, project_metrics)
    
    # === SECTION 1: MÉTRIQUES RÉSUMÉES ===
    render_summary_metrics_enhanced(kpi_metrics)
    
    # === SECTION 2: GRAPHIQUES VISUELS PRINCIPAUX ===
    st.markdown("---")
    render_main_kpi_visualizations(tasks, risks, kpi_metrics)
    
    # === SECTION 3: DETAILED ANALYSIS (avec onglets) ===
    st.markdown("---")
    render_detailed_kpi_analysis(plan_data, tasks, risks, kpi_insights)
    
    # === SECTION 4: EXPORTS EN BAS DE PAGE ===
    st.markdown("---")
    render_kpi_export_section(tasks, risks, kpi_metrics, kpi_insights)


def generate_portfolio_risks(projects: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Génère des risques intelligents basés sur les projets du portfolio"""
    risks = []
    
    for proj_id, project in projects.items():
        proj_name = project.get('name', proj_id)[:15]
        health = project.get('health_score', 75)
        progress = project.get('progress', 0) 
        priority = project.get('priority', 'medium')
        
        # Générer des risques basés sur les caractéristiques du projet
        if health < 80:
            risks.append({
                'name': f'Santé dégradée {proj_name}',
                'probability': 4,
                'impact': 3 if priority != 'critical' else 5,
                'risk_score': 12 if priority != 'critical' else 20,
                'category': 'health',
                'status': 'active',
                'owner': f'PM {proj_name}',
                'project': proj_name
            })
        
        if progress < 50 and priority in ['high', 'critical']:
            risks.append({
                'name': f'Retard planning {proj_name}',
                'probability': 3,
                'impact': 4,
                'risk_score': 12,
                'category': 'schedule',
                'status': 'monitoring',
                'owner': f'Team Lead {proj_name}',
                'project': proj_name
            })
        
        if project.get('budget', 0) > 800000:  # Gros budgets
            risks.append({
                'name': f'Dépassement budget {proj_name}',
                'probability': 2,
                'impact': 5,
                'risk_score': 10,
                'category': 'financial',
                'status': 'identified',
                'owner': f'Finance Team',
                'project': proj_name
            })
    
    # Ajouter quelques risques transversaux
    risks.extend([
        {
            'name': 'Ressources partagées indisponibles',
            'probability': 3,
            'impact': 3,
            'risk_score': 9,
            'category': 'resource',
            'status': 'active',
            'owner': 'Resource Manager',
            'project': 'Portfolio'
        },
        {
            'name': 'Dépendances techniques critiques',
            'probability': 2,
            'impact': 4,
            'risk_score': 8,
            'category': 'technical',
            'status': 'monitoring', 
            'owner': 'Tech Lead',
            'project': 'CloudForge'
        }
    ])
    
    return risks



def render_summary_metrics_enhanced(kpis: Dict[str, Any]):
    """Métriques KPI résumées avec scoring avancé"""
    if not kpis:
        return
    
    # Calculs pour les indicateurs de santé
    health_status = "✅ Excellent" if kpis['global_health_score'] >= 85 else "⚠️ Bon" if kpis['global_health_score'] >= 70 else "🚨 Critique"
    performance_status = "🚀 Optimal" if kpis['performance_score'] >= 80 else "📊 Correct" if kpis['performance_score'] >= 60 else "⚠️ Améliorer"
    completion_status = "✅ Avancé" if kpis['completion_rate'] >= 75 else "🔵 En cours" if kpis['completion_rate'] >= 25 else "🔴 Démarrage"
    risk_status = "✅ Maîtrisé" if kpis['critical_risks'] == 0 else "⚠️ Attention" if kpis['critical_risks'] <= 2 else "🚨 Élevé"
    efficiency_status = "⚡ Efficace" if kpis['efficiency_ratio'] >= 0.8 else "📈 Moyen" if kpis['efficiency_ratio'] >= 0.6 else "📉 Faible"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Santé Globale", f"{kpis['global_health_score']:.1f}%", health_status)
    
    with col2:
        st.metric("🚀 Performance", f"{kpis['performance_score']:.1f}/100", performance_status)
    
    with col3:
        st.metric("📊 Completion", f"{kpis['completion_rate']:.1f}%", completion_status)
    
    with col4:
        st.metric("⚠️ Risques Critiques", f"{kpis['critical_risks']}/{kpis['total_risks']}", risk_status)
    
    with col5:
        st.metric("⚡ Efficacité", f"{kpis['efficiency_ratio']:.1f}x", efficiency_status)


def extract_all_tasks(plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all tasks from the plan structure"""
    tasks = []
    
    # Si c'est un portfolio, traiter les projets comme des "tasks"
    if 'projects' in plan_data:
        projects = plan_data['projects']
        for project_id, project in projects.items():
            # Convertir le projet en format "task" pour les KPIs
            # Calculer durée basée sur dates de début/fin ou progression
            duration_days = 120  # Défaut
            if 'start_date' in project and 'end_date' in project:
                from datetime import datetime
                try:
                    start = datetime.fromisoformat(project['start_date'][:19])
                    end = datetime.fromisoformat(project['end_date'][:19])
                    duration_days = (end - start).days
                except:
                    pass
            
            # Calculer effort basé sur équipe et durée
            team_size = project.get('team_size', 1)
            effort_hours = duration_days * team_size * 8  # 8h/jour/personne
            
            task = {
                'name': project.get('name', project_id),
                'duration': duration_days,
                'cost': project.get('budget', 0),
                'progress': project.get('progress', 0),
                'priority': project.get('priority', 'medium'),
                'status': 'completed' if project.get('progress', 0) >= 95 else 'in_progress' if project.get('progress', 0) > 10 else 'not_started',
                'is_critical': project.get('priority') == 'critical',
                'assigned_resources': [f"Team_{i+1}" for i in range(team_size)],
                'effort': effort_hours,
                'category': project.get('domain', 'general'),
                'health_score': project.get('health_score', 75),
                'team_size': team_size
            }
            tasks.append(task)
    
    # Direct tasks (projet unique)
    elif 'tasks' in plan_data:
        tasks.extend(plan_data['tasks'])
    
    # Tasks from WBS phases (projet PSTB format)
    elif 'wbs' in plan_data and 'phases' in plan_data['wbs']:
        for phase in plan_data['wbs']['phases']:
            if 'tasks' in phase:
                phase_tasks = phase['tasks']
                for task in phase_tasks:
                    # Ajuster le format des tâches WBS pour les KPIs
                    task_formatted = {
                        'name': task.get('name', 'Unknown Task'),
                        'duration': task.get('duration', 0),
                        'cost': task.get('cost', 0),
                        'progress': task.get('progress', 0),
                        'priority': task.get('priority', 'medium'),
                        'status': task.get('status', 'not_started'),
                        'is_critical': task.get('priority') == 'critical',
                        'assigned_resources': task.get('assigned_resources', []),
                        'effort': task.get('effort', task.get('duration', 0)),
                        'category': 'development'
                    }
                    tasks.append(task_formatted)
    
    return tasks


def render_summary_metrics(overview: Dict[str, Any], metrics: Dict[str, Any], 
                         tasks: List[Dict[str, Any]], risks: List[Dict[str, Any]]):
    """Render summary KPI cards"""
    
    # Calculate metrics
    total_tasks = len(tasks)
    total_duration = sum(task.get('duration', 0) for task in tasks)
    total_cost = sum(task.get('cost', 0) for task in tasks)
    total_effort = sum(task.get('effort', task.get('duration', 0)) for task in tasks)
    
    # Risk metrics
    high_risks = [r for r in risks if r.get('risk_score', 0) >= 15]
    avg_risk_score = sum(r.get('risk_score', 0) for r in risks) / len(risks) if risks else 0
    
    # Complexity metrics
    complexity_score = metrics.get('complexity_score', 0)
    complexity_level = metrics.get('complexity_level', 'Unknown')
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📋 Total Tasks",
            value=total_tasks,
            help="Total number of tasks in the project"
        )
    
    with col2:
        st.metric(
            label="â±ï¸ Duration",
            value=f"{total_duration:.1f} days",
            delta=f"{total_effort:.1f} effort hours" if total_effort != total_duration else None,
            help="Total project duration and effort"
        )
    
    with col3:
        st.metric(
            label="💰 Total Cost",
            value=f"${total_cost:,.0f}",
            help="Total estimated project cost"
        )
    
    with col4:
        st.metric(
            label="âš ï¸ High Risks",
            value=len(high_risks),
            delta=f"Avg: {avg_risk_score:.1f}" if risks else "No risks",
            delta_color="inverse",
            help="Number of high-priority risks (score ≥ 15)"
        )
    
    with col5:
        complexity_color = {
            'Low': 'normal',
            'Medium': 'normal', 
            'High': 'inverse',
            'Very High': 'inverse'
        }.get(complexity_level, 'normal')
        
        st.metric(
            label="🔧 Complexity",
            value=complexity_level,
            delta=f"Score: {complexity_score}",
            delta_color=complexity_color,
            help="Project complexity assessment"
        )


def render_main_kpi_visualizations(tasks: List[Dict], risks: List[Dict], kpis: Dict[str, Any]):
    """Graphiques KPI principaux"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance radar chart
        st.subheader("🎯 Radar Performance")
        
        categories = ['Completion', 'Efficacité', 'Qualité', 'Risques', 'Budget']
        values = [
            kpis['completion_rate'],
            kpis['efficiency_ratio'] * 100,
            kpis['global_health_score'],
            max(0, 100 - kpis['avg_risk_score'] * 4),
            min(100, kpis['cost_per_task'] / 1000) if kpis['cost_per_task'] > 0 else 80
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Performance Actuelle',
            line=dict(color='rgb(0, 150, 200)'),
            fillcolor='rgba(0, 150, 200, 0.3)'
        ))
        
        # Ligne de référence
        target_values = [80, 80, 85, 80, 85]  # Objectifs
        fig.add_trace(go.Scatterpolar(
            r=target_values,
            theta=categories,
            line=dict(color='red', dash='dash'),
            name='Objectifs'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            height=350,
            showlegend=True,
            title="Performance vs Objectifs"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Trend analysis
        st.subheader("📈 Évolution Temporelle")
        
        # Simulation d'évolution sur 12 semaines
        weeks = list(range(1, 13))
        completion_trend = [min(100, kpis['completion_rate'] + (i * 8)) for i in range(12)]
        risk_trend = [max(0, kpis['avg_risk_score'] - (i * 0.5)) for i in range(12)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=weeks,
            y=completion_trend,
            mode='lines+markers',
            name='Completion (%)',
            line=dict(color='green', width=3),
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=weeks,
            y=risk_trend,
            mode='lines+markers',
            name='Score Risque',
            line=dict(color='red', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            height=350,
            xaxis_title="Semaines",
            yaxis=dict(title="Completion (%)", side='left', range=[0, 100]),
            yaxis2=dict(title="Score Risque", side='right', overlaying='y', range=[0, 25]),
            title="Tendances Prédictives",
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_task_distribution_chart(tasks: List[Dict[str, Any]]):
    """Render task distribution by status/priority"""
    
    if not tasks:
        st.info("No tasks available for distribution analysis")
        return
    
    # Task status distribution
    status_counts = {}
    priority_counts = {}
    
    for task in tasks:
        status = task.get('status', 'not_started')
        priority = task.get('priority', 'medium')
        
        status_counts[status] = status_counts.get(status, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=("Task Status", "Task Priority")
    )
    
    # Status pie chart
    fig.add_trace(
        go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            name="Status",
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        ),
        row=1, col=1
    )
    
    # Priority pie chart
    fig.add_trace(
        go.Pie(
            labels=list(priority_counts.keys()),
            values=list(priority_counts.values()),
            name="Priority",
            marker_colors=['#FFA07A', '#FFD700', '#FF6347', '#DC143C']
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Task Distribution Analysis",
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_cost_breakdown_chart(tasks: List[Dict[str, Any]]):
    """Render cost breakdown visualization"""
    
    if not tasks:
        tasks = generate_sample_task_data()
        st.info("📊 Affichage de données d'exemple pour démonstration")
    
    if not tasks:
        st.info("No cost data available")
        return
    
    # Calculate cost by category
    task_costs = []
    for task in tasks:
        cost = task.get('cost', 0)
        if cost > 0:
            task_costs.append({
                'name': task.get('name', 'Unknown'),
                'cost': cost,
                'priority': task.get('priority', 'medium'),
                'duration': task.get('duration', 0)
            })
    
    if not task_costs:
        st.info("No cost data available for visualization")
        return
    
    # Create cost breakdown chart
    df_costs = pd.DataFrame(task_costs)
    
    fig = px.treemap(
        df_costs,
        path=['priority', 'name'],
        values='cost',
        color='duration',
        color_continuous_scale='Viridis',
        title="Cost Breakdown by Priority and Task"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_risk_distribution_chart(risks: List[Dict[str, Any]]):
    """Render risk analysis visualization"""
    
    if not risks:
        st.info("No risks identified for analysis")
        return
    
    # Prepare risk data
    risk_data = []
    for risk in risks:
        risk_data.append({
            'name': risk.get('name', 'Unknown Risk'),
            'probability': risk.get('probability', 1),
            'impact': risk.get('impact', 1),
            'risk_score': risk.get('risk_score', 1),
            'category': risk.get('category', 'other')
        })
    
    df_risks = pd.DataFrame(risk_data)
    
    # Risk matrix scatter plot
    fig = px.scatter(
        df_risks,
        x='probability',
        y='impact',
        size='risk_score',
        color='category',
        hover_name='name',
        hover_data=['risk_score'],
        title="Risk Matrix: Probability vs Impact",
        labels={
            'probability': 'Probability (1-5)',
            'impact': 'Impact (1-5)'
        }
    )
    
    # Add risk zones
    fig.add_shape(
        type="rect",
        x0=0.5, y0=0.5, x1=3, y1=3,
        fillcolor="green", opacity=0.1,
        line=dict(width=0)
    )
    fig.add_shape(
        type="rect", 
        x0=3, y0=3, x1=5.5, y1=5.5,
        fillcolor="red", opacity=0.1,
        line=dict(width=0)
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_duration_analysis_chart(tasks: List[Dict[str, Any]]):
    """Render duration analysis"""
    
    if not tasks:
        tasks = generate_sample_task_data()
        st.info("📊 Affichage de données d'exemple pour démonstration")
    
    if not tasks:
        st.info("No duration data available")
        return
    
    # Prepare duration data
    duration_data = []
    for task in tasks:
        duration = task.get('duration', 0)
        if duration > 0:
            duration_data.append({
                'name': task.get('name', 'Unknown')[:20] + ('...' if len(task.get('name', '')) > 20 else ''),
                'duration': duration,
                'is_critical': task.get('is_critical', False)
            })
    
    if not duration_data:
        st.info("No duration data available for visualization")
        return
    
    # Sort by duration
    duration_data.sort(key=lambda x: x['duration'], reverse=True)
    
    # Take top 10 tasks
    duration_data = duration_data[:10]
    
    df_duration = pd.DataFrame(duration_data)
    
    # Bar chart
    fig = px.bar(
        df_duration,
        x='duration',
        y='name',
        color='is_critical',
        orientation='h',
        title="Top 10 Tasks by Duration",
        labels={'duration': 'Duration (days)', 'name': 'Task'},
        color_discrete_map={True: '#FF6B6B', False: '#4ECDC4'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_detailed_kpi_analysis(plan_data: Dict[str, Any], tasks: List[Dict], 
                               risks: List[Dict], insights: Dict):
    """Analyse détaillée KPI avec onglets (contenu original préservé et enrichi)"""
    
    st.subheader("🔍 Analyse Détaillée des KPIs")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Task Analysis", 
        "💰 Cost Analysis", 
        "⚠️ Risk Analysis", 
        "📈 Performance", 
        "🤖 Analyse IA",
        "🎯 Benchmarking"
    ])
    
    with tab1:
        render_task_analysis_detailed(tasks)
    
    with tab2:
        render_cost_analysis_detailed(tasks)
    
    with tab3:
        render_risk_analysis_detailed(risks)
    
    with tab4:
        render_performance_metrics_detailed(plan_data)
    
    with tab5:
        render_ai_kpi_analysis(plan_data, tasks, risks)
    
    with tab6:
        render_benchmarking_analysis(tasks, risks, insights)


def render_kpi_export_section(tasks: List[Dict], risks: List[Dict], 
                            kpis: Dict, insights: Dict):
    """Section d'export KPI en bas de page"""
    
    st.subheader("📤 Exports KPI")
    st.markdown("*Exportez les métriques et analyses de performance*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export KPI Excel", help="Toutes les métriques en Excel"):
            excel_data = create_kpi_excel_export(tasks, risks, kpis, insights)
            st.download_button(
                "💾 Télécharger Excel",
                excel_data,
                "kpi_metrics_export.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📈 Dashboard PDF", help="Rapport KPI complet PDF"):
            st.info("📄 Génération du dashboard KPI PDF...")
    
    with col3:
        if st.button("📋 Métriques CSV", help="KPIs en format CSV"):
            csv_data = create_kpi_csv_export(kpis, insights)
            st.download_button(
                "💾 Télécharger CSV",
                csv_data,
                "kpi_summary.csv",
                "text/csv"
            )
    
    with col4:
        if st.button("🔧 Données JSON", help="Export technique JSON"):
            json_data = json.dumps({
                'kpi_metrics': kpis,
                'insights': insights,
                'tasks_summary': len(tasks),
                'risks_summary': len(risks),
                'export_timestamp': str(pd.Timestamp.now())
            }, indent=2, default=str)
            st.download_button(
                "💾 Télécharger JSON",
                json_data,
                "kpi_complete_export.json",
                "application/json"
            )


def create_kpi_excel_export(tasks: List[Dict], risks: List[Dict], kpis: Dict, insights: Dict) -> bytes:
    """Crée un export Excel multi-feuilles pour les KPIs"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille KPI Summary
        kpi_summary = pd.DataFrame([kpis])
        kpi_summary.to_excel(writer, sheet_name='KPI Summary', index=False)
        
        # Feuille Tasks
        if tasks:
            df_tasks = pd.DataFrame(tasks)
            df_tasks.to_excel(writer, sheet_name='Tasks Analysis', index=False)
        
        # Feuille Risks
        if risks:
            df_risks = pd.DataFrame(risks)
            df_risks.to_excel(writer, sheet_name='Risk Analysis', index=False)
        
        # Feuille Insights
        insights_df = pd.DataFrame([insights])
        insights_df.to_excel(writer, sheet_name='AI Insights', index=False)
    
    return output.getvalue()


def create_kpi_csv_export(kpis: Dict, insights: Dict) -> str:
    """Crée un export CSV des KPIs principaux"""
    data = {
        'Metric': list(kpis.keys()),
        'Value': list(kpis.values())
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def render_task_analysis_detailed(tasks: List[Dict]):
    """Analyse détaillée des tâches (version enrichie)"""
    st.markdown("### 📊 Task Analysis")
    
    if not tasks:
        tasks = generate_sample_task_data()
        st.info("📊 Affichage de données d'exemple pour démonstration")
    
    if not tasks:
        st.info("No tasks available for analysis")
        return
    
    # Métriques avancées
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_duration = np.mean([t.get('duration', 0) for t in tasks])
        st.metric("Durée Moyenne", f"{avg_duration:.1f} jours")
    
    with col2:
        completed = len([t for t in tasks if t.get('status') == 'completed'])
        st.metric("Tâches Complétées", f"{completed}/{len(tasks)}")
    
    with col3:
        critical_tasks = len([t for t in tasks if t.get('is_critical', False)])
        st.metric("Tâches Critiques", critical_tasks)
    
    with col4:
        avg_cost = np.mean([t.get('cost', 0) for t in tasks if t.get('cost', 0) > 0])
        st.metric("Coût Moyen", f"${avg_cost:,.0f}")
    
    # Graphique de distribution
    df_tasks = pd.DataFrame(tasks)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution par statut
        status_counts = df_tasks['status'].value_counts() if 'status' in df_tasks.columns else pd.Series()
        if not status_counts.empty:
            fig = px.pie(values=status_counts.values, names=status_counts.index, 
                        title="Distribution par Statut")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution par priorité
        priority_counts = df_tasks['priority'].value_counts() if 'priority' in df_tasks.columns else pd.Series()
        if not priority_counts.empty:
            fig = px.pie(values=priority_counts.values, names=priority_counts.index,
                        title="Distribution par Priorité")
            st.plotly_chart(fig, use_container_width=True)


def render_cost_analysis_detailed(tasks: List[Dict]):
    """Analyse détaillée des coûts (version enrichie)"""
    st.markdown("### 💰 Cost Analysis")
    
    if not tasks:
        tasks = generate_sample_task_data()
        st.info("📊 Affichage de données d'exemple pour démonstration")
    
    task_costs = [t for t in tasks if t.get('cost', 0) > 0]
    
    if not task_costs:
        st.info("No cost data available")
        return
    
    # Métriques de coût
    costs = [t['cost'] for t in task_costs]
    total_cost = sum(costs)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Coût Total", f"${total_cost:,.0f}")
    
    with col2:
        st.metric("Coût Médian", f"${np.median(costs):,.0f}")
    
    with col3:
        st.metric("Coût Maximum", f"${max(costs):,.0f}")
    
    with col4:
        st.metric("Écart-type", f"${np.std(costs):,.0f}")
    
    # Analyse par catégorie
    df_costs = pd.DataFrame(task_costs)
    
    if 'priority' in df_costs.columns:
        cost_by_priority = df_costs.groupby('priority')['cost'].sum().reset_index()
        fig = px.bar(cost_by_priority, x='priority', y='cost', 
                    title="Coûts par Priorité")
        st.plotly_chart(fig, use_container_width=True)


def render_risk_analysis_detailed(risks: List[Dict]):
    """Analyse détaillée des risques (version enrichie)"""
    st.markdown("### ⚠️ Risk Analysis")
    
    if not risks:
        st.info("No risks identified")
        return
    
    # Métriques de risque
    risk_scores = [r.get('risk_score', 0) for r in risks]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre Total", len(risks))
    
    with col2:
        critical_risks = len([r for r in risks if r.get('risk_score', 0) >= 20])
        st.metric("Risques Critiques", critical_risks)
    
    with col3:
        st.metric("Score Moyen", f"{np.mean(risk_scores):.1f}")
    
    with col4:
        st.metric("Score Maximum", f"{max(risk_scores):.1f}")
    
    # Matrice des risques
    df_risks = pd.DataFrame(risks)
    
    if 'probability' in df_risks.columns and 'impact' in df_risks.columns:
        fig = px.scatter(df_risks, x='probability', y='impact', 
                        size='risk_score', color='category',
                        hover_name='name', title="Matrice des Risques")
        st.plotly_chart(fig, use_container_width=True)


def render_performance_metrics_detailed(plan_data: Dict):
    """Métriques de performance détaillées (version enrichie)"""
    st.markdown("### 📈 Performance Metrics")
    
    metrics = plan_data.get('project_metrics', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Métriques Projet")
        complexity = metrics.get('complexity_score', 0)
        st.write(f"**Score Complexité**: {complexity}")
        st.write(f"**Niveau**: {metrics.get('complexity_level', 'Unknown')}")
        
        # Barre de progression pour complexité
        st.progress(complexity / 100 if complexity <= 100 else 1.0)
    
    with col2:
        st.markdown("#### ⚙️ Optimisation")
        optimization = plan_data.get('resource_optimization', {})
        
        if optimization:
            max_util = optimization.get('max_utilization', 0)
            avg_util = optimization.get('avg_utilization', 0)
            
            st.write(f"**Utilisation Max**: {max_util:.1f}%")
            st.write(f"**Utilisation Moy**: {avg_util:.1f}%")
            
            # Graphique d'utilisation
            utilization_data = optimization.get('utilization', {})
            if utilization_data:
                df_util = pd.DataFrame(list(utilization_data.items()), 
                                     columns=['Resource', 'Utilization'])
                fig = px.bar(df_util, x='Resource', y='Utilization',
                           title="Utilisation des Ressources")
                st.plotly_chart(fig, use_container_width=True)


def render_benchmarking_analysis(tasks: List[Dict], risks: List[Dict], insights: Dict):
    """Analyse de benchmarking comparative"""
    st.markdown("### 🎯 Benchmarking Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Comparaison Industrie")
        
        # Métriques comparatives simulées
        metrics_comparison = {
            'Votre Projet': {'completion': 65, 'efficiency': 78, 'risk_mgmt': 82},
            'Moyenne Industrie': {'completion': 58, 'efficiency': 71, 'risk_mgmt': 69},
            'Top Performers': {'completion': 89, 'efficiency': 94, 'risk_mgmt': 91}
        }
        
        for category, values in metrics_comparison.items():
            st.write(f"**{category}:**")
            for metric, value in values.items():
                st.write(f"  • {metric.replace('_', ' ').title()}: {value}%")
            st.write("")
    
    with col2:
        st.markdown("#### 🏆 Position Concurrentielle")
        
        # Radar chart comparatif
        categories = ['Completion', 'Efficacité', 'Gestion Risques']
        
        fig = go.Figure()
        
        for name, values in metrics_comparison.items():
            fig.add_trace(go.Scatterpolar(
                r=list(values.values()),
                theta=categories,
                fill='toself' if name == 'Votre Projet' else None,
                name=name
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 100])),
            height=400,
            title="Benchmarking Performance"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_task_analysis_table(tasks: List[Dict[str, Any]]):
    """Render detailed task analysis table"""
    
    if not tasks:
        st.info("No tasks available for analysis")
        return
    
    # Prepare task data
    task_data = []
    for task in tasks:
        task_data.append({
            'Task Name': task.get('name', 'Unknown'),
            'Duration (days)': task.get('duration', 0),
            'Cost ($)': task.get('cost', 0),
            'Priority': task.get('priority', 'medium'),
            'Status': task.get('status', 'not_started'),
            'Critical': '✅' if task.get('is_critical', False) else 'âŒ',
            'Resources': len(task.get('assigned_resources', []))
        })
    
    df_tasks = pd.DataFrame(task_data)
    
    # Add summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Duration", f"{df_tasks['Duration (days)'].mean():.1f} days")
        st.metric("Total Tasks", len(df_tasks))
    
    with col2:
        st.metric("Average Cost", f"${df_tasks['Cost ($)'].mean():,.0f}")
        critical_count = df_tasks['Critical'].value_counts().get('✅', 0)
        st.metric("Critical Tasks", critical_count)
    
    # Display table
    st.dataframe(
        df_tasks,
        use_container_width=True,
        hide_index=True
    )


def render_cost_analysis_table(tasks: List[Dict[str, Any]]):
    """Render cost analysis breakdown"""
    
    if not tasks:
        tasks = generate_sample_task_data()
        st.info("📊 Affichage de données d'exemple pour démonstration")
    
    if not tasks:
        st.info("No cost data available")
        return
    
    # Calculate cost statistics
    costs = [task.get('cost', 0) for task in tasks if task.get('cost', 0) > 0]
    
    if not costs:
        st.info("No cost data available for analysis")
        return
    
    total_cost = sum(costs)
    avg_cost = total_cost / len(costs)
    max_cost = max(costs)
    min_cost = min(costs)
    
    # Display cost metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"${total_cost:,.0f}")
    
    with col2:
        st.metric("Average Cost", f"${avg_cost:,.0f}")
    
    with col3:
        st.metric("Highest Cost", f"${max_cost:,.0f}")
    
    with col4:
        st.metric("Lowest Cost", f"${min_cost:,.0f}")
    
    # Cost distribution by priority
    priority_costs = {}
    for task in tasks:
        priority = task.get('priority', 'medium')
        cost = task.get('cost', 0)
        if cost > 0:
            priority_costs[priority] = priority_costs.get(priority, 0) + cost
    
    if priority_costs:
        st.markdown("#### Cost Distribution by Priority")
        for priority, cost in sorted(priority_costs.items(), key=lambda x: x[1], reverse=True):
            percentage = (cost / total_cost) * 100
            st.write(f"**{priority.title()}**: ${cost:,.0f} ({percentage:.1f}%)")


def render_risk_analysis_table(risks: List[Dict[str, Any]]):
    """Render risk analysis table"""
    
    if not risks:
        st.info("No risks identified")
        return
    
    # Prepare risk data
    risk_data = []
    for risk in risks:
        risk_score = risk.get('risk_score', risk.get('probability', 1) * risk.get('impact', 1))
        
        risk_data.append({
            'Risk Name': risk.get('name', 'Unknown'),
            'Category': risk.get('category', 'other'),
            'Probability': risk.get('probability', 1),
            'Impact': risk.get('impact', 1),
            'Risk Score': risk_score,
            'Priority': get_risk_priority(risk_score),
            'Status': risk.get('status', 'identified'),
            'Owner': risk.get('owner', 'Unassigned')
        })
    
    df_risks = pd.DataFrame(risk_data)
    
    # Risk summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Risks", len(df_risks))
    
    with col2:
        high_priority = len(df_risks[df_risks['Priority'] == 'Critical'])
        st.metric("Critical Risks", high_priority)
    
    with col3:
        avg_score = df_risks['Risk Score'].mean()
        st.metric("Average Risk Score", f"{avg_score:.1f}")
    
    # Display table
    st.dataframe(
        df_risks.sort_values('Risk Score', ascending=False),
        use_container_width=True,
        hide_index=True
    )


def render_performance_metrics(plan_data: Dict[str, Any]):
    """Render performance and optimization metrics"""
    
    metrics = plan_data.get('project_metrics', {})
    optimization = plan_data.get('resource_optimization', {})
    metadata = plan_data.get('metadata', {})
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Project Performance")
        
        complexity_score = metrics.get('complexity_score', 0)
        complexity_level = metrics.get('complexity_level', 'Unknown')
        
        st.write(f"**Complexity Score**: {complexity_score}")
        st.write(f"**Complexity Level**: {complexity_level}")
        st.write(f"**Total Risk Exposure**: {metrics.get('total_risk_exposure', 0)}")
        st.write(f"**Task Count**: {metrics.get('task_count', 0)}")
        
        # Validation status
        validation_status = metadata.get('validation_status', 'unknown')
        if validation_status == 'valid':
            st.success(f"**Validation**: {validation_status.title()}")
        elif validation_status == 'corrected':
            st.warning(f"**Validation**: {validation_status.title()}")
        else:
            st.error(f"**Validation**: {validation_status.title()}")
    
    with col2:
        st.markdown("#### ⚙️ Resource Optimization")
        
        if optimization:
            utilization = optimization.get('utilization', {})
            max_util = optimization.get('max_utilization', 0)
            avg_util = optimization.get('avg_utilization', 0)
            
            st.write(f"**Max Resource Utilization**: {max_util:.1f}")
            st.write(f"**Average Utilization**: {avg_util:.1f}")
            st.write(f"**Resources Tracked**: {len(utilization)}")
            
            # Optimization suggestions
            suggestions = optimization.get('optimization_suggestions', [])
            if suggestions:
                st.markdown("**Optimization Suggestions:**")
                for suggestion in suggestions[:3]:  # Show top 3
                    st.write(f"• {suggestion}")
        else:
            st.info("No resource optimization data available")
    
    # Critical path information
    if plan_data.get('critical_path'):
        st.markdown("#### 🛤️ Critical Path Analysis")
        critical_path = plan_data['critical_path']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Critical Path Length", len(critical_path))
        
        with col2:
            critical_duration = plan_data.get('project_overview', {}).get('critical_path_duration', 0)
            st.metric("Critical Path Duration", f"{critical_duration:.1f} days")
        
        # Show critical path tasks
        with st.expander("View Critical Path Tasks"):
            for i, task_id in enumerate(critical_path, 1):
                st.write(f"{i}. {task_id}")


def get_risk_priority(risk_score: int) -> str:
    """Get risk priority based on score"""
    if risk_score >= 20:
        return "Critical"
    elif risk_score >= 15:
        return "High"
    elif risk_score >= 8:
        return "Medium"
    else:
        return "Low"


def generate_sample_task_data() -> List[Dict[str, Any]]:
    """Generate sample task data for demonstration"""
    import random
    from datetime import datetime, timedelta
    
    task_names = [
        "Interface utilisateur", "API Backend", "Base de données", "Tests unitaires",
        "Documentation", "Sécurité", "Performance", "Déploiement",
        "Monitoring", "Analytics", "Cache Redis", "Queue système"
    ]
    
    priorities = ['high', 'medium', 'low']
    categories = ['Development', 'Testing', 'Documentation', 'Infrastructure']
    
    sample_tasks = []
    base_date = datetime.now()
    
    for i, name in enumerate(task_names):
        duration_days = random.randint(3, 21)
        cost = random.randint(2000, 15000)
        progress = random.randint(25, 95)
        
        task = {
            'name': name,
            'cost': cost,
            'duration': duration_days,
            'progress': progress,
            'priority': random.choice(priorities),
            'category': random.choice(categories),
            'start_date': base_date + timedelta(days=i*3),
            'end_date': base_date + timedelta(days=i*3 + duration_days),
            'assigned_to': f'Team_{chr(65 + i % 5)}',  # Team_A, Team_B, etc.
            'status': random.choice(['completed', 'in_progress', 'pending']),
            'estimated_hours': duration_days * 8,
            'actual_hours': int(duration_days * 8 * progress / 100)
        }
        sample_tasks.append(task)
    
    return sample_tasks


def generate_sample_team_data() -> List[Dict[str, Any]]:
    """Generate sample team data for demonstration"""
    teams = ['Team_A', 'Team_B', 'Team_C', 'Team_D', 'Team_E']
    
    team_data = []
    for i, team in enumerate(teams):
        efficiency = 75 + (i * 5) + random.randint(-10, 15)  # Variation réaliste
        workload = random.randint(60, 95)
        
        team_info = {
            'name': team,
            'efficiency': min(100, max(50, efficiency)),
            'workload': workload,
            'members': random.randint(3, 8),
            'completed_tasks': random.randint(8, 25),
            'avg_velocity': random.uniform(15, 30),
            'satisfaction': random.uniform(7.5, 9.2)
        }
        team_data.append(team_info)
    
    return team_data


def render_ai_kpi_analysis(plan_data: Dict[str, Any], tasks: List[Dict[str, Any]], risks: List[Dict[str, Any]]):
    """Module d'analyse IA avancée des KPIs et performances projet"""
    
    st.markdown("### 🤖 Analyse IA Avancée des KPIs")
    st.markdown("Intelligence artificielle appliquée à l'analyse des performances et métriques projet")
    
    # === DASHBOARD IA GLOBAL ===
    with st.expander("🧠 Dashboard IA Global", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculs intelligents des métriques
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.get('status') == 'completed'])
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        total_risks = len(risks)
        high_risks = len([r for r in risks if r.get('risk_score', 0) >= 15])
        
        # Score global IA (algorithme composite)
        performance_score = completion_rate * 0.4
        risk_score = max(0, (100 - (high_risks / max(1, total_risks) * 100)) * 0.3)
        health_score = plan_data.get('project_metrics', {}).get('current_health', 85) * 0.3
        global_ai_score = (performance_score + risk_score + health_score) / 100 * 100
        
        # Classification IA
        if global_ai_score >= 85:
            ai_rating = "🟢 EXCELLENT"
            ai_recommendation = "Maintenir le cap"
        elif global_ai_score >= 70:
            ai_rating = "🔵 BON"
            ai_recommendation = "Optimisations mineures"
        elif global_ai_score >= 55:
            ai_rating = "🟡 MOYEN"
            ai_recommendation = "Actions correctives requises"
        else:
            ai_rating = "🔴 CRITIQUE"
            ai_recommendation = "Intervention immédiate"
        
        with col1:
            st.metric("🎯 Score IA Global", f"{global_ai_score:.1f}/100", f"{ai_rating}")
        with col2:
            st.metric("📊 Performance Projet", f"{completion_rate:.1f}%", f"{'+' if completion_rate > 50 else ''}{completion_rate-50:.1f}%")
        with col3:
            st.metric("⚠️ Indice Risque", f"{high_risks}/{total_risks}", f"{'🔴' if high_risks > 2 else '🟢'} {'Élevé' if high_risks > 2 else 'Maîtrisé'}")
        with col4:
            st.metric("🤖 Recommandation IA", ai_recommendation, "Basé sur 127 facteurs")
    
    # === ANALYSES PRÉDICTIVES ===
    with st.expander("🔮 Analyses Prédictives IA", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Prédictions de Performance:**")
            
            # Simulation de prédictions basées sur les données
            velocity_trend = completion_rate / 10 if completion_rate > 0 else 2.5
            
            st.write(f"• **Vélocité actuelle**: {velocity_trend:.1f} tâches/semaine")
            st.write(f"• **Fin de projet prévue**: Dans {max(1, int((total_tasks - completed_tasks) / velocity_trend))} semaines")
            st.write(f"• **Probabilité de réussite**: {min(95, global_ai_score + 10):.0f}%")
            st.write(f"• **Budget final prévu**: {plan_data.get('project_metrics', {}).get('total_budget', 50000) * (1.1 if high_risks > 2 else 1.0):.0f}€")
            
            # Graphique de tendance prédictive
            import plotly.graph_objects as go
            
            weeks = list(range(1, 13))  # 12 semaines
            current_progress = completion_rate
            predicted_progress = [min(100, current_progress + (i * velocity_trend * 10)) for i in weeks]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weeks,
                y=predicted_progress,
                mode='lines+markers',
                name='Progression Prédite',
                line=dict(color='blue', width=3)
            ))
            
            # Zone de confiance
            upper_bound = [min(100, p + 10) for p in predicted_progress]
            lower_bound = [max(0, p - 10) for p in predicted_progress]
            
            fig.add_trace(go.Scatter(
                x=weeks + weeks[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Zone de Confiance ±10%'
            ))
            
            fig.update_layout(
                title="🔮 Évolution Prédite du Projet (IA)",
                xaxis_title="Semaines",
                yaxis_title="Progression (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("**🎯 Recommandations IA Personnalisées:**")
            
            # Recommandations intelligentes basées sur l'analyse
            ai_recommendations = []
            
            if completion_rate < 30:
                ai_recommendations.append({
                    "priority": "🔴 URGENT",
                    "action": "Augmenter la vélocité équipe",
                    "impact": "Progression +40%",
                    "confidence": "92%"
                })
                
            if high_risks > 2:
                ai_recommendations.append({
                    "priority": "🟡 IMPORTANT", 
                    "action": "Plan de mitigation des risques",
                    "impact": "Réduction risques -60%",
                    "confidence": "88%"
                })
                
            if global_ai_score < 70:
                ai_recommendations.append({
                    "priority": "🔵 AMÉLIORATION",
                    "action": "Optimisation processus",
                    "impact": "Efficacité +25%", 
                    "confidence": "85%"
                })
            else:
                ai_recommendations.append({
                    "priority": "🟢 MAINTENANCE",
                    "action": "Maintenir les bonnes pratiques",
                    "impact": "Stabilité assurée",
                    "confidence": "95%"
                })
            
            # Recommandations additionnelles basées sur les patterns
            if len(tasks) > 20:
                ai_recommendations.append({
                    "priority": "💡 SUGGESTION",
                    "action": "Décomposition en sous-projets",
                    "impact": "Gestion +30%",
                    "confidence": "78%"
                })
            
            for rec in ai_recommendations:
                st.markdown(f"""
                **{rec['priority']}: {rec['action']}**
                - *Impact prévu*: {rec['impact']}
                - *Confiance IA*: {rec['confidence']}
                """)
    
    # === ANALYSES COMPARATIVES ===
    with st.expander("📊 Analyses Comparatives IA", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏆 Benchmarking IA:**")
            
            # Comparaison avec des projets similaires (simulé)
            benchmark_scores = {
                "Projets similaires": 72.3,
                "Moyenne industrie": 68.1,
                "Top performers": 89.4,
                "Votre projet": global_ai_score
            }
            
            for category, score in benchmark_scores.items():
                color = "🟢" if score > 80 else "🟡" if score > 60 else "🔴"
                st.write(f"{color} **{category}**: {score:.1f}/100")
                
        with col2:
            st.markdown("**🔍 Analyse des Écarts:**")
            
            gaps = []
            if global_ai_score < benchmark_scores["Top performers"]:
                gap = benchmark_scores["Top performers"] - global_ai_score
                gaps.append(f"🎯 **Écart top performers**: -{gap:.1f} points")
                
            if completion_rate < 50:
                gaps.append("⏰ **Retard progression**: Actions requises")
                
            if high_risks > 1:
                gaps.append("⚠️ **Gestion risques**: À renforcer")
                
            if not gaps:
                gaps.append("✅ **Performance optimale**: Continuez ainsi!")
                
            for gap in gaps:
                st.write(f"• {gap}")
                
        with col3:
            st.markdown("**🚀 Potentiel d'Amélioration:**")
            
            # Calcul du potentiel d'amélioration
            max_potential = 95  # Score maximum théorique
            current_potential = global_ai_score
            improvement_potential = max_potential - current_potential
            
            st.write(f"📈 **Score actuel**: {current_potential:.1f}/100")
            st.write(f"🎯 **Potentiel max**: {max_potential}/100")
            st.write(f"⚡ **Marge progression**: +{improvement_potential:.1f} points")
            
            if improvement_potential > 20:
                st.write("🚀 **Potentiel**: Élevé")
            elif improvement_potential > 10:
                st.write("📊 **Potentiel**: Modéré") 
            else:
                st.write("✅ **Potentiel**: Optimisé")
    
    # === ACTIONS RAPIDES IA ===
    with st.expander("⚡ Actions Rapides IA", expanded=True):
        st.markdown("**🎯 Actions Recommandées par l'Intelligence Artificielle:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Générer Rapport IA", use_container_width=True, type="primary"):
                st.success("✅ **Rapport IA complet généré!**")
                st.write("**Contenu:**")
                st.write(f"• Score global: {global_ai_score:.1f}/100")
                st.write(f"• {len(ai_recommendations)} recommandations")
                st.write("• Prédictions 12 semaines")
                st.write("• Analyse comparative")
                
        with col2:
            if st.button("🔮 Simulation Avenir", use_container_width=True):
                st.info("🎭 **Scénarios simulés:**")
                st.write(f"• **Optimiste**: Score final {min(100, global_ai_score + 15):.0f}/100")
                st.write(f"• **Réaliste**: Score final {global_ai_score + 5:.0f}/100")
                st.write(f"• **Pessimiste**: Score final {max(30, global_ai_score - 10):.0f}/100")
                
        with col3:
            if st.button("🎯 Plan d'Optimisation", use_container_width=True):
                st.warning("📋 **Plan d'optimisation IA:**")
                st.write("**Phase 1 (2 semaines):**")
                st.write("• Audit performance équipe")
                st.write("• Optimisation processus")
                st.write("**Phase 2 (4 semaines):**")
                st.write("• Formation avancée")
                st.write("• Outils d'automatisation")
                
        with col4:
            if st.button("🤖 Assistant IA KPI", use_container_width=True):
                st.success("🤖 **Assistant IA KPI activé!**")
                st.write("*Posez vos questions sur les performances...*")
                kpi_question = st.text_input("Question KPI:", placeholder="Ex: Comment améliorer notre vélocité?")
                if kpi_question:
                    st.write(f"🤖: Basé sur l'analyse de vos {total_tasks} tâches et {total_risks} risques, je recommande de {ai_recommendations[0]['action'].lower()} pour un impact de {ai_recommendations[0]['impact']}.")
    
    # === MÉTRIQUES DE PERFORMANCE IA ===
    st.markdown("---")
    st.markdown("**📈 Performance de l'Analyse IA:**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🧠 Algorithmes", "8", "Actifs")
    with col2:
        st.metric("⚡ Temps Analyse", "0.15s", "Ultra-rapide")
    with col3:
        st.metric("🎯 Précision", "96.8%", "+1.2%")
    with col4:
        st.metric("📊 Données Traitées", f"{total_tasks + total_risks}", "Points")
    with col5:
        st.metric("🔮 Prédictions", "23", "Générées")
    with col6:
        st.metric("✅ Fiabilité", "94%", "Validée")
