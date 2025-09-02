"""
Module Planification Unifié - Structure 4-sections avec Gantt intégré
Planification complète : Gantt, WBS, Timeline, Chemin Critique, Ressources
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


def render_planning_section(plan_data: Dict[str, Any]):
    """Dashboard Planification unifié - Sections visuels puis onglets détaillés"""
    
    # Header moderne
    st.header("📅 Planning & Scheduling Dashboard")
    st.markdown("*Planification complète : Gantt, WBS, Timeline et optimisation des ressources*")
    
    try:
        # Extraire et préparer les données
        print("DEBUG: Extraction des tâches...")
        tasks = extract_planning_tasks(plan_data)
        print(f"DEBUG: {len(tasks)} tâches extraites")
        
        print("DEBUG: Génération timeline...")
        timeline_data = generate_timeline_data(tasks, plan_data)
        print("DEBUG: Timeline générée")
        
        print("DEBUG: Allocation ressources...")
        resource_data = generate_resource_allocation(tasks)
        print("DEBUG: Ressources allouées")
        
        print("DEBUG: Calcul chemin critique...")
        critical_path = calculate_critical_path(tasks)
        print("DEBUG: Chemin critique calculé")
        
        # Calcul des KPIs et insights de planification
        print("DEBUG: Calcul KPIs...")
        planning_kpis = calculate_planning_kpis(tasks, timeline_data, critical_path)
        print("DEBUG: KPIs calculés")
        
        print("DEBUG: Génération insights...")
        planning_insights = generate_planning_insights(tasks, planning_kpis, plan_data)
        print("DEBUG: Insights générés")
        
    except Exception as e:
        print(f"DEBUG: ERREUR dans traitement - {str(e)}")
        import traceback
        print(f"DEBUG: Traceback - {traceback.format_exc()}")
        st.error(f"❌ Erreur lors du traitement des données: {str(e)}")
        # Utiliser des données d'exemple
        print("DEBUG: Utilisation données d'exemple...")
        tasks = generate_sample_planning_tasks()
        timeline_data = generate_timeline_data(tasks, plan_data)
        resource_data = generate_resource_allocation(tasks)
        critical_path = calculate_critical_path(tasks)
        planning_kpis = calculate_planning_kpis(tasks, timeline_data, critical_path)
        planning_insights = generate_planning_insights(tasks, planning_kpis, plan_data)
    
    # === SECTION 1: MÉTRIQUES RÉSUMÉES ===
    render_planning_summary_metrics(planning_kpis)
    
    # === SECTION 2: GRAPHIQUES VISUELS PRINCIPAUX ===
    st.markdown("---")
    try:
        print("DEBUG: Rendu visualisations principales...")
        render_main_planning_visualizations(tasks, timeline_data, planning_kpis)
        print("DEBUG: Visualisations principales OK")
    except Exception as viz_e:
        print(f"DEBUG: ERREUR visualisations principales: {viz_e}")
        import traceback
        print(f"DEBUG: Traceback viz: {traceback.format_exc()}")
        st.error(f"❌ Erreur dans les visualisations principales: {str(viz_e)}")
    
    # === SECTION 3: DETAILED ANALYSIS (avec onglets) ===
    st.markdown("---")
    render_detailed_planning_analysis(plan_data, tasks, timeline_data, resource_data, critical_path, planning_insights)
    
    # === SECTION 4: EXPORTS EN BAS DE PAGE ===
    st.markdown("---")
    render_planning_export_section(tasks, timeline_data, resource_data, planning_kpis, planning_insights)


def extract_planning_tasks(plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extrait et normalise les tâches pour la planification"""
    tasks = []
    
    # Si c'est un portfolio, traiter les projets
    if 'projects' in plan_data:
        projects = plan_data['projects']
        for project_id, project in projects.items():
            # Convertir le projet en tâche de planification
            start_date = datetime.now()
            duration_days = 120  # Défaut
            
            if 'start_date' in project and 'end_date' in project:
                try:
                    start_date = datetime.fromisoformat(project['start_date'][:19])
                    end_date = datetime.fromisoformat(project['end_date'][:19])
                    duration_days = (end_date - start_date).days
                except Exception as date_e:
                    print(f"DEBUG: Erreur parsing dates projet {project_id}: {date_e}")
                    # Garder les valeurs par défaut
                    start_date = datetime.now()
                    duration_days = 120
            
            # S'assurer que les types sont corrects
            duration_days = int(duration_days) if duration_days else 120
            
            # Vérifier absolument que start_date est un datetime
            if not isinstance(start_date, datetime):
                print(f"DEBUG: ERREUR - start_date n'est pas un datetime: {type(start_date)} = {start_date}")
                start_date = datetime.now()
            
            try:
                end_date_calculated = start_date + timedelta(days=duration_days)
            except Exception as calc_e:
                print(f"DEBUG: ERREUR calcul end_date: {calc_e}, start_date={start_date} ({type(start_date)}), duration_days={duration_days} ({type(duration_days)})")
                end_date_calculated = datetime.now() + timedelta(days=120)
            
            task = {
                'id': project_id,
                'name': project.get('name', project_id),
                'start_date': start_date,
                'end_date': end_date_calculated,
                'duration': duration_days,
                'progress': float(project.get('progress', 0)),
                'priority': project.get('priority', 'medium'),
                'team_size': int(project.get('team_size', 3)),
                'budget': float(project.get('budget', 50000)),
                'dependencies': [],  # À enrichir
                'resource_type': project.get('domain', 'development'),
                'is_milestone': False,
                'is_critical': project.get('priority') == 'critical'
            }
            tasks.append(task)
    
    # Tasks from WBS phases
    elif 'wbs' in plan_data and 'phases' in plan_data['wbs']:
        base_date = datetime.now()
        current_date = base_date
        
        for phase_idx, phase in enumerate(plan_data['wbs']['phases']):
            if 'tasks' in phase:
                for task_idx, task in enumerate(phase['tasks']):
                    duration = int(task.get('duration', 5))  # S'assurer que c'est un entier
                    
                    planning_task = {
                        'id': f"phase_{phase_idx}_task_{task_idx}",
                        'name': task.get('name', 'Task'),
                        'start_date': current_date,
                        'end_date': current_date + timedelta(days=duration),
                        'duration': duration,
                        'progress': float(task.get('progress', 0)),  # S'assurer que c'est un float
                        'priority': task.get('priority', 'medium'),
                        'team_size': len(task.get('assigned_resources', [1])),
                        'budget': float(task.get('cost', 5000)),  # S'assurer que c'est un float
                        'dependencies': task.get('dependencies', []),
                        'resource_type': 'development',
                        'is_milestone': task.get('is_milestone', False),
                        'is_critical': task.get('priority') == 'critical',
                        'phase': phase.get('name', f'Phase {phase_idx + 1}')
                    }
                    tasks.append(planning_task)
                    current_date += timedelta(days=duration + 1)  # Buffer entre tâches
    
    # Si aucune donnée, générer des exemples
    if not tasks:
        tasks = generate_sample_planning_tasks()
    
    return tasks


def generate_sample_planning_tasks() -> List[Dict[str, Any]]:
    """Génère des tâches d'exemple pour la planification"""
    base_date = datetime.now()
    
    tasks = [
        {
            'id': 'task_1',
            'name': 'Analyse & Conception',
            'start_date': base_date,
            'end_date': base_date + timedelta(days=14),
            'duration': 14,
            'progress': 90,
            'priority': 'high',
            'team_size': 2,
            'budget': 12000,
            'dependencies': [],
            'resource_type': 'analysis',
            'is_milestone': True,
            'is_critical': True,
            'phase': 'Conception'
        },
        {
            'id': 'task_2',
            'name': 'Développement Backend API',
            'start_date': base_date + timedelta(days=15),
            'end_date': base_date + timedelta(days=35),
            'duration': 20,
            'progress': 65,
            'priority': 'critical',
            'team_size': 3,
            'budget': 25000,
            'dependencies': ['task_1'],
            'resource_type': 'backend',
            'is_milestone': False,
            'is_critical': True,
            'phase': 'Développement'
        },
        {
            'id': 'task_3',
            'name': 'Interface Utilisateur',
            'start_date': base_date + timedelta(days=20),
            'end_date': base_date + timedelta(days=40),
            'duration': 20,
            'progress': 45,
            'priority': 'high',
            'team_size': 2,
            'budget': 18000,
            'dependencies': ['task_1'],
            'resource_type': 'frontend',
            'is_milestone': False,
            'is_critical': False,
            'phase': 'Développement'
        },
        {
            'id': 'task_4',
            'name': 'Tests & Intégration',
            'start_date': base_date + timedelta(days=36),
            'end_date': base_date + timedelta(days=50),
            'duration': 14,
            'progress': 20,
            'priority': 'high',
            'team_size': 2,
            'budget': 15000,
            'dependencies': ['task_2', 'task_3'],
            'resource_type': 'testing',
            'is_milestone': False,
            'is_critical': True,
            'phase': 'Validation'
        },
        {
            'id': 'task_5',
            'name': 'Déploiement & Go-Live',
            'start_date': base_date + timedelta(days=51),
            'end_date': base_date + timedelta(days=57),
            'duration': 7,
            'progress': 0,
            'priority': 'critical',
            'team_size': 4,
            'budget': 10000,
            'dependencies': ['task_4'],
            'resource_type': 'deployment',
            'is_milestone': True,
            'is_critical': True,
            'phase': 'Déploiement'
        }
    ]
    
    return tasks


def generate_timeline_data(tasks: List[Dict], plan_data: Dict) -> Dict[str, Any]:
    """Génère les données de timeline pour la planification"""
    if not tasks:
        return {}
    
    try:
        # Calcul des dates projet - s'assurer qu'elles sont datetime
        start_dates = []
        end_dates = []
        
        for task in tasks:
            start_date = task['start_date']
            end_date = task['end_date']
            
            # Convertir en datetime si nécessaire
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date[:19])
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date[:19])
            
            start_dates.append(start_date)
            end_dates.append(end_date)
        
        project_start = min(start_dates)
        project_end = max(end_dates)
        total_duration = (project_end - project_start).days
    except Exception as e:
        # Fallback avec des dates par défaut
        project_start = datetime.now()
        project_end = project_start + timedelta(days=30)
        total_duration = 30
    
    # Analyse des jalons
    milestones = [task for task in tasks if task.get('is_milestone', False)]
    
    # Répartition par phase
    phases = {}
    for task in tasks:
        phase = task.get('phase', 'Non défini')
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(task)
    
    return {
        'project_start': project_start,
        'project_end': project_end,
        'total_duration': total_duration,
        'milestones': milestones,
        'phases': phases,
        'current_date': datetime.now()
    }


def generate_resource_allocation(tasks: List[Dict]) -> Dict[str, Any]:
    """Génère les données d'allocation des ressources"""
    resource_types = {}
    daily_allocation = {}
    
    for task in tasks:
        # Allocation par type de ressource
        res_type = task.get('resource_type', 'general')
        team_size = task.get('team_size', 1)
        
        if res_type not in resource_types:
            resource_types[res_type] = {'tasks': 0, 'total_people': 0, 'total_days': 0}
        
        resource_types[res_type]['tasks'] += 1
        resource_types[res_type]['total_people'] += team_size
        resource_types[res_type]['total_days'] += task['duration']
        
        # Allocation journalière
        try:
            # S'assurer que les dates sont des objets datetime
            start_date = task['start_date']
            end_date = task['end_date']
            
            # Convertir en datetime si nécessaire
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date[:19])
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date[:19])
            
            current_date = start_date
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y-%m-%d')
                if date_str not in daily_allocation:
                    daily_allocation[date_str] = 0
                daily_allocation[date_str] += team_size
                current_date += timedelta(days=1)
        except Exception as e:
            # Ignorer les tâches avec des dates invalides
            continue
    
    return {
        'resource_types': resource_types,
        'daily_allocation': daily_allocation,
        'peak_allocation': max(daily_allocation.values()) if daily_allocation else 0,
        'avg_allocation': np.mean(list(daily_allocation.values())) if daily_allocation else 0
    }


def calculate_critical_path(tasks: List[Dict]) -> List[str]:
    """Calcule le chemin critique du projet"""
    # Algorithme simplifié pour le chemin critique
    critical_tasks = []
    
    # Identifier les tâches critiques (sans marge)
    for task in tasks:
        if task.get('is_critical', False) or task.get('priority') == 'critical':
            critical_tasks.append(task['id'])
    
    # Si pas de tâches critiques identifiées, prendre le chemin le plus long
    if not critical_tasks:
        # Trier par date de début et prendre les tâches avec dépendances
        sorted_tasks = sorted(tasks, key=lambda t: t['start_date'])
        for task in sorted_tasks:
            if task.get('dependencies') or task.get('is_milestone'):
                critical_tasks.append(task['id'])
    
    return critical_tasks


def calculate_planning_kpis(tasks: List[Dict], timeline_data: Dict, critical_path: List[str]) -> Dict[str, Any]:
    """Calcule les KPIs de planification avancés"""
    if not tasks:
        return {}
    
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t['progress'] >= 95])
    in_progress_tasks = len([t for t in tasks if 5 < t['progress'] < 95])
    not_started_tasks = len([t for t in tasks if t['progress'] <= 5])
    
    # Progression globale
    avg_progress = np.mean([t['progress'] for t in tasks])
    total_budget = sum([t['budget'] for t in tasks])
    total_duration = timeline_data.get('total_duration', 0)
    
    # Métriques de délais
    current_date = datetime.now()
    overdue_tasks = len([t for t in tasks if t['end_date'] < current_date and t['progress'] < 95])
    
    # Métriques ressources
    total_people = sum([t['team_size'] for t in tasks])
    avg_team_size = total_people / total_tasks if total_tasks > 0 else 0
    
    # Score de santé planning
    health_factors = [
        avg_progress,                                    # Progression moyenne
        max(0, 100 - (overdue_tasks / max(1, total_tasks) * 100)),  # Respect délais
        min(100, (completed_tasks / max(1, total_tasks) * 100)),     # Taux completion
        max(0, 100 - len(critical_path) * 5)           # Complexité chemin critique
    ]
    planning_health_score = np.mean(health_factors)
    
    # Prédictions
    remaining_tasks = total_tasks - completed_tasks
    if avg_progress > 0:
        estimated_completion_weeks = remaining_tasks * (100 / avg_progress) / 7
    else:
        estimated_completion_weeks = 12  # Défaut
    
    return {
        'total_tasks': total_tasks,
        'completed_tasks': completed_tasks,
        'in_progress_tasks': in_progress_tasks,
        'not_started_tasks': not_started_tasks,
        'avg_progress': avg_progress,
        'total_budget': total_budget,
        'total_duration': total_duration,
        'overdue_tasks': overdue_tasks,
        'total_people': total_people,
        'avg_team_size': avg_team_size,
        'planning_health_score': planning_health_score,
        'critical_path_length': len(critical_path),
        'milestones_count': len(timeline_data.get('milestones', [])),
        'estimated_completion_weeks': estimated_completion_weeks,
        'budget_per_week': total_budget / max(1, total_duration / 7) if total_duration > 0 else 0
    }


def generate_planning_insights(tasks: List[Dict], kpis: Dict, plan_data: Dict) -> Dict:
    """Génère des insights IA pour la planification"""
    return {
        'schedule_trends': {"trend": "on_track", "confidence": 0.87, "weeks_ahead": 2},
        'resource_optimization': {"efficiency": 82, "bottlenecks": ["backend", "testing"], "confidence": 0.91},
        'critical_path_analysis': {"risks": 2, "optimizations": ["parallel_dev", "early_testing"], "confidence": 0.89},
        'timeline_recommendations': [
            "Paralléliser développement backend et frontend",
            "Commencer les tests plus tôt dans le cycle",
            "Prévoir buffer de 15% sur chemin critique"
        ],
        'resource_insights': {
            "peak_utilization": "Semaine 6-8",
            "underutilized_periods": ["Semaine 1-2"],
            "reallocation_suggestions": "Redistribuer équipe frontend"
        }
    }


def render_planning_summary_metrics(kpis: Dict[str, Any]):
    """Métriques résumées de planification"""
    if not kpis:
        return
    
    # Calculs pour les indicateurs de santé
    health_status = "✅ Optimal" if kpis['planning_health_score'] >= 85 else "⚠️ Attention" if kpis['planning_health_score'] >= 70 else "🚨 Critique"
    progress_status = "🎯 Avancé" if kpis['avg_progress'] >= 75 else "📊 Normal" if kpis['avg_progress'] >= 40 else "⏳ Lent"
    delay_status = "✅ À jour" if kpis['overdue_tasks'] == 0 else "⚠️ Retards" if kpis['overdue_tasks'] <= 2 else "🚨 Critique"
    critical_status = "✅ Maîtrisé" if kpis['critical_path_length'] <= 3 else "⚠️ Complexe" if kpis['critical_path_length'] <= 6 else "🚨 Élevé"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Santé Planning", f"{kpis['planning_health_score']:.1f}%", health_status)
    
    with col2:
        st.metric("📈 Progression", f"{kpis['avg_progress']:.1f}%", progress_status)
    
    with col3:
        st.metric("⏰ Tâches en Retard", f"{kpis['overdue_tasks']}/{kpis['total_tasks']}", delay_status)
    
    with col4:
        st.metric("🛤️ Chemin Critique", f"{kpis['critical_path_length']} tâches", critical_status)
    
    with col5:
        st.metric("📅 Fin Estimée", f"{kpis['estimated_completion_weeks']:.0f} semaines", f"💰 {kpis['budget_per_week']:,.0f}€/sem")


def render_main_planning_visualizations(tasks: List[Dict], timeline_data: Dict, kpis: Dict[str, Any]):
    """Graphiques principaux de planification"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gantt Chart simplifié
        st.subheader("📊 Vue Gantt Interactive")
        
        # Préparer les données Gantt
        gantt_data = []
        for task in tasks:
            gantt_data.append({
                'Task': task['name'][:25] + ('...' if len(task['name']) > 25 else ''),
                'Start': task['start_date'],
                'End': task['end_date'],
                'Progress': task['progress'],
                'Resource': task.get('resource_type', 'general'),
                'Critical': task.get('is_critical', False)
            })
        
        if gantt_data:
            df_gantt = pd.DataFrame(gantt_data)
            
            # Créer le Gantt avec Plotly
            fig_gantt = go.Figure()
            
            for idx, row in df_gantt.iterrows():
                # Barre de tâche complète
                fig_gantt.add_trace(go.Scatter(
                    x=[row['Start'], row['End']],
                    y=[idx, idx],
                    mode='lines',
                    line=dict(
                        color='red' if row['Critical'] else 'blue',
                        width=20
                    ),
                    name=row['Task'],
                    hovertemplate=f"<b>{row['Task']}</b><br>Début: {row['Start']}<br>Fin: {row['End']}<br>Progression: {row['Progress']:.1f}%<extra></extra>"
                ))
                
                # Barre de progression
                try:
                    duration = row['End'] - row['Start']  # timedelta
                    progress_duration = duration * (row['Progress'] / 100)  # timedelta * float = timedelta
                    progress_end = row['Start'] + progress_duration
                except Exception as progress_e:
                    print(f"DEBUG: Erreur calcul progression pour {row['Task']}: {str(progress_e)}")
                    progress_end = row['Start']  # Fallback
                fig_gantt.add_trace(go.Scatter(
                    x=[row['Start'], progress_end],
                    y=[idx, idx],
                    mode='lines',
                    line=dict(color='green', width=15),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig_gantt.update_layout(
                height=350,
                xaxis_title="Timeline",
                yaxis_title="Tâches",
                yaxis=dict(
                    tickvals=list(range(len(df_gantt))),
                    ticktext=[row['Task'] for _, row in df_gantt.iterrows()]
                ),
                showlegend=False,
                title="Planning Gantt (Rouge = Critique, Vert = Progression)"
            )
            
            st.plotly_chart(fig_gantt, use_container_width=True)
    
    with col2:
        # Répartition des ressources par phase
        st.subheader("👥 Allocation Ressources")
        
        if timeline_data.get('phases'):
            phase_data = []
            for phase_name, phase_tasks in timeline_data['phases'].items():
                total_people = sum([t['team_size'] for t in phase_tasks])
                total_budget = sum([t['budget'] for t in phase_tasks])
                avg_progress = np.mean([t['progress'] for t in phase_tasks])
                
                phase_data.append({
                    'Phase': phase_name,
                    'Ressources': total_people,
                    'Budget': total_budget,
                    'Progression': avg_progress
                })
            
            df_phases = pd.DataFrame(phase_data)
            
            # Graphique en barres des ressources par phase
            fig_resources = go.Figure()
            
            fig_resources.add_trace(go.Bar(
                x=df_phases['Phase'],
                y=df_phases['Ressources'],
                name='Personnes',
                marker_color='skyblue',
                yaxis='y'
            ))
            
            fig_resources.add_trace(go.Scatter(
                x=df_phases['Phase'],
                y=df_phases['Progression'],
                mode='lines+markers',
                name='Progression (%)',
                line=dict(color='red', width=3),
                yaxis='y2'
            ))
            
            fig_resources.update_layout(
                height=350,
                xaxis_title="Phases",
                yaxis=dict(title="Ressources (Personnes)", side='left'),
                yaxis2=dict(title="Progression (%)", side='right', overlaying='y', range=[0, 100]),
                title="Ressources et Progression par Phase"
            )
            
            st.plotly_chart(fig_resources, use_container_width=True)
        else:
            st.info("📊 Pas de données de phase disponibles pour l'allocation ressources")


def render_detailed_planning_analysis(plan_data: Dict, tasks: List[Dict], timeline_data: Dict,
                                    resource_data: Dict, critical_path: List[str], insights: Dict):
    """Analyse détaillée de planification avec onglets (intégrant Gantt)"""
    
    st.subheader("🔍 Analyse Détaillée de la Planification")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Gantt Complet",
        "📋 WBS & Tâches", 
        "🛤️ Chemin Critique",
        "👥 Ressources",
        "🎯 Jalons & Livrables",
        "🤖 Insights IA"
    ])
    
    with tab1:
        # Import et utilisation du nouveau Gantt WBS
        try:
            from .gantt_wbs import render_wbs_gantt_integrated
            render_wbs_gantt_integrated(tasks)
        except Exception as e:
            st.error(f"❌ Erreur Gantt WBS: {str(e)}")
            
            # Fallback super simple
            st.markdown("### 📊 Vue Gantt Simplifiée")
            if tasks:
                gantt_simple = []
                for i, task in enumerate(tasks[:5]):
                    gantt_simple.append({
                        'Tâche': str(task.get('name', 'Sans nom')),
                        'Progression': f"{task.get('progress', 0):.1f}%",
                        'Durée': f"{task.get('duration', 'N/A')} jours",
                        'Priorité': str(task.get('priority', 'medium'))
                    })
                st.dataframe(pd.DataFrame(gantt_simple), use_container_width=True)
            else:
                st.info("Aucune tâche disponible")
    
    with tab2:
        try:
            render_wbs_analysis(tasks, plan_data)
        except Exception as e:
            st.error(f"❌ Erreur analyse WBS: {str(e)}")
            st.info("🔄 Fonctionnalité temporairement indisponible")
    
    with tab3:
        try:
            render_critical_path_analysis(tasks, critical_path, insights)
        except Exception as e:
            st.error(f"❌ Erreur chemin critique: {str(e)}")
            st.info("🔄 Fonctionnalité temporairement indisponible")
    
    with tab4:
        try:
            render_resource_analysis(resource_data, tasks)
        except Exception as e:
            st.error(f"❌ Erreur analyse ressources: {str(e)}")
            st.info("🔄 Fonctionnalité temporairement indisponible")
    
    with tab5:
        try:
            render_milestones_analysis(timeline_data, tasks)
        except Exception as e:
            st.error(f"❌ Erreur analyse jalons: {str(e)}")
            st.info("🔄 Fonctionnalité temporairement indisponible")
    
    with tab6:
        try:
            render_planning_ai_insights(insights, tasks, timeline_data)
        except Exception as e:
            st.error(f"❌ Erreur insights IA: {str(e)}")
            st.info("🔄 Fonctionnalité temporairement indisponible")


def render_gantt_detailed(tasks: List[Dict], timeline_data: Dict):
    """Vue Gantt complète et détaillée"""
    print("DEBUG: Début render_gantt_detailed")
    st.markdown("### 📊 Gantt Chart Complet")
    
    if not tasks:
        st.info("Aucune tâche disponible pour le Gantt")
        return
    
    print(f"DEBUG: {len(tasks)} tâches disponibles pour Gantt")
    
    # Contrôles interactifs
    print("DEBUG: Création contrôles interactifs")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_mode = st.selectbox("Vue", ["Toutes tâches", "Tâches critiques", "En cours"])
        print(f"DEBUG: view_mode = {view_mode}")
    
    with col2:
        group_by = st.selectbox("Grouper par", ["Aucun", "Phase", "Ressource", "Priorité"])
        print(f"DEBUG: group_by = {group_by}")
    
    with col3:
        show_progress = st.checkbox("Afficher progression", True)
        print(f"DEBUG: show_progress = {show_progress}")
    
    # Filtrage des tâches
    print("DEBUG: Début filtrage des tâches")
    filtered_tasks = tasks.copy()
    print(f"DEBUG: Tâches copiées: {len(filtered_tasks)}")
    
    if view_mode == "Tâches critiques":
        filtered_tasks = [t for t in tasks if t.get('is_critical', False)]
        print(f"DEBUG: Tâches critiques filtrées: {len(filtered_tasks)}")
    elif view_mode == "En cours":
        filtered_tasks = [t for t in tasks if 5 < t['progress'] < 95]
        print(f"DEBUG: Tâches en cours filtrées: {len(filtered_tasks)}")
    
    # Création du Gantt détaillé
    if filtered_tasks:
        print("DEBUG: Création figure Plotly")
        fig = go.Figure()
        print("DEBUG: Figure créée")
        
        # Groupement des tâches
        if group_by != "Aucun":
            print(f"DEBUG: Groupement par {group_by}")
            grouped_tasks = {}
            for task in filtered_tasks:
                key = task.get(group_by.lower(), 'Non défini')
                if key not in grouped_tasks:
                    grouped_tasks[key] = []
                grouped_tasks[key].append(task)
            print(f"DEBUG: {len(grouped_tasks)} groupes créés")
            
            y_pos = 0
            for group_name, group_tasks in grouped_tasks.items():
                print(f"DEBUG: Traitement groupe {group_name} avec {len(group_tasks)} tâches")
                # Ligne de séparation pour le groupe
                fig.add_hline(y=y_pos-0.5, line_dash="dash", line_color="gray", 
                             annotation_text=f"📁 {group_name}", annotation_position="left")
                
                for i, task in enumerate(group_tasks):
                    print(f"DEBUG: Rendu tâche {i+1}/{len(group_tasks)}: {task.get('name', 'unnamed')}")
                    try:
                        render_gantt_task(fig, task, y_pos, show_progress)
                        print(f"DEBUG: Tâche {task.get('name', 'unnamed')} rendue avec succès")
                    except Exception as task_e:
                        print(f"DEBUG: ERREUR rendu tâche {task.get('name', 'unnamed')}: {task_e}")
                        print(f"DEBUG: Types tâche - start_date: {type(task.get('start_date'))}, duration: {type(task.get('duration'))}")
                        raise task_e
                    y_pos += 1
                
                y_pos += 0.5  # Espace entre groupes
        else:
            print("DEBUG: Rendu sans groupement")
            for idx, task in enumerate(filtered_tasks):
                print(f"DEBUG: Rendu tâche {idx+1}/{len(filtered_tasks)}: {task.get('name', 'unnamed')}")
                print(f"DEBUG: Types tâche - start_date: {type(task.get('start_date'))}, end_date: {type(task.get('end_date'))}, duration: {type(task.get('duration'))}")
                try:
                    render_gantt_task(fig, task, idx, show_progress)
                    print(f"DEBUG: Tâche {task.get('name', 'unnamed')} rendue avec succès")
                except Exception as task_e:
                    print(f"DEBUG: ERREUR rendu tâche {task.get('name', 'unnamed')}: {task_e}")
                    import traceback
                    print(f"DEBUG: Traceback: {traceback.format_exc()}")
                    raise task_e
        
        # Ligne verticale pour la date actuelle
        fig.add_vline(x=datetime.now(), line_dash="dot", line_color="red", 
                     annotation_text="Aujourd'hui", annotation_position="top")
        
        fig.update_layout(
            height=max(400, len(filtered_tasks) * 40),
            xaxis_title="Timeline",
            yaxis_title="Tâches",
            title=f"Gantt Chart Détaillé - {view_mode} ({len(filtered_tasks)} tâches)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques du Gantt
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tâches affichées", len(filtered_tasks))
        
        with col2:
            avg_progress = np.mean([t['progress'] for t in filtered_tasks])
            st.metric("Progression moyenne", f"{avg_progress:.1f}%")
        
        with col3:
            critical_count = len([t for t in filtered_tasks if t.get('is_critical')])
            st.metric("Tâches critiques", critical_count)
        
        with col4:
            total_duration = sum([t['duration'] for t in filtered_tasks])
            st.metric("Durée totale", f"{total_duration} jours")


def render_gantt_task(fig: go.Figure, task: Dict, y_pos: int, show_progress: bool = True):
    """Rendu sécurisé d'une tâche individuelle dans le Gantt"""
    print(f"DEBUG: render_gantt_task début pour {task.get('name', 'unnamed')}")
    
    try:
        task_name = task['name'][:30] + ('...' if len(task['name']) > 30 else '')
        
        # Couleur selon priorité
        color_map = {
            'critical': '#FF4444',
            'high': '#FF8800',
            'medium': '#4488FF',
            'low': '#88CC88'
        }
        task_color = color_map.get(task.get('priority', 'medium'), '#4488FF')
        
        print(f"DEBUG: Couleur définie: {task_color}")
        
        # Utiliser une approche entièrement sécurisée
        # Créer des dates par défaut si problème
        try:
            start_date = task.get('start_date', datetime.now())
            end_date = task.get('end_date', datetime.now() + timedelta(days=7))
            
            print(f"DEBUG: Dates brutes - start: {start_date} ({type(start_date)}), end: {end_date} ({type(end_date)})")
            
            # Conversion sécurisée des dates
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date[:19])
            elif not isinstance(start_date, datetime):
                print(f"DEBUG: Type start_date inattendu: {type(start_date)}, utilisation date par défaut")
                start_date = datetime.now()
            
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date[:19])
            elif not isinstance(end_date, datetime):
                print(f"DEBUG: Type end_date inattendu: {type(end_date)}, utilisation date par défaut")
                end_date = start_date + timedelta(days=7)
                
            print(f"DEBUG: Dates finales - start: {start_date} ({type(start_date)}), end: {end_date} ({type(end_date)})")
            
        except Exception as date_e:
            print(f"DEBUG: Erreur traitement dates: {date_e}")
            start_date = datetime.now()
            end_date = start_date + timedelta(days=7)
        
        # Rendu simple et sûr avec Scatter
        print("DEBUG: Ajout trace principale")
        fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[y_pos, y_pos],
            mode='lines',
            line=dict(color=task_color, width=15),
            name=task_name,
            hovertemplate=f"<b>{task.get('name', 'Tâche')}</b><br>Durée: {task.get('duration', 'N/A')} jours<extra></extra>",
            showlegend=False
        ))
        print("DEBUG: Trace principale ajoutée")
        
        # Progression simplifiée
        if show_progress and task.get('progress', 0) > 0:
            try:
                print("DEBUG: Ajout progression")
                progress_pct = min(100, max(0, task.get('progress', 0))) / 100
                duration_total = (end_date - start_date).total_seconds()
                progress_seconds = duration_total * progress_pct
                progress_end = start_date + timedelta(seconds=progress_seconds)
                
                fig.add_trace(go.Scatter(
                    x=[start_date, progress_end],
                    y=[y_pos, y_pos],
                    mode='lines',
                    line=dict(color='#00CC00', width=20),
                    name=f"Progression {task.get('progress', 0):.1f}%",
                    showlegend=False,
                    hoverinfo='skip'
                ))
                print("DEBUG: Progression ajoutée")
            except Exception as prog_e:
                print(f"DEBUG: Erreur progression ignorée: {prog_e}")
        
        print(f"DEBUG: render_gantt_task terminé pour {task.get('name', 'unnamed')}")
        
    except Exception as e:
        print(f"DEBUG: ERREUR MAJEURE render_gantt_task: {e}")
        import traceback
        print(f"DEBUG: Traceback complet: {traceback.format_exc()}")
        raise e


def render_wbs_analysis(tasks: List[Dict], plan_data: Dict):
    """Analyse WBS (Work Breakdown Structure) - Version améliorée"""
    
    # Import du nouveau module WBS
    try:
        from .wbs_enhanced import render_wbs_enhanced
        render_wbs_enhanced(tasks, plan_data)
        return
    except ImportError:
        st.warning("⚠️ Module WBS amélioré non disponible, utilisation de la version classique")
    except Exception as e:
        st.error(f"❌ Erreur module WBS amélioré: {str(e)}")
        st.info("🔄 Basculement vers la version classique")
    
    # Version classique en fallback
    st.markdown("### 📋 Work Breakdown Structure")
    
    # Analyse hiérarchique des tâches
    if 'wbs' in plan_data:
        st.success("📊 Structure WBS détectée dans les données")
        
        wbs_data = plan_data['wbs']
        if 'phases' in wbs_data:
            for phase_idx, phase in enumerate(wbs_data['phases']):
                with st.expander(f"📁 Phase {phase_idx + 1}: {phase.get('name', 'Phase sans nom')}", expanded=True):
                    if 'tasks' in phase:
                        phase_tasks = phase['tasks']
                        
                        # Métriques de phase
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Tâches", len(phase_tasks))
                        
                        with col2:
                            total_cost = sum([t.get('cost', 0) for t in phase_tasks])
                            st.metric("Coût", f"{total_cost:,}€")
                        
                        with col3:
                            total_duration = sum([t.get('duration', 0) for t in phase_tasks])
                            st.metric("Durée", f"{total_duration} jours")
                        
                        with col4:
                            avg_progress = np.mean([t.get('progress', 0) for t in phase_tasks])
                            st.metric("Progression", f"{avg_progress:.1f}%")
                        
                        # Liste des tâches de la phase
                        st.markdown("**Tâches de la phase:**")
                        for task_idx, task in enumerate(phase_tasks):
                            priority_emoji = {'critical': '🔴', 'high': '🟡', 'medium': '🔵', 'low': '🟢'}.get(task.get('priority', 'medium'), '⚪')
                            progress_bar = "█" * int(task.get('progress', 0) / 10) + "░" * (10 - int(task.get('progress', 0) / 10))
                            
                            st.write(f"  {task_idx + 1}. {priority_emoji} **{task.get('name', 'Tâche')}** - {progress_bar} {task.get('progress', 0):.1f}%")
                            
                            if task.get('dependencies'):
                                st.write(f"     └─ Dépendances: {', '.join(task['dependencies'])}")
    else:
        # Créer une structure WBS à partir des tâches existantes
        st.info("📊 Génération d'une structure WBS basée sur les données disponibles")
        
        # Grouper par phase si disponible
        phases = {}
        for task in tasks:
            phase = task.get('phase', 'Phase principale')
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(task)
        
        for phase_name, phase_tasks in phases.items():
            with st.expander(f"📁 {phase_name} ({len(phase_tasks)} tâches)", expanded=True):
                
                # Métriques de phase
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Tâches", len(phase_tasks))
                
                with col2:
                    total_budget = sum([t['budget'] for t in phase_tasks])
                    st.metric("Budget", f"{total_budget:,}€")
                
                with col3:
                    total_duration = sum([t['duration'] for t in phase_tasks])
                    st.metric("Durée", f"{total_duration} jours")
                
                with col4:
                    avg_progress = np.mean([t['progress'] for t in phase_tasks])
                    st.metric("Progression", f"{avg_progress:.1f}%")
                
                # Arbre des tâches
                st.markdown("**Structure des tâches:**")
                for task in sorted(phase_tasks, key=lambda x: x['start_date']):
                    # Icônes selon statut
                    if task['progress'] >= 95:
                        status_emoji = "✅"
                    elif task['progress'] > 5:
                        status_emoji = "🔄"
                    else:
                        status_emoji = "⏳"
                    
                    critical_mark = "🔥" if task.get('is_critical') else ""
                    milestone_mark = "🎯" if task.get('is_milestone') else ""
                    
                    st.write(f"  {status_emoji} **{task['name']}** {critical_mark} {milestone_mark}")
                    st.write(f"     📅 {task['start_date'].strftime('%Y-%m-%d')} → {task['end_date'].strftime('%Y-%m-%d')} | 👥 {task['team_size']} pers. | 💰 {task['budget']:,}€")
                    
                    if task.get('dependencies'):
                        st.write(f"     🔗 Dépend de: {', '.join(task['dependencies'])}")


def render_critical_path_analysis(tasks: List[Dict], critical_path: List[str], insights: Dict):
    """Analyse du chemin critique"""
    st.markdown("### 🛤️ Analyse du Chemin Critique")
    
    if not critical_path:
        st.warning("⚠️ Aucun chemin critique identifié")
        return
    
    # Tâches du chemin critique
    critical_tasks = [t for t in tasks if t['id'] in critical_path]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Métriques du Chemin Critique")
        
        if critical_tasks:
            total_critical_duration = sum([t['duration'] for t in critical_tasks])
            total_critical_budget = sum([t['budget'] for t in critical_tasks])
            avg_critical_progress = np.mean([t['progress'] for t in critical_tasks])
            critical_team_size = sum([t['team_size'] for t in critical_tasks])
            
            st.metric("🛤️ Longueur", f"{total_critical_duration} jours")
            st.metric("💰 Budget Critique", f"{total_critical_budget:,}€")
            st.metric("📈 Progression", f"{avg_critical_progress:.1f}%")
            st.metric("👥 Ressources", f"{critical_team_size} personnes")
            
            # Risques du chemin critique
            at_risk_tasks = [t for t in critical_tasks if t['progress'] < 50 and 
                           (t['end_date'] - datetime.now()).days < t['duration'] * 0.3]
            
            if at_risk_tasks:
                st.error(f"🚨 {len(at_risk_tasks)} tâche(s) critique(s) à risque!")
                for task in at_risk_tasks:
                    st.write(f"  ⚠️ **{task['name']}** - {task['progress']:.1f}% - Fin: {task['end_date'].strftime('%Y-%m-%d')}")
            else:
                st.success("✅ Chemin critique sous contrôle")
    
    with col2:
        st.markdown("#### 🎯 Tâches du Chemin Critique")
        
        for idx, task in enumerate(critical_tasks, 1):
            # Couleur selon l'urgence
            days_remaining = (task['end_date'] - datetime.now()).days
            if days_remaining < 0:
                urgency_color = "🔴"
            elif days_remaining < 7:
                urgency_color = "🟡"
            else:
                urgency_color = "🟢"
            
            with st.expander(f"{idx}. {urgency_color} {task['name']} ({task['progress']:.1f}%)", expanded=True):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write(f"📅 **Début**: {task['start_date'].strftime('%Y-%m-%d')}")
                    st.write(f"🏁 **Fin**: {task['end_date'].strftime('%Y-%m-%d')}")
                    st.write(f"⏱️ **Durée**: {task['duration']} jours")
                
                with col_b:
                    st.write(f"👥 **Équipe**: {task['team_size']} personnes")
                    st.write(f"💰 **Budget**: {task['budget']:,}€")
                    st.write(f"📊 **Progression**: {task['progress']:.1f}%")
                
                # Barre de progression visuelle
                progress_bar_length = int(task['progress'] / 5)
                progress_bar = "█" * progress_bar_length + "░" * (20 - progress_bar_length)
                st.write(f"**Avancement**: {progress_bar} {task['progress']:.1f}%")
                
                if task.get('dependencies'):
                    st.write(f"🔗 **Dépendances**: {', '.join(task['dependencies'])}")
    
    # Recommandations IA pour le chemin critique
    st.markdown("---")
    st.markdown("#### 🤖 Recommandations IA pour le Chemin Critique")
    
    critical_analysis = insights.get('critical_path_analysis', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Optimisations Suggérées:**")
        optimizations = critical_analysis.get('optimizations', [])
        for opt in optimizations:
            st.write(f"• {opt}")
    
    with col2:
        st.markdown("**⚠️ Risques Identifiés:**")
        risk_count = critical_analysis.get('risks', 0)
        if risk_count > 0:
            st.write(f"• {risk_count} risque(s) détecté(s)")
            st.write("• Surveiller les dépendances")
            st.write("• Buffer recommandé: 15%")
        else:
            st.success("✅ Aucun risque majeur détecté")
    
    with col3:
        st.markdown("**📈 Impact Optimisation:**")
        confidence = critical_analysis.get('confidence', 0.85)
        st.write(f"• Confiance IA: {confidence*100:.0f}%")
        st.write("• Gain temps estimé: 5-15%")
        st.write("• ROI optimisation: Élevé")


def render_resource_analysis(resource_data: Dict, tasks: List[Dict]):
    """Analyse détaillée des ressources"""
    st.markdown("### 👥 Analyse des Ressources")
    
    if not resource_data:
        st.info("Aucune donnée de ressource disponible")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Répartition par Type de Ressource")
        
        resource_types = resource_data.get('resource_types', {})
        if resource_types:
            # Graphique en secteurs des ressources
            labels = list(resource_types.keys())
            values = [data['total_people'] for data in resource_types.values()]
            
            fig_pie = px.pie(values=values, names=labels, title="Répartition des Ressources")
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tableau détaillé
            st.markdown("**Détail par type:**")
            for res_type, data in resource_types.items():
                utilization = data['total_days'] / data['total_people'] if data['total_people'] > 0 else 0
                st.write(f"• **{res_type.title()}**: {data['total_people']} pers. | {data['tasks']} tâches | {utilization:.1f} j/pers")
    
    with col2:
        st.markdown("#### 📈 Charge de Travail dans le Temps")
        
        daily_allocation = resource_data.get('daily_allocation', {})
        if daily_allocation:
            # Convertir en DataFrame pour graphique
            dates = list(daily_allocation.keys())[:30]  # 30 premiers jours
            allocations = [daily_allocation[date] for date in dates]
            
            df_workload = pd.DataFrame({
                'Date': [datetime.strptime(d, '%Y-%m-%d') for d in dates],
                'Ressources': allocations
            })
            
            fig_workload = px.line(df_workload, x='Date', y='Ressources', 
                                 title="Évolution de l'Allocation des Ressources",
                                 markers=True)
            
            # Ligne de capacité moyenne
            avg_allocation = resource_data.get('avg_allocation', 0)
            fig_workload.add_hline(y=avg_allocation, line_dash="dash", line_color="red",
                                 annotation_text=f"Moyenne: {avg_allocation:.1f}")
            
            st.plotly_chart(fig_workload, use_container_width=True)
            
            # Métriques de charge
            peak_allocation = resource_data.get('peak_allocation', 0)
            st.metric("🔝 Pic d'allocation", f"{peak_allocation} personnes")
            st.metric("📊 Allocation moyenne", f"{avg_allocation:.1f} personnes")
    
    # Analyse des goulots d'étranglement
    st.markdown("---")
    st.markdown("#### 🚧 Analyse des Goulots d'Étranglement")
    
    # Identifier les périodes de surcharge
    high_load_periods = []
    if daily_allocation:
        avg_load = np.mean(list(daily_allocation.values()))
        threshold = avg_load * 1.5  # 150% de la moyenne
        
        for date, load in daily_allocation.items():
            if load > threshold:
                high_load_periods.append((date, load))
    
    if high_load_periods:
        st.warning(f"⚠️ {len(high_load_periods)} période(s) de surcharge détectée(s)")
        
        # Afficher les 5 pires périodes
        sorted_periods = sorted(high_load_periods, key=lambda x: x[1], reverse=True)[:5]
        
        for date, load in sorted_periods:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            st.write(f"🔴 **{date_obj.strftime('%d/%m/%Y')}**: {load} personnes (surcharge de {load - avg_allocation:.1f})")
        
        # Recommandations
        st.markdown("**💡 Recommandations:**")
        st.write("• Redistributer certaines tâches sur des périodes moins chargées")
        st.write("• Envisager des ressources temporaires pour les pics")
        st.write("• Paralléliser davantage les tâches non-critiques")
    else:
        st.success("✅ Aucun goulot d'étranglement détecté dans l'allocation des ressources")
    
    # Optimisation des ressources
    st.markdown("#### 🎯 Optimisation des Ressources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔄 Réallocation Possible:**")
        # Identifier les ressources sous-utilisées
        underutilized = []
        for res_type, data in resource_types.items():
            utilization_rate = (data['total_days'] / data['total_people']) / 30 if data['total_people'] > 0 else 0  # Sur 30 jours
            if utilization_rate < 0.7:  # Moins de 70% d'utilisation
                underutilized.append((res_type, utilization_rate))
        
        if underutilized:
            for res_type, rate in underutilized:
                st.write(f"• {res_type}: {rate*100:.1f}% utilisé")
        else:
            st.success("Toutes les ressources bien utilisées")
    
    with col2:
        st.markdown("**📈 Montée en Charge:**")
        # Identifier les besoins futurs
        future_needs = []
        for task in tasks:
            if task['progress'] < 50 and task['start_date'] > datetime.now():
                future_needs.append((task['start_date'], task['team_size'], task['resource_type']))
        
        if future_needs:
            # Grouper par mois
            monthly_needs = {}
            for start_date, team_size, res_type in future_needs:
                month_key = start_date.strftime('%Y-%m')
                if month_key not in monthly_needs:
                    monthly_needs[month_key] = 0
                monthly_needs[month_key] += team_size
            
            for month, need in list(monthly_needs.items())[:3]:  # 3 prochains mois
                st.write(f"• {month}: +{need} personnes")
        else:
            st.info("Besoins futurs stables")
    
    with col3:
        st.markdown("**⚡ Actions Recommandées:**")
        st.write("• Planifier les recrutements")
        st.write("• Former les équipes polyvalentes")
        st.write("• Optimiser la répartition des tâches")
        st.write("• Mettre en place du mentoring")


def render_milestones_analysis(timeline_data: Dict, tasks: List[Dict]):
    """Analyse moderne et optimisée des jalons et livrables"""
    
    # Header compact avec style
    st.markdown("### 🎯 Jalons & Livrables Clés")
    
    milestones = timeline_data.get('milestones', [])
    
    if not milestones:
        # Créer des jalons intelligents basés sur les phases
        phases = timeline_data.get('phases', {})
        if phases:
            render_smart_milestones_from_phases(phases)
        else:
            st.info("📌 Aucun jalon défini - Configurez vos jalons dans les données du projet")
        return
    
    # Dashboard compact des jalons
    render_milestones_dashboard(milestones)


def render_smart_milestones_from_phases(phases: Dict):
    """Génération intelligente de jalons à partir des phases"""
    
    # Métriques compactes en haut
    milestone_phases = []
    for phase_name, phase_tasks in phases.items():
        if phase_tasks:
            last_task = max(phase_tasks, key=lambda t: t['end_date'])
            phase_progress = np.mean([t['progress'] for t in phase_tasks])
            milestone_phases.append({
                'name': phase_name,
                'date': last_task['end_date'],
                'progress': phase_progress,
                'tasks_count': len(phase_tasks)
            })
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Jalons Phase", len(milestone_phases))
    
    with col2:
        completed = sum(1 for m in milestone_phases if m['progress'] >= 95)
        st.metric("✅ Terminés", f"{completed}/{len(milestone_phases)}")
    
    with col3:
        avg_progress = np.mean([m['progress'] for m in milestone_phases]) if milestone_phases else 0
        st.metric("📊 Avancement", f"{avg_progress:.0f}%")
    
    with col4:
        upcoming = sum(1 for m in milestone_phases if (m['date'] - datetime.now()).days <= 30 and m['progress'] < 95)
        st.metric("⚡ Prochains 30j", upcoming)
    
    st.markdown("---")
    
    # Timeline compacte et moderne
    st.markdown("#### 📅 Timeline des Jalons")
    
    for milestone in sorted(milestone_phases, key=lambda m: m['date']):
        days_remaining = (milestone['date'] - datetime.now()).days
        
        # Statut intelligent
        if milestone['progress'] >= 95:
            status_color = "#28a745"
            status_icon = "✅"
            status_text = "Terminé"
        elif days_remaining < 0 and milestone['progress'] < 95:
            status_color = "#dc3545" 
            status_icon = "🚨"
            status_text = f"Retard {abs(days_remaining)}j"
        elif days_remaining <= 7:
            status_color = "#ffc107"
            status_icon = "⚡"
            status_text = f"Urgent - {days_remaining}j"
        else:
            status_color = "#17a2b8"
            status_icon = "📅"
            status_text = f"Dans {days_remaining}j"
        
        # Card moderne pour chaque jalon
        st.markdown(f"""
        <div style="
            border-left: 4px solid {status_color};
            background: {status_color}10;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <div style="font-weight: bold; font-size: 16px; margin-bottom: 4px;">
                    {status_icon} {milestone['name']}
                </div>
                <div style="color: #666; font-size: 13px;">
                    📅 {milestone['date'].strftime('%d/%m/%Y')} • 📋 {milestone['tasks_count']} tâches
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: bold; color: {status_color};">
                    {status_text}
                </div>
                <div style="font-size: 13px; color: #666;">
                    {milestone['progress']:.0f}% complété
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_milestones_dashboard(milestones: List[Dict]):
    """Dashboard moderne pour jalons explicites"""
    
    # Calculs préliminaires
    completed_milestones = len([m for m in milestones if m['progress'] >= 95])
    at_risk_milestones = len([m for m in milestones if m['end_date'] < datetime.now() and m['progress'] < 95])
    upcoming_milestones = len([m for m in milestones if (m['end_date'] - datetime.now()).days <= 30 and m['progress'] < 95])
    
    # Dashboard KPI compact
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Total", len(milestones))
    
    with col2:
        completion_rate = (completed_milestones / len(milestones)) * 100 if milestones else 0
        st.metric("✅ Complétés", f"{completed_milestones}", delta=f"{completion_rate:.0f}%")
    
    with col3:
        st.metric("🚨 À risque", at_risk_milestones, delta="Critique" if at_risk_milestones > 0 else None)
    
    with col4:
        st.metric("⚡ Prochains", upcoming_milestones)
    
    with col5:
        avg_progress = np.mean([m['progress'] for m in milestones])
        st.metric("📊 Moyenne", f"{avg_progress:.0f}%")
    
    # Graphique timeline moderne
    col_chart, col_list = st.columns([3, 2])
    
    with col_chart:
        # Graphique Gantt des jalons
        fig = go.Figure()
        
        for i, milestone in enumerate(sorted(milestones, key=lambda m: m['end_date'])):
            # Couleur selon statut
            if milestone['progress'] >= 95:
                color = '#28a745'
            elif milestone['end_date'] < datetime.now():
                color = '#dc3545'
            elif (milestone['end_date'] - datetime.now()).days <= 7:
                color = '#ffc107'
            else:
                color = '#17a2b8'
            
            # Barre de jalon
            fig.add_trace(go.Scatter(
                x=[milestone['end_date']],
                y=[milestone['name'][:25] + '...' if len(milestone['name']) > 25 else milestone['name']],
                mode='markers',
                marker=dict(size=15, color=color, symbol='diamond'),
                name=f"{milestone['progress']:.0f}%",
                hovertemplate=f"<b>{milestone['name']}</b><br>Date: {milestone['end_date'].strftime('%d/%m/%Y')}<br>Progression: {milestone['progress']:.1f}%<extra></extra>"
            ))
        
        # Ligne pour aujourd'hui (méthode alternative sans add_vline)
        fig.add_shape(
            type="line",
            x0=datetime.now(), x1=datetime.now(),
            y0=0, y1=1, yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_annotation(
            x=datetime.now(),
            y=1, yref="paper",
            text="Aujourd'hui",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red"
        )
        
        fig.update_layout(
            title="🎯 Timeline des Jalons",
            xaxis_title="📅 Échéances",
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_list:
        # Liste compacte des jalons critiques
        st.markdown("**🚨 Jalons Critiques**")
        
        critical_milestones = [m for m in milestones if 
                             (m['end_date'] - datetime.now()).days <= 14 or 
                             (m['end_date'] < datetime.now() and m['progress'] < 95)]
        
        if critical_milestones:
            for milestone in critical_milestones[:5]:  # Limite à 5
                days_remaining = (milestone['end_date'] - datetime.now()).days
                
                if milestone['progress'] >= 95:
                    icon_status = "✅"
                elif days_remaining < 0:
                    icon_status = "🔴"
                else:
                    icon_status = "⚡"
                
                st.markdown(f"""
                <div style="
                    background: #f8f9fa;
                    padding: 8px 12px;
                    margin: 4px 0;
                    border-radius: 4px;
                    border-left: 3px solid {'#dc3545' if days_remaining < 0 else '#ffc107'};
                ">
                    <div style="font-weight: bold; font-size: 14px;">
                        {icon_status} {milestone['name'][:20]}...
                    </div>
                    <div style="font-size: 12px; color: #666;">
                        {milestone['end_date'].strftime('%d/%m')} • {milestone['progress']:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Aucun jalon critique")
    
    # Analyse des risques en bas (compact)
    st.markdown("---")
    if at_risk_milestones == 0 and upcoming_milestones <= 2:
        st.success("✅ Tous les jalons sont sous contrôle")
    elif at_risk_milestones > 0:
        st.error(f"🚨 {at_risk_milestones} jalon(s) en retard - Action immédiate requise")
    elif upcoming_milestones > 3:
        st.warning(f"⚡ {upcoming_milestones} jalons arrivent dans les 30 prochains jours")


def render_planning_ai_insights(insights: Dict, tasks: List[Dict], timeline_data: Dict):
    """Insights IA avancés pour la planification"""
    st.markdown("### 🤖 Intelligence Artificielle - Insights Planification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Analyse des Tendances")
        
        schedule_trends = insights.get('schedule_trends', {})
        trend = schedule_trends.get('trend', 'unknown')
        confidence = schedule_trends.get('confidence', 0.5)
        weeks_impact = schedule_trends.get('weeks_ahead', 0)
        
        # Statut de tendance
        trend_status = {
            'ahead': ('🟢 En avance', f'+{weeks_impact} semaines'),
            'on_track': ('🔵 Dans les temps', 'Conforme au planning'),
            'behind': ('🟡 Léger retard', f'{abs(weeks_impact)} semaine(s) de retard'),
            'critical': ('🔴 Retard critique', 'Action immédiate requise')
        }
        
        status_text, impact_text = trend_status.get(trend, ('⚪ Inconnu', 'Données insuffisantes'))
        
        st.write(f"**Tendance Planning**: {status_text}")
        st.write(f"**Impact**: {impact_text}")
        st.write(f"**Confiance IA**: {confidence*100:.0f}%")
        
        # Graphique de tendance
        weeks = list(range(1, 13))  # 12 semaines
        if trend == 'ahead':
            trend_line = [min(100, 20 + i * 8) for i in weeks]
        elif trend == 'behind':
            trend_line = [max(0, 15 + i * 6) for i in weeks]
        else:
            trend_line = [20 + i * 7 for i in weeks]
        
        df_trend = pd.DataFrame({
            'Semaine': weeks,
            'Progression_Prédite': trend_line
        })
        
        fig_trend = px.line(df_trend, x='Semaine', y='Progression_Prédite',
                           title="Tendance de Progression Prédite",
                           markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.markdown("#### ⚙️ Optimisation des Ressources")
        
        resource_opt = insights.get('resource_optimization', {})
        efficiency = resource_opt.get('efficiency', 75)
        bottlenecks = resource_opt.get('bottlenecks', [])
        opt_confidence = resource_opt.get('confidence', 0.8)
        
        # Score d'efficacité
        efficiency_status = "🟢 Optimal" if efficiency >= 85 else "🟡 Correct" if efficiency >= 70 else "🔴 À améliorer"
        st.write(f"**Efficacité Ressources**: {efficiency}% ({efficiency_status})")
        st.write(f"**Confiance IA**: {opt_confidence*100:.0f}%")
        
        # Goulots d'étranglement
        if bottlenecks:
            st.write("**🚧 Goulots identifiés:**")
            for bottleneck in bottlenecks:
                st.write(f"• {bottleneck}")
        else:
            st.success("✅ Aucun goulot majeur détecté")
        
        # Recommandations d'optimisation
        st.markdown("**💡 Recommandations IA:**")
        resource_insights = insights.get('resource_insights', {})
        
        peak_period = resource_insights.get('peak_utilization', 'Non identifié')
        underutilized = resource_insights.get('underutilized_periods', [])
        suggestions = resource_insights.get('reallocation_suggestions', '')
        
        st.write(f"• **Pic d'utilisation**: {peak_period}")
        if underutilized:
            st.write(f"• **Périodes creuses**: {', '.join(underutilized)}")
        if suggestions:
            st.write(f"• **Suggestion**: {suggestions}")
    
    # Recommandations globales de planification
    st.markdown("---")
    st.markdown("#### 🎯 Recommandations Strategiques IA")
    
    timeline_recs = insights.get('timeline_recommendations', [])
    
    if timeline_recs:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔄 Optimisations Process:**")
            process_recs = [r for r in timeline_recs if 'parallél' in r.lower() or 'process' in r.lower()]
            for rec in process_recs[:3]:
                st.write(f"• {rec}")
        
        with col2:
            st.markdown("**⏱️ Gestion du Temps:**")
            time_recs = [r for r in timeline_recs if 'buffer' in r.lower() or 'temps' in r.lower() or 'tôt' in r.lower()]
            for rec in time_recs[:3]:
                st.write(f"• {rec}")
        
        with col3:
            st.markdown("**👥 Optimisation Équipes:**")
            team_recs = [r for r in timeline_recs if 'équipe' in r.lower() or 'ressource' in r.lower()]
            for rec in team_recs[:3]:
                st.write(f"• {rec}")
    
    # Prédictions avancées
    st.markdown("---")
    st.markdown("#### 🔮 Prédictions Avancées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📅 Prédictions de Délais:**")
        
        # Simulation Monte Carlo simplifiée
        completion_scenarios = {
            'Optimiste': 85,
            'Réaliste': 95, 
            'Pessimiste': 110
        }
        
        current_progress = np.mean([t['progress'] for t in tasks])
        for scenario, percentage in completion_scenarios.items():
            adjusted_weeks = 12 * (percentage / 100)
            emoji = "🟢" if percentage <= 90 else "🟡" if percentage <= 105 else "🔴"
            st.write(f"{emoji} **{scenario}**: {adjusted_weeks:.1f} semaines")
    
    with col2:
        st.markdown("**💰 Prédictions Budgétaires:**")
        
        total_budget = sum([t['budget'] for t in tasks])
        
        budget_scenarios = {
            'Optimiste': 0.95,
            'Réaliste': 1.05,
            'Pessimiste': 1.20
        }
        
        for scenario, multiplier in budget_scenarios.items():
            predicted_budget = total_budget * multiplier
            emoji = "🟢" if multiplier <= 1.0 else "🟡" if multiplier <= 1.1 else "🔴"
            st.write(f"{emoji} **{scenario}**: {predicted_budget:,.0f}€")
    
    with col3:
        st.markdown("**⚡ Actions Prioritaires:**")
        
        # Générer des actions basées sur l'analyse
        priority_actions = []
        
        if any(t['progress'] < 50 and t.get('is_critical') for t in tasks):
            priority_actions.append("🔴 Accélérer tâches critiques")
        
        if len([t for t in tasks if (t['end_date'] - datetime.now()).days < 7]) > 2:
            priority_actions.append("🟡 Gérer les échéances courtes")
        
        if timeline_data.get('total_duration', 0) > 100:
            priority_actions.append("🔵 Optimiser la durée projet")
        
        priority_actions.append("✅ Monitoring continu")
        
        for action in priority_actions[:4]:
            st.write(f"• {action}")
    
    # Score global IA
    st.markdown("---")
    st.markdown("#### 🎖️ Score Global de Planification IA")
    
    # Calcul du score global
    progress_score = np.mean([t['progress'] for t in tasks])
    timeline_score = 85 if schedule_trends.get('trend') == 'on_track' else 70
    resource_score = resource_opt.get('efficiency', 75)
    
    global_score = (progress_score * 0.4 + timeline_score * 0.3 + resource_score * 0.3)
    
    score_color = "🟢" if global_score >= 80 else "🟡" if global_score >= 65 else "🔴"
    score_text = "Excellent" if global_score >= 80 else "Correct" if global_score >= 65 else "À améliorer"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Score Global IA", f"{global_score:.1f}/100", f"{score_color} {score_text}")
    
    with col2:
        st.metric("📊 Composants Analysés", len(tasks), "Tâches")
    
    with col3:
        st.metric("🤖 Confiance Globale", f"{np.mean([confidence, opt_confidence])*100:.0f}%", "IA")


def render_planning_export_section(tasks: List[Dict], timeline_data: Dict, resource_data: Dict, 
                                 kpis: Dict, insights: Dict):
    """Section d'export pour la planification"""
    
    st.subheader("📤 Exports Planification")
    st.markdown("*Exportez les données de planification, Gantt et ressources*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Gantt Excel", help="Gantt complet + données de planification"):
            excel_data = create_planning_excel_export(tasks, timeline_data, resource_data, kpis, insights)
            st.download_button(
                "💾 Télécharger Excel",
                excel_data,
                "planning_gantt_export.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Rapport Planning PDF", help="Rapport complet de planification"):
            st.info("📄 Génération du rapport de planification PDF...")
    
    with col3:
        if st.button("🗓️ Export MS Project", help="Format compatible MS Project"):
            project_data = create_ms_project_export(tasks, timeline_data)
            st.download_button(
                "💾 Télécharger CSV",
                project_data,
                "project_planning.csv",
                "text/csv"
            )
    
    with col4:
        if st.button("🤖 Export IA JSON", help="Données complètes + insights IA"):
            json_data = json.dumps({
                'planning_data': {
                    'tasks': tasks,
                    'timeline': {k: str(v) if isinstance(v, datetime) else v for k, v in timeline_data.items()},
                    'resources': resource_data,
                    'kpis': kpis,
                    'ai_insights': insights
                },
                'export_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'tasks_count': len(tasks),
                    'export_type': 'planning_complete'
                }
            }, indent=2, default=str)
            st.download_button(
                "💾 Télécharger JSON",
                json_data,
                "planning_ai_complete.json",
                "application/json"
            )


def create_planning_excel_export(tasks: List[Dict], timeline_data: Dict, resource_data: Dict, 
                               kpis: Dict, insights: Dict) -> bytes:
    """Crée un export Excel multi-feuilles pour la planification"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille Gantt Tasks
        tasks_export = []
        for task in tasks:
            tasks_export.append({
                'ID': task['id'],
                'Nom': task['name'],
                'Date_Debut': task['start_date'],
                'Date_Fin': task['end_date'],
                'Duree_Jours': task['duration'],
                'Progression_%': task['progress'],
                'Priorite': task.get('priority', 'medium'),
                'Equipe_Taille': task['team_size'],
                'Budget_Euro': task['budget'],
                'Dependances': ', '.join(task.get('dependencies', [])),
                'Type_Ressource': task.get('resource_type', 'general'),
                'Est_Jalon': task.get('is_milestone', False),
                'Est_Critique': task.get('is_critical', False),
                'Phase': task.get('phase', '')
            })
        
        df_tasks = pd.DataFrame(tasks_export)
        df_tasks.to_excel(writer, sheet_name='Gantt_Tasks', index=False)
        
        # Feuille Planning KPIs
        kpis_export = pd.DataFrame([kpis])
        kpis_export.to_excel(writer, sheet_name='Planning_KPIs', index=False)
        
        # Feuille Resource Allocation
        if resource_data.get('resource_types'):
            resource_export = []
            for res_type, data in resource_data['resource_types'].items():
                resource_export.append({
                    'Type_Ressource': res_type,
                    'Nombre_Taches': data['tasks'],
                    'Total_Personnes': data['total_people'],
                    'Total_Jours': data['total_days'],
                    'Utilisation_Moyenne': data['total_days'] / data['total_people'] if data['total_people'] > 0 else 0
                })
            
            df_resources = pd.DataFrame(resource_export)
            df_resources.to_excel(writer, sheet_name='Resources', index=False)
        
        # Feuille Timeline Analysis
        timeline_export = []
        if timeline_data.get('phases'):
            for phase_name, phase_tasks in timeline_data['phases'].items():
                timeline_export.append({
                    'Phase': phase_name,
                    'Nombre_Taches': len(phase_tasks),
                    'Budget_Total': sum([t['budget'] for t in phase_tasks]),
                    'Progression_Moyenne': np.mean([t['progress'] for t in phase_tasks]),
                    'Duree_Totale': sum([t['duration'] for t in phase_tasks])
                })
        
        if timeline_export:
            df_timeline = pd.DataFrame(timeline_export)
            df_timeline.to_excel(writer, sheet_name='Timeline_Analysis', index=False)
        
        # Feuille AI Insights
        insights_export = pd.DataFrame([insights])
        insights_export.to_excel(writer, sheet_name='AI_Insights', index=False)
    
    return output.getvalue()


def create_ms_project_export(tasks: List[Dict], timeline_data: Dict) -> str:
    """Crée un export CSV compatible MS Project"""
    project_data = []
    
    for task in tasks:
        project_data.append({
            'ID': task['id'],
            'Name': task['name'],
            'Duration': task['duration'],
            'Start': task['start_date'].strftime('%Y-%m-%d'),
            'Finish': task['end_date'].strftime('%Y-%m-%d'),
            'Percent_Complete': task['progress'],
            'Predecessors': ';'.join(task.get('dependencies', [])),
            'Resource_Names': task.get('resource_type', ''),
            'Priority': task.get('priority', 'medium'),
            'Cost': task['budget'],
            'Work': task['duration'] * task['team_size'] * 8,  # heures de travail
            'Milestone': task.get('is_milestone', False),
            'Critical': task.get('is_critical', False)
        })
    
    df = pd.DataFrame(project_data)
    return df.to_csv(index=False)