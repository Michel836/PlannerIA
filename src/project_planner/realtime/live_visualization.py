"""
Live Plan Generation Visualization
Visualisation en temps réel de la génération de plans avec animations
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

from .streaming_engine import (
    get_streaming_engine,
    StreamEvent,
    StreamEventType,
    AgentStatus
)

class LivePlanVisualization:
    """Visualisation live de la génération de plans"""
    
    def __init__(self):
        self.streaming_engine = get_streaming_engine()
        self.plan_data_buffer = []
        self.animation_frames = []
        self.current_frame = 0
        
    def render_live_generation_dashboard(self, project_data: Dict[str, Any]):
        """Rendre le dashboard de génération live"""
        
        st.markdown("# 🔴 Génération de Plan en Direct")
        st.markdown("---")
        
        # Conteneurs pour les métriques temps réel
        metrics_container = st.container()
        
        # Conteneur principal divisé en colonnes
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualisation principale
            main_viz_container = st.container()
            
        with col2:
            # Status des agents
            agents_container = st.container()
            
        # Logs en temps réel
        logs_container = st.container()
        
        # Initialiser les placeholders
        if 'live_viz_initialized' not in st.session_state:
            st.session_state['live_viz_initialized'] = True
            st.session_state['metrics_placeholder'] = metrics_container.empty()
            st.session_state['main_viz_placeholder'] = main_viz_container.empty()
            st.session_state['agents_placeholder'] = agents_container.empty()
            st.session_state['logs_placeholder'] = logs_container.empty()
            
        # Démarrer la mise à jour en continu
        self._start_live_updates()
        
    def _start_live_updates(self):
        """Démarrer les mises à jour en temps réel"""
        
        # Obtenir les données en temps réel
        streaming_status = self.streaming_engine.get_streaming_status()
        global_metrics = self.streaming_engine.get_global_metrics()
        agent_statuses = self.streaming_engine.get_all_agent_statuses()
        
        # Mettre à jour les métriques
        self._update_metrics_display(global_metrics)
        
        # Mettre à jour la visualisation principale
        self._update_main_visualization(streaming_status, global_metrics)
        
        # Mettre à jour le status des agents
        self._update_agents_display(agent_statuses)
        
        # Mettre à jour les logs
        self._update_logs_display()
        
        # Auto-refresh toutes les 500ms si streaming actif
        if streaming_status['streaming_active']:
            time.sleep(0.5)
            st.rerun()
            
    def _update_metrics_display(self, metrics: Dict[str, Any]):
        """Mettre à jour l'affichage des métriques"""
        
        with st.session_state['metrics_placeholder'].container():
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                completion = metrics.get('completion_percentage', 0)
                st.metric(
                    "🎯 Progression", 
                    f"{completion:.1f}%",
                    delta=f"{completion:.1f}%" if completion > 0 else None
                )
                
            with col2:
                active_agents = metrics.get('active_agents', 0)
                st.metric(
                    "🤖 Agents Actifs", 
                    active_agents,
                    delta=active_agents if active_agents > 0 else None
                )
                
            with col3:
                events_per_sec = metrics.get('events_per_second', 0)
                st.metric(
                    "📊 Événements/sec", 
                    f"{events_per_sec:.1f}",
                    delta=f"{events_per_sec:.1f}" if events_per_sec > 0 else None
                )
                
            with col4:
                time_remaining = metrics.get('estimated_time_remaining', 0)
                if time_remaining > 0:
                    minutes = int(time_remaining // 60)
                    seconds = int(time_remaining % 60)
                    time_str = f"{minutes}m {seconds}s"
                else:
                    time_str = "Calcul..."
                    
                st.metric(
                    "⏱️ Temps Restant", 
                    time_str
                )
                
            with col5:
                total_events = metrics.get('total_events', 0)
                st.metric(
                    "📈 Total Événements", 
                    total_events,
                    delta=total_events if total_events > 0 else None
                )
                
    def _update_main_visualization(self, streaming_status: Dict[str, Any], metrics: Dict[str, Any]):
        """Mettre à jour la visualisation principale"""
        
        with st.session_state['main_viz_placeholder'].container():
            
            # Graphique de progression globale
            st.markdown("### 📊 Progression Globale")
            
            completion = metrics.get('completion_percentage', 0)
            
            # Gauge de progression avec Plotly
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = completion,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Progression Globale (%)"},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"},
                        {'range': [50, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Timeline de génération avec agents
            st.markdown("### ⏰ Timeline de Génération")
            
            agent_statuses = streaming_status.get('agent_statuses', {})
            
            if agent_statuses:
                # Créer données pour timeline
                timeline_data = []
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
                
                for i, (agent_id, status_dict) in enumerate(agent_statuses.items()):
                    start_time = datetime.fromisoformat(status_dict['start_time'])
                    last_update = datetime.fromisoformat(status_dict['last_update'])
                    
                    timeline_data.append({
                        'Agent': status_dict['agent_name'],
                        'Start': start_time,
                        'Finish': last_update,
                        'Progress': status_dict['progress'],
                        'Status': status_dict['status'],
                        'Task': status_dict['current_task'],
                        'Color': colors[i % len(colors)]
                    })
                
                # Graphique Gantt-like pour les agents
                fig_timeline = go.Figure()
                
                for i, agent_data in enumerate(timeline_data):
                    # Barre de progression
                    duration = (agent_data['Finish'] - agent_data['Start']).total_seconds()
                    progress_duration = duration * (agent_data['Progress'] / 100)
                    
                    # Barre complète (gris)
                    fig_timeline.add_trace(go.Scatter(
                        x=[agent_data['Start'], agent_data['Finish']],
                        y=[i, i],
                        mode='lines',
                        line=dict(color='lightgray', width=20),
                        name=f"{agent_data['Agent']} (Total)",
                        showlegend=False
                    ))
                    
                    # Barre de progression (colorée)
                    progress_end = agent_data['Start'] + timedelta(seconds=progress_duration)
                    fig_timeline.add_trace(go.Scatter(
                        x=[agent_data['Start'], progress_end],
                        y=[i, i],
                        mode='lines',
                        line=dict(color=agent_data['Color'], width=20),
                        name=agent_data['Agent'],
                        hovertemplate=f"<b>{agent_data['Agent']}</b><br>" +
                                    f"Status: {agent_data['Status']}<br>" +
                                    f"Progress: {agent_data['Progress']:.1f}%<br>" +
                                    f"Task: {agent_data['Task']}<extra></extra>"
                    ))
                
                fig_timeline.update_layout(
                    title="Timeline des Agents IA",
                    xaxis_title="Temps",
                    yaxis_title="Agents",
                    height=300,
                    yaxis=dict(
                        tickvals=list(range(len(timeline_data))),
                        ticktext=[agent['Agent'] for agent in timeline_data]
                    ),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
                
            else:
                st.info("⏳ En attente du démarrage des agents...")
                
            # Graphique des événements par seconde
            st.markdown("### 📈 Activité Temps Réel")
            
            # Simuler des données d'activité (en production, ça viendrait du streaming)
            current_time = datetime.now()
            time_points = [current_time - timedelta(seconds=i*10) for i in range(30, 0, -1)]
            
            # Générer données simulées basées sur les métriques réelles
            events_per_sec = metrics.get('events_per_second', 0)
            activity_data = []
            
            for t in time_points:
                # Simuler variation d'activité
                base_activity = events_per_sec
                variation = np.random.normal(0, base_activity * 0.2) if base_activity > 0 else 0
                activity_value = max(0, base_activity + variation)
                
                activity_data.append({
                    'Time': t,
                    'Events_per_sec': activity_value,
                    'Active_Agents': metrics.get('active_agents', 0) + np.random.randint(-1, 2)
                })
                
            activity_df = pd.DataFrame(activity_data)
            
            if not activity_df.empty:
                fig_activity = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Événements par Seconde', 'Agents Actifs'),
                    vertical_spacing=0.1
                )
                
                # Événements par seconde
                fig_activity.add_trace(
                    go.Scatter(
                        x=activity_df['Time'],
                        y=activity_df['Events_per_sec'],
                        mode='lines+markers',
                        name='Événements/sec',
                        line=dict(color='blue'),
                        fill='tonexty'
                    ),
                    row=1, col=1
                )
                
                # Agents actifs
                fig_activity.add_trace(
                    go.Scatter(
                        x=activity_df['Time'],
                        y=activity_df['Active_Agents'],
                        mode='lines+markers',
                        name='Agents Actifs',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
                
                fig_activity.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                fig_activity.update_xaxes(title_text="Temps", row=2, col=1)
                fig_activity.update_yaxes(title_text="Événements/sec", row=1, col=1)
                fig_activity.update_yaxes(title_text="Nombre", row=2, col=1)
                
                st.plotly_chart(fig_activity, use_container_width=True)
                
    def _update_agents_display(self, agent_statuses: Dict[str, AgentStatus]):
        """Mettre à jour l'affichage des agents"""
        
        with st.session_state['agents_placeholder'].container():
            st.markdown("### 🤖 Status des Agents")
            
            if agent_statuses:
                for agent_id, status in agent_statuses.items():
                    
                    # Couleur selon le status
                    status_colors = {
                        'starting': '🟡',
                        'active': '🟢', 
                        'completed': '✅',
                        'error': '❌',
                        'paused': '⏸️'
                    }
                    
                    status_icon = status_colors.get(status.status, '⚪')
                    
                    with st.expander(f"{status_icon} {status.agent_name} ({status.progress:.1f}%)", 
                                    expanded=(status.status == 'active')):
                        
                        # Barre de progression
                        progress_bar = st.progress(status.progress / 100)
                        
                        # Informations détaillées
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Statut:** {status.status}")
                            st.markdown(f"**Tâche actuelle:** {status.current_task}")
                            
                        with col2:
                            elapsed = (status.last_update - status.start_time).total_seconds()
                            st.markdown(f"**Temps écoulé:** {elapsed:.0f}s")
                            st.markdown(f"**Dernière MAJ:** {status.last_update.strftime('%H:%M:%S')}")
                            
                        # Messages récents
                        if status.messages:
                            st.markdown("**Messages récents:**")
                            for msg in status.messages[-3:]:  # 3 derniers messages
                                st.markdown(f"- {msg}")
                                
                        # Métriques spécifiques à l'agent
                        if status.metrics:
                            st.markdown("**Métriques:**")
                            metrics_cols = st.columns(min(len(status.metrics), 3))
                            
                            for i, (key, value) in enumerate(status.metrics.items()):
                                if i < len(metrics_cols):
                                    with metrics_cols[i]:
                                        st.metric(key.replace('_', ' ').title(), value)
            else:
                st.info("Aucun agent actif pour le moment")
                
    def _update_logs_display(self):
        """Mettre à jour l'affichage des logs"""
        
        with st.session_state['logs_placeholder'].container():
            
            with st.expander("📋 Logs Temps Réel", expanded=False):
                
                # Container avec scroll pour les logs
                log_container = st.container()
                
                # Simuler des logs récents (en production, viendrait du streaming)
                recent_logs = [
                    {"timestamp": datetime.now() - timedelta(seconds=5), "level": "INFO", "message": "Agent Supervisor démarré"},
                    {"timestamp": datetime.now() - timedelta(seconds=3), "level": "DEBUG", "message": "Analyse des requirements en cours..."},
                    {"timestamp": datetime.now() - timedelta(seconds=1), "level": "INFO", "message": "Génération WBS: 45% complété"},
                    {"timestamp": datetime.now(), "level": "SUCCESS", "message": "Estimation des coûts terminée"}
                ]
                
                for log in recent_logs[-10:]:  # 10 derniers logs
                    level_colors = {
                        'DEBUG': '🔍',
                        'INFO': 'ℹ️',
                        'SUCCESS': '✅',
                        'WARNING': '⚠️',
                        'ERROR': '❌'
                    }
                    
                    icon = level_colors.get(log['level'], 'ℹ️')
                    timestamp_str = log['timestamp'].strftime('%H:%M:%S')
                    
                    log_container.markdown(f"`{timestamp_str}` {icon} {log['message']}")
                    
    def render_live_plan_preview(self, current_plan_data: Dict[str, Any]):
        """Rendre un aperçu live du plan en génération"""
        
        st.markdown("### 📄 Aperçu du Plan en Cours")
        
        if current_plan_data:
            # Onglets pour différentes sections du plan
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 Vue d'ensemble", "📋 Tâches", "👥 Équipe", "💰 Budget"])
            
            with tab1:
                self._render_plan_overview(current_plan_data)
                
            with tab2:
                self._render_plan_tasks(current_plan_data)
                
            with tab3:
                self._render_plan_team(current_plan_data)
                
            with tab4:
                self._render_plan_budget(current_plan_data)
                
        else:
            # Animation de chargement
            with st.container():
                st.markdown("#### 🔄 Génération en cours...")
                
                # Barre de progression animée
                progress_placeholder = st.empty()
                
                # Simuler progression
                if 'generation_progress' not in st.session_state:
                    st.session_state['generation_progress'] = 0
                    
                progress = st.session_state['generation_progress']
                progress_placeholder.progress(progress)
                
                # Incrémenter la progression
                if progress < 100:
                    st.session_state['generation_progress'] = min(100, progress + 5)
                    time.sleep(0.1)
                    st.rerun()
                    
    def _render_plan_overview(self, plan_data: Dict[str, Any]):
        """Rendre la vue d'ensemble du plan"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Métriques Clés**")
            if 'duration' in plan_data:
                st.metric("Durée", f"{plan_data['duration']} jours")
            if 'budget' in plan_data:
                st.metric("Budget", f"{plan_data['budget']:,.0f} €")
            if 'team_size' in plan_data:
                st.metric("Équipe", f"{plan_data['team_size']} personnes")
                
        with col2:
            st.markdown("**🎯 Objectifs**")
            objectives = plan_data.get('objectives', [])
            for obj in objectives[:3]:  # Limiter à 3
                st.markdown(f"- {obj}")
                
    def _render_plan_tasks(self, plan_data: Dict[str, Any]):
        """Rendre les tâches du plan"""
        
        tasks = plan_data.get('tasks', [])
        
        if tasks:
            # Tableau des tâches
            task_df = pd.DataFrame([
                {
                    'Tâche': task.get('name', ''),
                    'Durée': f"{task.get('duration', 0)} j",
                    'Status': task.get('status', 'Planifiée'),
                    'Assigné': task.get('assigned_to', 'Non assigné')
                }
                for task in tasks[:10]  # Limiter à 10 tâches
            ])
            
            st.dataframe(task_df, use_container_width=True)
        else:
            st.info("Tâches en cours de génération...")
            
    def _render_plan_team(self, plan_data: Dict[str, Any]):
        """Rendre l'équipe du plan"""
        
        team = plan_data.get('team', [])
        
        if team:
            for member in team:
                with st.container():
                    st.markdown(f"**{member.get('name', 'Membre')}**")
                    st.markdown(f"Role: {member.get('role', 'Non défini')}")
                    st.markdown("---")
        else:
            st.info("Composition de l'équipe en cours de définition...")
            
    def _render_plan_budget(self, plan_data: Dict[str, Any]):
        """Rendre le budget du plan"""
        
        budget_breakdown = plan_data.get('budget_breakdown', {})
        
        if budget_breakdown:
            # Graphique en camembert
            labels = list(budget_breakdown.keys())
            values = list(budget_breakdown.values())
            
            fig = px.pie(
                values=values,
                names=labels,
                title="Répartition Budgétaire"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Analyse budgétaire en cours...")
            
    def create_animated_progress_chart(self, agent_statuses: Dict[str, AgentStatus]) -> go.Figure:
        """Créer un graphique de progression animé"""
        
        if not agent_statuses:
            return go.Figure()
            
        # Préparer les données pour l'animation
        agent_names = [status.agent_name for status in agent_statuses.values()]
        progresses = [status.progress for status in agent_statuses.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=agent_names,
                y=progresses,
                marker_color=['green' if p >= 100 else 'orange' if p >= 50 else 'red' for p in progresses],
                text=[f"{p:.1f}%" for p in progresses],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Progression des Agents en Temps Réel",
            xaxis_title="Agents",
            yaxis_title="Progression (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        return fig

# Instance globale
def get_live_visualization():
    """Obtenir l'instance de visualisation live"""
    if 'live_visualization' not in st.session_state:
        st.session_state['live_visualization'] = LivePlanVisualization()
    return st.session_state['live_visualization']

# Fonctions utilitaires
def render_live_generation_dashboard(project_data: Dict[str, Any]):
    """Fonction utilitaire pour rendre le dashboard live"""
    viz = get_live_visualization()
    viz.render_live_generation_dashboard(project_data)

def render_live_plan_preview(plan_data: Dict[str, Any]):
    """Fonction utilitaire pour rendre l'aperçu live du plan"""
    viz = get_live_visualization()
    viz.render_live_plan_preview(plan_data)