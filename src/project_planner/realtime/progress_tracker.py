"""
Advanced Progress Indicators and Status Tracking
Indicateurs de progression avancés avec suivi détaillé du statut
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from .streaming_engine import (
    get_streaming_engine,
    StreamEvent,
    StreamEventType,
    AgentStatus
)

logger = logging.getLogger(__name__)

class ProgressType(Enum):
    """Types d'indicateurs de progression"""
    LINEAR = "linear"
    CIRCULAR = "circular"
    STEPPED = "stepped"
    ANIMATED = "animated"
    GAUGE = "gauge"
    TIMELINE = "timeline"

class StatusLevel(Enum):
    """Niveaux de statut"""
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    PROCESSING = "processing"

@dataclass
class ProgressStep:
    """Étape de progression"""
    step_id: str
    title: str
    description: str
    progress: float  # 0-100
    status: StatusLevel
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    duration: Optional[float] = None
    substeps: List['ProgressStep'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'title': self.title,
            'description': self.description,
            'progress': self.progress,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'completion_time': self.completion_time.isoformat() if self.completion_time else None,
            'duration': self.duration,
            'substeps': [substep.to_dict() for substep in self.substeps],
            'metadata': self.metadata
        }

@dataclass
class ProgressIndicator:
    """Indicateur de progression avancé"""
    indicator_id: str
    title: str
    progress_type: ProgressType
    current_progress: float
    max_progress: float
    status: StatusLevel
    steps: List[ProgressStep]
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_completion_percentage(self) -> float:
        """Calculer le pourcentage de completion"""
        if self.max_progress == 0:
            return 0
        return min(100, (self.current_progress / self.max_progress) * 100)
        
    def get_active_step(self) -> Optional[ProgressStep]:
        """Obtenir l'étape active actuelle"""
        for step in self.steps:
            if step.status == StatusLevel.PROCESSING:
                return step
        return None
        
    def update_progress(self, new_progress: float, step_id: str = None):
        """Mettre à jour la progression"""
        self.current_progress = min(self.max_progress, new_progress)
        
        if step_id:
            for step in self.steps:
                if step.step_id == step_id:
                    step.progress = min(100, (new_progress / self.max_progress) * 100)
                    break

class AdvancedProgressTracker:
    """Système de suivi de progression avancé"""
    
    def __init__(self):
        self.streaming_engine = get_streaming_engine()
        self.progress_indicators = {}
        self.status_history = []
        self.notification_queue = queue.Queue()
        self.update_callbacks = {}
        self.animation_states = {}
        
        # Démarrer le thread de mise à jour
        self._start_update_thread()
        
    def _start_update_thread(self):
        """Démarrer le thread de mise à jour en arrière-plan"""
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
    def _update_loop(self):
        """Boucle de mise à jour en continu"""
        while True:
            try:
                # Mettre à jour les indicateurs basés sur le streaming
                self._sync_with_streaming_engine()
                
                # Traiter les notifications
                self._process_notifications()
                
                # Attendre avant la prochaine mise à jour
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de mise à jour: {e}")
                time.sleep(1)
                
    def _sync_with_streaming_engine(self):
        """Synchroniser avec le moteur de streaming"""
        
        streaming_status = self.streaming_engine.get_streaming_status()
        agent_statuses = self.streaming_engine.get_all_agent_statuses()
        
        # Mettre à jour les indicateurs d'agents
        for agent_id, status in agent_statuses.items():
            if agent_id not in self.progress_indicators:
                self.create_agent_progress_indicator(agent_id, status)
            else:
                self.update_agent_progress_indicator(agent_id, status)
                
    def create_agent_progress_indicator(self, agent_id: str, agent_status: AgentStatus):
        """Créer un indicateur de progression pour un agent"""
        
        # Définir les étapes typiques d'un agent
        steps = [
            ProgressStep(
                step_id=f"{agent_id}_init",
                title="Initialisation",
                description="Démarrage de l'agent",
                progress=100 if agent_status.status != 'starting' else 50,
                status=StatusLevel.SUCCESS if agent_status.status != 'starting' else StatusLevel.PROCESSING
            ),
            ProgressStep(
                step_id=f"{agent_id}_processing",
                title="Traitement",
                description=agent_status.current_task,
                progress=agent_status.progress,
                status=StatusLevel.PROCESSING if agent_status.status == 'active' else StatusLevel.INFO
            ),
            ProgressStep(
                step_id=f"{agent_id}_completion",
                title="Finalisation",
                description="Finalisation des résultats",
                progress=100 if agent_status.status == 'completed' else 0,
                status=StatusLevel.SUCCESS if agent_status.status == 'completed' else StatusLevel.INFO
            )
        ]
        
        # Déterminer le statut global
        if agent_status.status == 'error':
            global_status = StatusLevel.ERROR
        elif agent_status.status == 'completed':
            global_status = StatusLevel.SUCCESS
        elif agent_status.status == 'active':
            global_status = StatusLevel.PROCESSING
        else:
            global_status = StatusLevel.INFO
            
        indicator = ProgressIndicator(
            indicator_id=agent_id,
            title=agent_status.agent_name,
            progress_type=ProgressType.STEPPED,
            current_progress=agent_status.progress,
            max_progress=100,
            status=global_status,
            steps=steps,
            start_time=agent_status.start_time
        )
        
        self.progress_indicators[agent_id] = indicator
        
    def update_agent_progress_indicator(self, agent_id: str, agent_status: AgentStatus):
        """Mettre à jour un indicateur de progression d'agent"""
        
        if agent_id not in self.progress_indicators:
            return
            
        indicator = self.progress_indicators[agent_id]
        indicator.current_progress = agent_status.progress
        
        # Mettre à jour le statut global
        if agent_status.status == 'error':
            indicator.status = StatusLevel.ERROR
        elif agent_status.status == 'completed':
            indicator.status = StatusLevel.SUCCESS
        elif agent_status.status == 'active':
            indicator.status = StatusLevel.PROCESSING
            
        # Mettre à jour les étapes
        for step in indicator.steps:
            if step.step_id == f"{agent_id}_processing":
                step.description = agent_status.current_task
                step.progress = agent_status.progress
                
                if agent_status.status == 'active':
                    step.status = StatusLevel.PROCESSING
                elif agent_status.status == 'completed':
                    step.status = StatusLevel.SUCCESS
                    step.progress = 100
                elif agent_status.status == 'error':
                    step.status = StatusLevel.ERROR
                    
    def _process_notifications(self):
        """Traiter les notifications en file d'attente"""
        try:
            while not self.notification_queue.empty():
                notification = self.notification_queue.get_nowait()
                self._handle_notification(notification)
        except queue.Empty:
            pass
            
    def _handle_notification(self, notification: Dict[str, Any]):
        """Traiter une notification"""
        # Logique de traitement des notifications
        notification_type = notification.get('type')
        
        if notification_type == 'progress_update':
            self._handle_progress_notification(notification)
        elif notification_type == 'status_change':
            self._handle_status_notification(notification)
            
    def render_progress_dashboard(self, container_key: str = "main_progress"):
        """Rendre le dashboard de progression"""
        
        st.markdown("# 📊 Suivi de Progression Temps Réel")
        st.markdown("---")
        
        if not self.progress_indicators:
            st.info("🔄 Aucune progression en cours. Démarrez une génération de plan pour voir les indicateurs.")
            return
            
        # Vue d'ensemble globale
        self._render_global_overview()
        
        # Indicateurs individuels
        st.markdown("## 🎯 Progression Détaillée")
        
        for indicator_id, indicator in self.progress_indicators.items():
            self._render_progress_indicator(indicator, indicator_id)
            
    def _render_global_overview(self):
        """Rendre la vue d'ensemble globale"""
        
        # Calculer les métriques globales
        total_indicators = len(self.progress_indicators)
        completed_indicators = len([i for i in self.progress_indicators.values() if i.status == StatusLevel.SUCCESS])
        processing_indicators = len([i for i in self.progress_indicators.values() if i.status == StatusLevel.PROCESSING])
        error_indicators = len([i for i in self.progress_indicators.values() if i.status == StatusLevel.ERROR])
        
        # Progression globale moyenne
        if total_indicators > 0:
            avg_progress = sum(i.get_completion_percentage() for i in self.progress_indicators.values()) / total_indicators
        else:
            avg_progress = 0
            
        # Métriques en colonnes
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Progression Globale", f"{avg_progress:.1f}%")
            
        with col2:
            st.metric("✅ Terminés", completed_indicators)
            
        with col3:
            st.metric("🔄 En Cours", processing_indicators)
            
        with col4:
            st.metric("❌ Erreurs", error_indicators)
            
        with col5:
            st.metric("📊 Total", total_indicators)
            
        # Graphique de progression globale
        self._render_global_progress_chart(avg_progress)
        
    def _render_global_progress_chart(self, avg_progress: float):
        """Rendre le graphique de progression globale"""
        
        # Gauge de progression
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_progress,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Progression Globale (%)"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"}, 
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_progress_indicator(self, indicator: ProgressIndicator, indicator_id: str):
        """Rendre un indicateur de progression spécifique"""
        
        # Container pour cet indicateur
        with st.container():
            
            # En-tête avec statut
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                status_icons = {
                    StatusLevel.SUCCESS: "✅",
                    StatusLevel.PROCESSING: "🔄", 
                    StatusLevel.WARNING: "⚠️",
                    StatusLevel.ERROR: "❌",
                    StatusLevel.INFO: "ℹ️"
                }
                
                icon = status_icons.get(indicator.status, "📊")
                st.markdown(f"### {icon} {indicator.title}")
                
            with col2:
                completion = indicator.get_completion_percentage()
                st.metric("Progression", f"{completion:.1f}%")
                
            with col3:
                # Temps écoulé
                if indicator.start_time:
                    elapsed = (datetime.now() - indicator.start_time).total_seconds()
                    st.metric("Temps", f"{elapsed:.0f}s")
                    
            # Rendu selon le type de progression
            if indicator.progress_type == ProgressType.STEPPED:
                self._render_stepped_progress(indicator)
            elif indicator.progress_type == ProgressType.LINEAR:
                self._render_linear_progress(indicator)
            elif indicator.progress_type == ProgressType.CIRCULAR:
                self._render_circular_progress(indicator)
            elif indicator.progress_type == ProgressType.GAUGE:
                self._render_gauge_progress(indicator)
            elif indicator.progress_type == ProgressType.TIMELINE:
                self._render_timeline_progress(indicator)
            else:
                # Par défaut: stepped
                self._render_stepped_progress(indicator)
                
            st.markdown("---")
            
    def _render_stepped_progress(self, indicator: ProgressIndicator):
        """Rendre une progression par étapes"""
        
        for i, step in enumerate(indicator.steps):
            col1, col2, col3 = st.columns([1, 4, 1])
            
            with col1:
                # Icône d'étape
                if step.status == StatusLevel.SUCCESS:
                    st.markdown("✅")
                elif step.status == StatusLevel.PROCESSING:
                    st.markdown("🔄")
                elif step.status == StatusLevel.ERROR:
                    st.markdown("❌")
                else:
                    st.markdown(f"{i+1}")
                    
            with col2:
                st.markdown(f"**{step.title}**")
                st.markdown(f"<small>{step.description}</small>", unsafe_allow_html=True)
                
                # Barre de progression pour l'étape
                if step.progress > 0:
                    progress_bar = st.progress(step.progress / 100)
                    
            with col3:
                st.markdown(f"{step.progress:.0f}%")
                
            # Sous-étapes si présentes
            if step.substeps:
                with st.expander("Voir détails", expanded=False):
                    for substep in step.substeps:
                        st.markdown(f"- {substep.title}: {substep.progress:.0f}%")
                        
    def _render_linear_progress(self, indicator: ProgressIndicator):
        """Rendre une progression linéaire"""
        
        completion = indicator.get_completion_percentage()
        
        # Barre de progression principale
        st.progress(completion / 100)
        
        # Détails textuels
        active_step = indicator.get_active_step()
        if active_step:
            st.markdown(f"**Étape actuelle:** {active_step.title}")
            st.markdown(f"**Description:** {active_step.description}")
            
    def _render_circular_progress(self, indicator: ProgressIndicator):
        """Rendre une progression circulaire"""
        
        completion = indicator.get_completion_percentage()
        
        # Graphique circulaire avec Plotly
        fig = go.Figure(go.Pie(
            values=[completion, 100 - completion],
            names=['Terminé', 'Restant'],
            hole=0.7,
            marker_colors=['green', 'lightgray'],
            showlegend=False,
            textinfo='none'
        ))
        
        # Ajouter le pourcentage au centre
        fig.add_annotation(
            text=f"{completion:.1f}%",
            x=0.5, y=0.5,
            font_size=20,
            font_color="black",
            showarrow=False
        )
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_gauge_progress(self, indicator: ProgressIndicator):
        """Rendre une progression en gauge"""
        
        completion = indicator.get_completion_percentage()
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = completion,
            title = {'text': indicator.title},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_timeline_progress(self, indicator: ProgressIndicator):
        """Rendre une progression en timeline"""
        
        # Créer les données de timeline
        timeline_data = []
        
        for i, step in enumerate(indicator.steps):
            start_time = step.start_time or indicator.start_time
            end_time = step.completion_time or datetime.now()
            
            timeline_data.append({
                'Task': step.title,
                'Start': start_time,
                'Finish': end_time,
                'Progress': step.progress,
                'Status': step.status.value
            })
            
        if timeline_data:
            # Graphique Gantt-like
            fig = go.Figure()
            
            colors = {
                'success': 'green',
                'processing': 'orange',
                'error': 'red',
                'info': 'blue',
                'warning': 'yellow'
            }
            
            for i, data in enumerate(timeline_data):
                color = colors.get(data['Status'], 'gray')
                
                fig.add_trace(go.Scatter(
                    x=[data['Start'], data['Finish']],
                    y=[i, i],
                    mode='lines',
                    line=dict(color=color, width=10),
                    name=data['Task'],
                    hovertemplate=f"<b>{data['Task']}</b><br>" +
                                f"Progress: {data['Progress']:.1f}%<br>" +
                                f"Status: {data['Status']}<extra></extra>"
                ))
                
            fig.update_layout(
                title="Timeline de Progression",
                xaxis_title="Temps",
                yaxis_title="Étapes",
                height=300,
                yaxis=dict(
                    tickvals=list(range(len(timeline_data))),
                    ticktext=[data['Task'] for data in timeline_data]
                ),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    def create_custom_progress_indicator(self, 
                                       indicator_id: str,
                                       title: str,
                                       progress_type: ProgressType = ProgressType.LINEAR,
                                       max_progress: float = 100) -> ProgressIndicator:
        """Créer un indicateur de progression personnalisé"""
        
        indicator = ProgressIndicator(
            indicator_id=indicator_id,
            title=title,
            progress_type=progress_type,
            current_progress=0,
            max_progress=max_progress,
            status=StatusLevel.INFO,
            steps=[],
            start_time=datetime.now()
        )
        
        self.progress_indicators[indicator_id] = indicator
        return indicator
        
    def update_progress(self, 
                       indicator_id: str,
                       progress: float,
                       status: StatusLevel = None,
                       message: str = None):
        """Mettre à jour la progression d'un indicateur"""
        
        if indicator_id not in self.progress_indicators:
            return
            
        indicator = self.progress_indicators[indicator_id]
        indicator.current_progress = min(indicator.max_progress, progress)
        
        if status:
            indicator.status = status
            
        # Ajouter notification si message fourni
        if message:
            self.notification_queue.put({
                'type': 'progress_update',
                'indicator_id': indicator_id,
                'progress': progress,
                'message': message,
                'timestamp': datetime.now()
            })
            
    def get_progress_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé de toutes les progressions"""
        
        summary = {
            'total_indicators': len(self.progress_indicators),
            'completed_count': 0,
            'processing_count': 0,
            'error_count': 0,
            'average_progress': 0,
            'indicators': {}
        }
        
        if not self.progress_indicators:
            return summary
            
        total_progress = 0
        
        for indicator_id, indicator in self.progress_indicators.items():
            completion = indicator.get_completion_percentage()
            total_progress += completion
            
            if indicator.status == StatusLevel.SUCCESS:
                summary['completed_count'] += 1
            elif indicator.status == StatusLevel.PROCESSING:
                summary['processing_count'] += 1
            elif indicator.status == StatusLevel.ERROR:
                summary['error_count'] += 1
                
            summary['indicators'][indicator_id] = {
                'title': indicator.title,
                'progress': completion,
                'status': indicator.status.value,
                'type': indicator.progress_type.value
            }
            
        summary['average_progress'] = total_progress / len(self.progress_indicators)
        
        return summary

# Instance globale
global_progress_tracker: Optional[AdvancedProgressTracker] = None

def get_progress_tracker() -> AdvancedProgressTracker:
    """Obtenir l'instance globale du tracker de progression"""
    global global_progress_tracker
    
    if global_progress_tracker is None:
        global_progress_tracker = AdvancedProgressTracker()
        
    return global_progress_tracker

# Fonctions utilitaires
def render_progress_dashboard():
    """Fonction utilitaire pour rendre le dashboard de progression"""
    tracker = get_progress_tracker()
    tracker.render_progress_dashboard()

def update_agent_progress(agent_id: str, progress: float, message: str = None):
    """Fonction utilitaire pour mettre à jour la progression d'un agent"""
    tracker = get_progress_tracker()
    tracker.update_progress(agent_id, progress, message=message)

def create_progress_indicator(indicator_id: str, title: str, progress_type: ProgressType = ProgressType.LINEAR):
    """Fonction utilitaire pour créer un indicateur de progression"""
    tracker = get_progress_tracker()
    return tracker.create_custom_progress_indicator(indicator_id, title, progress_type)