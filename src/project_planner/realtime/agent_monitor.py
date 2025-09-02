"""
Live AI Agent Status Monitoring System
Système de surveillance en temps réel des agents IA
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import queue
import numpy as np

from .streaming_engine import (
    get_streaming_engine,
    StreamEvent,
    StreamEventType,
    AgentStatus
)

from .websocket_manager import (
    get_websocket_manager,
    WebSocketMessage,
    WebSocketMessageType,
    WebSocketEventChannel
)

logger = logging.getLogger(__name__)

class AgentHealthStatus(Enum):
    """États de santé des agents"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

class MonitoringMetric(Enum):
    """Métriques de monitoring"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    SUCCESS_RATE = "success_rate"
    ERROR_COUNT = "error_count"
    TASK_COMPLETION_RATE = "task_completion_rate"
    THROUGHPUT = "throughput"

@dataclass
class AgentMetrics:
    """Métriques détaillées d'un agent"""
    agent_id: str
    agent_name: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    total_tasks: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    health_status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    alerts: List[str] = field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Calculer le taux de succès"""
        if self.total_tasks == 0:
            return 0.0
        return (self.success_count / self.total_tasks) * 100
        
    def get_avg_response_time(self) -> float:
        """Calculer le temps de réponse moyen"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
        
    def get_current_memory_usage(self) -> float:
        """Obtenir l'utilisation mémoire actuelle"""
        if not self.memory_usage:
            return 0.0
        return self.memory_usage[-1]
        
    def get_current_cpu_usage(self) -> float:
        """Obtenir l'utilisation CPU actuelle"""
        if not self.cpu_usage:
            return 0.0
        return self.cpu_usage[-1]
        
    def add_response_time(self, response_time: float):
        """Ajouter un temps de réponse"""
        self.response_times.append(response_time)
        self.last_activity = datetime.now()
        
    def add_memory_usage(self, memory: float):
        """Ajouter utilisation mémoire"""
        self.memory_usage.append(memory)
        
    def add_cpu_usage(self, cpu: float):
        """Ajouter utilisation CPU"""
        self.cpu_usage.append(cpu)
        
    def add_alert(self, alert_message: str):
        """Ajouter une alerte"""
        self.alerts.append(f"{datetime.now().strftime('%H:%M:%S')} - {alert_message}")
        if len(self.alerts) > 20:  # Garder seulement les 20 dernières
            self.alerts.pop(0)
            
    def update_health_status(self):
        """Mettre à jour le statut de santé"""
        
        # Vérifier l'activité récente
        time_since_activity = (datetime.now() - self.last_activity).total_seconds()
        
        if time_since_activity > 300:  # 5 minutes sans activité
            self.health_status = AgentHealthStatus.OFFLINE
            return
            
        # Vérifier les métriques
        critical_conditions = 0
        warning_conditions = 0
        
        # Temps de réponse
        if self.response_times:
            avg_response_time = self.get_avg_response_time()
            if avg_response_time > 10:  # Plus de 10 secondes
                critical_conditions += 1
            elif avg_response_time > 5:  # Plus de 5 secondes
                warning_conditions += 1
                
        # Taux d'erreur
        error_rate = (self.error_count / max(1, self.total_tasks)) * 100
        if error_rate > 20:  # Plus de 20% d'erreurs
            critical_conditions += 1
        elif error_rate > 10:  # Plus de 10% d'erreurs
            warning_conditions += 1
            
        # Utilisation mémoire
        if self.memory_usage and self.memory_usage[-1] > 90:  # Plus de 90%
            critical_conditions += 1
        elif self.memory_usage and self.memory_usage[-1] > 80:  # Plus de 80%
            warning_conditions += 1
            
        # Déterminer le statut final
        if critical_conditions > 0:
            self.health_status = AgentHealthStatus.CRITICAL
        elif warning_conditions > 0:
            self.health_status = AgentHealthStatus.WARNING
        else:
            self.health_status = AgentHealthStatus.HEALTHY

@dataclass
class SystemAlert:
    """Alerte système"""
    alert_id: str
    alert_type: str
    severity: str
    message: str
    agent_id: Optional[str]
    timestamp: datetime
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }

class LiveAgentMonitor:
    """Moniteur en temps réel des agents IA"""
    
    def __init__(self):
        self.streaming_engine = get_streaming_engine()
        self.websocket_manager = get_websocket_manager()
        self.agent_metrics = {}
        self.system_alerts = []
        self.monitoring_active = False
        self.alert_thresholds = {
            'response_time_critical': 10.0,  # secondes
            'response_time_warning': 5.0,
            'error_rate_critical': 20.0,     # pourcentage
            'error_rate_warning': 10.0,
            'memory_critical': 90.0,         # pourcentage
            'memory_warning': 80.0,
            'cpu_critical': 95.0,            # pourcentage
            'cpu_warning': 85.0
        }
        
        # Historique des événements
        self.event_history = deque(maxlen=1000)
        
        # Démarrer le monitoring
        self._start_monitoring()
        
    def _start_monitoring(self):
        """Démarrer le monitoring des agents"""
        
        # S'abonner aux événements du streaming engine
        self.streaming_engine.subscribe('all', self._handle_agent_event)
        
        # Démarrer les tâches de monitoring
        self.monitoring_active = True
        self._start_monitoring_thread()
        
        logger.info("🔍 Monitoring des agents démarré")
        
    def _start_monitoring_thread(self):
        """Démarrer le thread de monitoring"""
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
    def _monitoring_loop(self):
        """Boucle principale de monitoring"""
        
        while self.monitoring_active:
            try:
                # Mettre à jour les métriques des agents
                self._update_agent_metrics()
                
                # Vérifier les alertes
                self._check_alerts()
                
                # Nettoyer les anciennes données
                self._cleanup_old_data()
                
                # Attendre avant la prochaine itération
                time.sleep(2)  # Monitoring toutes les 2 secondes
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de monitoring: {e}")
                time.sleep(5)
                
    def _handle_agent_event(self, event: StreamEvent):
        """Gérer un événement d'agent"""
        
        # Ajouter à l'historique
        self.event_history.append(event)
        
        agent_id = event.agent_id
        if not agent_id:
            return
            
        # Créer les métriques de l'agent si elles n'existent pas
        if agent_id not in self.agent_metrics:
            self._create_agent_metrics(agent_id, event)
            
        metrics = self.agent_metrics[agent_id]
        
        # Traiter selon le type d'événement
        if event.event_type == StreamEventType.AGENT_START:
            self._handle_agent_start(metrics, event)
            
        elif event.event_type == StreamEventType.AGENT_PROGRESS:
            self._handle_agent_progress(metrics, event)
            
        elif event.event_type == StreamEventType.AGENT_COMPLETE:
            self._handle_agent_complete(metrics, event)
            
        elif event.event_type == StreamEventType.AGENT_ERROR:
            self._handle_agent_error(metrics, event)
            
    def _create_agent_metrics(self, agent_id: str, event: StreamEvent):
        """Créer les métriques pour un nouvel agent"""
        
        agent_name = event.data.get('agent_name', f'Agent-{agent_id[:8]}')
        
        self.agent_metrics[agent_id] = AgentMetrics(
            agent_id=agent_id,
            agent_name=agent_name
        )
        
        logger.info(f"📊 Métriques créées pour l'agent: {agent_name}")
        
    def _handle_agent_start(self, metrics: AgentMetrics, event: StreamEvent):
        """Gérer le démarrage d'un agent"""
        
        metrics.last_activity = event.timestamp
        metrics.health_status = AgentHealthStatus.HEALTHY
        
        # Simuler quelques métriques initiales
        metrics.add_memory_usage(np.random.uniform(20, 40))
        metrics.add_cpu_usage(np.random.uniform(10, 30))
        
    def _handle_agent_progress(self, metrics: AgentMetrics, event: StreamEvent):
        """Gérer le progrès d'un agent"""
        
        metrics.last_activity = event.timestamp
        
        # Simuler temps de réponse basé sur les données
        simulated_response_time = np.random.uniform(0.5, 3.0)
        metrics.add_response_time(simulated_response_time)
        
        # Simuler métriques système
        current_memory = metrics.get_current_memory_usage()
        memory_change = np.random.uniform(-5, 10)
        new_memory = max(10, min(95, current_memory + memory_change))
        metrics.add_memory_usage(new_memory)
        
        current_cpu = metrics.get_current_cpu_usage()
        cpu_change = np.random.uniform(-10, 15)
        new_cpu = max(5, min(98, current_cpu + cpu_change))
        metrics.add_cpu_usage(new_cpu)
        
        # Incrémenter tâches
        metrics.total_tasks += 1
        
    def _handle_agent_complete(self, metrics: AgentMetrics, event: StreamEvent):
        """Gérer la completion d'un agent"""
        
        metrics.last_activity = event.timestamp
        metrics.success_count += 1
        metrics.total_tasks += 1
        
        # Temps de réponse final
        final_response_time = np.random.uniform(1.0, 4.0)
        metrics.add_response_time(final_response_time)
        
        # Réduire les métriques système
        metrics.add_memory_usage(max(10, metrics.get_current_memory_usage() - 10))
        metrics.add_cpu_usage(max(5, metrics.get_current_cpu_usage() - 15))
        
    def _handle_agent_error(self, metrics: AgentMetrics, event: StreamEvent):
        """Gérer une erreur d'agent"""
        
        metrics.last_activity = event.timestamp
        metrics.error_count += 1
        metrics.total_tasks += 1
        
        error_message = event.data.get('error_message', 'Erreur inconnue')
        metrics.add_alert(f"Erreur: {error_message}")
        
        # Temps de réponse élevé en cas d'erreur
        error_response_time = np.random.uniform(5.0, 15.0)
        metrics.add_response_time(error_response_time)
        
    def _update_agent_metrics(self):
        """Mettre à jour les métriques de tous les agents"""
        
        for agent_id, metrics in self.agent_metrics.items():
            # Mettre à jour le statut de santé
            metrics.update_health_status()
            
            # Simuler métriques en continu pour agents actifs
            if metrics.health_status in [AgentHealthStatus.HEALTHY, AgentHealthStatus.WARNING]:
                
                # Variation légère des métriques système
                if metrics.memory_usage:
                    current_memory = metrics.get_current_memory_usage()
                    memory_variation = np.random.uniform(-2, 2)
                    new_memory = max(10, min(95, current_memory + memory_variation))
                    metrics.add_memory_usage(new_memory)
                    
                if metrics.cpu_usage:
                    current_cpu = metrics.get_current_cpu_usage()
                    cpu_variation = np.random.uniform(-3, 3)
                    new_cpu = max(5, min(98, current_cpu + cpu_variation))
                    metrics.add_cpu_usage(new_cpu)
                    
    def _check_alerts(self):
        """Vérifier et générer les alertes"""
        
        for agent_id, metrics in self.agent_metrics.items():
            
            # Alerte temps de réponse
            avg_response_time = metrics.get_avg_response_time()
            if avg_response_time > self.alert_thresholds['response_time_critical']:
                self._create_alert('response_time', 'critical', 
                                 f"Temps de réponse critique: {avg_response_time:.2f}s", agent_id)
            elif avg_response_time > self.alert_thresholds['response_time_warning']:
                self._create_alert('response_time', 'warning',
                                 f"Temps de réponse élevé: {avg_response_time:.2f}s", agent_id)
                                 
            # Alerte taux d'erreur
            if metrics.total_tasks > 0:
                error_rate = (metrics.error_count / metrics.total_tasks) * 100
                if error_rate > self.alert_thresholds['error_rate_critical']:
                    self._create_alert('error_rate', 'critical',
                                     f"Taux d'erreur critique: {error_rate:.1f}%", agent_id)
                elif error_rate > self.alert_thresholds['error_rate_warning']:
                    self._create_alert('error_rate', 'warning',
                                     f"Taux d'erreur élevé: {error_rate:.1f}%", agent_id)
                                     
            # Alerte mémoire
            current_memory = metrics.get_current_memory_usage()
            if current_memory > self.alert_thresholds['memory_critical']:
                self._create_alert('memory', 'critical',
                                 f"Utilisation mémoire critique: {current_memory:.1f}%", agent_id)
            elif current_memory > self.alert_thresholds['memory_warning']:
                self._create_alert('memory', 'warning',
                                 f"Utilisation mémoire élevée: {current_memory:.1f}%", agent_id)
                                 
            # Alerte CPU
            current_cpu = metrics.get_current_cpu_usage()
            if current_cpu > self.alert_thresholds['cpu_critical']:
                self._create_alert('cpu', 'critical',
                                 f"Utilisation CPU critique: {current_cpu:.1f}%", agent_id)
            elif current_cpu > self.alert_thresholds['cpu_warning']:
                self._create_alert('cpu', 'warning',
                                 f"Utilisation CPU élevée: {current_cpu:.1f}%", agent_id)
                                 
    def _create_alert(self, alert_type: str, severity: str, message: str, agent_id: str):
        """Créer une nouvelle alerte"""
        
        # Vérifier si une alerte similaire existe déjà récemment
        recent_alerts = [a for a in self.system_alerts[-10:] 
                        if a.alert_type == alert_type and a.agent_id == agent_id and a.severity == severity]
        
        # Ne créer l'alerte que si pas d'alerte similaire dans les 5 dernières minutes
        if not recent_alerts or (datetime.now() - recent_alerts[-1].timestamp).total_seconds() > 300:
            
            alert = SystemAlert(
                alert_id=f"{alert_type}_{agent_id}_{int(time.time())}",
                alert_type=alert_type,
                severity=severity,
                message=message,
                agent_id=agent_id,
                timestamp=datetime.now()
            )
            
            self.system_alerts.append(alert)
            
            # Notifier via WebSocket
            self._send_alert_notification(alert)
            
            logger.warning(f"🚨 Alerte {severity}: {message}")
            
    def _send_alert_notification(self, alert: SystemAlert):
        """Envoyer notification d'alerte via WebSocket"""
        
        try:
            alert_message = {
                'type': 'alert',
                'data': alert.to_dict()
            }
            
            # Diffuser l'alerte (implémentation simplifiée)
            # En production, cela utiliserait le WebSocket manager
            logger.info(f"Notification d'alerte envoyée: {alert.message}")
            
        except Exception as e:
            logger.error(f"Erreur envoi notification alerte: {e}")
            
    def _cleanup_old_data(self):
        """Nettoyer les anciennes données"""
        
        # Nettoyer les alertes anciennes (garder seulement les 100 dernières)
        if len(self.system_alerts) > 100:
            self.system_alerts = self.system_alerts[-100:]
            
        # Nettoyer les métriques d'agents inactifs depuis plus d'une heure
        current_time = datetime.now()
        agents_to_remove = []
        
        for agent_id, metrics in self.agent_metrics.items():
            time_since_activity = (current_time - metrics.last_activity).total_seconds()
            if time_since_activity > 3600:  # 1 heure
                agents_to_remove.append(agent_id)
                
        for agent_id in agents_to_remove:
            del self.agent_metrics[agent_id]
            logger.info(f"Métriques supprimées pour agent inactif: {agent_id}")
            
    def render_monitoring_dashboard(self):
        """Rendre le dashboard de monitoring"""
        
        st.markdown("# 🔍 Monitoring Agents IA Temps Réel")
        st.markdown("---")
        
        if not self.agent_metrics:
            st.info("🤖 Aucun agent actif à monitorer. Démarrez une génération de plan pour voir les métriques.")
            return
            
        # Vue d'ensemble système
        self._render_system_overview()
        
        # Alertes actives
        self._render_active_alerts()
        
        # Métriques détaillées par agent
        st.markdown("## 📊 Métriques Détaillées par Agent")
        
        for agent_id, metrics in self.agent_metrics.items():
            self._render_agent_metrics(metrics)
            
    def _render_system_overview(self):
        """Rendre la vue d'ensemble système"""
        
        # Calculer métriques système
        total_agents = len(self.agent_metrics)
        healthy_agents = len([m for m in self.agent_metrics.values() 
                             if m.health_status == AgentHealthStatus.HEALTHY])
        warning_agents = len([m for m in self.agent_metrics.values() 
                             if m.health_status == AgentHealthStatus.WARNING])
        critical_agents = len([m for m in self.agent_metrics.values() 
                              if m.health_status == AgentHealthStatus.CRITICAL])
        offline_agents = len([m for m in self.agent_metrics.values() 
                             if m.health_status == AgentHealthStatus.OFFLINE])
        
        # Métriques globales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🤖 Total Agents", total_agents)
            
        with col2:
            st.metric("✅ Sains", healthy_agents)
            
        with col3:
            st.metric("⚠️ Attention", warning_agents)
            
        with col4:
            st.metric("🚨 Critique", critical_agents)
            
        with col5:
            st.metric("📴 Hors Ligne", offline_agents)
            
        # Graphique de santé du système
        self._render_system_health_chart()
        
        # Métriques temps réel
        self._render_realtime_metrics()
        
    def _render_system_health_chart(self):
        """Rendre le graphique de santé système"""
        
        if not self.agent_metrics:
            return
            
        # Données pour le graphique en donut
        health_counts = {
            'Sains': len([m for m in self.agent_metrics.values() if m.health_status == AgentHealthStatus.HEALTHY]),
            'Attention': len([m for m in self.agent_metrics.values() if m.health_status == AgentHealthStatus.WARNING]),
            'Critique': len([m for m in self.agent_metrics.values() if m.health_status == AgentHealthStatus.CRITICAL]),
            'Hors Ligne': len([m for m in self.agent_metrics.values() if m.health_status == AgentHealthStatus.OFFLINE])
        }
        
        # Filtrer les catégories vides
        health_counts = {k: v for k, v in health_counts.items() if v > 0}
        
        if health_counts:
            fig = go.Figure(data=[go.Pie(
                labels=list(health_counts.keys()),
                values=list(health_counts.values()),
                hole=0.6,
                marker_colors=['green', 'orange', 'red', 'gray']
            )])
            
            fig.update_layout(
                title="Santé Globale des Agents",
                annotations=[dict(text='Agents', x=0.5, y=0.5, font_size=20, showarrow=False)],
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_realtime_metrics(self):
        """Rendre les métriques temps réel"""
        
        if not self.agent_metrics:
            return
            
        st.markdown("### ⏱️ Métriques Temps Réel")
        
        # Graphiques de métriques système
        col1, col2 = st.columns(2)
        
        with col1:
            # Temps de réponse moyens
            agent_names = [m.agent_name for m in self.agent_metrics.values()]
            avg_response_times = [m.get_avg_response_time() for m in self.agent_metrics.values()]
            
            fig_response = go.Figure(data=[
                go.Bar(x=agent_names, y=avg_response_times, 
                      marker_color=['red' if t > 5 else 'orange' if t > 2 else 'green' for t in avg_response_times])
            ])
            
            fig_response.update_layout(
                title="Temps de Réponse Moyen (s)",
                height=300
            )
            
            st.plotly_chart(fig_response, use_container_width=True)
            
        with col2:
            # Taux de succès
            success_rates = [m.get_success_rate() for m in self.agent_metrics.values()]
            
            fig_success = go.Figure(data=[
                go.Bar(x=agent_names, y=success_rates,
                      marker_color=['red' if r < 80 else 'orange' if r < 95 else 'green' for r in success_rates])
            ])
            
            fig_success.update_layout(
                title="Taux de Succès (%)",
                height=300
            )
            
            st.plotly_chart(fig_success, use_container_width=True)
            
    def _render_active_alerts(self):
        """Rendre les alertes actives"""
        
        active_alerts = [a for a in self.system_alerts if not a.acknowledged][-10:]  # 10 dernières
        
        if active_alerts:
            st.markdown("### 🚨 Alertes Actives")
            
            for alert in reversed(active_alerts):  # Plus récentes en premier
                severity_colors = {
                    'critical': '🔴',
                    'warning': '🟠',
                    'info': '🔵'
                }
                
                icon = severity_colors.get(alert.severity, '⚪')
                time_str = alert.timestamp.strftime('%H:%M:%S')
                
                with st.expander(f"{icon} {alert.message} ({time_str})", expanded=(alert.severity == 'critical')):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Type:** {alert.alert_type}")
                        st.markdown(f"**Agent:** {alert.agent_id}")
                        
                    with col2:
                        st.markdown(f"**Sévérité:** {alert.severity}")
                        
                    with col3:
                        if st.button("Acquitter", key=f"ack_{alert.alert_id}"):
                            alert.acknowledged = True
                            st.success("Alerte acquittée")
                            st.rerun()
                            
        else:
            st.success("✅ Aucune alerte active")
            
    def _render_agent_metrics(self, metrics: AgentMetrics):
        """Rendre les métriques d'un agent spécifique"""
        
        # Container pour cet agent
        with st.expander(f"🤖 {metrics.agent_name} ({metrics.health_status.value})", 
                        expanded=(metrics.health_status in [AgentHealthStatus.CRITICAL, AgentHealthStatus.WARNING])):
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Temps Réponse Moy.", f"{metrics.get_avg_response_time():.2f}s")
                
            with col2:
                st.metric("Taux de Succès", f"{metrics.get_success_rate():.1f}%")
                
            with col3:
                st.metric("Mémoire", f"{metrics.get_current_memory_usage():.1f}%")
                
            with col4:
                st.metric("CPU", f"{metrics.get_current_cpu_usage():.1f}%")
                
            # Graphiques détaillés
            self._render_agent_charts(metrics)
            
            # Alertes de cet agent
            agent_alerts = [a for a in self.system_alerts 
                           if a.agent_id == metrics.agent_id and not a.acknowledged]
            
            if agent_alerts:
                st.markdown("**Alertes récentes:**")
                for alert in agent_alerts[-3:]:  # 3 dernières
                    st.markdown(f"- {alert.timestamp.strftime('%H:%M:%S')}: {alert.message}")
                    
    def _render_agent_charts(self, metrics: AgentMetrics):
        """Rendre les graphiques pour un agent"""
        
        # Créer les sous-graphiques
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temps de Réponse', 'Utilisation Mémoire', 'Utilisation CPU', 'Historique des Tâches'),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Temps de réponse
        if metrics.response_times:
            time_points = list(range(len(metrics.response_times)))
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=list(metrics.response_times),
                    mode='lines+markers',
                    name='Temps de Réponse',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
        # Utilisation mémoire
        if metrics.memory_usage:
            time_points = list(range(len(metrics.memory_usage)))
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=list(metrics.memory_usage),
                    mode='lines+markers',
                    name='Mémoire',
                    line=dict(color='orange'),
                    fill='tonexty'
                ),
                row=1, col=2
            )
            
        # Utilisation CPU
        if metrics.cpu_usage:
            time_points = list(range(len(metrics.cpu_usage)))
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=list(metrics.cpu_usage),
                    mode='lines+markers',
                    name='CPU',
                    line=dict(color='red'),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
        # Historique des tâches
        success_data = [metrics.success_count, metrics.error_count]
        labels = ['Succès', 'Erreurs']
        
        fig.add_trace(
            go.Pie(
                values=success_data,
                labels=labels,
                name="Tâches",
                marker_colors=['green', 'red']
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_yaxes(title_text="Temps (s)", row=1, col=1)
        fig.update_yaxes(title_text="Pourcentage", row=1, col=2)
        fig.update_yaxes(title_text="Pourcentage", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé du monitoring"""
        
        return {
            'total_agents': len(self.agent_metrics),
            'healthy_agents': len([m for m in self.agent_metrics.values() 
                                  if m.health_status == AgentHealthStatus.HEALTHY]),
            'agents_with_issues': len([m for m in self.agent_metrics.values() 
                                      if m.health_status in [AgentHealthStatus.WARNING, AgentHealthStatus.CRITICAL]]),
            'active_alerts': len([a for a in self.system_alerts if not a.acknowledged]),
            'total_events': len(self.event_history),
            'monitoring_uptime': (datetime.now() - datetime.now()).total_seconds() if self.monitoring_active else 0
        }
        
    def stop_monitoring(self):
        """Arrêter le monitoring"""
        
        self.monitoring_active = False
        logger.info("🛑 Monitoring des agents arrêté")

# Instance globale
global_agent_monitor: Optional[LiveAgentMonitor] = None

def get_agent_monitor() -> LiveAgentMonitor:
    """Obtenir l'instance globale du monitoring des agents"""
    global global_agent_monitor
    
    if global_agent_monitor is None:
        global_agent_monitor = LiveAgentMonitor()
        
    return global_agent_monitor

# Fonctions utilitaires
def render_agent_monitoring_dashboard():
    """Fonction utilitaire pour rendre le dashboard de monitoring"""
    monitor = get_agent_monitor()
    monitor.render_monitoring_dashboard()

def get_agent_health_status(agent_id: str) -> Optional[AgentHealthStatus]:
    """Obtenir le statut de santé d'un agent"""
    monitor = get_agent_monitor()
    if agent_id in monitor.agent_metrics:
        return monitor.agent_metrics[agent_id].health_status
    return None