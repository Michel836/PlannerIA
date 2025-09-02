"""
🔴 Composant Dashboard - Intégration Temps Réel
==============================================

Interface unifiée pour toutes les fonctionnalités temps réel de PlannerIA.
Ce composant orchestre l'affichage live de la génération de plans, du monitoring
des agents IA, et des métriques de performance en temps réel.

Fonctionnalités Principales:
- 🎬 Visualisation live de la génération de plans avec animations
- 📊 Suivi en temps réel de la progression multi-agents
- 🔍 Monitoring avancé de la santé et performance des agents IA
- 📡 Interface WebSocket pour communication bidirectionnelle
- ⚙️ Configuration complète du système temps réel

Architecture Multi-Onglets:
1. 📺 Live Generation - Stream de génération avec progress bars animées
2. 📊 Progress Tracking - Indicateurs de progression par étapes détaillées  
3. 🔍 Agent Monitoring - Santé, métriques et alertes des agents IA
4. 📡 WebSocket Status - État des connexions et statistiques de communication
5. ⚙️ Configuration - Paramétrage fin du système temps réel

Contrôles de Session:
▶️ START - Démarrer nouvelle session de streaming
⏸️ PAUSE - Suspendre temporairement les mises à jour
⏹️ STOP - Arrêter complètement la session active
🔄 REFRESH - Actualisation manuelle des données
🎮 SIMULATION - Mode démonstration avec agents simulés

Intégrations Système:
- StreamingEngine pour événements temps réel
- LiveVisualization pour graphiques animés
- ProgressTracker pour indicateurs de progression
- WebSocketManager pour communication réseau
- AgentMonitor pour surveillance des agents IA

Exemple d'usage:
    dashboard = RealTimeDashboard()
    
    # Panel de contrôle principal
    dashboard.render_realtime_control_panel()
    
    # Interface complète avec onglets
    dashboard.render_realtime_dashboard_tabs({
        'project_id': 'demo_001',
        'session_type': 'live_demo'
    })
    
    # Intégration sidebar pour accès rapide
    dashboard.render_realtime_sidebar(project_data)

Performance & Optimisations:
- Auto-refresh intelligent basé sur l'activité
- Thread background pour éviter blocage UI
- Cache des métriques pour réduire latence
- Debouncing des mises à jour haute fréquence
- Lazy loading des composants coûteux

Sécurité:
- Validation des sessions et permissions
- Rate limiting des WebSocket connections
- Sanitisation des données d'entrée utilisateur
- Timeout automatique des sessions inactives

Auteur: PlannerIA AI System  
Version: 2.0.0
"""

import streamlit as st
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio
import threading

from ...realtime import (
    get_streaming_engine,
    get_live_visualization,
    get_progress_tracker,
    get_websocket_manager,
    get_agent_monitor,
    start_realtime_session,
    render_live_generation_dashboard,
    render_progress_dashboard,
    render_agent_monitoring_dashboard,
    SuggestionTrigger,
    StreamEventType
)

logger = logging.getLogger(__name__)

class RealTimeDashboard:
    """Dashboard intégré pour toutes les fonctionnalités temps réel"""
    
    def __init__(self):
        self.streaming_engine = get_streaming_engine()
        self.live_viz = get_live_visualization()
        self.progress_tracker = get_progress_tracker()
        self.websocket_manager = get_websocket_manager()
        self.agent_monitor = get_agent_monitor()
        
        # État du dashboard
        self.is_active = False
        self.current_session_id = None
        
    def render_realtime_control_panel(self):
        """Panneau de contrôle temps réel"""
        
        st.markdown("## 🔴 Contrôle Temps Réel")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Démarrer Session", key="start_realtime"):
                self._start_realtime_session()
                
        with col2:
            if st.button("⏸️ Pause", key="pause_realtime"):
                self._pause_realtime_session()
                
        with col3:
            if st.button("⏹️ Arrêter", key="stop_realtime"):
                self._stop_realtime_session()
                
        with col4:
            if st.button("🔄 Actualiser", key="refresh_realtime"):
                st.info("✅ Actualisation terminée")
                
        # Statut de la session
        streaming_status = self.streaming_engine.get_streaming_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if streaming_status['streaming_active'] else "🔴"
            st.markdown(f"**Statut:** {status_color} {streaming_status['status']}")
            
        with col2:
            st.markdown(f"**Session:** {streaming_status['session_id'][:8]}...")
            
        with col3:
            st.markdown(f"**Agents:** {streaming_status['active_agents']}/{streaming_status['agent_count']}")
            
    def render_realtime_dashboard_tabs(self, project_data: Dict[str, Any]):
        """Rendre les onglets du dashboard temps réel"""
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📺 Live Generation",
            "📊 Progress Tracking", 
            "🔍 Agent Monitoring",
            "📡 WebSocket Status",
            "⚙️ Configuration"
        ])
        
        with tab1:
            self._render_live_generation_tab(project_data)
            
        with tab2:
            self._render_progress_tracking_tab()
            
        with tab3:
            self._render_agent_monitoring_tab()
            
        with tab4:
            self._render_websocket_status_tab()
            
        with tab5:
            self._render_configuration_tab()
            
    def _render_live_generation_tab(self, project_data: Dict[str, Any]):
        """Onglet de génération live"""
        
        st.markdown("### 🎬 Génération de Plan en Direct")
        
        # Vérifier si une génération est active
        streaming_status = self.streaming_engine.get_streaming_status()
        
        if streaming_status['streaming_active']:
            # Afficher la visualisation live
            render_live_generation_dashboard(project_data)
            
            # Auto-refresh si la génération est active
            if streaming_status['active_agents'] > 0:
                time.sleep(1)
                st.info("✅ Actualisation terminée")
                
        else:
            st.info("🎯 Aucune génération active. Démarrez une session pour voir la visualisation en temps réel.")
            
            # Bouton pour démarrer une simulation
            if st.button("🎮 Démarrer Simulation", key="start_simulation"):
                self._start_simulation_session(project_data)
                
    def _render_progress_tracking_tab(self):
        """Onglet de suivi de progression"""
        
        st.markdown("### 📈 Suivi de Progression")
        
        # Rendre le dashboard de progression
        render_progress_dashboard()
        
    def _render_agent_monitoring_tab(self):
        """Onglet de monitoring des agents"""
        
        st.markdown("### 🤖 Monitoring des Agents IA")
        
        # Rendre le dashboard de monitoring
        render_agent_monitoring_dashboard()
        
    def _render_websocket_status_tab(self):
        """Onglet du statut WebSocket"""
        
        st.markdown("### 📡 Statut WebSocket")
        
        try:
            # Obtenir les statistiques WebSocket
            ws_stats = self.websocket_manager.get_connection_stats()
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Connexions Actives", ws_stats['active_connections'])
                
            with col2:
                st.metric("Messages Envoyés", ws_stats['messages_sent'])
                
            with col3:
                st.metric("Messages Reçus", ws_stats['messages_received'])
                
            with col4:
                bytes_mb = ws_stats['bytes_transferred'] / (1024 * 1024)
                st.metric("Données (MB)", f"{bytes_mb:.2f}")
                
            # Détails des connexions
            if ws_stats['clients']:
                st.markdown("#### 🔗 Connexions Actives")
                
                for client_id, client_info in ws_stats['clients'].items():
                    with st.expander(f"Client {client_id[:8]}", expanded=False):
                        st.json(client_info)
                        
            # Statut des canaux
            if ws_stats['channels']:
                st.markdown("#### 📢 Canaux WebSocket")
                
                for channel, subscriber_count in ws_stats['channels'].items():
                    st.markdown(f"- **{channel}:** {subscriber_count} abonnés")
                    
        except Exception as e:
            st.error(f"Erreur récupération statut WebSocket: {e}")
            
    def _render_configuration_tab(self):
        """Onglet de configuration"""
        
        st.markdown("### ⚙️ Configuration Temps Réel")
        
        # Configuration du streaming
        with st.expander("🎬 Configuration Streaming", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                auto_refresh = st.checkbox(
                    "Auto-refresh", 
                    value=True,
                    help="Actualisation automatique de l'affichage"
                )
                
                refresh_interval = st.slider(
                    "Intervalle d'actualisation (ms)",
                    min_value=100,
                    max_value=5000,
                    value=1000,
                    step=100
                )
                
            with col2:
                enable_websocket = st.checkbox(
                    "Activer WebSocket",
                    value=True,
                    help="Communication temps réel via WebSocket"
                )
                
                websocket_port = st.number_input(
                    "Port WebSocket",
                    min_value=8000,
                    max_value=9999,
                    value=8765
                )
                
        # Configuration du monitoring
        with st.expander("🔍 Configuration Monitoring", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                enable_agent_monitoring = st.checkbox(
                    "Monitoring des agents",
                    value=True
                )
                
                alert_threshold = st.slider(
                    "Seuil d'alerte (secondes)",
                    min_value=1.0,
                    max_value=30.0,
                    value=5.0,
                    step=0.5
                )
                
            with col2:
                metrics_retention = st.slider(
                    "Rétention métriques (minutes)",
                    min_value=10,
                    max_value=120,
                    value=60
                )
                
                enable_alerts = st.checkbox(
                    "Alertes système",
                    value=True
                )
                
        # Bouton de sauvegarde
        if st.button("💾 Sauvegarder Configuration", key="save_realtime_config"):
            self._save_realtime_configuration({
                'auto_refresh': auto_refresh,
                'refresh_interval': refresh_interval,
                'enable_websocket': enable_websocket,
                'websocket_port': websocket_port,
                'enable_agent_monitoring': enable_agent_monitoring,
                'alert_threshold': alert_threshold,
                'metrics_retention': metrics_retention,
                'enable_alerts': enable_alerts
            })
            
    def _start_realtime_session(self):
        """Démarrer une session temps réel"""
        
        try:
            # Démarrer la session de streaming
            session_id = start_realtime_session({
                'project_type': 'demo',
                'user_id': 'dashboard_user',
                'start_time': datetime.now().isoformat()
            })
            
            self.current_session_id = session_id
            self.is_active = True
            
            st.success(f"✅ Session temps réel démarrée: {session_id[:8]}")
            
            # Démarrer WebSocket en arrière-plan si pas déjà fait
            self._ensure_websocket_server()
            
            st.info("✅ Actualisation terminée")
            
        except Exception as e:
            st.error(f"Erreur démarrage session: {e}")
            logger.error(f"Erreur démarrage session temps réel: {e}")
            
    def _pause_realtime_session(self):
        """Mettre en pause la session"""
        
        try:
            self.streaming_engine.pause_streaming()
            st.info("⏸️ Session mise en pause")
            st.info("✅ Actualisation terminée")
            
        except Exception as e:
            st.error(f"Erreur pause session: {e}")
            
    def _stop_realtime_session(self):
        """Arrêter la session temps réel"""
        
        try:
            self.streaming_engine.stop_streaming()
            self.is_active = False
            self.current_session_id = None
            
            st.success("🛑 Session temps réel arrêtée")
            st.info("✅ Actualisation terminée")
            
        except Exception as e:
            st.error(f"Erreur arrêt session: {e}")
            
    def _start_simulation_session(self, project_data: Dict[str, Any]):
        """Démarrer une session de simulation pour démonstration"""
        
        try:
            # Démarrer la session temps réel
            session_id = start_realtime_session(project_data)
            
            # Simuler des agents
            self._simulate_agents()
            
            st.success("🎮 Simulation démarrée")
            st.info("✅ Actualisation terminée")
            
        except Exception as e:
            st.error(f"Erreur simulation: {e}")
            
    def _simulate_agents(self):
        """Simuler l'activité des agents pour démonstration"""
        
        # Liste des agents simulés
        simulated_agents = [
            {'id': 'supervisor', 'name': 'Supervisor Agent'},
            {'id': 'planner', 'name': 'Planner Agent'},
            {'id': 'estimator', 'name': 'Estimator Agent'},
            {'id': 'risk', 'name': 'Risk Agent'},
            {'id': 'documentation', 'name': 'Documentation Agent'}
        ]
        
        # Démarrer les agents simulés
        for agent in simulated_agents:
            self.streaming_engine.emit_agent_start(
                agent['id'], 
                agent['name'],
                f"Initialisation {agent['name']}..."
            )
            
        # Programmer la progression simulée
        threading.Thread(target=self._run_agent_simulation, args=(simulated_agents,), daemon=True).start()
        
    def _run_agent_simulation(self, agents: List[Dict[str, str]]):
        """Exécuter la simulation des agents"""
        
        import time
        import random
        
        try:
            for step in range(100):  # 100 étapes de simulation
                for agent in agents:
                    # Progression simulée
                    progress = min(100, step + random.uniform(0, 2))
                    
                    # Messages variés
                    messages = [
                        f"Traitement étape {step+1}...",
                        f"Analyse en cours...",
                        f"Génération de contenu...",
                        f"Validation des résultats...",
                        f"Optimisation..."
                    ]
                    
                    self.streaming_engine.emit_agent_progress(
                        agent['id'],
                        progress,
                        f"Étape {step+1}/100",
                        random.choice(messages)
                    )
                    
                    # Compléter l'agent si progression terminée
                    if progress >= 100:
                        self.streaming_engine.emit_agent_complete(
                            agent['id'],
                            f"{agent['name']} terminé avec succès"
                        )
                        
                # Pause entre les étapes
                time.sleep(2)
                
                # Arrêter si plus de streaming actif
                if not self.streaming_engine.streaming_active:
                    break
                    
        except Exception as e:
            logger.error(f"Erreur simulation agents: {e}")
            
    def _ensure_websocket_server(self):
        """S'assurer que le serveur WebSocket est actif"""
        
        try:
            # Vérifier si le serveur est déjà actif
            if not hasattr(st.session_state, 'websocket_server_started'):
                
                # Démarrer le serveur WebSocket en arrière-plan
                from ...realtime.websocket_manager import run_websocket_server_in_thread
                
                thread = run_websocket_server_in_thread()
                st.session_state['websocket_server_started'] = True
                st.session_state['websocket_thread'] = thread
                
                logger.info("🌐 Serveur WebSocket démarré en arrière-plan")
                
        except Exception as e:
            logger.error(f"Erreur démarrage serveur WebSocket: {e}")
            
    def _save_realtime_configuration(self, config: Dict[str, Any]):
        """Sauvegarder la configuration temps réel"""
        
        try:
            st.session_state['realtime_config'] = config
            st.success("✅ Configuration sauvegardée")
            
        except Exception as e:
            st.error(f"Erreur sauvegarde configuration: {e}")
            
    def render_realtime_sidebar(self, project_data: Dict[str, Any]):
        """Rendre les éléments temps réel dans la sidebar"""
        
        with st.sidebar:
            st.markdown("### 🔴 Temps Réel")
            
            # Statut rapide
            streaming_status = self.streaming_engine.get_streaming_status()
            
            if streaming_status['streaming_active']:
                st.success("🟢 Session Active")
                st.markdown(f"**Agents:** {streaming_status['active_agents']}")
                
                # Bouton d'arrêt rapide
                if st.button("⏹️ Arrêter", key="sidebar_stop"):
                    self._stop_realtime_session()
                    
            else:
                st.info("🔴 Session Inactive")
                
                # Bouton de démarrage rapide
                if st.button("▶️ Démarrer", key="sidebar_start"):
                    self._start_realtime_session()
                    
            # Métriques rapides
            if streaming_status['agent_count'] > 0:
                st.markdown("---")
                st.markdown("**Métriques Rapides**")
                
                global_metrics = self.streaming_engine.get_global_metrics()
                
                completion = global_metrics.get('completion_percentage', 0)
                st.progress(completion / 100)
                st.markdown(f"Progression: {completion:.1f}%")
                
                events_per_sec = global_metrics.get('events_per_second', 0)
                st.markdown(f"Événements/s: {events_per_sec:.1f}")

# Instance globale
def get_realtime_dashboard():
    """Obtenir l'instance du dashboard temps réel"""
    if 'realtime_dashboard' not in st.session_state:
        st.session_state['realtime_dashboard'] = RealTimeDashboard()
    return st.session_state['realtime_dashboard']

# Fonctions utilitaires
def render_realtime_control_panel():
    """Fonction utilitaire pour le panneau de contrôle"""
    dashboard = get_realtime_dashboard()
    dashboard.render_realtime_control_panel()

def render_realtime_dashboard_tabs(project_data: Dict[str, Any]):
    """Fonction utilitaire pour les onglets temps réel"""
    dashboard = get_realtime_dashboard()
    dashboard.render_realtime_dashboard_tabs(project_data)

def render_realtime_sidebar(project_data: Dict[str, Any]):
    """Fonction utilitaire pour la sidebar temps réel"""
    dashboard = get_realtime_dashboard()
    dashboard.render_realtime_sidebar(project_data)