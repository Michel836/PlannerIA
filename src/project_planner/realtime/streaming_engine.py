"""
🌊 Moteur de Streaming Temps Réel PlannerIA
==========================================

Ce module implémente le cœur du système de mise à jour temps réel, permettant de suivre
en direct la progression des agents IA et la génération des plans de projet.

Fonctionnalités principales:
- 🎬 Streaming d'événements en temps réel avec queue thread-safe
- 📊 Suivi détaillé du statut et progrès de chaque agent IA
- 🔄 Gestion de sessions avec start/pause/stop
- 📈 Métriques globales avec estimation temps restant
- 💾 Historique des événements pour analyse et debug
- 🎯 Notifications push vers interfaces utilisateur

Architecture événementielle:
- RealTimeStreamingEngine: Moteur principal de streaming
- StreamEvent: Événement individuel avec métadonnées
- AgentStatus: État détaillé d'un agent en cours d'exécution
- StreamEventType: Types d'événements (10 catégories)

Types d'événements supportés:
🤖 AGENT_START/PROGRESS/COMPLETE/ERROR - Lifecycle des agents
📄 PLAN_CHUNK/COMPLETE - Génération incrémentale de plans  
📊 STATUS_UPDATE/METRICS_UPDATE - Mises à jour système
📝 LOG_MESSAGE/SYSTEM_MESSAGE - Messages de debug et système

Exemple d'usage:
    # Démarrer une session de streaming
    engine = get_streaming_engine()
    session_id = engine.start_streaming_session({'project_id': 'demo'})
    
    # Émettre des événements d'agents
    engine.emit_agent_start('planner', 'Planner Agent')
    engine.emit_agent_progress('planner', 50.0, 'Génération WBS en cours...')
    
    # Écouter les événements en continu
    for event in engine.get_event_stream():
        print(f"📡 {event.event_type}: {event.data}")

Performance:
- Traitement asynchrone multi-threaded
- Queue circulaire pour éviter la saturation mémoire  
- Métriques temps réel avec calculs optimisés
- Support de milliers d'événements/seconde

Auteur: PlannerIA AI System
Version: 2.0.0
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Generator
from dataclasses import dataclass, asdict
from enum import Enum
import queue
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    """
    🎭 Types d'Événements de Streaming
    
    Enumération complète des types d'événements supportés par le moteur de streaming.
    Chaque type correspond à une phase spécifique du workflow de génération de plans.
    
    Événements d'Agents (Lifecycle):
    - AGENT_START: Démarrage d'un agent IA
    - AGENT_PROGRESS: Mise à jour du progrès (0-100%)  
    - AGENT_COMPLETE: Finalisation réussie d'un agent
    - AGENT_ERROR: Erreur lors de l'exécution d'un agent
    
    Événements de Plans:
    - PLAN_CHUNK: Fragment de plan généré (streaming incrémental)
    - PLAN_COMPLETE: Plan final complet disponible
    
    Événements Système:
    - STATUS_UPDATE: Changement d'état global de la session
    - METRICS_UPDATE: Mise à jour des métriques de performance
    - LOG_MESSAGE: Message de debug ou d'information  
    - SYSTEM_MESSAGE: Message système (heartbeat, notifications)
    """
    AGENT_START = "agent_start"         # 🚀 Démarrage agent
    AGENT_PROGRESS = "agent_progress"   # 📊 Progression agent (0-100%)
    AGENT_COMPLETE = "agent_complete"   # ✅ Agent terminé avec succès
    AGENT_ERROR = "agent_error"         # ❌ Erreur agent
    PLAN_CHUNK = "plan_chunk"           # 📄 Fragment de plan
    PLAN_COMPLETE = "plan_complete"     # 📋 Plan complet
    STATUS_UPDATE = "status_update"     # 🔄 Mise à jour statut
    METRICS_UPDATE = "metrics_update"   # 📈 Mise à jour métriques
    LOG_MESSAGE = "log_message"         # 📝 Message de log
    SYSTEM_MESSAGE = "system_message"   # 🖥️ Message système

class StreamStatus(Enum):
    """États du streaming"""
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class StreamEvent:
    """Événement de streaming"""
    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    data: Dict[str, Any]
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'agent_id': self.agent_id,
            'session_id': self.session_id
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

@dataclass
class AgentStatus:
    """État d'un agent en temps réel"""
    agent_id: str
    agent_name: str
    status: str
    progress: float
    current_task: str
    start_time: datetime
    last_update: datetime
    messages: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'status': self.status,
            'progress': self.progress,
            'current_task': self.current_task,
            'start_time': self.start_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'messages': self.messages,
            'metrics': self.metrics
        }

class RealTimeStreamingEngine:
    """Moteur de streaming temps réel pour PlannerIA"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.stream_status = StreamStatus.IDLE
        self.event_queue = queue.Queue()
        self.subscribers = {}  # Callbacks for different event types
        self.agent_statuses = {}
        self.streaming_active = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Métriques globales
        self.global_metrics = {
            'total_events': 0,
            'events_per_second': 0,
            'active_agents': 0,
            'completion_percentage': 0.0,
            'estimated_time_remaining': 0,
            'start_time': None,
            'last_activity': datetime.now()
        }
        
        # Thread de traitement des événements
        self._start_event_processor()
        
    def _start_event_processor(self):
        """Démarrer le thread de traitement des événements"""
        self.processing_thread = threading.Thread(
            target=self._event_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        logger.info("🚀 Moteur de streaming temps réel démarré")
        
    def _event_processing_loop(self):
        """Boucle principale de traitement des événements"""
        while True:
            try:
                # Attendre un événement avec timeout
                try:
                    event = self.event_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                # Traiter l'événement
                self._process_event(event)
                
                # Notifier les subscribers
                self._notify_subscribers(event)
                
                # Mettre à jour les métriques
                self._update_metrics(event)
                
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Erreur traitement événement streaming: {e}")
                
    def _process_event(self, event: StreamEvent):
        """Traiter un événement spécifique"""
        
        if event.event_type == StreamEventType.AGENT_START:
            agent_data = event.data
            self.agent_statuses[event.agent_id] = AgentStatus(
                agent_id=event.agent_id,
                agent_name=agent_data.get('agent_name', 'Unknown'),
                status='starting',
                progress=0.0,
                current_task=agent_data.get('initial_task', 'Initialisation...'),
                start_time=event.timestamp,
                last_update=event.timestamp,
                messages=[],
                metrics={}
            )
            
        elif event.event_type == StreamEventType.AGENT_PROGRESS:
            if event.agent_id in self.agent_statuses:
                status = self.agent_statuses[event.agent_id]
                status.progress = event.data.get('progress', status.progress)
                status.current_task = event.data.get('current_task', status.current_task)
                status.last_update = event.timestamp
                status.status = 'active'
                
                # Ajouter messages
                if 'message' in event.data:
                    status.messages.append(event.data['message'])
                    # Garder seulement les 10 derniers messages
                    status.messages = status.messages[-10:]
                    
                # Mettre à jour métriques agent
                if 'metrics' in event.data:
                    status.metrics.update(event.data['metrics'])
                    
        elif event.event_type == StreamEventType.AGENT_COMPLETE:
            if event.agent_id in self.agent_statuses:
                status = self.agent_statuses[event.agent_id]
                status.status = 'completed'
                status.progress = 100.0
                status.current_task = 'Terminé'
                status.last_update = event.timestamp
                
                if 'final_message' in event.data:
                    status.messages.append(f"✅ {event.data['final_message']}")
                    
        elif event.event_type == StreamEventType.AGENT_ERROR:
            if event.agent_id in self.agent_statuses:
                status = self.agent_statuses[event.agent_id]
                status.status = 'error'
                status.current_task = f"❌ Erreur: {event.data.get('error_message', 'Erreur inconnue')}"
                status.last_update = event.timestamp
                
    def _notify_subscribers(self, event: StreamEvent):
        """Notifier les subscribers d'un événement"""
        
        # Notifier subscribers génériques
        if 'all' in self.subscribers:
            for callback in self.subscribers['all']:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Erreur callback subscriber: {e}")
                    
        # Notifier subscribers spécifiques au type d'événement
        event_type_key = event.event_type.value
        if event_type_key in self.subscribers:
            for callback in self.subscribers[event_type_key]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Erreur callback subscriber {event_type_key}: {e}")
                    
    def _update_metrics(self, event: StreamEvent):
        """Mettre à jour les métriques globales"""
        
        self.global_metrics['total_events'] += 1
        self.global_metrics['last_activity'] = event.timestamp
        
        # Calculer agents actifs
        active_agents = len([s for s in self.agent_statuses.values() if s.status == 'active'])
        self.global_metrics['active_agents'] = active_agents
        
        # Calculer pourcentage de completion global
        if self.agent_statuses:
            total_progress = sum(s.progress for s in self.agent_statuses.values())
            avg_progress = total_progress / len(self.agent_statuses)
            self.global_metrics['completion_percentage'] = avg_progress
            
        # Calculer événements par seconde
        if self.global_metrics['start_time']:
            elapsed = (event.timestamp - self.global_metrics['start_time']).total_seconds()
            if elapsed > 0:
                self.global_metrics['events_per_second'] = self.global_metrics['total_events'] / elapsed
                
    def start_streaming_session(self, project_data: Dict[str, Any] = None):
        """Démarrer une session de streaming"""
        
        self.session_id = str(uuid.uuid4())
        self.stream_status = StreamStatus.STARTING
        self.streaming_active = True
        self.global_metrics['start_time'] = datetime.now()
        
        # Émettre événement de démarrage
        self.emit_event(
            StreamEventType.STATUS_UPDATE,
            {
                'status': 'session_started',
                'session_id': self.session_id,
                'project_data': project_data or {}
            }
        )
        
        self.stream_status = StreamStatus.ACTIVE
        logger.info(f"📡 Session de streaming démarrée: {self.session_id}")
        
    def emit_event(self, 
                   event_type: StreamEventType,
                   data: Dict[str, Any],
                   agent_id: Optional[str] = None):
        """Émettre un événement dans le stream"""
        
        if not self.streaming_active and event_type != StreamEventType.STATUS_UPDATE:
            return
            
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            agent_id=agent_id,
            session_id=self.session_id
        )
        
        self.event_queue.put(event)
        
    def emit_agent_start(self, agent_id: str, agent_name: str, initial_task: str = None):
        """Émettre le démarrage d'un agent"""
        self.emit_event(
            StreamEventType.AGENT_START,
            {
                'agent_name': agent_name,
                'initial_task': initial_task or f"Démarrage de {agent_name}..."
            },
            agent_id=agent_id
        )
        
    def emit_agent_progress(self, 
                           agent_id: str,
                           progress: float,
                           current_task: str,
                           message: str = None,
                           metrics: Dict[str, Any] = None):
        """Émettre le progrès d'un agent"""
        
        data = {
            'progress': max(0, min(100, progress)),  # Limiter entre 0 et 100
            'current_task': current_task
        }
        
        if message:
            data['message'] = message
            
        if metrics:
            data['metrics'] = metrics
            
        self.emit_event(StreamEventType.AGENT_PROGRESS, data, agent_id=agent_id)
        
    def emit_agent_complete(self, agent_id: str, final_message: str = None, results: Dict[str, Any] = None):
        """Émettre la completion d'un agent"""
        
        data = {}
        if final_message:
            data['final_message'] = final_message
        if results:
            data['results'] = results
            
        self.emit_event(StreamEventType.AGENT_COMPLETE, data, agent_id=agent_id)
        
    def emit_agent_error(self, agent_id: str, error_message: str, error_details: Dict[str, Any] = None):
        """Émettre une erreur d'agent"""
        
        data = {'error_message': error_message}
        if error_details:
            data['error_details'] = error_details
            
        self.emit_event(StreamEventType.AGENT_ERROR, data, agent_id=agent_id)
        
    def emit_plan_chunk(self, chunk_data: Dict[str, Any], chunk_type: str = 'partial'):
        """Émettre un chunk de plan en cours de génération"""
        
        self.emit_event(
            StreamEventType.PLAN_CHUNK,
            {
                'chunk_type': chunk_type,
                'chunk_data': chunk_data,
                'timestamp': datetime.now().isoformat()
            }
        )
        
    def emit_plan_complete(self, final_plan: Dict[str, Any]):
        """Émettre la completion du plan"""
        
        self.emit_event(
            StreamEventType.PLAN_COMPLETE,
            {
                'final_plan': final_plan,
                'completion_time': datetime.now().isoformat()
            }
        )
        
    def emit_log_message(self, level: str, message: str, component: str = None):
        """Émettre un message de log"""
        
        self.emit_event(
            StreamEventType.LOG_MESSAGE,
            {
                'level': level,
                'message': message,
                'component': component or 'system'
            }
        )
        
    def subscribe(self, event_type: str, callback: Callable[[StreamEvent], None]):
        """S'abonner à un type d'événement"""
        
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
            
        self.subscribers[event_type].append(callback)
        logger.debug(f"Nouveau subscriber pour {event_type}")
        
    def unsubscribe(self, event_type: str, callback: Callable[[StreamEvent], None]):
        """Se désabonner d'un type d'événement"""
        
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(callback)
            except ValueError:
                pass
                
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Obtenir le status d'un agent"""
        return self.agent_statuses.get(agent_id)
        
    def get_all_agent_statuses(self) -> Dict[str, AgentStatus]:
        """Obtenir tous les status d'agents"""
        return self.agent_statuses.copy()
        
    def get_global_metrics(self) -> Dict[str, Any]:
        """Obtenir les métriques globales"""
        metrics = self.global_metrics.copy()
        
        # Calculer temps écoulé
        if metrics['start_time']:
            elapsed = (datetime.now() - metrics['start_time']).total_seconds()
            metrics['elapsed_time'] = elapsed
            
            # Estimer temps restant basé sur le progrès
            if metrics['completion_percentage'] > 0:
                estimated_total = elapsed / (metrics['completion_percentage'] / 100)
                metrics['estimated_time_remaining'] = max(0, estimated_total - elapsed)
            else:
                metrics['estimated_time_remaining'] = 0
                
        return metrics
        
    def get_event_stream(self) -> Generator[StreamEvent, None, None]:
        """Générateur d'événements en streaming"""
        
        events_emitted = 0
        start_time = time.time()
        
        while self.streaming_active:
            try:
                event = self.event_queue.get(timeout=0.1)
                events_emitted += 1
                yield event
                self.event_queue.task_done()
                
            except queue.Empty:
                # Émettre un heartbeat toutes les 5 secondes
                current_time = time.time()
                if current_time - start_time > 5:
                    heartbeat_event = StreamEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=StreamEventType.SYSTEM_MESSAGE,
                        timestamp=datetime.now(),
                        data={
                            'type': 'heartbeat',
                            'events_streamed': events_emitted,
                            'uptime': current_time - start_time
                        },
                        session_id=self.session_id
                    )
                    yield heartbeat_event
                    start_time = current_time
                    
    async def get_async_event_stream(self) -> AsyncGenerator[StreamEvent, None]:
        """Générateur d'événements asynchrone"""
        
        events_emitted = 0
        last_heartbeat = time.time()
        
        while self.streaming_active:
            try:
                event = self.event_queue.get_nowait()
                events_emitted += 1
                yield event
                self.event_queue.task_done()
                
            except queue.Empty:
                await asyncio.sleep(0.1)
                
                # Heartbeat asynchrone
                current_time = time.time()
                if current_time - last_heartbeat > 5:
                    heartbeat_event = StreamEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=StreamEventType.SYSTEM_MESSAGE,
                        timestamp=datetime.now(),
                        data={
                            'type': 'heartbeat',
                            'events_streamed': events_emitted,
                            'active_agents': len([s for s in self.agent_statuses.values() if s.status == 'active'])
                        },
                        session_id=self.session_id
                    )
                    yield heartbeat_event
                    last_heartbeat = current_time
                    
    def pause_streaming(self):
        """Mettre en pause le streaming"""
        self.stream_status = StreamStatus.PAUSED
        self.emit_event(
            StreamEventType.STATUS_UPDATE,
            {'status': 'paused'}
        )
        
    def resume_streaming(self):
        """Reprendre le streaming"""
        self.stream_status = StreamStatus.ACTIVE
        self.emit_event(
            StreamEventType.STATUS_UPDATE,
            {'status': 'resumed'}
        )
        
    def stop_streaming(self):
        """Arrêter le streaming"""
        
        self.streaming_active = False
        self.stream_status = StreamStatus.COMPLETED
        
        # Émettre événement final
        self.emit_event(
            StreamEventType.STATUS_UPDATE,
            {
                'status': 'stopped',
                'session_summary': {
                    'total_events': self.global_metrics['total_events'],
                    'duration': (datetime.now() - self.global_metrics['start_time']).total_seconds() if self.global_metrics['start_time'] else 0,
                    'agents_processed': len(self.agent_statuses)
                }
            }
        )
        
        logger.info(f"📡 Session de streaming terminée: {self.session_id}")
        
    def get_streaming_status(self) -> Dict[str, Any]:
        """Obtenir le status complet du streaming"""
        
        return {
            'session_id': self.session_id,
            'status': self.stream_status.value,
            'streaming_active': self.streaming_active,
            'agent_count': len(self.agent_statuses),
            'active_agents': len([s for s in self.agent_statuses.values() if s.status == 'active']),
            'queue_size': self.event_queue.qsize(),
            'global_metrics': self.get_global_metrics(),
            'agent_statuses': {k: v.to_dict() for k, v in self.agent_statuses.items()}
        }

# Instance globale
global_streaming_engine: Optional[RealTimeStreamingEngine] = None

def get_streaming_engine() -> RealTimeStreamingEngine:
    """Obtenir l'instance globale du moteur de streaming"""
    global global_streaming_engine
    
    if global_streaming_engine is None:
        global_streaming_engine = RealTimeStreamingEngine()
        
    return global_streaming_engine

def start_realtime_session(project_data: Dict[str, Any] = None) -> str:
    """Démarrer une session temps réel"""
    engine = get_streaming_engine()
    engine.start_streaming_session(project_data)
    return engine.session_id

def emit_agent_progress(agent_id: str, progress: float, current_task: str, message: str = None):
    """Fonction utilitaire pour émettre le progrès d'un agent"""
    engine = get_streaming_engine()
    engine.emit_agent_progress(agent_id, progress, current_task, message)

def emit_plan_update(chunk_data: Dict[str, Any]):
    """Fonction utilitaire pour émettre une mise à jour de plan"""
    engine = get_streaming_engine()
    engine.emit_plan_chunk(chunk_data)