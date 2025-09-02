"""
Event System API - Système de communication inter-modules pour PlannerIA
Communication asynchrone par événements entre tous les modules du dashboard
"""

import asyncio
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import threading
import queue
import logging
from enum import Enum
import hashlib
import json
import copy

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types d'événements pour la communication inter-modules"""
    TASK_UPDATED = "task_updated"
    RISK_ADDED = "risk_added"
    RISK_MITIGATED = "risk_mitigated"
    BUDGET_EXCEEDED = "budget_exceeded"
    DEADLINE_APPROACHING = "deadline_approaching"
    RESOURCE_CONFLICT = "resource_conflict"
    CRITICAL_PATH_CHANGED = "critical_path_changed"
    KPI_THRESHOLD_BREACHED = "kpi_threshold_breached"
    SCENARIO_CREATED = "scenario_created"
    VALIDATION_FAILED = "validation_failed"
    PROJECT_HEALTH_CHANGED = "project_health_changed"
    DATA_INCONSISTENCY = "data_inconsistency"

@dataclass
class Event:
    """Structure d'événement pour communication inter-modules"""
    type: EventType
    source_module: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: str = "normal"  # low, normal, high, critical
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir l'événement en dictionnaire"""
        return {
            "type": self.type.value,
            "source_module": self.source_module,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "correlation_id": self.correlation_id
        }

class EventBus:
    """Bus d'événements centralisé pour communication asynchrone"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        self.event_history: List[Event] = []
        self.max_history = 1000
        self.stats = {
            "events_processed": 0,
            "events_failed": 0,
            "subscribers_count": 0
        }
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """S'abonner à un type d'événement"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        self.stats["subscribers_count"] = sum(len(subs) for subs in self.subscribers.values())
        logger.info(f"Nouvelle subscription: {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Se désabonner d'un type d'événement"""
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                cb for cb in self.subscribers[event_type] if cb != callback
            ]
            self.stats["subscribers_count"] = sum(len(subs) for subs in self.subscribers.values())
    
    def publish(self, event: Event):
        """Publier un événement"""
        try:
            self.event_queue.put(event, timeout=5.0)  # Timeout pour éviter blocage
            self._add_to_history(event)
            logger.debug(f"Événement publié: {event.type.value}")
        except queue.Full:
            logger.error("Queue d'événements pleine, événement perdu")
            self.stats["events_failed"] += 1
    
    def start_processing(self):
        """Démarrer le traitement des événements"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
            self.processing_thread.start()
            logger.info("Bus d'événements démarré")
    
    def stop_processing(self):
        """Arrêter le traitement des événements"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            logger.info("Bus d'événements arrêté")
    
    def _process_events(self):
        """Traiter les événements en arrière-plan"""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._dispatch_event(event)
                self.stats["events_processed"] += 1
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erreur lors du traitement d'événement: {e}")
                self.stats["events_failed"] += 1
    
    def _dispatch_event(self, event: Event):
        """Distribuer un événement aux abonnés"""
        if event.type in self.subscribers:
            for callback in self.subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Erreur dans callback pour {event.type.value}: {e}")
        else:
            logger.debug(f"Aucun abonné pour l'événement: {event.type.value}")
    
    def _add_to_history(self, event: Event):
        """Ajouter l'événement à l'historique"""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
    
    def get_recent_events(self, count: int = 50, event_type: Optional[EventType] = None) -> List[Event]:
        """Obtenir les événements récents avec filtrage optionnel"""
        events = self.event_history[-count:]
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques du bus d'événements"""
        return self.stats.copy()

# Instance globale du bus d'événements
event_bus = EventBus()

class ModuleAPI(ABC):
    """Interface de base pour les APIs de modules"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.event_bus = event_bus
        self.is_initialized = False
        self.cache = {}
        self.last_validation = None
        self.validation_errors = []
    
    @abstractmethod
    def initialize(self, plan_data: Dict[str, Any]):
        """Initialiser le module avec les données du plan"""
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valider les données spécifiques au module"""
        pass
    
    def emit_event(self, event_type: EventType, data: Dict[str, Any], priority: str = "normal"):
        """Émettre un événement"""
        event = Event(
            type=event_type,
            source_module=self.module_name,
            data=data,
            priority=priority
        )
        self.event_bus.publish(event)
    
    def clear_cache(self):
        """Vider le cache du module"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques du cache"""
        return {
            "entries": len(self.cache),
            "module": self.module_name,
            "last_validation": self.last_validation.isoformat() if self.last_validation else None
        }

class GanttAPI(ModuleAPI):
    """API pour le module Gantt"""
    
    def __init__(self):
        super().__init__("gantt")
        self.current_critical_path = []
        self.resource_assignments = {}
        self.task_dependencies = {}
        
        # S'abonner aux événements pertinents
        self.event_bus.subscribe(EventType.RISK_ADDED, self._handle_risk_added)
        self.event_bus.subscribe(EventType.BUDGET_EXCEEDED, self._handle_budget_change)
    
    def initialize(self, plan_data: Dict[str, Any]):
        """Initialiser avec les données du plan"""
        try:
            self.is_initialized = True
            self.current_critical_path = plan_data.get('critical_path', [])
            self._calculate_resource_assignments(plan_data)
            self._build_dependency_graph(plan_data)
            logger.info("Module Gantt initialisé")
        except Exception as e:
            logger.error(f"Erreur initialisation Gantt: {e}")
            self.is_initialized = False
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valider les données Gantt"""
        errors = {"errors": [], "warnings": []}
        
        try:
            tasks = self._extract_tasks(data)
            dependencies = data.get('dependencies', [])
            
            # Valider les tâches
            task_ids = set()
            for task in tasks:
                task_id = task.get('id')
                if not task_id:
                    errors["errors"].append("Tâche sans ID détectée")
                    continue
                
                if task_id in task_ids:
                    errors["errors"].append(f"ID de tâche dupliqué: {task_id}")
                
                task_ids.add(task_id)
                
                # Valider durée
                duration = task.get('duration', 0)
                if duration <= 0:
                    errors["warnings"].append(f"Tâche {task_id} sans durée valide")
                
                # Valider ressources
                resources = task.get('assigned_resources', [])
                if not resources:
                    errors["warnings"].append(f"Tâche {task_id} sans ressources assignées")
            
            # Valider les dépendances
            for dep in dependencies:
                pred = dep.get('predecessor')
                succ = dep.get('successor')
                
                if pred not in task_ids:
                    errors["errors"].append(f"Dépendance invalide: prédécesseur {pred} introuvable")
                if succ not in task_ids:
                    errors["errors"].append(f"Dépendance invalide: successeur {succ} introuvable")
                
                # Valider type de dépendance
                dep_type = dep.get('type', 'finish_to_start')
                valid_types = ['finish_to_start', 'start_to_start', 'finish_to_finish', 'start_to_finish']
                if dep_type not in valid_types:
                    errors["warnings"].append(f"Type de dépendance non standard: {dep_type}")
            
            # Détecter les boucles
            if self._has_circular_dependencies(dependencies):
                errors["errors"].append("Dépendances circulaires détectées")
            
            self.last_validation = datetime.now()
            self.validation_errors = errors["errors"]
            
        except Exception as e:
            errors["errors"].append(f"Erreur lors de la validation Gantt: {str(e)}")
            logger.error(f"Erreur validation Gantt: {e}")
        
        return errors
    
    def update_task_duration(self, task_id: str, new_duration: float, reason: str = ""):
        """Mettre à jour la durée d'une tâche"""
        self.emit_event(
            EventType.TASK_UPDATED,
            {
                "task_id": task_id,
                "field": "duration",
                "old_value": self.cache.get(f"duration_{task_id}", 0),
                "new_value": new_duration,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            },
            priority="high" if new_duration > 30 else "normal"
        )
        
        # Mettre à jour le cache
        self.cache[f"duration_{task_id}"] = new_duration
        
        # Recalculer le chemin critique si nécessaire
        if task_id in self.current_critical_path:
            self.emit_event(
                EventType.CRITICAL_PATH_CHANGED,
                {
                    "affected_task": task_id, 
                    "impact": "duration_change",
                    "new_duration": new_duration
                },
                priority="high"
            )
    
    def detect_resource_conflicts(self, plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Détecter les conflits de ressources"""
        conflicts = []
        
        try:
            tasks = self._extract_tasks(plan_data)
            resource_schedule = {}
            
            # Construire le planning des ressources
            for task in tasks:
                task_id = task.get('id', f"task_{tasks.index(task)}")
                resources = task.get('assigned_resources', [])
                start_date = task.get('start_date')
                end_date = task.get('end_date')
                duration = task.get('duration', 0)
                
                if not start_date or not resources:
                    continue
                
                for resource in resources:
                    if resource not in resource_schedule:
                        resource_schedule[resource] = []
                    
                    resource_schedule[resource].append({
                        'task_id': task_id,
                        'task_name': task.get('name', task_id),
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration': duration,
                        'priority': task.get('priority', 'medium')
                    })
            
            # Détecter les conflits
            for resource, assignments in resource_schedule.items():
                if len(assignments) <= 1:
                    continue
                
                # Trier par date de début
                assignments.sort(key=lambda x: x.get('start_date', ''))
                
                for i in range(len(assignments) - 1):
                    current_task = assignments[i]
                    next_task = assignments[i + 1]
                    
                    # Vérifier le chevauchement temporel
                    if (current_task.get('end_date') and next_task.get('start_date') and
                        current_task['end_date'] > next_task['start_date']):
                        
                        conflicts.append({
                            'type': 'temporal_overlap',
                            'resource': resource,
                            'task1': current_task['task_name'],
                            'task2': next_task['task_name'],
                            'task1_id': current_task['task_id'],
                            'task2_id': next_task['task_id'],
                            'severity': self._calculate_conflict_severity(current_task, next_task),
                            'overlap_days': self._calculate_overlap_days(current_task, next_task)
                        })
            
            # Émettre événement si conflits détectés
            if conflicts:
                self.emit_event(
                    EventType.RESOURCE_CONFLICT,
                    {
                        "conflicts": conflicts, 
                        "count": len(conflicts),
                        "affected_resources": list(set(c['resource'] for c in conflicts))
                    },
                    priority="critical" if len(conflicts) > 5 else "high"
                )
            
        except Exception as e:
            logger.error(f"Erreur détection conflits ressources: {e}")
        
        return conflicts
    
    def _handle_risk_added(self, event: Event):
        """Gérer l'ajout d'un nouveau risque"""
        risk_data = event.data
        if risk_data.get('affects_timeline', False):
            # Suggérer d'ajuster les buffers
            self.emit_event(
                EventType.TASK_UPDATED,
                {
                    "reason": "risk_mitigation",
                    "risk_id": risk_data.get('risk_id'),
                    "suggested_action": "add_buffer",
                    "recommended_buffer_days": 2
                },
                priority="high"
            )
    
    def _handle_budget_change(self, event: Event):
        """Gérer les changements budgétaires"""
        budget_data = event.data
        # Vérifier si cela affecte les ressources
        if budget_data.get('budget_cut', False):
            self.emit_event(
                EventType.RESOURCE_CONFLICT,
                {
                    "reason": "budget_constraint",
                    "suggested_action": "reduce_resource_allocation",
                    "impact": "schedule_extension_likely"
                },
                priority="high"
            )
    
    def _extract_tasks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraire toutes les tâches"""
        tasks = []
        if 'tasks' in data:
            tasks.extend(data['tasks'])
        if 'wbs' in data and 'phases' in data['wbs']:
            for phase in data['wbs']['phases']:
                tasks.extend(phase.get('tasks', []))
        return tasks
    
    def _has_circular_dependencies(self, dependencies: List[Dict[str, Any]]) -> bool:
        """Détecter les dépendances circulaires avec DFS"""
        if not dependencies:
            return False
        
        # Construire le graphe
        graph = {}
        for dep in dependencies:
            pred = dep.get('predecessor')
            succ = dep.get('successor')
            if pred and succ:
                if pred not in graph:
                    graph[pred] = []
                graph[pred].append(succ)
        
        # DFS pour détecter les cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def _calculate_resource_assignments(self, plan_data: Dict[str, Any]):
        """Calculer les assignations de ressources"""
        tasks = self._extract_tasks(plan_data)
        self.resource_assignments = {}
        
        for task in tasks:
            resources = task.get('assigned_resources', [])
            task_id = task.get('id', f"task_{tasks.index(task)}")
            
            for resource in resources:
                if resource not in self.resource_assignments:
                    self.resource_assignments[resource] = []
                self.resource_assignments[resource].append(task_id)
    
    def _build_dependency_graph(self, plan_data: Dict[str, Any]):
        """Construire le graphe des dépendances"""
        dependencies = plan_data.get('dependencies', [])
        self.task_dependencies = {}
        
        for dep in dependencies:
            pred = dep.get('predecessor')
            succ = dep.get('successor')
            if pred and succ:
                if succ not in self.task_dependencies:
                    self.task_dependencies[succ] = []
                self.task_dependencies[succ].append({
                    'predecessor': pred,
                    'type': dep.get('type', 'finish_to_start'),
                    'lag': dep.get('lag', 0)
                })
    
    def _calculate_conflict_severity(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> str:
        """Calculer la sévérité d'un conflit de ressources"""
        priority_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        task1_priority = priority_scores.get(task1.get('priority', 'medium'), 2)
        task2_priority = priority_scores.get(task2.get('priority', 'medium'), 2)
        
        avg_priority = (task1_priority + task2_priority) / 2
        overlap_days = self._calculate_overlap_days(task1, task2)
        
        if avg_priority >= 3.5 or overlap_days > 5:
            return 'critical'
        elif avg_priority >= 2.5 or overlap_days > 2:
            return 'high'
        else:
            return 'medium'
    
    def _calculate_overlap_days(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> int:
        """Calculer le nombre de jours de chevauchement"""
        try:
            end1 = datetime.fromisoformat(task1.get('end_date', ''))
            start2 = datetime.fromisoformat(task2.get('start_date', ''))
            return max(0, (end1 - start2).days)
        except:
            return 0

class RiskAPI(ModuleAPI):
    """API pour le module de gestion des risques"""
    
    def __init__(self):
        super().__init__("risks")
        self.risk_thresholds = {
            "high_risk_score": 15,
            "critical_risk_score": 20,
            "critical_risk_count": 5,
            "risk_exposure_limit": 100
        }
        self.active_risks = {}
        
        self.event_bus.subscribe(EventType.TASK_UPDATED, self._handle_task_update)
        self.event_bus.subscribe(EventType.CRITICAL_PATH_CHANGED, self._handle_critical_path_change)
    
    def initialize(self, plan_data: Dict[str, Any]):
        """Initialiser avec les données de risques"""
        try:
            self.is_initialized = True
            self._assess_initial_risks(plan_data)
            self._build_risk_registry(plan_data)
            logger.info("Module Risk initialisé")
        except Exception as e:
            logger.error(f"Erreur initialisation Risk: {e}")
            self.is_initialized = False
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valider les données de risques"""
        errors = {"errors": [], "warnings": []}
        
        try:
            risks = data.get('risks', [])
            risk_ids = set()
            
            for risk in risks:
                risk_id = risk.get('id')
                if not risk_id:
                    errors["warnings"].append("Risque sans ID détecté")
                elif risk_id in risk_ids:
                    errors["errors"].append(f"ID de risque dupliqué: {risk_id}")
                risk_ids.add(risk_id)
                
                # Validation du nom
                if not risk.get('name'):
                    errors["errors"].append(f"Risque {risk_id or 'inconnu'} sans nom")
                
                # Validation des scores
                probability = risk.get('probability', 0)
                impact = risk.get('impact', 0)
                risk_score = risk.get('risk_score', 0)
                
                if not (1 <= probability <= 5):
                    errors["warnings"].append(f"Probabilité hors limites pour {risk.get('name', risk_id)}: {probability}")
                
                if not (1 <= impact <= 5):
                    errors["warnings"].append(f"Impact hors limites pour {risk.get('name', risk_id)}: {impact}")
                
                # Vérifier cohérence du score de risque
                expected_score = probability * impact
                if abs(risk_score - expected_score) > 0.1:
                    errors["warnings"].append(
                        f"Incohérence score de risque pour {risk.get('name', risk_id)}: "
                        f"attendu {expected_score}, trouvé {risk_score}"
                    )
                
                # Validation des catégories
                category = risk.get('category', 'other')
                valid_categories = ['technical', 'schedule', 'budget', 'resource', 'external', 'other']
                if category not in valid_categories:
                    errors["warnings"].append(f"Catégorie de risque non standard: {category}")
                
                # Validation des stratégies de réponse
                strategy = risk.get('response_strategy', 'accept')
                valid_strategies = ['avoid', 'mitigate', 'transfer', 'accept']
                if strategy not in valid_strategies:
                    errors["warnings"].append(f"Stratégie de réponse non standard: {strategy}")
            
            self.last_validation = datetime.now()
            
        except Exception as e:
            errors["errors"].append(f"Erreur lors de la validation des risques: {str(e)}")
            logger.error(f"Erreur validation risques: {e}")
        
        return errors
    
    def add_risk(self, risk_data: Dict[str, Any]) -> str:
        """Ajouter un nouveau risque"""
        risk_id = risk_data.get('id') or f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Enrichir les données du risque
        enriched_data = {
            **risk_data,
            'id': risk_id,
            'created_at': datetime.now().isoformat(),
            'status': 'identified',
            'affects_timeline': self._affects_timeline(risk_data),
            'affects_budget': self._affects_budget(risk_data)
        }
        
        # Stocker dans le registre
        self.active_risks[risk_id] = enriched_data
        
        # Déterminer la priorité
        risk_score = risk_data.get('risk_score', 0)
        priority = "critical" if risk_score >= 20 else "high" if risk_score >= 15 else "normal"
        
        self.emit_event(
            EventType.RISK_ADDED,
            enriched_data,
            priority=priority
        )
        
        # Vérifier les seuils
        self._check_risk_thresholds()
        
        return risk_id
    
    def mitigate_risk(self, risk_id: str, mitigation_plan: Dict[str, Any]):
        """Appliquer une mitigation de risque"""
        if risk_id in self.active_risks:
            self.active_risks[risk_id]['status'] = 'mitigated'
            self.active_risks[risk_id]['mitigation_plan'] = mitigation_plan
            self.active_risks[risk_id]['mitigated_at'] = datetime.now().isoformat()
        
        self.emit_event(
            EventType.RISK_MITIGATED,
            {
                "risk_id": risk_id,
                "mitigation_plan": mitigation_plan,
                "cost_impact": mitigation_plan.get('cost', 0),
                "duration_impact": mitigation_plan.get('duration_impact', 0),
                "effectiveness": mitigation_plan.get('effectiveness', 0.5)
            },
            priority="normal"
        )
    
    def _handle_task_update(self, event: Event):
        """Gérer les mises à jour de tâches"""
        task_data = event.data
        
        # Tâche très longue - risque potentiel
        if task_data.get('field') == 'duration' and task_data.get('new_value', 0) > 30:
            risk_id = self.add_risk({
                'name': f"Tâche longue: {task_data.get('task_id')}",
                'category': 'schedule',
                'probability': 3,
                'impact': 3,
                'risk_score': 9,
                'description': f"Tâche avec durée élevée ({task_data.get('new_value')} jours)",
                'auto_generated': True,
                'source_task': task_data.get('task_id')
            })
            logger.info(f"Risque auto-généré: {risk_id}")
    
    def _handle_critical_path_change(self, event: Event):
        """Gérer les changements du chemin critique"""
        # Réévaluer les risques liés au chemin critique
        affected_task = event.data.get('affected_task')
        if affected_task:
            # Augmenter la priorité des risques affectant cette tâche
            for risk_id, risk_data in self.active_risks.items():
                if risk_data.get('source_task') == affected_task:
                    self.emit_event(
                        EventType.KPI_THRESHOLD_BREACHED,
                        {
                            "metric": "critical_path_risk",
                            "risk_id": risk_id,
                            "task_id": affected_task,
                            "impact": "schedule_critical"
                        },
                        priority="high"
                    )
    
    def _assess_initial_risks(self, plan_data: Dict[str, Any]):
        """Évaluer les risques initiaux"""
        risks = plan_data.get('risks', [])
        
        critical_risks = [r for r in risks if r.get('risk_score', 0) >= self.risk_thresholds['critical_risk_score']]
        high_risks = [r for r in risks if r.get('risk_score', 0) >= self.risk_thresholds['high_risk_score']]
        
        # Alerte pour risques critiques
        if len(critical_risks) >= self.risk_thresholds['critical_risk_count']:
            self.emit_event(
                EventType.KPI_THRESHOLD_BREACHED,
                {
                    "metric": "critical_risk_count",
                    "value": len(critical_risks),
                    "threshold": self.risk_thresholds['critical_risk_count'],
                    "affected_risks": [r.get('name', r.get('id')) for r in critical_risks]
                },
                priority="critical"
            )
        
        # Calculer l'exposition totale aux risques
        total_exposure = sum(r.get('risk_score', 0) for r in risks)
        if total_exposure > self.risk_thresholds['risk_exposure_limit']:
            self.emit_event(
                EventType.KPI_THRESHOLD_BREACHED,
                {
                    "metric": "total_risk_exposure",
                    "value": total_exposure,
                    "threshold": self.risk_thresholds['risk_exposure_limit']
                },
                priority="high"
            )
    
    def _build_risk_registry(self, plan_data: Dict[str, Any]):
        """Construire le registre des risques"""
        risks = plan_data.get('risks', [])
        for risk in risks:
            risk_id = risk.get('id') or f"risk_{len(self.active_risks)}"
            self.active_risks[risk_id] = {
                **risk,
                'id': risk_id,
                'loaded_at': datetime.now().isoformat()
            }
    
    def _check_risk_thresholds(self):
        """Vérifier les seuils de risques"""
        active_risks_list = list(self.active_risks.values())
        critical_count = len([r for r in active_risks_list if r.get('risk_score', 0) >= 20])
        
        if critical_count >= self.risk_thresholds['critical_risk_count']:
            self.emit_event(
                EventType.PROJECT_HEALTH_CHANGED,
                {
                    "health_factor": "risks",
                    "status": "critical",
                    "critical_risk_count": critical_count
                },
                priority="critical"
            )
    
    def _affects_timeline(self, risk_data: Dict[str, Any]) -> bool:
        """Détermine si le risque affecte la timeline"""
        timeline_categories = ['schedule', 'resource', 'technical', 'external']
        return risk_data.get('category') in timeline_categories
    
    def _affects_budget(self, risk_data: Dict[str, Any]) -> bool:
        """Détermine si le risque affecte le budget"""
        budget_categories = ['budget', 'scope', 'technical', 'external']
        return risk_data.get('category') in budget_categories

class KPIAPI(ModuleAPI):
    """API pour le module KPI"""
    
    def __init__(self):
        super().__init__("kpis")
        self.kpi_thresholds = {
            "budget_warning": 0.8,
            "budget_critical": 0.95,
            "schedule_warning": 0.9,
            "risk_score_critical": 15,
            "health_score_critical": 30,
            "health_score_warning": 50
        }
        self.current_metrics = {}
        
        # S'abonner aux événements
        self.event_bus.subscribe(EventType.TASK_UPDATED, self._recalculate_kpis)
        self.event_bus.subscribe(EventType.RISK_ADDED, self._update_risk_kpis)
        self.event_bus.subscribe(EventType.RISK_MITIGATED, self._update_risk_kpis)
        self.event_bus.subscribe(EventType.RESOURCE_CONFLICT, self._update_resource_kpis)
    
    def initialize(self, plan_data: Dict[str, Any]):
        """Initialiser les KPIs"""
        try:
            self.is_initialized = True
            self._calculate_baseline_kpis(plan_data)
            self._setup_monitoring(plan_data)
            logger.info("Module KPI initialisé")
        except Exception as e:
            logger.error(f"Erreur initialisation KPI: {e}")
            self.is_initialized = False
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valider la cohérence des données pour les KPIs"""
        errors = {"errors": [], "warnings": []}
        
        try:
            # Valider cohérence budget
            tasks = self._extract_tasks(data)
            task_costs = sum(task.get('cost', 0) for task in tasks if task.get('cost', 0) > 0)
            
            project_overview = data.get('project_overview', {})
            project_budget = project_overview.get('total_cost', 0)
            budget_limit = project_overview.get('budget_limit', project_budget)
            
            if task_costs > 0 and project_budget > 0:
                cost_variance = abs(task_costs - project_budget) / project_budget
                if cost_variance > 0.1:  # Plus de 10% d'écart
                    errors["warnings"].append(
                        f"Incohérence budget: tâches={task_costs:,.0f}€, projet={project_budget:,.0f}€ "
                        f"(écart: {cost_variance*100:.1f}%)"
                    )
            
            # Valider cohérence durée
            task_durations = sum(task.get('duration', 0) for task in tasks)
            project_duration = project_overview.get('total_duration', 0)
            
            if task_durations > 0 and project_duration > 0:
                duration_variance = abs(task_durations - project_duration) / project_duration
                if duration_variance > 0.2:  # Plus de 20% d'écart
                    errors["warnings"].append(
                        f"Incohérence durée: somme tâches={task_durations:.1f}j, "
                        f"projet={project_duration:.1f}j (écart: {duration_variance*100:.1f}%)"
                    )
            
            # Valider métriques de risques
            risks = data.get('risks', [])
            if risks:
                risk_scores = [r.get('risk_score', 0) for r in risks]
                invalid_scores = [s for s in risk_scores if not (1 <= s <= 25)]
                if invalid_scores:
                    errors["warnings"].append(f"Scores de risque hors limites: {invalid_scores}")
            
            self.last_validation = datetime.now()
            
        except Exception as e:
            errors["errors"].append(f"Erreur lors de la validation KPI: {str(e)}")
            logger.error(f"Erreur validation KPI: {e}")
        
        return errors
    
    def calculate_health_score(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculer le score de santé global du projet avec détails"""
        factors = {}
        
        try:
            # Facteur risques (40% du score)
            risks = plan_data.get('risks', [])
            if risks:
                critical_risks = len([r for r in risks if r.get('risk_score', 0) >= 20])
                high_risks = len([r for r in risks if 15 <= r.get('risk_score', 0) < 20])
                
                risk_ratio = (critical_risks * 2 + high_risks) / len(risks)
                risk_factor = max(0, 100 - (risk_ratio * 50))
            else:
                risk_factor = 80  # Score neutre sans risques identifiés
            factors['risks'] = {'score': risk_factor, 'weight': 0.4, 'status': self._get_factor_status(risk_factor)}
            
            # Facteur budget (25% du score)
            tasks = self._extract_tasks(plan_data)
            total_cost = sum(task.get('cost', 0) for task in tasks)
            project_overview = plan_data.get('project_overview', {})
            budget_limit = project_overview.get('budget_limit') or project_overview.get('total_cost', total_cost)
            
            if budget_limit > 0:
                budget_usage = min(total_cost / budget_limit, 2.0)  # Cap à 200%
                if budget_usage <= 0.8:
                    budget_factor = 100
                elif budget_usage <= 1.0:
                    budget_factor = 100 - ((budget_usage - 0.8) * 250)
                else:
                    budget_factor = max(0, 50 - ((budget_usage - 1.0) * 50))
            else:
                budget_factor = 50
            
            factors['budget'] = {
                'score': budget_factor, 
                'weight': 0.25, 
                'status': self._get_factor_status(budget_factor),
                'usage_percent': (total_cost / budget_limit * 100) if budget_limit > 0 else 0
            }
            
            # Facteur timeline/complexité (20% du score)
            if tasks:
                avg_duration = sum(task.get('duration', 0) for task in tasks) / len(tasks)
                critical_path_length = len(plan_data.get('critical_path', []))
                
                # Pénaliser les projets très complexes
                complexity_penalty = min(avg_duration * 2 + critical_path_length * 2, 60)
                timeline_factor = max(0, 100 - complexity_penalty)
            else:
                timeline_factor = 50
            
            factors['timeline'] = {
                'score': timeline_factor, 
                'weight': 0.2, 
                'status': self._get_factor_status(timeline_factor),
                'avg_task_duration': avg_duration if tasks else 0,
                'critical_path_length': len(plan_data.get('critical_path', []))
            }
            
            # Facteur validation (15% du score)
            validation_status = plan_data.get('metadata', {}).get('validation_status', 'unknown')
            validation_scores = {'valid': 100, 'corrected': 75, 'failed': 25, 'unknown': 50}
            validation_factor = validation_scores.get(validation_status, 50)
            factors['validation'] = {
                'score': validation_factor, 
                'weight': 0.15, 
                'status': self._get_factor_status(validation_factor)
            }
            
            # Calcul pondéré
            weighted_sum = sum(factor['score'] * factor['weight'] for factor in factors.values())
            overall_score = max(0, min(100, weighted_sum))
            
            # Stocker dans le cache
            health_data = {
                'overall_score': overall_score,
                'factors': factors,
                'calculated_at': datetime.now().isoformat(),
                'status': self._get_factor_status(overall_score)
            }
            
            self.current_metrics['health_score'] = health_data
            
            # Vérifier les seuils
            self.check_thresholds(plan_data)
            
            return health_data
            
        except Exception as e:
            logger.error(f"Erreur calcul score de santé: {e}")
            return {'overall_score': 50, 'factors': {}, 'status': 'error'}
    
    def check_thresholds(self, plan_data: Dict[str, Any]):
        """Vérifier les seuils KPI et émettre des alertes"""
        try:
            health_data = self.current_metrics.get('health_score', {})
            overall_score = health_data.get('overall_score', 50)
            
            if overall_score < self.kpi_thresholds['health_score_critical']:
                self.emit_event(
                    EventType.KPI_THRESHOLD_BREACHED,
                    {
                        "metric": "project_health",
                        "value": overall_score,
                        "threshold": self.kpi_thresholds['health_score_critical'],
                        "severity": "critical",
                        "factors": health_data.get('factors', {}),
                        "recommended_actions": self._get_health_recommendations(health_data)
                    },
                    priority="critical"
                )
            elif overall_score < self.kpi_thresholds['health_score_warning']:
                self.emit_event(
                    EventType.KPI_THRESHOLD_BREACHED,
                    {
                        "metric": "project_health",
                        "value": overall_score,
                        "threshold": self.kpi_thresholds['health_score_warning'],
                        "severity": "warning",
                        "factors": health_data.get('factors', {})
                    },
                    priority="high"
                )
            
            # Vérifier budget séparément
            factors = health_data.get('factors', {})
            budget_factor = factors.get('budget', {})
            budget_usage = budget_factor.get('usage_percent', 0)
            
            if budget_usage > self.kpi_thresholds['budget_critical'] * 100:
                self.emit_event(
                    EventType.BUDGET_EXCEEDED,
                    {
                        "usage_percent": budget_usage,
                        "threshold": self.kpi_thresholds['budget_critical'] * 100,
                        "severity": "critical"
                    },
                    priority="critical"
                )
            elif budget_usage > self.kpi_thresholds['budget_warning'] * 100:
                self.emit_event(
                    EventType.KPI_THRESHOLD_BREACHED,
                    {
                        "metric": "budget_usage",
                        "value": budget_usage,
                        "threshold": self.kpi_thresholds['budget_warning'] * 100,
                        "severity": "warning"
                    },
                    priority="high"
                )
                
        except Exception as e:
            logger.error(f"Erreur vérification seuils KPI: {e}")
    
    def _recalculate_kpis(self, event: Event):
        """Recalculer les KPIs suite à une mise à jour"""
        try:
            # Marquer les métriques comme nécessitant une mise à jour
            self.cache['needs_recalculation'] = True
            self.cache['last_update_event'] = event.to_dict()
            
            # Si changement majeur, recalculer immédiatement
            if event.priority in ['high', 'critical']:
                # Ici on devrait avoir accès aux plan_data, 
                # mais pour l'instant on marque juste le besoin
                pass
                
        except Exception as e:
            logger.error(f"Erreur recalcul KPI: {e}")
    
    def _update_risk_kpis(self, event: Event):
        """Mettre à jour les KPIs liés aux risques"""
        risk_data = event.data
        
        if event.type == EventType.RISK_ADDED:
            risk_score = risk_data.get('risk_score', 0)
            if risk_score >= 20:
                self.emit_event(
                    EventType.PROJECT_HEALTH_CHANGED,
                    {
                        "health_factor": "risks",
                        "change": "degraded",
                        "reason": "critical_risk_added",
                        "risk_id": risk_data.get('id')
                    },
                    priority="high"
                )
        elif event.type == EventType.RISK_MITIGATED:
            self.emit_event(
                EventType.PROJECT_HEALTH_CHANGED,
                {
                    "health_factor": "risks",
                    "change": "improved", 
                    "reason": "risk_mitigated",
                    "risk_id": risk_data.get('risk_id')
                },
                priority="normal"
            )
    
    def _update_resource_kpis(self, event: Event):
        """Mettre à jour les KPIs liés aux ressources"""
        conflict_data = event.data
        conflict_count = conflict_data.get('count', 0)
        
        if conflict_count > 5:
            self.emit_event(
                EventType.PROJECT_HEALTH_CHANGED,
                {
                    "health_factor": "resources",
                    "change": "degraded",
                    "reason": "major_resource_conflicts",
                    "conflict_count": conflict_count
                },
                priority="high"
            )
    
    def _calculate_baseline_kpis(self, plan_data: Dict[str, Any]):
        """Calculer les KPIs de base"""
        try:
            # Calculer le score de santé initial
            self.calculate_health_score(plan_data)
            
            # Autres métriques de base
            tasks = self._extract_tasks(plan_data)
            risks = plan_data.get('risks', [])
            
            self.current_metrics.update({
                'task_count': len(tasks),
                'total_duration': sum(task.get('duration', 0) for task in tasks),
                'total_cost': sum(task.get('cost', 0) for task in tasks),
                'risk_count': len(risks),
                'high_risk_count': len([r for r in risks if r.get('risk_score', 0) >= 15]),
                'critical_path_length': len(plan_data.get('critical_path', [])),
                'baseline_calculated_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Erreur calcul KPIs de base: {e}")
    
    def _setup_monitoring(self, plan_data: Dict[str, Any]):
        """Configurer le monitoring des seuils"""
        # Configuration du monitoring automatique
        self.cache['monitoring_active'] = True
        self.cache['last_check'] = datetime.now().isoformat()
        
    def _extract_tasks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraire toutes les tâches"""
        tasks = []
        if 'tasks' in data:
            tasks.extend(data['tasks'])
        if 'wbs' in data and 'phases' in data['wbs']:
            for phase in data['wbs']['phases']:
                tasks.extend(phase.get('tasks', []))
        return tasks
    
    def _get_factor_status(self, score: float) -> str:
        """Obtenir le statut d'un facteur basé sur son score"""
        if score >= 80:
            return 'excellent'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'warning'
        else:
            return 'critical'
    
    def _get_health_recommendations(self, health_data: Dict[str, Any]) -> List[str]:
        """Obtenir des recommandations basées sur la santé du projet"""
        recommendations = []
        factors = health_data.get('factors', {})
        
        # Recommandations basées sur les risques
        risk_factor = factors.get('risks', {})
        if risk_factor.get('status') == 'critical':
            recommendations.append("Mitigation urgente des risques critiques")
        elif risk_factor.get('status') == 'warning':
            recommendations.append("Révision et mise à jour du plan de gestion des risques")
        
        # Recommandations basées sur le budget
        budget_factor = factors.get('budget', {})
        if budget_factor.get('status') == 'critical':
            recommendations.append("Révision budgétaire d'urgence et contrôle des dépenses")
        elif budget_factor.get('status') == 'warning':
            recommendations.append("Surveillance renforcée du budget")
        
        # Recommandations basées sur la timeline
        timeline_factor = factors.get('timeline', {})
        if timeline_factor.get('status') == 'critical':
            recommendations.append("Optimisation du chemin critique et révision du planning")
        
        return recommendations

class WhatIfAPI(ModuleAPI):
    """API pour le simulateur What-If"""
    
    def __init__(self):
        super().__init__("whatif")
        self.active_scenarios = {}
        self.scenario_results_cache = {}
        
        # S'abonner aux événements pour mise à jour automatique des scénarios
        self.event_bus.subscribe(EventType.TASK_UPDATED, self._update_scenarios)
        self.event_bus.subscribe(EventType.RISK_ADDED, self._update_scenarios)
        self.event_bus.subscribe(EventType.BUDGET_EXCEEDED, self._update_scenarios)
    
    def initialize(self, plan_data: Dict[str, Any]):
        """Initialiser le simulateur"""
        try:
            self.is_initialized = True
            self._create_baseline_scenario(plan_data)
            self._setup_scenario_monitoring()
            logger.info("Module WhatIf initialisé")
        except Exception as e:
            logger.error(f"Erreur initialisation WhatIf: {e}")
            self.is_initialized = False
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Valider les données de simulation"""
        errors = {"errors": [], "warnings": []}
        
        try:
            # Valider les scénarios existants
            scenarios = data.get('scenarios', [])
            scenario_names = set()
            
            for scenario in scenarios:
                name = scenario.get('name')
                if not name:
                    errors["errors"].append("Scénario sans nom détecté")
                elif name in scenario_names:
                    errors["errors"].append(f"Nom de scénario dupliqué: {name}")
                scenario_names.add(name)
                
                # Valider les paramètres
                params = scenario.get('parameters', {})
                
                duration_mult = params.get('duration_multiplier', 1.0)
                if duration_mult <= 0:
                    errors["errors"].append(f"Multiplicateur de durée invalide dans {name}: {duration_mult}")
                elif duration_mult > 5.0:
                    errors["warnings"].append(f"Multiplicateur de durée très élevé dans {name}: {duration_mult}")
                
                cost_mult = params.get('cost_multiplier', 1.0)
                if cost_mult <= 0:
                    errors["errors"].append(f"Multiplicateur de coût invalide dans {name}: {cost_mult}")
                elif cost_mult > 10.0:
                    errors["warnings"].append(f"Multiplicateur de coût très élevé dans {name}: {cost_mult}")
                
                team_mult = params.get('team_multiplier', 1.0)
                if team_mult <= 0:
                    errors["errors"].append(f"Multiplicateur d'équipe invalide dans {name}: {team_mult}")
                elif team_mult > 5.0:
                    errors["warnings"].append(f"Multiplicateur d'équipe très élevé dans {name}: {team_mult}")
                
                risk_tolerance = params.get('risk_tolerance', 0.5)
                if not (0 <= risk_tolerance <= 1):
                    errors["warnings"].append(f"Tolérance au risque hors limites dans {name}: {risk_tolerance}")
            
            self.last_validation = datetime.now()
            
        except Exception as e:
            errors["errors"].append(f"Erreur lors de la validation What-If: {str(e)}")
            logger.error(f"Erreur validation WhatIf: {e}")
        
        return errors
    
    def create_scenario(self, scenario_name: str, parameters: Dict[str, Any]) -> str:
        """Créer un nouveau scénario"""
        scenario_id = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Valider les paramètres
        validated_params = self._validate_scenario_parameters(parameters)
        
        scenario_data = {
            "id": scenario_id,
            "name": scenario_name,
            "parameters": validated_params,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "results": None,
            "last_updated": None
        }
        
        self.active_scenarios[scenario_id] = scenario_data
        
        self.emit_event(
            EventType.SCENARIO_CREATED,
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "parameters": validated_params
            },
            priority="normal"
        )
        
        return scenario_id
    
    def simulate_scenario(self, scenario_id: str, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simuler un scénario spécifique"""
        if scenario_id not in self.active_scenarios:
            raise ValueError(f"Scénario {scenario_id} introuvable")
        
        scenario = self.active_scenarios[scenario_id]
        parameters = scenario['parameters']
        
        try:
            # Vérifier le cache
            cache_key = self._generate_cache_key(scenario_id, plan_data)
            if cache_key in self.scenario_results_cache:
                cached_result = self.scenario_results_cache[cache_key]
                if (datetime.now() - datetime.fromisoformat(cached_result['calculated_at'])).seconds < 300:  # 5 min
                    return cached_result
            
            # Calculer les impacts
            results = self._calculate_scenario_impacts(plan_data, parameters)
            
            # Enrichir avec des métriques additionnelles
            results.update({
                "scenario_id": scenario_id,
                "scenario_name": scenario['name'],
                "calculated_at": datetime.now().isoformat(),
                "baseline_comparison": self._compare_to_baseline(results, plan_data),
                "confidence_level": self._calculate_confidence(parameters),
                "recommendations": self._generate_scenario_recommendations(results, parameters)
            })
            
            # Stocker les résultats
            self.active_scenarios[scenario_id]['results'] = results
            self.active_scenarios[scenario_id]['last_updated'] = datetime.now().isoformat()
            self.active_scenarios[scenario_id]['status'] = 'completed'
            
            # Mettre en cache
            self.scenario_results_cache[cache_key] = results
            
            # Émettre événement si résultats critiques
            if results.get('success_probability', 1.0) < 0.3:
                self.emit_event(
                    EventType.KPI_THRESHOLD_BREACHED,
                    {
                        "metric": "scenario_success_probability",
                        "scenario_id": scenario_id,
                        "value": results['success_probability'],
                        "threshold": 0.3,
                        "severity": "critical"
                    },
                    priority="high"
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur simulation scénario {scenario_id}: {e}")
            error_result = {
                "scenario_id": scenario_id,
                "error": str(e),
                "calculated_at": datetime.now().isoformat(),
                "status": "failed"
            }
            self.active_scenarios[scenario_id]['status'] = 'failed'
            return error_result
    
    def _update_scenarios(self, event: Event):
        """Mettre à jour les scénarios suite aux changements"""
        try:
            # Marquer tous les scénarios comme nécessitant une mise à jour
            for scenario_id in self.active_scenarios:
                self.active_scenarios[scenario_id]['needs_update'] = True
                self.active_scenarios[scenario_id]['update_reason'] = f"{event.type.value} from {event.source_module}"
            
            # Vider le cache des résultats
            self.scenario_results_cache.clear()
            
            # Émettre notification de mise à jour
            self.emit_event(
                EventType.PROJECT_HEALTH_CHANGED,
                {
                    "health_factor": "scenarios",
                    "change": "needs_recalculation",
                    "trigger_event": event.type.value,
                    "affected_scenarios": len(self.active_scenarios)
                },
                priority="normal"
            )
            
        except Exception as e:
            logger.error(f"Erreur mise à jour scénarios: {e}")
    
    def _create_baseline_scenario(self, plan_data: Dict[str, Any]):
        """Créer le scénario de base"""
        baseline_params = {
            "duration_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "team_multiplier": 1.0,
            "complexity_factor": 1.0,
            "risk_tolerance": 0.5
        }
        
        baseline_id = self.create_scenario("Baseline", baseline_params)
        
        # Marquer comme scénario de référence
        self.active_scenarios[baseline_id]['is_baseline'] = True
        self.cache['baseline_scenario_id'] = baseline_id
    
    def _setup_scenario_monitoring(self):
        """Configurer le monitoring des scénarios"""
        self.cache['monitoring_enabled'] = True
        self.cache['auto_update_scenarios'] = True
        self.cache['last_monitoring_check'] = datetime.now().isoformat()
    
    def _validate_scenario_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Valider et normaliser les paramètres de scénario"""
        defaults = {
            "duration_multiplier": 1.0,
            "cost_multiplier": 1.0,
            "team_multiplier": 1.0,
            "complexity_factor": 1.0,
            "risk_tolerance": 0.5
        }
        
        validated = {}
        for key, default_value in defaults.items():
            value = parameters.get(key, default_value)
            
            # Appliquer des limites raisonnables
            if key == 'risk_tolerance':
                validated[key] = max(0.0, min(1.0, value))
            else:
                validated[key] = max(0.1, min(10.0, value))  # Entre 0.1x et 10x
        
        return validated
    
    def _calculate_scenario_impacts(self, plan_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculer les impacts d'un scénario avec logique avancée"""
        try:
            # Extraire les données de base
            tasks = self._extract_tasks(plan_data)
            risks = plan_data.get('risks', [])
            project_overview = plan_data.get('project_overview', {})
            
            base_duration = sum(task.get('duration', 0) for task in tasks)
            base_cost = sum(task.get('cost', 0) for task in tasks)
            base_team_size = len(set().union(*[task.get('assigned_resources', []) for task in tasks]))
            
            # Appliquer les multiplicateurs de base
            duration_mult = parameters.get('duration_multiplier', 1.0)
            cost_mult = parameters.get('cost_multiplier', 1.0)
            team_mult = parameters.get('team_multiplier', 1.0)
            complexity = parameters.get('complexity_factor', 1.0)
            risk_tolerance = parameters.get('risk_tolerance', 0.5)
            
            # Calculs avancés avec effets non-linéaires
            
            # Effet Brooks' Law: ajouter des ressources peut ralentir
            communication_overhead = 1.0
            if team_mult > 1.2:
                communication_overhead = 1 + (team_mult - 1) * 0.15
                duration_mult *= communication_overhead
            
            # Économies d'échelle pour les équipes optimales
            if 1.0 < team_mult <= 1.5:
                efficiency_gain = 0.95 - (0.05 / team_mult)
                duration_mult *= efficiency_gain
            
            # Effet de la complexité sur durée et coût
            complexity_duration_impact = complexity ** 0.8
            complexity_cost_impact = complexity ** 1.2
            
            # Calcul des nouvelles valeurs
            new_duration = base_duration * duration_mult * complexity_duration_impact
            new_cost = base_cost * cost_mult * team_mult * complexity_cost_impact
            new_team_size = base_team_size * team_mult
            
            # Calculs des facteurs de risque
            risk_factors = self._calculate_risk_factors(parameters, base_duration, base_cost, base_team_size)
            
            # Score de risque ajusté
            base_risk_score = sum(r.get('risk_score', 0) for r in risks) / len(risks) if risks else 5.0
            adjusted_risk_score = base_risk_score + risk_factors['total_risk_increase'] * (1 - risk_tolerance)
            
            # Calcul de la probabilité de succès
            success_factors = self._calculate_success_factors(parameters, risk_factors, complexity)
            success_probability = max(0.1, min(1.0, success_factors['overall_probability']))
            
            # Métriques de performance
            performance_metrics = self._calculate_performance_metrics(
                new_duration, new_cost, new_team_size, adjusted_risk_score
            )
            
            return {
                "duration_impact": new_duration,
                "cost_impact": new_cost,
                "team_size_impact": new_team_size,
                "risk_score": adjusted_risk_score,
                "success_probability": success_probability,
                "risk_factors": risk_factors,
                "success_factors": success_factors,
                "performance_metrics": performance_metrics,
                "efficiency_ratio": base_duration / new_duration if new_duration > 0 else 0,
                "cost_efficiency": base_cost / new_cost if new_cost > 0 else 0,
                "resource_efficiency": base_team_size / new_team_size if new_team_size > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul impacts scénario: {e}")
            return {
                "error": str(e),
                "duration_impact": 0,
                "cost_impact": 0,
                "success_probability": 0.5
            }
    
    def _calculate_risk_factors(self, parameters: Dict[str, Any], 
                               base_duration: float, base_cost: float, base_team_size: int) -> Dict[str, Any]:
        """Calculer les facteurs de risque pour un scénario"""
        factors = {
            "timeline_pressure": 0,
            "resource_risk": 0,
            "budget_risk": 0,
            "complexity_risk": 0,
            "coordination_risk": 0,
            "total_risk_increase": 0
        }
        
        duration_mult = parameters.get('duration_multiplier', 1.0)
        cost_mult = parameters.get('cost_multiplier', 1.0)
        team_mult = parameters.get('team_multiplier', 1.0)
        complexity = parameters.get('complexity_factor', 1.0)
        
        # Risque de pression temporelle
        if duration_mult < 0.8:
            factors["timeline_pressure"] = (0.8 - duration_mult) * 15
        
        # Risque de ressources
        if team_mult < 0.7:
            factors["resource_risk"] = (0.7 - team_mult) * 12
        elif team_mult > 2.0:
            factors["coordination_risk"] = (team_mult - 2.0) * 8
        
        # Risque budgétaire
        if cost_mult < 0.8:
            factors["budget_risk"] = (0.8 - cost_mult) * 10
        
        # Risque de complexité
        if complexity > 1.3:
            factors["complexity_risk"] = (complexity - 1.3) * 6
        
        factors["total_risk_increase"] = sum(v for k, v in factors.items() if k != "total_risk_increase")
        
        return factors
    
    def _calculate_success_factors(self, parameters: Dict[str, Any], 
                                 risk_factors: Dict[str, Any], complexity: float) -> Dict[str, Any]:
        """Calculer les facteurs de succès"""
        factors = {}
        
        duration_mult = parameters.get('duration_multiplier', 1.0)
        cost_mult = parameters.get('cost_multiplier', 1.0)
        team_mult = parameters.get('team_multiplier', 1.0)
        risk_tolerance = parameters.get('risk_tolerance', 0.5)
        
        # Facteur timeline
        factors['timeline_factor'] = max(0.2, 1 - abs(duration_mult - 1) * 0.4)
        
        # Facteur budget
        factors['budget_factor'] = max(0.3, 1 - abs(cost_mult - 1) * 0.3)
        
        # Facteur équipe
        if 0.8 <= team_mult <= 1.5:
            factors['team_factor'] = 1.0
        else:
            factors['team_factor'] = max(0.4, 1 - abs(team_mult - 1) * 0.5)
        
        # Facteur complexité
        factors['complexity_factor'] = max(0.2, 1 - (complexity - 1) * 0.6)
        
        # Facteur risque
        total_risk = risk_factors.get('total_risk_increase', 0)
        factors['risk_factor'] = max(0.1, 1 - (total_risk * 0.02))
        
        # Facteur tolérance au risque
        factors['risk_tolerance_factor'] = 0.5 + (risk_tolerance * 0.5)
        
        # Calcul de la probabilité globale
        base_probability = 1.0
        for factor_name, factor_value in factors.items():
            if factor_name != 'risk_tolerance_factor':
                base_probability *= factor_value
        
        # Ajustement par tolérance au risque
        factors['overall_probability'] = base_probability * factors['risk_tolerance_factor']
        
        return factors
    
    def _calculate_performance_metrics(self, duration: float, cost: float, 
                                     team_size: float, risk_score: float) -> Dict[str, Any]:
        """Calculer les métriques de performance du scénario"""
        return {
            "schedule_performance_index": 1.0,  # À calculer selon vos critères
            "cost_performance_index": 1.0,     # À calculer selon vos critères
            "quality_index": max(0.1, 1 - (risk_score / 25)),
            "efficiency_score": min(100, max(0, 100 - (duration * 0.5 + cost * 0.0001))),
            "overall_performance": (100 - risk_score * 2) / 100
        }
    
    def _compare_to_baseline(self, results: Dict[str, Any], plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comparer les résultats au scénario de base"""
        baseline_id = self.cache.get('baseline_scenario_id')
        if not baseline_id or baseline_id not in self.active_scenarios:
            return {"error": "Baseline scenario not found"}
        
        baseline_scenario = self.active_scenarios[baseline_id]
        baseline_results = baseline_scenario.get('results')
        
        if not baseline_results:
            # Calculer le baseline si nécessaire
            baseline_params = baseline_scenario['parameters']
            baseline_results = self._calculate_scenario_impacts(plan_data, baseline_params)
        
        return {
            "duration_change": results.get('duration_impact', 0) - baseline_results.get('duration_impact', 0),
            "cost_change": results.get('cost_impact', 0) - baseline_results.get('cost_impact', 0),
            "success_probability_change": results.get('success_probability', 0) - baseline_results.get('success_probability', 0),
            "risk_score_change": results.get('risk_score', 0) - baseline_results.get('risk_score', 0),
            "performance_change": (results.get('performance_metrics', {}).get('overall_performance', 0) - 
                                 baseline_results.get('performance_metrics', {}).get('overall_performance', 0))
        }
    
    def _calculate_confidence(self, parameters: Dict[str, Any]) -> float:
        """Calculer le niveau de confiance du scénario"""
        # Plus les paramètres sont proches de 1.0, plus la confiance est élevée
        deviations = []
        for param, value in parameters.items():
            if param != 'risk_tolerance':
                deviations.append(abs(value - 1.0))
        
        avg_deviation = sum(deviations) / len(deviations) if deviations else 0
        confidence = max(0.1, 1 - (avg_deviation * 0.5))
        
        return confidence
    
    def _generate_scenario_recommendations(self, results: Dict[str, Any], 
                                         parameters: Dict[str, Any]) -> List[str]:
        """Générer des recommandations pour le scénario"""
        recommendations = []
        
        success_prob = results.get('success_probability', 0.5)
        if success_prob < 0.3:
            recommendations.append("Probabilité de succès très faible - réviser les paramètres")
        elif success_prob < 0.5:
            recommendations.append("Probabilité de succès modérée - envisager des mesures de mitigation")
        
        risk_factors = results.get('risk_factors', {})
        if risk_factors.get('timeline_pressure', 0) > 10:
            recommendations.append("Pression temporelle élevée - ajouter des buffers au planning")
        
        if risk_factors.get('coordination_risk', 0) > 5:
            recommendations.append("Risque de coordination élevé - renforcer la communication d'équipe")
        
        if risk_factors.get('budget_risk', 0) > 8:
            recommendations.append("Contraintes budgétaires importantes - réviser le périmètre")
        
        return recommendations
    
    def _generate_cache_key(self, scenario_id: str, plan_data: Dict[str, Any]) -> str:
        """Générer une clé de cache pour les résultats de scénario"""
        plan_hash = hashlib.md5(json.dumps(plan_data, sort_keys=True).encode()).hexdigest()
        return f"{scenario_id}_{plan_hash[:8]}"
    
    def _extract_tasks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraire toutes les tâches"""
        tasks = []
        if 'tasks' in data:
            tasks.extend(data['tasks'])
        if 'wbs' in data and 'phases' in data['wbs']:
            for phase in data['wbs']['phases']:
                tasks.extend(phase.get('tasks', []))
        return tasks

class APIManager:
    """Gestionnaire central des APIs de modules"""
    
    def __init__(self):
        self.apis = {
            "gantt": GanttAPI(),
            "risks": RiskAPI(),
            "kpis": KPIAPI(),
            "whatif": WhatIfAPI()
        }
        
        self.is_initialized = False
        
        # Démarrer le bus d'événements
        try:
            event_bus.start_processing()
            self.is_initialized = True
            logger.info("APIManager initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur initialisation APIManager: {e}")
    
    def initialize_all(self, plan_data: Dict[str, Any]):
        """Initialiser tous les modules avec validation"""
        initialization_results = {}
        
        for name, api in self.apis.items():
            try:
                api.initialize(plan_data)
                initialization_results[name] = {
                    "status": "success",
                    "initialized": api.is_initialized,
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"Module {name} initialisé avec succès")
                
            except Exception as e:
                initialization_results[name] = {
                    "status": "failed",
                    "error": str(e),
                    "initialized": False,
                    "timestamp": datetime.now().isoformat()
                }
                logger.error(f"Erreur lors de l'initialisation de {name}: {e}")
        
        # Émettre un événement de fin d'initialisation
        event_bus.publish(Event(
            type=EventType.PROJECT_HEALTH_CHANGED,
            source_module="api_manager",
            data={
                "initialization_results": initialization_results,
                "all_modules_initialized": all(r["initialized"] for r in initialization_results.values())
            }
        ))
        
        return initialization_results
    
    def validate_all(self, plan_data: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        """Valider les données avec tous les modules"""
        validation_results = {}
        
        for name, api in self.apis.items():
            try:
                if api.is_initialized:
                    validation_results[name] = api.validate_data(plan_data)
                else:
                    validation_results[name] = {
                        "errors": ["Module non initialisé"],
                        "warnings": []
                    }
            except Exception as e:
                validation_results[name] = {
                    "errors": [f"Erreur de validation: {str(e)}"],
                    "warnings": []
                }
                logger.error(f"Erreur validation {name}: {e}")
        
        # Analyser les résultats globaux
        total_errors = sum(len(v["errors"]) for v in validation_results.values())
        total_warnings = sum(len(v["warnings"]) for v in validation_results.values())
        
        if total_errors > 0:
            event_bus.publish(Event(
                type=EventType.VALIDATION_FAILED,
                source_module="api_manager",
                data={
                    "total_errors": total_errors,
                    "total_warnings": total_warnings,
                    "validation_results": validation_results
                },
                priority="high"
            ))
        
        return validation_results
    
    def get_api(self, module_name: str) -> Optional[ModuleAPI]:
        """Obtenir l'API d'un module spécifique"""
        return self.apis.get(module_name)
    
    def get_module_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de tous les modules"""
        stats = {
            "event_bus": event_bus.get_stats(),
            "modules": {}
        }
        
        for name, api in self.apis.items():
            stats["modules"][name] = {
                "initialized": api.is_initialized,
                "last_validation": api.last_validation.isoformat() if api.last_validation else None,
                "validation_errors_count": len(api.validation_errors),
                "cache_stats": api.get_cache_stats()
            }
        
        return stats
    
    def clear_all_caches(self):
        """Vider tous les caches des modules"""
        for api in self.apis.values():
            api.clear_cache()
        
        logger.info("Tous les caches des modules ont été vidés")
    
    def shutdown(self):
        """Arrêter le gestionnaire et nettoyer les ressources"""
        try:
            # Arrêter le bus d'événements
            event_bus.stop_processing()
            
            # Nettoyer les caches
            self.clear_all_caches()
            
            self.is_initialized = False
            logger.info("APIManager arrêté proprement")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt de l'APIManager: {e}")

# Instance globale du gestionnaire d'APIs
api_manager = APIManager()

# === FONCTIONS UTILITAIRES ===

def get_event_history(count: int = 50, event_type: Optional[EventType] = None) -> List[Dict[str, Any]]:
    """Obtenir l'historique des événements sous forme de dictionnaires"""
    events = event_bus.get_recent_events(count, event_type)
    return [event.to_dict() for event in events]

def get_system_health() -> Dict[str, Any]:
    """Obtenir l'état de santé global du système d'API"""
    return {
        "api_manager_initialized": api_manager.is_initialized,
        "event_bus_running": event_bus.is_running,
        "modules_stats": api_manager.get_module_stats(),
        "last_check": datetime.now().isoformat()
    }

def trigger_system_validation(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Déclencher une validation complète du système"""
    if not api_manager.is_initialized:
        return {"error": "APIManager not initialized"}
    
    # Initialiser tous les modules
    init_results = api_manager.initialize_all(plan_data)
    
    # Valider toutes les données
    validation_results = api_manager.validate_all(plan_data)
    
    return {
        "initialization": init_results,
        "validation": validation_results,
        "system_health": get_system_health(),
        "timestamp": datetime.now().isoformat()
    }

# === EXEMPLES D'UTILISATION ===

def example_usage():
    """Exemples d'utilisation du système d'API complet"""
    
    # Simuler des données de plan
    sample_plan = {
        "project_overview": {
            "title": "Projet Test API",
            "total_cost": 100000,
            "budget_limit": 120000,
            "total_duration": 45
        },
        "tasks": [
            {
                "id": "task1", 
                "name": "Analyse des besoins", 
                "duration": 5, 
                "cost": 15000,
                "assigned_resources": ["Analyst1", "Designer1"],
                "priority": "high"
            },
            {
                "id": "task2", 
                "name": "Développement", 
                "duration": 25, 
                "cost": 60000,
                "assigned_resources": ["Dev1", "Dev2", "Designer1"],
                "priority": "critical"
            },
            {
                "id": "task3", 
                "name": "Tests", 
                "duration": 15, 
                "cost": 25000,
                "assigned_resources": ["Tester1", "Dev1"],
                "priority": "medium"
            }
        ],
        "dependencies": [
            {"predecessor": "task1", "successor": "task2", "type": "finish_to_start", "lag": 0},
            {"predecessor": "task2", "successor": "task3", "type": "finish_to_start", "lag": 2}
        ],
        "risks": [
            {
                "id": "risk1", 
                "name": "Risque technique majeur", 
                "category": "technical",
                "probability": 4, 
                "impact": 5, 
                "risk_score": 20,
                "response_strategy": "mitigate"
            },
            {
                "id": "risk2", 
                "name": "Retard fournisseur", 
                "category": "external",
                "probability": 3, 
                "impact": 3, 
                "risk_score": 9,
                "response_strategy": "transfer"
            }
        ],
        "critical_path": ["task1", "task2", "task3"],
        "metadata": {
            "validation_status": "valid",
            "model_used": "gpt-4",
            "created_at": datetime.now().isoformat()
        }
    }
    
    print("=== Test du système API complet ===")
    
    # 1. Validation système complète
    print("\n1. Validation système complète...")
    system_validation = trigger_system_validation(sample_plan)
    print(f"Statut validation: {system_validation}")
    
    # 2. Test des APIs individuelles
    print("\n2. Test des APIs individuelles...")
    
    # Test Gantt API
    gantt_api = api_manager.get_api("gantt")
    if gantt_api:
        print("Test Gantt API...")
        gantt_api.update_task_duration("task2", 30.0, "Révision des estimations")
        conflicts = gantt_api.detect_resource_conflicts(sample_plan)
        print(f"Conflits de ressources détectés: {len(conflicts)}")
    
    # Test Risk API
    risk_api = api_manager.get_api("risks")
    if risk_api:
        print("Test Risk API...")
        new_risk_id = risk_api.add_risk({
            "name": "Nouveau risque budget",
            "category": "budget",
            "probability": 3,
            "impact": 4,
            "risk_score": 12,
            "description": "Dépassement budgétaire potentiel"
        })
        print(f"Nouveau risque créé: {new_risk_id}")
    
    # Test KPI API
    kpi_api = api_manager.get_api("kpis")
    if kpi_api:
        print("Test KPI API...")
        health_score = kpi_api.calculate_health_score(sample_plan)
        print(f"Score de santé projet: {health_score.get('overall_score', 0):.1f}%")
    
    # Test What-If API
    whatif_api = api_manager.get_api("whatif")
    if whatif_api:
        print("Test What-If API...")
        scenario_id = whatif_api.create_scenario("Scénario accéléré", {
            "duration_multiplier": 0.8,
            "cost_multiplier": 1.2,
            "team_multiplier": 1.5,
            "risk_tolerance": 0.7
        })
        results = whatif_api.simulate_scenario(scenario_id, sample_plan)
        print(f"Résultats scénario - Probabilité succès: {results.get('success_probability', 0):.2f}")
    
    # 3. Test de l'historique des événements
    print("\n3. Historique des événements...")
    recent_events = get_event_history(10)
    print(f"Derniers événements: {len(recent_events)}")
    for event in recent_events[-3:]:  # Derniers 3 événements
        print(f"  - {event['type']} par {event['source_module']} à {event['timestamp'][:19]}")
    
    # 4. Statistiques système
    print("\n4. Statistiques système...")
    system_health = get_system_health()
    print(f"Santé système: {system_health}")
    
    print("\n=== Test terminé ===")

if __name__ == "__main__":
    example_usage()
    
    # Nettoyer à la fin
    import time
    time.sleep(2)  # Laisser le temps au bus d'événements de traiter
    api_manager.shutdown()