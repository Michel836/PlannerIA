"""
Behavioral Analysis Module for Smart Recommendations
Module d'analyse comportementale avec apprentissage des préférences utilisateur
"""

import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import statistics
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class PreferenceType(Enum):
    """Types de préférences utilisateur"""
    PLANNING_STYLE = "planning_style"
    TIME_HORIZON = "time_horizon"
    RISK_TOLERANCE = "risk_tolerance"
    BUDGET_APPROACH = "budget_approach"
    TEAM_SIZE = "team_size"
    METHODOLOGY = "methodology"
    COMMUNICATION = "communication"
    TOOL_PREFERENCE = "tool_preference"

class ActionType(Enum):
    """Types d'actions utilisateur trackées"""
    PLAN_GENERATED = "plan_generated"
    RECOMMENDATION_ACCEPTED = "recommendation_accepted"
    RECOMMENDATION_REJECTED = "recommendation_rejected"
    PARAMETER_ADJUSTED = "parameter_adjusted"
    EXPORT_REQUESTED = "export_requested"
    WHAT_IF_SCENARIO = "what_if_scenario"
    DASHBOARD_VIEW = "dashboard_view"
    RISK_ACKNOWLEDGED = "risk_acknowledged"

@dataclass
class UserPreference:
    """Préférence utilisateur avec métriques"""
    preference_type: PreferenceType
    value: Any
    confidence: float
    frequency: int
    last_updated: datetime
    context: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'preference_type': self.preference_type.value,
            'value': self.value,
            'confidence': self.confidence,
            'frequency': self.frequency,
            'last_updated': self.last_updated.isoformat(),
            'context': self.context or {}
        }

@dataclass
class BehaviorPattern:
    """Pattern comportemental détecté"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    triggers: List[str]
    outcomes: List[str]
    detected_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'description': self.description,
            'frequency': self.frequency,
            'confidence': self.confidence,
            'triggers': self.triggers,
            'outcomes': self.outcomes,
            'detected_at': self.detected_at.isoformat()
        }

class BehavioralAnalyzer:
    """Analyseur comportemental avec apprentissage des préférences"""
    
    def __init__(self, db_path: str = "data/user_behavior.db"):
        self.db_path = db_path
        self.user_preferences = {}
        self.behavior_patterns = {}
        self.action_history = []
        self.session_id = self._generate_session_id()
        self.pattern_cache = {}
        
        self._init_database()
        self._load_preferences()
        
    def _generate_session_id(self) -> str:
        """Générer ID de session unique"""
        return f"session_{int(time.time())}"
        
    def _init_database(self):
        """Initialiser la base de données comportementale"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table des actions utilisateur
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        action_type TEXT NOT NULL,
                        context TEXT,
                        project_id TEXT,
                        parameters TEXT,
                        outcome TEXT
                    )
                """)
                
                # Table des préférences
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        preference_type TEXT NOT NULL,
                        value TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        frequency INTEGER NOT NULL,
                        last_updated DATETIME NOT NULL,
                        context TEXT
                    )
                """)
                
                # Table des patterns comportementaux
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS behavior_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT UNIQUE NOT NULL,
                        pattern_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        frequency INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        triggers TEXT NOT NULL,
                        outcomes TEXT NOT NULL,
                        detected_at DATETIME NOT NULL
                    )
                """)
                
                conn.commit()
                logger.info("✅ Base de données comportementale initialisée")
                
        except Exception as e:
            logger.error(f"Erreur initialisation DB comportementale: {e}")
            
    def track_user_action(self, 
                         action_type: ActionType, 
                         context: Dict[str, Any] = None,
                         project_id: str = None,
                         parameters: Dict[str, Any] = None,
                         outcome: str = None):
        """Tracker une action utilisateur"""
        
        action_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now(),
            'action_type': action_type,
            'context': context or {},
            'project_id': project_id,
            'parameters': parameters or {},
            'outcome': outcome
        }
        
        self.action_history.append(action_data)
        
        # Sauvegarder en base
        self._save_action(action_data)
        
        # Analyser pour détecter des patterns
        self._analyze_recent_actions()
        
        # Mettre à jour les préférences
        self._update_preferences_from_action(action_data)
        
        logger.debug(f"Action trackée: {action_type.value} - {context}")
        
    def _save_action(self, action_data: Dict[str, Any]):
        """Sauvegarder une action en base"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_actions 
                    (session_id, timestamp, action_type, context, project_id, parameters, outcome)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    action_data['session_id'],
                    action_data['timestamp'],
                    action_data['action_type'].value,
                    json.dumps(action_data['context']),
                    action_data['project_id'],
                    json.dumps(action_data['parameters']),
                    action_data['outcome']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde action: {e}")
            
    def _analyze_recent_actions(self, window_hours: int = 24):
        """Analyser les actions récentes pour détecter des patterns"""
        
        # Récupérer actions récentes
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_actions = [
            action for action in self.action_history 
            if action['timestamp'] >= cutoff_time
        ]
        
        if len(recent_actions) < 3:
            return
            
        # Détecter différents types de patterns
        self._detect_recommendation_patterns(recent_actions)
        self._detect_workflow_patterns(recent_actions)
        self._detect_preference_patterns(recent_actions)
        
    def _detect_recommendation_patterns(self, actions: List[Dict[str, Any]]):
        """Détecter des patterns dans l'usage des recommandations"""
        
        rec_actions = [
            a for a in actions 
            if a['action_type'] in [ActionType.RECOMMENDATION_ACCEPTED, ActionType.RECOMMENDATION_REJECTED]
        ]
        
        if len(rec_actions) < 3:
            return
            
        # Analyser taux d'acceptation par type
        acceptance_by_type = defaultdict(list)
        for action in rec_actions:
            rec_type = action['context'].get('recommendation_type', 'unknown')
            accepted = action['action_type'] == ActionType.RECOMMENDATION_ACCEPTED
            acceptance_by_type[rec_type].append(accepted)
            
        # Créer patterns
        for rec_type, acceptances in acceptance_by_type.items():
            if len(acceptances) >= 3:
                acceptance_rate = sum(acceptances) / len(acceptances)
                
                if acceptance_rate >= 0.8:
                    self._register_pattern(
                        pattern_type="high_recommendation_acceptance",
                        description=f"Accepte régulièrement les recommandations {rec_type}",
                        triggers=[f"recommendation_{rec_type}"],
                        outcomes=["high_acceptance"],
                        confidence=acceptance_rate
                    )
                elif acceptance_rate <= 0.3:
                    self._register_pattern(
                        pattern_type="low_recommendation_acceptance", 
                        description=f"Rejette souvent les recommandations {rec_type}",
                        triggers=[f"recommendation_{rec_type}"],
                        outcomes=["frequent_rejection"],
                        confidence=1.0 - acceptance_rate
                    )
                    
    def _detect_workflow_patterns(self, actions: List[Dict[str, Any]]):
        """Détecter des patterns dans le workflow"""
        
        # Séquences d'actions communes
        action_sequence = [a['action_type'].value for a in actions[-10:]]  # Dernières 10 actions
        
        # Pattern: génération -> what-if -> export
        if self._contains_sequence(action_sequence, ['plan_generated', 'what_if_scenario', 'export_requested']):
            self._register_pattern(
                pattern_type="analytical_workflow",
                description="Génère un plan, fait de l'analyse what-if, puis exporte",
                triggers=['plan_generated'],
                outcomes=['what_if_analysis', 'export'],
                confidence=0.8
            )
            
        # Pattern: ajustements fréquents de paramètres
        parameter_adjustments = [
            a for a in actions 
            if a['action_type'] == ActionType.PARAMETER_ADJUSTED
        ]
        
        if len(parameter_adjustments) >= 3:
            # Analyser quels paramètres sont ajustés
            adjusted_params = [
                list(a['parameters'].keys()) for a in parameter_adjustments
                if a['parameters']
            ]
            
            if adjusted_params:
                common_params = Counter([param for params in adjusted_params for param in params])
                most_adjusted = common_params.most_common(1)[0][0] if common_params else None
                
                if most_adjusted:
                    self._register_pattern(
                        pattern_type="parameter_tuning",
                        description=f"Ajuste fréquemment le paramètre {most_adjusted}",
                        triggers=['parameter_display'],
                        outcomes=['parameter_adjustment'],
                        confidence=0.7
                    )
                    
    def _detect_preference_patterns(self, actions: List[Dict[str, Any]]):
        """Détecter des patterns de préférences"""
        
        # Analyser les préférences de timing
        planning_actions = [
            a for a in actions 
            if a['action_type'] == ActionType.PLAN_GENERATED
        ]
        
        if len(planning_actions) >= 3:
            # Analyser les heures de génération
            hours = [a['timestamp'].hour for a in planning_actions]
            avg_hour = statistics.mean(hours)
            
            if avg_hour < 10:
                preference = "morning_planner"
                description = "Préfère planifier le matin"
            elif avg_hour > 18:
                preference = "evening_planner"
                description = "Préfère planifier le soir"
            else:
                preference = "daytime_planner"
                description = "Préfère planifier en journée"
                
            self._register_pattern(
                pattern_type="timing_preference",
                description=description,
                triggers=['planning_session'],
                outcomes=[preference],
                confidence=0.6
            )
            
    def _contains_sequence(self, sequence: List[str], target: List[str]) -> bool:
        """Vérifier si une séquence contient une sous-séquence"""
        if len(target) > len(sequence):
            return False
            
        for i in range(len(sequence) - len(target) + 1):
            if sequence[i:i+len(target)] == target:
                return True
        return False
        
    def _register_pattern(self, 
                         pattern_type: str,
                         description: str,
                         triggers: List[str],
                         outcomes: List[str],
                         confidence: float):
        """Enregistrer un pattern comportemental"""
        
        pattern_id = f"{pattern_type}_{int(time.time())}"
        
        pattern = BehaviorPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            frequency=1,
            confidence=confidence,
            triggers=triggers,
            outcomes=outcomes,
            detected_at=datetime.now()
        )
        
        # Vérifier si pattern similaire existe
        existing = self._find_similar_pattern(pattern_type)
        if existing:
            existing.frequency += 1
            existing.confidence = (existing.confidence + confidence) / 2
            existing.detected_at = datetime.now()
        else:
            self.behavior_patterns[pattern_id] = pattern
            self._save_pattern(pattern)
            
        logger.info(f"🧠 Pattern détecté: {description}")
        
    def _find_similar_pattern(self, pattern_type: str) -> Optional[BehaviorPattern]:
        """Chercher un pattern similaire existant"""
        for pattern in self.behavior_patterns.values():
            if pattern.pattern_type == pattern_type:
                return pattern
        return None
        
    def _save_pattern(self, pattern: BehaviorPattern):
        """Sauvegarder un pattern en base"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO behavior_patterns
                    (pattern_id, pattern_type, description, frequency, confidence, triggers, outcomes, detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    pattern.description,
                    pattern.frequency,
                    pattern.confidence,
                    json.dumps(pattern.triggers),
                    json.dumps(pattern.outcomes),
                    pattern.detected_at
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde pattern: {e}")
            
    def _update_preferences_from_action(self, action_data: Dict[str, Any]):
        """Mettre à jour les préférences basées sur une action"""
        
        action_type = action_data['action_type']
        context = action_data['context']
        parameters = action_data['parameters']
        
        # Préférences de méthodologie
        if action_type == ActionType.PLAN_GENERATED:
            methodology = context.get('methodology', 'agile')
            self._update_preference(
                PreferenceType.METHODOLOGY,
                methodology,
                confidence_boost=0.1
            )
            
        # Préférences de tolérance au risque
        elif action_type == ActionType.RECOMMENDATION_ACCEPTED:
            if 'risk' in context.get('recommendation_type', '').lower():
                self._update_preference(
                    PreferenceType.RISK_TOLERANCE,
                    'cautious',
                    confidence_boost=0.15
                )
                
        elif action_type == ActionType.RECOMMENDATION_REJECTED:
            if 'risk' in context.get('recommendation_type', '').lower():
                self._update_preference(
                    PreferenceType.RISK_TOLERANCE,
                    'risk_taking',
                    confidence_boost=0.1
                )
                
        # Préférences d'horizon temporel
        if parameters and 'duration' in parameters:
            duration = parameters['duration']
            if isinstance(duration, (int, float)):
                if duration <= 30:
                    time_horizon = 'short_term'
                elif duration <= 180:
                    time_horizon = 'medium_term'
                else:
                    time_horizon = 'long_term'
                    
                self._update_preference(
                    PreferenceType.TIME_HORIZON,
                    time_horizon,
                    confidence_boost=0.05
                )
                
    def _update_preference(self, 
                          preference_type: PreferenceType,
                          value: Any,
                          confidence_boost: float = 0.1):
        """Mettre à jour une préférence utilisateur"""
        
        pref_key = preference_type.value
        
        if pref_key in self.user_preferences:
            existing = self.user_preferences[pref_key]
            if existing.value == value:
                existing.frequency += 1
                existing.confidence = min(1.0, existing.confidence + confidence_boost)
                existing.last_updated = datetime.now()
            else:
                # Conflit de préférence - réduire la confiance de l'ancienne
                existing.confidence = max(0.1, existing.confidence - confidence_boost)
        else:
            self.user_preferences[pref_key] = UserPreference(
                preference_type=preference_type,
                value=value,
                confidence=0.5 + confidence_boost,
                frequency=1,
                last_updated=datetime.now()
            )
            
        self._save_preference(self.user_preferences[pref_key])
        
    def _save_preference(self, preference: UserPreference):
        """Sauvegarder une préférence en base"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_preferences
                    (preference_type, value, confidence, frequency, last_updated, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    preference.preference_type.value,
                    json.dumps(preference.value),
                    preference.confidence,
                    preference.frequency,
                    preference.last_updated,
                    json.dumps(preference.context or {})
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde préférence: {e}")
            
    def _load_preferences(self):
        """Charger les préférences depuis la base"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT preference_type, value, confidence, frequency, last_updated, context
                    FROM user_preferences
                """)
                
                for row in cursor.fetchall():
                    pref_type_str, value_json, confidence, frequency, last_updated_str, context_json = row
                    
                    try:
                        preference = UserPreference(
                            preference_type=PreferenceType(pref_type_str),
                            value=json.loads(value_json),
                            confidence=confidence,
                            frequency=frequency,
                            last_updated=datetime.fromisoformat(last_updated_str),
                            context=json.loads(context_json) if context_json else {}
                        )
                        self.user_preferences[pref_type_str] = preference
                    except Exception as e:
                        logger.warning(f"Erreur chargement préférence: {e}")
                        
                logger.info(f"✅ {len(self.user_preferences)} préférences chargées")
                
        except Exception as e:
            logger.error(f"Erreur chargement préférences: {e}")
            
    def get_user_preferences(self) -> Dict[str, UserPreference]:
        """Obtenir toutes les préférences utilisateur"""
        return self.user_preferences.copy()
        
    def get_preference(self, preference_type: PreferenceType) -> Optional[UserPreference]:
        """Obtenir une préférence spécifique"""
        return self.user_preferences.get(preference_type.value)
        
    def get_behavior_patterns(self) -> Dict[str, BehaviorPattern]:
        """Obtenir tous les patterns comportementaux"""
        return self.behavior_patterns.copy()
        
    def get_recommendation_adaptations(self) -> Dict[str, Any]:
        """Obtenir les adaptations recommandées basées sur le comportement"""
        
        adaptations = {
            'ui_preferences': {},
            'default_parameters': {},
            'recommendation_filters': {},
            'workflow_suggestions': []
        }
        
        # Adaptations UI basées sur les patterns
        for pattern in self.behavior_patterns.values():
            if pattern.pattern_type == 'analytical_workflow':
                adaptations['workflow_suggestions'].append({
                    'type': 'quick_what_if',
                    'description': "Accès rapide à l'analyse What-If",
                    'confidence': pattern.confidence
                })
                
            elif pattern.pattern_type == 'high_recommendation_acceptance':
                adaptations['ui_preferences']['show_more_recommendations'] = True
                
            elif pattern.pattern_type == 'parameter_tuning':
                adaptations['ui_preferences']['advanced_parameters_visible'] = True
                
        # Adaptations basées sur les préférences
        methodology_pref = self.get_preference(PreferenceType.METHODOLOGY)
        if methodology_pref and methodology_pref.confidence > 0.7:
            adaptations['default_parameters']['methodology'] = methodology_pref.value
            
        risk_pref = self.get_preference(PreferenceType.RISK_TOLERANCE)
        if risk_pref and risk_pref.confidence > 0.6:
            if risk_pref.value == 'cautious':
                adaptations['recommendation_filters']['show_risk_mitigation'] = True
            else:
                adaptations['recommendation_filters']['show_optimization_opportunities'] = True
                
        return adaptations
        
    def get_session_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de la session courante"""
        
        session_actions = [
            a for a in self.action_history 
            if a['session_id'] == self.session_id
        ]
        
        action_counts = Counter([a['action_type'].value for a in session_actions])
        
        return {
            'session_id': self.session_id,
            'total_actions': len(session_actions),
            'action_breakdown': dict(action_counts),
            'session_duration_minutes': (datetime.now() - datetime.fromisoformat(self.session_id.split('_')[1])).seconds // 60,
            'patterns_detected': len(self.behavior_patterns),
            'preferences_updated': len(self.user_preferences)
        }

# Instance globale
global_behavioral_analyzer: Optional[BehavioralAnalyzer] = None

def get_behavioral_analyzer(db_path: str = "data/user_behavior.db") -> BehavioralAnalyzer:
    """Obtenir l'instance globale de l'analyseur comportemental"""
    global global_behavioral_analyzer
    
    if global_behavioral_analyzer is None:
        global_behavioral_analyzer = BehavioralAnalyzer(db_path)
        
    return global_behavioral_analyzer

def track_user_action(action_type: ActionType, 
                     context: Dict[str, Any] = None,
                     project_id: str = None,
                     parameters: Dict[str, Any] = None,
                     outcome: str = None):
    """Fonction utilitaire pour tracker une action utilisateur"""
    analyzer = get_behavioral_analyzer()
    analyzer.track_user_action(action_type, context, project_id, parameters, outcome)

def get_user_preferences() -> Dict[str, UserPreference]:
    """Fonction utilitaire pour obtenir les préférences utilisateur"""
    analyzer = get_behavioral_analyzer()
    return analyzer.get_user_preferences()