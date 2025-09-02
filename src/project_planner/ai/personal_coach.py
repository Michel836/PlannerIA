"""
🎯 Coach Personnel IA - PlannerIA
Système d'accompagnement personnalisé transversal avec analyse comportementale
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict, deque
import functools
import pickle
import os


class CoachingArea(Enum):
    """Domaines de coaching"""
    PLANNING = "planning"
    BUDGET = "budget"
    RISK_MANAGEMENT = "risk_management"
    TEAM_LEADERSHIP = "team_leadership"
    COMMUNICATION = "communication"
    DECISION_MAKING = "decision_making"
    TIME_MANAGEMENT = "time_management"
    CRISIS_HANDLING = "crisis_handling"


class PersonalityTrait(Enum):
    """Traits de personnalité détectés"""
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    COLLABORATIVE = "collaborative"
    INDEPENDENT = "independent"
    PERFECTIONIST = "perfectionist"
    PRAGMATIC = "pragmatic"
    RISK_AVERSE = "risk_averse"
    RISK_TAKER = "risk_taker"


class CoachingStyle(Enum):
    """Styles de coaching adaptatifs"""
    DIRECTIVE = "directive"  # Instructions claires
    SUPPORTIVE = "supportive"  # Encouragement
    ANALYTICAL = "analytical"  # Données et logique
    CREATIVE = "creative"  # Solutions innovantes


@dataclass
class UserAction:
    """Action utilisateur observée"""
    module: str
    action_type: str
    context: Dict[str, Any]
    timestamp: datetime
    outcome: Optional[str] = None
    success_score: Optional[float] = None


@dataclass
class CoachingInsight:
    """Insight généré par le coach"""
    message: str
    confidence: float
    coaching_area: CoachingArea
    priority: str  # "high", "medium", "low"
    evidence: List[str]
    recommended_action: Optional[str] = None
    coaching_style: CoachingStyle = CoachingStyle.SUPPORTIVE


@dataclass
class UserProfile:
    """Profil comportemental utilisateur"""
    user_id: str
    personality_traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    strengths: List[CoachingArea] = field(default_factory=list)
    improvement_areas: List[CoachingArea] = field(default_factory=list)
    preferred_coaching_style: CoachingStyle = CoachingStyle.SUPPORTIVE
    success_patterns: Dict[str, List[str]] = field(default_factory=dict)
    failure_patterns: Dict[str, List[str]] = field(default_factory=dict)
    total_actions: int = 0
    coaching_sessions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class AIPersonalCoach:
    """Coach personnel IA transversal"""
    
    def __init__(self, data_dir: str = "data/coaching"):
        self.data_dir = data_dir
        self.user_profiles: Dict[str, UserProfile] = {}
        self.action_history: deque = deque(maxlen=1000)
        self.coaching_insights: List[CoachingInsight] = []
        
        # Modèles de patterns comportementaux
        self.behavioral_patterns = {
            'planning_patterns': {
                'over_estimator': ['timeline_buffer > 20%', 'rarely_late'],
                'under_estimator': ['timeline_buffer < 10%', 'frequently_late'],
                'detailed_planner': ['task_breakdown_depth > 3', 'frequent_updates'],
                'high_level_planner': ['task_breakdown_depth < 2', 'infrequent_updates']
            },
            'decision_patterns': {
                'data_driven': ['requests_analytics', 'delay_for_data'],
                'intuitive': ['quick_decisions', 'trust_gut_feeling'],
                'collaborative': ['seeks_team_input', 'consensus_builder'],
                'independent': ['decides_alone', 'confident_choices']
            },
            'risk_patterns': {
                'risk_averse': ['conservative_estimates', 'multiple_contingencies'],
                'balanced': ['moderate_estimates', 'some_contingencies'],
                'risk_taker': ['aggressive_estimates', 'minimal_contingencies']
            }
        }
        
        # Templates de conseils personnalisés
        self.coaching_templates = {
            'planning_advice': {
                'over_estimator': "Votre tendance à surestimer est un atout pour la fiabilité, mais pourriez-vous optimiser 15% du buffer pour accelerer ?",
                'under_estimator': "Pattern détecté: vos estimations sont 23% trop optimistes. Je recommande un buffer de sécurité de 25%.",
                'detailed_planner': "Votre approche détaillée excellente ! Attention à ne pas sur-planifier les tâches simples.",
                'high_level_planner': "Votre vision macro est forte. Considérez plus de détails pour les phases critiques."
            },
            'decision_advice': {
                'data_driven': "Votre approche analytique est solide. Parfois, faites confiance à votre expérience pour accélérer.",
                'intuitive': "Votre intuition est souvent juste ! Validez avec quelques métriques clés pour renforcer.",
                'collaborative': "Votre leadership collaboratif engage l'équipe. Dans l'urgence, assumez vos décisions solo.",
                'independent': "Votre autonomie décisionnelle est efficace. Impliquez l'équipe sur les décisions qui l'affectent."
            }
        }
        
        # Chargement des données existantes
        self._load_user_data()
        
        # Compteurs de performance
        self.module_observations = defaultdict(int)
        self.coaching_effectiveness = defaultdict(list)
    
    def _load_user_data(self):
        """Charge les données utilisateur existantes"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            return
            
        profile_file = os.path.join(self.data_dir, "user_profiles.json")
        if os.path.exists(profile_file):
            try:
                with open(profile_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, profile_data in data.items():
                        profile = UserProfile(user_id=user_id)
                        profile.__dict__.update(profile_data)
                        # Conversion des enums
                        if 'personality_traits' in profile_data:
                            profile.personality_traits = {
                                PersonalityTrait(k): v for k, v in profile_data['personality_traits'].items()
                            }
                        if 'strengths' in profile_data:
                            profile.strengths = [CoachingArea(area) for area in profile_data['strengths']]
                        if 'improvement_areas' in profile_data:
                            profile.improvement_areas = [CoachingArea(area) for area in profile_data['improvement_areas']]
                        if 'preferred_coaching_style' in profile_data:
                            profile.preferred_coaching_style = CoachingStyle(profile_data['preferred_coaching_style'])
                        
                        self.user_profiles[user_id] = profile
            except Exception as e:
                print(f"Erreur chargement profils: {e}")
    
    def _save_user_data(self):
        """Sauvegarde les données utilisateur"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        profile_file = os.path.join(self.data_dir, "user_profiles.json")
        try:
            serializable_data = {}
            for user_id, profile in self.user_profiles.items():
                profile_dict = profile.__dict__.copy()
                # Conversion des enums en strings
                if 'personality_traits' in profile_dict:
                    profile_dict['personality_traits'] = {
                        trait.value: score for trait, score in profile_dict['personality_traits'].items()
                    }
                if 'strengths' in profile_dict:
                    profile_dict['strengths'] = [area.value for area in profile_dict['strengths']]
                if 'improvement_areas' in profile_dict:
                    profile_dict['improvement_areas'] = [area.value for area in profile_dict['improvement_areas']]
                if 'preferred_coaching_style' in profile_dict:
                    profile_dict['preferred_coaching_style'] = profile_dict['preferred_coaching_style'].value
                if 'last_updated' in profile_dict:
                    profile_dict['last_updated'] = profile_dict['last_updated'].isoformat()
                    
                serializable_data[user_id] = profile_dict
                
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde profils: {e}")
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """Récupère ou crée le profil utilisateur"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        return self.user_profiles[user_id]
    
    def observe_action(self, user_id: str, module: str, action_type: str, 
                      context: Dict[str, Any], outcome: Optional[str] = None):
        """Observe une action utilisateur"""
        
        action = UserAction(
            module=module,
            action_type=action_type,
            context=context,
            timestamp=datetime.now(),
            outcome=outcome
        )
        
        self.action_history.append(action)
        self.module_observations[module] += 1
        
        # Mise à jour du profil utilisateur
        profile = self.get_user_profile(user_id)
        profile.total_actions += 1
        profile.last_updated = datetime.now()
        
        # Analyse comportementale en temps réel
        self._analyze_behavioral_patterns(user_id, action)
        
        # Génération d'insights si nécessaire
        if profile.total_actions % 10 == 0:  # Tous les 10 actions
            self._generate_coaching_insights(user_id)
        
        # Sauvegarde périodique
        if profile.total_actions % 50 == 0:
            self._save_user_data()
    
    def _analyze_behavioral_patterns(self, user_id: str, action: UserAction):
        """Analyse les patterns comportementaux"""
        profile = self.get_user_profile(user_id)
        
        # Analyse du style de planification
        if action.module == "planning":
            if action.action_type == "create_timeline":
                buffer_ratio = action.context.get('buffer_ratio', 0)
                if buffer_ratio > 0.20:
                    self._update_personality_trait(profile, PersonalityTrait.PERFECTIONIST, 0.1)
                elif buffer_ratio < 0.10:
                    self._update_personality_trait(profile, PersonalityTrait.RISK_TAKER, 0.1)
        
        # Analyse du style décisionnel
        elif action.module in ["budget", "risk", "crisis"]:
            if action.action_type == "make_decision":
                analysis_time = action.context.get('analysis_time_minutes', 0)
                if analysis_time > 15:
                    self._update_personality_trait(profile, PersonalityTrait.ANALYTICAL, 0.1)
                elif analysis_time < 2:
                    self._update_personality_trait(profile, PersonalityTrait.INTUITIVE, 0.1)
        
        # Détection de patterns de succès/échec
        if action.outcome:
            pattern_key = f"{action.module}_{action.action_type}"
            if action.outcome == "success":
                if pattern_key not in profile.success_patterns:
                    profile.success_patterns[pattern_key] = []
                profile.success_patterns[pattern_key].append(str(action.context))
            elif action.outcome == "failure":
                if pattern_key not in profile.failure_patterns:
                    profile.failure_patterns[pattern_key] = []
                profile.failure_patterns[pattern_key].append(str(action.context))
    
    def _update_personality_trait(self, profile: UserProfile, trait: PersonalityTrait, increment: float):
        """Met à jour un trait de personnalité"""
        current = profile.personality_traits.get(trait, 0.5)
        new_value = min(1.0, max(0.0, current + increment))
        profile.personality_traits[trait] = new_value
    
    def _generate_coaching_insights(self, user_id: str):
        """Génère des insights de coaching personnalisés"""
        profile = self.get_user_profile(user_id)
        insights = []
        
        # Analyse des traits dominants
        if profile.personality_traits:
            dominant_trait = max(profile.personality_traits.items(), key=lambda x: x[1])
            
            if dominant_trait[1] > 0.7:  # Trait très marqué
                insight = self._generate_trait_insight(dominant_trait[0], profile)
                if insight:
                    insights.append(insight)
        
        # Analyse des patterns de succès
        if profile.success_patterns:
            most_successful = max(profile.success_patterns.items(), key=lambda x: len(x[1]))
            if len(most_successful[1]) >= 3:
                insight = self._generate_success_pattern_insight(most_successful, profile)
                if insight:
                    insights.append(insight)
        
        # Analyse des patterns d'échec
        if profile.failure_patterns:
            most_problematic = max(profile.failure_patterns.items(), key=lambda x: len(x[1]))
            if len(most_problematic[1]) >= 2:
                insight = self._generate_improvement_insight(most_problematic, profile)
                if insight:
                    insights.append(insight)
        
        # Ajout des insights générés
        self.coaching_insights.extend(insights)
        
        # Limitation du nombre d'insights actifs
        if len(self.coaching_insights) > 20:
            self.coaching_insights = self.coaching_insights[-15:]
    
    def _generate_trait_insight(self, trait: PersonalityTrait, profile: UserProfile) -> Optional[CoachingInsight]:
        """Génère un insight basé sur un trait de personnalité"""
        
        trait_insights = {
            PersonalityTrait.ANALYTICAL: CoachingInsight(
                message="Votre approche analytique est un atout majeur ! Parfois, faites confiance à votre expérience pour des décisions plus rapides.",
                confidence=0.8,
                coaching_area=CoachingArea.DECISION_MAKING,
                priority="medium",
                evidence=["Temps d'analyse élevé", "Recherche de données avant décision"],
                recommended_action="Fixez-vous un time-box de 10min pour les décisions simples",
                coaching_style=CoachingStyle.ANALYTICAL
            ),
            PersonalityTrait.INTUITIVE: CoachingInsight(
                message="Votre intuition vous guide bien ! Renforcez vos décisions avec quelques métriques clés pour convaincre l'équipe.",
                confidence=0.8,
                coaching_area=CoachingArea.DECISION_MAKING,
                priority="medium",
                evidence=["Décisions rapides", "Se fie à l'instinct"],
                recommended_action="Préparez 2-3 données pour justifier vos choix intuitifs",
                coaching_style=CoachingStyle.SUPPORTIVE
            ),
            PersonalityTrait.PERFECTIONIST: CoachingInsight(
                message="Votre attention au détail garantit la qualité ! Identifiez les 20% critiques qui nécessitent votre perfectionnisme.",
                confidence=0.85,
                coaching_area=CoachingArea.TIME_MANAGEMENT,
                priority="high",
                evidence=["Planning détaillé", "Buffers importants", "Révisions multiples"],
                recommended_action="Utilisez la règle 80/20: perfectionnisme sur 20% critique, 'good enough' sur le reste",
                coaching_style=CoachingStyle.DIRECTIVE
            )
        }
        
        return trait_insights.get(trait)
    
    def _generate_success_pattern_insight(self, success_pattern: Tuple, profile: UserProfile) -> Optional[CoachingInsight]:
        """Génère un insight basé sur les patterns de succès"""
        pattern_key, occurrences = success_pattern
        module, action = pattern_key.split('_', 1)
        
        return CoachingInsight(
            message=f"Pattern de succès identifié ! Quand vous {action} dans {module}, vous réussissez dans {len(occurrences)} cas sur vos dernières actions. Reproduisez cette approche !",
            confidence=min(0.9, len(occurrences) * 0.15),
            coaching_area=CoachingArea.DECISION_MAKING,
            priority="high",
            evidence=[f"{len(occurrences)} succès récents", f"Module: {module}"],
            recommended_action=f"Appliquez votre méthode éprouvée de {action} aux situations similaires",
            coaching_style=CoachingStyle.SUPPORTIVE
        )
    
    def _generate_improvement_insight(self, failure_pattern: Tuple, profile: UserProfile) -> Optional[CoachingInsight]:
        """Génère un insight basé sur les patterns d'échec"""
        pattern_key, occurrences = failure_pattern
        module, action = pattern_key.split('_', 1)
        
        return CoachingInsight(
            message=f"Attention ! Pattern récurrent détecté: {action} dans {module} pose des difficultés. Analysons ensemble une approche alternative.",
            confidence=min(0.8, len(occurrences) * 0.2),
            coaching_area=CoachingArea.DECISION_MAKING,
            priority="high",
            evidence=[f"{len(occurrences)} difficultés récentes", f"Module: {module}"],
            recommended_action=f"Préparez une checklist pour {action} ou demandez un avis externe",
            coaching_style=CoachingStyle.DIRECTIVE
        )
    
    def get_contextual_advice(self, user_id: str, current_module: str, 
                            current_action: str = None) -> List[CoachingInsight]:
        """Fournit des conseils contextuels pour le module actuel"""
        profile = self.get_user_profile(user_id)
        relevant_insights = []
        
        # Insights spécifiques au module actuel
        module_insights = [
            insight for insight in self.coaching_insights
            if current_module.lower() in insight.message.lower() or
               current_module in [area.value for area in [insight.coaching_area]]
        ]
        
        # Tri par priorité et confiance
        module_insights.sort(key=lambda x: (x.priority == "high", x.confidence), reverse=True)
        relevant_insights.extend(module_insights[:2])  # Top 2
        
        # Insight générique basé sur le profil
        if not relevant_insights and profile.personality_traits:
            dominant_trait = max(profile.personality_traits.items(), key=lambda x: x[1])
            generic_insight = self._generate_trait_insight(dominant_trait[0], profile)
            if generic_insight:
                relevant_insights.append(generic_insight)
        
        return relevant_insights
    
    def get_coaching_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Données pour le dashboard de coaching"""
        profile = self.get_user_profile(user_id)
        
        # Calcul des métriques
        recent_actions = [a for a in self.action_history if a.timestamp > datetime.now() - timedelta(days=7)]
        success_rate = 0.7 + np.random.random() * 0.2  # Simulation
        
        return {
            'profile': profile,
            'total_actions': profile.total_actions,
            'coaching_sessions': profile.coaching_sessions,
            'recent_activity': len(recent_actions),
            'success_rate': success_rate,
            'active_insights': len(self.coaching_insights),
            'personality_analysis': dict(profile.personality_traits),
            'strengths': [area.value for area in profile.strengths],
            'improvement_areas': [area.value for area in profile.improvement_areas],
            'coaching_effectiveness': np.mean(self.coaching_effectiveness.get(user_id, [0.7])),
            'module_usage': dict(self.module_observations)
        }
    
    def mark_insight_as_applied(self, insight_id: str, user_id: str, effectiveness_score: float):
        """Marque un insight comme appliqué et évalue son efficacité"""
        if user_id not in self.coaching_effectiveness:
            self.coaching_effectiveness[user_id] = []
        
        self.coaching_effectiveness[user_id].append(effectiveness_score)
        
        # Mise à jour du profil
        profile = self.get_user_profile(user_id)
        profile.coaching_sessions += 1
        
        # Nettoyage des insights appliqués
        self.coaching_insights = [i for i in self.coaching_insights if str(id(i)) != insight_id]


def coach_observe(action_type: str, module: str = None):
    """Décorateur pour observer les actions utilisateur"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Tentative de récupération du coach global
            try:
                # Récupération de l'user_id depuis session_state ou défaut
                user_id = getattr(st.session_state, 'user_id', f"user_{datetime.now().timestamp()}")
                
                # Contexte de l'action
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs': list(kwargs.keys()),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Module automatique si non spécifié
                detected_module = module or func.__module__.split('.')[-1]
                
                # Exécution de la fonction
                start_time = datetime.now()
                try:
                    result = func(*args, **kwargs)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    context['execution_time'] = execution_time
                    outcome = "success"
                except Exception as e:
                    context['error'] = str(e)
                    outcome = "failure"
                    raise
                finally:
                    # Observation par le coach (si disponible)
                    try:
                        if hasattr(st.session_state, 'personal_coach'):
                            coach = st.session_state.personal_coach
                            coach.observe_action(user_id, detected_module, action_type, context, outcome)
                    except Exception:
                        pass  # Coach non disponible, continuer silencieusement
                
                return result
            except Exception:
                # En cas d'erreur, exécuter la fonction normalement
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Instance globale du coach
ai_personal_coach = AIPersonalCoach()

# Initialisation pour Streamlit
def initialize_personal_coach():
    """Initialise le coach personnel dans la session Streamlit"""
    if 'personal_coach' not in st.session_state:
        st.session_state.personal_coach = ai_personal_coach
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{datetime.now().timestamp()}"
    
    return st.session_state.personal_coach


# Fonction d'aide pour les autres modules
def get_personal_coach() -> AIPersonalCoach:
    """Récupère l'instance du coach personnel"""
    return getattr(st.session_state, 'personal_coach', ai_personal_coach)