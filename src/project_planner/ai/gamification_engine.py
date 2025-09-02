"""
🎮 Système de Gamification Intelligente - PlannerIA
Défis adaptatifs, achievements et coaching virtuel avec IA
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import random
import math
from collections import defaultdict


class ChallengeType(Enum):
    ESTIMATION = "estimation"
    OPTIMIZATION = "optimization"
    RISK_MANAGEMENT = "risk_management"
    TEAM_BUILDING = "team_building"
    BUDGET_CONTROL = "budget_control"
    QUALITY_FOCUS = "quality_focus"
    INNOVATION = "innovation"


class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"


class AchievementCategory(Enum):
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    INNOVATION = "innovation"
    LEADERSHIP = "leadership"
    LEARNING = "learning"
    COLLABORATION = "collaboration"


@dataclass
class Challenge:
    """Défi gamifié généré par IA"""
    challenge_id: str
    type: ChallengeType
    difficulty: DifficultyLevel
    title: str
    description: str
    objectives: List[str]
    success_criteria: Dict[str, Any]
    rewards: Dict[str, int]
    time_limit: Optional[timedelta]
    hints: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    ai_generated: bool = True


@dataclass
class Achievement:
    """Achievement déblocable"""
    achievement_id: str
    category: AchievementCategory
    name: str
    description: str
    icon: str
    rarity: str  # common, rare, epic, legendary
    points: int
    unlock_criteria: Dict[str, Any]
    unlocked_at: Optional[datetime] = None
    progress: float = 0.0


@dataclass
class PlayerProfile:
    """Profil joueur avec progression"""
    player_id: str
    name: str
    level: int
    total_xp: int
    current_streak: int
    longest_streak: int
    challenges_completed: int
    achievements_unlocked: List[str]
    skill_ratings: Dict[str, float]
    preferred_challenge_types: List[ChallengeType]
    learning_style: str
    created_at: datetime
    badges: List[str] = field(default_factory=list)


@dataclass
class GameSession:
    """Session de jeu active"""
    session_id: str
    player_id: str
    start_time: datetime
    current_challenge: Optional[Challenge]
    score: int
    actions_taken: List[Dict[str, Any]]
    ai_coach_messages: List[str]
    session_type: str = "planning"


class IntelligentGamificationEngine:
    """Moteur de gamification intelligente avec IA adaptive"""
    
    def __init__(self):
        self.players = {}
        self.challenges_pool = self._initialize_challenges_pool()
        self.achievements_catalog = self._initialize_achievements()
        self.active_sessions = {}
        self.ai_coach = AICoach()
        self.difficulty_adapter = DifficultyAdapter()
        self.leaderboards = defaultdict(list)
        
    def _initialize_challenges_pool(self) -> Dict[str, List[Challenge]]:
        """Initialise le pool de défis par type et difficulté"""
        
        challenges = {
            ChallengeType.ESTIMATION.value: [
                Challenge(
                    challenge_id="est_basic_webapp",
                    type=ChallengeType.ESTIMATION,
                    difficulty=DifficultyLevel.BEGINNER,
                    title="🎯 Estimation Master: Web App Simple",
                    description="Estimez précisément la durée de développement d'une application web basique avec React/Node.js",
                    objectives=[
                        "Analyser les spécifications techniques",
                        "Estimer la durée de chaque composant",
                        "Atteindre une précision de ±15%"
                    ],
                    success_criteria={
                        "accuracy_threshold": 0.15,
                        "time_bonus_threshold": 300,  # 5 minutes
                        "methodology_points": True
                    },
                    rewards={"xp": 100, "coins": 50, "estimation_skill": 10},
                    time_limit=timedelta(minutes=10),
                    hints=[
                        "💡 Pensez aux composants réutilisables",
                        "🔧 N'oubliez pas les tests et la documentation",
                        "⚡ Les APIs externes ajoutent de la complexité"
                    ],
                    created_at=datetime.now(),
                    expires_at=None
                ),
                
                Challenge(
                    challenge_id="est_ecommerce_advanced", 
                    type=ChallengeType.ESTIMATION,
                    difficulty=DifficultyLevel.ADVANCED,
                    title="🛒 E-commerce Challenge: Marketplace Complex",
                    description="Projet complexe: marketplace multi-vendeurs avec paiements, stocks, et analytics",
                    objectives=[
                        "Identifier tous les modules nécessaires",
                        "Estimer les intégrations tierces",
                        "Prévoir les risques techniques",
                        "Précision ±10% requise"
                    ],
                    success_criteria={
                        "accuracy_threshold": 0.10,
                        "risk_identification": 8,
                        "module_completeness": 0.9
                    },
                    rewards={"xp": 500, "coins": 300, "estimation_skill": 50},
                    time_limit=timedelta(minutes=30),
                    hints=[
                        "🏪 Marketplace = complexité x3 vs e-commerce classique",
                        "💳 Intégrations paiement multi-devises",
                        "📊 Analytics temps réel = défi technique"
                    ],
                    created_at=datetime.now(),
                    expires_at=None
                )
            ],
            
            ChallengeType.OPTIMIZATION.value: [
                Challenge(
                    challenge_id="opt_budget_rescue",
                    type=ChallengeType.OPTIMIZATION,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    title="💰 Budget Rescue Mission",
                    description="Un projet dépasse son budget de 40%. Trouvez 3 stratégies pour le sauver sans compromettre la qualité.",
                    objectives=[
                        "Identifier les postes de coûts élevés",
                        "Proposer des alternatives moins chères",
                        "Maintenir 90% des fonctionnalités critiques"
                    ],
                    success_criteria={
                        "cost_reduction": 0.4,
                        "feature_retention": 0.9,
                        "feasibility_score": 0.8
                    },
                    rewards={"xp": 300, "coins": 200, "optimization_skill": 30},
                    time_limit=timedelta(minutes=15),
                    hints=[
                        "🔧 Technologies moins chères mais équivalentes?",
                        "⏱️ Phases ou fonctionnalités reportables?",
                        "👥 Optimisation de l'équipe possible?"
                    ],
                    created_at=datetime.now(),
                    expires_at=None
                )
            ],
            
            ChallengeType.RISK_MANAGEMENT.value: [
                Challenge(
                    challenge_id="risk_fintech_security",
                    type=ChallengeType.RISK_MANAGEMENT,
                    difficulty=DifficultyLevel.EXPERT,
                    title="🛡️ Fintech Security Master",
                    description="Application bancaire mobile: identifiez et mitigez les 10 risques critiques de sécurité et conformité.",
                    objectives=[
                        "Identifier risques réglementaires (RGPD, PCI-DSS)",
                        "Évaluer risques techniques et humains",
                        "Proposer stratégies de mitigation",
                        "Créer plan de contingence"
                    ],
                    success_criteria={
                        "risk_coverage": 0.95,
                        "mitigation_quality": 0.9,
                        "compliance_score": 1.0
                    },
                    rewards={"xp": 800, "coins": 500, "risk_skill": 80},
                    time_limit=timedelta(minutes=45),
                    hints=[
                        "🏦 RGPD + PCI-DSS + directives bancaires",
                        "🔐 Chiffrement, authentification, audit trails",
                        "👥 Formation équipe = risque humain réduit"
                    ],
                    created_at=datetime.now(),
                    expires_at=None
                )
            ],
            
            ChallengeType.TEAM_BUILDING.value: [
                Challenge(
                    challenge_id="team_remote_coordination",
                    type=ChallengeType.TEAM_BUILDING,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    title="🌍 Remote Team Coordination",
                    description="Équipe de 12 développeurs répartis sur 4 fuseaux horaires. Optimisez la collaboration et productivité.",
                    objectives=[
                        "Organiser les équipes par compétences et fuseaux",
                        "Définir processus de communication",
                        "Planifier les sprints et cérémonies",
                        "Minimiser les dépendances inter-équipes"
                    ],
                    success_criteria={
                        "communication_efficiency": 0.85,
                        "timezone_optimization": 0.9,
                        "dependency_reduction": 0.7
                    },
                    rewards={"xp": 400, "coins": 250, "leadership_skill": 40},
                    time_limit=timedelta(minutes=20),
                    hints=[
                        "⏰ 2-3h de chevauchement minimum entre équipes",
                        "📋 Documentation asynchrone cruciale",
                        "🎯 Équipes autonomes = moins de dépendances"
                    ],
                    created_at=datetime.now(),
                    expires_at=None
                )
            ]
        }
        
        # Générer plus de défis pour chaque catégorie
        for challenge_type in ChallengeType:
            if challenge_type.value not in challenges:
                challenges[challenge_type.value] = []
            
            # Ajouter des défis générés par IA si pas assez
            while len(challenges[challenge_type.value]) < 3:
                ai_challenge = self._generate_ai_challenge(challenge_type)
                challenges[challenge_type.value].append(ai_challenge)
        
        return challenges
    
    def _generate_ai_challenge(self, challenge_type: ChallengeType) -> Challenge:
        """Génère un défi avec IA"""
        
        # Templates par type de défi
        templates = {
            ChallengeType.QUALITY_FOCUS: {
                "titles": ["🏆 Code Quality Champion", "📊 Quality Metrics Master", "🔍 Bug Hunter Pro"],
                "descriptions": [
                    "Améliorer la qualité du code en implémentant des métriques avancées",
                    "Mettre en place un système de revue de code optimal",
                    "Atteindre 95% de couverture de tests automatisés"
                ]
            },
            ChallengeType.INNOVATION: {
                "titles": ["💡 Innovation Lab", "🚀 Tech Explorer", "🎨 Creative Solutions"],
                "descriptions": [
                    "Proposer une solution innovante pour un problème complexe",
                    "Intégrer une technologie émergente dans le projet",
                    "Créer un prototype disruptif en temps limité"
                ]
            }
        }
        
        template = templates.get(challenge_type, {
            "titles": [f"Défi {challenge_type.value.title()}"],
            "descriptions": [f"Défi personnalisé pour {challenge_type.value}"]
        })
        
        difficulty = random.choice(list(DifficultyLevel))
        
        return Challenge(
            challenge_id=f"ai_{challenge_type.value}_{random.randint(1000, 9999)}",
            type=challenge_type,
            difficulty=difficulty,
            title=random.choice(template["titles"]),
            description=random.choice(template["descriptions"]),
            objectives=[
                f"Objectif principal du défi {challenge_type.value}",
                f"Critère de succès technique",
                f"Validation par les pairs"
            ],
            success_criteria={"ai_generated": True, "threshold": 0.8},
            rewards={
                "xp": random.randint(100, 600),
                "coins": random.randint(50, 400),
                f"{challenge_type.value}_skill": random.randint(10, 60)
            },
            time_limit=timedelta(minutes=random.randint(10, 45)),
            hints=[
                "💡 Conseil IA généré",
                "🎯 Focus sur la qualité",
                "⚡ Optimisation recommandée"
            ],
            created_at=datetime.now(),
            expires_at=None,
            ai_generated=True
        )
    
    def _initialize_achievements(self) -> Dict[str, Achievement]:
        """Initialise le catalogue d'achievements"""
        
        achievements = {
            # Accuracy Achievements
            "estimation_sniper": Achievement(
                achievement_id="estimation_sniper",
                category=AchievementCategory.ACCURACY,
                name="🎯 Estimation Sniper",
                description="Atteindre une précision de ±5% sur 5 estimations consécutives",
                icon="🎯",
                rarity="epic",
                points=500,
                unlock_criteria={"consecutive_accurate_estimations": 5, "accuracy_threshold": 0.05}
            ),
            
            "budget_wizard": Achievement(
                achievement_id="budget_wizard",
                category=AchievementCategory.ACCURACY,
                name="💰 Budget Wizard",
                description="Prédire le budget final avec moins de 10% d'erreur sur 3 projets",
                icon="💰",
                rarity="rare",
                points=300,
                unlock_criteria={"accurate_budget_predictions": 3, "budget_accuracy": 0.10}
            ),
            
            # Efficiency Achievements
            "speed_demon": Achievement(
                achievement_id="speed_demon",
                category=AchievementCategory.EFFICIENCY,
                name="⚡ Speed Demon",
                description="Compléter 10 défis en moins de temps que la moyenne",
                icon="⚡",
                rarity="rare", 
                points=400,
                unlock_criteria={"fast_completions": 10}
            ),
            
            "optimization_guru": Achievement(
                achievement_id="optimization_guru",
                category=AchievementCategory.EFFICIENCY,
                name="🚀 Optimization Guru",
                description="Réduire les coûts de projet de 30% tout en maintenant la qualité",
                icon="🚀",
                rarity="legendary",
                points=1000,
                unlock_criteria={"cost_reduction_achievements": 5, "min_reduction": 0.30}
            ),
            
            # Innovation Achievements  
            "innovator": Achievement(
                achievement_id="innovator",
                category=AchievementCategory.INNOVATION,
                name="💡 Innovation Master",
                description="Proposer 5 solutions créatives acceptées par l'IA",
                icon="💡",
                rarity="epic",
                points=600,
                unlock_criteria={"creative_solutions_accepted": 5}
            ),
            
            # Leadership Achievements
            "team_builder": Achievement(
                achievement_id="team_builder",
                category=AchievementCategory.LEADERSHIP,
                name="👥 Team Builder",
                description="Créer des équipes avec 95%+ de compatibilité sur 10 projets",
                icon="👥",
                rarity="epic",
                points=700,
                unlock_criteria={"high_compatibility_teams": 10, "compatibility_threshold": 0.95}
            ),
            
            # Learning Achievements
            "knowledge_seeker": Achievement(
                achievement_id="knowledge_seeker",
                category=AchievementCategory.LEARNING,
                name="📚 Knowledge Seeker",
                description="Compléter des défis dans 5 domaines différents",
                icon="📚",
                rarity="rare",
                points=350,
                unlock_criteria={"different_domains_completed": 5}
            ),
            
            "streak_master": Achievement(
                achievement_id="streak_master",
                category=AchievementCategory.LEARNING,
                name="🔥 Streak Master",
                description="Maintenir une série de 30 jours d'activité",
                icon="🔥",
                rarity="legendary",
                points=1500,
                unlock_criteria={"daily_streak": 30}
            ),
            
            # Collaboration Achievements
            "mentor": Achievement(
                achievement_id="mentor",
                category=AchievementCategory.COLLABORATION,
                name="🎓 Mentor",
                description="Aider 20 autres joueurs à résoudre des défis",
                icon="🎓",
                rarity="epic",
                points=800,
                unlock_criteria={"players_helped": 20}
            )
        }
        
        return achievements
    
    def register_player(self, player_id: str, name: str) -> PlayerProfile:
        """Enregistre un nouveau joueur"""
        
        profile = PlayerProfile(
            player_id=player_id,
            name=name,
            level=1,
            total_xp=0,
            current_streak=0,
            longest_streak=0,
            challenges_completed=0,
            achievements_unlocked=[],
            skill_ratings={
                "estimation": 100,
                "optimization": 100,
                "risk_management": 100,
                "team_building": 100,
                "innovation": 100,
                "leadership": 100
            },
            preferred_challenge_types=[],
            learning_style="balanced",
            created_at=datetime.now()
        )
        
        self.players[player_id] = profile
        return profile
    
    def get_personalized_challenge(self, player_id: str) -> Challenge:
        """Génère un défi personnalisé basé sur le profil joueur"""
        
        if player_id not in self.players:
            return self._get_random_beginner_challenge()
        
        player = self.players[player_id]
        
        # Analyse du niveau et compétences
        suitable_difficulty = self._calculate_suitable_difficulty(player)
        preferred_types = self._analyze_preferred_types(player)
        
        # Sélection IA du meilleur défi
        best_challenge = self._select_optimal_challenge(player, suitable_difficulty, preferred_types)
        
        # Personnalisation du défi
        personalized_challenge = self._personalize_challenge(best_challenge, player)
        
        return personalized_challenge
    
    def _calculate_suitable_difficulty(self, player: PlayerProfile) -> DifficultyLevel:
        """Calcule le niveau de difficulté approprié"""
        
        # Basé sur le niveau, XP et skill ratings
        avg_skill = np.mean(list(player.skill_ratings.values()))
        
        if avg_skill < 150 or player.level < 3:
            return DifficultyLevel.BEGINNER
        elif avg_skill < 300 or player.level < 8:
            return DifficultyLevel.INTERMEDIATE  
        elif avg_skill < 500 or player.level < 15:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT
    
    def _analyze_preferred_types(self, player: PlayerProfile) -> List[ChallengeType]:
        """Analyse les types de défis préférés du joueur"""
        
        if player.preferred_challenge_types:
            return player.preferred_challenge_types
        
        # Analyse basée sur les compétences les plus développées
        sorted_skills = sorted(player.skill_ratings.items(), key=lambda x: x[1], reverse=True)
        
        skill_to_challenge = {
            "estimation": ChallengeType.ESTIMATION,
            "optimization": ChallengeType.OPTIMIZATION,
            "risk_management": ChallengeType.RISK_MANAGEMENT,
            "team_building": ChallengeType.TEAM_BUILDING,
            "innovation": ChallengeType.INNOVATION,
            "leadership": ChallengeType.TEAM_BUILDING
        }
        
        preferred = []
        for skill, _ in sorted_skills[:3]:
            if skill in skill_to_challenge:
                preferred.append(skill_to_challenge[skill])
        
        return preferred or [ChallengeType.ESTIMATION]
    
    def _select_optimal_challenge(self, player: PlayerProfile, 
                                difficulty: DifficultyLevel,
                                preferred_types: List[ChallengeType]) -> Challenge:
        """Sélectionne le défi optimal avec algorithme IA"""
        
        candidates = []
        
        # Collecte des candidats
        for challenge_type in preferred_types:
            type_challenges = self.challenges_pool.get(challenge_type.value, [])
            
            for challenge in type_challenges:
                if challenge.difficulty == difficulty:
                    candidates.append(challenge)
        
        # Si pas assez de candidats, élargir la recherche
        if len(candidates) < 3:
            for challenge_type_key, challenges in self.challenges_pool.items():
                for challenge in challenges:
                    if challenge.difficulty == difficulty and challenge not in candidates:
                        candidates.append(challenge)
        
        # Sélection par algorithme de scoring
        if not candidates:
            return self._get_random_beginner_challenge()
        
        best_challenge = max(candidates, key=lambda c: self._calculate_challenge_score(c, player))
        return best_challenge
    
    def _calculate_challenge_score(self, challenge: Challenge, player: PlayerProfile) -> float:
        """Calcule un score de pertinence pour un défi"""
        
        score = 0.0
        
        # Bonus pour type préféré
        if challenge.type in player.preferred_challenge_types:
            score += 0.3
        
        # Bonus pour compétence à développer
        skill_key = challenge.type.value
        if skill_key in player.skill_ratings:
            # Favoriser les compétences moyennement développées (zone d'apprentissage optimal)
            skill_level = player.skill_ratings[skill_key]
            if 200 <= skill_level <= 400:
                score += 0.4
            elif skill_level < 200:
                score += 0.2  # Peut être trop difficile
        
        # Bonus pour nouveauté
        if challenge.ai_generated:
            score += 0.1
        
        # Pénalité pour défis trop longs si joueur préfère du rapide
        if challenge.time_limit and challenge.time_limit > timedelta(minutes=30):
            score -= 0.1
        
        return score
    
    def _personalize_challenge(self, challenge: Challenge, player: PlayerProfile) -> Challenge:
        """Personnalise un défi pour un joueur spécifique"""
        
        personalized = Challenge(
            challenge_id=f"{challenge.challenge_id}_p_{player.player_id}",
            type=challenge.type,
            difficulty=challenge.difficulty,
            title=f"{challenge.title} - {player.name}",
            description=challenge.description,
            objectives=challenge.objectives.copy(),
            success_criteria=challenge.success_criteria.copy(),
            rewards=self._adjust_rewards(challenge.rewards, player),
            time_limit=challenge.time_limit,
            hints=challenge.hints.copy(),
            created_at=datetime.now(),
            expires_at=None,
            ai_generated=True
        )
        
        # Ajustements basés sur le profil
        if player.learning_style == "fast":
            personalized.time_limit = timedelta(minutes=max(5, personalized.time_limit.total_seconds() // 60 * 0.8))
        elif player.learning_style == "thorough":
            personalized.hints.append("🔍 Prenez votre temps pour analyser tous les aspects")
        
        return personalized
    
    def _adjust_rewards(self, base_rewards: Dict[str, int], player: PlayerProfile) -> Dict[str, int]:
        """Ajuste les récompenses selon le niveau du joueur"""
        
        multiplier = 1.0 + (player.level - 1) * 0.1  # 10% de bonus par niveau
        
        adjusted = {}
        for reward_type, amount in base_rewards.items():
            adjusted[reward_type] = int(amount * multiplier)
        
        return adjusted
    
    def _get_random_beginner_challenge(self) -> Challenge:
        """Retourne un défi débutant au hasard"""
        
        beginner_challenges = []
        for challenges in self.challenges_pool.values():
            for challenge in challenges:
                if challenge.difficulty == DifficultyLevel.BEGINNER:
                    beginner_challenges.append(challenge)
        
        return random.choice(beginner_challenges) if beginner_challenges else self._create_default_challenge()
    
    def _create_default_challenge(self) -> Challenge:
        """Crée un défi par défaut"""
        
        return Challenge(
            challenge_id="default_beginner",
            type=ChallengeType.ESTIMATION,
            difficulty=DifficultyLevel.BEGINNER,
            title="🎯 Premier Défi: Estimation Simple",
            description="Estimez la durée de développement d'un site web statique avec 5 pages.",
            objectives=[
                "Analyser les besoins",
                "Estimer chaque page",
                "Prévoir les tests et déploiement"
            ],
            success_criteria={"accuracy_threshold": 0.20},
            rewards={"xp": 50, "coins": 25},
            time_limit=timedelta(minutes=10),
            hints=["💡 Commencez simple", "⏱️ N'oubliez pas les tests"],
            created_at=datetime.now(),
            expires_at=None
        )
    
    def start_challenge_session(self, player_id: str, challenge: Challenge) -> GameSession:
        """Démarre une session de défi"""
        
        session = GameSession(
            session_id=f"session_{datetime.now().timestamp()}",
            player_id=player_id,
            start_time=datetime.now(),
            current_challenge=challenge,
            score=0,
            actions_taken=[],
            ai_coach_messages=[]
        )
        
        # Message de coaching initial
        initial_coaching = self.ai_coach.get_challenge_introduction(challenge, self.players.get(player_id))
        session.ai_coach_messages.append(initial_coaching)
        
        self.active_sessions[session.session_id] = session
        return session
    
    def process_challenge_action(self, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une action du joueur pendant un défi"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session non trouvée"}
        
        session = self.active_sessions[session_id]
        session.actions_taken.append({
            "timestamp": datetime.now(),
            "action": action
        })
        
        # Analyse de l'action par l'IA coach
        feedback = self.ai_coach.analyze_action(action, session.current_challenge)
        session.ai_coach_messages.append(feedback)
        
        # Vérification des critères de succès
        success_check = self._check_success_criteria(session)
        
        return {
            "feedback": feedback,
            "success_check": success_check,
            "hints_unlocked": self._get_contextual_hints(session)
        }
    
    def complete_challenge(self, session_id: str, final_result: Dict[str, Any]) -> Dict[str, Any]:
        """Complète un défi et calcule les récompenses"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session non trouvée"}
        
        session = self.active_sessions[session_id]
        challenge = session.current_challenge
        player = self.players[session.player_id]
        
        # Évaluation finale
        performance = self._evaluate_performance(session, final_result)
        
        # Calcul des récompenses
        earned_rewards = self._calculate_earned_rewards(challenge, performance)
        
        # Mise à jour du profil joueur
        self._update_player_profile(player, challenge, performance, earned_rewards)
        
        # Vérification des achievements
        new_achievements = self._check_achievements(player)
        
        # Nettoyage de la session
        del self.active_sessions[session_id]
        
        # Message de félicitations de l'IA coach
        completion_message = self.ai_coach.get_completion_message(performance, earned_rewards, new_achievements)
        
        return {
            "performance": performance,
            "rewards": earned_rewards,
            "new_achievements": new_achievements,
            "coach_message": completion_message,
            "next_challenge_suggestion": self.get_personalized_challenge(session.player_id)
        }
    
    def _evaluate_performance(self, session: GameSession, result: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue la performance du joueur"""
        
        challenge = session.current_challenge
        duration = datetime.now() - session.start_time
        
        performance = {
            "completion_time": duration.total_seconds(),
            "actions_count": len(session.actions_taken),
            "success": False,
            "accuracy": 0.0,
            "efficiency": 0.0,
            "creativity": 0.0
        }
        
        # Évaluation spécifique selon le type de défi
        if challenge.type == ChallengeType.ESTIMATION:
            actual_estimate = result.get("estimated_duration", 0)
            target_value = result.get("target_duration", 100)  # Valeur de référence
            
            if target_value > 0:
                error_ratio = abs(actual_estimate - target_value) / target_value
                accuracy_threshold = challenge.success_criteria.get("accuracy_threshold", 0.15)
                
                performance["accuracy"] = max(0, 1 - error_ratio / accuracy_threshold)
                performance["success"] = error_ratio <= accuracy_threshold
        
        # Évaluation de l'efficacité (basée sur le temps)
        if challenge.time_limit:
            time_ratio = duration / challenge.time_limit
            performance["efficiency"] = max(0, 2 - time_ratio)  # Bonus pour rapidité
        
        # Évaluation de la créativité (basée sur les actions non-standard)
        unique_actions = len(set(action["action"].get("type") for action in session.actions_taken))
        performance["creativity"] = min(1.0, unique_actions / 5)
        
        return performance
    
    def _calculate_earned_rewards(self, challenge: Challenge, performance: Dict[str, Any]) -> Dict[str, int]:
        """Calcule les récompenses gagnées"""
        
        base_rewards = challenge.rewards.copy()
        earned = {}
        
        # Facteur de performance global
        performance_factor = (performance["accuracy"] + performance["efficiency"] + performance["creativity"]) / 3
        
        # Bonus de succès
        success_multiplier = 1.5 if performance["success"] else 0.7
        
        for reward_type, amount in base_rewards.items():
            earned_amount = int(amount * performance_factor * success_multiplier)
            earned[reward_type] = earned_amount
        
        # Bonus spéciaux
        if performance["efficiency"] > 1.2:  # Très rapide
            earned["speed_bonus"] = 50
        
        if performance["accuracy"] > 0.9:  # Très précis
            earned["precision_bonus"] = 100
        
        return earned
    
    def _update_player_profile(self, player: PlayerProfile, challenge: Challenge, 
                             performance: Dict[str, Any], rewards: Dict[str, int]):
        """Met à jour le profil du joueur"""
        
        # XP et niveau
        xp_gained = rewards.get("xp", 0)
        player.total_xp += xp_gained
        
        new_level = 1 + int(player.total_xp / 1000)  # 1000 XP par niveau
        if new_level > player.level:
            player.level = new_level
            # Bonus de level up
            player.total_xp += 100
        
        # Défis complétés
        player.challenges_completed += 1
        
        # Streak
        # (Logique de streak simplifiée - à améliorer avec vraies dates)
        player.current_streak += 1
        player.longest_streak = max(player.longest_streak, player.current_streak)
        
        # Compétences
        skill_key = challenge.type.value
        if skill_key in player.skill_ratings:
            skill_gain = rewards.get(f"{skill_key}_skill", 0)
            player.skill_ratings[skill_key] += skill_gain
        
        # Préférences (apprentissage adaptatif)
        if performance["success"] and challenge.type not in player.preferred_challenge_types:
            player.preferred_challenge_types.append(challenge.type)
    
    def _check_achievements(self, player: PlayerProfile) -> List[Achievement]:
        """Vérifie et débloque les nouveaux achievements"""
        
        new_achievements = []
        
        for achievement_id, achievement in self.achievements_catalog.items():
            if achievement_id in player.achievements_unlocked:
                continue  # Déjà débloqué
            
            if self._check_achievement_criteria(player, achievement):
                achievement.unlocked_at = datetime.now()
                player.achievements_unlocked.append(achievement_id)
                player.total_xp += achievement.points
                new_achievements.append(achievement)
        
        return new_achievements
    
    def _check_achievement_criteria(self, player: PlayerProfile, achievement: Achievement) -> bool:
        """Vérifie si les critères d'un achievement sont remplis"""
        
        criteria = achievement.unlock_criteria
        
        # Exemples de vérifications
        if "consecutive_accurate_estimations" in criteria:
            # Logique simplifiée - à améliorer avec historique réel
            return player.challenges_completed >= criteria["consecutive_accurate_estimations"]
        
        if "daily_streak" in criteria:
            return player.current_streak >= criteria["daily_streak"]
        
        if "different_domains_completed" in criteria:
            # Logique simplifiée
            return len(player.preferred_challenge_types) >= criteria["different_domains_completed"]
        
        # Critères génériques
        for key, required_value in criteria.items():
            if hasattr(player, key):
                if getattr(player, key) < required_value:
                    return False
        
        return True
    
    def get_leaderboard(self, category: str = "global", limit: int = 10) -> List[Dict[str, Any]]:
        """Récupère le classement des joueurs"""
        
        players_list = list(self.players.values())
        
        if category == "global":
            players_list.sort(key=lambda p: p.total_xp, reverse=True)
        elif category == "level":
            players_list.sort(key=lambda p: p.level, reverse=True)
        elif category == "streak":
            players_list.sort(key=lambda p: p.longest_streak, reverse=True)
        elif category in ["estimation", "optimization", "risk_management"]:
            players_list.sort(key=lambda p: p.skill_ratings.get(category, 0), reverse=True)
        
        leaderboard = []
        for i, player in enumerate(players_list[:limit], 1):
            leaderboard.append({
                "rank": i,
                "player_name": player.name,
                "level": player.level,
                "total_xp": player.total_xp,
                "current_streak": player.current_streak,
                "achievements_count": len(player.achievements_unlocked),
                "specialty": max(player.skill_ratings.items(), key=lambda x: x[1])[0] if player.skill_ratings else "généraliste"
            })
        
        return leaderboard
    
    def get_player_dashboard(self, player_id: str) -> Dict[str, Any]:
        """Récupère le dashboard complet d'un joueur"""
        
        if player_id not in self.players:
            return {"error": "Joueur non trouvé"}
        
        player = self.players[player_id]
        
        # Prochain défi suggéré
        next_challenge = self.get_personalized_challenge(player_id)
        
        # Progression vers le prochain niveau
        current_level_xp = (player.level - 1) * 1000
        next_level_xp = player.level * 1000
        level_progress = (player.total_xp - current_level_xp) / (next_level_xp - current_level_xp)
        
        # Achievements récents
        recent_achievements = sorted(
            [self.achievements_catalog[aid] for aid in player.achievements_unlocked],
            key=lambda a: a.unlocked_at or datetime.min,
            reverse=True
        )[:5]
        
        return {
            "player_profile": {
                "name": player.name,
                "level": player.level,
                "total_xp": player.total_xp,
                "level_progress": level_progress,
                "current_streak": player.current_streak,
                "longest_streak": player.longest_streak,
                "challenges_completed": player.challenges_completed
            },
            "skills": player.skill_ratings,
            "next_challenge": {
                "title": next_challenge.title,
                "type": next_challenge.type.value,
                "difficulty": next_challenge.difficulty.value,
                "estimated_time": str(next_challenge.time_limit) if next_challenge.time_limit else "Flexible",
                "rewards": next_challenge.rewards
            },
            "recent_achievements": [
                {
                    "name": ach.name,
                    "description": ach.description,
                    "rarity": ach.rarity,
                    "points": ach.points
                }
                for ach in recent_achievements
            ],
            "global_rank": self._get_player_rank(player_id),
            "badges": player.badges,
            "learning_insights": self.ai_coach.get_learning_insights(player)
        }
    
    def _get_player_rank(self, player_id: str) -> int:
        """Calcule le rang global d'un joueur"""
        
        player_xp = self.players[player_id].total_xp
        higher_players = sum(1 for p in self.players.values() if p.total_xp > player_xp)
        
        return higher_players + 1


class AICoach:
    """Coach IA pour guidance et motivation"""
    
    def __init__(self):
        self.coaching_styles = ["motivational", "analytical", "supportive", "challenging"]
        self.player_coaching_preferences = {}
    
    def get_challenge_introduction(self, challenge: Challenge, player: Optional[PlayerProfile]) -> str:
        """Génère une introduction personnalisée pour un défi"""
        
        if not player:
            return f"🎯 Prêt pour le défi '{challenge.title}' ? Montrez-moi de quoi vous êtes capable !"
        
        style = self._get_coaching_style(player)
        
        if style == "motivational":
            return f"💪 {player.name}, voici un défi parfait pour votre niveau {player.level} ! '{challenge.title}' - Je sais que vous allez le réussir brillamment !"
        elif style == "analytical":
            return f"🔍 Défi: '{challenge.title}' | Difficulté: {challenge.difficulty.value} | Objectifs: {len(challenge.objectives)} | Temps estimé: {challenge.time_limit}"
        elif style == "supportive":
            return f"🤗 Ne vous inquiétez pas {player.name}, je serai là pour vous guider dans '{challenge.title}'. Prenez votre temps et faites de votre mieux !"
        else:  # challenging
            return f"🔥 {player.name}, niveau {player.level} hein ? Prouvez-le avec '{challenge.title}' ! Ce défi va vraiment tester vos compétences !"
    
    def analyze_action(self, action: Dict[str, Any], challenge: Challenge) -> str:
        """Analyse une action et fournit un feedback"""
        
        action_type = action.get("type", "unknown")
        
        feedback_templates = {
            "estimation": [
                "📊 Bonne approche ! Continuez à décomposer le problème.",
                "🎯 Cette estimation semble dans la bonne fourchette.",
                "⚠️ Attention, vous pourriez sous-estimer la complexité.",
                "💡 Pensez aux dépendances et intégrations."
            ],
            "optimization": [
                "⚡ Excellente idée d'optimisation !",
                "🔧 Cette solution pourrait poser des problèmes de maintenance.",
                "💰 Impact budgétaire positif de cette décision !",
                "⚖️ Bon équilibre entre coût et performance."
            ],
            "risk_analysis": [
                "🛡️ Risque bien identifié et documenté.",
                "📋 Stratégie de mitigation solide !",
                "⚠️ Ce risque pourrait en cacher d'autres...",
                "🎯 Priorité bien évaluée pour ce risque."
            ]
        }
        
        relevant_feedback = feedback_templates.get(action_type, [
            "👍 Action enregistrée, continuez !",
            "🤔 Intéressant, voyons où cela nous mène.",
            "📈 Progression notable sur ce défi."
        ])
        
        return random.choice(relevant_feedback)
    
    def get_completion_message(self, performance: Dict[str, Any], rewards: Dict[str, int], 
                              achievements: List[Achievement]) -> str:
        """Message de félicitations personnalisé"""
        
        if performance["success"]:
            base_message = "🎉 Félicitations ! Défi réussi avec brio !"
        else:
            base_message = "💪 Bon effort ! Chaque tentative vous fait progresser."
        
        # Ajout des détails de performance
        if performance["accuracy"] > 0.8:
            base_message += " 🎯 Précision exceptionnelle !"
        if performance["efficiency"] > 1.0:
            base_message += " ⚡ Impressionnante rapidité !"
        if performance["creativity"] > 0.7:
            base_message += " 💡 Approche créative remarquable !"
        
        # Récompenses
        xp_gained = rewards.get("xp", 0)
        if xp_gained > 0:
            base_message += f"\n📈 +{xp_gained} XP gagnés !"
        
        # Nouveaux achievements
        if achievements:
            base_message += f"\n🏆 Nouveaux achievements débloqués : {', '.join([a.name for a in achievements])}"
        
        return base_message
    
    def get_learning_insights(self, player: PlayerProfile) -> List[str]:
        """Génère des insights d'apprentissage personnalisés"""
        
        insights = []
        
        # Analyse des compétences
        skills = player.skill_ratings
        strongest_skill = max(skills.items(), key=lambda x: x[1])
        weakest_skill = min(skills.items(), key=lambda x: x[1])
        
        insights.append(f"🎯 Votre point fort: {strongest_skill[0]} (niveau {strongest_skill[1]})")
        insights.append(f"📈 À développer: {weakest_skill[0]} - Tentez plus de défis dans ce domaine !")
        
        # Analyse de progression
        if player.current_streak > 5:
            insights.append(f"🔥 Série impressionnante de {player.current_streak} jours ! Continuez sur cette lancée !")
        
        if len(player.achievements_unlocked) > 0:
            insights.append(f"🏆 {len(player.achievements_unlocked)} achievements débloqués - Vous progressez bien !")
        
        # Conseils adaptatifs
        if player.challenges_completed > 10:
            insights.append("🎖️ Expérience solide ! Tentez des défis de niveau supérieur.")
        else:
            insights.append("🌱 Continuez à explorer différents types de défis pour développer vos compétences.")
        
        return insights
    
    def _get_coaching_style(self, player: PlayerProfile) -> str:
        """Détermine le style de coaching optimal pour un joueur"""
        
        if player.player_id in self.player_coaching_preferences:
            return self.player_coaching_preferences[player.player_id]
        
        # Style adaptatif basé sur le profil
        if player.level < 5:
            return "supportive"
        elif player.current_streak > 10:
            return "challenging"
        elif np.mean(list(player.skill_ratings.values())) > 300:
            return "analytical"
        else:
            return "motivational"


class DifficultyAdapter:
    """Adaptateur intelligent de difficulté"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
    
    def record_performance(self, player_id: str, challenge_type: ChallengeType, 
                          difficulty: DifficultyLevel, success: bool, performance_score: float):
        """Enregistre une performance pour adaptation future"""
        
        self.performance_history[player_id].append({
            "timestamp": datetime.now(),
            "challenge_type": challenge_type,
            "difficulty": difficulty,
            "success": success,
            "performance_score": performance_score
        })
    
    def suggest_next_difficulty(self, player_id: str, challenge_type: ChallengeType) -> DifficultyLevel:
        """Suggère le niveau de difficulté optimal"""
        
        if player_id not in self.performance_history:
            return DifficultyLevel.BEGINNER
        
        # Analyse des performances récentes pour ce type de défi
        recent_performances = [
            p for p in self.performance_history[player_id][-10:]  # 10 dernières performances
            if p["challenge_type"] == challenge_type
        ]
        
        if not recent_performances:
            return DifficultyLevel.BEGINNER
        
        # Calcul du taux de succès et score moyen
        success_rate = sum(1 for p in recent_performances if p["success"]) / len(recent_performances)
        avg_score = np.mean([p["performance_score"] for p in recent_performances])
        
        current_difficulty = recent_performances[-1]["difficulty"]
        
        # Logique d'adaptation
        if success_rate > 0.8 and avg_score > 0.7:
            # Très bon, augmenter la difficulté
            return self._increase_difficulty(current_difficulty)
        elif success_rate < 0.4 or avg_score < 0.3:
            # Difficultés, réduire la difficulté
            return self._decrease_difficulty(current_difficulty)
        else:
            # Maintenir le niveau actuel
            return current_difficulty
    
    def _increase_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Augmente le niveau de difficulté"""
        
        mapping = {
            DifficultyLevel.BEGINNER: DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.INTERMEDIATE: DifficultyLevel.ADVANCED,
            DifficultyLevel.ADVANCED: DifficultyLevel.EXPERT,
            DifficultyLevel.EXPERT: DifficultyLevel.EXPERT  # Maximum
        }
        return mapping[current]
    
    def _decrease_difficulty(self, current: DifficultyLevel) -> DifficultyLevel:
        """Diminue le niveau de difficulté"""
        
        mapping = {
            DifficultyLevel.EXPERT: DifficultyLevel.ADVANCED,
            DifficultyLevel.ADVANCED: DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.INTERMEDIATE: DifficultyLevel.BEGINNER,
            DifficultyLevel.BEGINNER: DifficultyLevel.BEGINNER  # Minimum
        }
        return mapping[current]


# Instance globale
gamification_engine = IntelligentGamificationEngine()