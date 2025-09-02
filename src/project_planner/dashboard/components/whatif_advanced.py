"""
🎯 Advanced What-If Scenario Engine - PlannerIA
Système révolutionnaire d'aide à la décision avec IA, Monte Carlo, Gamification & Optimisation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.optimize as opt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import copy
import json
import asyncio
import random
from dataclasses import dataclass
from enum import Enum
import math

# === CORE DATA STRUCTURES ===

class ScenarioType(Enum):
    AGGRESSIVE = "Délais agressifs"
    CONSERVATIVE = "Approche prudente"  
    BALANCED = "Équilibré"
    INNOVATION = "Innovation focus"
    CRISIS = "Mode crise"

@dataclass
class ScenarioResult:
    """Résultat complet d'un scénario"""
    name: str
    duration: float
    cost: float
    quality_score: float
    risk_score: float
    success_probability: float
    team_size: float
    complexity_factor: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    recommendations: List[str]
    ai_generated: bool = False

@dataclass 
class GameState:
    """État du jeu de scénarios"""
    player_score: int = 0
    level: int = 1
    challenges_completed: int = 0
    achievements: List[str] = None
    current_challenge: str = ""
    time_bonus: float = 1.0

# === AI-POWERED SCENARIO PREDICTOR ===

class AIScenarioPredictor:
    """Intelligence Artificielle pour prédiction et génération de scénarios"""
    
    def __init__(self):
        self.historical_patterns = self._load_patterns()
        self.learning_rate = 0.1
        self.confidence_threshold = 0.75
        
    def _load_patterns(self) -> Dict[str, Any]:
        """Charge les patterns historiques (simulé)"""
        return {
            'domain_multipliers': {
                'ecommerce': {'duration': 1.2, 'cost': 1.1, 'risk': 0.8},
                'fintech': {'duration': 1.6, 'cost': 1.4, 'risk': 1.3},
                'healthcare': {'duration': 1.8, 'cost': 1.5, 'risk': 1.4},
                'enterprise': {'duration': 1.5, 'cost': 1.3, 'risk': 1.1}
            },
            'team_dynamics': {
                'small_team': (1, 3, 0.9),  # (min, max, efficiency)
                'medium_team': (4, 8, 1.0),
                'large_team': (9, 15, 0.8)
            },
            'success_patterns': {
                'high_success': {'duration_range': (0.8, 1.2), 'cost_range': (0.9, 1.3)},
                'medium_success': {'duration_range': (1.0, 1.5), 'cost_range': (1.0, 1.6)},
                'low_success': {'duration_range': (1.3, 2.0), 'cost_range': (1.4, 2.2)}
            }
        }
    
    def predict_scenario_outcome(self, scenario_params: Dict[str, Any], 
                               project_context: Dict[str, Any]) -> ScenarioResult:
        """Prédit le résultat d'un scénario avec IA"""
        
        # Analyse contextuelle
        domain = project_context.get('domain', 'enterprise')
        team_size = scenario_params.get('team_multiplier', 1.0) * project_context.get('base_team_size', 5)
        
        # Application des patterns IA
        domain_mult = self.historical_patterns['domain_multipliers'].get(domain, 
            {'duration': 1.0, 'cost': 1.0, 'risk': 1.0})
        
        # Calculs prédictifs avancés
        predicted_duration = self._predict_duration(scenario_params, domain_mult, team_size)
        predicted_cost = self._predict_cost(scenario_params, domain_mult, team_size)
        quality_score = self._predict_quality(scenario_params, project_context)
        risk_score = self._predict_risk(scenario_params, domain_mult)
        success_prob = self._calculate_success_probability(predicted_duration, predicted_cost, quality_score, risk_score)
        
        # Intervalles de confiance IA
        confidence_intervals = self._calculate_ai_confidence_intervals(
            predicted_duration, predicted_cost, quality_score
        )
        
        # Recommandations intelligentes
        recommendations = self._generate_ai_recommendations(
            scenario_params, predicted_duration, predicted_cost, risk_score
        )
        
        return ScenarioResult(
            name=f"AI Prediction {datetime.now().strftime('%H:%M:%S')}",
            duration=predicted_duration,
            cost=predicted_cost,
            quality_score=quality_score,
            risk_score=risk_score,
            success_probability=success_prob,
            team_size=team_size,
            complexity_factor=scenario_params.get('complexity_factor', 1.0),
            confidence_intervals=confidence_intervals,
            recommendations=recommendations,
            ai_generated=True
        )
    
    def _predict_duration(self, params: Dict, domain_mult: Dict, team_size: float) -> float:
        """Prédiction IA de la durée"""
        base_duration = params.get('base_duration', 100)
        duration_mult = params.get('duration_multiplier', 1.0)
        complexity = params.get('complexity_factor', 1.0)
        
        # Brooks' Law avec IA
        if team_size > 8:
            communication_overhead = 1 + (team_size - 8) * 0.08
        else:
            communication_overhead = 1.0
            
        # Courbe d'apprentissage
        learning_factor = 0.95 ** (team_size - 1) if team_size <= 5 else 0.85
        
        ai_duration = (base_duration * duration_mult * domain_mult['duration'] * 
                      complexity * communication_overhead * learning_factor)
        
        return max(10, ai_duration)
    
    def _predict_cost(self, params: Dict, domain_mult: Dict, team_size: float) -> float:
        """Prédiction IA du coût"""
        base_cost = params.get('base_cost', 50000)
        cost_mult = params.get('cost_multiplier', 1.0)
        
        # Économies d'échelle avec seuils
        if team_size > 10:
            scale_factor = 1.1  # Surcoût coordination
        elif team_size > 5:
            scale_factor = 0.95  # Efficacité optimale
        else:
            scale_factor = 1.0
            
        # Facteur expérience (simulé)
        experience_discount = random.uniform(0.9, 1.1)
        
        ai_cost = (base_cost * cost_mult * domain_mult['cost'] * 
                  team_size * scale_factor * experience_discount)
        
        return max(5000, ai_cost)
    
    def _predict_quality(self, params: Dict, context: Dict) -> float:
        """Prédiction de la qualité"""
        time_pressure = 1.0 / params.get('duration_multiplier', 1.0)
        team_quality = min(1.2, params.get('team_multiplier', 1.0))
        complexity_challenge = params.get('complexity_factor', 1.0)
        
        # Formule qualité sophistiquée
        quality_base = 0.8
        time_impact = max(0.3, 1.0 - (time_pressure - 1.0) * 0.4)
        team_impact = min(1.2, team_quality)
        complexity_impact = max(0.5, 1.0 - (complexity_challenge - 1.0) * 0.3)
        
        final_quality = quality_base * time_impact * team_impact * complexity_impact
        return max(0.2, min(1.0, final_quality))
    
    def _predict_risk(self, params: Dict, domain_mult: Dict) -> float:
        """Prédiction du risque"""
        base_risk = 5.0
        
        # Facteurs de risque
        timeline_risk = abs(params.get('duration_multiplier', 1.0) - 1.0) * 3
        budget_risk = abs(params.get('cost_multiplier', 1.0) - 1.0) * 2
        complexity_risk = (params.get('complexity_factor', 1.0) - 1.0) * 4
        domain_risk = domain_mult['risk'] - 1.0
        
        total_risk = base_risk + timeline_risk + budget_risk + complexity_risk + domain_risk
        return max(1.0, min(10.0, total_risk))
    
    def _calculate_success_probability(self, duration: float, cost: float, 
                                     quality: float, risk: float) -> float:
        """Calcul IA de la probabilité de succès"""
        # Normalisation des facteurs
        duration_factor = max(0.2, 1.0 - abs(duration - 100) / 200)
        cost_factor = max(0.3, 1.0 - abs(cost - 50000) / 100000)
        quality_factor = quality
        risk_factor = max(0.1, (10 - risk) / 10)
        
        # Pondération intelligente
        weights = [0.25, 0.2, 0.35, 0.2]  # duration, cost, quality, risk
        factors = [duration_factor, cost_factor, quality_factor, risk_factor]
        
        success_prob = sum(w * f for w, f in zip(weights, factors))
        return max(0.05, min(0.99, success_prob))
    
    def _calculate_ai_confidence_intervals(self, duration: float, cost: float, 
                                         quality: float) -> Dict[str, Tuple[float, float]]:
        """Intervalles de confiance IA"""
        duration_std = duration * 0.15  # 15% de variation
        cost_std = cost * 0.2  # 20% de variation
        quality_std = quality * 0.1  # 10% de variation
        
        return {
            'duration_90': (duration - 1.645*duration_std, duration + 1.645*duration_std),
            'duration_50': (duration - 0.675*duration_std, duration + 0.675*duration_std),
            'cost_90': (cost - 1.645*cost_std, cost + 1.645*cost_std),
            'cost_50': (cost - 0.675*cost_std, cost + 0.675*cost_std),
            'quality_90': (max(0, quality - 1.645*quality_std), min(1, quality + 1.645*quality_std))
        }
    
    def _generate_ai_recommendations(self, params: Dict, duration: float, 
                                   cost: float, risk: float) -> List[str]:
        """Recommandations intelligentes basées sur l'IA"""
        recommendations = []
        
        # Analyse de la durée
        if duration > 150:
            recommendations.append("🕒 Durée élevée détectée. Considérez l'ajout de ressources senior ou la parallélisation des tâches.")
        
        # Analyse du coût
        if cost > 100000:
            recommendations.append("💰 Budget élevé. Évaluez le ROI et négociez des taux préférentiels avec les fournisseurs.")
            
        # Analyse du risque
        if risk > 7:
            recommendations.append("⚠️ Risque élevé. Planifiez des points de contrôle fréquents et un plan de contingence.")
            
        # Recommandations d'optimisation
        duration_mult = params.get('duration_multiplier', 1.0)
        if duration_mult < 0.8:
            recommendations.append("⚡ Timeline agressive. Priorisez les fonctionnalités critiques et préparez un MVP.")
            
        team_mult = params.get('team_multiplier', 1.0)
        if team_mult > 1.5:
            recommendations.append("👥 Équipe large. Investissez dans les outils de communication et la gouvernance de projet.")
            
        return recommendations
    
    def auto_generate_scenarios(self, project_context: Dict[str, Any], 
                              num_scenarios: int = 8) -> List[ScenarioResult]:
        """Auto-génération de scénarios optimaux par IA"""
        scenarios = []
        
        # Templates de scénarios intelligents
        scenario_templates = [
            # Scénario optimiste
            {
                'name': '🚀 Scénario Optimiste IA',
                'duration_multiplier': 0.8,
                'cost_multiplier': 0.9,
                'team_multiplier': 1.2,
                'complexity_factor': 0.9
            },
            # Scénario réaliste
            {
                'name': '⚖️ Scénario Équilibré IA',
                'duration_multiplier': 1.0,
                'cost_multiplier': 1.0,
                'team_multiplier': 1.0,
                'complexity_factor': 1.0
            },
            # Scénario conservateur
            {
                'name': '🛡️ Scénario Sécurisé IA',
                'duration_multiplier': 1.3,
                'cost_multiplier': 1.2,
                'team_multiplier': 0.8,
                'complexity_factor': 1.1
            },
            # Scénario innovation
            {
                'name': '💡 Scénario Innovation IA',
                'duration_multiplier': 1.4,
                'cost_multiplier': 1.5,
                'team_multiplier': 1.1,
                'complexity_factor': 1.6
            },
            # Scénario budget serré
            {
                'name': '💸 Budget Optimisé IA',
                'duration_multiplier': 1.2,
                'cost_multiplier': 0.7,
                'team_multiplier': 0.6,
                'complexity_factor': 0.8
            },
            # Scénario délai court
            {
                'name': '⏱️ Délai Express IA',
                'duration_multiplier': 0.6,
                'cost_multiplier': 1.4,
                'team_multiplier': 1.8,
                'complexity_factor': 1.2
            },
            # Scénario qualité premium
            {
                'name': '⭐ Qualité Premium IA',
                'duration_multiplier': 1.1,
                'cost_multiplier': 1.3,
                'team_multiplier': 1.2,
                'complexity_factor': 0.9
            },
            # Scénario startup agile
            {
                'name': '🎯 Startup Agile IA',
                'duration_multiplier': 0.9,
                'cost_multiplier': 0.8,
                'team_multiplier': 0.7,
                'complexity_factor': 1.1
            }
        ]
        
        for template in scenario_templates[:num_scenarios]:
            # Ajout de variation aléatoire
            varied_params = copy.deepcopy(template)
            for key in ['duration_multiplier', 'cost_multiplier', 'team_multiplier', 'complexity_factor']:
                if key in varied_params:
                    variation = random.uniform(0.95, 1.05)  # ±5% de variation
                    varied_params[key] *= variation
            
            # Ajout du contexte projet
            varied_params.update({
                'base_duration': project_context.get('base_duration', 100),
                'base_cost': project_context.get('base_cost', 50000)
            })
            
            # Prédiction du scénario
            scenario_result = self.predict_scenario_outcome(varied_params, project_context)
            scenario_result.name = template['name']
            scenarios.append(scenario_result)
        
        return scenarios

# === MONTE CARLO SIMULATION ENGINE ===

class MonteCarloEngine:
    """
    Moteur de simulation Monte Carlo ultra-performant
    
    RÔLE: Interface utilisateur principal pour analyse de scénarios
    - Simulations interactives avec paramètres configurables
    - Visualisations avancées et matrices de corrélation  
    - Exploration approfondie de l'espace des possibles
    """
    
    def __init__(self, num_simulations: int = 50000):
        self.num_simulations = num_simulations
        self.random_state = np.random.RandomState(42)
        
    def run_simulation(self, scenario_params: Dict[str, Any], 
                      project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute simulation Monte Carlo complète"""
        
        # Variables d'entrée
        base_duration = project_context.get('base_duration', 100)
        base_cost = project_context.get('base_cost', 50000)
        
        # Paramètres de distribution
        duration_std = base_duration * 0.25
        cost_std = base_cost * 0.3
        
        # Arrays pour simulations parallèles
        durations = np.zeros(self.num_simulations)
        costs = np.zeros(self.num_simulations)
        qualities = np.zeros(self.num_simulations)
        success_flags = np.zeros(self.num_simulations, dtype=bool)
        
        # Simulation vectorisée pour performance
        for i in range(self.num_simulations):
            # Facteurs aléatoires corrélés
            correlation_factor = self.random_state.normal(1.0, 0.1)
            
            # Durée avec distribution log-normale
            duration_factor = self.random_state.lognormal(0, 0.2) * correlation_factor
            sim_duration = base_duration * scenario_params.get('duration_multiplier', 1.0) * duration_factor
            durations[i] = max(10, sim_duration)
            
            # Coût avec corrélation à la durée
            cost_correlation = 0.7  # Corrélation durée-coût
            cost_factor = (cost_correlation * duration_factor + 
                          (1 - cost_correlation) * self.random_state.lognormal(0, 0.25))
            sim_cost = base_cost * scenario_params.get('cost_multiplier', 1.0) * cost_factor
            costs[i] = max(5000, sim_cost)
            
            # Qualité inversement corrélée à la pression temporelle
            time_pressure = base_duration / sim_duration
            quality_base = 0.8
            quality_variance = self.random_state.normal(0, 0.1)
            sim_quality = quality_base - (time_pressure - 1.0) * 0.2 + quality_variance
            qualities[i] = max(0.1, min(1.0, sim_quality))
            
            # Critères de succès
            duration_ok = sim_duration <= base_duration * 1.5
            cost_ok = sim_cost <= base_cost * 1.8
            quality_ok = sim_quality >= 0.6
            success_flags[i] = duration_ok and cost_ok and quality_ok
        
        # Calcul des statistiques
        results = {
            'duration_stats': self._calculate_distribution_stats(durations),
            'cost_stats': self._calculate_distribution_stats(costs),
            'quality_stats': self._calculate_distribution_stats(qualities),
            'success_rate': np.mean(success_flags),
            'simulation_data': {
                'durations': durations.tolist()[:1000],  # Sample pour viz
                'costs': costs.tolist()[:1000],
                'qualities': qualities.tolist()[:1000]
            },
            'risk_metrics': self._calculate_risk_metrics(durations, costs, qualities, base_duration, base_cost),
            'confidence_intervals': self._calculate_confidence_intervals(durations, costs, qualities)
        }
        
        return results
    
    def _calculate_distribution_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calcule les statistiques de distribution"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'p10': float(np.percentile(data, 10)),
            'p25': float(np.percentile(data, 25)),
            'p75': float(np.percentile(data, 75)),
            'p90': float(np.percentile(data, 90)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }
    
    def _calculate_risk_metrics(self, durations: np.ndarray, costs: np.ndarray, 
                               qualities: np.ndarray, base_duration: float, base_cost: float) -> Dict[str, float]:
        """Calcule les métriques de risque avancées"""
        
        duration_overrun = np.sum(durations > base_duration * 1.2) / len(durations)
        cost_overrun = np.sum(costs > base_cost * 1.5) / len(costs)
        quality_risk = np.sum(qualities < 0.7) / len(qualities)
        
        # Value at Risk (VaR)
        duration_var_95 = np.percentile(durations, 95)
        cost_var_95 = np.percentile(costs, 95)
        
        # Expected Shortfall (ES)
        duration_es = np.mean(durations[durations >= duration_var_95])
        cost_es = np.mean(costs[costs >= cost_var_95])
        
        return {
            'duration_overrun_prob': duration_overrun,
            'cost_overrun_prob': cost_overrun,
            'quality_risk_prob': quality_risk,
            'duration_var_95': duration_var_95,
            'cost_var_95': cost_var_95,
            'duration_expected_shortfall': duration_es,
            'cost_expected_shortfall': cost_es,
            'overall_risk_score': (duration_overrun + cost_overrun + quality_risk) / 3
        }
    
    def _calculate_confidence_intervals(self, durations: np.ndarray, costs: np.ndarray, 
                                      qualities: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calcule les intervalles de confiance multiples"""
        confidence_levels = [50, 68, 80, 90, 95, 99]
        intervals = {}
        
        for level in confidence_levels:
            alpha = (100 - level) / 2
            
            intervals[f'duration_{level}'] = (
                float(np.percentile(durations, alpha)),
                float(np.percentile(durations, 100 - alpha))
            )
            intervals[f'cost_{level}'] = (
                float(np.percentile(costs, alpha)),
                float(np.percentile(costs, 100 - alpha))
            )
            intervals[f'quality_{level}'] = (
                float(np.percentile(qualities, alpha)),
                float(np.percentile(qualities, 100 - alpha))
            )
        
        return intervals

# === GAMIFICATION ENGINE ===

class GamificationEngine:
    """Moteur de gamification pour engagement utilisateur"""
    
    def __init__(self):
        self.achievements = self._init_achievements()
        self.challenges = self._init_challenges()
        
    def _init_achievements(self) -> List[Dict[str, Any]]:
        """Initialise les succès déblocables"""
        return [
            {
                'id': 'first_scenario',
                'name': '🎯 Premier Scénario',
                'description': 'Créez votre premier scénario what-if',
                'points': 50,
                'icon': '🎯'
            },
            {
                'id': 'optimizer',
                'name': '⚡ Optimiseur',
                'description': 'Trouvez un scénario avec 90%+ de réussite',
                'points': 200,
                'icon': '⚡'
            },
            {
                'id': 'risk_master',
                'name': '🛡️ Maître des Risques',
                'description': 'Créez 5 scénarios avec analyse de risque',
                'points': 300,
                'icon': '🛡️'
            },
            {
                'id': 'monte_carlo_expert',
                'name': '🎲 Expert Monte Carlo',
                'description': 'Lancez 10 simulations Monte Carlo',
                'points': 400,
                'icon': '🎲'
            },
            {
                'id': 'perfectionist',
                'name': '⭐ Perfectionniste',
                'description': 'Obtenez un score qualité de 95%+',
                'points': 500,
                'icon': '⭐'
            },
            {
                'id': 'speed_demon',
                'name': '🏎️ Vitesse Éclair',
                'description': 'Réduisez la durée de 40%+ tout en gardant 80%+ de qualité',
                'points': 600,
                'icon': '🏎️'
            }
        ]
    
    def _init_challenges(self) -> List[Dict[str, Any]]:
        """Initialise les défis quotidiens"""
        return [
            {
                'id': 'budget_challenge',
                'name': '💰 Défi Budget',
                'description': 'Réduisez le coût de 20% sans perdre plus de 10% de qualité',
                'target': {'cost_reduction': 0.2, 'quality_loss_max': 0.1},
                'reward': 250,
                'time_limit': '24h'
            },
            {
                'id': 'speed_challenge',
                'name': '⚡ Défi Vitesse',
                'description': 'Livrez en 70% du temps avec 85%+ de succès',
                'target': {'duration_reduction': 0.3, 'success_rate_min': 0.85},
                'reward': 300,
                'time_limit': '24h'
            },
            {
                'id': 'balance_challenge',
                'name': '⚖️ Défi Équilibre',
                'description': 'Trouvez le parfait équilibre temps/coût/qualité',
                'target': {'balance_score_min': 0.9},
                'reward': 400,
                'time_limit': '48h'
            }
        ]
    
    def calculate_scenario_score(self, scenario: ScenarioResult, 
                               baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule le score gamifié d'un scénario"""
        
        # Métriques normalisées (0-1)
        duration_score = self._normalize_score(scenario.duration, baseline.get('duration', 100), inverse=True)
        cost_score = self._normalize_score(scenario.cost, baseline.get('cost', 50000), inverse=True)
        quality_score = scenario.quality_score
        success_score = scenario.success_probability
        risk_score = self._normalize_score(scenario.risk_score, 10, inverse=True)
        
        # Pondération intelligente
        weights = {
            'duration': 0.2,
            'cost': 0.2,
            'quality': 0.25,
            'success': 0.2,
            'risk': 0.15
        }
        
        # Score composite
        composite_score = (
            weights['duration'] * duration_score +
            weights['cost'] * cost_score +
            weights['quality'] * quality_score +
            weights['success'] * success_score +
            weights['risk'] * risk_score
        )
        
        # Conversion en points (0-1000)
        points = int(composite_score * 1000)
        
        # Bonus spéciaux
        bonuses = []
        if scenario.success_probability > 0.9:
            bonuses.append(('Haute Réussite', 100))
        if scenario.quality_score > 0.95:
            bonuses.append(('Qualité Exceptionnelle', 150))
        if scenario.risk_score < 3:
            bonuses.append(('Faible Risque', 75))
            
        total_bonus = sum(bonus[1] for bonus in bonuses)
        final_score = points + total_bonus
        
        # Niveau et rang
        level = self._calculate_level(final_score)
        rank = self._get_rank(final_score)
        
        return {
            'base_points': points,
            'bonuses': bonuses,
            'total_score': final_score,
            'level': level,
            'rank': rank,
            'component_scores': {
                'duration': duration_score,
                'cost': cost_score,
                'quality': quality_score,
                'success': success_score,
                'risk': risk_score
            }
        }
    
    def _normalize_score(self, value: float, reference: float, inverse: bool = False) -> float:
        """Normalise un score entre 0 et 1"""
        if reference == 0:
            return 0.5
            
        ratio = value / reference
        
        if inverse:
            # Plus c'est bas, mieux c'est
            if ratio <= 0.5:
                return 1.0
            elif ratio <= 1.0:
                return 1.0 - (ratio - 0.5) * 1.0
            else:
                return max(0.0, 1.0 - (ratio - 1.0) * 0.5)
        else:
            # Plus c'est haut, mieux c'est
            return min(1.0, ratio)
    
    def _calculate_level(self, score: int) -> int:
        """Calcule le niveau basé sur le score"""
        if score < 200:
            return 1
        elif score < 500:
            return 2
        elif score < 800:
            return 3
        elif score < 1200:
            return 4
        else:
            return 5
    
    def _get_rank(self, score: int) -> str:
        """Détermine le rang basé sur le score"""
        if score < 200:
            return "🥉 Apprenti"
        elif score < 400:
            return "🥈 Stratège"
        elif score < 600:
            return "🥇 Expert"
        elif score < 800:
            return "💎 Maître"
        else:
            return "👑 Grand Maître"
    
    def check_achievements(self, game_state: GameState, scenario_history: List[ScenarioResult]) -> List[Dict[str, Any]]:
        """Vérifie les succès déblocables"""
        new_achievements = []
        
        for achievement in self.achievements:
            if achievement['id'] in (game_state.achievements or []):
                continue
                
            if self._check_achievement_condition(achievement, game_state, scenario_history):
                new_achievements.append(achievement)
                if game_state.achievements is None:
                    game_state.achievements = []
                game_state.achievements.append(achievement['id'])
        
        return new_achievements
    
    def _check_achievement_condition(self, achievement: Dict[str, Any], 
                                   game_state: GameState, scenario_history: List[ScenarioResult]) -> bool:
        """Vérifie si une condition de succès est remplie"""
        
        achievement_id = achievement['id']
        
        if achievement_id == 'first_scenario' and len(scenario_history) >= 1:
            return True
        elif achievement_id == 'optimizer' and any(s.success_probability >= 0.9 for s in scenario_history):
            return True
        elif achievement_id == 'risk_master' and len([s for s in scenario_history if s.risk_score < 5]) >= 5:
            return True
        elif achievement_id == 'perfectionist' and any(s.quality_score >= 0.95 for s in scenario_history):
            return True
        elif achievement_id == 'speed_demon':
            # Vérifier si un scénario réduit la durée de 40%+ avec qualité 80%+
            for scenario in scenario_history:
                if (hasattr(scenario, 'duration_reduction') and 
                    scenario.duration_reduction >= 0.4 and scenario.quality_score >= 0.8):
                    return True
        
        return False

# === OPTIMIZATION ENGINE ===

class OptimizationEngine:
    """Moteur d'optimisation automatique multi-objectifs"""
    
    def __init__(self):
        self.optimization_methods = ['genetic', 'particle_swarm', 'differential_evolution']
        self.objectives = ['minimize_duration', 'minimize_cost', 'maximize_quality', 'minimize_risk']
        
    def auto_optimize(self, project_context: Dict[str, Any], 
                     constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimisation automatique avec algorithme génétique"""
        
        # Définir l'espace de recherche
        bounds = [
            (0.5, 2.0),  # duration_multiplier
            (0.5, 2.5),  # cost_multiplier  
            (0.5, 3.0),  # team_multiplier
            (0.5, 2.0),  # complexity_factor
        ]
        
        # Fonction objectif multi-critères
        def objective_function(params):
            duration_mult, cost_mult, team_mult, complexity_factor = params
            
            scenario_params = {
                'duration_multiplier': duration_mult,
                'cost_multiplier': cost_mult,
                'team_multiplier': team_mult,
                'complexity_factor': complexity_factor,
                'base_duration': project_context.get('base_duration', 100),
                'base_cost': project_context.get('base_cost', 50000)
            }
            
            # Simuler le résultat
            ai_predictor = AIScenarioPredictor()
            result = ai_predictor.predict_scenario_outcome(scenario_params, project_context)
            
            # Vérifier les contraintes
            if not self._check_constraints(result, constraints):
                return 10000  # Pénalité pour contraintes violées
            
            # Score multi-objectif (à minimiser)
            duration_penalty = (result.duration - project_context.get('target_duration', 100)) ** 2
            cost_penalty = (result.cost - project_context.get('target_cost', 50000)) ** 2
            quality_bonus = -result.quality_score * 10000  # Négatif car on maximise
            risk_penalty = result.risk_score ** 2
            
            return duration_penalty + cost_penalty + quality_bonus + risk_penalty
        
        # Optimisation avec differential evolution
        try:
            result = opt.differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42,
                atol=1e-6
            )
            
            if result.success:
                optimal_params = {
                    'duration_multiplier': result.x[0],
                    'cost_multiplier': result.x[1], 
                    'team_multiplier': result.x[2],
                    'complexity_factor': result.x[3],
                    'base_duration': project_context.get('base_duration', 100),
                    'base_cost': project_context.get('base_cost', 50000)
                }
                
                # Générer le scénario optimal
                ai_predictor = AIScenarioPredictor()
                optimal_scenario = ai_predictor.predict_scenario_outcome(optimal_params, project_context)
                optimal_scenario.name = "🎯 Scénario Optimal (Auto-Généré)"
                
                return {
                    'status': 'success',
                    'optimal_scenario': optimal_scenario,
                    'optimization_details': {
                        'iterations': result.nit,
                        'function_evaluations': result.nfev,
                        'final_score': result.fun,
                        'convergence': result.success
                    },
                    'recommendations': self._generate_optimization_recommendations(optimal_scenario)
                }
            else:
                return {
                    'status': 'failed',
                    'message': 'Optimisation échouée - contraintes trop restrictives',
                    'fallback_recommendations': self._generate_fallback_recommendations()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Erreur d\'optimisation: {str(e)}',
                'fallback_recommendations': self._generate_fallback_recommendations()
            }
    
    def _check_constraints(self, scenario: ScenarioResult, constraints: Dict[str, Any]) -> bool:
        """Vérifie si un scénario respecte les contraintes"""
        
        max_duration = constraints.get('max_duration')
        if max_duration and scenario.duration > max_duration:
            return False
            
        max_cost = constraints.get('max_cost')
        if max_cost and scenario.cost > max_cost:
            return False
            
        min_quality = constraints.get('min_quality')
        if min_quality and scenario.quality_score < min_quality:
            return False
            
        max_risk = constraints.get('max_risk')
        if max_risk and scenario.risk_score > max_risk:
            return False
            
        min_success_prob = constraints.get('min_success_probability')
        if min_success_prob and scenario.success_probability < min_success_prob:
            return False
        
        return True
    
    def _generate_optimization_recommendations(self, scenario: ScenarioResult) -> List[str]:
        """Génère des recommandations pour le scénario optimal"""
        recommendations = [
            f"✅ Scénario optimal identifié avec {scenario.success_probability:.1%} de succès",
            f"🎯 Équipe recommandée: {scenario.team_size:.1f} personnes",
            f"📊 Score qualité prévu: {scenario.quality_score:.1%}",
        ]
        
        if scenario.risk_score < 5:
            recommendations.append("🛡️ Profil de risque acceptable pour ce scénario")
        else:
            recommendations.append("⚠️ Surveiller les facteurs de risque identifiés")
            
        return recommendations
    
    def _generate_fallback_recommendations(self) -> List[str]:
        """Recommandations de secours si l'optimisation échoue"""
        return [
            "🔍 Relâchez certaines contraintes pour permettre l'optimisation",
            "📈 Augmentez le budget ou la durée si possible",
            "👥 Considérez l'ajustement de la taille d'équipe",
            "🎯 Priorisez les objectifs les plus critiques"
        ]
    
    def pareto_optimization(self, project_context: Dict[str, Any], 
                           num_solutions: int = 20) -> List[ScenarioResult]:
        """Génère un front de Pareto pour optimisation multi-objectifs"""
        
        solutions = []
        ai_predictor = AIScenarioPredictor()
        
        # Génération de solutions diverses
        for _ in range(num_solutions * 3):  # On génère plus pour filtrer
            # Paramètres aléatoires dans les bornes
            params = {
                'duration_multiplier': random.uniform(0.6, 1.8),
                'cost_multiplier': random.uniform(0.7, 2.0),
                'team_multiplier': random.uniform(0.6, 2.5),
                'complexity_factor': random.uniform(0.7, 1.8),
                'base_duration': project_context.get('base_duration', 100),
                'base_cost': project_context.get('base_cost', 50000)
            }
            
            scenario = ai_predictor.predict_scenario_outcome(params, project_context)
            scenario.name = f"Pareto Solution {len(solutions) + 1}"
            solutions.append(scenario)
        
        # Filtrage de Pareto
        pareto_solutions = self._extract_pareto_front(solutions)
        
        # Limitation au nombre demandé
        return pareto_solutions[:num_solutions]
    
    def _extract_pareto_front(self, solutions: List[ScenarioResult]) -> List[ScenarioResult]:
        """Extrait le front de Pareto d'un ensemble de solutions"""
        
        pareto_solutions = []
        
        for i, solution_i in enumerate(solutions):
            is_pareto_optimal = True
            
            for j, solution_j in enumerate(solutions):
                if i == j:
                    continue
                    
                # Vérifier si solution_j domine solution_i
                # (meilleur sur tous les critères)
                if (solution_j.duration <= solution_i.duration and
                    solution_j.cost <= solution_i.cost and 
                    solution_j.quality_score >= solution_i.quality_score and
                    solution_j.risk_score <= solution_i.risk_score):
                    
                    # Au moins un critère strictement meilleur
                    if (solution_j.duration < solution_i.duration or
                        solution_j.cost < solution_i.cost or
                        solution_j.quality_score > solution_i.quality_score or
                        solution_j.risk_score < solution_i.risk_score):
                        
                        is_pareto_optimal = False
                        break
            
            if is_pareto_optimal:
                pareto_solutions.append(solution_i)
        
        # Tri par score de performance global
        pareto_solutions.sort(key=lambda s: s.success_probability, reverse=True)
        
        return pareto_solutions

# Import du système d'intelligence graphique
try:
    from .whatif_chart_intelligence import (
        IntelligentChartAnalyzer, 
        ChartInsight, 
        ChartAnalysis,
        render_chart_intelligence_panel
    )
    CHART_INTELLIGENCE_AVAILABLE = True
    print("Intelligence graphique avancée chargée!")
except ImportError as e:
    CHART_INTELLIGENCE_AVAILABLE = False
    print(f"Intelligence graphique non disponible: {e}")

def get_current_project_id(plan_data: Dict[str, Any]) -> str:
    """Identifie de manière unique le projet actuel"""
    # Essayer plusieurs méthodes pour identifier le projet
    if 'project_overview' in plan_data:
        # Format PSTB
        return plan_data['project_overview'].get('id', 'unknown_project')
    elif 'projects' in plan_data:
        # Format Portfolio
        purpose = plan_data.get('purpose', 'portfolio')
        return f"portfolio_{purpose.lower().replace(' ', '_')}"
    else:
        # Projet simple - utiliser hash des données clés pour identifier
        import hashlib
        key_data = str(plan_data.get('project_name', '')) + str(len(plan_data.get('tasks', [])))
        return hashlib.md5(key_data.encode()).hexdigest()[:8]

# === MAIN INTERFACE FUNCTIONS ===

def render_advanced_whatif_simulator(plan_data: Dict[str, Any]):
    """Interface principale du simulateur What-If avancé"""
    
    st.markdown("""
    # 🎯 What-If Simulator **RÉVOLUTIONNAIRE**
    
    Système d'aide à la décision nouvelle génération avec **IA**, **Monte Carlo**, **Gamification** et **Optimisation Automatique**.
    """)
    
    # Identifier le projet actuel
    current_project_id = get_current_project_id(plan_data)
    
    # Initialisation des engines
    if 'ai_predictor' not in st.session_state:
        st.session_state.ai_predictor = AIScenarioPredictor()
        st.session_state.monte_carlo = MonteCarloEngine()
        st.session_state.gamification = GamificationEngine()
        st.session_state.optimizer = OptimizationEngine()
        st.session_state.game_state = GameState()
        st.session_state.scenario_histories = {}  # Dictionnaire par projet
        
        # Intelligence graphique
        if CHART_INTELLIGENCE_AVAILABLE:
            st.session_state.chart_analyzer = IntelligentChartAnalyzer()
        else:
            st.session_state.chart_analyzer = None
    
    # S'assurer que scenario_histories existe
    if not hasattr(st.session_state, 'scenario_histories'):
        st.session_state.scenario_histories = {}
        
    # Initialiser l'historique pour le projet actuel si nécessaire
    if current_project_id not in st.session_state.scenario_histories:
        st.session_state.scenario_histories[current_project_id] = []
    
    # Pointer vers l'historique du projet actuel pour compatibilité
    st.session_state.scenario_history = st.session_state.scenario_histories[current_project_id]
    
    # Interface à onglets
    if CHART_INTELLIGENCE_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🤖 IA Predictive", 
            "🎲 Monte Carlo", 
            "🎮 Gamification", 
            "⚡ Auto-Optimisation",
            "🧠 Intelligence Graphique",
            "📊 Dashboard Global"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 IA Predictive", 
            "🎲 Monte Carlo", 
            "🎮 Gamification", 
            "⚡ Auto-Optimisation",
            "📊 Dashboard Global"
        ])
        tab6 = None
    
    # Contexte projet
    project_context = extract_project_context(plan_data)
    
    with tab1:
        render_ai_prediction_tab(project_context)
    
    with tab2:
        render_monte_carlo_tab(project_context)
    
    with tab3:
        render_gamification_tab(project_context)
    
    with tab4:
        render_optimization_tab(project_context)
    
    if tab6:  # Onglet intelligence graphique
        with tab6:
            render_chart_intelligence_tab(project_context)
        
        with tab5:
            render_global_dashboard(project_context)
    else:
        with tab5:
            render_global_dashboard(project_context)

def extract_project_context(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extrait le contexte projet pour les engines"""
    
    overview = plan_data.get('project_overview', {})
    
    return {
        'base_duration': overview.get('total_duration', 100),
        'base_cost': overview.get('total_cost', 50000),
        'base_team_size': len(plan_data.get('resources', [5])),
        'domain': plan_data.get('domain', 'enterprise'),
        'complexity_level': overview.get('complexity_level', 'medium'),
        'target_duration': overview.get('total_duration', 100) * 1.1,
        'target_cost': overview.get('total_cost', 50000) * 1.1
    }

def render_ai_prediction_tab(project_context: Dict[str, Any]):
    """Onglet de prédiction IA"""
    
    st.header("🤖 Intelligence Artificielle Prédictive")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Paramètres de Scénario")
        
        # Contrôles utilisateur
        duration_mult = st.slider("Multiplicateur Durée", 0.5, 2.0, 1.0, 0.1)
        cost_mult = st.slider("Multiplicateur Coût", 0.5, 2.5, 1.0, 0.1)  
        team_mult = st.slider("Multiplicateur Équipe", 0.5, 3.0, 1.0, 0.1)
        complexity = st.slider("Facteur Complexité", 0.5, 2.0, 1.0, 0.1)
        
        # Bouton de prédiction
        if st.button("🔮 Prédire avec IA", type="primary", key="ai_predict_btn"):
            params = {
                'duration_multiplier': duration_mult,
                'cost_multiplier': cost_mult,
                'team_multiplier': team_mult,
                'complexity_factor': complexity,
                'base_duration': project_context['base_duration'],
                'base_cost': project_context['base_cost']
            }
            
            # Prédiction
            with st.spinner("IA en cours d'analyse..."):
                prediction = st.session_state.ai_predictor.predict_scenario_outcome(params, project_context)
                st.session_state.current_prediction = prediction
                st.session_state.scenario_history.append(prediction)
    
    with col2:
        if hasattr(st.session_state, 'current_prediction'):
            prediction = st.session_state.current_prediction
            
            st.subheader("🎯 Prédiction IA")
            
            # Métriques principales
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Durée Prédite", f"{prediction.duration:.1f} jours")
                st.metric("Coût Prédit", f"${prediction.cost:,.0f}")
                
            with col_b:
                st.metric("Score Qualité", f"{prediction.quality_score:.1%}")
                st.metric("Probabilité Succès", f"{prediction.success_probability:.1%}")
            
            # Recommandations IA
            if prediction.recommendations:
                st.subheader("💡 Recommandations IA")
                for rec in prediction.recommendations:
                    st.info(rec)
    
    # Auto-génération de scénarios
    st.subheader("⚡ Auto-Génération IA")
    
    if st.button("🎲 Générer 8 Scénarios Optimaux", key="generate_scenarios_btn"):
        with st.spinner("IA génère des scénarios..."):
            generated_scenarios = st.session_state.ai_predictor.auto_generate_scenarios(project_context, 8)
            st.session_state.generated_scenarios = generated_scenarios
    
    # Affichage des scénarios générés
    if hasattr(st.session_state, 'generated_scenarios'):
        st.subheader("🎯 Scénarios Générés par IA")
        
        scenarios_df = pd.DataFrame([
            {
                'Nom': s.name,
                'Durée': f"{s.duration:.1f}j",
                'Coût': f"${s.cost:,.0f}",
                'Qualité': f"{s.quality_score:.1%}",
                'Succès': f"{s.success_probability:.1%}",
                'Équipe': f"{s.team_size:.1f}"
            }
            for s in st.session_state.generated_scenarios
        ])
        
        st.dataframe(scenarios_df, use_container_width=True)

def render_monte_carlo_tab(project_context: Dict[str, Any]):
    """Onglet de simulation Monte Carlo - Moteur d'analyse principal"""
    
    st.header("🎲 Simulation Monte Carlo Ultra-Performante")
    st.markdown("*Moteur principal d'analyse de scénarios - Exploration approfondie avec milliers de simulations*")
    
    # Contrôles de simulation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_simulations = st.selectbox(
            "Nombre de Simulations",
            [1000, 10000, 50000, 100000],
            index=2,
            help="Plus de simulations = plus de précision"
        )
    
    with col2:
        duration_mult = st.slider("Durée Multiplier (MC)", 0.6, 2.0, 1.0, 0.1)
        cost_mult = st.slider("Coût Multiplier (MC)", 0.6, 2.0, 1.0, 0.1)
    
    with col3:
        if st.button("🚀 Lancer Simulation MC", type="primary", key="monte_carlo_btn"):
            st.session_state.monte_carlo.num_simulations = num_simulations
            
            scenario_params = {
                'duration_multiplier': duration_mult,
                'cost_multiplier': cost_mult
            }
            
            # Sauvegarder les paramètres pour utilisation ultérieure
            st.session_state.scenario_params = scenario_params
            
            with st.spinner(f"Simulation de {num_simulations:,} scénarios..."):
                mc_results = st.session_state.monte_carlo.run_simulation(scenario_params, project_context)
                st.session_state.mc_results = mc_results
    
    # Affichage des résultats Monte Carlo
    if hasattr(st.session_state, 'mc_results'):
        results = st.session_state.mc_results
        
        st.subheader("📊 Résultats Simulation")
        
        # Métriques clés
        col_a, col_b, col_c, col_d = st.columns(4)
        
        duration_stats = results['duration_stats']
        cost_stats = results['cost_stats']
        
        with col_a:
            st.metric("Durée Médiane", f"{duration_stats['median']:.1f} jours")
            st.metric("IC 90% Durée", f"{duration_stats['p10']:.0f}-{duration_stats['p90']:.0f}j")
            
        with col_b:
            st.metric("Coût Médian", f"${cost_stats['median']:,.0f}")
            st.metric("IC 90% Coût", f"${cost_stats['p10']:,.0f}-${cost_stats['p90']:,.0f}")
        
        with col_c:
            st.metric("Taux de Succès", f"{results['success_rate']:.1%}")
            st.metric("Risque Global", f"{results['risk_metrics']['overall_risk_score']:.1%}")
            
        with col_d:
            st.metric("VaR 95% Durée", f"{results['risk_metrics']['duration_var_95']:.1f}j")
            st.metric("VaR 95% Coût", f"${results['risk_metrics']['cost_var_95']:,.0f}")
        
        # Graphiques intelligents avec analyse automatique
        if hasattr(st.session_state, 'chart_analyzer') and st.session_state.chart_analyzer and CHART_INTELLIGENCE_AVAILABLE:
            st.subheader("🧠 Analyse Monte Carlo Intelligente")
            
            # Récupération des paramètres sauvegardés ou défaut
            scenario_params = getattr(st.session_state, 'scenario_params', {
                'duration_multiplier': 1.0,
                'cost_multiplier': 1.0
            })
            
            # Création du graphique intelligent
            intelligent_fig, chart_analysis = st.session_state.chart_analyzer.create_intelligent_monte_carlo_chart(
                results, scenario_params
            )
            
            # Affichage du graphique
            st.plotly_chart(intelligent_fig, use_container_width=True)
            
            # Panneau d'intelligence automatique
            render_chart_intelligence_panel(chart_analysis)
            
        else:
            # Fallback vers graphiques standards
            st.subheader("📈 Distributions Monte Carlo")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Histogramme durée
                fig_duration = px.histogram(
                    x=results['simulation_data']['durations'][:1000],
                    nbins=30,
                    title="Distribution Durée (jours)",
                    labels={'x': 'Durée', 'y': 'Fréquence'}
                )
                fig_duration.add_vline(x=duration_stats['median'], line_dash="dash", line_color="red", annotation_text="Médiane")
                st.plotly_chart(fig_duration, use_container_width=True)
            
            with col_viz2:
                # Histogramme coût  
                fig_cost = px.histogram(
                    x=results['simulation_data']['costs'][:1000],
                    nbins=30,
                    title="Distribution Coût ($)",
                    labels={'x': 'Coût', 'y': 'Fréquence'}
                )
                fig_cost.add_vline(x=cost_stats['median'], line_dash="dash", line_color="red", annotation_text="Médiane")
                st.plotly_chart(fig_cost, use_container_width=True)
            
            # Analyse de corrélation
            correlation_df = pd.DataFrame({
                'Durée': results['simulation_data']['durations'][:1000],
                'Coût': results['simulation_data']['costs'][:1000],
                'Qualité': results['simulation_data']['qualities'][:1000]
            })
            
            fig_scatter = px.scatter_matrix(
                correlation_df,
                title="Matrice de Corrélation Monte Carlo"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

def render_gamification_tab(project_context: Dict[str, Any]):
    """Onglet de gamification"""
    
    st.header("🎮 Mode Gamifié - Devenez un Maître de la Planification!")
    
    game_state = st.session_state.game_state
    
    # Panneau de score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏆 Score Total", f"{game_state.player_score:,}")
    with col2:
        st.metric("📊 Niveau", f"{game_state.level}")
    with col3:
        st.metric("🎯 Défis Réussis", f"{game_state.challenges_completed}")
    with col4:
        achievements_count = len(game_state.achievements or [])
        st.metric("🏅 Succès", f"{achievements_count}")
    
    # Zone de jeu
    st.subheader("🎯 Défi Actuel")
    
    # Sélection de défi
    challenges = st.session_state.gamification.challenges
    current_challenge = st.selectbox(
        "Choisissez votre défi",
        [c['name'] for c in challenges],
        format_func=lambda x: x
    )
    
    selected_challenge = next(c for c in challenges if c['name'] == current_challenge)
    
    st.info(f"**{selected_challenge['description']}**")
    st.markdown(f"🏆 Récompense: {selected_challenge['reward']} points")
    st.markdown(f"⏰ Temps limite: {selected_challenge['time_limit']}")
    
    # Contrôles de jeu
    st.subheader("🎮 Votre Stratégie")
    
    col_game1, col_game2 = st.columns(2)
    
    with col_game1:
        game_duration_mult = st.slider("⚡ Vitesse d'Exécution", 0.5, 2.0, 1.0, 0.1, key="game_duration")
        game_cost_mult = st.slider("💰 Budget Alloué", 0.5, 2.0, 1.0, 0.1, key="game_cost")
        
    with col_game2:
        game_team_mult = st.slider("👥 Taille Équipe", 0.5, 2.5, 1.0, 0.1, key="game_team")
        game_quality_focus = st.slider("⭐ Focus Qualité", 0.5, 1.5, 1.0, 0.1, key="game_quality")
    
    # Bouton de tentative
    if st.button("🚀 Tenter le Défi!", type="primary", key="challenge_btn"):
        # Création du scénario de jeu
        game_params = {
            'duration_multiplier': game_duration_mult,
            'cost_multiplier': game_cost_mult,
            'team_multiplier': game_team_mult,
            'complexity_factor': 1.0 / game_quality_focus,  # Inverse pour qualité
            'base_duration': project_context['base_duration'],
            'base_cost': project_context['base_cost']
        }
        
        # Prédiction du résultat
        game_result = st.session_state.ai_predictor.predict_scenario_outcome(game_params, project_context)
        
        # Calcul du score
        game_score_data = st.session_state.gamification.calculate_scenario_score(
            game_result, 
            {'duration': project_context['base_duration'], 'cost': project_context['base_cost']}
        )
        
        # Affichage des résultats
        st.subheader("🎯 Résultat de votre Tentative")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("📈 Score Obtenu", f"{game_score_data['total_score']:,}")
            st.metric("🎖️ Rang", game_score_data['rank'])
            
        with col_result2:
            st.metric("✅ Probabilité Succès", f"{game_result.success_probability:.1%}")
            st.metric("⭐ Score Qualité", f"{game_result.quality_score:.1%}")
            
        with col_result3:
            st.metric("⏱️ Durée Finale", f"{game_result.duration:.1f}j")
            st.metric("💸 Coût Final", f"${game_result.cost:,.0f}")
        
        # Bonus obtenus
        if game_score_data['bonuses']:
            st.subheader("🎁 Bonus Obtenus!")
            for bonus_name, bonus_points in game_score_data['bonuses']:
                st.success(f"🎉 {bonus_name}: +{bonus_points} points")
        
        # Mise à jour du score
        st.session_state.game_state.player_score += game_score_data['total_score']
        st.session_state.game_state.level = game_score_data['level']
        
        # Vérification des succès
        new_achievements = st.session_state.gamification.check_achievements(
            st.session_state.game_state,
            st.session_state.scenario_history
        )
        
        if new_achievements:
            st.subheader("🏆 Nouveau Succès Débloqué!")
            for achievement in new_achievements:
                st.success(f"🎉 {achievement['icon']} **{achievement['name']}** - {achievement['description']} (+{achievement['points']} points)")
    
    # Tableau des succès
    st.subheader("🏅 Vos Succès")
    
    achievements_df = pd.DataFrame([
        {
            'Succès': f"{a['icon']} {a['name']}",
            'Description': a['description'],
            'Points': a['points'],
            'Statut': '✅ Débloqué' if a['id'] in (game_state.achievements or []) else '🔒 Verrouillé'
        }
        for a in st.session_state.gamification.achievements
    ])
    
    st.dataframe(achievements_df, use_container_width=True)

def render_optimization_tab(project_context: Dict[str, Any]):
    """Onglet d'optimisation automatique"""
    
    st.header("⚡ Optimisation Automatique Multi-Objectifs")
    
    # Définition des contraintes
    st.subheader("🎯 Définissez vos Contraintes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Limites Temporelles et Budgétaires**")
        
        max_duration = st.number_input(
            "Durée Maximale (jours)",
            min_value=10,
            max_value=500,
            value=int(project_context['base_duration'] * 1.5),
            help="Durée maximale acceptable"
        )
        
        max_cost = st.number_input(
            "Budget Maximum ($)",
            min_value=5000,
            max_value=1000000,
            value=int(project_context['base_cost'] * 1.8),
            help="Budget maximum acceptable"
        )
    
    with col2:
        st.markdown("**Exigences de Qualité et Risque**")
        
        min_quality = st.slider(
            "Qualité Minimale",
            0.3, 1.0, 0.7, 0.05,
            format="%.2f",
            help="Score de qualité minimum requis"
        )
        
        max_risk = st.slider(
            "Risque Maximum",
            1.0, 10.0, 7.0, 0.5,
            help="Score de risque maximum acceptable"
        )
        
        min_success_prob = st.slider(
            "Probabilité Succès Minimale",
            0.3, 0.95, 0.75, 0.05,
            format="%.2f",
            help="Probabilité de succès minimum"
        )
    
    # Lancement de l'optimisation
    if st.button("🎯 Lancer Optimisation Automatique", type="primary", key="optimization_btn"):
        constraints = {
            'max_duration': max_duration,
            'max_cost': max_cost,
            'min_quality': min_quality,
            'max_risk': max_risk,
            'min_success_probability': min_success_prob
        }
        
        with st.spinner("🤖 Optimisation en cours... (algorithme génétique)"):
            optimization_result = st.session_state.optimizer.auto_optimize(project_context, constraints)
            st.session_state.optimization_result = optimization_result
    
    # Affichage des résultats d'optimisation
    if hasattr(st.session_state, 'optimization_result'):
        result = st.session_state.optimization_result
        
        if result['status'] == 'success':
            optimal_scenario = result['optimal_scenario']
            
            st.subheader("🎯 Scénario Optimal Trouvé!")
            st.success(f"✅ Optimisation réussie en {result['optimization_details']['iterations']} itérations")
            
            # Métriques du scénario optimal
            col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
            
            with col_opt1:
                st.metric("⏱️ Durée Optimale", f"{optimal_scenario.duration:.1f} jours")
                
            with col_opt2:
                st.metric("💰 Coût Optimal", f"${optimal_scenario.cost:,.0f}")
                
            with col_opt3:
                st.metric("⭐ Score Qualité", f"{optimal_scenario.quality_score:.1%}")
                
            with col_opt4:
                st.metric("🎯 Probabilité Succès", f"{optimal_scenario.success_probability:.1%}")
            
            # Détails d'optimisation
            st.subheader("📊 Détails de l'Optimisation")
            
            opt_details = result['optimization_details']
            
            details_df = pd.DataFrame([
                {'Métrique': 'Itérations', 'Valeur': opt_details['iterations']},
                {'Métrique': 'Évaluations de Fonction', 'Valeur': opt_details['function_evaluations']},
                {'Métrique': 'Score Final', 'Valeur': f"{opt_details['final_score']:.2f}"},
                {'Métrique': 'Convergence', 'Valeur': '✅ Oui' if opt_details['convergence'] else '❌ Non'}
            ])
            
            st.dataframe(details_df, use_container_width=True)
            
            # Recommandations
            if result.get('recommendations'):
                st.subheader("💡 Recommandations d'Optimisation")
                for rec in result['recommendations']:
                    st.info(rec)
                    
        else:
            st.error(f"❌ Optimisation échouée: {result.get('message', 'Erreur inconnue')}")
            
            if result.get('fallback_recommendations'):
                st.subheader("🔧 Suggestions d'Amélioration")
                for rec in result['fallback_recommendations']:
                    st.warning(rec)
    
    # Front de Pareto
    st.subheader("📈 Analyse Multi-Objectifs (Pareto)")
    
    if st.button("🎯 Générer Front de Pareto", key="pareto_front_btn"):
        with st.spinner("Calcul du front de Pareto..."):
            pareto_solutions = st.session_state.optimizer.pareto_optimization(project_context, 15)
            st.session_state.pareto_solutions = pareto_solutions
    
    if hasattr(st.session_state, 'pareto_solutions'):
        solutions = st.session_state.pareto_solutions
        
        st.subheader("🎯 Solutions Pareto-Optimales")
        
        # Graphique Pareto intelligent
        if hasattr(st.session_state, 'chart_analyzer') and st.session_state.chart_analyzer and CHART_INTELLIGENCE_AVAILABLE:
            st.subheader("🧠 Analyse Pareto 3D Intelligente")
            
            # Conversion des solutions en format compatible
            scenario_data = []
            for s in solutions:
                scenario_data.append({
                    'name': s.name,
                    'duration': s.duration,
                    'cost': s.cost,
                    'quality_score': s.quality_score,
                    'success_probability': s.success_probability,
                    'risk_score': s.risk_score
                })
            
            # Graphique intelligent avec analyse
            pareto_fig, pareto_analysis = st.session_state.chart_analyzer.create_intelligent_pareto_chart(scenario_data)
            st.plotly_chart(pareto_fig, use_container_width=True)
            
            # Panneau d'intelligence Pareto
            render_chart_intelligence_panel(pareto_analysis)
            
        else:
            # Fallback graphique standard
            pareto_df = pd.DataFrame([
                {
                    'Nom': s.name[:20] + "...",  # Truncate long names
                    'Durée': s.duration,
                    'Coût': s.cost,
                    'Qualité': s.quality_score,
                    'Succès': s.success_probability,
                    'Risque': s.risk_score
                }
                for s in solutions
            ])
            
            # Scatter plot interactif
            fig_pareto = px.scatter(
                pareto_df,
                x='Durée',
                y='Coût', 
                size='Succès',
                color='Qualité',
                hover_data=['Risque'],
                title="Front de Pareto: Durée vs Coût (Taille=Succès, Couleur=Qualité)"
            )
            
            st.plotly_chart(fig_pareto, use_container_width=True)
            
            # Tableau des solutions
            st.dataframe(pareto_df, use_container_width=True)

def render_global_dashboard(project_context: Dict[str, Any]):
    """Dashboard global avec toutes les analyses"""
    
    st.header("📊 Dashboard Global - Vue d'Ensemble")
    
    # Historique des scénarios
    st.subheader("📈 Évolution de vos Scénarios")
    
    if st.session_state.scenario_history:
        history_df = pd.DataFrame([
            {
                'Nom': s.name[:30],
                'Durée (jours)': s.duration,
                'Coût (k€)': s.cost / 1000 if s.cost > 1000 else s.cost,  # Convertir en k€ si > 1000
                'Qualité (%)': s.quality_score * 100 if s.quality_score <= 1 else s.quality_score,  # Convertir en %
                'Succès (%)': s.success_probability * 100 if s.success_probability <= 1 else s.success_probability,  # Convertir en %
                'Risque': s.risk_score,
                'IA': '🤖' if s.ai_generated else '👤'
            }
            for s in st.session_state.scenario_history
        ])
        
        # Graphique temporel avec données réelles normalisées
        fig_timeline = px.line(
            history_df.reset_index(),
            x='index',
            y=['Durée (jours)', 'Coût (k€)', 'Qualité (%)', 'Succès (%)'],
            title="Évolution des Métriques par Scénario",
            labels={'index': 'Scénario', 'value': 'Valeur', 'variable': 'Métriques'}
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Tableau récapitulatif
        st.dataframe(history_df, use_container_width=True)
    else:
        # Affichage de données d'exemple quand aucun scénario n'existe
        st.info("💡 Créez votre premier scénario pour voir l'évolution des métriques ici")
        
        # Données d'exemple pour démonstration avec normalisation
        sample_df = pd.DataFrame([
            {
                'Scénario': 'Optimiste', 
                'Durée (jours)': 80, 
                'Coût (k€)': 45, 
                'Qualité (%)': 85, 
                'Succès (%)': 75
            },
            {
                'Scénario': 'Réaliste', 
                'Durée (jours)': 100, 
                'Coût (k€)': 50, 
                'Qualité (%)': 80, 
                'Succès (%)': 70
            },
            {
                'Scénario': 'Pessimiste', 
                'Durée (jours)': 120, 
                'Coût (k€)': 60, 
                'Qualité (%)': 75, 
                'Succès (%)': 60
            }
        ])
        
        fig_sample = px.line(
            sample_df,
            x='Scénario',
            y=['Durée (jours)', 'Coût (k€)', 'Qualité (%)', 'Succès (%)'],
            title="Exemple d'Évolution des Métriques par Scénario",
            labels={'value': 'Valeur', 'variable': 'Métriques'}
        )
        
        st.plotly_chart(fig_sample, use_container_width=True)
        
        st.markdown("*Cet exemple montre comment vos scénarios apparaîtront une fois créés*")
    
    # Statistiques globales
    st.subheader("🎯 Statistiques de Session")
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("Scénarios Créés", len(st.session_state.scenario_history))
    
    with col_stats2:
        ai_scenarios = len([s for s in st.session_state.scenario_history if s.ai_generated])
        st.metric("Scénarios IA", ai_scenarios)
    
    with col_stats3:
        if st.session_state.scenario_history:
            avg_success = np.mean([s.success_probability for s in st.session_state.scenario_history])
            st.metric("Succès Moyen", f"{avg_success:.1%}")
        else:
            st.metric("Succès Moyen", "N/A")
    
    with col_stats4:
        st.metric("Score de Jeu", f"{st.session_state.game_state.player_score:,}")
    
    # Export des données
    st.subheader("💾 Export des Données")
    
    if st.button("📄 Exporter Rapport Complet", key="export_whatif_report"):
        # Génération du rapport
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'project_context': project_context,
            'scenario_history': [
                {
                    'name': s.name,
                    'duration': s.duration,
                    'cost': s.cost,
                    'quality_score': s.quality_score,
                    'success_probability': s.success_probability,
                    'risk_score': s.risk_score,
                    'ai_generated': s.ai_generated,
                    'recommendations': s.recommendations
                }
                for s in st.session_state.scenario_history
            ],
            'game_statistics': {
                'player_score': st.session_state.game_state.player_score,
                'level': st.session_state.game_state.level,
                'achievements': st.session_state.game_state.achievements or []
            }
        }
        
        # Création du fichier JSON
        report_json = json.dumps(report_data, indent=2, default=str)
        
        st.download_button(
            label="⬇️ Télécharger Rapport JSON",
            data=report_json,
            file_name=f"whatif_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Rapport généré avec succès!")

def render_chart_intelligence_tab(project_context: Dict[str, Any]):
    """Onglet dédié à l'intelligence graphique avancée"""
    
    st.header("🧠 Intelligence Graphique Révolutionnaire")
    st.markdown("Analyse automatique et interprétation intelligente de tous vos graphiques")
    
    if not CHART_INTELLIGENCE_AVAILABLE or not hasattr(st.session_state, 'chart_analyzer') or not st.session_state.chart_analyzer:
        st.error("❌ Module d'intelligence graphique non disponible")
        return
    
    # Menu de sélection du type d'analyse
    analysis_type = st.selectbox(
        "🎯 Type d'Analyse Intelligente",
        [
            "Monte Carlo Avancé",
            "Analyse Pareto 3D", 
            "Heatmap de Risques",
            "Analyse de Sensibilité",
            "Comparaison Multi-Scénarios"
        ]
    )
    
    if analysis_type == "Monte Carlo Avancé":
        st.subheader("🎲 Démonstration Monte Carlo Intelligent")
        st.info("💡 **Note**: Ceci est une démonstration avancée. Le moteur Monte Carlo principal se trouve dans l'onglet **🎲 Monte Carlo** pour vos analyses interactives.")
        
        # Paramètres de démonstration
        col1, col2 = st.columns(2)
        
        with col1:
            demo_duration_mult = st.slider("Multiplicateur Durée (Demo)", 0.6, 2.0, 1.0, 0.1)
            demo_cost_mult = st.slider("Multiplicateur Coût (Demo)", 0.6, 2.0, 1.0, 0.1)
            
        with col2:
            demo_simulations = st.selectbox("Nombre de Simulations", [1000, 5000, 10000], index=1)
            
        if st.button("🚀 Générer Analyse Monte Carlo Intelligente", type="primary", key="intelligent_mc_btn"):
            # Simulation des données
            simulation_data = {
                'durations': list(np.random.lognormal(0, 0.3, demo_simulations) * project_context['base_duration'] * demo_duration_mult),
                'costs': list(np.random.lognormal(0, 0.4, demo_simulations) * project_context['base_cost'] * demo_cost_mult),
                'qualities': list(np.random.beta(2, 1, demo_simulations))
            }
            
            scenario_params = {
                'duration_multiplier': demo_duration_mult,
                'cost_multiplier': demo_cost_mult
            }
            
            # Création du graphique intelligent
            with st.spinner("🧠 Analyse intelligente en cours..."):
                intelligent_fig, analysis = st.session_state.chart_analyzer.create_intelligent_monte_carlo_chart(
                    {'simulation_data': simulation_data}, scenario_params
                )
                
                # Affichage
                st.plotly_chart(intelligent_fig, use_container_width=True)
                
                # Panneau d'analyse
                render_chart_intelligence_panel(analysis)
    
    elif analysis_type == "Analyse Pareto 3D":
        st.subheader("🎯 Démonstration Pareto 3D Intelligent")
        
        # Génération de scénarios de démonstration
        demo_scenarios = []
        for i in range(8):
            scenario = {
                'name': f'Scénario Demo {i+1}',
                'duration': np.random.uniform(50, 150),
                'cost': np.random.uniform(30000, 80000),
                'quality_score': np.random.uniform(0.6, 0.95),
                'success_probability': np.random.uniform(0.5, 0.9),
                'risk_score': np.random.uniform(2, 8)
            }
            demo_scenarios.append(scenario)
        
        if st.button("🎯 Analyser le Front de Pareto", type="primary", key="analyze_pareto_btn"):
            with st.spinner("🧠 Analyse Pareto intelligente..."):
                pareto_fig, pareto_analysis = st.session_state.chart_analyzer.create_intelligent_pareto_chart(demo_scenarios)
                
                st.plotly_chart(pareto_fig, use_container_width=True)
                render_chart_intelligence_panel(pareto_analysis)
    
    elif analysis_type == "Heatmap de Risques":
        st.subheader("🚨 Analyse de Risques Intelligente")
        
        risk_factors = ['Timeline', 'Budget', 'Quality', 'Team', 'Complexity', 'External']
        
        # Génération de scénarios pour heatmap
        risk_scenarios = []
        for i in range(6):
            scenario = {
                'name': f'Stratégie {i+1}',
                'duration': np.random.uniform(60, 140),
                'cost': np.random.uniform(25000, 75000),
                'team_size': np.random.randint(3, 10),
                'complexity_factor': np.random.uniform(0.8, 1.8)
            }
            risk_scenarios.append(scenario)
        
        if st.button("🚨 Analyser Matrice de Risques", type="primary", key="risk_matrix_btn"):
            with st.spinner("🧠 Analyse de risques intelligente..."):
                risk_fig, risk_analysis = st.session_state.chart_analyzer.create_intelligent_risk_heatmap(
                    risk_scenarios, risk_factors
                )
                
                st.plotly_chart(risk_fig, use_container_width=True)
                render_chart_intelligence_panel(risk_analysis)
    
    elif analysis_type == "Analyse de Sensibilité":
        st.subheader("📊 Tornado Chart Intelligent")
        
        sensitivity_params = ['Durée Base', 'Coût Base', 'Taille Équipe', 'Complexité', 'Qualité Requise']
        base_scenario = {
            'duration': project_context['base_duration'],
            'cost': project_context['base_cost'],
            'team_size': project_context['base_team_size'],
            'complexity': 1.0,
            'quality': 0.8
        }
        
        if st.button("📊 Analyser Sensibilité", type="primary", key="sensitivity_btn"):
            with st.spinner("🧠 Analyse de sensibilité intelligente..."):
                sensitivity_fig, sensitivity_analysis = st.session_state.chart_analyzer.create_intelligent_sensitivity_chart(
                    base_scenario, sensitivity_params
                )
                
                st.plotly_chart(sensitivity_fig, use_container_width=True)
                render_chart_intelligence_panel(sensitivity_analysis)
    
    elif analysis_type == "Comparaison Multi-Scénarios":
        st.subheader("🔀 Comparaison Intelligente de Scénarios")
        st.info("🚀 Cette fonctionnalité utilise l'historique de vos scénarios créés dans les autres onglets.")
        
        if st.session_state.scenario_history:
            st.success(f"✅ {len(st.session_state.scenario_history)} scénarios disponibles pour analyse")
            
            # Conversion en format compatible
            scenario_data = []
            for s in st.session_state.scenario_history:
                scenario_data.append({
                    'name': s.name,
                    'duration': s.duration,
                    'cost': s.cost,
                    'quality_score': s.quality_score,
                    'success_probability': s.success_probability,
                    'risk_score': s.risk_score
                })
            
            if st.button("🔍 Analyser Historique des Scénarios", key="analyze_history_btn"):
                # Analyse comparative intelligente
                comparison_fig, comparison_analysis = st.session_state.chart_analyzer.create_intelligent_pareto_chart(scenario_data)
                
                st.plotly_chart(comparison_fig, use_container_width=True)
                render_chart_intelligence_panel(comparison_analysis)
        else:
            st.warning("📝 Créez d'abord quelques scénarios dans les onglets IA ou Monte Carlo pour activer cette analyse.")
    
    # Guide d'utilisation
    with st.expander("📖 Guide de l'Intelligence Graphique"):
        st.markdown("""
        ## 🎯 **Fonctionnalités d'Intelligence**
        
        ### **🧠 Analyse Automatique**
        - **Détection de patterns** : Asymétrie, variance élevée, corrélations
        - **Identification de risques** : Zones critiques, scénarios dominés
        - **Recommandations intelligentes** : Actions concrètes basées sur l'analyse
        
        ### **📊 Types d'Insights**
        - **🔴 CRITIQUE** : Risques majeurs nécessitant action immédiate
        - **🟠 ÉLEVÉ** : Points d'attention importants
        - **🟡 MOYEN** : Surveillances recommandées
        - **🟢 FAIBLE** : Informations complémentaires
        
        ### **🎯 Interprétation Business**
        - **Traduction technique→business** des résultats statistiques
        - **Plans d'action concrets** avec priorités
        - **Export des analyses** pour reporting exécutif
        """)

# Export des fonctions principales
__all__ = [
    'render_advanced_whatif_simulator',
    'AIScenarioPredictor',
    'MonteCarloEngine', 
    'GamificationEngine',
    'OptimizationEngine'
]