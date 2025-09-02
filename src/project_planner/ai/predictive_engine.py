"""
🧠 Système d'IA Prédictive Avancée - PlannerIA
Intelligence artificielle révolutionnaire pour prédiction et optimisation de projets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from enum import Enum
import asyncio


class ProjectDomain(Enum):
    ECOMMERCE = "ecommerce"
    FINTECH = "fintech" 
    HEALTHCARE = "healthcare"
    ENTERPRISE = "enterprise"
    MOBILE_APP = "mobile_app"
    WEB_APP = "web_app"
    AI_ML = "ai_ml"
    BLOCKCHAIN = "blockchain"


class PredictionConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ProjectPattern:
    """Pattern de projet identifié par l'IA"""
    pattern_id: str
    domain: ProjectDomain
    team_size_range: Tuple[int, int]
    duration_multiplier: float
    cost_multiplier: float
    success_rate: float
    common_risks: List[str]
    optimal_practices: List[str]
    confidence: PredictionConfidence


@dataclass
class PredictionResult:
    """Résultat de prédiction IA"""
    predicted_duration: float
    predicted_cost: float
    success_probability: float
    confidence_interval: Tuple[float, float]
    risk_factors: Dict[str, float]
    recommendations: List[str]
    similar_projects: List[Dict[str, Any]]
    optimization_suggestions: List[str]


class AIProjectPredictor:
    """Système d'IA prédictive pour projets logiciels"""
    
    def __init__(self):
        self.patterns_db = self._initialize_patterns_database()
        self.learning_models = self._initialize_ml_models()
        self.scaler = StandardScaler()
        self.clustering_model = KMeans(n_clusters=8, random_state=42)
        self.prediction_cache = {}
        
    def _initialize_patterns_database(self) -> Dict[str, List[ProjectPattern]]:
        """Initialise la base de données de patterns"""
        patterns = {
            ProjectDomain.ECOMMERCE.value: [
                ProjectPattern(
                    pattern_id="ecom_standard",
                    domain=ProjectDomain.ECOMMERCE,
                    team_size_range=(3, 8),
                    duration_multiplier=1.2,
                    cost_multiplier=1.1,
                    success_rate=0.78,
                    common_risks=["Payment integration", "Scalability", "Security"],
                    optimal_practices=["Microservices", "Progressive deployment", "A/B testing"],
                    confidence=PredictionConfidence.HIGH
                ),
                ProjectPattern(
                    pattern_id="ecom_marketplace",
                    domain=ProjectDomain.ECOMMERCE,
                    team_size_range=(8, 15),
                    duration_multiplier=1.8,
                    cost_multiplier=1.6,
                    success_rate=0.65,
                    common_risks=["Multi-vendor complexity", "Performance", "Data consistency"],
                    optimal_practices=["Event-driven architecture", "CQRS", "Distributed caching"],
                    confidence=PredictionConfidence.MEDIUM
                )
            ],
            ProjectDomain.FINTECH.value: [
                ProjectPattern(
                    pattern_id="fintech_core",
                    domain=ProjectDomain.FINTECH,
                    team_size_range=(5, 12),
                    duration_multiplier=2.0,
                    cost_multiplier=1.8,
                    success_rate=0.62,
                    common_risks=["Regulatory compliance", "Security audits", "Integration complexity"],
                    optimal_practices=["Zero-trust security", "Audit trails", "Immutable data"],
                    confidence=PredictionConfidence.HIGH
                )
            ],
            ProjectDomain.AI_ML.value: [
                ProjectPattern(
                    pattern_id="ml_standard",
                    domain=ProjectDomain.AI_ML,
                    team_size_range=(3, 10),
                    duration_multiplier=1.6,
                    cost_multiplier=1.4,
                    success_rate=0.58,
                    common_risks=["Data quality", "Model drift", "Computational resources"],
                    optimal_practices=["MLOps pipeline", "Data versioning", "Continuous training"],
                    confidence=PredictionConfidence.MEDIUM
                )
            ]
        }
        
        # Ajouter des patterns pour tous les domaines
        for domain in ProjectDomain:
            if domain.value not in patterns:
                patterns[domain.value] = [self._generate_default_pattern(domain)]
                
        return patterns
    
    def _generate_default_pattern(self, domain: ProjectDomain) -> ProjectPattern:
        """Génère un pattern par défaut pour un domaine"""
        base_multipliers = {
            ProjectDomain.ENTERPRISE: (1.5, 1.3, 0.72),
            ProjectDomain.MOBILE_APP: (1.1, 1.0, 0.82),
            ProjectDomain.WEB_APP: (1.0, 0.9, 0.85),
            ProjectDomain.HEALTHCARE: (2.2, 1.9, 0.55),
            ProjectDomain.BLOCKCHAIN: (1.8, 1.6, 0.48)
        }
        
        duration_mult, cost_mult, success_rate = base_multipliers.get(domain, (1.0, 1.0, 0.75))
        
        return ProjectPattern(
            pattern_id=f"{domain.value}_default",
            domain=domain,
            team_size_range=(3, 8),
            duration_multiplier=duration_mult,
            cost_multiplier=cost_mult,
            success_rate=success_rate,
            common_risks=["Technical complexity", "Requirements changes"],
            optimal_practices=["Agile methodology", "Code reviews"],
            confidence=PredictionConfidence.MEDIUM
        )
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialise les modèles ML"""
        return {
            'duration_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'cost_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'success_classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'risk_predictor': RandomForestRegressor(n_estimators=50, random_state=42)
        }
    
    async def predict_project_outcome(self, project_data: Dict[str, Any]) -> PredictionResult:
        """Prédit le résultat d'un projet avec IA avancée"""
        
        # Extraction des caractéristiques du projet
        features = self._extract_project_features(project_data)
        
        # Identification du pattern le plus proche
        best_pattern = self._find_best_matching_pattern(features)
        
        # Prédictions ML
        ml_predictions = await self._run_ml_predictions(features)
        
        # Combinaison pattern + ML
        final_prediction = self._combine_predictions(best_pattern, ml_predictions, features)
        
        # Génération des recommandations
        recommendations = self._generate_ai_recommendations(features, best_pattern, final_prediction)
        
        # Recherche de projets similaires
        similar_projects = self._find_similar_projects(features)
        
        return PredictionResult(
            predicted_duration=final_prediction['duration'],
            predicted_cost=final_prediction['cost'],
            success_probability=final_prediction['success_probability'],
            confidence_interval=final_prediction['confidence_interval'],
            risk_factors=final_prediction['risk_factors'],
            recommendations=recommendations,
            similar_projects=similar_projects,
            optimization_suggestions=self._generate_optimization_suggestions(features, final_prediction)
        )
    
    def _extract_project_features(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les caractéristiques importantes du projet"""
        # Safe extraction with type checking
        def safe_get(key, default, expected_type=None):
            value = project_data.get(key, default)
            if expected_type and not isinstance(value, expected_type):
                return default
            return value
        
        features = {
            'domain': safe_get('domain', 'web_app', str),
            'team_size': safe_get('team_size', 5, (int, float)),
            'complexity': safe_get('complexity', 'medium', str),
            'budget': safe_get('budget', 100000, (int, float)),
            'deadline_pressure': safe_get('deadline_pressure', 0.5, (int, float)),
            'team_experience': safe_get('team_experience', 'mixed', str),
            'technology_stack': safe_get('technology_stack', [], list),
            'integration_points': len(safe_get('integrations', [], list)),
            'requirements_clarity': safe_get('requirements_clarity', 0.7, (int, float)),
            'stakeholder_involvement': safe_get('stakeholder_involvement', 0.6, (int, float))
        }
        
        # Calcul de scores dérivés
        features['complexity_score'] = self._calculate_complexity_score(features)
        features['risk_score'] = self._calculate_initial_risk_score(features)
        features['experience_factor'] = self._calculate_experience_factor(features)
        
        return features
    
    def _calculate_complexity_score(self, features: Dict[str, Any]) -> float:
        """Calcule un score de complexité normalisé"""
        complexity_map = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'very_high': 1.0}
        base_complexity = complexity_map.get(features['complexity'], 0.6)
        
        # Ajustements basés sur d'autres facteurs
        tech_complexity = len(features['technology_stack']) * 0.05
        integration_complexity = features['integration_points'] * 0.1
        
        return min(1.0, base_complexity + tech_complexity + integration_complexity)
    
    def _calculate_initial_risk_score(self, features: Dict[str, Any]) -> float:
        """Calcule un score de risque initial"""
        risk = 0.0
        
        # Facteurs de risque
        if features['deadline_pressure'] > 0.7:
            risk += 0.3
        if features['requirements_clarity'] < 0.5:
            risk += 0.4
        if features['team_experience'] == 'junior':
            risk += 0.3
        if features['stakeholder_involvement'] < 0.4:
            risk += 0.2
        if features['complexity_score'] > 0.8:
            risk += 0.3
            
        return min(1.0, risk)
    
    def _calculate_experience_factor(self, features: Dict[str, Any]) -> float:
        """Calcule un facteur d'expérience de l'équipe"""
        experience_map = {
            'junior': 0.7,
            'mixed': 0.85,
            'senior': 1.0,
            'expert': 1.15
        }
        return experience_map.get(features['team_experience'], 0.85)
    
    def _find_best_matching_pattern(self, features: Dict[str, Any]) -> ProjectPattern:
        """Trouve le pattern le plus proche du projet"""
        domain = features['domain']
        team_size = features['team_size']
        
        domain_patterns = self.patterns_db.get(domain, [])
        if not domain_patterns:
            domain_patterns = self.patterns_db.get('web_app', [])
        
        best_pattern = domain_patterns[0]
        best_score = 0
        
        for pattern in domain_patterns:
            score = self._calculate_pattern_similarity(features, pattern)
            if score > best_score:
                best_score = score
                best_pattern = pattern
                
        return best_pattern
    
    def _calculate_pattern_similarity(self, features: Dict[str, Any], pattern: ProjectPattern) -> float:
        """Calcule la similarité entre un projet et un pattern"""
        score = 0.0
        
        # Similarité de taille d'équipe
        team_size = features['team_size']
        if pattern.team_size_range[0] <= team_size <= pattern.team_size_range[1]:
            score += 0.4
        else:
            # Pénalité proportionnelle à l'écart
            min_range, max_range = pattern.team_size_range
            if team_size < min_range:
                score += max(0, 0.4 - (min_range - team_size) * 0.1)
            else:
                score += max(0, 0.4 - (team_size - max_range) * 0.1)
        
        # Bonus pour domaine exact
        if features['domain'] == pattern.domain.value:
            score += 0.6
        
        return score
    
    async def _run_ml_predictions(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute les prédictions ML (simulées pour cette démo)"""
        # Simulation de prédictions ML avancées
        base_duration = 30 + features['complexity_score'] * 60
        base_cost = features['budget'] * 0.8
        
        # Ajustements basés sur l'expérience
        experience_adj = features['experience_factor']
        adjusted_duration = base_duration / experience_adj
        adjusted_cost = base_cost * (2.0 - experience_adj)
        
        return {
            'duration': adjusted_duration,
            'cost': adjusted_cost,
            'success_probability': max(0.3, 0.9 - features['risk_score'] * 0.4),
            'confidence': 0.8 if features['requirements_clarity'] > 0.7 else 0.6
        }
    
    def _combine_predictions(self, pattern: ProjectPattern, ml_pred: Dict[str, Any], 
                           features: Dict[str, Any]) -> Dict[str, Any]:
        """Combine les prédictions pattern et ML"""
        
        # Pondération intelligente (plus de poids au ML si confiance élevée)
        ml_weight = 0.7 if ml_pred['confidence'] > 0.7 else 0.5
        pattern_weight = 1.0 - ml_weight
        
        # Prédictions du pattern
        base_duration = 50  # Baseline
        base_cost = features['budget']
        
        pattern_duration = base_duration * pattern.duration_multiplier
        pattern_cost = base_cost * pattern.cost_multiplier
        
        # Combinaison
        final_duration = (ml_pred['duration'] * ml_weight + 
                         pattern_duration * pattern_weight)
        final_cost = (ml_pred['cost'] * ml_weight + 
                     pattern_cost * pattern_weight)
        
        # Calcul de l'intervalle de confiance
        uncertainty = 0.2 if pattern.confidence == PredictionConfidence.HIGH else 0.35
        confidence_interval = (
            final_duration * (1 - uncertainty),
            final_duration * (1 + uncertainty)
        )
        
        # Facteurs de risque détaillés
        risk_factors = {
            'technical_risk': features['complexity_score'],
            'team_risk': 1.0 - features['experience_factor'],
            'timeline_risk': features['deadline_pressure'],
            'requirements_risk': 1.0 - features['requirements_clarity'],
            'domain_risk': 1.0 - pattern.success_rate
        }
        
        return {
            'duration': final_duration,
            'cost': final_cost,
            'success_probability': (ml_pred['success_probability'] + pattern.success_rate) / 2,
            'confidence_interval': confidence_interval,
            'risk_factors': risk_factors
        }
    
    def _generate_ai_recommendations(self, features: Dict[str, Any], 
                                   pattern: ProjectPattern,
                                   prediction: Dict[str, Any]) -> List[str]:
        """Génère des recommandations intelligentes"""
        recommendations = []
        
        # Recommandations basées sur les patterns
        if pattern.optimal_practices:
            recommendations.extend([f"✅ Appliquer: {practice}" for practice in pattern.optimal_practices[:3]])
        
        # Recommandations basées sur les risques
        high_risks = [k for k, v in prediction['risk_factors'].items() if v > 0.7]
        
        risk_recommendations = {
            'technical_risk': "🔧 Prévoir plus de temps pour les spikes techniques et POCs",
            'team_risk': "👥 Renforcer l'équipe avec des seniors ou prévoir plus de formation",
            'timeline_risk': "⏰ Négocier des délais réalistes ou réduire le scope",
            'requirements_risk': "📋 Organiser des ateliers utilisateur pour clarifier les besoins",
            'domain_risk': "🎯 S'inspirer des meilleures pratiques du domaine"
        }
        
        for risk in high_risks:
            if risk in risk_recommendations:
                recommendations.append(risk_recommendations[risk])
        
        # Recommandations optimisation
        if features['team_size'] > 10:
            recommendations.append("🏗️ Considérer une architecture en équipes autonomes")
        
        if prediction['cost'] > features['budget'] * 1.2:
            recommendations.append("💰 Revoir le scope ou négocier le budget")
        
        return recommendations[:5]  # Top 5 recommandations
    
    def _find_similar_projects(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trouve des projets similaires (simulé)"""
        # Simulation de recherche dans une base de projets historiques
        similar_projects = [
            {
                'name': f"Projet {features['domain']} similaire #{i+1}",
                'domain': features['domain'],
                'team_size': features['team_size'] + np.random.randint(-2, 3),
                'actual_duration': np.random.normal(60, 15),
                'actual_cost': np.random.normal(features['budget'], features['budget'] * 0.2),
                'success_rate': np.random.uniform(0.6, 0.9),
                'lessons_learned': [f"Leçon {j+1}" for j in range(2)]
            }
            for i in range(3)
        ]
        
        return similar_projects
    
    def _generate_optimization_suggestions(self, features: Dict[str, Any], 
                                         prediction: Dict[str, Any]) -> List[str]:
        """Génère des suggestions d'optimisation"""
        suggestions = []
        
        # Optimisations basées sur la prédiction
        if prediction['success_probability'] < 0.7:
            suggestions.append("🎯 Réduire le scope pour augmenter les chances de succès")
            
        if prediction['cost'] > features['budget']:
            suggestions.append("💡 Explorer des solutions no-code/low-code pour réduire les coûts")
            
        if prediction['risk_factors']['timeline_risk'] > 0.8:
            suggestions.append("⚡ Adopter une approche MVP pour livrer plus rapidement")
            
        # Optimisations techniques
        if features['complexity_score'] > 0.8:
            suggestions.append("🔬 Diviser en phases avec des POCs pour valider l'approche")
            
        if features['integration_points'] > 5:
            suggestions.append("🔌 Prévoir une couche d'abstraction pour les intégrations")
            
        return suggestions
    
    def learn_from_project(self, project_data: Dict[str, Any], actual_results: Dict[str, Any]):
        """Apprend d'un projet terminé pour améliorer les prédictions"""
        # Mise à jour des patterns si nécessaire
        # Réentraînement des modèles ML
        # Ajustement des poids de combinaison
        pass
    
    def get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """Retourne les insights IA pour un domaine spécifique"""
        patterns = self.patterns_db.get(domain, [])
        
        if not patterns:
            return {'error': f'Domaine {domain} non reconnu'}
        
        # Agrégation des insights
        avg_duration_mult = np.mean([p.duration_multiplier for p in patterns])
        avg_cost_mult = np.mean([p.cost_multiplier for p in patterns])
        avg_success_rate = np.mean([p.success_rate for p in patterns])
        
        all_risks = []
        all_practices = []
        
        for pattern in patterns:
            all_risks.extend(pattern.common_risks)
            all_practices.extend(pattern.optimal_practices)
        
        return {
            'domain': domain,
            'average_duration_multiplier': avg_duration_mult,
            'average_cost_multiplier': avg_cost_mult,
            'average_success_rate': avg_success_rate,
            'top_risks': list(set(all_risks))[:5],
            'best_practices': list(set(all_practices))[:5],
            'patterns_count': len(patterns)
        }


# Instance globale pour utilisation dans l'app
ai_predictor = AIProjectPredictor()