"""
💰 Optimisateur Budgétaire IA - PlannerIA
Allocation intelligente des ressources et optimisation budgétaire avec IA
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go


class BudgetCategory(Enum):
    DEVELOPMENT = "development"
    DESIGN = "design"
    INFRASTRUCTURE = "infrastructure"
    TESTING = "testing"
    PROJECT_MANAGEMENT = "project_management"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    TRAINING = "training"
    CONTINGENCY = "contingency"


class OptimizationObjective(Enum):
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_ROI = "maximize_roi"
    BALANCE_ALL = "balance_all"


@dataclass
class BudgetAllocation:
    """Allocation budgétaire pour une catégorie"""
    category: BudgetCategory
    current_amount: float
    optimized_amount: float
    minimum_required: float
    maximum_feasible: float
    roi_score: float
    impact_on_timeline: float
    impact_on_quality: float
    risk_factor: float
    justification: str


@dataclass
class OptimizationResult:
    """Résultat d'optimisation budgétaire"""
    total_current_budget: float
    total_optimized_budget: float
    savings: float
    savings_percentage: float
    allocations: List[BudgetAllocation]
    quality_impact: float
    timeline_impact: float
    risk_assessment: float
    confidence_score: float
    recommendations: List[str]
    alternative_scenarios: List[Dict[str, Any]]


@dataclass
class CostOptimizationStrategy:
    """Stratégie d'optimisation des coûts"""
    strategy_id: str
    name: str
    description: str
    category: BudgetCategory
    potential_savings: float
    implementation_difficulty: float
    risk_level: float
    prerequisites: List[str]
    expected_timeline: timedelta
    success_probability: float


class AIBudgetOptimizer:
    """Optimisateur budgétaire intelligent avec IA"""
    
    def __init__(self):
        self.optimization_models = self._initialize_optimization_models()
        self.cost_strategies = self._load_optimization_strategies()
        self.benchmark_data = self._load_industry_benchmarks()
        self.roi_models = self._initialize_roi_models()
        self.constraint_solver = BudgetConstraintSolver()
        
    def _initialize_optimization_models(self) -> Dict[str, Any]:
        """Initialise les modèles d'optimisation"""
        
        return {
            'cost_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'roi_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'quality_impact_model': RandomForestRegressor(n_estimators=50, random_state=42),
            'timeline_impact_model': RandomForestRegressor(n_estimators=50, random_state=42),
            'category_optimizer': KMeans(n_clusters=5, random_state=42)
        }
    
    def _load_optimization_strategies(self) -> Dict[str, List[CostOptimizationStrategy]]:
        """Charge les stratégies d'optimisation par catégorie"""
        
        strategies = {
            BudgetCategory.DEVELOPMENT.value: [
                CostOptimizationStrategy(
                    strategy_id="offshore_development",
                    name="Développement Offshore",
                    description="Délocaliser une partie du développement vers des équipes offshore qualifiées",
                    category=BudgetCategory.DEVELOPMENT,
                    potential_savings=0.4,  # 40% de réduction
                    implementation_difficulty=0.7,
                    risk_level=0.6,
                    prerequisites=["Processus well-defined", "Communication tools", "Time zone management"],
                    expected_timeline=timedelta(weeks=4),
                    success_probability=0.75
                ),
                CostOptimizationStrategy(
                    strategy_id="junior_senior_mix",
                    name="Mix Junior/Senior Optimal",
                    description="Optimiser le ratio junior/senior pour maximiser l'efficacité coût",
                    category=BudgetCategory.DEVELOPMENT,
                    potential_savings=0.25,
                    implementation_difficulty=0.3,
                    risk_level=0.4,
                    prerequisites=["Mentoring process", "Clear documentation"],
                    expected_timeline=timedelta(weeks=2),
                    success_probability=0.85
                ),
                CostOptimizationStrategy(
                    strategy_id="automation_tools",
                    name="Outils d'Automatisation",
                    description="Implémenter des outils d'automatisation pour réduire le travail manuel",
                    category=BudgetCategory.DEVELOPMENT,
                    potential_savings=0.3,
                    implementation_difficulty=0.5,
                    risk_level=0.3,
                    prerequisites=["Tool evaluation", "Team training"],
                    expected_timeline=timedelta(weeks=6),
                    success_probability=0.8
                )
            ],
            
            BudgetCategory.INFRASTRUCTURE.value: [
                CostOptimizationStrategy(
                    strategy_id="cloud_optimization",
                    name="Optimisation Cloud",
                    description="Optimiser l'utilisation des ressources cloud (auto-scaling, reserved instances)",
                    category=BudgetCategory.INFRASTRUCTURE,
                    potential_savings=0.35,
                    implementation_difficulty=0.4,
                    risk_level=0.2,
                    prerequisites=["Cloud monitoring", "Usage analysis"],
                    expected_timeline=timedelta(weeks=3),
                    success_probability=0.9
                ),
                CostOptimizationStrategy(
                    strategy_id="containerization",
                    name="Conteneurisation",
                    description="Migrer vers une architecture conteneurisée pour optimiser les ressources",
                    category=BudgetCategory.INFRASTRUCTURE,
                    potential_savings=0.45,
                    implementation_difficulty=0.8,
                    risk_level=0.5,
                    prerequisites=["Docker expertise", "Kubernetes knowledge", "Migration plan"],
                    expected_timeline=timedelta(weeks=8),
                    success_probability=0.7
                )
            ],
            
            BudgetCategory.TESTING.value: [
                CostOptimizationStrategy(
                    strategy_id="test_automation",
                    name="Automatisation des Tests",
                    description="Automatiser les tests répétitifs pour réduire les coûts de QA manuel",
                    category=BudgetCategory.TESTING,
                    potential_savings=0.5,
                    implementation_difficulty=0.6,
                    risk_level=0.3,
                    prerequisites=["Test framework", "CI/CD pipeline"],
                    expected_timeline=timedelta(weeks=5),
                    success_probability=0.85
                ),
                CostOptimizationStrategy(
                    strategy_id="risk_based_testing",
                    name="Tests Basés sur les Risques",
                    description="Concentrer les efforts de test sur les zones à haut risque",
                    category=BudgetCategory.TESTING,
                    potential_savings=0.3,
                    implementation_difficulty=0.4,
                    risk_level=0.4,
                    prerequisites=["Risk analysis", "Test prioritization"],
                    expected_timeline=timedelta(weeks=2),
                    success_probability=0.8
                )
            ],
            
            BudgetCategory.DESIGN.value: [
                CostOptimizationStrategy(
                    strategy_id="design_systems",
                    name="Système de Design",
                    description="Créer un système de design réutilisable pour réduire les coûts futurs",
                    category=BudgetCategory.DESIGN,
                    potential_savings=0.4,
                    implementation_difficulty=0.5,
                    risk_level=0.2,
                    prerequisites=["Design standards", "Component library"],
                    expected_timeline=timedelta(weeks=4),
                    success_probability=0.9
                ),
                CostOptimizationStrategy(
                    strategy_id="template_approach",
                    name="Approche Template",
                    description="Utiliser des templates et composants pré-conçus de qualité",
                    category=BudgetCategory.DESIGN,
                    potential_savings=0.6,
                    implementation_difficulty=0.2,
                    risk_level=0.3,
                    prerequisites=["Template evaluation", "Customization plan"],
                    expected_timeline=timedelta(weeks=1),
                    success_probability=0.85
                )
            ]
        }
        
        # Ajouter des stratégies pour les autres catégories
        for category in BudgetCategory:
            if category.value not in strategies:
                strategies[category.value] = [
                    CostOptimizationStrategy(
                        strategy_id=f"{category.value}_generic",
                        name=f"Optimisation {category.value.title()}",
                        description=f"Stratégies génériques d'optimisation pour {category.value}",
                        category=category,
                        potential_savings=0.2,
                        implementation_difficulty=0.5,
                        risk_level=0.4,
                        prerequisites=["Analysis required"],
                        expected_timeline=timedelta(weeks=3),
                        success_probability=0.7
                    )
                ]
        
        return strategies
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Charge les benchmarks industrie par domaine"""
        
        return {
            'web_app': {
                BudgetCategory.DEVELOPMENT.value: 0.60,
                BudgetCategory.DESIGN.value: 0.15,
                BudgetCategory.INFRASTRUCTURE.value: 0.10,
                BudgetCategory.TESTING.value: 0.10,
                BudgetCategory.PROJECT_MANAGEMENT.value: 0.05
            },
            'mobile_app': {
                BudgetCategory.DEVELOPMENT.value: 0.55,
                BudgetCategory.DESIGN.value: 0.20,
                BudgetCategory.INFRASTRUCTURE.value: 0.08,
                BudgetCategory.TESTING.value: 0.12,
                BudgetCategory.PROJECT_MANAGEMENT.value: 0.05
            },
            'enterprise': {
                BudgetCategory.DEVELOPMENT.value: 0.50,
                BudgetCategory.DESIGN.value: 0.12,
                BudgetCategory.INFRASTRUCTURE.value: 0.18,
                BudgetCategory.TESTING.value: 0.15,
                BudgetCategory.PROJECT_MANAGEMENT.value: 0.05
            },
            'ecommerce': {
                BudgetCategory.DEVELOPMENT.value: 0.45,
                BudgetCategory.DESIGN.value: 0.25,
                BudgetCategory.INFRASTRUCTURE.value: 0.15,
                BudgetCategory.TESTING.value: 0.10,
                BudgetCategory.PROJECT_MANAGEMENT.value: 0.05
            },
            'fintech': {
                BudgetCategory.DEVELOPMENT.value: 0.40,
                BudgetCategory.DESIGN.value: 0.15,
                BudgetCategory.INFRASTRUCTURE.value: 0.20,
                BudgetCategory.TESTING.value: 0.20,
                BudgetCategory.PROJECT_MANAGEMENT.value: 0.05
            }
        }
    
    def _initialize_roi_models(self) -> Dict[str, Any]:
        """Initialise les modèles de calcul de ROI"""
        
        # Modèles simplifiés de ROI par catégorie
        return {
            BudgetCategory.DEVELOPMENT.value: {
                'base_roi': 2.5,
                'quality_multiplier': 1.3,
                'time_factor': -0.1  # Négatif car plus de temps = moins de ROI
            },
            BudgetCategory.DESIGN.value: {
                'base_roi': 3.2,  # Design a souvent un ROI élevé
                'quality_multiplier': 1.5,
                'time_factor': -0.05
            },
            BudgetCategory.INFRASTRUCTURE.value: {
                'base_roi': 1.8,
                'quality_multiplier': 1.1,
                'time_factor': -0.2
            },
            BudgetCategory.TESTING.value: {
                'base_roi': 4.0,  # Tests ont un ROI très élevé (évitent les bugs coûteux)
                'quality_multiplier': 2.0,
                'time_factor': -0.05
            },
            BudgetCategory.PROJECT_MANAGEMENT.value: {
                'base_roi': 2.8,
                'quality_multiplier': 1.2,
                'time_factor': -0.15
            }
        }
    
    async def optimize_budget(self, project_data: Dict[str, Any], 
                            current_budget: Dict[BudgetCategory, float],
                            objective: OptimizationObjective = OptimizationObjective.BALANCE_ALL,
                            constraints: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimise l'allocation budgétaire avec IA"""
        
        # Préparation des données
        total_budget = sum(current_budget.values())
        project_domain = project_data.get('domain', 'web_app')
        
        # Analyse des benchmarks industrie
        benchmark_analysis = self._analyze_industry_benchmarks(current_budget, project_domain)
        
        # Identification des opportunités d'optimisation
        optimization_opportunities = await self._identify_optimization_opportunities(
            project_data, current_budget
        )
        
        # Optimisation mathématique
        optimized_allocation = await self._perform_mathematical_optimization(
            current_budget, optimization_opportunities, objective, constraints
        )
        
        # Calcul des impacts
        impacts = self._calculate_optimization_impacts(current_budget, optimized_allocation)
        
        # Génération des recommandations
        recommendations = self._generate_optimization_recommendations(
            optimization_opportunities, benchmark_analysis, impacts
        )
        
        # Scénarios alternatifs
        alternative_scenarios = self._generate_alternative_scenarios(
            current_budget, optimization_opportunities
        )
        
        # Construction du résultat
        result = OptimizationResult(
            total_current_budget=total_budget,
            total_optimized_budget=sum(allocation.optimized_amount for allocation in optimized_allocation),
            savings=total_budget - sum(allocation.optimized_amount for allocation in optimized_allocation),
            savings_percentage=((total_budget - sum(allocation.optimized_amount for allocation in optimized_allocation)) / total_budget) * 100,
            allocations=optimized_allocation,
            quality_impact=impacts['quality_impact'],
            timeline_impact=impacts['timeline_impact'],
            risk_assessment=impacts['risk_assessment'],
            confidence_score=impacts['confidence_score'],
            recommendations=recommendations,
            alternative_scenarios=alternative_scenarios
        )
        
        return result
    
    def _analyze_industry_benchmarks(self, current_budget: Dict[BudgetCategory, float], 
                                   domain: str) -> Dict[str, Any]:
        """Analyse les écarts avec les benchmarks industrie"""
        
        benchmarks = self.benchmark_data.get(domain, self.benchmark_data['web_app'])
        total_budget = sum(current_budget.values())
        
        analysis = {
            'deviations': {},
            'recommendations': [],
            'domain': domain,
            'total_budget': total_budget
        }
        
        for category, current_amount in current_budget.items():
            current_ratio = current_amount / total_budget if total_budget > 0 else 0
            benchmark_ratio = benchmarks.get(category.value, 0.2)
            
            deviation = current_ratio - benchmark_ratio
            analysis['deviations'][category.value] = {
                'current_ratio': current_ratio,
                'benchmark_ratio': benchmark_ratio,
                'deviation': deviation,
                'deviation_percentage': (deviation / benchmark_ratio) * 100 if benchmark_ratio > 0 else 0
            }
            
            # Recommandations basées sur les écarts
            if abs(deviation) > 0.05:  # Écart significatif de 5%
                if deviation > 0:
                    analysis['recommendations'].append(
                        f"💡 {category.value.title()}: Sur-allocation de {deviation*100:.1f}% vs industrie"
                    )
                else:
                    analysis['recommendations'].append(
                        f"⚠️ {category.value.title()}: Sous-allocation de {abs(deviation)*100:.1f}% vs industrie"
                    )
        
        return analysis
    
    async def _identify_optimization_opportunities(self, project_data: Dict[str, Any],
                                                 current_budget: Dict[BudgetCategory, float]) -> List[CostOptimizationStrategy]:
        """Identifie les opportunités d'optimisation"""
        
        opportunities = []
        
        # Analyse par catégorie
        for category, amount in current_budget.items():
            category_strategies = self.cost_strategies.get(category.value, [])
            
            for strategy in category_strategies:
                # Évaluation de la pertinence de la stratégie
                relevance_score = self._calculate_strategy_relevance(
                    strategy, project_data, amount
                )
                
                if relevance_score > 0.6:  # Seuil de pertinence
                    # Ajustement des métriques selon le contexte
                    adjusted_strategy = self._adjust_strategy_for_context(
                        strategy, project_data, amount, relevance_score
                    )
                    opportunities.append(adjusted_strategy)
        
        # Tri par potentiel d'économies ajusté par le risque
        opportunities.sort(
            key=lambda s: s.potential_savings * s.success_probability / (1 + s.risk_level),
            reverse=True
        )
        
        return opportunities[:10]  # Top 10 opportunités
    
    def _calculate_strategy_relevance(self, strategy: CostOptimizationStrategy,
                                    project_data: Dict[str, Any], budget_amount: float) -> float:
        """Calcule la pertinence d'une stratégie pour le projet"""
        
        relevance = 0.5  # Base
        
        # Facteurs de pertinence
        team_size = len(project_data.get('resources', []))
        project_duration = project_data.get('project_overview', {}).get('total_duration', 90)
        complexity = project_data.get('complexity', 'medium')
        
        # Ajustements selon la stratégie
        if strategy.strategy_id == "offshore_development":
            if team_size > 5 and budget_amount > 50000:
                relevance += 0.3
            if project_duration > 120:  # Projets longs
                relevance += 0.2
        
        elif strategy.strategy_id == "junior_senior_mix":
            if team_size > 3:
                relevance += 0.4
            if complexity in ['low', 'medium']:
                relevance += 0.3
        
        elif strategy.strategy_id == "cloud_optimization":
            if project_data.get('uses_cloud', True):
                relevance += 0.4
            if budget_amount > 20000:
                relevance += 0.2
        
        elif strategy.strategy_id == "test_automation":
            if project_duration > 60:
                relevance += 0.3
            if complexity in ['high', 'very_high']:
                relevance += 0.4
        
        return min(1.0, relevance)
    
    def _adjust_strategy_for_context(self, strategy: CostOptimizationStrategy,
                                   project_data: Dict[str, Any], budget_amount: float,
                                   relevance_score: float) -> CostOptimizationStrategy:
        """Ajuste une stratégie selon le contexte du projet"""
        
        # Copie de la stratégie pour ajustement
        adjusted = CostOptimizationStrategy(
            strategy_id=strategy.strategy_id,
            name=strategy.name,
            description=strategy.description,
            category=strategy.category,
            potential_savings=strategy.potential_savings,
            implementation_difficulty=strategy.implementation_difficulty,
            risk_level=strategy.risk_level,
            prerequisites=strategy.prerequisites.copy(),
            expected_timeline=strategy.expected_timeline,
            success_probability=strategy.success_probability
        )
        
        # Ajustements contextuels
        team_size = len(project_data.get('resources', []))
        complexity = project_data.get('complexity', 'medium')
        
        # Ajustement des économies potentielles selon le budget
        budget_factor = min(2.0, budget_amount / 50000)  # Normalisation
        adjusted.potential_savings *= budget_factor * relevance_score
        
        # Ajustement du risque selon la complexité
        complexity_risk_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3,
            'very_high': 1.6
        }
        adjusted.risk_level *= complexity_risk_multiplier.get(complexity, 1.0)
        
        # Ajustement de la probabilité de succès selon l'équipe
        if team_size < 3:
            adjusted.success_probability *= 0.8  # Petite équipe = plus risqué
        elif team_size > 10:
            adjusted.success_probability *= 0.9  # Grande équipe = coordination difficile
        
        return adjusted
    
    async def _perform_mathematical_optimization(self, current_budget: Dict[BudgetCategory, float],
                                               opportunities: List[CostOptimizationStrategy],
                                               objective: OptimizationObjective,
                                               constraints: Optional[Dict[str, Any]]) -> List[BudgetAllocation]:
        """Effectue l'optimisation mathématique avec contraintes"""
        
        allocations = []
        
        for category, current_amount in current_budget.items():
            # Recherche des opportunités pour cette catégorie
            category_opportunities = [
                opp for opp in opportunities 
                if opp.category == category
            ]
            
            # Calcul de l'optimisation pour cette catégorie
            optimized_result = self._optimize_category_budget(
                category, current_amount, category_opportunities, objective, constraints
            )
            
            allocations.append(optimized_result)
        
        # Optimisation globale avec contraintes inter-catégories
        return self._apply_global_constraints(allocations, constraints)
    
    def _optimize_category_budget(self, category: BudgetCategory, current_amount: float,
                                opportunities: List[CostOptimizationStrategy],
                                objective: OptimizationObjective,
                                constraints: Optional[Dict[str, Any]]) -> BudgetAllocation:
        """Optimise le budget d'une catégorie spécifique"""
        
        # Calcul des limites
        min_required = current_amount * 0.3  # Minimum 30% du budget actuel
        max_feasible = current_amount * 1.5  # Maximum 150% du budget actuel
        
        # Application des contraintes spécifiques
        if constraints:
            category_constraints = constraints.get(category.value, {})
            min_required = max(min_required, category_constraints.get('min_amount', 0))
            max_feasible = min(max_feasible, category_constraints.get('max_amount', float('inf')))
        
        # Calcul de l'optimisation basée sur les opportunités
        potential_savings = 0
        weighted_risk = 0
        
        for opportunity in opportunities:
            savings = current_amount * opportunity.potential_savings * opportunity.success_probability
            potential_savings += savings
            weighted_risk += opportunity.risk_level * (savings / current_amount) if current_amount > 0 else 0
        
        # Optimisation selon l'objectif
        if objective == OptimizationObjective.MINIMIZE_COST:
            optimized_amount = max(min_required, current_amount - potential_savings * 0.8)
        elif objective == OptimizationObjective.MAXIMIZE_QUALITY:
            optimized_amount = min(max_feasible, current_amount * 1.2)
        elif objective == OptimizationObjective.MINIMIZE_TIME:
            optimized_amount = min(max_feasible, current_amount * 1.1)
        else:  # BALANCE_ALL
            savings_factor = min(0.5, potential_savings / current_amount) if current_amount > 0 else 0
            optimized_amount = current_amount * (1 - savings_factor * 0.6)
            optimized_amount = max(min_required, min(max_feasible, optimized_amount))
        
        # Calcul du ROI
        roi_data = self.roi_models.get(category.value, {'base_roi': 2.0})
        roi_score = roi_data['base_roi'] * (optimized_amount / current_amount) if current_amount > 0 else 0
        
        # Impacts sur timeline et qualité
        timeline_impact = self._calculate_timeline_impact(category, current_amount, optimized_amount)
        quality_impact = self._calculate_quality_impact(category, current_amount, optimized_amount)
        
        # Justification
        if optimized_amount < current_amount:
            savings_pct = ((current_amount - optimized_amount) / current_amount) * 100
            justification = f"Optimisation identifiée: -{savings_pct:.1f}% grâce aux stratégies disponibles"
        elif optimized_amount > current_amount:
            increase_pct = ((optimized_amount - current_amount) / current_amount) * 100
            justification = f"Investissement recommandé: +{increase_pct:.1f}% pour améliorer ROI et qualité"
        else:
            justification = "Allocation optimale actuelle, pas de changement recommandé"
        
        return BudgetAllocation(
            category=category,
            current_amount=current_amount,
            optimized_amount=optimized_amount,
            minimum_required=min_required,
            maximum_feasible=max_feasible,
            roi_score=roi_score,
            impact_on_timeline=timeline_impact,
            impact_on_quality=quality_impact,
            risk_factor=weighted_risk,
            justification=justification
        )
    
    def _apply_global_constraints(self, allocations: List[BudgetAllocation],
                                constraints: Optional[Dict[str, Any]]) -> List[BudgetAllocation]:
        """Applique les contraintes globales et réajuste si nécessaire"""
        
        total_optimized = sum(alloc.optimized_amount for alloc in allocations)
        total_current = sum(alloc.current_amount for alloc in allocations)
        
        # Contrainte de budget total
        if constraints and 'max_total_budget' in constraints:
            max_budget = constraints['max_total_budget']
            if total_optimized > max_budget:
                # Réduction proportionnelle
                reduction_factor = max_budget / total_optimized
                for allocation in allocations:
                    allocation.optimized_amount *= reduction_factor
                    allocation.justification += f" (Ajusté: contrainte budget total {max_budget:,.0f}€)"
        
        # Contrainte de variation maximale
        if constraints and 'max_variation_percent' in constraints:
            max_variation = constraints['max_variation_percent'] / 100
            
            for allocation in allocations:
                if allocation.current_amount > 0:
                    variation = abs(allocation.optimized_amount - allocation.current_amount) / allocation.current_amount
                    
                    if variation > max_variation:
                        # Limiter la variation
                        if allocation.optimized_amount > allocation.current_amount:
                            allocation.optimized_amount = allocation.current_amount * (1 + max_variation)
                        else:
                            allocation.optimized_amount = allocation.current_amount * (1 - max_variation)
                        
                        allocation.justification += f" (Limitée: variation max {max_variation*100:.1f}%)"
        
        return allocations
    
    def _calculate_timeline_impact(self, category: BudgetCategory, 
                                 current_amount: float, optimized_amount: float) -> float:
        """Calcule l'impact sur le timeline"""
        
        if current_amount == 0:
            return 0
        
        change_ratio = optimized_amount / current_amount
        
        # Impact différent selon la catégorie
        timeline_sensitivity = {
            BudgetCategory.DEVELOPMENT: 0.8,  # Très sensible
            BudgetCategory.TESTING: 0.6,     # Modérément sensible
            BudgetCategory.DESIGN: 0.4,      # Peu sensible
            BudgetCategory.INFRASTRUCTURE: 0.5,
            BudgetCategory.PROJECT_MANAGEMENT: 0.7
        }
        
        sensitivity = timeline_sensitivity.get(category, 0.5)
        impact = (change_ratio - 1) * sensitivity
        
        return np.clip(impact, -0.5, 0.5)  # Impact entre -50% et +50%
    
    def _calculate_quality_impact(self, category: BudgetCategory,
                                current_amount: float, optimized_amount: float) -> float:
        """Calcule l'impact sur la qualité"""
        
        if current_amount == 0:
            return 0
        
        change_ratio = optimized_amount / current_amount
        
        # Impact différent selon la catégorie
        quality_sensitivity = {
            BudgetCategory.TESTING: 0.9,     # Très sensible
            BudgetCategory.DEVELOPMENT: 0.7,  # Sensible
            BudgetCategory.DESIGN: 0.6,      # Modérément sensible
            BudgetCategory.INFRASTRUCTURE: 0.4,
            BudgetCategory.PROJECT_MANAGEMENT: 0.3
        }
        
        sensitivity = quality_sensitivity.get(category, 0.5)
        impact = (change_ratio - 1) * sensitivity
        
        return np.clip(impact, -0.6, 0.6)  # Impact entre -60% et +60%
    
    def _calculate_optimization_impacts(self, current_budget: Dict[BudgetCategory, float],
                                      optimized_allocation: List[BudgetAllocation]) -> Dict[str, float]:
        """Calcule les impacts globaux de l'optimisation"""
        
        # Impact qualité global
        quality_impacts = [alloc.impact_on_quality for alloc in optimized_allocation]
        quality_impact = np.mean(quality_impacts)
        
        # Impact timeline global
        timeline_impacts = [alloc.impact_on_timeline for alloc in optimized_allocation]
        timeline_impact = np.mean(timeline_impacts)
        
        # Évaluation du risque global
        risk_factors = [alloc.risk_factor for alloc in optimized_allocation]
        risk_assessment = np.mean(risk_factors)
        
        # Score de confiance basé sur la cohérence des optimisations
        variations = []
        for alloc in optimized_allocation:
            if alloc.current_amount > 0:
                variation = abs(alloc.optimized_amount - alloc.current_amount) / alloc.current_amount
                variations.append(variation)
        
        # Plus les variations sont modérées, plus la confiance est élevée
        avg_variation = np.mean(variations) if variations else 0
        confidence_score = max(0.5, 1.0 - avg_variation)
        
        return {
            'quality_impact': quality_impact,
            'timeline_impact': timeline_impact,
            'risk_assessment': risk_assessment,
            'confidence_score': confidence_score
        }
    
    def _generate_optimization_recommendations(self, opportunities: List[CostOptimizationStrategy],
                                             benchmark_analysis: Dict[str, Any],
                                             impacts: Dict[str, float]) -> List[str]:
        """Génère des recommandations d'optimisation"""
        
        recommendations = []
        
        # Recommandations basées sur les opportunités
        top_opportunities = opportunities[:5]
        for opp in top_opportunities:
            if opp.potential_savings > 0.2 and opp.success_probability > 0.7:
                recommendations.append(
                    f"💰 **{opp.name}**: Économies potentielles de {opp.potential_savings*100:.0f}% "
                    f"({opp.success_probability*100:.0f}% de chance de succès)"
                )
        
        # Recommandations basées sur les benchmarks
        for category, deviation_data in benchmark_analysis['deviations'].items():
            deviation_pct = deviation_data['deviation_percentage']
            
            if abs(deviation_pct) > 20:  # Écart significatif de 20%
                if deviation_pct > 0:
                    recommendations.append(
                        f"📊 **{category.title()}**: Réduire de {abs(deviation_pct):.0f}% pour aligner sur l'industrie"
                    )
                else:
                    recommendations.append(
                        f"📈 **{category.title()}**: Augmenter de {abs(deviation_pct):.0f}% (sous-investi vs industrie)"
                    )
        
        # Recommandations basées sur les impacts
        if impacts['quality_impact'] < -0.2:
            recommendations.append(
                "⚠️ **Attention Qualité**: L'optimisation pourrait impacter la qualité. "
                "Considérez des investissements compensatoires en tests et revues."
            )
        
        if impacts['timeline_impact'] > 0.15:
            recommendations.append(
                "⏱️ **Accélération Possible**: L'optimisation pourrait accélérer le projet. "
                "Profitez-en pour renforcer la qualité ou réduire les risques."
            )
        
        # Recommandations de mise en œuvre
        recommendations.append(
            "🎯 **Mise en œuvre**: Implémenter les optimisations par phases, "
            "en commençant par les moins risquées et à fort impact."
        )
        
        return recommendations[:8]  # Maximum 8 recommandations
    
    def _generate_alternative_scenarios(self, current_budget: Dict[BudgetCategory, float],
                                      opportunities: List[CostOptimizationStrategy]) -> List[Dict[str, Any]]:
        """Génère des scénarios alternatifs d'optimisation"""
        
        scenarios = []
        total_current = sum(current_budget.values())
        
        # Scénario conservateur (économies minimales, risque faible)
        conservative_savings = sum(
            current_budget.get(opp.category, 0) * opp.potential_savings * 0.5
            for opp in opportunities
            if opp.risk_level < 0.4 and opp.success_probability > 0.8
        )
        
        scenarios.append({
            'name': 'Conservateur',
            'description': 'Optimisations à faible risque seulement',
            'total_budget': total_current - conservative_savings,
            'savings': conservative_savings,
            'savings_percentage': (conservative_savings / total_current) * 100,
            'risk_level': 0.2,
            'implementation_time': 'Court terme (2-4 semaines)'
        })
        
        # Scénario agressif (économies maximales)
        aggressive_savings = sum(
            current_budget.get(opp.category, 0) * opp.potential_savings * 0.9
            for opp in opportunities[:8]  # Top 8 opportunités
        )
        
        scenarios.append({
            'name': 'Agressif',
            'description': 'Optimisations maximales avec tous les leviers',
            'total_budget': total_current - aggressive_savings,
            'savings': aggressive_savings,
            'savings_percentage': (aggressive_savings / total_current) * 100,
            'risk_level': 0.7,
            'implementation_time': 'Long terme (3-6 mois)'
        })
        
        # Scénario équilibré
        balanced_savings = sum(
            current_budget.get(opp.category, 0) * opp.potential_savings * 0.7
            for opp in opportunities
            if 0.3 < opp.risk_level < 0.6 and opp.success_probability > 0.7
        )
        
        scenarios.append({
            'name': 'Équilibré',
            'description': 'Bon compromis économies/risques',
            'total_budget': total_current - balanced_savings,
            'savings': balanced_savings,
            'savings_percentage': (balanced_savings / total_current) * 100,
            'risk_level': 0.4,
            'implementation_time': 'Moyen terme (1-3 mois)'
        })
        
        return scenarios
    
    def generate_budget_insights(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """Génère des insights détaillés sur l'optimisation budgétaire"""
        
        insights = {
            'summary': {},
            'category_analysis': {},
            'roi_analysis': {},
            'risk_assessment': {},
            'implementation_roadmap': []
        }
        
        # Résumé exécutif
        insights['summary'] = {
            'total_savings': optimization_result.savings,
            'savings_percentage': optimization_result.savings_percentage,
            'confidence_level': 'Élevé' if optimization_result.confidence_score > 0.8 else 'Moyen',
            'overall_recommendation': self._get_overall_recommendation(optimization_result)
        }
        
        # Analyse par catégorie
        for allocation in optimization_result.allocations:
            category_name = allocation.category.value
            
            change_amount = allocation.optimized_amount - allocation.current_amount
            change_percentage = (change_amount / allocation.current_amount) * 100 if allocation.current_amount > 0 else 0
            
            insights['category_analysis'][category_name] = {
                'current_amount': allocation.current_amount,
                'optimized_amount': allocation.optimized_amount,
                'change_amount': change_amount,
                'change_percentage': change_percentage,
                'roi_score': allocation.roi_score,
                'risk_level': allocation.risk_factor,
                'recommendation': 'Réduire' if change_amount < 0 else 'Augmenter' if change_amount > 0 else 'Maintenir'
            }
        
        # Analyse ROI
        roi_scores = [alloc.roi_score for alloc in optimization_result.allocations]
        insights['roi_analysis'] = {
            'average_roi': np.mean(roi_scores),
            'best_roi_category': max(optimization_result.allocations, key=lambda a: a.roi_score).category.value,
            'total_roi_improvement': sum(alloc.roi_score * alloc.optimized_amount for alloc in optimization_result.allocations)
        }
        
        # Évaluation des risques
        insights['risk_assessment'] = {
            'overall_risk': optimization_result.risk_assessment,
            'risk_level': 'Faible' if optimization_result.risk_assessment < 0.3 else 'Modéré' if optimization_result.risk_assessment < 0.6 else 'Élevé',
            'high_risk_categories': [
                alloc.category.value for alloc in optimization_result.allocations
                if alloc.risk_factor > 0.6
            ]
        }
        
        # Feuille de route d'implémentation
        insights['implementation_roadmap'] = self._create_implementation_roadmap(optimization_result)
        
        return insights
    
    def _get_overall_recommendation(self, result: OptimizationResult) -> str:
        """Génère une recommandation globale"""
        
        if result.savings_percentage > 20 and result.confidence_score > 0.7:
            return "Optimisation fortement recommandée - potentiel d'économies élevé avec risque maîtrisé"
        elif result.savings_percentage > 10 and result.risk_assessment < 0.5:
            return "Optimisation recommandée - économies substantielles avec risque acceptable"
        elif result.savings_percentage > 5:
            return "Optimisation modérée recommandée - amélioration progressive possible"
        else:
            return "Budget déjà bien optimisé - optimisations mineures possibles"
    
    def _create_implementation_roadmap(self, result: OptimizationResult) -> List[Dict[str, Any]]:
        """Crée une feuille de route d'implémentation"""
        
        roadmap = []
        
        # Phase 1: Quick wins (0-4 semaines)
        quick_wins = [
            alloc for alloc in result.allocations
            if alloc.risk_factor < 0.3 and abs(alloc.optimized_amount - alloc.current_amount) > 1000
        ]
        
        if quick_wins:
            roadmap.append({
                'phase': 'Phase 1 - Quick Wins',
                'duration': '0-4 semaines',
                'categories': [alloc.category.value for alloc in quick_wins],
                'expected_savings': sum(alloc.current_amount - alloc.optimized_amount for alloc in quick_wins if alloc.current_amount > alloc.optimized_amount),
                'risk_level': 'Faible',
                'priority': 'Haute'
            })
        
        # Phase 2: Optimisations structurelles (1-3 mois)
        structural_changes = [
            alloc for alloc in result.allocations
            if 0.3 <= alloc.risk_factor < 0.6 and abs(alloc.optimized_amount - alloc.current_amount) > 2000
        ]
        
        if structural_changes:
            roadmap.append({
                'phase': 'Phase 2 - Optimisations Structurelles',
                'duration': '1-3 mois',
                'categories': [alloc.category.value for alloc in structural_changes],
                'expected_savings': sum(alloc.current_amount - alloc.optimized_amount for alloc in structural_changes if alloc.current_amount > alloc.optimized_amount),
                'risk_level': 'Modéré',
                'priority': 'Moyenne'
            })
        
        # Phase 3: Transformations majeures (3-6 mois)
        major_transformations = [
            alloc for alloc in result.allocations
            if alloc.risk_factor >= 0.6 and abs(alloc.optimized_amount - alloc.current_amount) > 5000
        ]
        
        if major_transformations:
            roadmap.append({
                'phase': 'Phase 3 - Transformations Majeures',
                'duration': '3-6 mois',
                'categories': [alloc.category.value for alloc in major_transformations],
                'expected_savings': sum(alloc.current_amount - alloc.optimized_amount for alloc in major_transformations if alloc.current_amount > alloc.optimized_amount),
                'risk_level': 'Élevé',
                'priority': 'À évaluer'
            })
        
        return roadmap
    
    def render_optimization_dashboard(self, result: OptimizationResult):
        """Rend le dashboard d'optimisation budgétaire dans Streamlit"""
        
        st.subheader("💰 Optimisation Budgétaire IA")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Économies Potentielles",
                f"{result.savings:,.0f}€",
                f"-{result.savings_percentage:.1f}%"
            )
        
        with col2:
            st.metric(
                "Nouveau Budget Total",
                f"{result.total_optimized_budget:,.0f}€",
                f"vs {result.total_current_budget:,.0f}€"
            )
        
        with col3:
            confidence_color = "normal" if result.confidence_score > 0.7 else "inverse"
            st.metric(
                "Confiance IA",
                f"{result.confidence_score:.1%}",
                delta_color=confidence_color
            )
        
        with col4:
            risk_color = "inverse" if result.risk_assessment > 0.6 else "normal"
            st.metric(
                "Niveau de Risque",
                f"{result.risk_assessment:.1%}",
                delta_color=risk_color
            )
        
        # Graphiques d'optimisation
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_budget_comparison_chart(result)
        
        with col2:
            self._render_impact_analysis_chart(result)
        
        # Détails par catégorie
        st.subheader("📊 Détails par Catégorie")
        self._render_category_details_table(result)
        
        # Recommandations
        st.subheader("💡 Recommandations IA")
        for i, rec in enumerate(result.recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Scénarios alternatifs
        st.subheader("🎯 Scénarios Alternatifs")
        self._render_alternative_scenarios(result)
    
    def _render_budget_comparison_chart(self, result: OptimizationResult):
        """Graphique de comparaison budgétaire"""
        
        categories = [alloc.category.value.replace('_', ' ').title() for alloc in result.allocations]
        current_amounts = [alloc.current_amount for alloc in result.allocations]
        optimized_amounts = [alloc.optimized_amount for alloc in result.allocations]
        
        fig = go.Figure(data=[
            go.Bar(name='Budget Actuel', x=categories, y=current_amounts, marker_color='lightblue'),
            go.Bar(name='Budget Optimisé', x=categories, y=optimized_amounts, marker_color='darkblue')
        ])
        
        fig.update_layout(
            title='Comparaison Budget Actuel vs Optimisé',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_impact_analysis_chart(self, result: OptimizationResult):
        """Graphique d'analyse d'impact"""
        
        categories = [alloc.category.value.replace('_', ' ').title() for alloc in result.allocations]
        timeline_impacts = [alloc.impact_on_timeline * 100 for alloc in result.allocations]
        quality_impacts = [alloc.impact_on_quality * 100 for alloc in result.allocations]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timeline_impacts,
            y=quality_impacts,
            mode='markers+text',
            text=categories,
            textposition="middle right",
            marker=dict(
                size=[alloc.risk_factor * 50 + 10 for alloc in result.allocations],
                color=[alloc.roi_score for alloc in result.allocations],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="ROI Score")
            ),
            name='Catégories'
        ))
        
        fig.update_layout(
            title='Impact Timeline vs Qualité (Taille = Risque, Couleur = ROI)',
            xaxis_title='Impact Timeline (%)',
            yaxis_title='Impact Qualité (%)',
            height=400
        )
        
        # Lignes de référence
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_category_details_table(self, result: OptimizationResult):
        """Tableau détaillé des catégories"""
        
        data = []
        
        for alloc in result.allocations:
            change_amount = alloc.optimized_amount - alloc.current_amount
            change_pct = (change_amount / alloc.current_amount) * 100 if alloc.current_amount > 0 else 0
            
            data.append({
                'Catégorie': alloc.category.value.replace('_', ' ').title(),
                'Budget Actuel (€)': f"{alloc.current_amount:,.0f}",
                'Budget Optimisé (€)': f"{alloc.optimized_amount:,.0f}",
                'Changement (€)': f"{change_amount:+,.0f}",
                'Changement (%)': f"{change_pct:+.1f}%",
                'ROI Score': f"{alloc.roi_score:.2f}",
                'Risque': f"{alloc.risk_factor:.1%}",
                'Justification': alloc.justification
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    def _render_alternative_scenarios(self, result: OptimizationResult):
        """Affichage des scénarios alternatifs"""
        
        for scenario in result.alternative_scenarios:
            with st.expander(f"📋 Scénario {scenario['name']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Budget Total",
                        f"{scenario['total_budget']:,.0f}€"
                    )
                
                with col2:
                    st.metric(
                        "Économies",
                        f"{scenario['savings']:,.0f}€",
                        f"-{scenario['savings_percentage']:.1f}%"
                    )
                
                with col3:
                    risk_color = "inverse" if scenario['risk_level'] > 0.6 else "normal"
                    st.metric(
                        "Niveau de Risque",
                        f"{scenario['risk_level']:.1%}",
                        delta_color=risk_color
                    )
                
                st.markdown(f"**Description**: {scenario['description']}")
                st.markdown(f"**Temps d'implémentation**: {scenario['implementation_time']}")


class BudgetConstraintSolver:
    """Solveur de contraintes budgétaires"""
    
    def solve_with_constraints(self, variables: Dict[str, float], 
                              constraints: List[Dict[str, Any]], 
                              objective: str = "minimize") -> Dict[str, float]:
        """Résout l'optimisation avec contraintes"""
        
        # Implémentation simplifiée - à améliorer avec des solveurs plus avancés
        return variables  # Placeholder


# Instance globale
ai_budget_optimizer = AIBudgetOptimizer()