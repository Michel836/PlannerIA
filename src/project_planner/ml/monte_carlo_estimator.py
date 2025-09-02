"""
Système d'estimation Monte Carlo pour PlannerIA
Remplace les modèles ML défaillants par des simulations probabilistes robustes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
import pickle


@dataclass
class TaskEstimate:
    """Résultat d'estimation pour une tâche"""
    task_name: str
    optimistic: float
    realistic: float
    pessimistic: float
    confidence_interval_50: Tuple[float, float]
    confidence_interval_90: Tuple[float, float]
    mean_duration: float
    std_deviation: float
    risk_factors: Dict[str, float]


@dataclass
class ProjectEstimate:
    """Résultat d'estimation pour un projet complet"""
    project_name: str
    total_optimistic: float
    total_realistic: float
    total_pessimistic: float
    probability_on_time: float
    expected_duration: float
    tasks: List[TaskEstimate]
    risk_assessment: Dict[str, float]
    budget_simulation: Dict[str, float]


class MonteCarloEstimator:
    """Estimateur Monte Carlo pour projets logiciels"""
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        self.random_state = np.random.RandomState(42)  # Reproductibilité
        
        # Facteurs de risque par défaut basés sur des données empiriques
        self.default_risk_factors = {
            'complexity_multiplier': {'low': 0.8, 'medium': 1.0, 'high': 1.5, 'very_high': 2.0},
            'team_experience': {'junior': 1.4, 'mixed': 1.2, 'senior': 1.0, 'expert': 0.9},
            'requirements_clarity': {'unclear': 1.6, 'partial': 1.3, 'clear': 1.0, 'detailed': 0.9},
            'technology_risk': {'new': 1.5, 'familiar': 1.1, 'proven': 1.0},
            'integration_complexity': {'none': 1.0, 'simple': 1.2, 'complex': 1.5, 'very_complex': 2.0}
        }
        
        # Distributions d'incertitude (écart-type relatif)
        self.uncertainty_factors = {
            'analysis': 0.25,      # Tâches d'analyse: 25% d'incertitude
            'development': 0.35,   # Développement: 35% d'incertitude
            'testing': 0.30,       # Tests: 30% d'incertitude
            'integration': 0.40,   # Intégration: 40% d'incertitude
            'deployment': 0.20     # Déploiement: 20% d'incertitude
        }
    
    def estimate_task(self, task_data: Dict) -> TaskEstimate:
        """
        Estime une tâche individuelle avec simulation Monte Carlo
        
        Args:
            task_data: Dictionnaire avec les paramètres de la tâche
                - name: nom de la tâche
                - base_estimate: estimation de base en heures/jours
                - task_type: type de tâche (analysis, development, testing, etc.)
                - complexity: complexité (low, medium, high, very_high)
                - team_experience: expérience équipe
                - requirements_clarity: clarté des exigences
                - technology_risk: risque technologique
                - integration_complexity: complexité d'intégration
        """
        task_name = task_data.get('name', 'Tâche sans nom')
        base_estimate = task_data.get('base_estimate', 8.0)  # 1 jour par défaut
        task_type = task_data.get('task_type', 'development')
        
        # Calcul des multiplicateurs de risque
        risk_multiplier = 1.0
        risk_details = {}
        
        for factor, value in task_data.items():
            if factor in self.default_risk_factors:
                factor_mult = self.default_risk_factors[factor].get(value, 1.0)
                risk_multiplier *= factor_mult
                risk_details[factor] = factor_mult
        
        # Estimation ajustée par les risques
        adjusted_estimate = base_estimate * risk_multiplier
        
        # Incertitude selon le type de tâche
        uncertainty = self.uncertainty_factors.get(task_type, 0.30)
        
        # Simulation Monte Carlo
        simulations = []
        for _ in range(self.n_simulations):
            # Distribution log-normale pour éviter les valeurs négatives
            # et capturer l'asymétrie naturelle des estimations
            random_factor = self.random_state.lognormal(
                mean=0,
                sigma=uncertainty
            )
            simulated_duration = adjusted_estimate * random_factor
            simulations.append(simulated_duration)
        
        simulations = np.array(simulations)
        
        # Calcul des statistiques
        mean_duration = np.mean(simulations)
        std_dev = np.std(simulations)
        
        # Percentiles pour les scénarios
        optimistic = np.percentile(simulations, 10)  # 10% de chance d'être plus rapide
        realistic = np.percentile(simulations, 50)   # Médiane
        pessimistic = np.percentile(simulations, 90) # 90% de chance d'être plus rapide
        
        # Intervalles de confiance
        ci_50 = (np.percentile(simulations, 25), np.percentile(simulations, 75))
        ci_90 = (np.percentile(simulations, 5), np.percentile(simulations, 95))
        
        return TaskEstimate(
            task_name=task_name,
            optimistic=optimistic,
            realistic=realistic,
            pessimistic=pessimistic,
            confidence_interval_50=ci_50,
            confidence_interval_90=ci_90,
            mean_duration=mean_duration,
            std_deviation=std_dev,
            risk_factors=risk_details
        )
    
    def estimate_project(self, project_data: Dict) -> ProjectEstimate:
        """
        Estime un projet complet avec dépendances et corrélations
        
        Args:
            project_data: Dictionnaire avec:
                - name: nom du projet
                - tasks: liste des tâches
                - dependencies: dépendances entre tâches (optionnel)
                - target_deadline: délai cible (optionnel)
                - budget_constraints: contraintes budgétaires (optionnel)
        """
        project_name = project_data.get('name', 'Projet sans nom')
        tasks_data = project_data.get('tasks', [])
        target_deadline = project_data.get('target_deadline')
        
        # Estimation de chaque tâche
        task_estimates = []
        for task_data in tasks_data:
            task_estimate = self.estimate_task(task_data)
            task_estimates.append(task_estimate)
        
        # Simulation du projet complet avec corrélations
        project_simulations = []
        
        for _ in range(self.n_simulations):
            # Facteur de corrélation globale (les projets tendent à être cohérents)
            project_correlation = self.random_state.normal(1.0, 0.15)
            project_correlation = max(0.5, min(1.5, project_correlation))  # Bornage
            
            total_duration = 0
            for task_est in task_estimates:
                # Simulation individuelle de chaque tâche
                task_factor = self.random_state.lognormal(0, 0.2)  # Variation individuelle
                combined_factor = (project_correlation + task_factor) / 2
                
                simulated_task_duration = task_est.mean_duration * combined_factor
                total_duration += simulated_task_duration
            
            project_simulations.append(total_duration)
        
        project_simulations = np.array(project_simulations)
        
        # Statistiques du projet
        total_optimistic = np.percentile(project_simulations, 10)
        total_realistic = np.percentile(project_simulations, 50)
        total_pessimistic = np.percentile(project_simulations, 90)
        expected_duration = np.mean(project_simulations)
        
        # Probabilité de respecter le délai cible
        probability_on_time = 1.0
        if target_deadline:
            probability_on_time = np.sum(project_simulations <= target_deadline) / len(project_simulations)
        
        # Analyse des risques
        risk_assessment = self._analyze_project_risks(task_estimates, project_simulations)
        
        # Simulation budgétaire
        budget_simulation = self._simulate_budget(task_estimates, project_data.get('hourly_rate', 75))
        
        return ProjectEstimate(
            project_name=project_name,
            total_optimistic=total_optimistic,
            total_realistic=total_realistic,
            total_pessimistic=total_pessimistic,
            probability_on_time=probability_on_time,
            expected_duration=expected_duration,
            tasks=task_estimates,
            risk_assessment=risk_assessment,
            budget_simulation=budget_simulation
        )
    
    def _analyze_project_risks(self, task_estimates: List[TaskEstimate], 
                              project_simulations: np.ndarray) -> Dict[str, float]:
        """Analyse les risques du projet"""
        
        # Variance du projet (instabilité)
        project_variance = np.var(project_simulations)
        project_cv = np.std(project_simulations) / np.mean(project_simulations)  # Coefficient de variation
        
        # Risques par catégorie
        high_risk_tasks = sum(1 for task in task_estimates if task.std_deviation / task.mean_duration > 0.4)
        risk_concentration = high_risk_tasks / len(task_estimates) if task_estimates else 0
        
        # Probabilité de dépassement significatif (>50%)
        mean_duration = np.mean(project_simulations)
        prob_major_overrun = np.sum(project_simulations > mean_duration * 1.5) / len(project_simulations)
        
        return {
            'overall_risk_score': min(100, project_cv * 100),  # Score de 0 à 100
            'schedule_volatility': project_cv,
            'risk_concentration': risk_concentration,
            'probability_major_overrun': prob_major_overrun,
            'high_risk_tasks_count': high_risk_tasks
        }
    
    def _simulate_budget(self, task_estimates: List[TaskEstimate], hourly_rate: float) -> Dict[str, float]:
        """Simulation budgétaire basée sur les estimations de temps"""
        
        # Coûts par scénario
        optimistic_cost = sum(task.optimistic for task in task_estimates) * hourly_rate
        realistic_cost = sum(task.realistic for task in task_estimates) * hourly_rate
        pessimistic_cost = sum(task.pessimistic for task in task_estimates) * hourly_rate
        expected_cost = sum(task.mean_duration for task in task_estimates) * hourly_rate
        
        # Simulation des coûts avec variations
        cost_simulations = []
        for _ in range(1000):  # Simulation réduite pour la performance
            total_hours = 0
            for task in task_estimates:
                # Variation aléatoire dans la plage de confiance
                random_hours = self.random_state.normal(task.mean_duration, task.std_deviation)
                random_hours = max(task.optimistic, min(task.pessimistic, random_hours))
                total_hours += random_hours
            
            # Coûts supplémentaires aléatoires (overhead, imprévus)
            overhead_factor = self.random_state.normal(1.15, 0.05)  # 15% d'overhead moyen
            total_cost = total_hours * hourly_rate * overhead_factor
            cost_simulations.append(total_cost)
        
        cost_simulations = np.array(cost_simulations)
        
        return {
            'optimistic_budget': optimistic_cost,
            'realistic_budget': realistic_cost,
            'pessimistic_budget': pessimistic_cost,
            'expected_budget': expected_cost,
            'budget_std_dev': np.std(cost_simulations),
            'prob_over_realistic': np.sum(cost_simulations > realistic_cost) / len(cost_simulations)
        }
    
    def generate_recommendations(self, project_estimate: ProjectEstimate) -> List[str]:
        """Génère des recommandations basées sur l'estimation"""
        recommendations = []
        
        # Analyse des risques
        risk_score = project_estimate.risk_assessment['overall_risk_score']
        if risk_score > 50:
            recommendations.append(
                f"Risque élevé détecté (score: {risk_score:.0f}/100). "
                "Considérez une décomposition plus fine des tâches complexes."
            )
        
        # Analyse budgétaire
        budget_risk = project_estimate.budget_simulation['prob_over_realistic']
        if budget_risk > 0.3:
            recommendations.append(
                f"Risque budgétaire: {budget_risk*100:.0f}% de chance de dépassement. "
                "Prévoir une marge de sécurité de 20-30%."
            )
        
        # Analyse des délais
        if project_estimate.probability_on_time < 0.7:
            recommendations.append(
                "Faible probabilité de respecter les délais. "
                "Réviser le planning ou réduire le périmètre."
            )
        
        # Tâches à risque
        high_risk_tasks = [
            task for task in project_estimate.tasks 
            if task.std_deviation / task.mean_duration > 0.4
        ]
        
        if high_risk_tasks:
            task_names = [task.task_name for task in high_risk_tasks[:3]]
            recommendations.append(
                f"Tâches à forte incertitude: {', '.join(task_names)}. "
                "Prévoir des prototypes ou spike solutions."
            )
        
        return recommendations
    
    def export_results(self, project_estimate: ProjectEstimate, filename: str):
        """Exporte les résultats vers un fichier JSON"""
        
        # Conversion en format sérialisable
        export_data = {
            'project_name': project_estimate.project_name,
            'summary': {
                'optimistic_duration': project_estimate.total_optimistic,
                'realistic_duration': project_estimate.total_realistic,
                'pessimistic_duration': project_estimate.total_pessimistic,
                'expected_duration': project_estimate.expected_duration,
                'probability_on_time': project_estimate.probability_on_time
            },
            'tasks': [
                {
                    'name': task.task_name,
                    'optimistic': task.optimistic,
                    'realistic': task.realistic,
                    'pessimistic': task.pessimistic,
                    'mean_duration': task.mean_duration,
                    'confidence_50': task.confidence_interval_50,
                    'confidence_90': task.confidence_interval_90,
                    'risk_factors': task.risk_factors
                }
                for task in project_estimate.tasks
            ],
            'risk_assessment': project_estimate.risk_assessment,
            'budget_simulation': project_estimate.budget_simulation,
            'recommendations': self.generate_recommendations(project_estimate),
            'export_date': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


# Fonctions utilitaires pour l'intégration avec PlannerIA

def create_sample_project() -> Dict:
    """Crée un projet d'exemple pour démonstration"""
    return {
        'name': 'Développement Module E-commerce',
        'target_deadline': 120,  # 120 heures
        'hourly_rate': 75,
        'tasks': [
            {
                'name': 'Analyse des besoins',
                'base_estimate': 16,
                'task_type': 'analysis',
                'complexity': 'medium',
                'team_experience': 'senior',
                'requirements_clarity': 'partial',
                'technology_risk': 'proven',
                'integration_complexity': 'simple'
            },
            {
                'name': 'Conception de la base de données',
                'base_estimate': 12,
                'task_type': 'analysis',
                'complexity': 'medium',
                'team_experience': 'senior',
                'requirements_clarity': 'clear',
                'technology_risk': 'proven',
                'integration_complexity': 'complex'
            },
            {
                'name': 'Développement API produits',
                'base_estimate': 32,
                'task_type': 'development',
                'complexity': 'high',
                'team_experience': 'mixed',
                'requirements_clarity': 'clear',
                'technology_risk': 'familiar',
                'integration_complexity': 'complex'
            },
            {
                'name': 'Interface utilisateur panier',
                'base_estimate': 24,
                'task_type': 'development',
                'complexity': 'medium',
                'team_experience': 'senior',
                'requirements_clarity': 'detailed',
                'technology_risk': 'proven',
                'integration_complexity': 'simple'
            },
            {
                'name': 'Système de paiement',
                'base_estimate': 28,
                'task_type': 'integration',
                'complexity': 'very_high',
                'team_experience': 'mixed',
                'requirements_clarity': 'unclear',
                'technology_risk': 'new',
                'integration_complexity': 'very_complex'
            },
            {
                'name': 'Tests unitaires et intégration',
                'base_estimate': 20,
                'task_type': 'testing',
                'complexity': 'medium',
                'team_experience': 'senior',
                'requirements_clarity': 'clear',
                'technology_risk': 'proven',
                'integration_complexity': 'complex'
            },
            {
                'name': 'Déploiement et documentation',
                'base_estimate': 8,
                'task_type': 'deployment',
                'complexity': 'low',
                'team_experience': 'expert',
                'requirements_clarity': 'detailed',
                'technology_risk': 'proven',
                'integration_complexity': 'simple'
            }
        ]
    }


def quick_estimate(base_hours: float, complexity: str = 'medium', 
                  team_experience: str = 'mixed') -> Dict:
    """Estimation rapide pour intégration dans PlannerIA"""
    
    estimator = MonteCarloEstimator(n_simulations=1000)  # Simulation réduite pour la rapidité
    
    task_data = {
        'name': 'Estimation rapide',
        'base_estimate': base_hours,
        'task_type': 'development',
        'complexity': complexity,
        'team_experience': team_experience,
        'requirements_clarity': 'partial',
        'technology_risk': 'familiar',
        'integration_complexity': 'simple'
    }
    
    task_estimate = estimator.estimate_task(task_data)
    
    return {
        'optimistic': round(task_estimate.optimistic, 1),
        'realistic': round(task_estimate.realistic, 1),
        'pessimistic': round(task_estimate.pessimistic, 1),
        'confidence_range': f"{task_estimate.confidence_interval_50[0]:.1f}h - {task_estimate.confidence_interval_50[1]:.1f}h"
    }


if __name__ == "__main__":
    # Test du système
    print("Test du système d'estimation Monte Carlo")
    print("=" * 50)
    
    estimator = MonteCarloEstimator()
    
    # Test projet d'exemple
    sample_project = create_sample_project()
    project_estimate = estimator.estimate_project(sample_project)
    
    print(f"Projet: {project_estimate.project_name}")
    print(f"Durée optimiste: {project_estimate.total_optimistic:.1f}h")
    print(f"Durée réaliste: {project_estimate.total_realistic:.1f}h") 
    print(f"Durée pessimiste: {project_estimate.total_pessimistic:.1f}h")
    print(f"Probabilité respect délais: {project_estimate.probability_on_time:.1%}")
    
    recommendations = estimator.generate_recommendations(project_estimate)
    print("\nRecommandations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # Export des résultats
    estimator.export_results(project_estimate, "estimation_monte_carlo.json")
    print("\nRésultats exportés vers 'estimation_monte_carlo.json'")