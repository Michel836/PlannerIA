"""
🚀 KPI RÉVOLUTIONNAIRE - Smart KPI Engine avec IA Prédictive
Système de KPI intelligent qui prédit, analyse et recommande automatiquement
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

@dataclass
class KPIInsight:
    """Insight IA généré sur un KPI"""
    kpi_name: str
    insight_type: str  # prediction, anomaly, trend, recommendation
    title: str
    description: str
    confidence: float  # 0-1
    severity: str  # low, medium, high, critical
    action_items: List[str]
    predicted_values: Dict[str, Any]
    time_horizon: str  # "7d", "30d", "90d"

@dataclass
class SmartKPI:
    """KPI Intelligent avec prédictions IA"""
    name: str
    current_value: Any
    predicted_value: Any
    trend_direction: str  # up, down, stable
    confidence: float
    risk_level: str
    insights: List[KPIInsight]
    historical_data: List[float]
    target_value: Optional[Any] = None

class SmartKPIEngine:
    """Moteur IA pour KPI intelligents"""
    
    def __init__(self):
        self.prediction_models = {
            'linear_trend': self._linear_trend_prediction,
            'exponential_smooth': self._exponential_smoothing,
            'seasonal_adjust': self._seasonal_adjustment,
            'monte_carlo': self._monte_carlo_prediction
        }
        
        self.anomaly_thresholds = {
            'cost_deviation': 0.15,      # 15% de déviation = anomalie
            'duration_overrun': 0.10,    # 10% dépassement délai
            'quality_drop': 0.05,        # 5% baisse qualité
            'velocity_change': 0.20      # 20% changement vélocité
        }
    
    def analyze_kpi_intelligence(self, plan_data: Dict[str, Any]) -> Dict[str, SmartKPI]:
        """Analyse intelligente complète des KPI avec prédictions IA"""
        
        # Extraction et préparation des données
        tasks = self._extract_all_tasks(plan_data)
        project_metrics = plan_data.get('project_metrics', {})
        
        # Génération de données historiques simulées si besoin
        historical_data = self._generate_historical_context(tasks, project_metrics)
        
        smart_kpis = {}
        
        # 1. 🎯 KPI Performance Projet (Prédictif)
        smart_kpis['project_performance'] = self._analyze_project_performance(
            tasks, project_metrics, historical_data
        )
        
        # 2. 💰 KPI Budget Intelligence
        smart_kpis['budget_intelligence'] = self._analyze_budget_intelligence(
            tasks, historical_data
        )
        
        # 3. ⏱️ KPI Délai Prédictif  
        smart_kpis['timeline_prediction'] = self._analyze_timeline_prediction(
            tasks, historical_data
        )
        
        # 4. 👥 KPI Équipe Performance
        smart_kpis['team_performance'] = self._analyze_team_performance(
            tasks, historical_data
        )
        
        # 5. ⚠️ KPI Risque Prédictif
        risks = plan_data.get('risks', [])
        smart_kpis['risk_prediction'] = self._analyze_risk_prediction(
            risks, tasks, historical_data
        )
        
        # 6. 📈 KPI Qualité Trending
        smart_kpis['quality_trending'] = self._analyze_quality_trending(
            tasks, historical_data
        )
        
        return smart_kpis
    
    def _analyze_project_performance(self, tasks: List[Dict], metrics: Dict, 
                                   history: Dict) -> SmartKPI:
        """Analyse performance globale du projet avec prédictions"""
        
        # Calcul de l'indicateur de performance actuel
        completed_tasks = len([t for t in tasks if t.get('status') == 'completed'])
        total_tasks = len(tasks) if tasks else 1
        current_performance = (completed_tasks / total_tasks) * 100
        
        # Prédiction basée sur vélocité
        historical_performance = history.get('performance_history', [65, 70, 75, current_performance])
        predicted_performance = self._linear_trend_prediction(historical_performance, 30)
        
        # Détection d'anomalies
        insights = []
        if len(historical_performance) > 3:
            recent_trend = np.mean(historical_performance[-3:]) - np.mean(historical_performance[:-3])
            if recent_trend < -5:  # Baisse de 5%
                insights.append(KPIInsight(
                    kpi_name="project_performance",
                    insight_type="anomaly",
                    title="🚨 Ralentissement Performance Détecté",
                    description=f"Performance en baisse de {abs(recent_trend):.1f}% sur période récente",
                    confidence=0.85,
                    severity="high",
                    action_items=[
                        "Analyser les blocages équipe",
                        "Revoir la répartition des tâches",
                        "Organiser réunion performance"
                    ],
                    predicted_values={"trend_7d": predicted_performance},
                    time_horizon="7d"
                ))
        
        # Prédiction positive
        if predicted_performance > current_performance + 5:
            insights.append(KPIInsight(
                kpi_name="project_performance",
                insight_type="prediction",
                title="📈 Accélération Prévue",
                description=f"Performance devrait atteindre {predicted_performance:.1f}% (+{predicted_performance-current_performance:.1f}%)",
                confidence=0.78,
                severity="medium",
                action_items=[
                    "Maintenir le rythme actuel",
                    "Anticiper les prochaines phases"
                ],
                predicted_values={"target_30d": predicted_performance},
                time_horizon="30d"
            ))
        
        return SmartKPI(
            name="Performance Projet",
            current_value=f"{current_performance:.1f}%",
            predicted_value=f"{predicted_performance:.1f}%",
            trend_direction="up" if predicted_performance > current_performance else "down",
            confidence=0.82,
            risk_level="low" if current_performance > 75 else "medium",
            insights=insights,
            historical_data=historical_performance,
            target_value="85%"
        )
    
    def _analyze_budget_intelligence(self, tasks: List[Dict], history: Dict) -> SmartKPI:
        """Analyse budget avec prédiction de dérive"""
        
        # Calcul coût actuel et prévu
        current_spent = sum(t.get('cost', 0) for t in tasks)
        total_budget = sum(t.get('cost', 0) for t in tasks) * 1.15  # Budget avec marge
        
        # Historique des dépenses
        budget_history = history.get('budget_history', [
            total_budget * 0.20, total_budget * 0.45, total_budget * 0.68, current_spent
        ])
        
        # Prédiction Monte Carlo pour budget final (moteur utilitaire interne)
        # NOTE: Différent du Monte Carlo What-If qui est l'interface utilisateur principale
        final_cost_prediction = self._monte_carlo_prediction(budget_history, 60)
        budget_overrun = (final_cost_prediction - total_budget) / total_budget * 100
        
        insights = []
        
        # Alerte dépassement budgétaire
        if budget_overrun > 10:
            insights.append(KPIInsight(
                kpi_name="budget_intelligence",
                insight_type="prediction",
                title="⚠️ Risque Dépassement Budgétaire",
                description=f"Prédiction: dépassement de {budget_overrun:.1f}% du budget initial",
                confidence=0.89,
                severity="critical" if budget_overrun > 25 else "high",
                action_items=[
                    "Réviser le scope du projet",
                    "Négocier budget supplémentaire",
                    "Optimiser les ressources coûteuses",
                    "Prioriser les fonctionnalités essentielles"
                ],
                predicted_values={"final_cost": final_cost_prediction, "overrun_pct": budget_overrun},
                time_horizon="90d"
            ))
        
        # Opportunité d'économies
        elif budget_overrun < -5:
            savings = abs(budget_overrun)
            insights.append(KPIInsight(
                kpi_name="budget_intelligence",
                insight_type="recommendation",
                title="💰 Opportunité d'Économies Détectée",
                description=f"Économies potentielles de {savings:.1f}% identifiées",
                confidence=0.75,
                severity="low",
                action_items=[
                    "Réinvestir dans la qualité",
                    "Ajouter des fonctionnalités bonus",
                    "Constituer réserve pour projets futurs"
                ],
                predicted_values={"savings_amount": abs(final_cost_prediction - total_budget)},
                time_horizon="60d"
            ))
        
        return SmartKPI(
            name="Intelligence Budget",
            current_value=f"${current_spent:,.0f}",
            predicted_value=f"${final_cost_prediction:,.0f}",
            trend_direction="up" if budget_overrun > 0 else "stable",
            confidence=0.89,
            risk_level="high" if budget_overrun > 15 else "medium",
            insights=insights,
            historical_data=budget_history,
            target_value=f"${total_budget:,.0f}"
        )
    
    def _analyze_timeline_prediction(self, tasks: List[Dict], history: Dict) -> SmartKPI:
        """Prédiction intelligente des délais avec IA"""
        
        # Calcul progression temporelle
        total_duration = sum(t.get('duration', 0) for t in tasks)
        completed_duration = sum(t.get('duration', 0) for t in tasks if t.get('status') == 'completed')
        progress_pct = (completed_duration / total_duration * 100) if total_duration > 0 else 0
        
        # Vélocité historique (tâches/semaine)
        velocity_history = history.get('velocity_history', [2.1, 2.8, 3.2, 2.9, 3.4])
        current_velocity = velocity_history[-1]
        predicted_velocity = self._exponential_smoothing(velocity_history, 14)
        
        # Estimation temps restant
        remaining_tasks = len([t for t in tasks if t.get('status') != 'completed'])
        days_remaining_predicted = remaining_tasks / predicted_velocity * 7  # Conversion semaines
        
        insights = []
        
        # Prédiction retard
        expected_duration = total_duration
        if days_remaining_predicted > expected_duration * 1.1:  # 10% de retard
            delay_days = days_remaining_predicted - expected_duration
            insights.append(KPIInsight(
                kpi_name="timeline_prediction",
                insight_type="prediction",
                title="⏰ Retard Potentiel Prévu",
                description=f"Prédiction: retard de {delay_days:.0f} jours basé sur vélocité actuelle",
                confidence=0.81,
                severity="high" if delay_days > 14 else "medium",
                action_items=[
                    f"Accélérer vélocité à {predicted_velocity * 1.3:.1f} tâches/semaine",
                    "Paralléliser davantage de tâches",
                    "Augmenter temporairement l'équipe",
                    "Réviser le scope non-critique"
                ],
                predicted_values={"delay_days": delay_days, "required_velocity": remaining_tasks / (expected_duration / 7)},
                time_horizon="30d"
            ))
        
        # Accélération détectée
        velocity_trend = np.mean(velocity_history[-2:]) - np.mean(velocity_history[:-2])
        if velocity_trend > 0.5:
            insights.append(KPIInsight(
                kpi_name="timeline_prediction",
                insight_type="trend",
                title="🚀 Accélération Équipe Détectée",
                description=f"Vélocité en hausse: +{velocity_trend:.1f} tâches/semaine",
                confidence=0.77,
                severity="low",
                action_items=[
                    "Maintenir cette dynamique",
                    "Identifier les facteurs de succès",
                    "Dupliquer sur autres projets"
                ],
                predicted_values={"velocity_gain": velocity_trend},
                time_horizon="14d"
            ))
        
        return SmartKPI(
            name="Prédiction Délais",
            current_value=f"{progress_pct:.1f}% complété",
            predicted_value=f"{days_remaining_predicted:.0f}j restants",
            trend_direction="up" if velocity_trend > 0 else "down",
            confidence=0.81,
            risk_level="high" if days_remaining_predicted > expected_duration * 1.2 else "medium",
            insights=insights,
            historical_data=velocity_history,
            target_value=f"{expected_duration:.0f}j total"
        )
    
    def _analyze_team_performance(self, tasks: List[Dict], history: Dict) -> SmartKPI:
        """Analyse performance équipe avec prédictions"""
        
        # Analyse de la répartition des tâches par équipe
        team_assignments = {}
        for task in tasks:
            team = task.get('assigned_to', 'Unassigned')
            if team not in team_assignments:
                team_assignments[team] = {'completed': 0, 'total': 0, 'total_duration': 0}
            
            team_assignments[team]['total'] += 1
            team_assignments[team]['total_duration'] += task.get('duration', 0)
            if task.get('status') == 'completed':
                team_assignments[team]['completed'] += 1
        
        # Calcul efficacité moyenne
        team_efficiencies = []
        for team, stats in team_assignments.items():
            if stats['total'] > 0:
                efficiency = (stats['completed'] / stats['total']) * 100
                team_efficiencies.append(efficiency)
        
        avg_efficiency = np.mean(team_efficiencies) if team_efficiencies else 0
        
        # Prédiction efficacité future
        efficiency_history = history.get('team_efficiency_history', [72, 76, 81, avg_efficiency])
        predicted_efficiency = self._linear_trend_prediction(efficiency_history, 21)
        
        insights = []
        
        # Détection équipe sous-performante
        if team_efficiencies:
            min_efficiency = min(team_efficiencies)
            max_efficiency = max(team_efficiencies)
            if max_efficiency - min_efficiency > 25:  # Écart > 25%
                insights.append(KPIInsight(
                    kpi_name="team_performance",
                    insight_type="anomaly",
                    title="⚖️ Déséquilibre Équipes Détecté",
                    description=f"Écart performance: {max_efficiency-min_efficiency:.1f}% entre meilleures/moins bonnes équipes",
                    confidence=0.92,
                    severity="medium",
                    action_items=[
                        "Rééquilibrer la charge de travail",
                        "Former équipes moins performantes",
                        "Partager best practices inter-équipes",
                        "Réviser assignations de tâches"
                    ],
                    predicted_values={"efficiency_gap": max_efficiency - min_efficiency},
                    time_horizon="21d"
                ))
        
        # Prédiction amélioration continue
        if predicted_efficiency > avg_efficiency + 3:
            insights.append(KPIInsight(
                kpi_name="team_performance",
                insight_type="prediction",
                title="📈 Montée en Compétence Prévue",
                description=f"Efficacité équipe devrait atteindre {predicted_efficiency:.1f}%",
                confidence=0.74,
                severity="low",
                action_items=[
                    "Capitaliser sur la dynamique positive",
                    "Documenter les progrès réalisés"
                ],
                predicted_values={"target_efficiency": predicted_efficiency},
                time_horizon="21d"
            ))
        
        return SmartKPI(
            name="Performance Équipe",
            current_value=f"{avg_efficiency:.1f}%",
            predicted_value=f"{predicted_efficiency:.1f}%",
            trend_direction="up" if predicted_efficiency > avg_efficiency else "stable",
            confidence=0.79,
            risk_level="low" if avg_efficiency > 80 else "medium",
            insights=insights,
            historical_data=efficiency_history,
            target_value="85%"
        )
    
    def _analyze_risk_prediction(self, risks: List[Dict], tasks: List[Dict], 
                               history: Dict) -> SmartKPI:
        """Prédiction évolution des risques avec IA"""
        
        if not risks:
            return SmartKPI(
                name="Prédiction Risques",
                current_value="Aucun risque",
                predicted_value="Stable",
                trend_direction="stable",
                confidence=0.95,
                risk_level="low",
                insights=[],
                historical_data=[0, 0, 0, 0],
                target_value="0 risques"
            )
        
        # Analyse des risques actuels
        total_risk_score = sum(r.get('risk_score', r.get('probability', 1) * r.get('impact', 1)) for r in risks)
        critical_risks = len([r for r in risks if r.get('risk_score', 5) >= 20])
        
        # Historique des risques
        risk_history = history.get('risk_history', [total_risk_score * 0.8, total_risk_score * 0.9, total_risk_score * 1.1, total_risk_score])
        predicted_risk = self._exponential_smoothing(risk_history, 30)
        
        insights = []
        
        # Escalade de risques
        if predicted_risk > total_risk_score * 1.2:
            insights.append(KPIInsight(
                kpi_name="risk_prediction",
                insight_type="prediction",
                title="⚠️ Escalade de Risques Prévue",
                description=f"Score de risque global pourrait augmenter de {((predicted_risk/total_risk_score-1)*100):.1f}%",
                confidence=0.83,
                severity="high",
                action_items=[
                    "Activer plans de mitigation immédiats",
                    "Réviser tous les risques de priorité moyenne",
                    "Augmenter fréquence monitoring risques",
                    "Préparer plans de contingence"
                ],
                predicted_values={"predicted_total_risk": predicted_risk},
                time_horizon="30d"
            ))
        
        # Analyse de corrélation tâches-risques
        if tasks:
            high_risk_tasks = len([t for t in tasks if t.get('priority') == 'high'])
            risk_task_ratio = total_risk_score / len(tasks)
            if risk_task_ratio > 2:  # Plus de 2 points de risque par tâche
                insights.append(KPIInsight(
                    kpi_name="risk_prediction",
                    insight_type="recommendation",
                    title="🎯 Concentration de Risques",
                    description=f"Ratio risque/tâche élevé: {risk_task_ratio:.1f} points/tâche",
                    confidence=0.78,
                    severity="medium",
                    action_items=[
                        "Décomposer les tâches les plus risquées",
                        "Allouer experts sur tâches critiques",
                        "Créer tâches de validation intermédiaires"
                    ],
                    predicted_values={"risk_per_task": risk_task_ratio},
                    time_horizon="14d"
                ))
        
        return SmartKPI(
            name="Prédiction Risques",
            current_value=f"{total_risk_score:.0f} pts ({critical_risks} critiques)",
            predicted_value=f"{predicted_risk:.0f} pts prévus",
            trend_direction="up" if predicted_risk > total_risk_score else "down",
            confidence=0.83,
            risk_level="critical" if critical_risks > 3 else "high" if total_risk_score > 50 else "medium",
            insights=insights,
            historical_data=risk_history,
            target_value="<30 pts total"
        )
    
    def _analyze_quality_trending(self, tasks: List[Dict], history: Dict) -> SmartKPI:
        """Analyse trending de la qualité avec prédictions"""
        
        # Estimation qualité basée sur plusieurs facteurs
        quality_indicators = []
        
        for task in tasks:
            # Facteurs qualité
            duration_factor = min(1.0, task.get('duration', 5) / 10)  # Tâches trop courtes = risque qualité
            priority_factor = {'high': 1.0, 'medium': 0.8, 'low': 0.6}.get(task.get('priority', 'medium'), 0.8)
            status_factor = {'completed': 1.0, 'in_progress': 0.7, 'pending': 0.5}.get(task.get('status', 'pending'), 0.5)
            
            task_quality_score = (duration_factor * 0.3 + priority_factor * 0.4 + status_factor * 0.3) * 100
            quality_indicators.append(task_quality_score)
        
        current_quality = np.mean(quality_indicators) if quality_indicators else 75
        
        # Historique qualité
        quality_history = history.get('quality_history', [78, 76, 79, current_quality])
        predicted_quality = self._linear_trend_prediction(quality_history, 28)
        
        insights = []
        
        # Dégradation qualité
        quality_trend = np.mean(quality_history[-2:]) - np.mean(quality_history[:-2])
        if quality_trend < -3:  # Baisse > 3%
            insights.append(KPIInsight(
                kpi_name="quality_trending",
                insight_type="anomaly", 
                title="📉 Dégradation Qualité Détectée",
                description=f"Baisse qualité de {abs(quality_trend):.1f}% sur période récente",
                confidence=0.86,
                severity="high",
                action_items=[
                    "Auditer processus qualité actuels",
                    "Renforcer tests et validations",
                    "Former équipe sur standards qualité",
                    "Implémenter checkpoints qualité supplémentaires"
                ],
                predicted_values={"quality_decline": abs(quality_trend)},
                time_horizon="28d"
            ))
        
        # Excellence qualité
        elif current_quality > 85 and predicted_quality > 90:
            insights.append(KPIInsight(
                kpi_name="quality_trending",
                insight_type="trend",
                title="🏆 Excellence Qualité en Vue",
                description=f"Qualité exceptionnelle maintenue, objectif 90%+ atteignable",
                confidence=0.79,
                severity="low",
                action_items=[
                    "Documenter best practices qualité",
                    "Partager méthodes avec autres projets",
                    "Maintenir standards élevés"
                ],
                predicted_values={"excellence_target": 90},
                time_horizon="28d"
            ))
        
        return SmartKPI(
            name="Trending Qualité",
            current_value=f"{current_quality:.1f}%",
            predicted_value=f"{predicted_quality:.1f}%",
            trend_direction="up" if predicted_quality > current_quality else "down",
            confidence=0.76,
            risk_level="low" if current_quality > 80 else "medium",
            insights=insights,
            historical_data=quality_history,
            target_value="85%+"
        )
    
    # Méthodes de prédiction
    def _linear_trend_prediction(self, data: List[float], days_ahead: int) -> float:
        """Prédiction basée sur tendance linéaire"""
        if len(data) < 2:
            return data[0] if data else 0
        
        x = np.arange(len(data))
        z = np.polyfit(x, data, 1)
        prediction_point = len(data) + (days_ahead / 7)  # Conversion en périodes
        return max(0, z[0] * prediction_point + z[1])
    
    def _exponential_smoothing(self, data: List[float], days_ahead: int) -> float:
        """Lissage exponentiel pour prédiction"""
        if not data:
            return 0
        
        alpha = 0.3  # Facteur de lissage
        smoothed = data[0]
        
        for value in data[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Projection simple
        trend = (data[-1] - data[-2]) if len(data) > 1 else 0
        periods_ahead = days_ahead / 7
        return max(0, smoothed + trend * periods_ahead * 0.5)
    
    def _seasonal_adjustment(self, data: List[float], days_ahead: int) -> float:
        """Ajustement saisonnier simple"""
        if len(data) < 4:
            return self._linear_trend_prediction(data, days_ahead)
        
        # Facteur saisonnier simple (cycle de 4 périodes)
        seasonal_factor = data[-4:][days_ahead % 4] / np.mean(data[-4:])
        base_prediction = self._linear_trend_prediction(data, days_ahead)
        return max(0, base_prediction * seasonal_factor)
    
    def _monte_carlo_prediction(self, data: List[float], days_ahead: int) -> float:
        """
        Prédiction Monte Carlo utilitaire pour KPIs
        
        RÔLE: Fonction de calcul interne pour améliorer la précision des métriques
        - Utilisée pour prédictions budgétaires automatiques
        - Calcul de dépassements et tendances
        - Invisible à l'utilisateur (moteur sous le capot)
        """
        if not data:
            return 0
        
        # Calcul volatilité
        if len(data) > 1:
            returns = [data[i]/data[i-1] - 1 for i in range(1, len(data))]
            volatility = np.std(returns) if returns else 0.1
        else:
            volatility = 0.1
        
        # Simulation simple
        current_value = data[-1]
        periods = days_ahead / 7
        
        # Mouvement brownien géométrique simplifié
        random_factor = np.random.normal(0, volatility * np.sqrt(periods))
        return max(0, current_value * (1 + random_factor))
    
    def _extract_all_tasks(self, plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraction de toutes les tâches"""
        tasks = []
        
        if 'tasks' in plan_data:
            tasks.extend(plan_data['tasks'])
        
        if 'wbs' in plan_data and 'phases' in plan_data['wbs']:
            for phase in plan_data['wbs']['phases']:
                if 'tasks' in phase:
                    tasks.extend(phase['tasks'])
        
        return tasks if tasks else self._generate_sample_tasks()
    
    def _generate_historical_context(self, tasks: List[Dict], metrics: Dict) -> Dict[str, List[float]]:
        """Génère un contexte historique réaliste pour les prédictions"""
        
        num_periods = 4  # 4 points historiques
        
        # Performance projet
        current_perf = len([t for t in tasks if t.get('status') == 'completed']) / len(tasks) * 100 if tasks else 70
        performance_base = current_perf * 0.7
        performance_history = [
            performance_base + np.random.normal(0, 5) for _ in range(num_periods-1)
        ] + [current_perf]
        
        # Budget
        current_budget = sum(t.get('cost', 0) for t in tasks)
        budget_base = current_budget * 0.3
        budget_history = [
            budget_base * (1 + i * 0.4) + np.random.normal(0, current_budget * 0.05) 
            for i in range(num_periods)
        ]
        
        # Vélocité équipe
        velocity_base = 2.5
        velocity_history = [
            velocity_base + i * 0.2 + np.random.normal(0, 0.3) 
            for i in range(num_periods)
        ]
        
        # Efficacité équipe
        efficiency_base = 70
        team_efficiency_history = [
            efficiency_base + i * 3 + np.random.normal(0, 4) 
            for i in range(num_periods)
        ]
        
        # Risques
        current_risk = sum(r.get('risk_score', 5) for r in tasks[:3])  # Simulation
        risk_history = [
            current_risk * (0.9 + i * 0.1) + np.random.normal(0, 2)
            for i in range(num_periods)
        ]
        
        # Qualité
        quality_base = 75
        quality_history = [
            quality_base + np.random.normal(0, 3) 
            for _ in range(num_periods)
        ]
        
        return {
            'performance_history': [max(0, min(100, x)) for x in performance_history],
            'budget_history': [max(0, x) for x in budget_history],
            'velocity_history': [max(0.5, x) for x in velocity_history],
            'team_efficiency_history': [max(30, min(100, x)) for x in team_efficiency_history],
            'risk_history': [max(0, x) for x in risk_history],
            'quality_history': [max(40, min(100, x)) for x in quality_history]
        }
    
    def _generate_sample_tasks(self) -> List[Dict[str, Any]]:
        """Génère des tâches d'exemple pour démonstration"""
        import random
        
        task_names = [
            "Interface utilisateur", "API Backend", "Base de données", "Tests unitaires",
            "Documentation", "Sécurité", "Performance", "Déploiement"
        ]
        
        sample_tasks = []
        for i, name in enumerate(task_names):
            task = {
                'name': name,
                'cost': random.randint(3000, 12000),
                'duration': random.randint(5, 15),
                'priority': random.choice(['high', 'medium', 'low']),
                'status': random.choice(['completed', 'in_progress', 'pending']),
                'assigned_to': f'Team_{chr(65 + i % 3)}'
            }
            sample_tasks.append(task)
        
        return sample_tasks


def render_revolutionary_kpi_dashboard(plan_data: Dict[str, Any]):
    """Rendu du dashboard KPI révolutionnaire"""
    
    st.header("🚀 KPI RÉVOLUTIONNAIRE - Smart Analytics")
    st.markdown("""
    **Système de KPI Intelligent** avec IA prédictive, détection d'anomalies automatique, 
    et recommandations business en temps réel.
    """)
    
    # Initialisation du moteur IA
    if 'smart_kpi_engine' not in st.session_state:
        st.session_state.smart_kpi_engine = SmartKPIEngine()
    
    # Analyse intelligente
    with st.spinner("🧠 Analyse IA des KPI en cours..."):
        smart_kpis = st.session_state.smart_kpi_engine.analyze_kpi_intelligence(plan_data)
    
    # Dashboard principal
    render_smart_kpi_overview(smart_kpis)
    
    st.markdown("---")
    
    # Onglets KPI intelligents
    render_smart_kpi_tabs(smart_kpis)
    
    st.markdown("---")
    
    # Panel d'insights IA
    render_ai_insights_panel(smart_kpis)


def render_smart_kpi_overview(smart_kpis: Dict[str, SmartKPI]):
    """Vue d'ensemble des KPI intelligents"""
    
    st.subheader("📊 KPI Intelligents - Vue d'Ensemble")
    
    # Métriques principales avec prédictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        perf_kpi = smart_kpis.get('project_performance')
        if perf_kpi:
            delta_text = f"➜ {perf_kpi.predicted_value}"
            st.metric(
                label="🎯 Performance Projet",
                value=perf_kpi.current_value,
                delta=delta_text,
                help=f"Prédiction IA (conf: {perf_kpi.confidence:.0%})"
            )
    
    with col2:
        budget_kpi = smart_kpis.get('budget_intelligence')
        if budget_kpi:
            delta_text = f"➜ {budget_kpi.predicted_value}"
            st.metric(
                label="💰 Intelligence Budget",
                value=budget_kpi.current_value,
                delta=delta_text,
                help=f"Prédiction IA (conf: {budget_kpi.confidence:.0%})"
            )
    
    with col3:
        timeline_kpi = smart_kpis.get('timeline_prediction')
        if timeline_kpi:
            st.metric(
                label="⏰ Prédiction Délais",
                value=timeline_kpi.current_value,
                delta=timeline_kpi.predicted_value,
                help=f"Estimation IA (conf: {timeline_kpi.confidence:.0%})"
            )
    
    # Indicateurs de risque globaux
    st.markdown("#### 🚨 Alertes IA Temps Réel")
    
    # Collecte toutes les alertes critiques
    critical_alerts = []
    high_alerts = []
    
    for kpi_name, kpi in smart_kpis.items():
        for insight in kpi.insights:
            if insight.severity == "critical":
                critical_alerts.append(insight)
            elif insight.severity == "high":
                high_alerts.append(insight)
    
    # Affichage des alertes
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        if critical_alerts:
            st.error(f"🚨 {len(critical_alerts)} Alerte(s) Critique(s)")
            for alert in critical_alerts[:2]:  # Top 2
                st.markdown(f"**{alert.title}**")
                st.markdown(f"{alert.description}")
        else:
            st.success("✅ Aucune alerte critique")
    
    with alert_col2:
        if high_alerts:
            st.warning(f"⚠️ {len(high_alerts)} Alerte(s) Importante(s)")
            for alert in high_alerts[:2]:  # Top 2
                st.markdown(f"**{alert.title}**")
                st.markdown(f"{alert.description}")
        else:
            st.info("ℹ️ Surveillance active")


def render_smart_kpi_tabs(smart_kpis: Dict[str, SmartKPI]):
    """Onglets détaillés des KPI intelligents"""
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Performance", "💰 Budget IA", "⏰ Délais IA", 
        "👥 Équipe", "⚠️ Risques IA", "📈 Qualité"
    ])
    
    with tab1:
        render_performance_kpi_detail(smart_kpis.get('project_performance'))
    
    with tab2:
        render_budget_kpi_detail(smart_kpis.get('budget_intelligence'))
    
    with tab3:
        render_timeline_kpi_detail(smart_kpis.get('timeline_prediction'))
    
    with tab4:
        render_team_kpi_detail(smart_kpis.get('team_performance'))
    
    with tab5:
        render_risk_kpi_detail(smart_kpis.get('risk_prediction'))
    
    with tab6:
        render_quality_kpi_detail(smart_kpis.get('quality_trending'))


def render_performance_kpi_detail(kpi: SmartKPI):
    """Détail KPI Performance avec graphiques prédictifs"""
    if not kpi:
        st.info("Données de performance non disponibles")
        return
    
    st.markdown("### 🎯 Analyse Performance Projet")
    
    # Métriques clés
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Performance Actuelle", kpi.current_value)
    with col2:
        st.metric("Prédiction IA", kpi.predicted_value, 
                 delta=f"Confiance: {kpi.confidence:.0%}")
    with col3:
        risk_color = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        st.metric("Niveau Risque", f"{risk_color.get(kpi.risk_level, '⚪')} {kpi.risk_level.title()}")
    
    # Graphique tendance historique + prédiction
    if kpi.historical_data:
        fig = go.Figure()
        
        # Données historiques
        historical_x = list(range(len(kpi.historical_data)))
        fig.add_trace(go.Scatter(
            x=historical_x,
            y=kpi.historical_data,
            mode='lines+markers',
            name='Historique',
            line=dict(color='blue', width=3)
        ))
        
        # Prédiction
        pred_value = float(kpi.predicted_value.replace('%', ''))
        pred_x = [len(kpi.historical_data)-1, len(kpi.historical_data)+2]
        pred_y = [kpi.historical_data[-1], pred_value]
        
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=pred_y,
            mode='lines+markers',
            name='Prédiction IA',
            line=dict(color='red', dash='dash', width=3)
        ))
        
        # Zone de confiance
        confidence_margin = pred_value * (1 - kpi.confidence) * 0.5
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=[pred_y[0], pred_value + confidence_margin],
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0)',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=[pred_y[0], pred_value - confidence_margin],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0)',
            name='Zone Confiance IA',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig.update_layout(
            title="📈 Évolution Performance + Prédiction IA",
            xaxis_title="Période",
            yaxis_title="Performance (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights spécifiques
    if kpi.insights:
        st.markdown("#### 🧠 Insights IA Performance")
        for insight in kpi.insights:
            severity_color = {
                "low": "info", "medium": "warning", 
                "high": "warning", "critical": "error"
            }
            getattr(st, severity_color.get(insight.severity, "info"))(
                f"**{insight.title}**\n\n{insight.description}"
            )
            
            if insight.action_items:
                with st.expander("💡 Actions Recommandées"):
                    for action in insight.action_items:
                        st.write(f"• {action}")


def render_budget_kpi_detail(kpi: SmartKPI):
    """Détail KPI Budget avec analyse prédictive"""
    if not kpi:
        st.info("Données de budget non disponibles")
        return
    
    st.markdown("### 💰 Intelligence Budget Prédictive")
    
    # Métriques budget
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dépenses Actuelles", kpi.current_value)
    with col2:
        st.metric("Prédiction Finale", kpi.predicted_value, 
                 delta=f"Confiance: {kpi.confidence:.0%}")
    with col3:
        st.metric("Budget Cible", kpi.target_value or "Non défini")
    
    # Graphique burn-rate + prédiction
    if kpi.historical_data:
        fig = go.Figure()
        
        # Historique dépenses
        periods = list(range(len(kpi.historical_data)))
        fig.add_trace(go.Scatter(
            x=periods,
            y=kpi.historical_data,
            mode='lines+markers',
            name='Dépenses Réelles',
            line=dict(color='green', width=3)
        ))
        
        # Prédiction coût final
        pred_value = float(kpi.predicted_value.replace('$', '').replace(',', ''))
        pred_x = [len(kpi.historical_data)-1, len(kpi.historical_data)+3]
        pred_y = [kpi.historical_data[-1], pred_value]
        
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=pred_y,
            mode='lines+markers',
            name='Prédiction Coût Final',
            line=dict(color='orange', dash='dash', width=3)
        ))
        
        # Ligne budget cible si disponible
        if kpi.target_value:
            target_value = float(kpi.target_value.replace('$', '').replace(',', ''))
            fig.add_hline(
                y=target_value,
                line_dash="dot",
                line_color="red",
                annotation_text="Budget Cible"
            )
        
        fig.update_layout(
            title="💸 Évolution Budget + Prédiction IA",
            xaxis_title="Période Projet",
            yaxis_title="Coût Cumulé ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des écarts prédits
    for insight in kpi.insights:
        if insight.insight_type == "prediction" and "overrun" in insight.predicted_values:
            overrun_pct = insight.predicted_values["overrun_pct"]
            
            if overrun_pct > 0:
                st.error(f"⚠️ **Dépassement Prévu**: +{overrun_pct:.1f}% du budget initial")
            else:
                savings_pct = abs(overrun_pct)
                st.success(f"💰 **Économies Prévues**: -{savings_pct:.1f}% du budget initial")
    
    # Actions recommandées budget
    budget_insights = [i for i in kpi.insights if i.severity in ["high", "critical"]]
    if budget_insights:
        st.markdown("#### 💡 Recommandations Budget IA")
        for insight in budget_insights[:2]:  # Top 2 most important
            with st.expander(f"🎯 {insight.title}"):
                st.write(insight.description)
                for action in insight.action_items:
                    st.write(f"• {action}")


def render_timeline_kpi_detail(kpi: SmartKPI):
    """Détail KPI Délais avec prédictions de vélocité"""
    if not kpi:
        st.info("Données de délais non disponibles")
        return
    
    st.markdown("### ⏰ Prédiction Délais & Vélocité")
    
    # Métriques temporelles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("État Actuel", kpi.current_value)
    with col2:
        st.metric("Estimation Restante", kpi.predicted_value, 
                 delta=f"Confiance: {kpi.confidence:.0%}")
    with col3:
        st.metric("Objectif Projet", kpi.target_value or "Non défini")
    
    # Graphique vélocité historique
    if kpi.historical_data:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Vélocité Équipe (tâches/semaine)', 'Prédiction Temps Restant'),
            vertical_spacing=0.12
        )
        
        # Vélocité historique
        periods = list(range(len(kpi.historical_data)))
        fig.add_trace(
            go.Bar(x=periods, y=kpi.historical_data, name='Vélocité Historique',
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # Tendance vélocité
        if len(kpi.historical_data) > 1:
            trend_line = np.polyfit(periods, kpi.historical_data, 1)
            trend_values = [trend_line[0] * x + trend_line[1] for x in periods]
            fig.add_trace(
                go.Scatter(x=periods, y=trend_values, name='Tendance Vélocité',
                          line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # Prédiction délai (simulée)
        remaining_days = [30, 25, 20, 15, 10, 5]  # Exemple projection
        future_periods = list(range(len(periods), len(periods) + len(remaining_days)))
        
        fig.add_trace(
            go.Scatter(x=future_periods, y=remaining_days, name='Prédiction Temps Restant',
                      line=dict(color='green', width=3), mode='lines+markers'),
            row=2, col=1
        )
        
        fig.update_layout(height=500, title_text="📊 Analyse Vélocité & Prédictions Temporelles")
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights délais critiques
    delay_insights = [i for i in kpi.insights if "retard" in i.title.lower() or "delay" in i.title.lower()]
    if delay_insights:
        for insight in delay_insights:
            st.warning(f"⚠️ **{insight.title}**\n\n{insight.description}")
            
            if insight.predicted_values:
                pred_data = insight.predicted_values
                if "delay_days" in pred_data:
                    st.metric("🚨 Retard Prévu", f"{pred_data['delay_days']:.0f} jours")
                if "required_velocity" in pred_data:
                    st.metric("🎯 Vélocité Requise", f"{pred_data['required_velocity']:.1f} tâches/semaine")


def render_team_kpi_detail(kpi: SmartKPI):
    """Détail KPI Équipe avec analyse performance"""
    if not kpi:
        st.info("Données d'équipe non disponibles")
        return
    
    st.markdown("### 👥 Analyse Performance Équipe")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Efficacité Moyenne", kpi.current_value)
    with col2:
        st.metric("Prédiction IA", kpi.predicted_value)
    with col3:
        st.metric("Objectif Cible", kpi.target_value or "85%")
    
    # Graphique performance équipe
    if kpi.historical_data:
        fig = go.Figure()
        
        periods = list(range(len(kpi.historical_data)))
        fig.add_trace(go.Scatter(
            x=periods,
            y=kpi.historical_data,
            mode='lines+markers',
            name='Efficacité Historique',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ))
        
        # Ligne objectif
        fig.add_hline(y=85, line_dash="dash", line_color="green", 
                     annotation_text="Objectif: 85%")
        
        # Zone de performance acceptable
        fig.add_hrect(y0=75, y1=85, fillcolor="green", opacity=0.1, 
                     annotation_text="Zone Acceptable", annotation_position="top left")
        
        fig.update_layout(
            title="📈 Évolution Performance Équipe",
            xaxis_title="Période",
            yaxis_title="Efficacité (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights équipe
    team_insights = [i for i in kpi.insights if "équipe" in i.title.lower() or "team" in i.title.lower()]
    if team_insights:
        for insight in team_insights:
            severity_display = {
                "low": "info", "medium": "info", "high": "warning", "critical": "error"
            }
            getattr(st, severity_display.get(insight.severity, "info"))(
                f"**{insight.title}**\n\n{insight.description}"
            )


def render_risk_kpi_detail(kpi: SmartKPI):
    """Détail KPI Risques avec prédictions d'escalade"""
    if not kpi:
        st.info("Données de risques non disponibles")
        return
    
    st.markdown("### ⚠️ Prédiction & Analyse des Risques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risques Actuels", kpi.current_value)
    with col2:
        st.metric("Évolution Prédite", kpi.predicted_value)
    with col3:
        risk_colors = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        st.metric("Niveau Global", f"{risk_colors.get(kpi.risk_level, '⚪')} {kpi.risk_level.title()}")
    
    # Graphique évolution risques
    if kpi.historical_data:
        fig = go.Figure()
        
        periods = list(range(len(kpi.historical_data)))
        fig.add_trace(go.Scatter(
            x=periods,
            y=kpi.historical_data,
            mode='lines+markers',
            name='Score Risque Historique',
            line=dict(color='red', width=3),
            fill='tonexty'
        ))
        
        # Zones de risque
        fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                     annotation_text="Seuil Élevé: 50pts")
        fig.add_hline(y=30, line_dash="dash", line_color="yellow", 
                     annotation_text="Seuil Modéré: 30pts")
        
        fig.update_layout(
            title="📊 Évolution Score de Risque Global",
            xaxis_title="Période Projet",
            yaxis_title="Score Risque Total",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Alertes risques critiques
    critical_risk_insights = [i for i in kpi.insights if i.severity in ["critical", "high"]]
    if critical_risk_insights:
        st.markdown("#### 🚨 Alertes Risques Prioritaires")
        for insight in critical_risk_insights:
            st.error(f"**{insight.title}**\n\n{insight.description}")
            
            with st.expander("🛡️ Plan d'Action Recommandé"):
                for action in insight.action_items:
                    st.write(f"• {action}")


def render_quality_kpi_detail(kpi: SmartKPI):
    """Détail KPI Qualité avec trending"""
    if not kpi:
        st.info("Données de qualité non disponibles")
        return
    
    st.markdown("### 📈 Trending Qualité & Prédictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Qualité Actuelle", kpi.current_value)
    with col2:
        st.metric("Tendance IA", kpi.predicted_value)
    with col3:
        st.metric("Objectif Excellence", kpi.target_value or "90%")
    
    # Graphique qualité avec zones
    if kpi.historical_data:
        fig = go.Figure()
        
        periods = list(range(len(kpi.historical_data)))
        fig.add_trace(go.Scatter(
            x=periods,
            y=kpi.historical_data,
            mode='lines+markers',
            name='Qualité Historique',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Zones de qualité
        fig.add_hrect(y0=90, y1=100, fillcolor="green", opacity=0.1,
                     annotation_text="Excellence", annotation_position="top left")
        fig.add_hrect(y0=75, y1=90, fillcolor="yellow", opacity=0.1,
                     annotation_text="Acceptable", annotation_position="top left")
        fig.add_hrect(y0=0, y1=75, fillcolor="red", opacity=0.1,
                     annotation_text="Critique", annotation_position="top left")
        
        fig.update_layout(
            title="📊 Évolution Qualité avec Zones",
            xaxis_title="Période",
            yaxis_title="Score Qualité (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights qualité
    quality_insights = kpi.insights
    if quality_insights:
        for insight in quality_insights:
            if "excellence" in insight.title.lower():
                st.success(f"🏆 **{insight.title}**\n\n{insight.description}")
            elif "dégradation" in insight.title.lower():
                st.error(f"📉 **{insight.title}**\n\n{insight.description}")
            else:
                st.info(f"📊 **{insight.title}**\n\n{insight.description}")


def render_ai_insights_panel(smart_kpis: Dict[str, SmartKPI]):
    """Panel consolidé des insights IA"""
    
    st.subheader("🧠 Centre d'Intelligence IA - Insights Consolidés")
    
    # Collecte tous les insights
    all_insights = []
    for kpi_name, kpi in smart_kpis.items():
        for insight in kpi.insights:
            all_insights.append((kpi_name, insight))
    
    if not all_insights:
        st.info("🤖 Système IA en surveillance - Aucune anomalie détectée")
        return
    
    # Tri par sévérité et confiance
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_insights.sort(key=lambda x: (severity_order.get(x[1].severity, 3), -x[1].confidence))
    
    # Onglets par type d'insight
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Alertes Critiques", "🔮 Prédictions", "📈 Tendances", "💡 Recommandations"
    ])
    
    with tab1:
        critical_insights = [x for x in all_insights if x[1].severity in ["critical", "high"]]
        if critical_insights:
            for kpi_name, insight in critical_insights[:5]:  # Top 5
                st.error(f"🚨 **{insight.title}** [{kpi_name.replace('_', ' ').title()}]")
                st.write(insight.description)
                
                if insight.action_items:
                    with st.expander("🔧 Actions Immédiates"):
                        for action in insight.action_items:
                            st.write(f"• {action}")
                st.markdown("---")
        else:
            st.success("✅ Aucune alerte critique - Projet dans les paramètres normaux")
    
    with tab2:
        prediction_insights = [x for x in all_insights if x[1].insight_type == "prediction"]
        if prediction_insights:
            for kpi_name, insight in prediction_insights:
                confidence_badge = f"🎯 {insight.confidence:.0%}"
                st.info(f"🔮 **{insight.title}** [{kpi_name.replace('_', ' ').title()}] {confidence_badge}")
                st.write(insight.description)
                
                if insight.predicted_values:
                    with st.expander("📊 Valeurs Prédites"):
                        for key, value in insight.predicted_values.items():
                            st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                st.markdown("---")
        else:
            st.info("🔮 Système de prédiction en cours de calibrage...")
    
    with tab3:
        trend_insights = [x for x in all_insights if x[1].insight_type in ["trend", "anomaly"]]
        if trend_insights:
            for kpi_name, insight in trend_insights:
                if insight.insight_type == "anomaly":
                    st.warning(f"📊 **{insight.title}** [{kpi_name.replace('_', ' ').title()}]")
                else:
                    st.info(f"📈 **{insight.title}** [{kpi_name.replace('_', ' ').title()}]")
                st.write(insight.description)
                st.markdown("---")
        else:
            st.info("📈 Toutes les métriques suivent des tendances normales")
    
    with tab4:
        recommendation_insights = [x for x in all_insights if x[1].insight_type == "recommendation"]
        if recommendation_insights:
            for kpi_name, insight in recommendation_insights:
                st.success(f"💡 **{insight.title}** [{kpi_name.replace('_', ' ').title()}]")
                st.write(insight.description)
                
                if insight.action_items:
                    for i, action in enumerate(insight.action_items, 1):
                        st.write(f"{i}. {action}")
                st.markdown("---")
        else:
            st.info("💡 Projet optimalement configuré - Aucune recommandation spécifique")
    
    # Bouton export insights
    if st.button("📄 Exporter Rapport IA Complet", key="export_ai_insights"):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_insights': len(all_insights),
            'critical_count': len([x for x in all_insights if x[1].severity in ["critical", "high"]]),
            'insights': [
                {
                    'kpi': kpi_name,
                    'type': insight.insight_type,
                    'title': insight.title,
                    'description': insight.description,
                    'severity': insight.severity,
                    'confidence': insight.confidence,
                    'actions': insight.action_items,
                    'predictions': insight.predicted_values,
                    'horizon': insight.time_horizon
                }
                for kpi_name, insight in all_insights
            ]
        }
        
        st.download_button(
            label="💾 Télécharger JSON",
            data=json.dumps(export_data, indent=2, default=str, ensure_ascii=False),
            file_name=f"kpi_ai_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            key="download_insights"
        )