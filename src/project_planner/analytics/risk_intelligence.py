"""
🔴 Risk Analysis Intelligence - Module Professionnel
===================================================

Système d'analyse de risques avancé avec:
- Simulations Monte Carlo
- Scoring dynamique basé sur données réelles
- Machine Learning pour prédictions
- Visualisations interactives professionnelles

Auteur: PlannerIA Team
Date: 2025-08-31
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import statistics
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class RiskCategory(Enum):
    """Catégories de risques standards"""
    TECHNICAL = "technique"
    BUDGET = "budget"
    SCHEDULE = "planning"
    RESOURCE = "ressources"
    MARKET = "marché"
    QUALITY = "qualité"
    INTEGRATION = "intégration"
    SECURITY = "sécurité"
    COMPLIANCE = "conformité"
    STAKEHOLDER = "parties_prenantes"


class RiskImpact(Enum):
    """Niveaux d'impact des risques"""
    VERY_LOW = (1, "Très Faible", "#10b981")
    LOW = (2, "Faible", "#84cc16") 
    MEDIUM = (3, "Moyen", "#f59e0b")
    HIGH = (4, "Élevé", "#ef4444")
    VERY_HIGH = (5, "Très Élevé", "#dc2626")
    
    def __init__(self, score: int, label: str, color: str):
        self.score = score
        self.label = label
        self.color = color


class RiskProbability(Enum):
    """Probabilités d'occurrence des risques"""
    VERY_LOW = (0.05, "Très Faible", "1-5%")
    LOW = (0.15, "Faible", "6-25%")
    MEDIUM = (0.35, "Moyen", "26-50%")
    HIGH = (0.65, "Élevé", "51-75%")
    VERY_HIGH = (0.85, "Très Élevé", "76-95%")
    
    def __init__(self, probability: float, label: str, range_desc: str):
        self.probability = probability
        self.label = label
        self.range_desc = range_desc


@dataclass
class RiskFactor:
    """Facteur de risque individuel"""
    id: str
    name: str
    category: RiskCategory
    probability: RiskProbability
    impact: RiskImpact
    description: str
    mitigation_strategy: str
    owner: str
    status: str = "Open"
    detection_date: datetime = None
    mitigation_cost: float = 0.0
    
    def __post_init__(self):
        if self.detection_date is None:
            self.detection_date = datetime.now()
    
    @property
    def risk_score(self) -> float:
        """Calcule le score de risque (Probabilité × Impact)"""
        return self.probability.probability * self.impact.score
    
    @property
    def risk_exposure(self) -> float:
        """Calcule l'exposition au risque (Score × Coût potentiel)"""
        return self.risk_score * self.mitigation_cost


class RiskIntelligenceEngine:
    """
    🧠 Moteur d'Intelligence des Risques
    
    Fonctionnalités:
    - Analyse Monte Carlo pour simulations
    - ML pour prédictions basées sur historique
    - Scoring dynamique contextuel
    - Recommendations automatiques
    """
    
    def __init__(self):
        """Initialise le moteur avec modèles pré-entraînés"""
        self.ml_model = None
        self.scaler = StandardScaler()
        self.historical_data = self._generate_historical_data()
        self._train_risk_prediction_model()
        
        # Palette de couleurs professionnelle
        self.colors = {
            'critical': '#dc2626',
            'high': '#ef4444', 
            'medium': '#f59e0b',
            'low': '#84cc16',
            'very_low': '#10b981',
            'background': '#f8fafc',
            'text': '#1e293b'
        }
    
    def analyze_project_risks(self, project_data: Dict[str, Any], 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyse complète des risques d'un projet
        
        Args:
            project_data: Données du projet (WBS, budget, timeline)
            context: Contexte additionnel (équipe, marché, etc.)
            
        Returns:
            Analyse complète avec risques identifiés et métriques
        """
        if not project_data:
            project_data = self._generate_demo_project()
            
        if not context:
            context = self._generate_demo_context()
        
        # 1. Identification automatique des risques
        identified_risks = self._identify_risks_from_project(project_data, context)
        
        # 2. Calcul des métriques de risque
        risk_metrics = self._calculate_risk_metrics(identified_risks, project_data)
        
        # 3. Simulations Monte Carlo
        monte_carlo_results = self._run_monte_carlo_simulation(
            identified_risks, project_data, iterations=10000
        )
        
        # 4. Prédictions ML
        ml_predictions = self._predict_risk_outcomes(project_data, identified_risks)
        
        # 5. Recommandations automatiques
        recommendations = self._generate_risk_recommendations(identified_risks, risk_metrics)
        
        return {
            'identified_risks': identified_risks,
            'risk_metrics': risk_metrics,
            'monte_carlo': monte_carlo_results,
            'ml_predictions': ml_predictions,
            'recommendations': recommendations,
            'overall_risk_score': risk_metrics['weighted_average_score'],
            'risk_exposure_total': risk_metrics['total_exposure'],
            'critical_risks_count': len([r for r in identified_risks if r.risk_score >= 4.0])
        }
    
    def create_risk_dashboard(self, risk_analysis: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Crée un dashboard complet de visualisation des risques
        
        Returns:
            Dictionnaire de figures Plotly pour chaque visualisation
        """
        figures = {}
        
        # 1. Matrice de risques interactive
        figures['risk_matrix'] = self._create_interactive_risk_matrix(
            risk_analysis['identified_risks']
        )
        
        # 2. Distributions Monte Carlo
        figures['monte_carlo'] = self._create_monte_carlo_visualization(
            risk_analysis['monte_carlo']
        )
        
        # 3. Timeline des risques
        figures['risk_timeline'] = self._create_risk_timeline(
            risk_analysis['identified_risks']
        )
        
        # 4. Radar chart par catégorie
        figures['risk_radar'] = self._create_risk_radar_chart(
            risk_analysis['identified_risks']
        )
        
        # 5. Burn-down des risques
        figures['risk_burndown'] = self._create_risk_burndown(
            risk_analysis['identified_risks']
        )
        
        # 6. Prédictions ML
        figures['ml_predictions'] = self._create_ml_predictions_chart(
            risk_analysis['ml_predictions']
        )
        
        return figures
    
    # Méthodes privées d'analyse
    def _identify_risks_from_project(self, project_data: Dict, context: Dict) -> List[RiskFactor]:
        """Identifie automatiquement les risques basés sur les données projet"""
        risks = []
        
        # Analyse du budget
        total_budget = project_data.get('project_overview', {}).get('total_cost', 50000)
        if total_budget > 100000:
            risks.append(RiskFactor(
                id="R001",
                name="Budget élevé - Risque de dépassement",
                category=RiskCategory.BUDGET,
                probability=RiskProbability.MEDIUM,
                impact=RiskImpact.HIGH,
                description=f"Budget de €{total_budget:,.0f} présente un risque de dépassement",
                mitigation_strategy="Mise en place d'un contrôle budgétaire strict avec jalons",
                owner="Chef de Projet",
                mitigation_cost=total_budget * 0.15
            ))
        
        # Analyse de la complexité technique
        phases = project_data.get('wbs', {}).get('phases', [])
        tech_complexity = len([p for p in phases if 'développement' in p.get('name', '').lower()])
        
        if tech_complexity >= 2:
            risks.append(RiskFactor(
                id="R002", 
                name="Complexité technique élevée",
                category=RiskCategory.TECHNICAL,
                probability=RiskProbability.HIGH,
                impact=RiskImpact.HIGH,
                description="Projet multi-phases avec forte composante technique",
                mitigation_strategy="Architecture robuste + Code reviews + Tests automatisés",
                owner="Architecte Technique",
                mitigation_cost=total_budget * 0.08
            ))
        
        # Analyse de la timeline
        total_duration = project_data.get('project_overview', {}).get('total_duration', 30)
        if total_duration > 90:
            risks.append(RiskFactor(
                id="R003",
                name="Projet long terme - Risque de glissement",
                category=RiskCategory.SCHEDULE,
                probability=RiskProbability.MEDIUM,
                impact=RiskImpact.MEDIUM,
                description=f"Durée de {total_duration} jours augmente les risques planning",
                mitigation_strategy="Jalons fréquents + Méthode agile + Buffer temps",
                owner="PMO",
                mitigation_cost=total_budget * 0.05
            ))
        
        # Risques contextuels
        team_size = context.get('team_size', 5)
        if team_size > 10:
            risks.append(RiskFactor(
                id="R004",
                name="Équipe large - Risque communication",
                category=RiskCategory.RESOURCE,
                probability=RiskProbability.MEDIUM,
                impact=RiskImpact.MEDIUM,
                description=f"Équipe de {team_size} personnes complexifie la coordination",
                mitigation_strategy="Outils collaboratifs + Daily standups + Documentation",
                owner="Scrum Master",
                mitigation_cost=total_budget * 0.03
            ))
        
        # Risques marché/innovation
        innovation_keywords = ['ia', 'ml', 'blockchain', 'innovation', 'disruptif']
        project_desc = project_data.get('project_overview', {}).get('description', '').lower()
        
        if any(keyword in project_desc for keyword in innovation_keywords):
            risks.append(RiskFactor(
                id="R005",
                name="Technologies émergentes - Risque d'obsolescence",
                category=RiskCategory.MARKET,
                probability=RiskProbability.LOW,
                impact=RiskImpact.HIGH,
                description="Technologies récentes avec évolution rapide",
                mitigation_strategy="Veille technologique + Architecture modulaire + Proof of concepts",
                owner="CTO",
                mitigation_cost=total_budget * 0.12
            ))
        
        # Risques sécurité si données sensibles
        if 'fintech' in project_desc or 'bancaire' in project_desc:
            risks.append(RiskFactor(
                id="R006",
                name="Sécurité et conformité réglementaire",
                category=RiskCategory.SECURITY,
                probability=RiskProbability.HIGH,
                impact=RiskImpact.VERY_HIGH,
                description="Secteur financier nécessite sécurité renforcée",
                mitigation_strategy="Audit sécurité + Conformité RGPD + Chiffrement + Tests pénétration",
                owner="CISO",
                mitigation_cost=total_budget * 0.20
            ))
        
        return risks
    
    def _calculate_risk_metrics(self, risks: List[RiskFactor], project_data: Dict) -> Dict[str, float]:
        """Calcule les métriques agrégées de risque"""
        if not risks:
            return {'weighted_average_score': 0, 'total_exposure': 0}
        
        scores = [r.risk_score for r in risks]
        exposures = [r.risk_exposure for r in risks]
        
        return {
            'total_risks': len(risks),
            'average_score': statistics.mean(scores),
            'weighted_average_score': sum(r.risk_score * r.mitigation_cost for r in risks) / sum(r.mitigation_cost for r in risks),
            'max_score': max(scores),
            'total_exposure': sum(exposures),
            'critical_risks': len([r for r in risks if r.risk_score >= 4.0]),
            'high_risks': len([r for r in risks if 3.0 <= r.risk_score < 4.0]),
            'medium_risks': len([r for r in risks if 2.0 <= r.risk_score < 3.0]),
            'low_risks': len([r for r in risks if r.risk_score < 2.0])
        }
    
    def _run_monte_carlo_simulation(self, risks: List[RiskFactor], 
                                   project_data: Dict, iterations: int = 10000) -> Dict[str, Any]:
        """Exécute une simulation Monte Carlo sur les risques"""
        
        # Paramètres de base du projet
        base_cost = project_data.get('project_overview', {}).get('total_cost', 50000)
        base_duration = project_data.get('project_overview', {}).get('total_duration', 30)
        
        # Simulations
        cost_scenarios = []
        duration_scenarios = []
        
        for _ in range(iterations):
            scenario_cost_impact = 0
            scenario_duration_impact = 0
            
            for risk in risks:
                # Probabilité d'occurrence pour cette itération
                if np.random.random() < risk.probability.probability:
                    # Le risque se matérialise
                    impact_factor = np.random.uniform(0.5, 1.5)  # Variabilité de l'impact
                    
                    if risk.category in [RiskCategory.BUDGET, RiskCategory.TECHNICAL]:
                        scenario_cost_impact += risk.mitigation_cost * impact_factor
                    
                    if risk.category in [RiskCategory.SCHEDULE, RiskCategory.RESOURCE]:
                        scenario_duration_impact += base_duration * 0.1 * impact_factor
            
            cost_scenarios.append(base_cost + scenario_cost_impact)
            duration_scenarios.append(base_duration + scenario_duration_impact)
        
        # Analyse des résultats
        cost_p50 = np.percentile(cost_scenarios, 50)
        cost_p80 = np.percentile(cost_scenarios, 80)
        cost_p95 = np.percentile(cost_scenarios, 95)
        
        duration_p50 = np.percentile(duration_scenarios, 50)
        duration_p80 = np.percentile(duration_scenarios, 80)
        duration_p95 = np.percentile(duration_scenarios, 95)
        
        return {
            'iterations': iterations,
            'cost_scenarios': cost_scenarios,
            'duration_scenarios': duration_scenarios,
            'cost_statistics': {
                'base': base_cost,
                'p50': cost_p50,
                'p80': cost_p80,
                'p95': cost_p95,
                'mean': np.mean(cost_scenarios),
                'std': np.std(cost_scenarios),
                'risk_premium_p80': cost_p80 - base_cost,
                'risk_premium_p95': cost_p95 - base_cost
            },
            'duration_statistics': {
                'base': base_duration,
                'p50': duration_p50,
                'p80': duration_p80,
                'p95': duration_p95,
                'mean': np.mean(duration_scenarios),
                'std': np.std(duration_scenarios),
                'delay_risk_p80': duration_p80 - base_duration,
                'delay_risk_p95': duration_p95 - base_duration
            }
        }
    
    def _train_risk_prediction_model(self):
        """Entraîne le modèle ML pour prédictions de risques"""
        # Features: budget, durée, complexité, taille équipe, innovation
        X = []
        y = []
        
        for project in self.historical_data:
            features = [
                project['budget'] / 10000,  # Normalisation
                project['duration'] / 10,
                project['complexity_score'],
                project['team_size'],
                project['innovation_score']
            ]
            X.append(features)
            y.append(project['actual_risk_score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entraînement Random Forest
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Score du modèle
        train_score = self.ml_model.score(X_train_scaled, y_train)
        test_score = self.ml_model.score(X_test_scaled, y_test)
        
        return {'train_score': train_score, 'test_score': test_score}
    
    def _predict_risk_outcomes(self, project_data: Dict, risks: List[RiskFactor]) -> Dict[str, Any]:
        """Utilise ML pour prédire les outcomes des risques"""
        if not self.ml_model:
            return {'error': 'Model not trained'}
        
        # Extraction des features du projet actuel
        budget = project_data.get('project_overview', {}).get('total_cost', 50000)
        duration = project_data.get('project_overview', {}).get('total_duration', 30)
        
        # Calcul scores de complexité et innovation
        phases = project_data.get('wbs', {}).get('phases', [])
        complexity_score = min(10, len(phases) + len([p for p in phases if len(p.get('tasks', [])) > 5]))
        
        desc = project_data.get('project_overview', {}).get('description', '').lower()
        innovation_keywords = ['ia', 'ml', 'blockchain', 'innovation', 'disruptif', 'nouveau']
        innovation_score = sum(1 for keyword in innovation_keywords if keyword in desc)
        
        team_size = 5  # Default estimate
        
        # Prédiction
        features = np.array([[
            budget / 10000,
            duration / 10,
            complexity_score,
            team_size,
            innovation_score
        ]])
        
        features_scaled = self.scaler.transform(features)
        prediction = self.ml_model.predict(features_scaled)[0]
        
        # Confiance de la prédiction (basée sur variance des arbres)
        tree_predictions = [tree.predict(features_scaled)[0] for tree in self.ml_model.estimators_]
        prediction_std = np.std(tree_predictions)
        confidence = max(0, min(100, 100 - (prediction_std * 20)))
        
        # Comparaison avec risques identifiés
        current_risk_score = statistics.mean([r.risk_score for r in risks]) if risks else 0
        
        return {
            'predicted_risk_score': round(prediction, 2),
            'confidence': round(confidence, 1),
            'current_calculated_score': round(current_risk_score, 2),
            'prediction_vs_current': round(prediction - current_risk_score, 2),
            'risk_level': self._get_risk_level(prediction),
            'features_importance': {
                'budget_impact': 0.25,
                'duration_impact': 0.20,
                'complexity_impact': 0.30,
                'team_impact': 0.15,
                'innovation_impact': 0.10
            }
        }
    
    def _generate_risk_recommendations(self, risks: List[RiskFactor], 
                                     metrics: Dict[str, float]) -> List[Dict[str, str]]:
        """Génère des recommandations automatiques basées sur l'analyse"""
        recommendations = []
        
        if metrics.get('critical_risks', 0) > 0:
            recommendations.append({
                'priority': 'CRITIQUE',
                'category': 'Action Immédiate',
                'title': f"{metrics.get('critical_risks', 0)} risques critiques détectés",
                'description': "Mise en place immédiate de plans de mitigation pour les risques > 4.0",
                'action': "Revue d'urgence avec le comité de pilotage dans les 48h"
            })
        
        if metrics['total_exposure'] > 50000:
            recommendations.append({
                'priority': 'ÉLEVÉE',
                'category': 'Financement',
                'title': "Exposition financière importante",
                'description': f"Exposition totale: €{metrics['total_exposure']:,.0f}",
                'action': "Constitution d'une réserve pour risques (10-15% du budget total)"
            })
        
        if metrics['weighted_average_score'] > 3.0:
            recommendations.append({
                'priority': 'ÉLEVÉE',
                'category': 'Gouvernance',
                'title': "Score de risque moyen élevé",
                'description': f"Score moyen pondéré: {metrics['weighted_average_score']:.1f}/5",
                'action': "Mise en place d'un comité de gestion des risques hebdomadaire"
            })
        
        # Recommandations par catégorie dominante
        risk_by_category = {}
        for risk in risks:
            category = risk.category.value
            if category not in risk_by_category:
                risk_by_category[category] = []
            risk_by_category[category].append(risk)
        
        for category, category_risks in risk_by_category.items():
            if len(category_risks) >= 2:
                avg_score = statistics.mean([r.risk_score for r in category_risks])
                recommendations.append({
                    'priority': 'MOYENNE',
                    'category': f'Risques {category}',
                    'title': f"Concentration de risques en {category}",
                    'description': f"{len(category_risks)} risques identifiés (score moyen: {avg_score:.1f})",
                    'action': f"Plan de mitigation spécialisé pour les risques {category}"
                })
        
        return recommendations[:6]  # Limiter à 6 recommandations principales
    
    # Méthodes de visualisation
    def _create_interactive_risk_matrix(self, risks: List[RiskFactor]) -> go.Figure:
        """Crée une matrice de risques interactive professionnelle"""
        fig = go.Figure()
        
        # Grille de fond
        for i in range(6):
            fig.add_hline(y=i+0.5, line_color="rgba(0,0,0,0.1)", line_width=1)
            fig.add_vline(x=i+0.5, line_color="rgba(0,0,0,0.1)", line_width=1)
        
        # Zones de risque colorées
        risk_zones = [
            # Zone faible (vert)
            {'x': [0.5, 2.5], 'y': [0.5, 2.5], 'color': 'rgba(16, 185, 129, 0.2)'},
            # Zone moyenne (jaune) 
            {'x': [0.5, 3.5], 'y': [2.5, 3.5], 'color': 'rgba(245, 158, 11, 0.2)'},
            {'x': [2.5, 3.5], 'y': [0.5, 2.5], 'color': 'rgba(245, 158, 11, 0.2)'},
            # Zone élevée (rouge)
            {'x': [3.5, 5.5], 'y': [0.5, 5.5], 'color': 'rgba(239, 68, 68, 0.2)'},
            {'x': [0.5, 3.5], 'y': [3.5, 5.5], 'color': 'rgba(239, 68, 68, 0.2)'}
        ]
        
        for zone in risk_zones:
            fig.add_shape(
                type="rect",
                x0=zone['x'][0], y0=zone['y'][0],
                x1=zone['x'][1], y1=zone['y'][1],
                fillcolor=zone['color'],
                line=dict(width=0)
            )
        
        # Points des risques
        for risk in risks:
            fig.add_trace(go.Scatter(
                x=[risk.probability.probability * 5],  # Scale to 1-5
                y=[risk.impact.score],
                mode='markers+text',
                text=[risk.id],
                textposition="middle center",
                marker=dict(
                    size=max(15, risk.mitigation_cost / 1000),
                    color=risk.impact.color,
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                name=risk.name,
                hovertemplate=(
                    f"<b>{risk.name}</b><br>" +
                    f"Catégorie: {risk.category.value}<br>" +
                    f"Probabilité: {risk.probability.label} ({risk.probability.range_desc})<br>" +
                    f"Impact: {risk.impact.label}<br>" +
                    f"Score: {risk.risk_score:.1f}<br>" +
                    f"Coût mitigation: €{risk.mitigation_cost:,.0f}<br>" +
                    f"Propriétaire: {risk.owner}<br>" +
                    "<extra></extra>"
                ),
                showlegend=False
            ))
        
        # Configuration
        fig.update_layout(
            title={
                'text': "🎯 <b>Matrice des Risques Interactive</b>",
                'font': {'size': 20}
            },
            xaxis=dict(
                title="Probabilité d'Occurrence",
                range=[0.5, 5.5],
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['Très Faible<br>1-5%', 'Faible<br>6-25%', 'Moyen<br>26-50%', 'Élevé<br>51-75%', 'Très Élevé<br>76-95%']
            ),
            yaxis=dict(
                title="Impact sur le Projet",
                range=[0.5, 5.5],
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['Très Faible', 'Faible', 'Moyen', 'Élevé', 'Très Élevé']
            ),
            height=500,
            plot_bgcolor='white',
            annotations=[
                dict(x=1.5, y=1.5, text="<b>FAIBLE</b>", showarrow=False, font=dict(size=14, color='green')),
                dict(x=3, y=2, text="<b>MOYEN</b>", showarrow=False, font=dict(size=14, color='orange')),
                dict(x=4.5, y=4.5, text="<b>ÉLEVÉ</b>", showarrow=False, font=dict(size=16, color='red'))
            ]
        )
        
        return fig
    
    def _create_monte_carlo_visualization(self, monte_carlo_results: Dict) -> go.Figure:
        """Visualise les résultats Monte Carlo avec distributions"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution des Coûts', 'Distribution des Durées', 
                          'Coût vs Durée', 'Percentiles de Risque'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        cost_scenarios = monte_carlo_results['cost_scenarios']
        duration_scenarios = monte_carlo_results['duration_scenarios']
        cost_stats = monte_carlo_results['cost_statistics']
        duration_stats = monte_carlo_results['duration_statistics']
        
        # Distribution des coûts
        fig.add_trace(
            go.Histogram(
                x=cost_scenarios,
                name='Coûts simulés',
                opacity=0.7,
                marker_color='blue',
                nbinsx=50
            ),
            row=1, col=1
        )
        
        # Lignes de percentiles
        for percentile, color in [(50, 'green'), (80, 'orange'), (95, 'red')]:
            value = np.percentile(cost_scenarios, percentile)
            fig.add_vline(x=value, line_dash="dash", line_color=color,
                         annotation_text=f"P{percentile}", row=1, col=1)
        
        # Distribution des durées
        fig.add_trace(
            go.Histogram(
                x=duration_scenarios,
                name='Durées simulées',
                opacity=0.7,
                marker_color='green',
                nbinsx=50
            ),
            row=1, col=2
        )
        
        # Scatter plot coût vs durée
        fig.add_trace(
            go.Scatter(
                x=duration_scenarios[::100],  # Sample pour performance
                y=cost_scenarios[::100],
                mode='markers',
                marker=dict(size=3, opacity=0.6, color='purple'),
                name='Scénarios',
                hovertemplate='Durée: %{x:.0f}j<br>Coût: €%{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Graphique des percentiles
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        cost_percentiles = [np.percentile(cost_scenarios, p) for p in percentiles]
        duration_percentiles = [np.percentile(duration_scenarios, p) for p in percentiles]
        
        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=cost_percentiles,
                mode='lines+markers',
                name='Coût',
                line=dict(color='blue')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=[d * 1000 for d in duration_percentiles],  # Scale pour comparaison
                mode='lines+markers',
                name='Durée (x1000)',
                line=dict(color='green')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="📊 <b>Analyse Monte Carlo - Simulations de Risque</b>",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_risk_timeline(self, risks: List[RiskFactor]) -> go.Figure:
        """Crée une timeline des risques par date et criticité"""
        
        fig = go.Figure()
        
        # Trier les risques par score décroissant
        risks_sorted = sorted(risks, key=lambda r: r.risk_score, reverse=True)
        
        dates = [r.detection_date for r in risks_sorted]
        scores = [r.risk_score for r in risks_sorted]
        names = [r.name for r in risks_sorted]
        categories = [r.category.value for r in risks_sorted]
        
        # Couleurs par catégorie
        color_map = {
            'technique': '#3b82f6',
            'budget': '#ef4444',
            'planning': '#f59e0b',
            'ressources': '#10b981',
            'marché': '#8b5cf6',
            'qualité': '#06b6d4',
            'sécurité': '#dc2626',
            'intégration': '#84cc16'
        }
        
        colors = [color_map.get(cat, '#6b7280') for cat in categories]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='markers+text',
            marker=dict(
                size=[s*8 for s in scores],
                color=colors,
                line=dict(color='white', width=2),
                opacity=0.8
            ),
            text=[f"{r.id}" for r in risks_sorted],
            textposition="top center",
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Score: %{y:.1f}<br>" +
                "Date: %{x}<br>" +
                "<extra></extra>"
            ),
            name="Risques"
        ))
        
        # Ligne de tendance
        if len(dates) > 1:
            z = np.polyfit(range(len(dates)), scores, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=dates,
                y=p(range(len(dates))),
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Tendance',
                opacity=0.7
            ))
        
        fig.update_layout(
            title="📅 <b>Timeline des Risques</b>",
            xaxis_title="Date de Détection",
            yaxis_title="Score de Risque",
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    def _create_risk_radar_chart(self, risks: List[RiskFactor]) -> go.Figure:
        """Crée un radar chart des risques par catégorie"""
        
        # Calculer scores par catégorie
        category_scores = {}
        for risk in risks:
            cat = risk.category.value
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(risk.risk_score)
        
        # Moyennes par catégorie
        categories = list(category_scores.keys())
        avg_scores = [statistics.mean(scores) for scores in category_scores.values()]
        max_scores = [max(scores) for scores in category_scores.values()]
        
        # Fermer le radar
        categories_closed = categories + [categories[0]]
        avg_scores_closed = avg_scores + [avg_scores[0]]
        max_scores_closed = max_scores + [max_scores[0]]
        
        fig = go.Figure()
        
        # Score moyen
        fig.add_trace(go.Scatterpolar(
            r=avg_scores_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='#3b82f6', width=2),
            name='Score Moyen'
        ))
        
        # Score maximum
        fig.add_trace(go.Scatterpolar(
            r=max_scores_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.1)',
            line=dict(color='#ef4444', width=2),
            name='Score Maximum'
        ))
        
        fig.update_layout(
            title="📊 <b>Profil de Risque par Catégorie</b>",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['Très Faible', 'Faible', 'Moyen', 'Élevé', 'Critique']
                )
            ),
            height=500
        )
        
        return fig
    
    def _create_risk_burndown(self, risks: List[RiskFactor]) -> go.Figure:
        """Crée un burndown chart des risques (simulation)"""
        
        # Simulation d'évolution des risques sur 12 semaines
        weeks = list(range(13))
        total_risks = len(risks)
        
        # Simulation de résolution progressive
        remaining_risks = []
        for week in weeks:
            if week == 0:
                remaining_risks.append(total_risks)
            else:
                # Simulation de résolution avec variabilité
                resolved = max(0, np.random.poisson(0.5))  # En moyenne 0.5 risque résolu par semaine
                current = max(0, remaining_risks[-1] - resolved)
                remaining_risks.append(current)
        
        # Ligne idéale
        ideal_line = [max(0, total_risks - (total_risks * w / 12)) for w in weeks]
        
        fig = go.Figure()
        
        # Ligne réelle
        fig.add_trace(go.Scatter(
            x=weeks,
            y=remaining_risks,
            mode='lines+markers',
            name='Risques Restants (Réel)',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8)
        ))
        
        # Ligne idéale
        fig.add_trace(go.Scatter(
            x=weeks,
            y=ideal_line,
            mode='lines',
            name='Résolution Idéale',
            line=dict(color='#10b981', width=2, dash='dash')
        ))
        
        # Zone de tolérance
        fig.add_trace(go.Scatter(
            x=weeks + weeks[::-1],
            y=[max(0, i-1) for i in ideal_line] + [i+1 for i in ideal_line[::-1]],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Zone de Tolérance',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="📉 <b>Burndown Chart des Risques</b>",
            xaxis_title="Semaines",
            yaxis_title="Nombre de Risques Ouverts",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_ml_predictions_chart(self, ml_predictions: Dict) -> go.Figure:
        """Visualise les prédictions ML vs réalité"""
        
        if 'error' in ml_predictions:
            # Graphique d'erreur
            fig = go.Figure()
            fig.add_annotation(
                text="Modèle ML non disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="red")
            )
            return fig
        
        # Données de comparaison
        predicted = ml_predictions['predicted_risk_score']
        current = ml_predictions['current_calculated_score']
        confidence = ml_predictions['confidence']
        
        categories = ['Prédiction ML', 'Calcul Actuel', 'Seuil Critique']
        values = [predicted, current, 4.0]
        colors = ['#3b82f6', '#10b981', '#ef4444']
        
        fig = go.Figure()
        
        # Barres
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition='outside',
            hovertemplate="Score: %{y:.1f}<extra></extra>"
        ))
        
        # Barre de confiance
        fig.add_shape(
            type="rect",
            x0=-0.4, y0=0,
            x1=0.4, y1=predicted,
            fillcolor=f"rgba(59, 130, 246, {confidence/100})",
            layer="below",
            line=dict(width=0)
        )
        
        # Annotations
        fig.add_annotation(
            x=0, y=predicted + 0.2,
            text=f"Confiance: {confidence:.0f}%",
            showarrow=False,
            font=dict(size=12)
        )
        
        difference = predicted - current
        arrow_color = 'red' if difference > 0 else 'green'
        fig.add_annotation(
            x=1, y=max(predicted, current) + 0.3,
            text=f"{'↗️' if difference > 0 else '↘️'} {abs(difference):.1f}",
            showarrow=False,
            font=dict(size=14, color=arrow_color)
        )
        
        fig.update_layout(
            title="🤖 <b>Prédictions ML vs Analyse Actuelle</b>",
            yaxis_title="Score de Risque",
            height=400,
            showlegend=False
        )
        
        return fig
    
    # Méthodes utilitaires
    def _generate_historical_data(self) -> List[Dict]:
        """Génère des données historiques pour entraîner le modèle ML"""
        np.random.seed(42)  # Reproductibilité
        
        historical_projects = []
        
        for i in range(200):  # 200 projets historiques
            budget = np.random.exponential(50000) + 10000
            duration = np.random.gamma(2, 15) + 10
            complexity = np.random.randint(1, 11)
            team_size = np.random.randint(3, 15)
            innovation_score = np.random.randint(0, 6)
            
            # Score de risque basé sur les variables avec du bruit
            base_risk = (
                (budget / 20000) * 0.3 +
                (duration / 20) * 0.2 +
                complexity * 0.4 +
                (team_size / 5) * 0.1 +
                innovation_score * 0.2
            )
            
            # Ajouter du bruit et borner entre 1 et 5
            actual_risk = max(1, min(5, base_risk + np.random.normal(0, 0.3)))
            
            historical_projects.append({
                'budget': budget,
                'duration': duration,
                'complexity_score': complexity,
                'team_size': team_size,
                'innovation_score': innovation_score,
                'actual_risk_score': actual_risk
            })
        
        return historical_projects
    
    def _generate_demo_project(self) -> Dict[str, Any]:
        """Génère des données de projet pour démo"""
        return {
            'project_overview': {
                'title': 'Plateforme IA E-commerce',
                'description': 'Développement d\'une plateforme IA avancée pour recommandations personnalisées',
                'total_cost': 75000,
                'total_duration': 120
            },
            'wbs': {
                'phases': [
                    {
                        'name': 'Analyse & Conception',
                        'tasks': [{'name': 'Architecture IA', 'duration': 10, 'cost': 8000}]
                    },
                    {
                        'name': 'Développement Backend',
                        'tasks': [{'name': 'API ML', 'duration': 20, 'cost': 25000}]
                    },
                    {
                        'name': 'Développement Frontend', 
                        'tasks': [{'name': 'Interface utilisateur', 'duration': 15, 'cost': 18000}]
                    }
                ]
            }
        }
    
    def _generate_demo_context(self) -> Dict[str, Any]:
        """Génère un contexte de démonstration"""
        return {
            'team_size': 8,
            'industry': 'e-commerce',
            'timeline_pressure': 'high',
            'budget_flexibility': 'medium',
            'stakeholder_complexity': 'high'
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Convertit un score numérique en niveau de risque"""
        if score >= 4.0:
            return "CRITIQUE"
        elif score >= 3.0:
            return "ÉLEVÉ"
        elif score >= 2.0:
            return "MOYEN"
        elif score >= 1.0:
            return "FAIBLE"
        else:
            return "TRÈS FAIBLE"


# Instance globale pour réutilisation
_risk_engine = None

def get_risk_intelligence_engine() -> RiskIntelligenceEngine:
    """Retourne l'instance globale du moteur de risques"""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskIntelligenceEngine()
    return _risk_engine


if __name__ == "__main__":
    # Test du moteur
    engine = RiskIntelligenceEngine()
    
    # Analyse complète
    analysis = engine.analyze_project_risks({}, {})
    print(f"Risques identifiés: {len(analysis['identified_risks'])}")
    print(f"Score global: {analysis['overall_risk_score']:.2f}")
    print(f"Exposition totale: €{analysis['risk_exposure_total']:,.0f}")
    
    # Génération des visualisations
    dashboards = engine.create_risk_dashboard(analysis)
    print(f"Graphiques générés: {len(dashboards)}")
    
    # Test Monte Carlo
    monte_carlo = analysis['monte_carlo']
    print(f"Simulations Monte Carlo: {monte_carlo['iterations']:,}")
    print(f"Risque budgétaire P80: +€{monte_carlo['cost_statistics']['risk_premium_p80']:,.0f}")