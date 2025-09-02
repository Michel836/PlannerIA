"""
Advanced Business Intelligence and Analytics Module for PlannerIA
Tableaux de bord intelligents avec insights automatiques et prédictions business
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict
import sqlite3
import os

logger = logging.getLogger(__name__)

class KPICategory(Enum):
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    RISK = "risk"
    INNOVATION = "innovation"
    STAKEHOLDER = "stakeholder"

class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

@dataclass
class BusinessKPI:
    name: str
    category: KPICategory
    value: float
    target: float
    unit: str
    trend: TrendDirection
    change_percent: float
    description: str
    recommendation: str

@dataclass
class BusinessInsight:
    title: str
    description: str
    impact: str  # high, medium, low
    category: KPICategory
    confidence: float
    actionable_steps: List[str]
    data_source: str
    timestamp: datetime

class AdvancedAnalytics:
    """Module d'analytics avancé avec IA intégrée"""
    
    def __init__(self, db_path: str = "data/analytics.db"):
        self.db_path = db_path
        self._setup_database()
        self.kpi_history = defaultdict(list)
        self.insights_cache = []
        
    def _setup_database(self):
        """Initialiser la base de données pour l'historique"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table pour l'historique des KPIs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kpi_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    kpi_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    value REAL NOT NULL,
                    target REAL,
                    project_id TEXT,
                    metadata TEXT
                )
            """)
            
            # Table pour les insights
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS business_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    impact TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    actionable_steps TEXT,
                    data_source TEXT
                )
            """)
            
            conn.commit()
            
    def analyze_project_portfolio(self, projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyser un portfolio de projets et générer des insights business"""
        
        if not projects:
            return {"error": "Aucun projet à analyser"}
            
        # Calculer les KPIs du portfolio
        portfolio_kpis = self._calculate_portfolio_kpis(projects)
        
        # Générer des insights automatiques
        auto_insights = self._generate_automatic_insights(projects, portfolio_kpis)
        
        # Créer des visualisations interactives
        charts = self._create_portfolio_visualizations(projects, portfolio_kpis)
        
        # Prédictions et recommandations
        predictions = self._generate_predictions(projects)
        
        # Matrice de risque/valeur
        risk_value_matrix = self._create_risk_value_matrix(projects)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_summary": {
                "total_projects": len(projects),
                "total_budget": sum(p.get("metrics", {}).get("cost", 0) for p in projects),
                "avg_duration": np.mean([p.get("metrics", {}).get("duration", 0) for p in projects]),
                "success_probability": self._calculate_portfolio_success_rate(projects)
            },
            "kpis": [kpi.__dict__ for kpi in portfolio_kpis],
            "insights": [insight.__dict__ for insight in auto_insights],
            "charts": charts,
            "predictions": predictions,
            "risk_value_matrix": risk_value_matrix,
            "recommendations": self._generate_strategic_recommendations(projects, portfolio_kpis)
        }
        
    def _calculate_portfolio_kpis(self, projects: List[Dict[str, Any]]) -> List[BusinessKPI]:
        """Calculer les KPIs clés du portfolio"""
        
        kpis = []
        
        # KPI Performance: Budget moyen
        budgets = [p.get("metrics", {}).get("cost", 0) for p in projects]
        avg_budget = np.mean(budgets) if budgets else 0
        budget_target = 50000  # Target example
        budget_trend = TrendDirection.STABLE  # À améliorer avec historique
        
        kpis.append(BusinessKPI(
            name="Budget Moyen Projet",
            category=KPICategory.PERFORMANCE,
            value=avg_budget,
            target=budget_target,
            unit="€",
            trend=budget_trend,
            change_percent=((avg_budget - budget_target) / budget_target * 100) if budget_target > 0 else 0,
            description="Budget moyen par projet dans le portfolio",
            recommendation="Optimiser l'allocation des ressources" if avg_budget > budget_target else "Budget sous contrôle"
        ))
        
        # KPI Efficacité: Durée moyenne
        durations = [p.get("metrics", {}).get("duration", 0) for p in projects]
        avg_duration = np.mean(durations) if durations else 0
        duration_target = 45  # jours
        
        kpis.append(BusinessKPI(
            name="Durée Moyenne Projet",
            category=KPICategory.EFFICIENCY,
            value=avg_duration,
            target=duration_target,
            unit="jours",
            trend=TrendDirection.IMPROVING if avg_duration < duration_target else TrendDirection.DECLINING,
            change_percent=((avg_duration - duration_target) / duration_target * 100) if duration_target > 0 else 0,
            description="Temps moyen de réalisation des projets",
            recommendation="Accélérer la livraison" if avg_duration > duration_target else "Performance temporelle excellente"
        ))
        
        # KPI Qualité: Taux de complexité
        complexity_scores = []
        for project in projects:
            # Calculer un score de complexité basé sur les phases et tâches
            phases = project.get("phases", [])
            tasks_count = sum(len(phase.get("tasks", [])) for phase in phases)
            complexity_scores.append(min(100, tasks_count * 5))  # Score normalisé
            
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0
        complexity_target = 60
        
        kpis.append(BusinessKPI(
            name="Indice de Complexité",
            category=KPICategory.QUALITY,
            value=avg_complexity,
            target=complexity_target,
            unit="pts",
            trend=TrendDirection.STABLE,
            change_percent=((avg_complexity - complexity_target) / complexity_target * 100) if complexity_target > 0 else 0,
            description="Niveau de complexité moyen des projets",
            recommendation="Simplifier les processus" if avg_complexity > complexity_target else "Complexité maîtrisée"
        ))
        
        # KPI Innovation: Score technologique
        tech_scores = []
        for project in projects:
            title = project.get("title", "").lower()
            tech_keywords = ["ia", "ai", "ml", "blockchain", "cloud", "api", "automation", "intelligence"]
            tech_score = sum(10 for keyword in tech_keywords if keyword in title)
            tech_scores.append(min(100, tech_score))
            
        avg_tech_score = np.mean(tech_scores) if tech_scores else 0
        tech_target = 40
        
        kpis.append(BusinessKPI(
            name="Score Innovation Tech",
            category=KPICategory.INNOVATION,
            value=avg_tech_score,
            target=tech_target,
            unit="pts",
            trend=TrendDirection.IMPROVING if avg_tech_score > tech_target else TrendDirection.STABLE,
            change_percent=((avg_tech_score - tech_target) / tech_target * 100) if tech_target > 0 else 0,
            description="Niveau d'innovation technologique du portfolio",
            recommendation="Augmenter l'innovation" if avg_tech_score < tech_target else "Innovation excellente"
        ))
        
        # KPI Risque: Score de risque moyen
        risk_scores = []
        for project in projects:
            # Calculer un score de risque basé sur budget et durée
            budget = project.get("metrics", {}).get("cost", 0)
            duration = project.get("metrics", {}).get("duration", 0)
            risk_score = min(100, (budget / 1000 + duration) / 2)
            risk_scores.append(risk_score)
            
        avg_risk = np.mean(risk_scores) if risk_scores else 0
        risk_target = 30  # Plus bas = mieux
        
        kpis.append(BusinessKPI(
            name="Score de Risque Portfolio",
            category=KPICategory.RISK,
            value=avg_risk,
            target=risk_target,
            unit="pts",
            trend=TrendDirection.DECLINING if avg_risk > risk_target else TrendDirection.STABLE,
            change_percent=((avg_risk - risk_target) / risk_target * 100) if risk_target > 0 else 0,
            description="Niveau de risque global du portfolio",
            recommendation="Mitigation des risques nécessaire" if avg_risk > risk_target else "Risques sous contrôle"
        ))
        
        return kpis
        
    def _generate_automatic_insights(self, projects: List[Dict[str, Any]], kpis: List[BusinessKPI]) -> List[BusinessInsight]:
        """Générer des insights automatiques basés sur les données"""
        
        insights = []
        
        # Insight 1: Analyse budgétaire
        budgets = [p.get("metrics", {}).get("cost", 0) for p in projects]
        if budgets:
            budget_variance = np.std(budgets)
            avg_budget = np.mean(budgets)
            
            if budget_variance / avg_budget > 0.5:  # Forte variance
                insights.append(BusinessInsight(
                    title="Forte Disparité Budgétaire Détectée",
                    description=f"Les budgets varient significativement (écart-type: €{budget_variance:,.0f}). Cette disparité peut indiquer un manque de standardisation dans l'estimation.",
                    impact="medium",
                    category=KPICategory.PERFORMANCE,
                    confidence=0.85,
                    actionable_steps=[
                        "Standardiser les méthodes d'estimation",
                        "Créer des templates budgétaires par type de projet",
                        "Former les équipes à l'estimation"
                    ],
                    data_source="Analyse variance budgétaire",
                    timestamp=datetime.now()
                ))
        
        # Insight 2: Efficacité temporelle
        durations = [p.get("metrics", {}).get("duration", 0) for p in projects]
        if durations:
            short_projects = sum(1 for d in durations if d < 30)
            long_projects = sum(1 for d in durations if d > 60)
            
            if long_projects > len(projects) * 0.3:  # Plus de 30% de projets longs
                insights.append(BusinessInsight(
                    title="Tendance aux Projets Long-Terme",
                    description=f"{long_projects}/{len(projects)} projets dépassent 60 jours. Cela peut impacter l'agilité organisationnelle.",
                    impact="medium",
                    category=KPICategory.EFFICIENCY,
                    confidence=0.78,
                    actionable_steps=[
                        "Découper les projets longs en phases",
                        "Implémenter des livraisons incrémentales",
                        "Réviser la méthodologie projet"
                    ],
                    data_source="Analyse distribution durées",
                    timestamp=datetime.now()
                ))
        
        # Insight 3: Innovation technologique
        tech_projects = 0
        for project in projects:
            title = project.get("title", "").lower()
            if any(keyword in title for keyword in ["ia", "ai", "ml", "intelligence"]):
                tech_projects += 1
                
        if tech_projects / len(projects) > 0.6:  # Plus de 60% de projets tech
            insights.append(BusinessInsight(
                title="Portfolio Hautement Technologique",
                description=f"{tech_projects}/{len(projects)} projets intègrent des technologies avancées (IA/ML). Opportunité de positionnement leader.",
                impact="high",
                category=KPICategory.INNOVATION,
                confidence=0.92,
                actionable_steps=[
                    "Capitaliser sur l'expertise tech développée",
                    "Créer un centre d'excellence IA",
                    "Développer des offres tech propriétaires"
                ],
                data_source="Analyse contenu projets",
                timestamp=datetime.now()
            ))
        
        # Insight 4: Concentration sectorielle
        sectors = defaultdict(int)
        for project in projects:
            title = project.get("title", "").lower()
            if "medai" in title or "medical" in title or "santé" in title:
                sectors["healthcare"] += 1
            elif "fintech" in title or "financial" in title or "banc" in title:
                sectors["finance"] += 1
            elif "saas" in title or "startup" in title or "platform" in title:
                sectors["tech"] += 1
            else:
                sectors["other"] += 1
                
        dominant_sector = max(sectors.items(), key=lambda x: x[1])
        if dominant_sector[1] / len(projects) > 0.5:
            insights.append(BusinessInsight(
                title="Concentration Sectorielle Élevée",
                description=f"Forte concentration sur le secteur {dominant_sector[0]} ({dominant_sector[1]}/{len(projects)} projets). Risque de dépendance sectorielle.",
                impact="medium",
                category=KPICategory.RISK,
                confidence=0.80,
                actionable_steps=[
                    "Diversifier le portfolio sectoriel",
                    "Explorer de nouveaux marchés",
                    "Analyser les risques sectoriels"
                ],
                data_source="Analyse sectorielle",
                timestamp=datetime.now()
            ))
            
        return insights
        
    def _create_portfolio_visualizations(self, projects: List[Dict[str, Any]], kpis: List[BusinessKPI]) -> Dict[str, Any]:
        """Créer des visualisations interactives pour le portfolio"""
        
        charts = {}
        
        # 1. Graphique KPIs en radar
        kpi_names = [kpi.name for kpi in kpis]
        kpi_values = [min(100, (kpi.value / kpi.target * 100)) if kpi.target > 0 else 50 for kpi in kpis]
        
        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=kpi_values,
            theta=kpi_names,
            fill='toself',
            name='Performance Actuelle',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color='rgba(59, 130, 246, 1)', width=2)
        ))
        
        radar_fig.add_trace(go.Scatterpolar(
            r=[100] * len(kpi_names),
            theta=kpi_names,
            fill='toself',
            name='Objectifs',
            fillcolor='rgba(34, 197, 94, 0.2)',
            line=dict(color='rgba(34, 197, 94, 1)', width=1, dash='dash')
        ))
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            title="Performance Portfolio vs Objectifs",
            height=500
        )
        
        charts['kpi_radar'] = radar_fig.to_json()
        
        # 2. Distribution Budget vs Durée
        budgets = [p.get("metrics", {}).get("cost", 0) for p in projects]
        durations = [p.get("metrics", {}).get("duration", 0) for p in projects]
        titles = [p.get("title", f"Projet {i+1}")[:30] for i, p in enumerate(projects)]
        
        scatter_fig = px.scatter(
            x=durations, y=budgets, text=titles,
            labels={"x": "Durée (jours)", "y": "Budget (€)"},
            title="Distribution Budget/Durée des Projets",
            size_max=20
        )
        scatter_fig.update_traces(textposition="top center", marker=dict(size=15, opacity=0.7))
        scatter_fig.update_layout(height=500)
        
        charts['budget_duration_scatter'] = scatter_fig.to_json()
        
        # 3. Évolution des KPIs (simulée pour la démo)
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        timeline_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Budget Moyen", "Durée Moyenne", "Score Innovation", "Score Risque"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Simulation de données temporelles
        for i, kpi in enumerate(kpis[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Générer des données simulées avec tendance
            base_value = kpi.value
            noise = np.random.normal(0, base_value * 0.1, len(dates))
            trend_factor = 1 + (np.arange(len(dates)) / len(dates)) * 0.1
            values = base_value * trend_factor + noise
            
            timeline_fig.add_trace(
                go.Scatter(
                    x=dates, y=values,
                    mode='lines+markers',
                    name=kpi.name,
                    line=dict(width=2)
                ),
                row=row, col=col
            )
            
            # Ligne d'objectif
            timeline_fig.add_hline(
                y=kpi.target, line_dash="dash", line_color="red",
                row=row, col=col
            )
        
        timeline_fig.update_layout(
            height=600,
            title_text="Évolution des KPIs sur 30 jours",
            showlegend=False
        )
        
        charts['kpi_timeline'] = timeline_fig.to_json()
        
        return charts
        
    def _generate_predictions(self, projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Générer des prédictions business basées sur les tendances"""
        
        predictions = {}
        
        # Prédiction budget futur
        budgets = [p.get("metrics", {}).get("cost", 0) for p in projects]
        if budgets:
            avg_budget = np.mean(budgets)
            budget_growth = np.random.normal(0.05, 0.02)  # Simulation croissance
            
            predictions["budget_forecast"] = {
                "current_avg": avg_budget,
                "predicted_3m": avg_budget * (1 + budget_growth),
                "predicted_6m": avg_budget * (1 + budget_growth * 2),
                "confidence": 0.75,
                "trend": "increasing" if budget_growth > 0 else "decreasing"
            }
        
        # Prédiction succès projet
        success_factors = []
        for project in projects:
            score = 50  # Base
            
            # Facteurs positifs
            if project.get("metrics", {}).get("cost", 0) < 50000:
                score += 20  # Budget raisonnable
            if project.get("metrics", {}).get("duration", 0) < 45:
                score += 20  # Durée courte
                
            # Facteurs négatifs
            if "complex" in project.get("title", "").lower():
                score -= 15
                
            success_factors.append(min(100, max(0, score)))
            
        avg_success = np.mean(success_factors) if success_factors else 70
        
        predictions["success_probability"] = {
            "portfolio_success_rate": avg_success,
            "high_risk_projects": sum(1 for s in success_factors if s < 40),
            "safe_projects": sum(1 for s in success_factors if s > 80),
            "recommendation": "Renforcer l'accompagnement des projets à risque" if avg_success < 70 else "Portfolio bien équilibré"
        }
        
        return predictions
        
    def _create_risk_value_matrix(self, projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Créer une matrice risque/valeur pour les projets"""
        
        project_matrix = []
        
        for i, project in enumerate(projects):
            budget = project.get("metrics", {}).get("cost", 0)
            duration = project.get("metrics", {}).get("duration", 0)
            
            # Calculer score de valeur (basé sur potentiel business)
            value_score = min(100, budget / 1000)  # Simplification
            
            # Calculer score de risque
            risk_score = min(100, (duration + budget/1000) / 2)
            
            project_matrix.append({
                "name": project.get("title", f"Projet {i+1}")[:30],
                "value": value_score,
                "risk": risk_score,
                "budget": budget,
                "duration": duration,
                "quadrant": self._determine_quadrant(value_score, risk_score)
            })
        
        # Créer la visualisation
        matrix_fig = px.scatter(
            x=[p["risk"] for p in project_matrix],
            y=[p["value"] for p in project_matrix],
            text=[p["name"] for p in project_matrix],
            labels={"x": "Score de Risque", "y": "Score de Valeur"},
            title="Matrice Risque/Valeur des Projets"
        )
        
        # Ajouter les quadrants
        matrix_fig.add_hline(y=50, line_dash="dash", line_color="gray")
        matrix_fig.add_vline(x=50, line_dash="dash", line_color="gray")
        
        # Annotations des quadrants
        matrix_fig.add_annotation(x=25, y=75, text="Perles<br>(Faible risque,<br>Haute valeur)", showarrow=False, bgcolor="lightgreen")
        matrix_fig.add_annotation(x=75, y=75, text="Paris<br>(Haut risque,<br>Haute valeur)", showarrow=False, bgcolor="yellow")
        matrix_fig.add_annotation(x=25, y=25, text="Routine<br>(Faible risque,<br>Faible valeur)", showarrow=False, bgcolor="lightblue")
        matrix_fig.add_annotation(x=75, y=25, text="À éviter<br>(Haut risque,<br>Faible valeur)", showarrow=False, bgcolor="lightcoral")
        
        matrix_fig.update_traces(textposition="top center", marker=dict(size=12))
        matrix_fig.update_layout(height=500)
        
        return {
            "projects": project_matrix,
            "chart": matrix_fig.to_json(),
            "quadrant_analysis": self._analyze_quadrants(project_matrix)
        }
        
    def _determine_quadrant(self, value: float, risk: float) -> str:
        """Déterminer le quadrant d'un projet"""
        if value >= 50 and risk < 50:
            return "pearls"  # Perles
        elif value >= 50 and risk >= 50:
            return "gambles"  # Paris
        elif value < 50 and risk < 50:
            return "routine"  # Routine
        else:
            return "avoid"  # À éviter
            
    def _analyze_quadrants(self, project_matrix: List[Dict]) -> Dict[str, Any]:
        """Analyser la distribution des projets par quadrant"""
        
        quadrant_counts = defaultdict(int)
        for project in project_matrix:
            quadrant_counts[project["quadrant"]] += 1
            
        total_projects = len(project_matrix)
        
        return {
            "pearls": {
                "count": quadrant_counts["pearls"],
                "percentage": (quadrant_counts["pearls"] / total_projects * 100) if total_projects > 0 else 0,
                "recommendation": "Prioriser ces projets"
            },
            "gambles": {
                "count": quadrant_counts["gambles"], 
                "percentage": (quadrant_counts["gambles"] / total_projects * 100) if total_projects > 0 else 0,
                "recommendation": "Évaluer soigneusement le rapport risque/bénéfice"
            },
            "routine": {
                "count": quadrant_counts["routine"],
                "percentage": (quadrant_counts["routine"] / total_projects * 100) if total_projects > 0 else 0,
                "recommendation": "Maintenir en portfolio de base"
            },
            "avoid": {
                "count": quadrant_counts["avoid"],
                "percentage": (quadrant_counts["avoid"] / total_projects * 100) if total_projects > 0 else 0,
                "recommendation": "Reconsidérer ou annuler"
            }
        }
        
    def _calculate_portfolio_success_rate(self, projects: List[Dict[str, Any]]) -> float:
        """Calculer le taux de succès estimé du portfolio"""
        
        if not projects:
            return 0.0
            
        success_scores = []
        for project in projects:
            score = 70  # Score de base
            
            budget = project.get("metrics", {}).get("cost", 0)
            duration = project.get("metrics", {}).get("duration", 0)
            
            # Ajustements basés sur les caractéristiques
            if budget < 30000:
                score += 10
            elif budget > 70000:
                score -= 15
                
            if duration < 30:
                score += 10
            elif duration > 60:
                score -= 15
                
            # Bonus innovation tech
            title = project.get("title", "").lower()
            if any(keyword in title for keyword in ["ia", "ai", "ml", "intelligence"]):
                score += 5
                
            success_scores.append(min(100, max(0, score)))
            
        return np.mean(success_scores)
        
    def _generate_strategic_recommendations(self, projects: List[Dict[str, Any]], kpis: List[BusinessKPI]) -> List[Dict[str, Any]]:
        """Générer des recommandations stratégiques"""
        
        recommendations = []
        
        # Analyse des KPIs pour recommandations
        for kpi in kpis:
            if kpi.value > kpi.target * 1.2:  # Dépassement significatif
                if kpi.category == KPICategory.PERFORMANCE:
                    recommendations.append({
                        "title": f"Optimisation {kpi.name}",
                        "priority": "medium",
                        "description": f"Le KPI {kpi.name} dépasse la cible de {((kpi.value/kpi.target-1)*100):.0f}%",
                        "actions": [
                            "Analyser les causes du dépassement",
                            "Mettre en place des mesures correctives",
                            "Réviser les objectifs si nécessaire"
                        ]
                    })
                    
        # Recommandations basées sur la composition du portfolio
        budgets = [p.get("metrics", {}).get("cost", 0) for p in projects]
        if budgets:
            budget_variance = np.std(budgets) / np.mean(budgets)
            if budget_variance > 0.6:  # Forte variation
                recommendations.append({
                    "title": "Standardisation Budgétaire",
                    "priority": "high",
                    "description": "Forte disparité dans les budgets projet détectée",
                    "actions": [
                        "Créer des templates budgétaires",
                        "Former les équipes à l'estimation",
                        "Implémenter un processus de validation"
                    ]
                })
        
        # Recommandation innovation
        tech_count = sum(1 for p in projects 
                        if any(keyword in p.get("title", "").lower() 
                              for keyword in ["ia", "ai", "ml", "intelligence"]))
        
        if tech_count / len(projects) > 0.7:  # Portfolio très tech
            recommendations.append({
                "title": "Capitalisation Expertise Tech",
                "priority": "high", 
                "description": "Portfolio hautement technologique - opportunité de leadership",
                "actions": [
                    "Créer un centre d'excellence IA",
                    "Développer des offres propriétaires",
                    "Renforcer la R&D interne"
                ]
            })
            
        return recommendations

# Instance globale pour l'utilisation dans le dashboard
global_analytics = AdvancedAnalytics()

def get_analytics_engine() -> AdvancedAnalytics:
    """Obtenir l'instance globale du moteur d'analytics"""
    return global_analytics

def analyze_portfolio(projects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyser un portfolio de projets"""
    return global_analytics.analyze_project_portfolio(projects)

def get_kpi_dashboard_data(projects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Obtenir les données pour le dashboard KPI"""
    analysis = global_analytics.analyze_project_portfolio(projects)
    return {
        "kpis": analysis.get("kpis", []),
        "summary": analysis.get("portfolio_summary", {}),
        "health_score": global_analytics._calculate_portfolio_success_rate(projects)
    }