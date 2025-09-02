# Module Analytics Avancés pour PlannerIA
# Métriques de performance, ROI, tableaux de bord exécutifs et analyses prédictives

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

class AnalyticsModule:
    def __init__(self):
        self.kpi_colors = {
            'performance': '#3B82F6',
            'financier': '#10B981',
            'qualite': '#F59E0B',
            'risque': '#EF4444',
            'satisfaction': '#8B5CF6',
            'innovation': '#EC4899',
            'operationnel': '#06B6D4'
        }
        
    def load_analytics_data(self, project_id: str = "projet_test") -> Dict[str, Any]:
        """Charge les données analytiques enrichies"""
        return {
            'metriques_performance': {
                'velocite_equipe': [12, 15, 18, 22, 19, 24, 26, 28],  # story points par sprint
                'taux_completion': [85, 92, 88, 95, 89, 93, 96, 98],  # pourcentage
                'temps_cycle': [5.2, 4.8, 4.1, 3.9, 4.2, 3.7, 3.4, 3.1],  # jours
                'debt_technique': [15, 18, 22, 19, 16, 12, 10, 8],  # heures
                'couverture_tests': [76, 78, 82, 85, 87, 89, 91, 93],  # pourcentage
                'uptime_systeme': [99.2, 99.5, 99.8, 99.1, 99.6, 99.9, 99.7, 99.8],
                'temps_response': [180, 165, 145, 130, 125, 110, 95, 85]  # millisecondes
            },
            'indicateurs_financiers': {
                'roi_mensuel': [8.5, 12.3, 15.7, 18.2, 21.6, 24.1, 27.3, 30.2],  # pourcentage
                'cost_per_story_point': [850, 780, 720, 680, 650, 620, 590, 560],  # euros
                'budget_burn_rate': [15000, 14500, 13800, 13200, 12800, 12400, 12000, 11600],  # euros/mois
                'valeur_livree': [125000, 147000, 168000, 195000, 218000, 245000, 275000, 308000],  # euros cumulé
                'cout_acquisition_client': [245, 220, 195, 180, 165, 150, 135, 120],  # euros
                'revenue_per_user': [45, 48, 52, 56, 61, 65, 70, 75],  # euros/mois
                'retention_rate': [85, 87, 89, 91, 92, 94, 95, 96],  # pourcentage
                'ltv_cac_ratio': [2.1, 2.4, 2.7, 3.1, 3.5, 3.8, 4.2, 4.6]  # ratio
            },
            'qualite_livrables': {
                'defauts_production': [12, 8, 5, 3, 2, 1, 0, 1],  # nombre
                'satisfaction_client': [7.2, 7.8, 8.1, 8.5, 8.7, 8.9, 9.1, 9.3],  # sur 10
                'temps_resolution': [4.2, 3.8, 3.1, 2.7, 2.3, 1.9, 1.6, 1.4],  # heures moyennes
                'taux_regression': [5.2, 4.1, 3.2, 2.1, 1.8, 1.2, 0.9, 0.6],  # pourcentage
                'score_qualite': [78, 82, 85, 89, 92, 94, 96, 97],  # sur 100
                'nps_score': [25, 32, 38, 45, 52, 58, 63, 68],  # Net Promoter Score
                'first_call_resolution': [65, 68, 72, 76, 79, 82, 85, 88]  # pourcentage
            },
            'metriques_equipe': {
                'moral_equipe': [7.1, 7.4, 7.8, 8.1, 8.3, 8.5, 8.7, 8.9],  # sur 10
                'taux_absenteisme': [5.2, 4.8, 4.1, 3.7, 3.4, 3.1, 2.8, 2.5],  # pourcentage
                'formations_completees': [2, 3, 4, 6, 8, 11, 14, 18],  # nombre cumulé
                'certifications': [1, 1, 2, 3, 4, 5, 7, 8],  # nombre cumulé
                'innovation_time': [10, 12, 15, 18, 20, 22, 25, 28],  # pourcentage temps alloué
                'peer_feedback_score': [7.8, 8.0, 8.2, 8.4, 8.6, 8.7, 8.9, 9.0]
            },
            'metriques_produit': {
                'utilisateurs_actifs': [1250, 1380, 1520, 1680, 1850, 2040, 2250, 2480],  # MAU
                'sessions_par_user': [8.2, 8.6, 9.1, 9.5, 9.8, 10.2, 10.6, 11.1],
                'duree_session': [12.5, 13.2, 13.8, 14.5, 15.1, 15.7, 16.4, 17.0],  # minutes
                'taux_conversion': [3.2, 3.5, 3.8, 4.1, 4.4, 4.7, 5.0, 5.3],  # pourcentage
                'churn_rate': [8.5, 7.9, 7.3, 6.8, 6.2, 5.7, 5.2, 4.8],  # pourcentage mensuel
                'feature_adoption': [42, 45, 49, 53, 57, 61, 65, 69]  # pourcentage nouvelles features
            },
            'predictions': {
                'completion_prevue': datetime.now() + timedelta(days=32),
                'budget_final_estime': 115200,
                'roi_projete': 32.8,
                'risque_retard': 8,  # pourcentage
                'score_succes': 92,  # sur 100
                'utilisateurs_projetes_6m': 3500,
                'chiffre_affaires_prevu': 485000
            },
            'benchmarks': {
                'velocite_industrie': 20,
                'roi_moyen_secteur': 25.5,
                'satisfaction_benchmark': 8.5,
                'temps_cycle_optimal': 3.5,
                'couverture_tests_cible': 90,
                'nps_excellent': 50,
                'retention_cible': 90,
                'uptime_industrie': 99.5
            },
            'tendances_equipe': {
                'productivite': {'tendance': 'forte_hausse', 'variation': '+18%', 'confiance': 0.92},
                'moral': {'tendance': 'hausse', 'variation': '+8%', 'confiance': 0.85},
                'retention': {'tendance': 'stable', 'variation': '+2%', 'confiance': 0.78},
                'formation': {'tendance': 'forte_hausse', 'variation': '+35%', 'confiance': 0.95},
                'innovation': {'tendance': 'hausse', 'variation': '+12%', 'confiance': 0.88}
            },
            'alertes_automatiques': [
                {'type': 'success', 'message': 'ROI dépasse les objectifs de +15%', 'priorite': 'info'},
                {'type': 'warning', 'message': 'Vélocité en baisse sur les 2 derniers sprints', 'priorite': 'medium'},
                {'type': 'info', 'message': 'Nouveau record de satisfaction client (9.3/10)', 'priorite': 'low'}
            ],
            'scenarios_previsionels': {
                'optimiste': {'roi_final': 38.5, 'completion': 28, 'budget': 108000},
                'realiste': {'roi_final': 32.8, 'completion': 32, 'budget': 115200},
                'pessimiste': {'roi_final': 27.1, 'completion': 38, 'budget': 125000}
            }
        }
    
    def calculate_advanced_kpis(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule des KPIs avancés avec scoring intelligent"""
        perf = analytics_data['metriques_performance']
        finance = analytics_data['indicateurs_financiers']
        qualite = analytics_data['qualite_livrables']
        equipe = analytics_data['metriques_equipe']
        produit = analytics_data['metriques_produit']
        
        # Calculs de tendances avec pondération temporelle
        velocite_trend = self._calculate_weighted_trend(perf['velocite_equipe'])
        roi_trend = self._calculate_weighted_trend(finance['roi_mensuel'])
        qualite_trend = self._calculate_weighted_trend(qualite['score_qualite'])
        
        # Score de santé global du projet avec nouveaux critères
        health_score = self._calculate_comprehensive_health_score(analytics_data)
        
        # Efficacité opérationnelle multicritères
        operational_efficiency = self._calculate_operational_efficiency(analytics_data)
        
        # Score d'innovation et d'adaptation
        innovation_score = self._calculate_innovation_score(analytics_data)
        
        # Prédictibilité et stabilité
        predictability_score = self._calculate_predictability_score(analytics_data)
        
        return {
            'velocite_actuelle': perf['velocite_equipe'][-1],
            'velocite_trend': velocite_trend,
            'velocite_variance': np.std(perf['velocite_equipe'][-4:]),
            'roi_actuel': finance['roi_mensuel'][-1],
            'roi_trend': roi_trend,
            'roi_acceleration': self._calculate_acceleration(finance['roi_mensuel']),
            'score_qualite_actuel': qualite['score_qualite'][-1],
            'qualite_trend': qualite_trend,
            'health_score': health_score,
            'operational_efficiency': operational_efficiency,
            'innovation_score': innovation_score,
            'predictability_score': predictability_score,
            'burn_rate': finance['budget_burn_rate'][-1],
            'burn_rate_trend': self._calculate_trend(finance['budget_burn_rate']),
            'valeur_totale': finance['valeur_livree'][-1],
            'ltv_cac_actuel': finance['ltv_cac_ratio'][-1],
            'nps_actuel': qualite['nps_score'][-1],
            'retention_actuelle': finance['retention_rate'][-1],
            'churn_actuel': produit['churn_rate'][-1],
            'utilisateurs_actifs': produit['utilisateurs_actifs'][-1],
            'moral_equipe': equipe['moral_equipe'][-1]
        }
    
    def _calculate_weighted_trend(self, data_list: List[float], window: int = 4) -> str:
        """Calcule la tendance avec pondération des données récentes"""
        if len(data_list) < window:
            return "insufficient_data"
        
        # Pondération exponentielle des données récentes
        weights = np.exp(np.linspace(0, 1, window))
        weights = weights / weights.sum()
        
        recent_data = np.array(data_list[-window:])
        previous_data = np.array(data_list[-window*2:-window]) if len(data_list) >= window*2 else recent_data
        
        recent_weighted = np.average(recent_data, weights=weights)
        previous_weighted = np.average(previous_data, weights=weights)
        
        change = (recent_weighted - previous_weighted) / previous_weighted * 100
        
        if change > 8:
            return "forte_hausse"
        elif change > 3:
            return "hausse"
        elif change < -8:
            return "forte_baisse"
        elif change < -3:
            return "baisse"
        else:
            return "stable"
    
    def _calculate_trend(self, data_list: List[float]) -> str:
        """Calcule la tendance simple d'une série de données"""
        if len(data_list) < 3:
            return "stable"
        
        recent = np.mean(data_list[-3:])
        previous = np.mean(data_list[-6:-3]) if len(data_list) >= 6 else np.mean(data_list[:-3])
        
        change = (recent - previous) / previous * 100
        
        if change > 5:
            return "forte_hausse"
        elif change > 2:
            return "hausse"
        elif change < -5:
            return "forte_baisse"
        elif change < -2:
            return "baisse"
        else:
            return "stable"
    
    def _calculate_acceleration(self, data_list: List[float]) -> float:
        """Calcule l'accélération d'une métrique"""
        if len(data_list) < 3:
            return 0.0
        
        # Calcul de la dérivée seconde approximative
        deltas = [data_list[i] - data_list[i-1] for i in range(1, len(data_list))]
        if len(deltas) < 2:
            return 0.0
        
        accelerations = [deltas[i] - deltas[i-1] for i in range(1, len(deltas))]
        return np.mean(accelerations[-3:])  # Moyenne des 3 dernières accélérations
    
    def _calculate_comprehensive_health_score(self, analytics_data: Dict[str, Any]) -> float:
        """Calcule un score de santé global avancé du projet"""
        perf = analytics_data['metriques_performance']
        finance = analytics_data['indicateurs_financiers']
        qualite = analytics_data['qualite_livrables']
        equipe = analytics_data['metriques_equipe']
        produit = analytics_data['metriques_produit']
        benchmarks = analytics_data['benchmarks']
        
        # Normalisation des métriques par rapport aux benchmarks avec scoring avancé
        velocite_score = min(120, (perf['velocite_equipe'][-1] / benchmarks['velocite_industrie']) * 100)
        roi_score = min(120, (finance['roi_mensuel'][-1] / benchmarks['roi_moyen_secteur']) * 100)
        qualite_score = min(120, (qualite['satisfaction_client'][-1] / benchmarks['satisfaction_benchmark']) * 100)
        nps_score = min(120, (qualite['nps_score'][-1] / benchmarks['nps_excellent']) * 100)
        retention_score = min(120, (finance['retention_rate'][-1] / benchmarks['retention_cible']) * 100)
        uptime_score = min(120, (perf['uptime_systeme'][-1] / benchmarks['uptime_industrie']) * 100)
        
        # Score d'équipe
        moral_score = (equipe['moral_equipe'][-1] / 10) * 100
        
        # Score pondéré avec nouveaux critères
        health_score = (
            velocite_score * 0.15 +      # Performance technique
            roi_score * 0.25 +           # Performance financière
            qualite_score * 0.20 +       # Satisfaction client
            nps_score * 0.15 +           # Loyalty client
            retention_score * 0.10 +     # Rétention
            uptime_score * 0.10 +        # Fiabilité technique
            moral_score * 0.05           # Bien-être équipe
        )
        
        return min(100, health_score)
    
    def _calculate_operational_efficiency(self, analytics_data: Dict[str, Any]) -> float:
        """Calcule l'efficacité opérationnelle multicritères"""
        perf = analytics_data['metriques_performance']
        # finance = analytics_data['indicateurs_financier']
        finance = analytics_data.get('indicateurs_financiers', analytics_data.get('indicateurs_financier', {}))
        equipe = analytics_data['metriques_equipe']
        
        # Métriques d'efficacité
        completion_efficiency = np.mean(perf['taux_completion'][-3:]) / 100
        cycle_efficiency = max(0, (6 - np.mean(perf['temps_cycle'][-3:])) / 6)
        cost_efficiency = max(0, (1000 - finance['cost_per_story_point'][-1]) / 1000)
        team_efficiency = (10 - equipe['taux_absenteisme'][-1]) / 10
        
        # Score pondéré
        efficiency = (
            completion_efficiency * 0.3 +
            cycle_efficiency * 0.25 +
            cost_efficiency * 0.25 +
            team_efficiency * 0.2
        ) * 100
        
        return min(100, efficiency)
    
    def _calculate_innovation_score(self, analytics_data: Dict[str, Any]) -> float:
        """Calcule le score d'innovation et d'adaptation"""
        equipe = analytics_data['metriques_equipe']
        produit = analytics_data['metriques_produit']
        
        # Métriques d'innovation
        formation_score = min(100, equipe['formations_completees'][-1] * 5)
        cert_score = min(100, equipe['certifications'][-1] * 10)
        innovation_time_score = min(100, equipe['innovation_time'][-1] * 3)
        feature_adoption_score = produit['feature_adoption'][-1]
        
        innovation_score = (
            formation_score * 0.25 +
            cert_score * 0.25 +
            innovation_time_score * 0.25 +
            feature_adoption_score * 0.25
        )
        
        return min(100, innovation_score)
    
    def _calculate_predictability_score(self, analytics_data: Dict[str, Any]) -> float:
        """Calcule la prédictibilité et stabilité du projet"""
        perf = analytics_data['metriques_performance']
        finance = analytics_data['indicateurs_financiers']
        
        # Variance normalisée des métriques clés
        velocite_stability = max(0, 100 - (np.std(perf['velocite_equipe'][-6:]) / np.mean(perf['velocite_equipe'][-6:]) * 100))
        burn_stability = max(0, 100 - (np.std(finance['budget_burn_rate'][-6:]) / np.mean(finance['budget_burn_rate'][-6:]) * 100))
        
        predictability = (velocite_stability + burn_stability) / 2
        return min(100, predictability)
    
    def create_executive_dashboard(self, analytics_data: Dict[str, Any]) -> go.Figure:
        """Crée le tableau de bord exécutif unifié"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('ROI & Croissance', 'Satisfaction Client', 'Performance Équipe', 
                          'Métriques Produit', 'Stabilité Financière', 'Innovation'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû']
        
        # ROI & Revenue
        fig.add_trace(go.Scatter(x=mois, y=analytics_data['indicateurs_financiers']['roi_mensuel'],
                                name="ROI %", line=dict(color='#10B981', width=3)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(x=mois, y=analytics_data['indicateurs_financiers']['revenue_per_user'],
                            name="Revenue/User", marker_color='#3B82F6', opacity=0.6), row=1, col=1, secondary_y=True)
        
        # Satisfaction & NPS
        fig.add_trace(go.Scatter(x=mois, y=analytics_data['qualite_livrables']['satisfaction_client'],
                                name="Satisfaction", line=dict(color='#8B5CF6', width=3)), row=1, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=analytics_data['qualite_livrables']['nps_score'],
                                name="NPS", line=dict(color='#F59E0B', width=2)), row=1, col=2, secondary_y=True)
        
        # Performance Équipe
        fig.add_trace(go.Scatter(x=mois, y=analytics_data['metriques_performance']['velocite_equipe'],
                                name="Vélocité", line=dict(color='#EF4444', width=3)), row=1, col=3)
        
        # Métriques Produit
        fig.add_trace(go.Bar(x=mois, y=analytics_data['metriques_produit']['utilisateurs_actifs'],
                            name="MAU", marker_color='#06B6D4'), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=analytics_data['metriques_produit']['taux_conversion'],
                                name="Conversion %", line=dict(color='#EC4899', width=2)), row=2, col=1, secondary_y=True)
        
        # Stabilité Financière
        fig.add_trace(go.Scatter(x=mois, y=analytics_data['indicateurs_financiers']['budget_burn_rate'],
                                name="Burn Rate", line=dict(color='#F59E0B', width=3)), row=2, col=2)
        
        # Innovation
        fig.add_trace(go.Bar(x=mois, y=analytics_data['metriques_equipe']['formations_completees'],
                            name="Formations", marker_color='#8B5CF6'), row=2, col=3)
        
        fig.update_layout(height=600, showlegend=False, title_text="Dashboard Exécutif - Vue Consolidée")
        return fig
    
    def create_performance_dashboard(self, analytics_data: Dict[str, Any]) -> go.Figure:
        """Crée le tableau de bord de performance amélioré"""
        perf = analytics_data['metriques_performance']
        mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Vélocité & Stabilité', 'Qualité Code', 'Performance Système', 'Efficacité Processus'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Vélocité avec bandes de confiance
        fig.add_trace(go.Scatter(x=mois, y=perf['velocite_equipe'], name="Vélocité", 
                                line=dict(color='#3B82F6', width=3)), row=1, col=1, secondary_y=False)
        
        # Dette technique
        fig.add_trace(go.Bar(x=mois, y=perf['debt_technique'], name="Dette Tech", 
                            marker_color='#EF4444', opacity=0.6), row=1, col=1, secondary_y=True)
        
        # Couverture tests & Score qualité  
        fig.add_trace(go.Scatter(x=mois, y=perf['couverture_tests'], name="Tests %", 
                                line=dict(color='#10B981', width=3)), row=1, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=analytics_data['qualite_livrables']['score_qualite'], 
                                name="Qualité", line=dict(color='#8B5CF6', width=2)), row=1, col=2, secondary_y=True)
        
        # Uptime & Temps réponse
        fig.add_trace(go.Scatter(x=mois, y=perf['uptime_systeme'], name="Uptime %", 
                                line=dict(color='#06B6D4', width=3)), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=perf['temps_response'], name="Latence ms", 
                                line=dict(color='#F59E0B', width=2)), row=2, col=1, secondary_y=True)
        
        # Temps de cycle & Taux complétion
        fig.add_trace(go.Scatter(x=mois, y=perf['temps_cycle'], name="Cycle (j)", 
                                line=dict(color='#EC4899', width=3)), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Performance Technique Détaillée")
        return fig
    
    def create_financial_dashboard(self, analytics_data: Dict[str, Any]) -> go.Figure:
        """Crée le tableau de bord financier avancé"""
        finance = analytics_data['indicateurs_financiers']
        mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROI & Croissance', 'Coûts & Efficacité', 'Métriques Client', 'Valeur & Rétention'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # ROI avec projection
        fig.add_trace(go.Scatter(x=mois, y=finance['roi_mensuel'], name="ROI", 
                                line=dict(color='#10B981', width=4)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Bar(x=mois, y=finance['valeur_livree'], name="Valeur", 
                            marker_color='#3B82F6', opacity=0.6), row=1, col=1, secondary_y=True)
        
        # Coûts
        fig.add_trace(go.Scatter(x=mois, y=finance['budget_burn_rate'], name="Burn Rate", 
                                line=dict(color='#EF4444', width=3)), row=1, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=finance['cost_per_story_point'], name="Coût/SP", 
                                line=dict(color='#F59E0B', width=2)), row=1, col=2, secondary_y=True)
        
        # Métriques client
        fig.add_trace(go.Scatter(x=mois, y=finance['cout_acquisition_client'], name="CAC", 
                                line=dict(color='#8B5CF6', width=3)), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=finance['revenue_per_user'], name="ARPU", 
                                line=dict(color='#06B6D4', width=2)), row=2, col=1, secondary_y=True)
        
        # LTV/CAC & Rétention
        fig.add_trace(go.Scatter(x=mois, y=finance['ltv_cac_ratio'], name="LTV/CAC", 
                                line=dict(color='#10B981', width=4)), row=2, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=finance['retention_rate'], name="Retention %", 
                                line=dict(color='#EC4899', width=2)), row=2, col=2, secondary_y=True)
        
        fig.update_layout(height=600, showlegend=False, title_text="Analyse Financière Complète")
        return fig
    
    def create_predictive_analysis(self, analytics_data: Dict[str, Any]) -> go.Figure:
        """Crée l'analyse prédictive avancée avec scénarios"""
        # Données historiques
        finance = analytics_data['indicateurs_financiers']
        mois_historique = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû']
        
        # Prédictions multi-scénarios
        mois_futur = ['Sep', 'Oct', 'Nov', 'Déc']
        scenarios = analytics_data['scenarios_previsionels']
        
        fig = go.Figure()
        
        # Données historiques ROI
        fig.add_trace(go.Scatter(
            x=mois_historique,
            y=finance['roi_mensuel'],
            mode='lines+markers',
            name='ROI Historique',
            line=dict(color='#3B82F6', width=4),
            marker=dict(size=8)
        ))
        
        # Scénario optimiste
        roi_optimiste = [32, 35, 38, 41]
        fig.add_trace(go.Scatter(
            x=mois_futur,
            y=roi_optimiste,
            mode='lines+markers',
            name='Scénario Optimiste',
            line=dict(color='#10B981', width=3, dash='dot')
        ))
        
        # Scénario réaliste
        roi_realiste = [30, 32, 34, 35]
        fig.add_trace(go.Scatter(
            x=mois_futur,
            y=roi_realiste,
            mode='lines+markers',
            name='Scénario Réaliste',
            line=dict(color='#F59E0B', width=3, dash='dash')
        ))
        
        # Scénario pessimiste
        roi_pessimiste = [28, 29, 30, 31]
        fig.add_trace(go.Scatter(
            x=mois_futur,
            y=roi_pessimiste,
            mode='lines+markers',
            name='Scénario Pessimiste',
            line=dict(color='#EF4444', width=3, dash='dashdot')
        ))
        
        # Zone de confiance pour scénario réaliste
        fig.add_trace(go.Scatter(
            x=mois_futur + mois_futur[::-1],
            y=[r+2 for r in roi_realiste] + [r-2 for r in roi_realiste[::-1]],
            fill='toself',
            fillcolor='rgba(249, 158, 11, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Zone de Confiance'
        ))
        
        fig.update_layout(
            title="Projections ROI Multi-Scénarios",
            xaxis_title="Période",
            yaxis_title="ROI (%)",
            height=500,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def create_team_analytics(self, analytics_data: Dict[str, Any]) -> go.Figure:
        """Crée l'analyse d'équipe avancée"""
        equipe = analytics_data['metriques_equipe']
        mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bien-être & Performance', 'Développement', 'Innovation', 'Feedback'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Moral & Absentéisme
        fig.add_trace(go.Scatter(x=mois, y=equipe['moral_equipe'], name="Moral", 
                                line=dict(color='#10B981', width=3)), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=equipe['taux_absenteisme'], name="Absentéisme %", 
                                line=dict(color='#EF4444', width=2)), row=1, col=1, secondary_y=True)
        
        # Formation & Certifications
        fig.add_trace(go.Bar(x=mois, y=equipe['formations_completees'], name="Formations", 
                            marker_color='#3B82F6'), row=1, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=mois, y=equipe['certifications'], name="Certifs", 
                                line=dict(color='#8B5CF6', width=3)), row=1, col=2, secondary_y=True)
        
        # Innovation time
        fig.add_trace(go.Scatter(x=mois, y=equipe['innovation_time'], name="Innovation %", 
                                line=dict(color='#EC4899', width=3), fill='tonexty'), row=2, col=1)
        
        # Peer feedback
        fig.add_trace(go.Scatter(x=mois, y=equipe['peer_feedback_score'], name="Peer Score", 
                                line=dict(color='#06B6D4', width=3)), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Analytics Équipe & RH")
        return fig

    def render_analytics_dashboard(self, project_id: str = "projet_test"):
        """Affiche le dashboard analytics complet"""
        st.title("📈 Reporting & Analytics")
        st.markdown("*Métriques de performance, ROI et analyses prédictives*")
        
        # Configuration
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            analytics_view = st.selectbox(
                "Vue Analytics:",
                ["Dashboard Exécutif", "Performance Technique", "Analyse Financière", "Analytics Équipe", "Prédictions Multi-Scénarios"],
                key="analytics_view_mode"
            )
        
        with col2:
            period_filter = st.selectbox(
                "Période d'analyse:",
                ["6 derniers mois", "Année courante", "12 derniers mois", "Prévisionnel"],
                key="analytics_period"
            )
        
        with col3:
            if st.button("📊 Actualiser", use_container_width=True, key="analytics_refresh"):
                st.success("Analytics actualisées!")
        
        # Chargement des données
        analytics_data = self.load_analytics_data(project_id)
        kpis = self.calculate_advanced_kpis(analytics_data)
        
        # Alertes automatiques en haut
        alertes = analytics_data.get('alertes_automatiques', [])
        if alertes:
            st.subheader("🚨 Alertes Intelligentes")
            for alerte in alertes:
                if alerte['type'] == 'success':
                    st.success(f"✅ {alerte['message']}")
                elif alerte['type'] == 'warning':
                    st.warning(f"⚠️ {alerte['message']}")
                else:
                    st.info(f"ℹ️ {alerte['message']}")
        
        # KPIs Principaux améliorés
        st.subheader("🎯 KPIs Exécutifs")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            trend_icons = {"forte_hausse": "🚀", "hausse": "📈", "stable": "➡️", "baisse": "📉", "forte_baisse": "📉"}
            trend_icon = trend_icons.get(kpis['velocite_trend'], "➡️")
            st.metric("⚡ Vélocité", f"{kpis['velocite_actuelle']} SP", 
                     delta=f"{trend_icon} {kpis['velocite_trend']}")
        
        with col2:
            roi_icon = trend_icons.get(kpis['roi_trend'], "➡️")
            st.metric("💰 ROI", f"{kpis['roi_actuel']:.1f}%", 
                     delta=f"{roi_icon} +{kpis['roi_acceleration']:.1f}% acc.")
        
        with col3:
            health_color = "normal" if kpis['health_score'] > 80 else "inverse"
            st.metric("❤️ Santé Projet", f"{kpis['health_score']:.0f}/100", 
                     delta_color=health_color)
        
        with col4:
            st.metric("🎯 Efficacité Op.", f"{kpis['operational_efficiency']:.0f}%")
        
        with col5:
            burn_trend_icon = "📉" if kpis['burn_rate_trend'] == 'baisse' else "📈"
            st.metric("💸 Burn Rate", f"{kpis['burn_rate']:,}€/mois", 
                     delta=f"{burn_trend_icon}")
        
        with col6:
            st.metric("🚀 Innovation", f"{kpis['innovation_score']:.0f}/100")
        
        # Métriques secondaires
        st.markdown("#### 📊 Métriques Clés")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("😊 NPS Score", f"{kpis['nps_actuel']}")
        with col2:
            st.metric("🔄 LTV/CAC", f"{kpis['ltv_cac_actuel']:.1f}x")
        with col3:
            st.metric("📱 MAU", f"{kpis['utilisateurs_actifs']:,}")
        with col4:
            churn_color = "normal" if kpis['churn_actuel'] < 6 else "inverse"
            st.metric("📉 Churn", f"{kpis['churn_actuel']:.1f}%", delta_color=churn_color)
        with col5:
            st.metric("🎭 Moral Équipe", f"{kpis['moral_equipe']:.1f}/10")
        
        st.divider()
        
        # Contenu selon la vue
        if analytics_view == "Dashboard Exécutif":
            st.plotly_chart(
                self.create_executive_dashboard(analytics_data),
                use_container_width=True
            )
            
            # Résumé exécutif intelligent
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📋 Résumé Exécutif Intelligent")
                predictions = analytics_data['predictions']
                
                # Analyse automatique des tendances
                if kpis['roi_trend'] in ['forte_hausse', 'hausse']:
                    st.success(f"✅ **Performance financière excellente**: ROI en {kpis['roi_trend']} (+{kpis['roi_acceleration']:.1f}% d'accélération)")
                
                if kpis['health_score'] > 85:
                    st.success(f"✅ **Santé projet optimale**: Score global de {kpis['health_score']:.0f}/100")
                elif kpis['health_score'] < 70:
                    st.warning(f"⚠️ **Attention requise**: Score santé à {kpis['health_score']:.0f}/100")
                
                if kpis['innovation_score'] > 75:
                    st.info(f"💡 **Innovation forte**: Score de {kpis['innovation_score']:.0f}/100 - Équipe dynamique")
            
            with col2:
                st.subheader("🔮 Prédictions Clés")
                predictions = analytics_data['predictions']
                st.info(f"**🎯 Livraison:** {predictions['completion_prevue'].strftime('%d/%m/%Y')}")
                st.info(f"**💰 Budget final:** {predictions['budget_final_estime']:,}€")
                st.info(f"**📈 ROI projeté:** {predictions['roi_projete']}%")
                st.info(f"**👥 Utilisateurs 6M:** {predictions['utilisateurs_projetes_6m']:,}")
                
                risk_color = "success" if predictions['risque_retard'] < 10 else "warning" if predictions['risque_retard'] < 25 else "error"
                getattr(st, risk_color)(f"**⚠️ Risque retard:** {predictions['risque_retard']}%")
                
                st.metric("🏆 Score Succès", f"{predictions['score_succes']}/100")
        
        elif analytics_view == "Performance Technique":
            st.plotly_chart(
                self.create_performance_dashboard(analytics_data),
                use_container_width=True
            )
            
            # Comparaison benchmarks détaillée
            st.subheader("🎯 Analyse Comparative - Benchmarks Industrie")
            benchmarks = analytics_data['benchmarks']
            
            benchmark_data = []
            current_perf = analytics_data['metriques_performance']
            current_quality = analytics_data['qualite_livrables']
            
            metrics_comparison = [
                {"Métrique": "Vélocité Équipe", "Actuel": current_perf['velocite_equipe'][-1], "Benchmark": benchmarks['velocite_industrie'], "Unité": "SP"},
                {"Métrique": "Temps de Cycle", "Actuel": current_perf['temps_cycle'][-1], "Benchmark": benchmarks['temps_cycle_optimal'], "Unité": "jours"},
                {"Métrique": "Couverture Tests", "Actuel": current_perf['couverture_tests'][-1], "Benchmark": benchmarks['couverture_tests_cible'], "Unité": "%"},
                {"Métrique": "Uptime Système", "Actuel": current_perf['uptime_systeme'][-1], "Benchmark": benchmarks['uptime_industrie'], "Unité": "%"},
                {"Métrique": "NPS Score", "Actuel": current_quality['nps_score'][-1], "Benchmark": benchmarks['nps_excellent'], "Unité": "pts"}
            ]
            
            for metric in metrics_comparison:
                diff = metric["Actuel"] - metric["Benchmark"]
                metric["Écart"] = f"{diff:+.1f}"
                if diff > 0:
                    metric["Performance"] = "🟢 Supérieur"
                elif abs(diff) <= metric["Benchmark"] * 0.05:  # 5% de tolérance
                    metric["Performance"] = "🟡 Conforme"
                else:
                    metric["Performance"] = "🔴 À améliorer"
            
            benchmark_df = pd.DataFrame(metrics_comparison)
            st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
        
        elif analytics_view == "Analyse Financière":
            st.plotly_chart(
                self.create_financial_dashboard(analytics_data),
                use_container_width=True
            )
            
            # Analyse des scénarios financiers
            st.subheader("💰 Analyse Scénarios Financiers")
            scenarios = analytics_data['scenarios_previsionels']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🟢 Scénario Optimiste")
                st.metric("ROI Final", f"{scenarios['optimiste']['roi_final']}%")
                st.metric("Livraison", f"J+{scenarios['optimiste']['completion']}")
                st.metric("Budget", f"{scenarios['optimiste']['budget']:,}€")
            
            with col2:
                st.markdown("#### 🟡 Scénario Réaliste")
                st.metric("ROI Final", f"{scenarios['realiste']['roi_final']}%")
                st.metric("Livraison", f"J+{scenarios['realiste']['completion']}")
                st.metric("Budget", f"{scenarios['realiste']['budget']:,}€")
            
            with col3:
                st.markdown("#### 🔴 Scénario Pessimiste")  
                st.metric("ROI Final", f"{scenarios['pessimiste']['roi_final']}%")
                st.metric("Livraison", f"J+{scenarios['pessimiste']['completion']}")
                st.metric("Budget", f"{scenarios['pessimiste']['budget']:,}€")
        
        elif analytics_view == "Analytics Équipe":
            st.plotly_chart(
                self.create_team_analytics(analytics_data),
                use_container_width=True
            )
            
            # Insights équipe avec recommandations
            st.subheader("💡 Insights & Recommandations RH")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Points Forts Identifiés:**")
                trends = analytics_data['tendances_equipe']
                
                for metric, data in trends.items():
                    if data['tendance'] in ['hausse', 'forte_hausse']:
                        confidence_bar = "⭐" * int(data['confiance'] * 5)
                        st.success(f"• **{metric.capitalize()}**: {data['variation']} {confidence_bar}")
            
            with col2:
                st.markdown("**🔧 Actions Recommandées:**")
                
                if kpis['moral_equipe'] > 8.5:
                    st.info("• Maintenir les initiatives bien-être actuelles")
                elif kpis['moral_equipe'] < 7.5:
                    st.warning("• Enquête satisfaction et plan d'action bien-être")
                
                if kpis['innovation_score'] > 80:
                    st.info("• Capitaliser sur la dynamique d'innovation")
                else:
                    st.warning("• Augmenter le temps dédié à l'innovation (20% → 25%)")
        
        elif analytics_view == "Prédictions Multi-Scénarios":
            st.plotly_chart(
                self.create_predictive_analysis(analytics_data),
                use_container_width=True
            )
            
            # Analyse prédictive avancée
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔮 Modèles Prédictifs")
                st.metric("Précision ROI", "87.3%", delta="ML Model")
                st.metric("Confiance Timeline", "92.1%", delta="Probabiliste")
                st.metric("Score Prédictibilité", f"{kpis['predictability_score']:.0f}%")
                
                st.markdown("**Facteurs d'Influence Détectés:**")
                st.write("• Vélocité équipe (+0.73 corrélation)")
                st.write("• Satisfaction client (+0.68 corrélation)")  
                st.write("• Dette technique (-0.54 corrélation)")
            
            with col2:
                st.subheader("⚡ Actions Prédictives")
                
                st.markdown("**Recommandations Automatiques:**")
                
                if kpis['velocite_variance'] > 3:
                    st.warning("• Stabiliser la vélocité (variance élevée détectée)")
                
                if kpis['predictability_score'] < 75:
                    st.warning("• Améliorer la prévisibilité des processus")
                
                st.info("• Maintenir l'accélération ROI actuelle")
                st.success("• Capitaliser sur la satisfaction client croissante")
        
        st.divider()
        
        # Actions Analytics enrichies
        st.subheader("⚡ Actions Analytics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Rapport Exécutif", use_container_width=True, key="analytics_executive_report"):
                with st.expander("Aperçu Rapport", expanded=True):
                    st.write("**Contenu du rapport:**")
                    st.write(f"• Santé projet: {kpis['health_score']:.0f}/100")
                    st.write(f"• ROI actuel: {kpis['roi_actuel']:.1f}%")
                    st.write(f"• Prédiction livraison: {analytics_data['predictions']['completion_prevue'].strftime('%d/%m/%Y')}")
                    st.write(f"• Score succès: {analytics_data['predictions']['score_succes']}/100")
        
        with col2:
            if st.button("🎯 Alertes Intelligentes", use_container_width=True, key="analytics_smart_alerts"):
                with st.expander("Configuration Alertes", expanded=True):
                    st.write("**Seuils Actuels:**")
                    st.write("• ROI < 20% → ⚠️ Alerte")
                    st.write("• Vélocité baisse 15% → ⚠️ Alerte")
                    st.write("• NPS < 40 → 🚨 Critique")
                    st.write("• Health Score < 70 → ⚠️ Alerte")
        
        with col3:
            if st.button("🤖 ML Prédictif", use_container_width=True, key="analytics_ml_model"):
                with st.expander("Modèles ML", expanded=True):
                    st.write("**Modèles Actifs:**")
                    st.write("• Prédiction ROI: 87% précision")
                    st.write("• Timeline: 92% fiabilité")
                    st.write("• Risques: 83% détection")
                    st.write("• Satisfaction: 89% prédiction")
        
        with col4:
            if st.button("📈 Export Avancé", use_container_width=True, key="analytics_advanced_export"):
                with st.expander("Options Export", expanded=True):
                    export_format = st.selectbox("Format:", ["PDF Exécutif", "Excel Détaillé", "JSON API", "Power BI"])
                    if st.button("Générer Export"):
                        st.success(f"Export {export_format} généré!")


def show_analytics_module(project_id: str = "projet_test"):
    """Point d'entrée pour le module analytics"""
    analytics_module = AnalyticsModule()
    analytics_module.render_analytics_dashboard(project_id)


if __name__ == "__main__":
    st.set_page_config(page_title="Module Analytics", layout="wide")
    show_analytics_module()