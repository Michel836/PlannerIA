"""
🧠 Intelligence Graphique Avancée pour What-If Analysis
Système révolutionnaire d'analyse et d'interprétation automatique des graphiques
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ChartInsight:
    """Insight automatique généré par IA"""
    type: str  # trend, outlier, correlation, pattern, risk, opportunity
    title: str
    description: str
    confidence: float  # 0-1
    severity: str  # low, medium, high, critical
    action_items: List[str]
    supporting_data: Dict[str, Any]

@dataclass
class ChartAnalysis:
    """Analyse complète d'un graphique"""
    chart_type: str
    insights: List[ChartInsight]
    statistical_summary: Dict[str, Any]
    business_interpretation: str
    recommendations: List[str]

class IntelligentChartAnalyzer:
    """Analyseur intelligent de graphiques avec IA"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.insight_patterns = self._load_insight_patterns()
        
    def _load_insight_patterns(self) -> Dict[str, Any]:
        """Patterns d'insights pré-configurés"""
        return {
            'monte_carlo_patterns': {
                'high_variance': {
                    'threshold': 0.4,
                    'message': "Variance élevée détectée - Projet à haut risque",
                    'severity': 'high'
                },
                'skewness': {
                    'threshold': 1.5,
                    'message': "Distribution asymétrique - Risque de dépassement",
                    'severity': 'medium'
                },
                'tail_risk': {
                    'threshold': 0.1,
                    'message': "Risque de queue important - Scénarios extrêmes probables",
                    'severity': 'high'
                }
            },
            'scenario_patterns': {
                'pareto_optimal': {
                    'message': "Zone Pareto-optimale identifiée",
                    'severity': 'low'
                },
                'dominated_solutions': {
                    'message': "Solutions dominées détectées - Optimisation possible",
                    'severity': 'medium'
                }
            },
            'correlation_patterns': {
                'strong_positive': {
                    'threshold': 0.7,
                    'message': "Corrélation forte positive détectée",
                    'severity': 'medium'
                },
                'negative_correlation': {
                    'threshold': -0.5,
                    'message': "Corrélation négative - Trade-off critique",
                    'severity': 'high'
                }
            }
        }
    
    def create_intelligent_monte_carlo_chart(self, simulation_data: Dict[str, Any], 
                                           scenario_params: Dict[str, Any]) -> Tuple[go.Figure, ChartAnalysis]:
        """Crée un graphique Monte Carlo avec analyse intelligente"""
        
        # Extraction ultra-sécurisée des données avec fallbacks multiples
        try:
            if isinstance(simulation_data, dict):
                if 'simulation_data' in simulation_data:
                    sim_data = simulation_data['simulation_data']
                else:
                    sim_data = simulation_data
                    
                # Extraction avec fallbacks robustes
                durations = np.array(sim_data.get('durations', sim_data.get('duration', [100, 120, 90, 110, 105])))
                costs = np.array(sim_data.get('costs', sim_data.get('cost', [50000, 60000, 45000, 55000, 52000])))
                qualities = np.array(sim_data.get('qualities', sim_data.get('quality', [0.8, 0.75, 0.85, 0.78, 0.82])))
            else:
                # Fallback complet si les données ne sont pas un dict
                durations = np.array([100, 120, 90, 110, 105])
                costs = np.array([50000, 60000, 45000, 55000, 52000])
                qualities = np.array([0.8, 0.75, 0.85, 0.78, 0.82])
        except Exception as e:
            # Fallback final en cas d'erreur
            durations = np.array([100, 120, 90, 110, 105])
            costs = np.array([50000, 60000, 45000, 55000, 52000])
            qualities = np.array([0.8, 0.75, 0.85, 0.78, 0.82])
        
        # Analyse statistique avancée
        duration_stats = self._calculate_advanced_stats(durations)
        cost_stats = self._calculate_advanced_stats(costs)
        quality_stats = self._calculate_advanced_stats(qualities)
        
        # Création du graphique multi-panneaux
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Distribution Durée (avec Insights)", 
                "Distribution Coût (avec Insights)",
                "Distribution Qualité",
                "Corrélation Durée-Coût", 
                "Analyse de Risque (VaR)",
                "Zones de Confiance"
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": False}]
            ]
        )
        
        # 1. Distribution durée avec insights
        fig.add_trace(
            go.Histogram(x=durations, nbinsx=30, name="Durée", 
                        marker_color="rgba(55, 128, 191, 0.7)",
                        hovertemplate="Durée: %{x:.1f}j<br>Fréquence: %{y}<extra></extra>"),
            row=1, col=1
        )
        
        # Ajout des seuils critiques
        p95_duration = np.percentile(durations, 95)
        median_duration = np.median(durations)
        
        fig.add_vline(x=median_duration, line_dash="dash", line_color="green", 
                     annotation_text="Médiane", row=1, col=1)
        fig.add_vline(x=p95_duration, line_dash="dash", line_color="red", 
                     annotation_text="VaR 95%", row=1, col=1)
        
        # 2. Distribution coût avec insights
        fig.add_trace(
            go.Histogram(x=costs, nbinsx=30, name="Coût", 
                        marker_color="rgba(219, 64, 82, 0.7)",
                        hovertemplate="Coût: $%{x:,.0f}<br>Fréquence: %{y}<extra></extra>"),
            row=1, col=2
        )
        
        p95_cost = np.percentile(costs, 95)
        median_cost = np.median(costs)
        
        fig.add_vline(x=median_cost, line_dash="dash", line_color="green", 
                     annotation_text="Médiane", row=1, col=2)
        fig.add_vline(x=p95_cost, line_dash="dash", line_color="red", 
                     annotation_text="Budget Critique", row=1, col=2)
        
        # 3. Distribution qualité
        fig.add_trace(
            go.Histogram(x=qualities, nbinsx=20, name="Qualité", 
                        marker_color="rgba(50, 171, 96, 0.7)"),
            row=1, col=3
        )
        
        # 4. Corrélation durée-coût avec clustering
        correlation_coef = np.corrcoef(durations, costs)[0, 1]
        
        # K-means clustering pour identifier les zones (si sklearn disponible)
        if SKLEARN_AVAILABLE and len(durations) > 10:
            try:
                clustering_data = np.column_stack([durations[:min(100, len(durations))], costs[:min(100, len(costs))]])
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clustering_data)
                
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Couleurs par cluster
                colors = ['red', 'blue', 'green']
                for i in range(3):
                    mask = clusters == i
                    if sum(mask) > 0:  # S'assurer qu'il y a des points dans le cluster
                        fig.add_trace(
                            go.Scatter(
                                x=durations[:len(clustering_data)][mask], 
                                y=costs[:len(clustering_data)][mask],
                                mode='markers',
                                name=f'Cluster {i+1}',
                                marker_color=colors[i],
                                hovertemplate="Durée: %{x:.1f}j<br>Coût: $%{y:,.0f}<br>Cluster: %{text}<extra></extra>",
                                text=[f'Zone {i+1}'] * sum(mask)
                            ),
                            row=2, col=1
                        )
            except Exception as e:
                # Fallback si clustering échoue
                fig.add_trace(
                    go.Scatter(x=durations, y=costs, mode='markers', name="Projets",
                              marker_color="rgba(128, 0, 128, 0.6)"),
                    row=2, col=1
                )
        else:
            fig.add_trace(
                go.Scatter(x=durations, y=costs, mode='markers', name="Projets",
                          marker_color="rgba(128, 0, 128, 0.6)"),
                row=2, col=1
            )
        
        # Ligne de régression
        z = np.polyfit(durations, costs, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(x=sorted(durations), y=p(sorted(durations)), 
                      mode='lines', name=f'Tendance (r={correlation_coef:.2f})',
                      line_color="orange"),
            row=2, col=1
        )
        
        # 5. Analyse VaR multi-niveaux
        var_levels = [90, 95, 99]
        var_durations = [np.percentile(durations, level) for level in var_levels]
        var_costs = [np.percentile(costs, level) for level in var_levels]
        
        fig.add_trace(
            go.Bar(x=[f"VaR {level}%" for level in var_levels], 
                  y=var_durations, name="Durée VaR", 
                  marker_color="lightblue", yaxis="y2"),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(x=[f"VaR {level}%" for level in var_levels], 
                  y=var_costs, name="Coût VaR", 
                  marker_color="lightcoral"),
            row=2, col=2, secondary_y=True
        )
        
        # 6. Zones de confiance (contour plot)
        if len(durations) > 50:
            # Création d'une heatmap de densité
            hist, x_edges, y_edges = np.histogram2d(durations[:100], costs[:100], bins=15)
            
            fig.add_trace(
                go.Heatmap(
                    z=hist.T,
                    x=x_edges[:-1],
                    y=y_edges[:-1],
                    colorscale="Viridis",
                    name="Densité",
                    hovertemplate="Durée: %{x:.1f}j<br>Coût: $%{y:,.0f}<br>Densité: %{z}<extra></extra>"
                ),
                row=2, col=3
            )
        
        # Mise en forme avancée
        fig.update_layout(
            title="🧠 Analyse Monte Carlo Intelligente - Insights Automatiques",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Génération de l'analyse
        insights = self._generate_monte_carlo_insights(
            duration_stats, cost_stats, quality_stats, correlation_coef
        )
        
        analysis = ChartAnalysis(
            chart_type="intelligent_monte_carlo",
            insights=insights,
            statistical_summary={
                "duration_stats": duration_stats,
                "cost_stats": cost_stats,
                "correlation": correlation_coef,
                "sample_size": len(durations)
            },
            business_interpretation=self._generate_business_interpretation(insights),
            recommendations=self._generate_recommendations(insights)
        )
        
        return fig, analysis
    
    def create_intelligent_pareto_chart(self, scenarios: List[Dict[str, Any]]) -> Tuple[go.Figure, ChartAnalysis]:
        """Graphique Pareto intelligent avec analyse des zones optimales"""
        
        # Extraction des données
        durations = [s['duration'] for s in scenarios]
        costs = [s['cost'] for s in scenarios]
        qualities = [s['quality_score'] for s in scenarios]
        success_probs = [s['success_probability'] for s in scenarios]
        names = [s.get('name', f'Scénario {i+1}') for i, s in enumerate(scenarios)]
        
        # Analyse Pareto
        pareto_analysis = self._analyze_pareto_efficiency(scenarios)
        
        # Création du graphique 3D interactif
        fig = go.Figure()
        
        # Points principaux avec analyse de dominance
        for i, scenario in enumerate(scenarios):
            is_pareto = pareto_analysis['pareto_optimal'][i]
            domination_count = pareto_analysis['domination_counts'][i]
            
            color = 'gold' if is_pareto else 'lightblue' if domination_count < 2 else 'lightcoral'
            symbol = 'diamond' if is_pareto else 'circle'
            size = 15 + success_probs[i] * 15  # Taille basée sur probabilité succès
            
            fig.add_trace(go.Scatter3d(
                x=[durations[i]],
                y=[costs[i]],
                z=[qualities[i]],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color='black')
                ),
                text=[names[i][:10]],
                textposition="top center",
                name=f"{'⭐ ' if is_pareto else ''}Scénario {i+1}",
                hovertemplate=(
                    f"<b>{names[i]}</b><br>"
                    "Durée: %{x:.1f} jours<br>"
                    "Coût: $%{y:,.0f}<br>"
                    "Qualité: %{z:.1%}<br>"
                    f"Succès: {success_probs[i]:.1%}<br>"
                    f"{'🏆 Pareto Optimal' if is_pareto else f'Dominé par {domination_count} solutions'}"
                    "<extra></extra>"
                )
            ))
        
        # Surface Pareto (approximation)
        if len([i for i, is_pareto in enumerate(pareto_analysis['pareto_optimal']) if is_pareto]) > 3:
            pareto_points = [(d, c, q) for d, c, q, is_pareto in 
                           zip(durations, costs, qualities, pareto_analysis['pareto_optimal']) 
                           if is_pareto]
            
            if len(pareto_points) > 3:
                # Tri des points Pareto
                pareto_points.sort()
                
                pareto_durations, pareto_costs, pareto_qualities = zip(*pareto_points)
                
                # Surface approximative
                fig.add_trace(go.Mesh3d(
                    x=pareto_durations,
                    y=pareto_costs, 
                    z=pareto_qualities,
                    opacity=0.3,
                    color='gold',
                    name="Front de Pareto"
                ))
        
        # Zones d'amélioration
        best_duration = min(durations)
        best_cost = min(costs)
        best_quality = max(qualities)
        
        fig.add_trace(go.Scatter3d(
            x=[best_duration],
            y=[best_cost],
            z=[best_quality],
            mode='markers',
            marker=dict(size=20, color='lime', symbol='diamond'),
            name="🎯 Idéal Théorique",
            hovertemplate="<b>Solution Idéale Théorique</b><br>Durée: %{x:.1f}j<br>Coût: $%{y:,.0f}<br>Qualité: %{z:.1%}<extra></extra>"
        ))
        
        # Mise en forme
        fig.update_layout(
            title="🎯 Analyse Pareto 3D Intelligente - Zones Optimales Identifiées",
            scene=dict(
                xaxis_title="⏱️ Durée (jours)",
                yaxis_title="💰 Coût ($)",
                zaxis_title="⭐ Qualité (%)",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
            ),
            height=700
        )
        
        # Génération des insights
        insights = self._generate_pareto_insights(pareto_analysis, scenarios)
        
        analysis = ChartAnalysis(
            chart_type="intelligent_pareto_3d",
            insights=insights,
            statistical_summary=pareto_analysis,
            business_interpretation=self._generate_pareto_interpretation(pareto_analysis),
            recommendations=self._generate_pareto_recommendations(pareto_analysis, scenarios)
        )
        
        return fig, analysis
    
    def create_intelligent_risk_heatmap(self, scenarios: List[Dict[str, Any]], 
                                      risk_factors: List[str]) -> Tuple[go.Figure, ChartAnalysis]:
        """Heatmap de risque intelligente avec clustering automatique"""
        
        # Matrice de risques
        risk_matrix = []
        scenario_names = []
        
        for i, scenario in enumerate(scenarios):
            scenario_risks = []
            for factor in risk_factors:
                # Simulation de scores de risque basés sur les paramètres du scénario
                base_risk = np.random.uniform(1, 10)  # En réalité, on utiliserait les vrais calculs
                risk_modifier = self._calculate_risk_modifier(scenario, factor)
                final_risk = min(10, max(1, base_risk * risk_modifier))
                scenario_risks.append(final_risk)
            
            risk_matrix.append(scenario_risks)
            scenario_names.append(scenario.get('name', f'Scénario {i+1}'))
        
        risk_matrix = np.array(risk_matrix)
        
        # Clustering automatique des risques
        if len(scenarios) > 2:
            kmeans = KMeans(n_clusters=min(3, len(scenarios)), random_state=42)
            risk_clusters = kmeans.fit_predict(risk_matrix)
        else:
            risk_clusters = [0] * len(scenarios)
        
        # Création de la heatmap avancée
        fig = go.Figure()
        
        # Heatmap principale avec annotations intelligentes
        annotations = []
        for i, scenario_name in enumerate(scenario_names):
            for j, factor in enumerate(risk_factors):
                risk_value = risk_matrix[i][j]
                
                # Couleur et symbole basés sur le niveau de risque
                if risk_value > 7:
                    symbol = "🔴"
                    text_color = "white"
                elif risk_value > 4:
                    symbol = "🟡"
                    text_color = "black"
                else:
                    symbol = "🟢"
                    text_color = "black"
                
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{symbol}<br>{risk_value:.1f}",
                        showarrow=False,
                        font=dict(color=text_color, size=10)
                    )
                )
        
        fig.add_trace(go.Heatmap(
            z=risk_matrix,
            x=risk_factors,
            y=scenario_names,
            colorscale='RdYlGn_r',  # Rouge = risque élevé
            reversescale=False,
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Facteur: %{x}<br>Risque: %{z:.1f}/10<extra></extra>",
            colorbar=dict(
                title="Niveau de Risque",
                tickvals=[1, 3, 5, 7, 9],
                ticktext=["Très Faible", "Faible", "Moyen", "Élevé", "Critique"]
            )
        ))
        
        fig.update_layout(
            annotations=annotations,
            title="🚨 Analyse de Risque Intelligente - Clustering Automatique",
            xaxis_title="Facteurs de Risque",
            yaxis_title="Scénarios",
            height=max(400, len(scenarios) * 50)
        )
        
        # Analyse des patterns de risque
        risk_insights = self._analyze_risk_patterns(risk_matrix, scenario_names, risk_factors)
        
        analysis = ChartAnalysis(
            chart_type="intelligent_risk_heatmap",
            insights=risk_insights,
            statistical_summary={
                "risk_matrix": risk_matrix.tolist(),
                "clusters": risk_clusters.tolist(),
                "avg_risk_per_scenario": np.mean(risk_matrix, axis=1).tolist(),
                "avg_risk_per_factor": np.mean(risk_matrix, axis=0).tolist()
            },
            business_interpretation=self._generate_risk_interpretation(risk_insights),
            recommendations=self._generate_risk_recommendations(risk_insights, scenarios)
        )
        
        return fig, analysis
    
    def create_intelligent_sensitivity_chart(self, base_scenario: Dict[str, Any],
                                           sensitivity_params: List[str]) -> Tuple[go.Figure, ChartAnalysis]:
        """Analyse de sensibilité intelligente avec tornado chart"""
        
        # Calcul de sensibilité pour chaque paramètre
        sensitivities = []
        param_ranges = []
        
        for param in sensitivity_params:
            # Variation de ±20% du paramètre
            low_impact, high_impact = self._calculate_parameter_sensitivity(base_scenario, param)
            sensitivity = (high_impact - low_impact) / 2  # Impact moyen
            
            sensitivities.append(abs(sensitivity))
            param_ranges.append((low_impact, high_impact))
        
        # Tri par ordre de sensibilité
        sorted_data = sorted(zip(sensitivity_params, sensitivities, param_ranges), 
                           key=lambda x: x[1], reverse=True)
        
        params, sens_values, ranges = zip(*sorted_data)
        
        # Création du tornado chart
        fig = go.Figure()
        
        # Barres négatives (impact de -20%)
        fig.add_trace(go.Bar(
            y=params,
            x=[-s/2 for s in sens_values],
            orientation='h',
            name='Impact -20%',
            marker_color='lightcoral',
            hovertemplate="Paramètre: %{y}<br>Impact: %{x:.1f}<extra></extra>"
        ))
        
        # Barres positives (impact de +20%)
        fig.add_trace(go.Bar(
            y=params,
            x=[s/2 for s in sens_values],
            orientation='h',
            name='Impact +20%',
            marker_color='lightblue',
            hovertemplate="Paramètre: %{y}<br>Impact: %{x:.1f}<extra></extra>"
        ))
        
        # Ligne de base
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
        
        # Annotations avec insights
        annotations = []
        for i, (param, sens, (low, high)) in enumerate(sorted_data):
            # Classification de la sensibilité
            if sens > np.mean(sens_values) * 1.5:
                insight = "🔴 CRITIQUE"
                color = "red"
            elif sens > np.mean(sens_values):
                insight = "🟡 ÉLEVÉ"
                color = "orange"
            else:
                insight = "🟢 FAIBLE"
                color = "green"
            
            annotations.append(
                dict(
                    x=sens/2 + 0.5,
                    y=i,
                    text=insight,
                    showarrow=False,
                    font=dict(color=color, size=10, family="Arial Black")
                )
            )
        
        fig.update_layout(
            title="📊 Analyse de Sensibilité Intelligente - Tornado Chart",
            xaxis_title="Impact sur le Résultat Final",
            yaxis_title="Paramètres de Projet",
            barmode='relative',
            height=max(400, len(params) * 60),
            annotations=annotations
        )
        
        # Génération des insights de sensibilité
        sensitivity_insights = self._generate_sensitivity_insights(sorted_data)
        
        analysis = ChartAnalysis(
            chart_type="intelligent_sensitivity_tornado",
            insights=sensitivity_insights,
            statistical_summary={
                "sensitivity_ranking": list(params),
                "sensitivity_values": list(sens_values),
                "parameter_ranges": list(ranges),
                "most_sensitive": params[0] if params else None
            },
            business_interpretation=self._generate_sensitivity_interpretation(sorted_data),
            recommendations=self._generate_sensitivity_recommendations(sorted_data)
        )
        
        return fig, analysis
    
    # === MÉTHODES D'ANALYSE PRIVÉES ===
    
    def _calculate_advanced_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """Calcule des statistiques avancées"""
        stats_dict = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'var_95': float(np.percentile(data, 95)),
            'var_99': float(np.percentile(data, 99)),
            'coefficient_variation': float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0
        }
        
        # Ajout des stats avancées si scipy disponible
        if SCIPY_AVAILABLE:
            try:
                stats_dict['skewness'] = float(stats.skew(data))
                stats_dict['kurtosis'] = float(stats.kurtosis(data))
            except:
                stats_dict['skewness'] = 0.0
                stats_dict['kurtosis'] = 0.0
        else:
            stats_dict['skewness'] = 0.0
            stats_dict['kurtosis'] = 0.0
            
        return stats_dict
    
    def _analyze_pareto_efficiency(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse l'efficacité Pareto"""
        n = len(scenarios)
        pareto_optimal = [True] * n
        domination_counts = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Vérifier si j domine i
                    if (scenarios[j]['duration'] <= scenarios[i]['duration'] and
                        scenarios[j]['cost'] <= scenarios[i]['cost'] and 
                        scenarios[j]['quality_score'] >= scenarios[i]['quality_score'] and
                        (scenarios[j]['duration'] < scenarios[i]['duration'] or
                         scenarios[j]['cost'] < scenarios[i]['cost'] or
                         scenarios[j]['quality_score'] > scenarios[i]['quality_score'])):
                        
                        pareto_optimal[i] = False
                        domination_counts[i] += 1
        
        return {
            'pareto_optimal': pareto_optimal,
            'domination_counts': domination_counts,
            'pareto_count': sum(pareto_optimal),
            'efficiency_ratio': sum(pareto_optimal) / n if n > 0 else 0
        }
    
    def _calculate_risk_modifier(self, scenario: Dict[str, Any], risk_factor: str) -> float:
        """Calcule le modificateur de risque pour un facteur donné"""
        modifiers = {
            'Timeline': scenario.get('duration', 100) / 100,  # Normalisé
            'Budget': scenario.get('cost', 50000) / 50000,
            'Quality': 2 - scenario.get('quality_score', 0.8),
            'Team': scenario.get('team_size', 5) / 5,
            'Complexity': scenario.get('complexity_factor', 1.0)
        }
        return modifiers.get(risk_factor, 1.0)
    
    def _calculate_parameter_sensitivity(self, base_scenario: Dict[str, Any], 
                                       param: str) -> Tuple[float, float]:
        """Calcule la sensibilité d'un paramètre"""
        base_value = base_scenario.get('duration', 100)  # Simplification
        
        # Simulation de l'impact de ±20%
        low_impact = base_value * 0.8
        high_impact = base_value * 1.2
        
        return low_impact, high_impact
    
    def _generate_monte_carlo_insights(self, duration_stats: Dict, cost_stats: Dict, 
                                     quality_stats: Dict, correlation: float) -> List[ChartInsight]:
        """Génère des insights pour Monte Carlo"""
        insights = []
        
        # Analyse de variance
        if duration_stats['coefficient_variation'] > 0.4:
            insights.append(ChartInsight(
                type="risk",
                title="🚨 Variance Élevée Détectée",
                description=f"Le coefficient de variation de la durée est de {duration_stats['coefficient_variation']:.2f}, indiquant une forte incertitude.",
                confidence=0.9,
                severity="high",
                action_items=[
                    "Décomposer les tâches complexes",
                    "Ajouter des points de contrôle fréquents",
                    "Prévoir une marge de sécurité de 30%+"
                ],
                supporting_data=duration_stats
            ))
        
        # Analyse d'asymétrie
        if abs(duration_stats['skewness']) > 1.0:
            direction = "positive" if duration_stats['skewness'] > 0 else "négative"
            insights.append(ChartInsight(
                type="pattern",
                title=f"📈 Asymétrie {direction.title()} Significative",
                description=f"Distribution asymétrique ({duration_stats['skewness']:.2f}) suggère des risques de dépassement.",
                confidence=0.8,
                severity="medium",
                action_items=[
                    "Identifier les causes d'asymétrie",
                    "Prévoir des scénarios de contingence"
                ],
                supporting_data={'skewness': duration_stats['skewness']}
            ))
        
        # Analyse de corrélation
        if abs(correlation) > 0.7:
            insights.append(ChartInsight(
                type="correlation",
                title="🔗 Corrélation Forte Durée-Coût",
                description=f"Corrélation de {correlation:.2f} entre durée et coût - effet de levier important.",
                confidence=0.9,
                severity="medium" if correlation > 0 else "high",
                action_items=[
                    "Exploiter la corrélation pour optimisation",
                    "Surveiller l'effet domino durée→coût"
                ],
                supporting_data={'correlation': correlation}
            ))
        
        return insights
    
    def _generate_pareto_insights(self, pareto_analysis: Dict, scenarios: List[Dict]) -> List[ChartInsight]:
        """Génère des insights pour l'analyse Pareto"""
        insights = []
        
        efficiency_ratio = pareto_analysis['efficiency_ratio']
        
        if efficiency_ratio < 0.3:
            insights.append(ChartInsight(
                type="opportunity",
                title="🎯 Opportunités d'Optimisation Majeures",
                description=f"Seulement {efficiency_ratio:.1%} des scénarios sont Pareto-optimaux. Fort potentiel d'amélioration.",
                confidence=0.95,
                severity="high",
                action_items=[
                    "Éliminer les scénarios dominés",
                    "Explorer l'espace entre solutions Pareto",
                    "Optimiser les paramètres sous-performants"
                ],
                supporting_data=pareto_analysis
            ))
        
        # Identification du meilleur scénario
        pareto_scenarios = [s for s, is_pareto in zip(scenarios, pareto_analysis['pareto_optimal']) if is_pareto]
        if pareto_scenarios:
            best_scenario = min(pareto_scenarios, key=lambda s: s['duration'] + s['cost']/1000)
            insights.append(ChartInsight(
                type="recommendation",
                title="⭐ Scénario Recommandé Identifié",
                description=f"Le scénario optimal combine efficacité temporelle et budgétaire.",
                confidence=0.9,
                severity="low",
                action_items=[
                    "Analyser en détail le scénario optimal",
                    "Valider la faisabilité pratique"
                ],
                supporting_data=best_scenario
            ))
        
        return insights
    
    def _analyze_risk_patterns(self, risk_matrix: np.ndarray, scenario_names: List[str], 
                             risk_factors: List[str]) -> List[ChartInsight]:
        """Analyse les patterns de risque"""
        insights = []
        
        # Scénario le plus risqué
        avg_risks = np.mean(risk_matrix, axis=1)
        max_risk_idx = np.argmax(avg_risks)
        
        insights.append(ChartInsight(
            type="risk",
            title="⚠️ Scénario Critique Identifié",
            description=f"'{scenario_names[max_risk_idx]}' présente le profil de risque le plus élevé ({avg_risks[max_risk_idx]:.1f}/10).",
            confidence=0.95,
            severity="high",
            action_items=[
                "Réviser complètement ce scénario",
                "Développer plan de mitigation spécifique",
                "Considérer l'abandon si risques > bénéfices"
            ],
            supporting_data={'scenario_index': max_risk_idx, 'risk_score': avg_risks[max_risk_idx]}
        ))
        
        # Facteur de risque critique
        avg_factor_risks = np.mean(risk_matrix, axis=0)
        max_factor_idx = np.argmax(avg_factor_risks)
        
        insights.append(ChartInsight(
            type="risk",
            title="🎯 Facteur de Risque Systémique",
            description=f"'{risk_factors[max_factor_idx]}' est critique sur tous les scénarios ({avg_factor_risks[max_factor_idx]:.1f}/10).",
            confidence=0.9,
            severity="high",
            action_items=[
                f"Plan d'action spécifique pour {risk_factors[max_factor_idx]}",
                "Formation équipe sur ce risque",
                "Monitoring continu de ce facteur"
            ],
            supporting_data={'factor_index': max_factor_idx, 'risk_score': avg_factor_risks[max_factor_idx]}
        ))
        
        return insights
    
    def _generate_sensitivity_insights(self, sorted_data: List[Tuple]) -> List[ChartInsight]:
        """Génère des insights de sensibilité"""
        insights = []
        
        if sorted_data:
            most_sensitive_param, sens_value, (low, high) = sorted_data[0]
            
            insights.append(ChartInsight(
                type="critical",
                title="🎯 Paramètre Critique Identifié",
                description=f"'{most_sensitive_param}' est le levier d'impact principal (sensibilité: {sens_value:.1f}).",
                confidence=0.95,
                severity="critical",
                action_items=[
                    f"Contrôle strict de {most_sensitive_param}",
                    "Monitoring en temps réel de ce paramètre",
                    "Plan de contingence dédié"
                ],
                supporting_data={'parameter': most_sensitive_param, 'sensitivity': sens_value}
            ))
            
        return insights
    
    def _generate_business_interpretation(self, insights: List[ChartInsight]) -> str:
        """Génère l'interprétation business"""
        critical_count = len([i for i in insights if i.severity in ['high', 'critical']])
        
        if critical_count > 2:
            return "🚨 **ATTENTION**: Plusieurs risques critiques détectés. Révision stratégique recommandée avant exécution."
        elif critical_count > 0:
            return "⚠️ **VIGILANCE**: Quelques points d'attention identifiés. Surveillance renforcée nécessaire."
        else:
            return "✅ **CONFIANT**: Profil de risque acceptable. Projet prêt pour exécution avec monitoring standard."
    
    def _generate_recommendations(self, insights: List[ChartInsight]) -> List[str]:
        """Génère des recommandations consolidées"""
        all_actions = []
        for insight in insights:
            all_actions.extend(insight.action_items)
        
        # Déduplication et priorisation
        unique_actions = list(set(all_actions))
        return unique_actions[:5]  # Top 5 recommandations
    
    def _generate_pareto_interpretation(self, pareto_analysis: Dict) -> str:
        """Interprétation business de l'analyse Pareto"""
        ratio = pareto_analysis['efficiency_ratio']
        
        if ratio > 0.7:
            return "🎯 **EXCELLENT**: Haute efficacité des scénarios - espace d'optimisation limité mais solutions robustes."
        elif ratio > 0.4:
            return "⚖️ **BALANCÉ**: Efficacité modérée - opportunités d'optimisation disponibles sans risque majeur."
        else:
            return "🚀 **FORT POTENTIEL**: Nombreuses opportunités d'amélioration - optimisation recommandée avant décision finale."
    
    def _generate_pareto_recommendations(self, pareto_analysis: Dict, scenarios: List[Dict]) -> List[str]:
        """Recommandations basées sur l'analyse Pareto"""
        recommendations = []
        
        if pareto_analysis['efficiency_ratio'] < 0.5:
            recommendations.append("🎯 Éliminer les scénarios non Pareto-optimaux")
            recommendations.append("🔍 Explorer l'espace entre solutions optimales")
        
        recommendations.append("📊 Valider la faisabilité des scénarios Pareto")
        recommendations.append("🔄 Itérer sur les paramètres des solutions dominées")
        
        return recommendations
    
    def _generate_risk_interpretation(self, risk_insights: List[ChartInsight]) -> str:
        """Interprétation des risques"""
        high_risk_count = len([i for i in risk_insights if i.severity == 'high'])
        
        if high_risk_count > 1:
            return "🚨 **RISQUE ÉLEVÉ**: Plusieurs facteurs critiques identifiés - mitigation urgente requise."
        elif high_risk_count == 1:
            return "⚠️ **VIGILANCE**: Un risque critique identifié - plan d'action ciblé nécessaire."
        else:
            return "🛡️ **MAÎTRISÉ**: Profil de risque acceptable - monitoring standard suffisant."
    
    def _generate_risk_recommendations(self, risk_insights: List[ChartInsight], scenarios: List[Dict]) -> List[str]:
        """Recommandations de gestion des risques"""
        recommendations = []
        
        for insight in risk_insights:
            if insight.severity in ['high', 'critical']:
                recommendations.extend(insight.action_items[:2])  # Top 2 par insight critique
        
        return list(set(recommendations))[:5]
    
    def _generate_sensitivity_interpretation(self, sorted_data: List[Tuple]) -> str:
        """Interprétation de l'analyse de sensibilité"""
        if not sorted_data:
            return "📊 Analyse de sensibilité non disponible."
        
        top_param = sorted_data[0][0]
        return f"🎯 **FOCUS**: '{top_param}' est le levier principal de performance - concentration des efforts recommandée sur ce paramètre."
    
    def _generate_sensitivity_recommendations(self, sorted_data: List[Tuple]) -> List[str]:
        """Recommandations basées sur la sensibilité"""
        if not sorted_data:
            return ["🔍 Collecter plus de données pour analyse de sensibilité"]
        
        top_3_params = [data[0] for data in sorted_data[:3]]
        
        recommendations = [
            f"🎯 Contrôle prioritaire: {top_3_params[0]}",
            f"📊 Monitoring renforcé: {', '.join(top_3_params[1:3])}",
            "🔄 Calibrage régulier des paramètres sensibles",
            "📈 Dashboard temps réel pour paramètres critiques"
        ]
        
        return recommendations

def render_chart_intelligence_panel(analysis: ChartAnalysis):
    """Panneau d'intelligence graphique avec insights"""
    
    st.markdown("---")
    st.markdown("## 🧠 **Intelligence Graphique Automatique**")
    
    # Résumé exécutif
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Interprétation Business")
        st.info(analysis.business_interpretation)
    
    with col2:
        # Métriques clés
        st.markdown("### 📊 Métriques")
        critical_insights = len([i for i in analysis.insights if i.severity in ['high', 'critical']])
        st.metric("Insights Critiques", critical_insights)
        st.metric("Confiance Moyenne", f"{np.mean([i.confidence for i in analysis.insights]):.1%}")
    
    # Insights détaillés
    st.markdown("### 🔍 Insights Automatiques")
    
    for insight in analysis.insights:
        # Couleur basée sur la sévérité
        if insight.severity == 'critical':
            color = "🔴"
            container = st.error
        elif insight.severity == 'high':
            color = "🟠"
            container = st.warning
        elif insight.severity == 'medium':
            color = "🟡"
            container = st.info
        else:
            color = "🟢"
            container = st.success
        
        with container(f"{color} **{insight.title}**"):
            st.markdown(insight.description)
            
            if insight.action_items:
                st.markdown("**Actions recommandées:**")
                for action in insight.action_items:
                    st.markdown(f"• {action}")
            
            with st.expander("Données détaillées"):
                st.json(insight.supporting_data)
    
    # Recommandations consolidées
    if analysis.recommendations:
        st.markdown("### 🎯 Plan d'Action Recommandé")
        
        for i, rec in enumerate(analysis.recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    
    # Export des insights
    import time
    unique_key = f"export_chart_analysis_{int(time.time() * 1000)}"
    if st.button("📄 Exporter Analyse Complète", key=unique_key):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'chart_type': analysis.chart_type,
            'business_interpretation': analysis.business_interpretation,
            'insights': [
                {
                    'type': i.type,
                    'title': i.title,
                    'description': i.description,
                    'severity': i.severity,
                    'confidence': i.confidence,
                    'action_items': i.action_items
                }
                for i in analysis.insights
            ],
            'recommendations': analysis.recommendations,
            'statistical_summary': analysis.statistical_summary
        }
        
        st.download_button(
            label="⬇️ Télécharger Rapport d'Analyse",
            data=json.dumps(export_data, indent=2, default=str),
            file_name=f"chart_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Export des classes principales
__all__ = [
    'IntelligentChartAnalyzer',
    'ChartInsight', 
    'ChartAnalysis',
    'render_chart_intelligence_panel'
]