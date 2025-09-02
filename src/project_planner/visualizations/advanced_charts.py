"""
🎨 Module de Visualisations Avancées pour PlannerIA
==================================================

Graphiques interactifs et visuellement impactants pour
maximiser l'impact lors de la session d'évaluation.

Auteur: PlannerIA Team
Date: 2025-08-31
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import colorsys
from datetime import datetime, timedelta


class AdvancedVisualizations:
    """
    🚀 Générateur de Visualisations Premium
    
    Graphiques optimisés pour:
    - Impact visuel maximal
    - Interactivité fluide
    - Pertinence métier
    - Performance temps réel
    """
    
    def __init__(self):
        """Initialise le thème et les couleurs"""
        self.color_palette = {
            'primary': '#1e40af',      # Bleu profond
            'secondary': '#7c3aed',    # Violet
            'success': '#10b981',      # Vert
            'warning': '#f59e0b',      # Orange
            'danger': '#ef4444',       # Rouge
            'info': '#06b6d4',         # Cyan
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        }
        
        self.theme = {
            'background': 'rgba(17, 17, 17, 0)',
            'paper': '#ffffff',
            'font_family': 'Inter, sans-serif',
            'font_color': '#1f2937'
        }
    
    def create_performance_gauge(self, value: float, title: str = "Performance",
                                max_value: float = 100, target: float = 90) -> go.Figure:
        """
        Crée un gauge chart style speedometer pour les métriques de performance
        
        Parfait pour: Score global, Santé système, Accuracy IA
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{title}</b>", 'font': {'size': 24}},
            delta={'reference': target, 'increasing': {'color': self.color_palette['success']}},
            gauge={
                'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': self._get_performance_color(value, max_value)},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, max_value*0.5], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [max_value*0.5, max_value*0.8], 'color': 'rgba(245, 158, 11, 0.1)'},
                    {'range': [max_value*0.8, max_value], 'color': 'rgba(16, 185, 129, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=self.theme['paper'],
            font={'family': self.theme['font_family']}
        )
        
        return fig
    
    def create_ai_systems_heatmap(self, systems_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Crée une heatmap de l'activité des 15 systèmes IA
        
        Parfait pour: Vue d'ensemble des modules actifs
        """
        # Données par défaut si non fournies
        if not systems_data:
            systems_data = {
                'Supervisor': {'activity': 95, 'latency': 12, 'accuracy': 96},
                'Planner': {'activity': 88, 'latency': 18, 'accuracy': 94},
                'Estimator': {'activity': 92, 'latency': 15, 'accuracy': 95},
                'Risk Analyzer': {'activity': 78, 'latency': 22, 'accuracy': 93},
                'Documentation': {'activity': 85, 'latency': 8, 'accuracy': 98},
                'Strategy Advisor': {'activity': 90, 'latency': 25, 'accuracy': 92},
                'Learning Agent': {'activity': 94, 'latency': 10, 'accuracy': 97},
                'Stakeholder Intel': {'activity': 76, 'latency': 20, 'accuracy': 91},
                'Monitor': {'activity': 100, 'latency': 5, 'accuracy': 99},
                'Innovation': {'activity': 82, 'latency': 30, 'accuracy': 89},
                'RAG Manager': {'activity': 96, 'latency': 16, 'accuracy': 95},
                'Coach': {'activity': 88, 'latency': 14, 'accuracy': 94},
                'BI Analytics': {'activity': 91, 'latency': 19, 'accuracy': 93},
                'Comparator': {'activity': 79, 'latency': 17, 'accuracy': 92},
                'Error Handler': {'activity': 87, 'latency': 7, 'accuracy': 98}
            }
        
        systems = list(systems_data.keys())
        metrics = ['Activité %', 'Latence ms', 'Précision %']
        
        # Créer la matrice de données
        z_data = []
        for metric in metrics:
            row = []
            for system in systems:
                if metric == 'Activité %':
                    row.append(systems_data[system].get('activity', 0))
                elif metric == 'Latence ms':
                    # Inverser pour que faible latence = bonne perf
                    row.append(100 - min(100, systems_data[system].get('latency', 0)))
                else:
                    row.append(systems_data[system].get('accuracy', 0))
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=systems,
            y=metrics,
            colorscale='Viridis',
            text=[[f"{val:.0f}" for val in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Performance"),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="<b>🧠 Activité des 15 Systèmes IA en Temps Réel</b>",
            height=300,
            xaxis={'tickangle': -45},
            yaxis={'autorange': 'reversed'},
            font={'family': self.theme['font_family']},
            paper_bgcolor=self.theme['paper']
        )
        
        return fig
    
    def create_workflow_sankey(self, plan_data: Dict[str, Any]) -> go.Figure:
        """
        Crée un diagramme de Sankey du flux de travail multi-agents
        
        Parfait pour: Montrer la coordination des agents
        """
        # Nœuds du workflow
        nodes = [
            "User Input",           # 0
            "Supervisor Agent",     # 1
            "Planner Agent",       # 2
            "Estimator Agent",     # 3
            "Risk Agent",          # 4
            "Documentation Agent", # 5
            "Strategy Advisor",    # 6
            "Learning Agent",      # 7
            "Final Plan"          # 8
        ]
        
        # Liens entre les nœuds (source -> target avec valeur)
        links = [
            (0, 1, 100),  # User -> Supervisor
            (1, 2, 95),   # Supervisor -> Planner
            (1, 3, 90),   # Supervisor -> Estimator
            (1, 4, 85),   # Supervisor -> Risk
            (2, 5, 80),   # Planner -> Documentation
            (2, 6, 75),   # Planner -> Strategy
            (3, 7, 70),   # Estimator -> Learning
            (4, 6, 65),   # Risk -> Strategy
            (5, 8, 90),   # Documentation -> Final
            (6, 8, 85),   # Strategy -> Final
            (7, 8, 80),   # Learning -> Final
        ]
        
        # Couleurs pour les liens
        link_colors = [
            f"rgba({30}, {64}, {175}, 0.4)",   # Bleu
            f"rgba({124}, {58}, {237}, 0.4)",  # Violet
            f"rgba({16}, {185}, {129}, 0.4)",  # Vert
            f"rgba({245}, {158}, {11}, 0.4)",  # Orange
            f"rgba({239}, {68}, {68}, 0.4)",   # Rouge
            f"rgba({6}, {182}, {212}, 0.4)",   # Cyan
            f"rgba({168}, {85}, {247}, 0.4)",  # Purple
            f"rgba({34}, {197}, {94}, 0.4)",   # Green
            f"rgba({251}, {191}, {36}, 0.4)",  # Yellow
            f"rgba({239}, {68}, {68}, 0.4)",   # Red
            f"rgba({59}, {130}, {246}, 0.4)",  # Blue
        ]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=[self.color_palette['primary'] if i == 0 or i == 8 else self.color_palette['secondary'] 
                      for i in range(len(nodes))],
                hovertemplate='<b>%{label}</b><br>Flux total: %{value}<extra></extra>'
            ),
            link=dict(
                source=[link[0] for link in links],
                target=[link[1] for link in links],
                value=[link[2] for link in links],
                color=link_colors,
                hovertemplate='De %{source.label}<br>Vers %{target.label}<br>Flux: %{value}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title="<b>🔄 Flux de Travail Multi-Agents PlannerIA</b>",
            font={'family': self.theme['font_family'], 'size': 11},
            height=500,
            paper_bgcolor=self.theme['paper']
        )
        
        return fig
    
    def create_budget_sunburst(self, plan_data: Dict[str, Any]) -> go.Figure:
        """
        Crée un sunburst chart pour la décomposition hiérarchique du budget
        
        Parfait pour: Visualiser la répartition budget/WBS
        """
        # Extraire les données du plan ou utiliser des données démo
        phases = plan_data.get("wbs", {}).get("phases", []) if plan_data else []
        
        if not phases:
            # Données démo pour illustration
            phases = [
                {
                    "name": "Conception",
                    "cost": 15000,
                    "tasks": [
                        {"name": "Architecture", "cost": 5000},
                        {"name": "Design UX", "cost": 6000},
                        {"name": "Prototypage", "cost": 4000}
                    ]
                },
                {
                    "name": "Développement",
                    "cost": 35000,
                    "tasks": [
                        {"name": "Backend", "cost": 15000},
                        {"name": "Frontend", "cost": 12000},
                        {"name": "API", "cost": 8000}
                    ]
                },
                {
                    "name": "Tests",
                    "cost": 10000,
                    "tasks": [
                        {"name": "Tests unitaires", "cost": 3000},
                        {"name": "Tests intégration", "cost": 4000},
                        {"name": "Tests UAT", "cost": 3000}
                    ]
                }
            ]
        
        # Construire les données pour le sunburst
        labels = ["Projet Total"]
        parents = [""]
        
        # Calculer le coût total du projet à partir des tâches
        total_cost = 0
        for phase in phases:
            phase_cost = sum(task.get("cost", 0) for task in phase.get("tasks", []))
            total_cost += phase_cost
        
        # Si aucun coût n'est défini, utiliser les données démo
        if total_cost == 0:
            return self.create_budget_sunburst({})  # Utiliser les données démo
        
        values = [total_cost]
        colors = [self.color_palette['primary']]
        
        # Ajouter les phases
        for i, phase in enumerate(phases):
            phase_name = phase.get("name", f"Phase {i+1}")
            labels.append(phase_name)
            parents.append("Projet Total")
            
            # Calculer le coût de la phase à partir de ses tâches
            phase_cost = sum(task.get("cost", 0) for task in phase.get("tasks", []))
            values.append(phase_cost)
            colors.append(self._get_color_shade(self.color_palette['secondary'], i * 0.2))
            
            # Ajouter les tâches
            for j, task in enumerate(phase.get("tasks", [])[:5]):  # Limiter à 5 tâches
                labels.append(task.get("name", f"Tâche {j+1}"))
                parents.append(phase_name)
                values.append(task.get("cost", 0))
                colors.append(self._get_color_shade(self.color_palette['info'], j * 0.15))
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent parent",
            hovertemplate='<b>%{label}</b><br>Budget: €%{value:,.0f}<br>%{percentParent}<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>💰 Décomposition Hiérarchique du Budget</b>",
            height=500,
            font={'family': self.theme['font_family']},
            paper_bgcolor=self.theme['paper']
        )
        
        return fig
    
    def create_network_dependencies(self, plan_data: Dict[str, Any]) -> go.Figure:
        """
        Crée un graphe de réseau des dépendances entre tâches
        
        Parfait pour: Visualiser le chemin critique
        """
        # Générer des positions pour les nœuds
        phases = plan_data.get("wbs", {}).get("phases", []) if plan_data else []
        
        if not phases:
            # Données démo
            phases = [
                {"name": "Phase 1", "tasks": [{"name": "T1.1"}, {"name": "T1.2"}]},
                {"name": "Phase 2", "tasks": [{"name": "T2.1"}, {"name": "T2.2"}, {"name": "T2.3"}]},
                {"name": "Phase 3", "tasks": [{"name": "T3.1"}, {"name": "T3.2"}]}
            ]
        
        # Créer les nœuds et arêtes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        edge_x = []
        edge_y = []
        
        # Positionner les nœuds
        x_offset = 0
        for i, phase in enumerate(phases):
            tasks = phase.get("tasks", [])
            for j, task in enumerate(tasks[:8]):  # Limiter à 8 tâches par phase
                x = x_offset + j * 2
                y = i * 3
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{phase['name']}<br>{task.get('name', f'Tâche {j+1}')}")
                
                # Couleur selon criticité
                is_critical = task.get("is_critical", j % 3 == 0)  # Demo: 1/3 critique
                node_color.append(self.color_palette['danger'] if is_critical else self.color_palette['info'])
                
                # Créer des liens vers la phase suivante
                if i < len(phases) - 1:
                    next_phase_tasks = phases[i + 1].get("tasks", [])
                    for k in range(min(2, len(next_phase_tasks))):  # Connecter à 2 tâches max
                        edge_x.extend([x, x_offset + (j+1)*2 + k*2, None])
                        edge_y.extend([y, (i+1)*3, None])
            
            x_offset += len(tasks) * 2 + 2
        
        # Créer les traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=15,
                color=node_color,
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{text}</b><extra></extra>'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title="<b>🔗 Graphe des Dépendances et Chemin Critique</b>",
            showlegend=False,
            hovermode='closest',
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font={'family': self.theme['font_family']},
            paper_bgcolor=self.theme['paper']
        )
        
        # Ajouter une légende
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text="🔴 Tâche critique | 🔵 Tâche normale",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#888",
            borderwidth=1
        )
        
        return fig
    
    def create_real_time_metrics(self, metrics_history: List[Dict[str, float]]) -> go.Figure:
        """
        Crée des graphiques de métriques en temps réel
        
        Parfait pour: Dashboard de monitoring
        """
        # Générer des données démo si non fournies
        if not metrics_history:
            times = pd.date_range(end=datetime.now(), periods=50, freq='1min')
            metrics_history = [
                {
                    'timestamp': t,
                    'latency': 15 + np.random.normal(0, 3),
                    'accuracy': 95 + np.random.normal(0, 2),
                    'throughput': 100 + np.random.normal(0, 10),
                    'error_rate': max(0, 2 + np.random.normal(0, 1))
                }
                for t in times
            ]
        
        # Créer des subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("⚡ Latence IA (ms)", "🎯 Précision (%)", 
                          "📊 Throughput (req/min)", "⚠️ Taux d'erreur (%)"),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        df = pd.DataFrame(metrics_history)
        
        # Latence
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['latency'],
                mode='lines',
                name='Latence',
                line=dict(color=self.color_palette['primary'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(30, 64, 175, 0.1)"
            ),
            row=1, col=1
        )
        
        # Ligne cible latence
        fig.add_hline(y=20, line_dash="dash", line_color="red", 
                     annotation_text="Cible <20ms", row=1, col=1)
        
        # Précision
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['accuracy'],
                mode='lines+markers',
                name='Précision',
                line=dict(color=self.color_palette['success'], width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        # Ligne cible précision
        fig.add_hline(y=95, line_dash="dash", line_color="green",
                     annotation_text="Cible >95%", row=1, col=2)
        
        # Throughput
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['throughput'],
                name='Throughput',
                marker_color=self.color_palette['info']
            ),
            row=2, col=1
        )
        
        # Taux d'erreur
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['error_rate'],
                mode='lines',
                name='Erreurs',
                line=dict(color=self.color_palette['danger'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(239, 68, 68, 0.1)"
            ),
            row=2, col=2
        )
        
        # Ligne cible erreurs
        fig.add_hline(y=2, line_dash="dash", line_color="orange",
                     annotation_text="Seuil <2%", row=2, col=2)
        
        fig.update_layout(
            title="<b>📈 Métriques de Performance Temps Réel</b>",
            height=600,
            showlegend=False,
            font={'family': self.theme['font_family']},
            paper_bgcolor=self.theme['paper']
        )
        
        # Mise à jour des axes
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        
        return fig
    
    def create_3d_portfolio_scatter(self, projects: List[Dict[str, Any]]) -> go.Figure:
        """
        Crée un scatter 3D pour le positionnement de portfolio
        
        Parfait pour: Comparaison multi-projets
        """
        if not projects:
            # Données démo variées pour démonstration
            projects = [
                {"name": "MedIA (Actuel)", "cost": 48000, "duration": 60, "risk": 4.5, "complexity": 7},
                {"name": "FinTech App", "cost": 50000, "duration": 90, "risk": 6.2, "complexity": 8},
                {"name": "SaaS PME", "cost": 48000, "duration": 60, "risk": 4.8, "complexity": 6},
                {"name": "E-commerce B2B", "cost": 75000, "duration": 120, "risk": 5.5, "complexity": 9},
                {"name": "IoT Platform", "cost": 85000, "duration": 150, "risk": 7.8, "complexity": 10},
                {"name": "Mobile Banking", "cost": 65000, "duration": 95, "risk": 6.8, "complexity": 8},
                {"name": "AI Chatbot", "cost": 35000, "duration": 45, "risk": 3.2, "complexity": 5}
            ]
        
        # Extraire les données
        names = [p.get("name", f"Projet {i+1}") for i, p in enumerate(projects)]
        costs = [p.get("cost", 50000) / 1000 for p in projects]  # En k€
        durations = [p.get("duration", 60) for p in projects]
        risks = [p.get("risk", 5) for p in projects]
        complexities = [p.get("complexity", 5) for p in projects]
        
        # Couleurs selon le risque
        colors = [self._get_risk_color(r) for r in risks]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=costs,
            y=durations,
            z=risks,
            mode='markers+text',
            text=names,
            textposition="top center",
            marker=dict(
                size=[c*3 for c in complexities],  # Taille = complexité
                color=risks,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Score Risque"),
                line=dict(color='white', width=1)
            ),
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Budget: %{x:.0f}k€<br>' +
                'Durée: %{y:.0f} jours<br>' +
                'Risque: %{z:.1f}/10<br>' +
                '<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title="<b>🎯 Positionnement Portfolio 3D</b>",
            scene=dict(
                xaxis_title="Budget (k€)",
                yaxis_title="Durée (jours)",
                zaxis_title="Score de Risque",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            font={'family': self.theme['font_family']},
            paper_bgcolor=self.theme['paper']
        )
        
        return fig
    
    def create_waterfall_roi(self, financial_data: Dict[str, float]) -> go.Figure:
        """
        Crée un waterfall chart pour l'analyse ROI
        
        Parfait pour: Justification business/financière
        """
        if not financial_data:
            # Données démo
            financial_data = {
                'initial_investment': -65000,
                'development_cost': -35000,
                'maintenance_year1': -10000,
                'revenue_year1': 45000,
                'revenue_year2': 85000,
                'cost_savings': 30000,
                'productivity_gains': 25000
            }
        
        # Préparer les données pour le waterfall
        x = ["Investissement<br>Initial", "Coûts<br>Développement", "Maintenance<br>An 1",
             "Revenus<br>An 1", "Revenus<br>An 2", "Économies<br>Coûts", "Gains<br>Productivité", "ROI Net"]
        
        y = [
            financial_data.get('initial_investment', -65000),
            financial_data.get('development_cost', -35000),
            financial_data.get('maintenance_year1', -10000),
            financial_data.get('revenue_year1', 45000),
            financial_data.get('revenue_year2', 85000),
            financial_data.get('cost_savings', 30000),
            financial_data.get('productivity_gains', 25000),
            None  # Total calculé automatiquement
        ]
        
        measure = ["relative", "relative", "relative", "relative", 
                  "relative", "relative", "relative", "total"]
        
        # Les couleurs sont gérées par les paramètres increasing/decreasing/totals
        
        fig = go.Figure(go.Waterfall(
            x=x,
            y=y,
            measure=measure,
            text=[f"€{abs(v):,.0f}" if v else "" for v in y],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": self.color_palette['success']}},
            decreasing={"marker": {"color": self.color_palette['danger']}},
            totals={"marker": {"color": self.color_palette['primary']}},
            hovertemplate='<b>%{x}</b><br>Montant: €%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="<b>💼 Analyse ROI - Cascade Financière</b>",
            showlegend=False,
            height=450,
            font={'family': self.theme['font_family']},
            paper_bgcolor=self.theme['paper']
        )
        
        return fig
    
    def create_radar_comparison(self, projects_comparison: List[Dict[str, Any]]) -> go.Figure:
        """
        Crée un radar chart pour comparer plusieurs projets
        
        Parfait pour: Analyse comparative multi-critères
        """
        if not projects_comparison:
            # Données démo
            projects_comparison = [
                {
                    "name": "Projet Actuel",
                    "metrics": {
                        "Performance": 92,
                        "Budget": 85,
                        "Délai": 88,
                        "Qualité": 95,
                        "Innovation": 90,
                        "Risque": 75  # Inversé: 75 = faible risque
                    }
                },
                {
                    "name": "Moyenne Secteur",
                    "metrics": {
                        "Performance": 78,
                        "Budget": 72,
                        "Délai": 70,
                        "Qualité": 80,
                        "Innovation": 65,
                        "Risque": 60
                    }
                },
                {
                    "name": "Best Practice",
                    "metrics": {
                        "Performance": 95,
                        "Budget": 90,
                        "Délai": 92,
                        "Qualité": 98,
                        "Innovation": 95,
                        "Risque": 85
                    }
                }
            ]
        
        fig = go.Figure()
        
        categories = list(projects_comparison[0]["metrics"].keys())
        
        colors = [self.color_palette['primary'], self.color_palette['secondary'], self.color_palette['success']]
        
        for i, project in enumerate(projects_comparison):
            values = list(project["metrics"].values())
            values.append(values[0])  # Fermer le radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(30, 100, 200, 0.1)' if i == 0 else 'rgba(124, 58, 237, 0.1)' if i == 1 else 'rgba(16, 185, 129, 0.1)',
                name=project["name"],
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate='<b>%{theta}</b><br>Score: %{r}<br><extra></extra>'
            ))
        
        fig.update_layout(
            title="<b>📊 Analyse Comparative Multi-Critères</b>",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=True,
            height=500,
            font={'family': self.theme['font_family']},
            paper_bgcolor=self.theme['paper']
        )
        
        return fig
    
    def create_timeline_roadmap(self, milestones: List[Dict[str, Any]]) -> go.Figure:
        """
        Crée une timeline/roadmap interactive
        
        Parfait pour: Vue d'ensemble du planning
        """
        # Création d'un graphique vide si pas de données pour éviter les erreurs Plotly
        if not milestones:
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée de timeline disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="<b>📅 Roadmap du Projet</b>",
                height=250,
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return fig
        
        # Préparer les données
        dates = [m["date"] for m in milestones]
        names = [m["name"] for m in milestones]
        
        # Couleurs selon le statut
        colors = []
        sizes = []
        for m in milestones:
            status = m.get("status", "planned")
            if status == "completed":
                colors.append(self.color_palette['success'])
                sizes.append(20)
            elif status == "in_progress":
                colors.append(self.color_palette['warning'])
                sizes.append(25)
            else:
                colors.append(self.color_palette['info'])
                sizes.append(15)
        
        fig = go.Figure()
        
        # Ajouter les traces seulement s'il y a des données
        if dates and len(dates) > 0:
            # Ligne de base
            fig.add_trace(go.Scatter(
                x=dates,
                y=[1] * len(dates),
                mode='lines',
                line=dict(color='rgba(0,0,0,0.2)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Milestones
            fig.add_trace(go.Scatter(
                x=dates,
                y=[1] * len(dates),
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=names,
                textposition="top center",
                textfont=dict(size=11),
                hovertemplate='<b>%{text}</b><br>Timing: %{x}<extra></extra>'
            ))
        
        # Ligne verticale pour aujourd'hui désactivée temporairement pour éviter les conflits
        # Le problème vient de l'interaction entre Streamlit et Plotly add_vline
        # TODO: Réactiver quand le problème sera résolu
        pass
        
        fig.update_layout(
            title="<b>📅 Roadmap du Projet</b>",
            showlegend=False,
            height=250,
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[0.5, 1.5]
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                type='category'
            ),
            font={'family': self.theme['font_family']},
            paper_bgcolor=self.theme['paper'],
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    # Méthodes utilitaires
    def _get_performance_color(self, value: float, max_value: float) -> str:
        """Retourne une couleur selon la performance"""
        ratio = value / max_value
        if ratio > 0.8:
            return self.color_palette['success']
        elif ratio > 0.5:
            return self.color_palette['warning']
        else:
            return self.color_palette['danger']
    
    def _get_risk_color(self, risk_score: float) -> str:
        """Retourne une couleur selon le score de risque"""
        if risk_score < 3:
            return self.color_palette['success']
        elif risk_score < 6:
            return self.color_palette['warning']
        else:
            return self.color_palette['danger']
    
    def _get_color_shade(self, base_color: str, factor: float) -> str:
        """Génère une nuance de couleur"""
        # Convertir hex en RGB
        hex_color = base_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Ajuster la luminosité
        factor = max(0, min(1, 1 - factor))
        new_rgb = tuple(int(c * factor) for c in rgb)
        
        # Retourner en hex
        return f"#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}"


if __name__ == "__main__":
    # Test des visualisations
    viz = AdvancedVisualizations()
    
    # Test gauge
    fig = viz.create_performance_gauge(92.5, "Performance Globale")
    fig.show()
    
    # Test heatmap
    fig = viz.create_ai_systems_heatmap({})
    fig.show()
    
    # Test sankey
    fig = viz.create_workflow_sankey({})
    fig.show()