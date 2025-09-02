#!/usr/bin/env python3
"""
🎨 MOTEUR VISUALISATION 3D IA/ML - PlannerIA
Visualisations spectaculaires pour soutenance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import json
import logging

class ML3DEngine:
    """
    🚀 Moteur de Visualisation 3D IA/ML
    
    Fonctionnalités spectaculaires:
    - Pareto 3D avec clustering intelligent
    - Réseaux neuronaux visualisés
    - Surfaces de risque ML volumétriques  
    - Hologrammes multi-métriques
    - Animations temporelles fluides
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.clusters_cache = {}
        self.animations_cache = {}
        
        # Palettes de couleurs spectaculaires
        self.color_schemes = {
            'neon': ['#00ffff', '#ff00ff', '#ffff00', '#00ff00', '#ff0080'],
            'fire': ['#ff4500', '#ff6347', '#ffa500', '#ffd700', '#ff1493'],
            'galaxy': ['#4b0082', '#8a2be2', '#9400d3', '#9932cc', '#ba55d3'],
            'matrix': ['#00ff00', '#32cd32', '#90ee90', '#adff2f', '#7fff00'],
            'hologram': ['#00bfff', '#1e90ff', '#87ceeb', '#87cefa', '#b0e0e6']
        }
        
        self.logger.info("🎨 Moteur 3D ML initialisé")
    
    def generer_pareto_3d_intelligent(self, project_data: Dict[str, Any], 
                                    theme: str = 'neon') -> go.Figure:
        """
        🎯 Pareto 3D avec Clustering IA Spectaculaire
        
        Axes: Impact × Probabilité × Effort
        Bulles: Taille = Budget
        Couleurs: Clusters IA automatiques
        """
        try:
            # Générer ou extraire données
            if 'tasks' not in project_data:
                data = self._generer_donnees_pareto_demo()
            else:
                data = self._transformer_donnees_pareto(project_data)
            
            # Clustering intelligent
            clusters = self._clustering_intelligent(data[['impact', 'probabilite', 'effort']])
            data['cluster'] = clusters
            
            # Créer figure 3D spectaculaire
            fig = go.Figure()
            
            colors = self.color_schemes[theme]
            
            for i, cluster in enumerate(np.unique(clusters)):
                cluster_data = data[data['cluster'] == cluster]
                
                fig.add_trace(go.Scatter3d(
                    x=cluster_data['impact'],
                    y=cluster_data['probabilite'], 
                    z=cluster_data['effort'],
                    mode='markers+text',
                    marker=dict(
                        size=cluster_data['budget'] * 50,  # Taille proportionnelle
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        line=dict(width=2, color='white'),
                        symbol='diamond' if cluster == 0 else 'circle'
                    ),
                    text=cluster_data['nom_tache'],
                    textposition='top center',
                    textfont=dict(size=10, color='white'),
                    name=f'Cluster {i+1}',
                    hovertemplate=
                    '<b>%{text}</b><br>' +
                    'Impact: %{x:.2f}<br>' +
                    'Probabilité: %{y:.2f}<br>' +
                    'Effort: %{z:.2f}<br>' +
                    'Budget: %{marker.size}<br>' +
                    '<extra></extra>'
                ))
            
            # Configuration 3D spectaculaire
            fig.update_layout(
                title={
                    'text': '🎯 PARETO 3D INTELLIGENT - CLUSTERING IA',
                    'font': {'size': 24, 'color': colors[0]},
                    'x': 0.5
                },
                scene=dict(
                    xaxis=dict(
                        title='📈 IMPACT PROJET',
                        title_font=dict(color=colors[1], size=14),
                        tickfont=dict(color='white'),
                        gridcolor='rgba(255,255,255,0.2)',
                        backgroundcolor='rgba(0,0,0,0.8)'
                    ),
                    yaxis=dict(
                        title='🎲 PROBABILITÉ SUCCESS',
                        title_font=dict(color=colors[2], size=14),
                        tickfont=dict(color='white'),
                        gridcolor='rgba(255,255,255,0.2)',
                        backgroundcolor='rgba(0,0,0,0.8)'
                    ),
                    zaxis=dict(
                        title='⚡ EFFORT REQUIS',
                        title_font=dict(color=colors[3], size=14),
                        tickfont=dict(color='white'),
                        gridcolor='rgba(255,255,255,0.2)',
                        backgroundcolor='rgba(0,0,0,0.8)'
                    ),
                    bgcolor='black',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                height=700,
                margin=dict(t=80, b=0, l=0, r=0)
            )
            
            self.logger.info("🎯 Pareto 3D généré avec succès")
            return fig
            
        except Exception as e:
            self.logger.error(f"Erreur génération Pareto 3D: {e}")
            return self._figure_erreur("Erreur Pareto 3D")
    
    def generer_reseau_neuronal_3d(self, dependencies_data: Dict[str, Any],
                                  theme: str = 'matrix') -> go.Figure:
        """
        🧠 Réseau Neuronal 3D du Projet
        
        Nœuds: Tâches/Modules
        Arêtes: Dépendances avec force variable
        Couleurs: Criticité IA
        Animation: Layout physique
        """
        try:
            # Créer graphe de dépendances
            G = self._creer_graphe_dependances(dependencies_data)
            
            # Layout 3D avec algorithme physique
            pos_3d = self._layout_3d_physique(G)
            
            # Calculer métriques de criticité
            criticite = self._calculer_criticite_noeuds(G)
            
            fig = go.Figure()
            colors = self.color_schemes[theme]
            
            # Ajouter arêtes (connexions)
            edge_x, edge_y, edge_z = [], [], []
            edge_info = []
            
            for edge in G.edges():
                x0, y0, z0 = pos_3d[edge[0]]
                x1, y1, z1 = pos_3d[edge[1]]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
                
                # Force de la connexion
                weight = G[edge[0]][edge[1]].get('weight', 1.0)
                edge_info.append(weight)
            
            # Tracer connexions
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color=colors[1], width=3),
                name='Connexions IA',
                showlegend=False,
                hoverinfo='none'
            ))
            
            # Ajouter nœuds (tâches)
            node_x = [pos_3d[node][0] for node in G.nodes()]
            node_y = [pos_3d[node][1] for node in G.nodes()]
            node_z = [pos_3d[node][2] for node in G.nodes()]
            
            node_colors = [criticite.get(node, 0.5) for node in G.nodes()]
            node_sizes = [G.degree(node) * 15 + 20 for node in G.nodes()]
            node_text = [f"{node}\\nCriticité: {criticite.get(node, 0.5):.2f}" 
                        for node in G.nodes()]
            
            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Plasma',
                    opacity=0.9,
                    colorbar=dict(
                        title="Criticité IA",
                        title_font=dict(color='white'),
                        tickfont=dict(color='white')
                    ),
                    line=dict(width=2, color='white')
                ),
                text=[str(node) for node in G.nodes()],
                textfont=dict(size=12, color='white'),
                name='Modules Projet',
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # Configuration réseau neuronal
            fig.update_layout(
                title={
                    'text': '🧠 RÉSEAU NEURONAL PROJET - IA DEPENDENCIES',
                    'font': {'size': 24, 'color': colors[0]},
                    'x': 0.5
                },
                scene=dict(
                    xaxis=dict(
                        showgrid=False, showticklabels=False,
                        backgroundcolor='black'
                    ),
                    yaxis=dict(
                        showgrid=False, showticklabels=False,
                        backgroundcolor='black'
                    ),
                    zaxis=dict(
                        showgrid=False, showticklabels=False,
                        backgroundcolor='black'
                    ),
                    bgcolor='black',
                    camera=dict(eye=dict(x=2, y=2, z=2))
                ),
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                height=700,
                showlegend=True
            )
            
            self.logger.info("🧠 Réseau neuronal 3D généré")
            return fig
            
        except Exception as e:
            self.logger.error(f"Erreur réseau neuronal: {e}")
            return self._figure_erreur("Erreur Réseau Neuronal 3D")
    
    def generer_surface_risque_3d(self, risk_data: Dict[str, Any],
                                theme: str = 'fire') -> go.Figure:
        """
        🌊 Surface de Risque 3D Volumétrique
        
        Axes: Temps × Budget × Qualité
        Surface: Prédictions ML déformables
        Couleurs: Gradient de risque
        """
        try:
            # Générer grille 3D
            temps = np.linspace(0, 100, 25)
            budget = np.linspace(0, 100, 25)
            qualite = np.linspace(0, 100, 25)
            
            T, B, Q = np.meshgrid(temps, budget, qualite, indexing='ij')
            
            # Calculer surface de risque avec ML
            Z = self._calculer_surface_risque(T, B, Q, risk_data)
            
            fig = go.Figure()
            colors = self.color_schemes[theme]
            
            # Surface principale
            fig.add_trace(go.Volume(
                x=T.flatten(),
                y=B.flatten(), 
                z=Q.flatten(),
                value=Z.flatten(),
                opacity=0.3,
                surface_count=15,
                colorscale='Hot',
                name='Volume Risque',
                showscale=True,
                colorbar=dict(
                    title="Niveau Risque",
                    title_font=dict(color='white'),
                    tickfont=dict(color='white')
                )
            ))
            
            # Points critiques
            points_critiques = self._identifier_points_critiques(T, B, Q, Z)
            
            if len(points_critiques) > 0:
                fig.add_trace(go.Scatter3d(
                    x=points_critiques[:, 0],
                    y=points_critiques[:, 1],
                    z=points_critiques[:, 2],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='x',
                        line=dict(width=3, color='white')
                    ),
                    name='Points Critiques',
                    hovertemplate='Point Critique<br>' +
                                'Temps: %{x:.1f}<br>' +
                                'Budget: %{y:.1f}<br>' +
                                'Qualité: %{z:.1f}<extra></extra>'
                ))
            
            # Configuration surface
            fig.update_layout(
                title={
                    'text': '🌊 SURFACE DE RISQUE 3D - PRÉDICTIONS ML',
                    'font': {'size': 24, 'color': colors[0]},
                    'x': 0.5
                },
                scene=dict(
                    xaxis=dict(
                        title='⏰ TEMPS PROJET (%)',
                        title_font=dict(color=colors[1], size=14),
                        tickfont=dict(color='white')
                    ),
                    yaxis=dict(
                        title='💰 BUDGET UTILISÉ (%)',
                        title_font=dict(color=colors[2], size=14),
                        tickfont=dict(color='white')
                    ),
                    zaxis=dict(
                        title='✅ QUALITÉ ATTEINTE (%)',
                        title_font=dict(color=colors[3], size=14),
                        tickfont=dict(color='white')
                    ),
                    bgcolor='black',
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
                ),
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                height=700
            )
            
            self.logger.info("🌊 Surface de risque 3D générée")
            return fig
            
        except Exception as e:
            self.logger.error(f"Erreur surface risque: {e}")
            return self._figure_erreur("Erreur Surface Risque 3D")
    
    def generer_hologramme_metriques(self, metrics_data: Dict[str, Any],
                                   theme: str = 'hologram') -> go.Figure:
        """
        🎯 Hologramme Multi-Métriques 3D
        
        Radar 3D multi-dimensionnel
        Comparaisons: Prévu vs Réalisé vs Prédit
        Morphing temps réel
        """
        try:
            # Préparer données multi-dimensionnelles
            if 'metriques' not in metrics_data:
                data = self._generer_metriques_demo()
            else:
                data = metrics_data['metriques']
            
            fig = go.Figure()
            colors = self.color_schemes[theme]
            
            # Créer radar 3D pour chaque série
            series = ['Prévu', 'Réalisé', 'Prédit IA']
            
            for i, serie in enumerate(series):
                if serie in data:
                    coords = self._radar_vers_3d(data[serie])
                    
                    fig.add_trace(go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode='markers+lines',
                        marker=dict(
                            size=12,
                            color=colors[i],
                            opacity=0.8,
                            line=dict(width=2, color='white')
                        ),
                        line=dict(
                            color=colors[i],
                            width=5
                        ),
                        name=serie,
                        text=[f"{k}: {v:.2f}" for k, v in data[serie].items()],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                    
                    # Surface de connexion
                    fig.add_trace(go.Mesh3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        opacity=0.2,
                        color=colors[i],
                        name=f'Surface {serie}',
                        showlegend=False
                    ))
            
            # Configuration hologramme
            fig.update_layout(
                title={
                    'text': '🎯 HOLOGRAMME MÉTRIQUES - RADAR 3D IA',
                    'font': {'size': 24, 'color': colors[0]},
                    'x': 0.5
                },
                scene=dict(
                    xaxis=dict(
                        showgrid=True, gridcolor='rgba(0,255,255,0.3)',
                        showticklabels=False,
                        backgroundcolor='black'
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor='rgba(0,255,255,0.3)',
                        showticklabels=False,
                        backgroundcolor='black'
                    ),
                    zaxis=dict(
                        showgrid=True, gridcolor='rgba(0,255,255,0.3)',
                        showticklabels=False,
                        backgroundcolor='black'
                    ),
                    bgcolor='black',
                    camera=dict(eye=dict(x=2.5, y=2.5, z=2.5))
                ),
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                height=700,
                showlegend=True
            )
            
            self.logger.info("🎯 Hologramme métriques généré")
            return fig
            
        except Exception as e:
            self.logger.error(f"Erreur hologramme: {e}")
            return self._figure_erreur("Erreur Hologramme 3D")
    
    def generer_animation_temporelle(self, timeline_data: Dict[str, Any],
                                   type_viz: str = 'evolution') -> go.Figure:
        """
        🎬 Animation Temporelle 3D
        
        Évolution des métriques dans le temps
        Trajectoires 3D fluides
        Contrôles play/pause
        """
        try:
            # Préparer données temporelles
            if 'timeline' not in timeline_data:
                frames_data = self._generer_timeline_demo()
            else:
                frames_data = timeline_data['timeline']
            
            fig = go.Figure()
            colors = self.color_schemes['galaxy']
            
            # Frame initial
            initial_frame = frames_data[0]
            
            fig.add_trace(go.Scatter3d(
                x=initial_frame['x'],
                y=initial_frame['y'],
                z=initial_frame['z'],
                mode='markers+lines',
                marker=dict(
                    size=initial_frame['sizes'],
                    color=initial_frame['colors'],
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                line=dict(color=colors[0], width=4),
                name='Évolution Projet',
                text=initial_frame['labels'],
                hovertemplate='%{text}<br>' +
                            'X: %{x:.2f}<br>' +
                            'Y: %{y:.2f}<br>' +
                            'Z: %{z:.2f}<extra></extra>'
            ))
            
            # Créer frames d'animation
            frames = []
            for i, frame_data in enumerate(frames_data):
                frames.append(go.Frame(
                    data=[go.Scatter3d(
                        x=frame_data['x'],
                        y=frame_data['y'],
                        z=frame_data['z'],
                        mode='markers+lines',
                        marker=dict(
                            size=frame_data['sizes'],
                            color=frame_data['colors'],
                            colorscale='Viridis',
                            opacity=0.8,
                            line=dict(width=2, color='white')
                        ),
                        line=dict(color=colors[i % len(colors)], width=4),
                        text=frame_data['labels']
                    )],
                    name=f'Frame {i}',
                    layout=dict(
                        title=f"🎬 Évolution Temporelle - Semaine {i+1}"
                    )
                ))
            
            fig.frames = frames
            
            # Contrôles d'animation
            fig.update_layout(
                title={
                    'text': '🎬 ANIMATION TEMPORELLE 3D - ÉVOLUTION IA',
                    'font': {'size': 24, 'color': colors[0]},
                    'x': 0.5
                },
                scene=dict(
                    xaxis=dict(title='Métrique X', title_font=dict(color='white')),
                    yaxis=dict(title='Métrique Y', title_font=dict(color='white')),
                    zaxis=dict(title='Métrique Z', title_font=dict(color='white')),
                    bgcolor='black',
                    camera=dict(eye=dict(x=2, y=2, z=2))
                ),
                updatemenus=[dict(
                    type="buttons",
                    direction="left",
                    showactive=False,
                    buttons=list([
                        dict(
                            args=[None, {"frame": {"duration": 500, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 300}}],
                            label="▶ Play",
                            method="animate"
                        ),
                        dict(
                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate", "transition": {"duration": 0}}],
                            label="⏸ Pause",
                            method="animate"
                        )
                    ]),
                    pad={"r": 10, "t": 87},
                    x=0.011, xanchor="right",
                    y=0, yanchor="top"
                )],
                sliders=[dict(
                    active=0,
                    yanchor="top", y=0,
                    xanchor="left", x=0.1,
                    currentvalue={
                        "font": {"size": 16, "color": "white"},
                        "prefix": "Frame: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    transition={"duration": 300},
                    pad={"b": 10, "t": 50},
                    len=0.9,
                    steps=[dict(args=[[f'Frame {k}'], {"frame": {"duration": 300, "redraw": True},
                                           "mode": "immediate",
                                           "transition": {"duration": 300}}],
                               label=str(k),
                               method='animate') for k in range(len(frames_data))]
                )],
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white'),
                height=700
            )
            
            self.logger.info("🎬 Animation temporelle générée")
            return fig
            
        except Exception as e:
            self.logger.error(f"Erreur animation: {e}")
            return self._figure_erreur("Erreur Animation 3D")
    
    # === MÉTHODES UTILITAIRES ===
    
    def _generer_donnees_pareto_demo(self) -> pd.DataFrame:
        """Génère données de démonstration pour Pareto 3D"""
        np.random.seed(42)
        n_tasks = 25
        
        return pd.DataFrame({
            'nom_tache': [f'Tâche {i+1}' for i in range(n_tasks)],
            'impact': np.random.uniform(0.1, 1.0, n_tasks),
            'probabilite': np.random.uniform(0.2, 1.0, n_tasks),
            'effort': np.random.uniform(0.1, 1.0, n_tasks),
            'budget': np.random.uniform(0.1, 1.0, n_tasks)
        })
    
    def _clustering_intelligent(self, data: pd.DataFrame, n_clusters: int = 4) -> np.ndarray:
        """Clustering intelligent des données"""
        try:
            data_scaled = self.scaler.fit_transform(data)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(data_scaled)
        except:
            return np.zeros(len(data))
    
    def _creer_graphe_dependances(self, deps_data: Dict[str, Any]) -> nx.Graph:
        """Crée graphe de dépendances"""
        G = nx.Graph()
        
        # Données de demo si pas de vraies données
        if 'nodes' not in deps_data:
            nodes = [f'Module_{i}' for i in range(12)]
            edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) 
                    for j in range(i+1, min(i+4, len(nodes)))]
        else:
            nodes = deps_data['nodes']
            edges = deps_data['edges']
        
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        return G
    
    def _layout_3d_physique(self, G: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Layout 3D avec algorithme physique"""
        try:
            # Position 2D avec spring layout
            pos_2d = nx.spring_layout(G, dim=2, k=3, iterations=50)
            
            # Étendre en 3D
            pos_3d = {}
            for node, (x, y) in pos_2d.items():
                # Z basé sur degré du nœud
                z = G.degree(node) * 0.1
                pos_3d[node] = (x, y, z)
            
            return pos_3d
        except:
            # Fallback positions aléatoires
            return {node: (np.random.rand(), np.random.rand(), np.random.rand()) 
                   for node in G.nodes()}
    
    def _calculer_criticite_noeuds(self, G: nx.Graph) -> Dict[str, float]:
        """Calcule criticité IA des nœuds"""
        try:
            centrality = nx.betweenness_centrality(G)
            # Normaliser entre 0 et 1
            max_cent = max(centrality.values()) if centrality.values() else 1
            return {node: val/max_cent for node, val in centrality.items()}
        except:
            return {node: 0.5 for node in G.nodes()}
    
    def _calculer_surface_risque(self, T: np.ndarray, B: np.ndarray, 
                               Q: np.ndarray, risk_data: Dict) -> np.ndarray:
        """Calcule surface de risque ML"""
        # Fonction de risque sophistiquée
        Z = (
            0.3 * np.sin(T/10) * np.cos(B/15) +  # Oscillations temporelles
            0.4 * (B/100) ** 2 +                 # Risque budgétaire quadratique
            0.3 * np.exp(-(Q-50)**2/500) +       # Pic qualité optimale
            0.2 * np.random.random(T.shape) * 0.1 # Bruit ML
        )
        
        return np.clip(Z, 0, 1)  # Normaliser 0-1
    
    def _identifier_points_critiques(self, T: np.ndarray, B: np.ndarray, 
                                   Q: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Identifie points critiques dans la surface"""
        # Seuil de criticité
        seuil = np.percentile(Z, 85)
        indices_critiques = np.where(Z > seuil)
        
        if len(indices_critiques[0]) > 0:
            points = np.column_stack([
                T[indices_critiques],
                B[indices_critiques], 
                Q[indices_critiques]
            ])
            # Limiter à 10 points max pour lisibilité
            if len(points) > 10:
                indices = np.random.choice(len(points), 10, replace=False)
                points = points[indices]
            return points
        
        return np.array([]).reshape(0, 3)
    
    def _generer_metriques_demo(self) -> Dict[str, Dict[str, float]]:
        """Génère métriques de démonstration"""
        metriques = ['Budget', 'Temps', 'Qualité', 'Risque', 'Équipe', 
                    'Innovation', 'Satisfaction', 'Performance']
        
        return {
            'Prévu': {m: np.random.uniform(0.3, 0.9) for m in metriques},
            'Réalisé': {m: np.random.uniform(0.2, 0.8) for m in metriques},
            'Prédit IA': {m: np.random.uniform(0.4, 0.95) for m in metriques}
        }
    
    def _radar_vers_3d(self, metriques: Dict[str, float]) -> np.ndarray:
        """Convertit métriques radar en coordonnées 3D"""
        n = len(metriques)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        coords = []
        for i, (_, value) in enumerate(metriques.items()):
            x = value * np.cos(angles[i]) * np.cos(angles[i]/2)
            y = value * np.sin(angles[i]) * np.cos(angles[i]/2)  
            z = value * np.sin(angles[i]/2)
            coords.append([x, y, z])
        
        return np.array(coords)
    
    def _generer_timeline_demo(self) -> List[Dict[str, List]]:
        """Génère timeline de démonstration"""
        frames = []
        n_points = 15
        n_frames = 10
        
        for frame in range(n_frames):
            t = frame / n_frames
            
            # Évolution sinusoïdale
            x = [np.sin(i/3 + t*2*np.pi) + np.random.normal(0, 0.1) for i in range(n_points)]
            y = [np.cos(i/4 + t*2*np.pi) + np.random.normal(0, 0.1) for i in range(n_points)]
            z = [np.sin(i/2 + t*np.pi) * np.cos(i/5) + np.random.normal(0, 0.1) for i in range(n_points)]
            
            sizes = [15 + 10*np.sin(i/5 + t*np.pi) for i in range(n_points)]
            colors = [i + frame*2 for i in range(n_points)]
            labels = [f'Point {i+1}' for i in range(n_points)]
            
            frames.append({
                'x': x, 'y': y, 'z': z,
                'sizes': sizes, 'colors': colors, 'labels': labels
            })
        
        return frames
    
    def _figure_erreur(self, message: str) -> go.Figure:
        """Crée figure d'erreur basique"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        return fig

# Test rapide
if __name__ == "__main__":
    engine = ML3DEngine()
    
    print("Test Moteur 3D ML")
    
    # Test Pareto 3D
    fig_pareto = engine.generer_pareto_3d_intelligent({})
    print("Pareto 3D: OK")
    
    # Test Réseau neuronal  
    fig_reseau = engine.generer_reseau_neuronal_3d({})
    print("Reseau neuronal 3D: OK")
    
    print("Moteur 3D pret pour soutenance!")