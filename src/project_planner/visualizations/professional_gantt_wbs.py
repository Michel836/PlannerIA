"""
📊 Module Professionnel de Visualisation Gantt & WBS
====================================================

Visualisations de niveau entreprise pour la planification de projet
avec support complet du chemin critique, dépendances et jalons.

Auteur: PlannerIA Team
Date: 2025-08-31
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx


class ProfessionalProjectVisualizations:
    """
    🎯 Générateur de Gantt et WBS Professionnels
    
    Caractéristiques:
    - Gantt avec dépendances et chemin critique
    - WBS interactif avec vue hiérarchique
    - Calcul automatique des dates
    - Support des jalons et livrables
    """
    
    def __init__(self):
        """Initialise les thèmes et configurations"""
        self.colors = {
            'critical': '#dc2626',      # Rouge pour chemin critique
            'normal': '#3b82f6',        # Bleu pour tâches normales
            'milestone': '#f59e0b',     # Orange pour jalons
            'completed': '#10b981',     # Vert pour complété
            'in_progress': '#8b5cf6',   # Violet pour en cours
            'planned': '#6b7280',       # Gris pour planifié
            'dependency': '#94a3b8',    # Gris clair pour liens
            'phase_bg': '#f1f5f9',      # Fond des phases
            'weekend': '#fef3c7'        # Jaune clair pour weekends
        }
    
    def create_professional_gantt(self, plan_data: Dict[str, Any], 
                                 start_date: Optional[datetime] = None) -> go.Figure:
        """
        Crée un diagramme de Gantt professionnel avec toutes les fonctionnalités
        
        Caractéristiques:
        - Calcul automatique des dates basé sur les durées réelles
        - Visualisation du chemin critique en rouge
        - Dépendances entre tâches (flèches)
        - Jalons et livrables marqués
        - Weekends grisés
        - Progression des tâches
        """
        if not start_date:
            start_date = datetime.now()
        
        # Extraire et structurer les données
        tasks_data = self._extract_tasks_with_dependencies(plan_data, start_date)
        
        if not tasks_data:
            # Données démo réalistes si plan vide
            tasks_data = self._generate_demo_gantt_data(start_date)
        
        # Calculer le chemin critique
        critical_path = self._calculate_critical_path(tasks_data)
        
        # Créer le DataFrame pour Plotly
        df = pd.DataFrame(tasks_data)
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter les barres de phases (fond)
        phases = df['phase'].unique()
        phase_colors = px.colors.qualitative.Pastel
        
        for i, phase in enumerate(phases):
            phase_tasks = df[df['phase'] == phase]
            phase_start = phase_tasks['start_date'].min()
            phase_end = phase_tasks['end_date'].max()
            
            # Barre de fond pour la phase
            fig.add_trace(go.Scatter(
                x=[phase_start, phase_end, phase_end, phase_start, phase_start],
                y=[i-0.4, i-0.4, i+len(phase_tasks)+0.4, i+len(phase_tasks)+0.4, i-0.4],
                fill='toself',
                fillcolor=f'rgba(200,200,200,0.1)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Ajouter les weekends en arrière-plan
        current = start_date
        end_date = df['end_date'].max()
        while current <= end_date:
            if current.weekday() >= 5:  # Samedi ou Dimanche
                fig.add_vrect(
                    x0=current,
                    x1=current + timedelta(days=1),
                    fillcolor=self.colors['weekend'],
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
            current += timedelta(days=1)
        
        # Ajouter les tâches
        for idx, row in df.iterrows():
            # Déterminer la couleur selon criticité et statut
            if row['task_id'] in critical_path:
                color = self.colors['critical']
                line_width = 3
            else:
                color = self.colors[row['status']]
                line_width = 1
            
            # Barre principale de la tâche
            fig.add_trace(go.Scatter(
                x=[row['start_date'], row['end_date']],
                y=[idx, idx],
                mode='lines',
                line=dict(color=color, width=20),
                name=row['task_name'],
                hovertemplate=(
                    f"<b>{row['task_name']}</b><br>"
                    f"Phase: {row['phase']}<br>"
                    f"Début: {row['start_date'].strftime('%d/%m/%Y')}<br>"
                    f"Fin: {row['end_date'].strftime('%d/%m/%Y')}<br>"
                    f"Durée: {row['duration']} jours<br>"
                    f"Progression: {row['progress']}%<br>"
                    f"Ressources: {row['resources']}<br>"
                    f"{'🔴 CHEMIN CRITIQUE' if row['task_id'] in critical_path else ''}"
                    "<extra></extra>"
                ),
                showlegend=False
            ))
            
            # Barre de progression
            if row['progress'] > 0:
                progress_end = row['start_date'] + timedelta(
                    days=int(row['duration'] * row['progress'] / 100)
                )
                fig.add_trace(go.Scatter(
                    x=[row['start_date'], progress_end],
                    y=[idx, idx],
                    mode='lines',
                    line=dict(color=self.colors['completed'], width=10),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Ajouter le texte de la tâche
            fig.add_annotation(
                x=row['start_date'],
                y=idx,
                text=f" {row['task_name'][:30]}",
                showarrow=False,
                font=dict(size=10, color='black'),
                xanchor='left',
                bgcolor='rgba(255,255,255,0.8)'
            )
            
            # Marquer les jalons
            if row['is_milestone']:
                fig.add_trace(go.Scatter(
                    x=[row['end_date']],
                    y=[idx],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color=self.colors['milestone'],
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>⭐ JALON: {row['task_name']}</b><extra></extra>"
                ))
        
        # Ajouter les dépendances (flèches)
        for idx, row in df.iterrows():
            if row['dependencies']:
                for dep_id in row['dependencies']:
                    dep_task = df[df['task_id'] == dep_id]
                    if not dep_task.empty:
                        dep_idx = dep_task.index[0]
                        # Ligne de dépendance
                        fig.add_trace(go.Scatter(
                            x=[dep_task.iloc[0]['end_date'], row['start_date']],
                            y=[dep_idx, idx],
                            mode='lines',
                            line=dict(
                                color=self.colors['dependency'],
                                width=1,
                                dash='dot'
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Flèche au bout
                        fig.add_annotation(
                            x=row['start_date'],
                            y=idx,
                            ax=dep_task.iloc[0]['end_date'],
                            ay=dep_idx,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor=self.colors['dependency']
                        )
        
        # Ligne verticale pour aujourd'hui - désactivée temporairement (conflit de types)
        # TODO: Corriger l'ajout de ligne verticale sur axe temporel
        # fig.add_vline(
        #     x=datetime.now(),
        #     line_dash="dash",
        #     line_color="red",
        #     annotation_text="Aujourd'hui",
        #     annotation_position="top"
        # )
        
        # Configuration du layout
        fig.update_layout(
            title={
                'text': "📅 <b>Diagramme de Gantt Professionnel avec Chemin Critique</b>",
                'font': {'size': 20}
            },
            xaxis=dict(
                title="Timeline",
                type='date',
                tickformat='%d %b',
                tickmode='linear',
                dtick=86400000.0 * 7,  # Tick hebdomadaire
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                rangeslider=dict(visible=True, thickness=0.05)
            ),
            yaxis=dict(
                title="Tâches",
                showticklabels=False,
                showgrid=False,
                range=[-1, len(df)]
            ),
            height=max(400, len(df) * 30 + 150),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Ajouter la légende personnalisée
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['critical']),
            name='🔴 Chemin Critique'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['completed']),
            name='✅ Complété'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['in_progress']),
            name='🔄 En Cours'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=self.colors['planned']),
            name='📋 Planifié'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='diamond', size=10, color=self.colors['milestone']),
            name='⭐ Jalon'
        ))
        
        return fig
    
    def create_interactive_wbs(self, plan_data: Dict[str, Any]) -> go.Figure:
        """
        Crée une visualisation WBS interactive et hiérarchique
        
        Caractéristiques:
        - Vue en arbre hiérarchique
        - Codes WBS automatiques (1.1, 1.2, 2.1, etc.)
        - Indicateurs visuels (coût, durée, criticité)
        - Expandable/Collapsible
        - Export possible
        """
        wbs_data = self._extract_wbs_structure(plan_data)
        
        if not wbs_data:
            wbs_data = self._generate_demo_wbs_data()
        
        # Créer un treemap pour la vue hiérarchique
        fig = go.Figure(go.Treemap(
            labels=[item['label'] for item in wbs_data],
            parents=[item['parent'] for item in wbs_data],
            values=[item['value'] for item in wbs_data],
            text=[item['text'] for item in wbs_data],
            ids=[item['id'] for item in wbs_data],
            marker=dict(
                colors=[item['color_value'] for item in wbs_data],
                colorscale='RdYlGn_r',
                cmid=50,
                colorbar=dict(title="Criticité<br>Score")
            ),
            texttemplate=(
                "<b>%{label}</b><br>"
                "%{text}<br>"
                "%{value:,.0f}€<br>"
                "%{color:.0f}% critique"
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "WBS: %{id}<br>"
                "Budget: €%{value:,.0f}<br>"
                "Criticité: %{color:.0f}%<br>"
                "<extra></extra>"
            ),
            textposition="middle center",
            pathbar=dict(
                visible=True,
                thickness=20,
                edgeshape='>'
            )
        ))
        
        fig.update_layout(
            title={
                'text': "🗂️ <b>Structure de Découpage du Projet (WBS) Interactive</b>",
                'font': {'size': 20}
            },
            height=600,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig
    
    def create_wbs_network(self, plan_data: Dict[str, Any]) -> go.Figure:
        """
        Crée une vue réseau du WBS avec les relations
        
        Caractéristiques:
        - Visualisation en graphe
        - Taille des nœuds = budget
        - Couleur = criticité
        - Liens = dépendances
        """
        G = nx.DiGraph()
        
        # Extraire la structure WBS
        wbs_structure = self._extract_wbs_for_network(plan_data)
        
        if not wbs_structure['nodes']:
            wbs_structure = self._generate_demo_network_data()
        
        # Ajouter les nœuds et arêtes
        for node in wbs_structure['nodes']:
            G.add_node(node['id'], **node)
        
        for edge in wbs_structure['edges']:
            G.add_edge(edge['source'], edge['target'])
        
        # Calculer le layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Créer les traces pour Plotly
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none'
            ))
        
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='RdYlGn_r',
                size=[],
                color=[],
                colorbar=dict(
                    thickness=15,
                    title="Criticité",
                    xanchor="left"
                ),
                line=dict(width=2, color='white')
            )
        )
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            node_info = G.nodes[node]
            node_trace['text'] += tuple([node_info.get('label', node)])
            node_trace['marker']['size'] += tuple([node_info.get('size', 20)])
            node_trace['marker']['color'] += tuple([node_info.get('criticality', 50)])
            
            hover_text = (
                f"<b>{node_info.get('label', node)}</b><br>"
                f"WBS: {node}<br>"
                f"Budget: €{node_info.get('budget', 0):,.0f}<br>"
                f"Durée: {node_info.get('duration', 0)} jours<br>"
                f"Criticité: {node_info.get('criticality', 0):.0f}%"
            )
            # Ajouter le hover text pour le noeud
            current_hovertexts = list(node_trace.hovertext) if hasattr(node_trace, 'hovertext') and node_trace.hovertext else []
            current_hovertexts.append(hover_text)
            node_trace.hovertext = current_hovertexts
        
        # Créer la figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title="🔗 <b>Vue Réseau du WBS avec Criticité</b>",
            showlegend=False,
            hovermode='closest',
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_resource_timeline(self, plan_data: Dict[str, Any]) -> go.Figure:
        """
        Crée une timeline des ressources (qui fait quoi quand)
        
        Caractéristiques:
        - Vue par ressource
        - Charge de travail visible
        - Conflits d'allocation
        """
        resources_data = self._extract_resources_timeline(plan_data)
        
        if not resources_data:
            resources_data = self._generate_demo_resources_data()
        
        fig = go.Figure()
        
        resources = list(set([r['resource'] for r in resources_data]))
        colors = px.colors.qualitative.Set3
        
        for i, resource in enumerate(resources):
            resource_tasks = [r for r in resources_data if r['resource'] == resource]
            
            for task in resource_tasks:
                fig.add_trace(go.Scatter(
                    x=[task['start'], task['end']],
                    y=[i, i],
                    mode='lines',
                    line=dict(
                        color=colors[i % len(colors)],
                        width=15
                    ),
                    name=resource,
                    legendgroup=resource,
                    showlegend=task == resource_tasks[0],
                    hovertemplate=(
                        f"<b>{task['task']}</b><br>"
                        f"Ressource: {resource}<br>"
                        f"Période: {task['start'].strftime('%d/%m')} - {task['end'].strftime('%d/%m')}<br>"
                        f"Charge: {task['workload']}%<br>"
                        "<extra></extra>"
                    )
                ))
                
                # Ajouter le nom de la tâche
                fig.add_annotation(
                    x=task['start'],
                    y=i,
                    text=f" {task['task'][:20]}",
                    showarrow=False,
                    font=dict(size=9),
                    xanchor='left'
                )
                
                # Indicateur de surcharge
                if task['workload'] > 100:
                    fig.add_trace(go.Scatter(
                        x=[task['start']],
                        y=[i],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=10,
                            color='red'
                        ),
                        showlegend=False,
                        hovertemplate="⚠️ Surcharge!<extra></extra>"
                    ))
        
        fig.update_layout(
            title="👥 <b>Planning des Ressources</b>",
            xaxis=dict(
                title="Timeline",
                type='date',
                tickformat='%d %b'
            ),
            yaxis=dict(
                title="Ressources",
                tickmode='array',
                tickvals=list(range(len(resources))),
                ticktext=resources
            ),
            height=max(300, len(resources) * 60 + 100),
            hovermode='closest'
        )
        
        return fig
    
    # Méthodes privées d'extraction et calcul
    def _extract_tasks_with_dependencies(self, plan_data: Dict[str, Any], 
                                        start_date: datetime) -> List[Dict]:
        """Extrait les tâches avec calcul des dates et dépendances"""
        tasks = []
        current_date = start_date
        task_id_counter = 0
        
        phases = plan_data.get("wbs", {}).get("phases", [])
        
        for phase_idx, phase in enumerate(phases):
            phase_name = phase.get("name", f"Phase {phase_idx + 1}")
            
            for task_idx, task in enumerate(phase.get("tasks", [])):
                task_id = f"T{task_id_counter}"
                duration = task.get("duration", 5)
                
                # Calculer les dates réelles
                task_start = current_date
                task_end = task_start + timedelta(days=duration)
                
                # Déterminer le statut (simulation)
                if task_idx == 0 and phase_idx == 0:
                    status = "completed"
                    progress = 100
                elif task_idx <= 1 and phase_idx == 0:
                    status = "in_progress"
                    progress = 60
                else:
                    status = "planned"
                    progress = 0
                
                tasks.append({
                    'task_id': task_id,
                    'task_name': task.get("name", f"Tâche {task_idx + 1}"),
                    'phase': phase_name,
                    'start_date': task_start,
                    'end_date': task_end,
                    'duration': duration,
                    'status': status,
                    'progress': progress,
                    'resources': task.get("resources", "Non assigné"),
                    'cost': task.get("cost", 0),
                    'is_milestone': task.get("is_milestone", False),
                    'is_critical': task.get("is_critical", task_idx % 3 == 0),
                    'dependencies': [f"T{task_id_counter-1}"] if task_id_counter > 0 else []
                })
                
                current_date = task_end
                task_id_counter += 1
        
        return tasks
    
    def _calculate_critical_path(self, tasks: List[Dict]) -> List[str]:
        """Calcule le chemin critique en utilisant l'algorithme CPM"""
        if not tasks:
            return []
        
        # Construire le graphe des dépendances
        G = nx.DiGraph()
        
        for task in tasks:
            G.add_node(task['task_id'], duration=task['duration'])
            for dep in task['dependencies']:
                if dep in [t['task_id'] for t in tasks]:
                    G.add_edge(dep, task['task_id'])
        
        # Trouver le chemin le plus long (critique)
        if not G.nodes():
            return []
        
        # Identifier les nœuds de début et fin
        starts = [n for n in G.nodes() if G.in_degree(n) == 0]
        ends = [n for n in G.nodes() if G.out_degree(n) == 0]
        
        if not starts or not ends:
            return []
        
        # Ajouter des nœuds virtuels début/fin
        G.add_node('START', duration=0)
        G.add_node('END', duration=0)
        
        for s in starts:
            G.add_edge('START', s)
        for e in ends:
            G.add_edge(e, 'END')
        
        try:
            # Calculer le chemin critique
            critical_path = nx.dag_longest_path(G, weight='duration')
            # Retirer les nœuds virtuels
            critical_path = [n for n in critical_path if n not in ['START', 'END']]
            return critical_path
        except:
            # Fallback: marquer quelques tâches comme critiques
            return [t['task_id'] for t in tasks if t.get('is_critical', False)]
    
    def _extract_wbs_structure(self, plan_data: Dict[str, Any]) -> List[Dict]:
        """Extrait la structure WBS pour le treemap"""
        wbs_items = []
        
        # Racine du projet
        project_name = plan_data.get("project_overview", {}).get("title", "Projet")
        total_cost = plan_data.get("project_overview", {}).get("total_cost", 0)
        
        wbs_items.append({
            'id': '0',
            'label': project_name,
            'parent': '',
            'value': total_cost,
            'text': f"Budget Total: €{total_cost:,.0f}",
            'color_value': 50  # Criticité moyenne par défaut
        })
        
        phases = plan_data.get("wbs", {}).get("phases", [])
        
        for phase_idx, phase in enumerate(phases):
            phase_id = f"1.{phase_idx + 1}"
            phase_cost = sum(t.get("cost", 0) for t in phase.get("tasks", []))
            
            wbs_items.append({
                'id': phase_id,
                'label': phase.get("name", f"Phase {phase_idx + 1}"),
                'parent': '0',
                'value': phase_cost,
                'text': f"Phase {phase_idx + 1}/{len(phases)}",
                'color_value': 30 + phase_idx * 10  # Criticité progressive
            })
            
            for task_idx, task in enumerate(phase.get("tasks", [])):
                task_id = f"{phase_id}.{task_idx + 1}"
                task_cost = task.get("cost", 1000)
                
                wbs_items.append({
                    'id': task_id,
                    'label': task.get("name", f"Tâche {task_idx + 1}"),
                    'parent': phase_id,
                    'value': task_cost,
                    'text': f"{task.get('duration', 5)}j",
                    'color_value': 80 if task.get("is_critical", False) else 20
                })
        
        return wbs_items
    
    def _extract_wbs_for_network(self, plan_data: Dict[str, Any]) -> Dict:
        """Extrait la structure WBS pour le graphe réseau"""
        nodes = []
        edges = []
        
        phases = plan_data.get("wbs", {}).get("phases", [])
        
        # Nœud racine
        nodes.append({
            'id': 'root',
            'label': 'Projet',
            'size': 40,
            'budget': plan_data.get("project_overview", {}).get("total_cost", 0),
            'duration': plan_data.get("project_overview", {}).get("total_duration", 0),
            'criticality': 50
        })
        
        for phase_idx, phase in enumerate(phases):
            phase_id = f"phase_{phase_idx}"
            phase_cost = sum(t.get("cost", 0) for t in phase.get("tasks", []))
            phase_duration = sum(t.get("duration", 0) for t in phase.get("tasks", []))
            
            nodes.append({
                'id': phase_id,
                'label': phase.get("name", f"Phase {phase_idx + 1}"),
                'size': 30,
                'budget': phase_cost,
                'duration': phase_duration,
                'criticality': 40 + phase_idx * 10
            })
            
            edges.append({'source': 'root', 'target': phase_id})
            
            for task_idx, task in enumerate(phase.get("tasks", [])):
                task_id = f"{phase_id}_task_{task_idx}"
                
                nodes.append({
                    'id': task_id,
                    'label': task.get("name", f"Tâche {task_idx + 1}"),
                    'size': 15 + task.get("cost", 1000) / 1000,
                    'budget': task.get("cost", 0),
                    'duration': task.get("duration", 0),
                    'criticality': 80 if task.get("is_critical", False) else 20
                })
                
                edges.append({'source': phase_id, 'target': task_id})
        
        return {'nodes': nodes, 'edges': edges}
    
    def _extract_resources_timeline(self, plan_data: Dict[str, Any]) -> List[Dict]:
        """Extrait les données de ressources pour la timeline"""
        resources_data = []
        start_date = datetime.now()
        
        # Simuler l'allocation des ressources
        resources_list = ["Chef de Projet", "Développeur Senior", "Développeur Junior", 
                         "Designer UX", "DevOps", "Testeur QA"]
        
        phases = plan_data.get("wbs", {}).get("phases", [])
        
        current_date = start_date
        for phase in phases:
            for task in phase.get("tasks", []):
                # Assigner des ressources (simulation)
                assigned_resources = np.random.choice(resources_list, 
                                                     size=np.random.randint(1, 3), 
                                                     replace=False)
                
                for resource in assigned_resources:
                    duration = task.get("duration", 5)
                    workload = np.random.randint(50, 120)  # % de charge
                    
                    resources_data.append({
                        'resource': resource,
                        'task': task.get("name", "Tâche"),
                        'start': current_date,
                        'end': current_date + timedelta(days=duration),
                        'workload': workload
                    })
                
                current_date += timedelta(days=duration)
        
        return resources_data
    
    # Méthodes de génération de données démo
    def _generate_demo_gantt_data(self, start_date: datetime) -> List[Dict]:
        """Génère des données démo réalistes pour le Gantt"""
        tasks = []
        current_date = start_date
        
        demo_phases = [
            {
                "name": "Phase 1: Analyse et Conception",
                "tasks": [
                    {"name": "Analyse des besoins", "duration": 5, "critical": True},
                    {"name": "Architecture technique", "duration": 3, "critical": True},
                    {"name": "Design UX/UI", "duration": 4, "critical": False},
                    {"name": "Validation client", "duration": 2, "critical": True, "milestone": True}
                ]
            },
            {
                "name": "Phase 2: Développement",
                "tasks": [
                    {"name": "Setup environnement", "duration": 2, "critical": False},
                    {"name": "Développement Backend", "duration": 15, "critical": True},
                    {"name": "Développement Frontend", "duration": 12, "critical": True},
                    {"name": "Intégration API", "duration": 5, "critical": False},
                    {"name": "Tests unitaires", "duration": 3, "critical": False}
                ]
            },
            {
                "name": "Phase 3: Tests et Déploiement",
                "tasks": [
                    {"name": "Tests d'intégration", "duration": 4, "critical": True},
                    {"name": "Tests utilisateurs", "duration": 5, "critical": True},
                    {"name": "Corrections bugs", "duration": 3, "critical": False},
                    {"name": "Déploiement production", "duration": 2, "critical": True, "milestone": True},
                    {"name": "Formation utilisateurs", "duration": 3, "critical": False}
                ]
            }
        ]
        
        task_id = 0
        for phase_idx, phase in enumerate(demo_phases):
            for task_idx, task_info in enumerate(phase["tasks"]):
                task_start = current_date
                task_end = task_start + timedelta(days=task_info["duration"])
                
                # Statut basé sur la position
                if phase_idx == 0:
                    status = "completed"
                    progress = 100
                elif phase_idx == 1 and task_idx < 2:
                    status = "in_progress"
                    progress = 60
                else:
                    status = "planned"
                    progress = 0
                
                tasks.append({
                    'task_id': f"T{task_id}",
                    'task_name': task_info["name"],
                    'phase': phase["name"],
                    'start_date': task_start,
                    'end_date': task_end,
                    'duration': task_info["duration"],
                    'status': status,
                    'progress': progress,
                    'resources': f"Équipe {(task_id % 3) + 1}",
                    'cost': task_info["duration"] * 1000,
                    'is_milestone': task_info.get("milestone", False),
                    'is_critical': task_info.get("critical", False),
                    'dependencies': [f"T{task_id-1}"] if task_id > 0 else []
                })
                
                current_date = task_end
                task_id += 1
        
        return tasks
    
    def _generate_demo_wbs_data(self) -> List[Dict]:
        """Génère des données WBS démo"""
        return [
            {'id': '0', 'label': 'Projet IA', 'parent': '', 'value': 100000, 
             'text': 'Budget Total', 'color_value': 50},
            {'id': '1', 'label': 'Phase 1: Conception', 'parent': '0', 'value': 25000,
             'text': 'Q1 2025', 'color_value': 30},
            {'id': '1.1', 'label': 'Analyse', 'parent': '1', 'value': 10000,
             'text': '10j', 'color_value': 20},
            {'id': '1.2', 'label': 'Architecture', 'parent': '1', 'value': 15000,
             'text': '15j', 'color_value': 80},
            {'id': '2', 'label': 'Phase 2: Développement', 'parent': '0', 'value': 50000,
             'text': 'Q2 2025', 'color_value': 60},
            {'id': '2.1', 'label': 'Backend', 'parent': '2', 'value': 25000,
             'text': '20j', 'color_value': 90},
            {'id': '2.2', 'label': 'Frontend', 'parent': '2', 'value': 20000,
             'text': '18j', 'color_value': 70},
            {'id': '2.3', 'label': 'Tests', 'parent': '2', 'value': 5000,
             'text': '5j', 'color_value': 40},
            {'id': '3', 'label': 'Phase 3: Déploiement', 'parent': '0', 'value': 25000,
             'text': 'Q3 2025', 'color_value': 45}
        ]
    
    def _generate_demo_network_data(self) -> Dict:
        """Génère des données réseau démo"""
        return {
            'nodes': [
                {'id': 'root', 'label': 'Projet', 'size': 40, 'budget': 100000, 
                 'duration': 60, 'criticality': 50},
                {'id': 'p1', 'label': 'Conception', 'size': 25, 'budget': 25000,
                 'duration': 15, 'criticality': 30},
                {'id': 'p2', 'label': 'Développement', 'size': 35, 'budget': 50000,
                 'duration': 30, 'criticality': 80},
                {'id': 'p3', 'label': 'Déploiement', 'size': 20, 'budget': 25000,
                 'duration': 15, 'criticality': 60}
            ],
            'edges': [
                {'source': 'root', 'target': 'p1'},
                {'source': 'root', 'target': 'p2'},
                {'source': 'root', 'target': 'p3'},
                {'source': 'p1', 'target': 'p2'},
                {'source': 'p2', 'target': 'p3'}
            ]
        }
    
    def _generate_demo_resources_data(self) -> List[Dict]:
        """Génère des données de ressources démo"""
        start = datetime.now()
        return [
            {'resource': 'Chef de Projet', 'task': 'Coordination', 
             'start': start, 'end': start + timedelta(days=60), 'workload': 80},
            {'resource': 'Développeur Senior', 'task': 'Architecture',
             'start': start, 'end': start + timedelta(days=15), 'workload': 100},
            {'resource': 'Développeur Senior', 'task': 'Backend API',
             'start': start + timedelta(days=15), 'end': start + timedelta(days=35), 'workload': 90},
            {'resource': 'Designer UX', 'task': 'Maquettes',
             'start': start, 'end': start + timedelta(days=10), 'workload': 100},
            {'resource': 'Designer UX', 'task': 'Tests utilisateurs',
             'start': start + timedelta(days=40), 'end': start + timedelta(days=45), 'workload': 60}
        ]