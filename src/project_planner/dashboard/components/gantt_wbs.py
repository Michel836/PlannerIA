"""
Gantt Chart complet basé sur WBS - Reconstruction propre
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np


def create_gantt_from_wbs(tasks: List[Dict]) -> pd.DataFrame:
    """Crée un DataFrame Gantt propre à partir des tâches WBS"""
    
    gantt_data = []
    current_date = datetime(2025, 8, 15)  # Date de début alignée avec le projet 2025
    
    for i, task in enumerate(tasks):
        # Extraction sécurisée des données
        task_name = str(task.get('name', f'Tâche {i+1}'))
        task_id = str(task.get('id', f'task_{i}'))
        
        # Durée (conversion sécurisée)
        duration_raw = task.get('duration', 7)
        try:
            duration_days = int(float(str(duration_raw)))  # Triple conversion sécurisée
        except:
            duration_days = 7
        
        # Dates calculées proprement
        start_date = current_date + timedelta(days=i * 2)  # Espacement de 2 jours
        end_date = start_date + timedelta(days=duration_days)
        
        # Progression
        try:
            progress = float(task.get('progress', 0))
            progress = min(100, max(0, progress))  # Clamp entre 0 et 100
        except:
            progress = 0.0
        
        # Priorité et couleur
        priority = str(task.get('priority', 'medium'))
        
        # Statut basé sur progression
        if progress >= 95:
            status = 'Terminé'
        elif progress > 5:
            status = 'En cours'
        else:
            status = 'Non démarré'
        
        gantt_data.append({
            'ID': task_id,
            'Tâche': task_name,
            'Début': start_date,  # Garder comme datetime pour Plotly
            'Fin': end_date,      # Garder comme datetime pour Plotly
            'Début_str': start_date.strftime('%Y-%m-%d'),  # String pour affichage
            'Fin_str': end_date.strftime('%Y-%m-%d'),      # String pour affichage
            'Durée': duration_days,
            'Progression': progress,
            'Priorité': priority,
            'Statut': status,
            'Équipe': task.get('team_size', 1),
            'Budget': task.get('budget', 0)
        })
    
    return pd.DataFrame(gantt_data)


def render_gantt_chart_wbs(tasks: List[Dict]):
    """Affiche un Gantt chart basé sur WBS - Version propre"""
    
    st.markdown("### 📊 Diagramme de Gantt WBS")
    
    if not tasks:
        st.info("Aucune tâche disponible")
        return
    
    # Création du DataFrame Gantt
    try:
        df_gantt = create_gantt_from_wbs(tasks)
        
        # Contrôles
        col1, col2, col3 = st.columns(3)
        with col1:
            show_progress = st.checkbox("Afficher progression", True)
        with col2:
            filter_status = st.selectbox("Filtrer par statut", 
                                       ["Tous", "En cours", "Terminé", "Non démarré"])
        with col3:
            sort_by = st.selectbox("Trier par", ["Ordre", "Début", "Durée", "Progression"])
        
        # Filtrage
        df_filtered = df_gantt.copy()
        if filter_status != "Tous":
            df_filtered = df_filtered[df_filtered['Statut'] == filter_status]
        
        # Tri
        if sort_by == "Début":
            df_filtered = df_filtered.sort_values('Début')
        elif sort_by == "Durée":
            df_filtered = df_filtered.sort_values('Durée', ascending=False)
        elif sort_by == "Progression":
            df_filtered = df_filtered.sort_values('Progression', ascending=False)
        
        if df_filtered.empty:
            st.warning("Aucune tâche ne correspond aux filtres")
            return
        
        # Graphique Gantt avec Plotly Express (plus stable)
        st.subheader("📈 Timeline Gantt")
        
        # Créer le diagramme de Gantt avec Plotly Express
        try:
            # Timeline Gantt principal
            fig = px.timeline(
                df_filtered,
                x_start="Début",
                x_end="Fin", 
                y="Tâche",
                color="Priorité",
                title="📊 Diagramme de Gantt - Timeline du Projet",
                color_discrete_map={
                    'critical': '#FF4444',
                    'high': '#FF8800', 
                    'medium': '#4488FF',
                    'low': '#88CC88'
                },
                hover_data=['Durée', 'Progression', 'Équipe', 'Budget']
            )
            
            # Configuration du graphique
            fig.update_layout(
                height=max(500, len(df_filtered) * 40),  # Plus haut pour visibilité
                xaxis_title="📅 Timeline",
                yaxis_title="📋 Tâches",
                showlegend=True,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            # Améliorer les barres
            fig.update_traces(
                marker_line_width=2,
                marker_line_color="white"
            )
            
            # Ligne aujourd'hui
            try:
                today = datetime.now()
                fig.add_shape(
                    type="line",
                    x0=today, x1=today,
                    y0=-0.5, y1=len(df_filtered)-0.5,
                    line=dict(color="red", width=3, dash="dash"),
                    name="Aujourd'hui"
                )
                # Annotation pour la ligne
                fig.add_annotation(
                    x=today,
                    y=len(df_filtered)-0.5,
                    text="📍 Aujourd'hui",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1
                )
            except:
                pass  # Ignorer si problème avec ligne aujourd'hui
            
            # Afficher le graphique principal
            st.plotly_chart(fig, use_container_width=True, key="gantt_main")
            
        except Exception as timeline_e:
            st.error(f"❌ Erreur timeline Plotly: {timeline_e}")
            
            # Graphique de secours simple mais visible
            st.subheader("📊 Gantt simplifié (mode secours)")
            
            fig_backup = go.Figure()
            colors = {'critical': '#FF4444', 'high': '#FF8800', 'medium': '#4488FF', 'low': '#88CC88'}
            
            for idx, row in df_filtered.iterrows():
                color = colors.get(row['Priorité'], '#4488FF')
                fig_backup.add_trace(go.Scatter(
                    x=[row['Début'], row['Fin']],
                    y=[row['Tâche'], row['Tâche']],
                    mode='lines',
                    line=dict(width=25, color=color),
                    name=row['Tâche'],
                    showlegend=False,
                    hovertemplate=f"<b>{row['Tâche']}</b><br>Début: {row['Début_str']}<br>Fin: {row['Fin_str']}<br>Durée: {row['Durée']} jours<extra></extra>"
                ))
            
            fig_backup.update_layout(
                title="Gantt Chart - Mode Secours",
                height=max(400, len(df_filtered) * 30),
                xaxis_title="Timeline",
                yaxis_title="Tâches"
            )
            
            st.plotly_chart(fig_backup, use_container_width=True, key="gantt_backup")
        
        # Graphique de progression (barres horizontales)
        if show_progress:
            st.subheader("📊 Progression par tâche")
            fig_progress = px.bar(
                df_filtered,
                x='Progression',
                y='Tâche',
                orientation='h',
                color='Statut',
                title="Progression des tâches (%)",
                color_discrete_map={
                    'Terminé': '#00AA00',
                    'En cours': '#FFAA00',
                    'Non démarré': '#CCCCCC'
                }
            )
            fig_progress.update_layout(height=max(300, len(df_filtered) * 25))
            st.plotly_chart(fig_progress, use_container_width=True)
        
        # Tableau WBS détaillé
        st.subheader("📋 Tableau WBS - Work Breakdown Structure")
        st.dataframe(
            df_filtered[['ID', 'Tâche', 'Début', 'Fin', 'Durée', 'Progression', 'Priorité', 'Statut']],
            use_container_width=True
        )
        
        # Métriques de synthèse
        st.subheader("📊 Métriques du projet")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total tâches", len(df_filtered))
        with col2:
            terminées = len(df_filtered[df_filtered['Statut'] == 'Terminé'])
            st.metric("Terminées", f"{terminées}/{len(df_filtered)}")
        with col3:
            avg_progress = df_filtered['Progression'].mean()
            st.metric("Progression moy.", f"{avg_progress:.1f}%")
        with col4:
            total_duration = df_filtered['Durée'].sum()
            st.metric("Durée totale", f"{total_duration} jours")
        
    except Exception as e:
        st.error(f"❌ Erreur création Gantt WBS: {str(e)}")
        
        # Fallback : tableau simple
        st.subheader("📋 Vue tableau (fallback)")
        simple_data = []
        for task in tasks[:10]:
            simple_data.append({
                'Tâche': str(task.get('name', 'Sans nom')),
                'Progression': f"{task.get('progress', 0):.1f}%",
                'Priorité': str(task.get('priority', 'medium')),
                'Durée': f"{task.get('duration', 'N/A')} jours"
            })
        
        if simple_data:
            st.dataframe(pd.DataFrame(simple_data), use_container_width=True)


def render_wbs_gantt_integrated(tasks: List[Dict]):
    """Version intégrée WBS + Gantt pour l'onglet Planning"""
    render_gantt_chart_wbs(tasks)