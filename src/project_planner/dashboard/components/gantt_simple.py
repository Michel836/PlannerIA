"""
Module Gantt Simplifié - Version sécurisée sans erreurs datetime
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any


def render_gantt_simple(tasks: List[Dict], timeline_data: Dict = None):
    """Rendu Gantt ultra-simplifié et sécurisé"""
    st.markdown("### 📊 Gantt Chart Complet")
    
    if not tasks:
        st.info("Aucune tâche disponible pour le Gantt")
        return
    
    # Contrôles simples
    col1, col2 = st.columns(2)
    with col1:
        view_mode = st.selectbox("Vue", ["Toutes tâches", "Tâches critiques", "En cours"])
    with col2:
        show_progress = st.checkbox("Afficher progression", True)
    
    # Filtrage
    filtered_tasks = tasks.copy()
    if view_mode == "Tâches critiques":
        filtered_tasks = [t for t in tasks if t.get('is_critical', False)]
    elif view_mode == "En cours":
        filtered_tasks = [t for t in tasks if 5 < t.get('progress', 0) < 95]
    
    if not filtered_tasks:
        st.info("Aucune tâche ne correspond aux filtres")
        return
    
    try:
        # Préparation des données ultra-sécurisée
        gantt_data = []
        base_date = datetime.now()
        print(f"DEBUG GANTT: Base date = {base_date} ({type(base_date)})")
        
        for i, task in enumerate(filtered_tasks):
            print(f"DEBUG GANTT: Traitement tâche {i}: {task.get('name', 'unnamed')}")
            
            # Extraction sécurisée
            name = str(task.get('name', f'Tâche {i+1}'))[:40]
            print(f"DEBUG GANTT: Name = {name}")
            
            # Dates par défaut - AVEC VERIFICATION TYPES
            print(f"DEBUG GANTT: Calcul start = base_date + timedelta(days={i*7})")
            print(f"DEBUG GANTT: base_date type = {type(base_date)}, i*7 = {i*7} ({type(i*7)})")
            
            try:
                start = base_date + timedelta(days=i*7)
                print(f"DEBUG GANTT: Start calculé = {start} ({type(start)})")
            except Exception as start_e:
                print(f"DEBUG GANTT: ERREUR calcul start: {start_e}")
                start = datetime.now()
                
            try:
                end = start + timedelta(days=7)
                print(f"DEBUG GANTT: End calculé = {end} ({type(end)})")
            except Exception as end_e:
                print(f"DEBUG GANTT: ERREUR calcul end: {end_e}")
                end = datetime.now() + timedelta(days=7)
            
            # Tentative d'extraction des vraies dates
            try:
                if 'start_date' in task and task['start_date']:
                    raw_start = task['start_date']
                    if isinstance(raw_start, datetime):
                        start = raw_start
                    elif isinstance(raw_start, str):
                        start = datetime.fromisoformat(raw_start[:19])
                
                if 'end_date' in task and task['end_date']:
                    raw_end = task['end_date'] 
                    if isinstance(raw_end, datetime):
                        end = raw_end
                    elif isinstance(raw_end, str):
                        end = datetime.fromisoformat(raw_end[:19])
            except:
                # Garder les dates par défaut si erreur
                pass
            
            # Validation finale
            if end <= start:
                end = start + timedelta(days=7)
            
            gantt_data.append({
                'Task': name,
                'Start': start,
                'End': end,
                'Progress': min(100, max(0, float(task.get('progress', 0)))),
                'Priority': str(task.get('priority', 'medium'))
            })
        
        # Création DataFrame
        df = pd.DataFrame(gantt_data)
        
        # Graphique Plotly simple
        fig = go.Figure()
        
        colors = {
            'critical': '#FF4444', 'high': '#FF8800', 
            'medium': '#4488FF', 'low': '#88CC88'
        }
        
        for i, row in df.iterrows():
            color = colors.get(row['Priority'], '#4488FF')
            
            # Barre de tâche
            fig.add_trace(go.Scatter(
                x=[row['Start'], row['End']],
                y=[i, i],
                mode='lines',
                line=dict(color=color, width=20),
                name=row['Task'],
                hovertemplate=(
                    f"<b>{row['Task']}</b><br>"
                    f"Début: {row['Start'].strftime('%Y-%m-%d')}<br>"
                    f"Fin: {row['End'].strftime('%Y-%m-%d')}<br>"
                    f"Progression: {row['Progress']:.1f}%<br>"
                    f"Priorité: {row['Priority']}<extra></extra>"
                ),
                showlegend=False
            ))
            
            # Barre de progression
            if show_progress and row['Progress'] > 0:
                total_seconds = (row['End'] - row['Start']).total_seconds()
                progress_seconds = total_seconds * (row['Progress'] / 100)
                progress_end = row['Start'] + timedelta(seconds=progress_seconds)
                
                fig.add_trace(go.Scatter(
                    x=[row['Start'], progress_end],
                    y=[i, i],
                    mode='lines',
                    line=dict(color='#00CC00', width=25),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Configuration
        fig.update_layout(
            title="📊 Diagramme de Gantt",
            xaxis_title="Timeline",
            yaxis=dict(
                title="Tâches",
                tickmode='array',
                tickvals=list(range(len(df))),
                ticktext=df['Task'].tolist(),
                autorange='reversed'
            ),
            height=max(400, len(df) * 50),
            showlegend=False
        )
        
        # Ligne aujourd'hui
        fig.add_vline(
            x=datetime.now(), 
            line_dash="dot", 
            line_color="red",
            annotation_text="Aujourd'hui"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Résumé
        st.subheader("📋 Résumé")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tâches", len(filtered_tasks))
        with col2:
            completed = sum(1 for t in filtered_tasks if t.get('progress', 0) >= 95)
            st.metric("Terminées", f"{completed}/{len(filtered_tasks)}")
        with col3:
            avg_progress = sum(t.get('progress', 0) for t in filtered_tasks) / len(filtered_tasks)
            st.metric("Progression", f"{avg_progress:.1f}%")
            
    except Exception as e:
        st.error(f"❌ Erreur Gantt: {str(e)}")
        st.info("🔄 Fonctionnalité en cours de développement")
        
        # Fallback ultra-simple
        st.subheader("📋 Liste des tâches")
        for i, task in enumerate(filtered_tasks[:10]):  # Limite à 10
            with st.expander(f"Tâche {i+1}: {task.get('name', 'Sans nom')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Progression:** {task.get('progress', 0):.1f}%")
                    st.write(f"**Priorité:** {task.get('priority', 'medium')}")
                with col2:
                    st.write(f"**Durée:** {task.get('duration', 'N/A')} jours")
                    st.write(f"**Équipe:** {task.get('team_size', 'N/A')} personnes")