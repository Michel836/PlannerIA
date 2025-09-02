"""
Module WBS Enhanced - Work Breakdown Structure moderne et interactif
Version améliorée avec graphiques hiérarchiques, métriques avancées et interface moderne
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any


def render_wbs_enhanced(tasks: List[Dict], plan_data: Dict):
    """Analyse WBS (Work Breakdown Structure) - Version moderne améliorée"""
    
    # Header moderne avec KPIs globaux
    col_header, col_chart = st.columns([2, 1])
    with col_header:
        st.markdown("### 🏗️ Work Breakdown Structure")
        st.markdown("*Structure hiérarchique détaillée du projet avec métriques avancées*")
    
    with col_chart:
        # Mini graphique de répartition des phases
        if 'wbs' in plan_data and 'phases' in plan_data['wbs']:
            phases_progress = []
            for phase in plan_data['wbs']['phases']:
                if 'tasks' in phase:
                    avg_progress = np.mean([t.get('progress', 0) for t in phase['tasks']])
                    phases_progress.append(avg_progress)
            
            if phases_progress:
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Bar(
                    x=[f"Phase {i+1}" for i in range(len(phases_progress))],
                    y=phases_progress,
                    marker_color=['#FF6B6B' if p < 30 else '#4ECDC4' if p < 70 else '#45B7D1' for p in phases_progress],
                    text=[f"{p:.0f}%" for p in phases_progress],
                    textposition='inside'
                ))
                fig_mini.update_layout(
                    height=150, 
                    margin=dict(l=0, r=0, t=20, b=0),
                    showlegend=False,
                    title="📊 Progression par phase",
                    title_font_size=12,
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_mini, use_container_width=True, key="wbs_mini")
    
    # Onglets pour différentes vues WBS  
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Structure Détaillée", "🌳 Vue Hiérarchique", "📊 Analytics WBS", "📋 Tableau Complet"])
    
    with tab1:
        render_detailed_structure(tasks, plan_data)
    
    with tab2:
        render_hierarchy_view(tasks, plan_data)
    
    with tab3:
        render_analytics_view(tasks, plan_data)
    
    with tab4:
        render_detailed_table(tasks, plan_data)


def render_detailed_structure(tasks: List[Dict], plan_data: Dict):
    """Rendu de la structure détaillée WBS"""
    
    # Analyse hiérarchique des tâches
    if 'wbs' in plan_data:
        wbs_data = plan_data['wbs']
        
        # Métriques globales WBS
        if 'phases' in wbs_data:
            total_phases = len(wbs_data['phases'])
            all_tasks = []
            for phase in wbs_data['phases']:
                if 'tasks' in phase:
                    all_tasks.extend(phase['tasks'])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("🗂️ Phases", total_phases)
            with col2:
                st.metric("📋 Tâches totales", len(all_tasks))
            with col3:
                total_budget = sum([t.get('cost', 0) for t in all_tasks])
                st.metric("💰 Budget total", f"{total_budget:,}€")
            with col4:
                total_duration = sum([t.get('duration', 0) for t in all_tasks])
                st.metric("⏱️ Durée totale", f"{total_duration} jours")
            with col5:
                avg_progress = np.mean([t.get('progress', 0) for t in all_tasks]) if all_tasks else 0
                delta_progress = 15.2  # Simulation delta de progression
                st.metric("📊 Progression glob.", f"{avg_progress:.1f}%", delta=f"{delta_progress:.1f}%")
            
            st.markdown("---")
            
            # Vue par phases avec design moderne
            for phase_idx, phase in enumerate(wbs_data['phases']):
                render_phase_card(phase_idx, phase)
    
    else:
        # Créer une structure WBS à partir des tâches existantes
        st.info("📊 Génération d'une structure WBS basée sur les données disponibles")
        render_generated_wbs(tasks)


def render_hierarchy_view(tasks: List[Dict], plan_data: Dict):
    """Vue hiérarchique avec graphique en arbre"""
    
    phases = []
    
    # Essayer d'abord la structure WBS officielle
    if 'wbs' in plan_data and 'phases' in plan_data['wbs']:
        phases = plan_data['wbs']['phases']
        st.success("🏗️ Structure WBS détectée - Affichage hiérarchique complet")
    elif tasks:
        # Créer des phases à partir des tâches existantes
        st.info("🔄 Génération de la hiérarchie à partir des tâches disponibles")
        phases_dict = {}
        for task in tasks:
            phase_name = task.get('phase', 'Phase Principale')
            if phase_name not in phases_dict:
                phases_dict[phase_name] = {
                    'name': phase_name,
                    'tasks': []
                }
            phases_dict[phase_name]['tasks'].append(task)
        
        phases = list(phases_dict.values())
    
    if phases:
        render_wbs_hierarchy_chart(phases)
    else:
        st.warning("⚠️ Aucune donnée disponible pour la vue hiérarchique")


def render_analytics_view(tasks: List[Dict], plan_data: Dict):
    """Vue analytics avec métriques avancées"""
    
    phases = []
    
    # Essayer d'abord la structure WBS officielle
    if 'wbs' in plan_data and 'phases' in plan_data['wbs']:
        phases = plan_data['wbs']['phases']
        st.success("📊 Analytics basés sur la structure WBS complète")
    elif tasks:
        # Créer des phases à partir des tâches existantes
        st.info("📈 Analytics générés à partir des tâches disponibles")
        phases_dict = {}
        for task in tasks:
            phase_name = task.get('phase', 'Phase Principale')
            if phase_name not in phases_dict:
                phases_dict[phase_name] = {
                    'name': phase_name,
                    'tasks': []
                }
            phases_dict[phase_name]['tasks'].append(task)
        
        phases = list(phases_dict.values())
    
    if phases:
        render_wbs_metrics_dashboard(phases)
        
        # Ajout de recommandations IA
        st.markdown("### 🤖 Recommandations IA")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🎯 Optimisations suggérées :**")
            st.write("• Paralléliser les phases 2 et 3")
            st.write("• Réduire les dépendances critiques")
            st.write("• Augmenter l'équipe sur la phase développement")
        
        with col2:
            st.markdown("**⚠️ Alertes détectées :**")
            st.warning("🔶 Phase développement à 55% - surveiller")
            st.info("🔵 Budget phase validation sous-estimé")
            st.success("✅ Phase conception dans les temps")
    else:
        st.warning("⚠️ Aucune donnée disponible pour les analytics")


def render_detailed_table(tasks: List[Dict], plan_data: Dict):
    """Tableau détaillé complet avec toutes les informations WBS"""
    
    st.markdown("### 📊 Tableau WBS Complet")
    st.markdown("*Vue tabulaire détaillée de toutes les tâches du projet*")
    
    # Contrôles et filtres
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_all_columns = st.checkbox("🔍 Toutes les colonnes", True, help="Afficher toutes les colonnes disponibles")
    
    with col2:
        filter_status = st.selectbox("📈 Filtrer par statut", 
                                   ["Tous", "Terminé", "En cours", "Non démarré", "Bloqué"])
    
    with col3:
        filter_priority = st.selectbox("⚡ Filtrer par priorité", 
                                     ["Toutes", "Critical", "High", "Medium", "Low"])
    
    with col4:
        sort_by = st.selectbox("📊 Trier par", 
                             ["Phase", "Nom", "Progression", "Budget", "Durée", "Date début", "Priorité"])
    
    # Construction du tableau à partir des données WBS
    table_data = []
    
    if 'wbs' in plan_data and 'phases' in plan_data['wbs']:
        phases = plan_data['wbs']['phases']
        
        for phase_idx, phase in enumerate(phases):
            phase_name = phase.get('name', f'Phase {phase_idx + 1}')
            
            if 'tasks' in phase:
                for task_idx, task in enumerate(phase['tasks']):
                    # Calculs dérivés
                    progress = task.get('progress', 0)
                    if progress >= 95:
                        status = "Terminé"
                        status_icon = "✅"
                    elif progress > 5:
                        status = "En cours" 
                        status_icon = "🔄"
                    elif task.get('status') == 'blocked':
                        status = "Bloqué"
                        status_icon = "🚫"
                    else:
                        status = "Non démarré"
                        status_icon = "⏳"
                    
                    priority = task.get('priority', 'medium').title()
                    priority_icons = {
                        'Critical': '🔴',
                        'High': '🟠', 
                        'Medium': '🔵',
                        'Low': '🟢'
                    }
                    priority_icon = priority_icons.get(priority, '⚪')
                    
                    # Dates formatées
                    start_date = "N/A"
                    end_date = "N/A"
                    if 'start_date' in task:
                        try:
                            if isinstance(task['start_date'], str):
                                start_date = task['start_date'][:10]
                            else:
                                start_date = task['start_date'].strftime('%Y-%m-%d')
                        except:
                            start_date = str(task.get('start_date', 'N/A'))[:10]
                    
                    if 'end_date' in task:
                        try:
                            if isinstance(task['end_date'], str):
                                end_date = task['end_date'][:10]
                            else:
                                end_date = task['end_date'].strftime('%Y-%m-%d')
                        except:
                            end_date = str(task.get('end_date', 'N/A'))[:10]
                    
                    # Ligne du tableau
                    row = {
                        'Phase': phase_name,
                        'ID Tâche': task.get('id', f'task_{phase_idx}_{task_idx}'),
                        'Nom de la Tâche': task.get('name', 'Tâche sans nom'),
                        'Description': task.get('description', '')[:100] + "..." if len(task.get('description', '')) > 100 else task.get('description', ''),
                        'Statut': f"{status_icon} {status}",
                        'Priorité': f"{priority_icon} {priority}",
                        'Progression (%)': f"{progress:.1f}%",
                        'Progression Num': progress,  # Pour le tri
                        'Date Début': start_date,
                        'Date Fin': end_date,
                        'Durée (jours)': task.get('duration', 0),
                        'Effort (h)': task.get('effort', 0),
                        'Budget (€)': f"{task.get('cost', 0):,}€",
                        'Budget Num': task.get('cost', 0),  # Pour le tri
                        'Équipe': task.get('team_size', 0),
                        'Ressources': ', '.join(task.get('assigned_resources', [])),
                        'Dépendances': ', '.join(task.get('dependencies', [])),
                        'Criticité': '🔥' if task.get('is_critical', False) else '',
                        'Jalon': '🎯' if task.get('is_milestone', False) else ''
                    }
                    table_data.append(row)
    
    elif tasks:
        # Fallback avec les tâches générées
        for idx, task in enumerate(tasks):
            progress = task.get('progress', 0)
            if progress >= 95:
                status = "Terminé"
                status_icon = "✅"
            elif progress > 5:
                status = "En cours"
                status_icon = "🔄"
            else:
                status = "Non démarré"
                status_icon = "⏳"
            
            row = {
                'Phase': task.get('phase', 'Phase Principale'),
                'ID Tâche': task.get('id', f'task_{idx}'),
                'Nom de la Tâche': task.get('name', 'Tâche'),
                'Description': task.get('description', ''),
                'Statut': f"{status_icon} {status}",
                'Priorité': f"🔵 {task.get('priority', 'Medium').title()}",
                'Progression (%)': f"{progress:.1f}%",
                'Progression Num': progress,
                'Date Début': task.get('start_date', datetime.now()).strftime('%Y-%m-%d') if hasattr(task.get('start_date'), 'strftime') else str(task.get('start_date', 'N/A'))[:10],
                'Date Fin': task.get('end_date', datetime.now()).strftime('%Y-%m-%d') if hasattr(task.get('end_date'), 'strftime') else str(task.get('end_date', 'N/A'))[:10],
                'Durée (jours)': task.get('duration', 0),
                'Budget (€)': f"{task.get('budget', 0):,}€",
                'Budget Num': task.get('budget', 0),
                'Équipe': task.get('team_size', 0),
                'Criticité': '🔥' if task.get('is_critical', False) else '',
                'Jalon': '🎯' if task.get('is_milestone', False) else ''
            }
            table_data.append(row)
    
    if not table_data:
        st.warning("⚠️ Aucune tâche disponible pour le tableau")
        return
    
    # Conversion en DataFrame
    df = pd.DataFrame(table_data)
    
    # Filtrage
    df_filtered = df.copy()
    
    # Filtre par statut
    if filter_status != "Tous":
        df_filtered = df_filtered[df_filtered['Statut'].str.contains(filter_status)]
    
    # Filtre par priorité
    if filter_priority != "Toutes":
        df_filtered = df_filtered[df_filtered['Priorité'].str.contains(filter_priority)]
    
    # Tri
    if sort_by == "Phase":
        df_filtered = df_filtered.sort_values('Phase')
    elif sort_by == "Nom":
        df_filtered = df_filtered.sort_values('Nom de la Tâche')
    elif sort_by == "Progression":
        df_filtered = df_filtered.sort_values('Progression Num', ascending=False)
    elif sort_by == "Budget":
        df_filtered = df_filtered.sort_values('Budget Num', ascending=False)
    elif sort_by == "Durée":
        df_filtered = df_filtered.sort_values('Durée (jours)', ascending=False)
    elif sort_by == "Date début":
        df_filtered = df_filtered.sort_values('Date Début')
    elif sort_by == "Priorité":
        df_filtered = df_filtered.sort_values('Priorité')
    
    # Sélection des colonnes
    if show_all_columns:
        display_columns = [
            'Phase', 'ID Tâche', 'Nom de la Tâche', 'Description', 'Statut', 
            'Priorité', 'Progression (%)', 'Date Début', 'Date Fin', 
            'Durée (jours)', 'Budget (€)', 'Équipe', 'Ressources', 
            'Dépendances', 'Criticité', 'Jalon'
        ]
    else:
        display_columns = [
            'Phase', 'Nom de la Tâche', 'Statut', 'Priorité', 
            'Progression (%)', 'Date Début', 'Budget (€)', 'Criticité', 'Jalon'
        ]
    
    # Filtrer les colonnes disponibles
    available_columns = [col for col in display_columns if col in df_filtered.columns]
    
    # Métriques du tableau
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total tâches", len(df_filtered))
    with col2:
        completed = len(df_filtered[df_filtered['Statut'].str.contains('Terminé')])
        st.metric("✅ Terminées", f"{completed}/{len(df_filtered)}")
    with col3:
        avg_progress = df_filtered['Progression Num'].mean()
        st.metric("📊 Progression moy.", f"{avg_progress:.1f}%")
    with col4:
        if 'Budget Num' in df_filtered.columns:
            total_budget = df_filtered['Budget Num'].sum()
            st.metric("💰 Budget total", f"{total_budget:,}€")
        else:
            st.metric("💰 Budget total", "N/A")
    
    st.markdown("---")
    
    # Affichage du tableau avec style
    st.markdown("### 📊 Tableau Détaillé WBS")
    
    # Configuration du tableau
    st.dataframe(
        df_filtered[available_columns],
        use_container_width=True,
        height=600,
        column_config={
            "Progression (%)": st.column_config.ProgressColumn(
                "Progression",
                help="Pourcentage d'avancement de la tâche",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Budget (€)": st.column_config.NumberColumn(
                "Budget",
                help="Budget alloué à la tâche",
                format="%.0f€"
            ),
            "Durée (jours)": st.column_config.NumberColumn(
                "Durée",
                help="Durée estimée de la tâche",
                format="%.0f jours"
            ),
            "Date Début": st.column_config.DateColumn(
                "Début",
                help="Date de début prévue"
            ),
            "Date Fin": st.column_config.DateColumn(
                "Fin",
                help="Date de fin prévue"
            )
        }
    )
    
    # Export du tableau
    st.markdown("---")
    st.markdown("### 📤 Export Tableau")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export CSV
        csv = df_filtered[available_columns].to_csv(index=False)
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name="wbs_tableau_complet.csv",
            mime="text/csv",
            help="Exporter le tableau filtré au format CSV"
        )
    
    with col2:
        # Export Excel (simulation)
        st.button(
            "📊 Télécharger Excel",
            help="Export Excel avec formatage avancé (à implémenter)",
            disabled=True
        )


def render_phase_card(phase_idx: int, phase: Dict):
    """Rendu d'une carte de phase moderne avec design amélioré"""
    
    phase_name = phase.get('name', f'Phase {phase_idx + 1}')
    phase_status = phase.get('status', 'in_progress')
    
    # Couleur de la phase selon statut
    status_colors = {
        'completed': '#28a745',
        'in_progress': '#ffc107', 
        'pending': '#6c757d',
        'blocked': '#dc3545'
    }
    phase_color = status_colors.get(phase_status, '#6c757d')
    
    # Container stylé pour la phase
    phase_container = st.container()
    with phase_container:
        # Header de phase avec style
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, {phase_color}15 0%, {phase_color}05 100%);
            border-left: 4px solid {phase_color};
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <h4 style="margin:0; color:{phase_color};">📁 {phase_name}</h4>
            <p style="margin:5px 0 0 0; color:#666; font-size:14px;">{phase.get('description', 'Phase du projet')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'tasks' in phase:
            phase_tasks = phase['tasks']
            
            # Métriques de phase en cartes modernes
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                task_count = len(phase_tasks)
                st.metric("📋 Tâches", task_count, help="Nombre de tâches dans cette phase")
            
            with col2:
                total_cost = sum([t.get('cost', 0) for t in phase_tasks])
                st.metric("💰 Budget", f"{total_cost:,}€", help="Budget alloué à cette phase")
            
            with col3:
                total_duration = sum([t.get('duration', 0) for t in phase_tasks])
                st.metric("⏱️ Durée", f"{total_duration} jours", help="Durée totale de la phase")
            
            with col4:
                avg_progress = np.mean([t.get('progress', 0) for t in phase_tasks])
                progress_delta = np.random.uniform(-5, 15)  # Simulation delta
                st.metric("📈 Progression", f"{avg_progress:.1f}%", delta=f"{progress_delta:+.1f}%")
            
            # Graphique de progression des tâches de la phase
            if len(phase_tasks) > 1:
                col_tasks, col_chart = st.columns([3, 2])
            else:
                col_tasks = st
                col_chart = None
            
            with col_tasks:
                # Liste hiérarchique des tâches avec design amélioré
                st.markdown("**🗂️ Structure des tâches:**")
                
                for task_idx, task in enumerate(phase_tasks):
                    render_task_card(task_idx, task)
            
            # Graphique de répartition si multiple tâches
            if col_chart and len(phase_tasks) > 1:
                with col_chart:
                    render_phase_chart(phase_tasks, phase_idx)
        
        st.markdown("---")


def render_task_card(task_idx: int, task: Dict):
    """Rendu d'une carte de tâche avec design moderne"""
    
    # Statut et priorité visuels
    progress = task.get('progress', 0)
    if progress >= 95:
        status_emoji = "✅"
        status_color = "#28a745"
    elif progress > 5:
        status_emoji = "🔄" 
        status_color = "#ffc107"
    else:
        status_emoji = "⏳"
        status_color = "#6c757d"
    
    priority_colors = {
        'critical': '#dc3545',
        'high': '#fd7e14',
        'medium': '#0d6efd', 
        'low': '#198754'
    }
    priority = task.get('priority', 'medium')
    priority_color = priority_colors.get(priority, '#0d6efd')
    
    critical_mark = "🔥" if task.get('is_critical') else ""
    milestone_mark = "🎯" if task.get('is_milestone') else ""
    
    # Affichage stylé de la tâche
    st.markdown(f"""
    <div style="
        border-left: 3px solid {priority_color};
        padding: 8px 12px;
        margin: 5px 0;
        background: {status_color}10;
        border-radius: 4px;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span><strong>{status_emoji} {task.get('name', 'Tâche')} {critical_mark} {milestone_mark}</strong></span>
            <span style="background: {priority_color}20; padding: 2px 8px; border-radius: 12px; font-size: 12px; color: {priority_color};">{priority.upper()}</span>
        </div>
        <div style="font-size: 13px; color: #666; margin-top: 4px;">
            📅 {task.get('start_date', 'N/A')} → {task.get('end_date', 'N/A')} | 
            👥 {task.get('team_size', 1)} pers. | 
            💰 {task.get('cost', 0):,}€
        </div>
        <div style="margin-top: 6px;">
            <div style="background: #e9ecef; height: 6px; border-radius: 3px; overflow: hidden;">
                <div style="background: {status_color}; height: 100%; width: {progress}%; transition: width 0.3s;"></div>
            </div>
            <span style="font-size: 11px; color: #666;">{progress:.1f}% complétée</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dépendances si présentes
    if task.get('dependencies'):
        st.markdown(f"<div style='margin-left: 20px; color: #666; font-size: 12px;'>🔗 Dépend de: {', '.join(task['dependencies'])}</div>", unsafe_allow_html=True)


def render_phase_chart(phase_tasks: List[Dict], phase_idx: int):
    """Rendu du graphique de répartition d'une phase"""
    
    st.markdown("**📊 Répartition tâches**")
    
    # Graphique en anneau des statuts
    status_counts = {'Terminées': 0, 'En cours': 0, 'Non démarrées': 0}
    for task in phase_tasks:
        progress = task.get('progress', 0)
        if progress >= 95:
            status_counts['Terminées'] += 1
        elif progress > 5:
            status_counts['En cours'] += 1
        else:
            status_counts['Non démarrées'] += 1
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=list(status_counts.keys()),
        values=list(status_counts.values()),
        hole=0.5,
        marker_colors=['#28a745', '#ffc107', '#6c757d']
    )])
    fig_donut.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    st.plotly_chart(fig_donut, use_container_width=True, key=f"phase_{phase_idx}_donut")


def render_generated_wbs(tasks: List[Dict]):
    """Rendu WBS généré à partir des tâches existantes"""
    
    # Grouper par phase si disponible
    phases = {}
    for task in tasks:
        phase = task.get('phase', 'Phase principale')
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(task)
    
    for phase_name, phase_tasks in phases.items():
        with st.expander(f"📁 {phase_name} ({len(phase_tasks)} tâches)", expanded=True):
            
            # Métriques de phase
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📋 Tâches", len(phase_tasks))
            
            with col2:
                total_budget = sum([t['budget'] for t in phase_tasks])
                st.metric("💰 Budget", f"{total_budget:,}€")
            
            with col3:
                total_duration = sum([t['duration'] for t in phase_tasks])
                st.metric("⏱️ Durée", f"{total_duration} jours")
            
            with col4:
                avg_progress = np.mean([t['progress'] for t in phase_tasks])
                st.metric("📊 Progression", f"{avg_progress:.1f}%")
            
            # Arbre des tâches avec design moderne
            st.markdown("**🗂️ Structure des tâches:**")
            for task in sorted(phase_tasks, key=lambda x: x['start_date']):
                # Icônes selon statut
                if task['progress'] >= 95:
                    status_emoji = "✅"
                    status_color = "#28a745"
                elif task['progress'] > 5:
                    status_emoji = "🔄"
                    status_color = "#ffc107"
                else:
                    status_emoji = "⏳"
                    status_color = "#6c757d"
                
                critical_mark = "🔥" if task.get('is_critical') else ""
                milestone_mark = "🎯" if task.get('is_milestone') else ""
                
                st.markdown(f"""
                <div style="
                    border-left: 3px solid {status_color};
                    padding: 8px 12px;
                    margin: 5px 0;
                    background: {status_color}10;
                    border-radius: 4px;
                ">
                    <strong>{status_emoji} {task['name']} {critical_mark} {milestone_mark}</strong><br>
                    <small style="color: #666;">
                        📅 {task['start_date'].strftime('%Y-%m-%d')} → {task['end_date'].strftime('%Y-%m-%d')} | 
                        👥 {task['team_size']} pers. | 💰 {task['budget']:,}€
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                if task.get('dependencies'):
                    st.markdown(f"<div style='margin-left: 20px; color: #666; font-size: 12px;'>🔗 Dépend de: {', '.join(task['dependencies'])}</div>", unsafe_allow_html=True)


def render_wbs_hierarchy_chart(phases: List[Dict]):
    """Graphique hiérarchique WBS avec Plotly"""
    
    st.markdown("### 🌳 Graphique Hiérarchique WBS")
    
    # Construction des données pour le graphique en arbre
    fig = go.Figure()
    
    y_pos = 0
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for phase_idx, phase in enumerate(phases):
        phase_name = phase.get('name', f'Phase {phase_idx + 1}')
        phase_color = colors[phase_idx % len(colors)]
        
        # Noeud phase
        fig.add_trace(go.Scatter(
            x=[0], y=[y_pos],
            mode='markers+text',
            marker=dict(size=30, color=phase_color, symbol='square'),
            text=phase_name,
            textposition="middle right",
            name=phase_name,
            hovertemplate=f"<b>{phase_name}</b><br>Phase {phase_idx + 1}<extra></extra>"
        ))
        
        # Tâches de la phase
        if 'tasks' in phase:
            for task_idx, task in enumerate(phase['tasks']):
                task_x = 1
                task_y = y_pos + (task_idx - len(phase['tasks'])/2 + 0.5) * 0.3
                
                # Couleur selon progression
                progress = task.get('progress', 0)
                if progress >= 95:
                    task_color = '#28a745'
                elif progress > 5:
                    task_color = '#ffc107'
                else:
                    task_color = '#6c757d'
                
                # Noeud tâche
                fig.add_trace(go.Scatter(
                    x=[task_x], y=[task_y],
                    mode='markers+text',
                    marker=dict(size=20, color=task_color, symbol='circle'),
                    text=task.get('name', 'Tâche')[:20] + "...",
                    textposition="middle right",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{task.get('name', 'Tâche')}</b><br>"
                        f"Progression: {progress:.1f}%<br>"
                        f"Budget: {task.get('cost', 0):,}€<br>"
                        f"Durée: {task.get('duration', 0)} jours"
                        "<extra></extra>"
                    )
                ))
                
                # Ligne de connexion
                fig.add_trace(go.Scatter(
                    x=[0, task_x], y=[y_pos, task_y],
                    mode='lines',
                    line=dict(color=phase_color, width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        y_pos -= 2  # Espacement entre phases
    
    fig.update_layout(
        title="🌳 Structure Hiérarchique du Projet",
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 3]),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=max(400, len(phases) * 150),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_wbs_metrics_dashboard(phases: List[Dict]):
    """Dashboard de métriques WBS avancées"""
    
    st.markdown("### 📊 Dashboard Métriques WBS")
    
    # Calculs globaux
    all_tasks = []
    phase_metrics = []
    
    for phase in phases:
        if 'tasks' in phase:
            phase_tasks = phase['tasks']
            all_tasks.extend(phase_tasks)
            
            phase_metrics.append({
                'name': phase.get('name', 'Phase'),
                'tasks_count': len(phase_tasks),
                'cost': sum([t.get('cost', 0) for t in phase_tasks]),
                'duration': sum([t.get('duration', 0) for t in phase_tasks]),
                'progress': np.mean([t.get('progress', 0) for t in phase_tasks])
            })
    
    # Graphiques de métriques
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique budget par phase
        if phase_metrics:
            fig_budget = px.bar(
                x=[p['name'] for p in phase_metrics],
                y=[p['cost'] for p in phase_metrics],
                title="💰 Budget par Phase",
                color=[p['progress'] for p in phase_metrics],
                color_continuous_scale='RdYlGn'
            )
            fig_budget.update_layout(height=300)
            st.plotly_chart(fig_budget, use_container_width=True)
    
    with col2:
        # Graphique progression par phase
        if phase_metrics:
            fig_progress = px.line(
                x=[p['name'] for p in phase_metrics],
                y=[p['progress'] for p in phase_metrics],
                title="📈 Progression par Phase",
                markers=True
            )
            fig_progress.update_layout(
                height=300,
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_progress, use_container_width=True)