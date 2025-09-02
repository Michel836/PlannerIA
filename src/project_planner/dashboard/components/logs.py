"""
Module Logs & History - Restructuré selon le modèle Reporting/Qualité/Risques
Historique des exécutions, logs système et analytics d'usage
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io


def render_logs_section(plan_data: Dict[str, Any], current_run_id: str):
    """Dashboard Logs & History restructuré - Sections visuels puis onglets détaillés"""
    
    # Header moderne
    st.header("📋 Logs & History Dashboard")
    st.markdown("*Historique des exécutions, monitoring système et analytics d'usage*")
    
    # Charger les données de logs
    runs_data = load_all_runs()
    system_logs = generate_system_logs()
    usage_analytics = generate_usage_analytics(runs_data)
    
    # Calcul des KPIs et insights
    logs_kpis = calculate_logs_kpis(runs_data, system_logs, usage_analytics)
    logs_insights = generate_logs_insights(runs_data, system_logs)
    
    # === SECTION 1: MÉTRIQUES RÉSUMÉES ===
    render_summary_metrics(logs_kpis)
    
    # === SECTION 2: GRAPHIQUES VISUELS PRINCIPAUX ===
    st.markdown("---")
    render_main_visualizations(runs_data, usage_analytics, logs_kpis)
    
    # === SECTION 3: DETAILED ANALYSIS (avec onglets) ===
    st.markdown("---")
    render_detailed_logs_analysis(runs_data, system_logs, usage_analytics, logs_insights, current_run_id)
    
    # === SECTION 4: EXPORTS EN BAS DE PAGE ===
    st.markdown("---")
    render_export_section(runs_data, system_logs, usage_analytics, logs_insights)


# Note: Les fonctions originales ont été remplacées par les nouvelles dans la structure 4-sections
# Si vous voyez des erreurs, c'est que l'ancienne structure est encore appelée quelque part


def load_all_runs() -> List[Dict[str, Any]]:
    """Load all available project runs"""
    # Mock data for demonstration with unique IDs
    import uuid
    
    base_runs = [
        {
            'name': 'Application E-commerce avec Paiement',
            'status': 'completed',
            'duration': 28,
            'cost': 32000,
            'task_count': 45,
            'created_at': '2024-01-15',
            'completed_at': '2024-02-12'
        },
        {
            'name': 'Application E-commerce avec Paiement',
            'status': 'completed',
            'duration': 22,
            'cost': 18500,
            'task_count': 38,
            'created_at': '2024-01-10',
            'completed_at': '2024-02-01'
        },
        {
            'name': 'Application E-commerce',
            'status': 'failed',
            'duration': 15,
            'cost': 12000,
            'task_count': 28,
            'created_at': '2024-01-05',
            'completed_at': '2024-01-20'
        },
        {
            'name': 'Unknown Project',
            'status': 'in_progress',
            'duration': 0,
            'cost': 5000,
            'task_count': 12,
            'created_at': '2024-02-01',
            'completed_at': None
        }
    ]
    
    # Create unique runs with different IDs
    unique_runs = []
    for i, base_run in enumerate(base_runs):
        for j in range(4):  # 4 variants of each base run
            run = base_run.copy()
            run['id'] = str(uuid.uuid4())[:8] + f"_{i}_{j}"
            unique_runs.append(run)
    
    return unique_runs


def calculate_logs_kpis(runs_data: List[Dict], system_logs: List[Dict], usage_analytics: Dict) -> Dict[str, Any]:
    """Calcule les KPIs avancés pour les logs"""
    if not runs_data:
        return {}
    
    # Statistiques des runs
    total_runs = len(runs_data)
    completed_runs = sum(1 for run in runs_data if run.get('status') == 'completed')
    failed_runs = sum(1 for run in runs_data if run.get('status') == 'failed')
    success_rate = (completed_runs / total_runs * 100) if total_runs > 0 else 0
    
    avg_duration = sum(run.get('duration', 0) for run in runs_data) / total_runs if total_runs > 0 else 0
    avg_cost = sum(run.get('cost', 0) for run in runs_data) / total_runs if total_runs > 0 else 0
    
    # Statistiques système
    system_health = 95 - (len([log for log in system_logs if log.get('level') == 'ERROR']) * 5)
    error_count = len([log for log in system_logs if log.get('level') == 'ERROR'])
    warning_count = len([log for log in system_logs if log.get('level') == 'WARNING'])
    
    # Score de santé global
    health_factors = [
        success_rate,                    # Taux de succès des runs
        min(100, system_health),         # Santé système
        min(100, 100 - error_count * 2), # Pénalité erreurs
        usage_analytics.get('uptime', 95)  # Temps de fonctionnement
    ]
    global_health_score = np.mean(health_factors)
    
    return {
        'total_runs': total_runs,
        'completed_runs': completed_runs,
        'failed_runs': failed_runs,
        'success_rate': success_rate,
        'avg_duration': avg_duration,
        'avg_cost': avg_cost,
        'system_health': system_health,
        'error_count': error_count,
        'warning_count': warning_count,
        'global_health_score': global_health_score,
        'uptime': usage_analytics.get('uptime', 95.5),
        'total_users': usage_analytics.get('total_users', 24),
        'avg_session_duration': usage_analytics.get('avg_session_duration', 18.5)
    }


def generate_logs_insights(runs_data: List[Dict], system_logs: List[Dict]) -> Dict:
    """Génère des insights IA pour les logs"""
    return {
        'performance_trends': {"trend": "improving", "confidence": 0.88},
        'error_patterns': {"most_common": "timeout_errors", "frequency": 12, "confidence": 0.91},
        'usage_insights': ["Peak usage at 14h-16h", "Success rate increased 8% this week"],
        'recommendations': ["Monitor timeout patterns", "Scale during peak hours", "Archive old logs"]
    }


def generate_system_logs() -> List[Dict]:
    """Génère des logs système simulés"""
    logs = []
    for i in range(50):
        level = np.random.choice(['INFO', 'WARNING', 'ERROR', 'DEBUG'], p=[0.6, 0.2, 0.1, 0.1])
        logs.append({
            'timestamp': datetime.now() - timedelta(hours=i),
            'level': level,
            'message': f"System message {i+1}",
            'module': np.random.choice(['api', 'dashboard', 'ml', 'core']),
            'user_id': np.random.choice(['user_001', 'user_002', 'user_003', 'system'])
        })
    return logs


def generate_usage_analytics(runs_data: List[Dict]) -> Dict:
    """Génère des analytics d'usage"""
    return {
        'uptime': 95.8,
        'total_users': 24,
        'active_users_today': 8,
        'avg_session_duration': 18.5,
        'peak_hours': [14, 15, 16],
        'most_used_features': ['planning', 'risk_analysis', 'reporting'],
        'geographic_distribution': {'EU': 65, 'NA': 25, 'ASIA': 10}
    }


def render_summary_metrics(kpis: Dict[str, Any]):
    """Métriques résumées en haut du dashboard"""
    if not kpis:
        return
    
    # Calculs pour les indicateurs de santé
    health_status = "✅ Excellent" if kpis['global_health_score'] >= 90 else "⚠️ Bon" if kpis['global_health_score'] >= 75 else "🚨 Critique"
    success_status = "✅ Optimal" if kpis['success_rate'] >= 90 else "⚠️ Correct" if kpis['success_rate'] >= 70 else "🚨 Problème"
    uptime_status = "✅ Stable" if kpis['uptime'] >= 99 else "⚠️ Acceptable" if kpis['uptime'] >= 95 else "🚨 Instable"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Santé Globale", f"{kpis['global_health_score']:.1f}%", health_status)
    
    with col2:
        st.metric("✅ Taux Succès", f"{kpis['success_rate']:.1f}%", success_status)
    
    with col3:
        st.metric("🔄 Uptime", f"{kpis['uptime']:.1f}%", uptime_status)
    
    with col4:
        st.metric("👥 Utilisateurs", kpis['total_users'], f"📊 {kpis.get('active_users_today', 8)} actifs")
    
    with col5:
        st.metric("🚫 Erreurs", kpis['error_count'], f"{'🔴' if kpis['error_count'] > 5 else '🟢'}")


def render_main_visualizations(runs_data: List[Dict], usage_analytics: Dict, kpis: Dict[str, Any]):
    """Graphiques visuels principaux"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Timeline des exécutions
        st.subheader("📈 Timeline des Exécutions")
        
        df_runs = pd.DataFrame(runs_data)
        df_runs['created_at'] = pd.to_datetime(df_runs['created_at'])
        df_runs['success'] = df_runs['status'].map({'completed': 1, 'failed': 0, 'in_progress': 0.5})
        
        # Graphique en ligne du taux de succès dans le temps
        daily_stats = df_runs.groupby(df_runs['created_at'].dt.date).agg({
            'success': 'mean',
            'duration': 'mean'
        }).reset_index()
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=daily_stats['created_at'],
            y=daily_stats['success'] * 100,
            mode='lines+markers',
            name='Taux de Succès (%)',
            line=dict(color='#10B981', width=3),
            marker=dict(size=8)
        ))
        
        fig_timeline.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Date",
            yaxis_title="Taux de Succès (%)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        # Distribution des durées
        st.subheader("⏱️ Distribution des Durées")
        
        fig_duration = px.histogram(
            df_runs,
            x='duration',
            nbins=15,
            title="Distribution des Durées d'Exécution",
            color='status',
            color_discrete_map={
                'completed': '#10B981',
                'failed': '#EF4444',
                'in_progress': '#F59E0B'
            }
        )
        
        fig_duration.update_layout(
            height=300,
            showlegend=True,
            xaxis_title="Durée (jours)",
            yaxis_title="Nombre d'Exécutions",
            template="plotly_white"
        )
        
        st.plotly_chart(fig_duration, use_container_width=True)


def render_detailed_logs_analysis(runs_data: List[Dict], system_logs: List[Dict], 
                                usage_analytics: Dict, insights: Dict, current_run_id: str):
    """Analyse détaillée avec onglets (contenu original préservé)"""
    
    st.subheader("🔍 Analyse Détaillée")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏃 Run History", 
        "📊 System Logs", 
        "💬 User Feedback", 
        "📈 Analytics",
        "🤖 Insights IA"
    ])
    
    with tab1:
        render_run_history_detailed(runs_data, current_run_id)
    
    with tab2:
        render_system_logs_detailed(system_logs)
    
    with tab3:
        render_user_feedback_detailed()
    
    with tab4:
        render_usage_analytics_detailed(usage_analytics)
    
    with tab5:
        render_ai_insights(insights, runs_data)


def render_export_section(runs_data: List[Dict], system_logs: List[Dict], 
                         usage_analytics: Dict, insights: Dict):
    """Section d'export en bas de page"""
    
    st.subheader("📤 Exports")
    st.markdown("*Exportez les données de logs et historiques*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Excel", help="Exporter toutes les données en Excel"):
            excel_data = create_excel_export(runs_data, system_logs, usage_analytics)
            st.download_button(
                "💾 Télécharger Excel",
                excel_data,
                "logs_history_export.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📄 Export PDF", help="Rapport PDF complet"):
            st.info("📄 Génération du rapport PDF...")
    
    with col3:
        if st.button("📋 Export CSV", help="Données tabulaires CSV"):
            csv_data = pd.DataFrame(runs_data).to_csv(index=False)
            st.download_button(
                "💾 Télécharger CSV",
                csv_data,
                "runs_history.csv",
                "text/csv"
            )
    
    with col4:
        if st.button("🔧 Export JSON", help="Données complètes JSON"):
            json_data = json.dumps({
                'runs_data': runs_data,
                'system_logs': system_logs[:20],  # Limiter pour taille
                'usage_analytics': usage_analytics,
                'insights': insights
            }, indent=2, default=str)
            st.download_button(
                "💾 Télécharger JSON",
                json_data,
                "logs_complete_export.json",
                "application/json"
            )


def create_excel_export(runs_data: List[Dict], system_logs: List[Dict], usage_analytics: Dict) -> bytes:
    """Crée un export Excel multi-feuilles"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Feuille Runs
        df_runs = pd.DataFrame(runs_data)
        df_runs.to_excel(writer, sheet_name='Runs History', index=False)
        
        # Feuille System Logs (limitée)
        df_logs = pd.DataFrame(system_logs[:100])
        df_logs.to_excel(writer, sheet_name='System Logs', index=False)
        
        # Feuille Analytics
        df_analytics = pd.DataFrame([usage_analytics])
        df_analytics.to_excel(writer, sheet_name='Usage Analytics', index=False)
    
    return output.getvalue()


def render_run_history_detailed(runs_data: List[Dict], current_run_id: str):
    """Historique détaillé des exécutions (contenu original préservé et enrichi)"""
    st.markdown("### 🏃 Project Run History")
    
    if not runs_data:
        st.info("No project runs found")
        return
    
    # Contrôles de filtrage
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("🔍 **Search runs**")
        search_query = st.text_input("Search runs", placeholder="Search by title or brief...", label_visibility="collapsed")
    
    with col2:
        st.markdown("📅 **Time Period**")
        time_filter = st.selectbox("Time Period", ["All Time", "Last 30 days", "Last 7 days", "Today"], label_visibility="collapsed")
    
    with col3:
        st.markdown("📊 **Status**")
        status_filter = st.selectbox("Status", ["All", "Completed", "Failed", "In Progress"], label_visibility="collapsed")
    
    # Statistiques détaillées
    st.markdown("---")
    st.markdown("### 📊 Statistiques Détaillées")
    
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    
    total_runs = len(runs_data)
    completed_runs = sum(1 for run in runs_data if run.get('status') == 'completed')
    failed_runs = sum(1 for run in runs_data if run.get('status') == 'failed')
    avg_duration = sum(run.get('duration', 0) for run in runs_data) / total_runs if total_runs > 0 else 0
    avg_cost = sum(run.get('cost', 0) for run in runs_data) / total_runs if total_runs > 0 else 0
    
    with stat_col1:
        st.metric("**Total Runs**", total_runs)
    
    with stat_col2:
        st.metric("**Completed**", completed_runs)
    
    with stat_col3:
        st.metric("**Failed**", failed_runs)
    
    with stat_col4:
        st.metric("**Avg Duration**", f"{avg_duration:.1f} days")
    
    with stat_col5:
        st.metric("**Avg Cost**", f"${avg_cost:,.0f}")
    
    # Affichage détaillé des runs
    st.markdown("---")
    st.markdown("### 📋 Run Details")
    
    filtered_runs = filter_runs(runs_data, search_query, time_filter, status_filter)
    
    for run in filtered_runs:
        with st.expander(f"📁 {run.get('name', 'Unknown Project')} - {run.get('id', 'N/A')[:8]}..."):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {run.get('status', 'Unknown')}")
                st.write(f"**Duration:** {run.get('duration', 0)} days")
            
            with col2:
                st.write(f"**Cost:** ${run.get('cost', 0):,}")
                st.write(f"**Tasks:** {run.get('task_count', 0)}")
            
            with col3:
                st.write(f"**Created:** {run.get('created_at', 'Unknown')}")
                st.write(f"**Completed:** {run.get('completed_at', 'N/A')}")
            
            if st.button(f"View Details", key=f"view_{run.get('id')}"):
                st.session_state.selected_run_id = run.get('id')
                st.success("✅ Détails sélectionnés")


def render_system_logs_detailed(system_logs: List[Dict]):
    """Logs système détaillés"""
    st.markdown("### 📊 System Logs")
    
    if not system_logs:
        st.info("No system logs available")
        return
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        level_filter = st.selectbox("Log Level", ["All", "ERROR", "WARNING", "INFO", "DEBUG"])
    with col2:
        module_filter = st.selectbox("Module", ["All", "api", "dashboard", "ml", "core"])
    
    # Filtrer logs
    filtered_logs = system_logs
    if level_filter != "All":
        filtered_logs = [log for log in filtered_logs if log.get('level') == level_filter]
    if module_filter != "All":
        filtered_logs = [log for log in filtered_logs if log.get('module') == module_filter]
    
    # Affichage des logs
    for log in filtered_logs[:20]:  # Limiter affichage
        level_color = {
            'ERROR': '🔴', 'WARNING': '🟡', 'INFO': '🔵', 'DEBUG': '⚪'
        }.get(log.get('level'), '⚫')
        
        with st.expander(f"{level_color} {log.get('level')} - {log.get('module')} - {log.get('timestamp')}"):
            st.code(log.get('message'))


def render_user_feedback_detailed():
    """Feedback utilisateur détaillé"""
    st.markdown("### 💬 User Feedback")
    st.info("📝 Module de feedback utilisateur à implémenter")
    
    # Placeholder pour feedback
    with st.expander("💡 Fonctionnalités prévues"):
        st.markdown("""
        - **Collecte de feedback** après chaque exécution
        - **Ratings** et commentaires utilisateurs
        - **Analyse sentiment** des retours
        - **Suggestions d'amélioration** automatiques
        """)


def render_usage_analytics_detailed(usage_analytics: Dict):
    """Analytics d'usage détaillés"""
    st.markdown("### 📈 Usage Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Utilisateurs")
        st.metric("Total Users", usage_analytics.get('total_users', 24))
        st.metric("Active Today", usage_analytics.get('active_users_today', 8))
        st.metric("Avg Session", f"{usage_analytics.get('avg_session_duration', 18.5)} min")
    
    with col2:
        st.markdown("#### 🌍 Répartition Géographique")
        geo_data = usage_analytics.get('geographic_distribution', {})
        for region, percentage in geo_data.items():
            st.metric(region, f"{percentage}%")


def render_ai_insights(insights: Dict, runs_data: List[Dict]):
    """Insights IA pour les logs"""
    st.markdown("### 🤖 Insights IA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Tendances Performance")
        trend_data = insights.get('performance_trends', {})
        st.write(f"**Tendance:** {trend_data.get('trend', 'stable')}")
        st.write(f"**Confiance:** {trend_data.get('confidence', 0.88)*100:.0f}%")
        
        st.markdown("#### ⚠️ Patterns d'Erreurs")
        error_data = insights.get('error_patterns', {})
        st.write(f"**Type fréquent:** {error_data.get('most_common', 'timeout_errors')}")
        st.write(f"**Fréquence:** {error_data.get('frequency', 12)} occurrences")
    
    with col2:
        st.markdown("#### 💡 Recommandations")
        recommendations = insights.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        st.markdown("#### 🔮 Insights Usage")
        usage_insights = insights.get('usage_insights', [])
        for insight in usage_insights:
            st.write(f"• {insight}")


def filter_runs(runs: List[Dict], search: str, time_filter: str, status_filter: str) -> List[Dict]:
    """Filter runs based on search criteria"""
    filtered = runs
    
    if search:
        filtered = [run for run in filtered if search.lower() in run.get('name', '').lower()]
    
    if status_filter != "All":
        filtered = [run for run in filtered if run.get('status') == status_filter.lower()]
    
    return filtered[:10]  # Limit to first 10 for display