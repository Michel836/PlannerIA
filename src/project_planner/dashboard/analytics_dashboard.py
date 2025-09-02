"""
Advanced Analytics Dashboard Component for PlannerIA
Interface de tableaux de bord business intelligence
"""

import streamlit as st
import plotly.graph_objects as go
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

def display_analytics_dashboard(projects: List[Dict[str, Any]]):
    """Afficher le tableau de bord analytics avancé"""
    
    if not projects:
        st.info("📊 Générez plusieurs projets pour accéder aux analytics avancées")
        return
    
    # Import ici pour éviter les problèmes de circularité
    try:
        from src.project_planner.analytics.business_intelligence import analyze_portfolio
        analytics_available = True
    except ImportError:
        analytics_available = False
        st.error("Module d'analytics avancé non disponible")
        return
    
    st.markdown("## 📊 Business Intelligence Dashboard")
    st.markdown("---")
    
    # Analyser le portfolio
    with st.spinner("🧠 Analyse du portfolio en cours..."):
        analysis = analyze_portfolio(projects)
    
    if "error" in analysis:
        st.error(f"Erreur d'analyse: {analysis['error']}")
        return
    
    # Résumé du portfolio
    display_portfolio_summary(analysis.get("portfolio_summary", {}))
    
    # KPIs principaux
    display_kpi_section(analysis.get("kpis", []))
    
    # Visualisations interactives
    display_interactive_charts(analysis.get("charts", {}))
    
    # Insights automatiques
    display_business_insights(analysis.get("insights", []))
    
    # Matrice Risque/Valeur
    display_risk_value_matrix(analysis.get("risk_value_matrix", {}))
    
    # Prédictions et recommandations
    col1, col2 = st.columns(2)
    with col1:
        display_predictions(analysis.get("predictions", {}))
    with col2:
        display_recommendations(analysis.get("recommendations", []))

def display_portfolio_summary(summary: Dict[str, Any]):
    """Afficher le résumé du portfolio"""
    
    st.markdown("### 🎯 Résumé Portfolio")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_projects = summary.get("total_projects", 0)
        st.metric(
            label="Projets Totaux",
            value=total_projects,
            help="Nombre total de projets dans le portfolio"
        )
    
    with col2:
        total_budget = summary.get("total_budget", 0)
        st.metric(
            label="Budget Global",
            value=f"€{total_budget:,.0f}",
            help="Investissement total du portfolio"
        )
    
    with col3:
        avg_duration = summary.get("avg_duration", 0)
        st.metric(
            label="Durée Moyenne",
            value=f"{avg_duration:.0f} jours",
            help="Durée moyenne des projets"
        )
    
    with col4:
        success_prob = summary.get("success_probability", 0)
        delta_color = "normal" if success_prob >= 70 else "inverse"
        st.metric(
            label="Taux de Succès",
            value=f"{success_prob:.0f}%",
            delta=f"{'✅' if success_prob >= 70 else '⚠️'} {'Excellent' if success_prob >= 80 else 'Bon' if success_prob >= 70 else 'À améliorer'}",
            help="Probabilité de succès estimée du portfolio"
        )

def display_kpi_section(kpis: List[Dict[str, Any]]):
    """Afficher la section KPIs avec code couleur"""
    
    st.markdown("### 📈 Indicateurs Clés de Performance")
    
    if not kpis:
        st.warning("Aucun KPI disponible")
        return
    
    # Organiser les KPIs par catégorie
    kpi_categories = {}
    for kpi in kpis:
        category = kpi.get("category", "other")
        if category not in kpi_categories:
            kpi_categories[category] = []
        kpi_categories[category].append(kpi)
    
    # Créer des onglets par catégorie
    category_names = list(kpi_categories.keys())
    if category_names:
        tabs = st.tabs([f"📊 {cat.value.replace('_', ' ').title()}" for cat in category_names])
        
        for tab, category in zip(tabs, category_names):
            with tab:
                display_category_kpis(kpi_categories[category])

def display_category_kpis(category_kpis: List[Dict[str, Any]]):
    """Afficher les KPIs d'une catégorie"""
    
    for i in range(0, len(category_kpis), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(category_kpis):
                display_single_kpi(category_kpis[i])
        
        with col2:
            if i + 1 < len(category_kpis):
                display_single_kpi(category_kpis[i + 1])

def display_single_kpi(kpi: Dict[str, Any]):
    """Afficher un KPI individuel avec visualisation"""
    
    name = kpi.get("name", "KPI")
    value = kpi.get("value", 0)
    target = kpi.get("target", 0)
    unit = kpi.get("unit", "")
    trend = kpi.get("trend", "stable")
    change_percent = kpi.get("change_percent", 0)
    description = kpi.get("description", "")
    recommendation = kpi.get("recommendation", "")
    
    # Déterminer la couleur basée sur la performance
    if target > 0:
        performance_ratio = value / target
        if performance_ratio >= 1.0:
            color = "success"
            status_icon = "✅"
        elif performance_ratio >= 0.8:
            color = "info"
            status_icon = "📊"
        elif performance_ratio >= 0.6:
            color = "warning"
            status_icon = "⚠️"
        else:
            color = "error"
            status_icon = "🚨"
    else:
        color = "info"
        status_icon = "📊"
    
    # Afficher dans une carte stylisée
    with st.container():
        st.markdown(f"""
        <div style="
            background: #ffffff; 
            padding: 1.5rem; 
            border-radius: 16px; 
            margin: 1rem 0;
            border: 2px solid #e2e8f0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        ">
            <h4 style="color: #1e293b; margin-bottom: 0.5rem; font-size: 1.2rem;">
                {status_icon} {name}
            </h4>
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 1rem 0;">
                <div style="font-size: 2rem; font-weight: 700; color: #1e293b;">
                    {value:.1f}{unit}
                </div>
                <div style="text-align: right;">
                    <div style="color: #64748b; font-size: 0.9rem;">Objectif: {target:.1f}{unit}</div>
                    <div style="color: {'#10b981' if change_percent >= 0 else '#ef4444'}; font-size: 0.9rem; font-weight: 600;">
                        {'+' if change_percent >= 0 else ''}{change_percent:.1f}%
                    </div>
                </div>
            </div>
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">
                {description}
            </div>
            <div style="background: #f1f5f9; padding: 0.75rem; border-radius: 8px; font-size: 0.9rem; color: #334155;">
                💡 {recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphique de progression
        progress_value = min(100, (value / target * 100)) if target > 0 else 50
        st.progress(progress_value / 100)

def display_interactive_charts(charts: Dict[str, Any]):
    """Afficher les graphiques interactifs"""
    
    st.markdown("### 📊 Visualisations Interactives")
    
    if not charts:
        st.warning("Aucune visualisation disponible")
        return
    
    # Créer des onglets pour les différents graphiques
    chart_tabs = st.tabs(["🎯 KPIs Radar", "💰 Budget/Durée", "📈 Évolution Temporelle"])
    
    with chart_tabs[0]:
        if "kpi_radar" in charts:
            st.markdown("#### Performance Portfolio vs Objectifs")
            fig_data = json.loads(charts["kpi_radar"])
            st.plotly_chart(fig_data, use_container_width=True, key="radar_chart")
        else:
            st.info("Graphique radar non disponible")
    
    with chart_tabs[1]:
        if "budget_duration_scatter" in charts:
            st.markdown("#### Distribution Budget/Durée des Projets")
            fig_data = json.loads(charts["budget_duration_scatter"])
            st.plotly_chart(fig_data, use_container_width=True, key="scatter_chart")
        else:
            st.info("Graphique de dispersion non disponible")
    
    with chart_tabs[2]:
        if "kpi_timeline" in charts:
            st.markdown("#### Évolution des KPIs sur 30 jours")
            fig_data = json.loads(charts["kpi_timeline"])
            st.plotly_chart(fig_data, use_container_width=True, key="timeline_chart")
        else:
            st.info("Graphique temporel non disponible")

def display_business_insights(insights: List[Dict[str, Any]]):
    """Afficher les insights business automatiques"""
    
    st.markdown("### 🧠 Insights Automatiques")
    
    if not insights:
        st.info("Aucun insight disponible")
        return
    
    for insight in insights:
        title = insight.get("title", "Insight")
        description = insight.get("description", "")
        impact = insight.get("impact", "medium")
        category = insight.get("category", "")
        confidence = insight.get("confidence", 0)
        actionable_steps = insight.get("actionable_steps", [])
        timestamp = insight.get("timestamp", "")
        
        # Déterminer la couleur selon l'impact
        if impact == "high":
            border_color = "#ef4444"
            bg_color = "#fef2f2"
            icon = "🔴"
        elif impact == "medium":
            border_color = "#f59e0b"
            bg_color = "#fefbf3"
            icon = "🟡"
        else:
            border_color = "#3b82f6"
            bg_color = "#eff6ff"
            icon = "🔵"
        
        with st.expander(f"{icon} {title} (Impact: {impact.title()})", expanded=impact == "high"):
            st.markdown(f"""
            <div style="
                background: {bg_color}; 
                padding: 1.5rem; 
                border-radius: 12px; 
                border-left: 4px solid {border_color};
                margin: 1rem 0;
            ">
                <p style="color: #374151; margin-bottom: 1rem; font-size: 1rem;">
                    {description}
                </p>
                <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 1rem;">
                    📊 Catégorie: {category.value.replace('_', ' ').title()} | 🎯 Confiance: {confidence:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if actionable_steps:
                st.markdown("**🎯 Actions Recommandées:**")
                for i, step in enumerate(actionable_steps, 1):
                    st.markdown(f"{i}. {step}")

def display_risk_value_matrix(matrix_data: Dict[str, Any]):
    """Afficher la matrice risque/valeur"""
    
    st.markdown("### 🎯 Matrice Risque/Valeur")
    
    if not matrix_data:
        st.warning("Matrice non disponible")
        return
    
    # Afficher le graphique
    if "chart" in matrix_data:
        fig_data = json.loads(matrix_data["chart"])
        st.plotly_chart(fig_data, use_container_width=True, key="risk_value_matrix")
    
    # Analyse par quadrant
    if "quadrant_analysis" in matrix_data:
        st.markdown("#### 📊 Analyse par Quadrant")
        
        quadrant_analysis = matrix_data["quadrant_analysis"]
        
        quad_col1, quad_col2 = st.columns(2)
        
        with quad_col1:
            # Perles et Paris
            pearls = quadrant_analysis.get("pearls", {})
            gambles = quadrant_analysis.get("gambles", {})
            
            st.markdown("##### 🟢 Projets Perles (Faible risque, Haute valeur)")
            st.metric("Nombre", pearls.get("count", 0), 
                     delta=f"{pearls.get('percentage', 0):.1f}% du portfolio")
            st.info(f"💡 {pearls.get('recommendation', '')}")
            
            st.markdown("##### 🟡 Projets Paris (Haut risque, Haute valeur)")
            st.metric("Nombre", gambles.get("count", 0),
                     delta=f"{gambles.get('percentage', 0):.1f}% du portfolio")
            st.warning(f"⚠️ {gambles.get('recommendation', '')}")
        
        with quad_col2:
            # Routine et À éviter
            routine = quadrant_analysis.get("routine", {})
            avoid = quadrant_analysis.get("avoid", {})
            
            st.markdown("##### 🔵 Projets Routine (Faible risque, Faible valeur)")
            st.metric("Nombre", routine.get("count", 0),
                     delta=f"{routine.get('percentage', 0):.1f}% du portfolio")
            st.info(f"📋 {routine.get('recommendation', '')}")
            
            st.markdown("##### 🔴 Projets À Éviter (Haut risque, Faible valeur)")
            st.metric("Nombre", avoid.get("count", 0),
                     delta=f"{avoid.get('percentage', 0):.1f}% du portfolio")
            if avoid.get("count", 0) > 0:
                st.error(f"🚨 {avoid.get('recommendation', '')}")
            else:
                st.success("✅ Aucun projet à éviter détecté")

def display_predictions(predictions: Dict[str, Any]):
    """Afficher les prédictions business"""
    
    st.markdown("### 🔮 Prédictions Business")
    
    if not predictions:
        st.info("Aucune prédiction disponible")
        return
    
    # Prédiction budget
    if "budget_forecast" in predictions:
        budget_pred = predictions["budget_forecast"]
        
        st.markdown("#### 💰 Évolution Budgétaire")
        
        current = budget_pred.get("current_avg", 0)
        pred_3m = budget_pred.get("predicted_3m", 0)
        pred_6m = budget_pred.get("predicted_6m", 0)
        
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            st.metric("Actuel", f"€{current:,.0f}")
        
        with pred_col2:
            change_3m = ((pred_3m - current) / current * 100) if current > 0 else 0
            st.metric("Dans 3 mois", f"€{pred_3m:,.0f}", 
                     delta=f"{change_3m:+.1f}%")
        
        with pred_col3:
            change_6m = ((pred_6m - current) / current * 100) if current > 0 else 0
            st.metric("Dans 6 mois", f"€{pred_6m:,.0f}",
                     delta=f"{change_6m:+.1f}%")
        
        confidence = budget_pred.get("confidence", 0)
        st.progress(confidence, text=f"Confiance: {confidence:.0%}")
    
    # Prédiction succès
    if "success_probability" in predictions:
        success_pred = predictions["success_probability"]
        
        st.markdown("#### 🎯 Probabilité de Succès")
        
        portfolio_success = success_pred.get("portfolio_success_rate", 0)
        high_risk = success_pred.get("high_risk_projects", 0)
        safe_projects = success_pred.get("safe_projects", 0)
        
        succ_col1, succ_col2, succ_col3 = st.columns(3)
        
        with succ_col1:
            st.metric("Taux Global", f"{portfolio_success:.0f}%",
                     help="Probabilité de succès du portfolio")
        
        with succ_col2:
            st.metric("Projets à Risque", high_risk,
                     help="Projets avec probabilité < 40%")
        
        with succ_col3:
            st.metric("Projets Sûrs", safe_projects,
                     help="Projets avec probabilité > 80%")
        
        recommendation = success_pred.get("recommendation", "")
        if recommendation:
            st.info(f"💡 {recommendation}")

def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Afficher les recommandations stratégiques"""
    
    st.markdown("### 🎯 Recommandations Stratégiques")
    
    if not recommendations:
        st.info("Aucune recommandation disponible")
        return
    
    # Trier par priorité
    priority_order = {"high": 1, "medium": 2, "low": 3}
    sorted_recs = sorted(recommendations, 
                        key=lambda x: priority_order.get(x.get("priority", "low"), 3))
    
    for rec in sorted_recs:
        title = rec.get("title", "Recommandation")
        priority = rec.get("priority", "low")
        description = rec.get("description", "")
        actions = rec.get("actions", [])
        
        # Déterminer la couleur selon la priorité
        if priority == "high":
            color = "#ef4444"
            icon = "🚨"
            bg_color = "#fef2f2"
        elif priority == "medium":
            color = "#f59e0b"
            icon = "⚠️"
            bg_color = "#fefbf3"
        else:
            color = "#3b82f6"
            icon = "💡"
            bg_color = "#eff6ff"
        
        with st.expander(f"{icon} {title} (Priorité: {priority.title()})", 
                        expanded=priority == "high"):
            
            st.markdown(f"""
            <div style="
                background: {bg_color}; 
                padding: 1rem; 
                border-radius: 8px; 
                border-left: 4px solid {color};
                margin-bottom: 1rem;
            ">
                <p style="color: #374151; margin: 0;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if actions:
                st.markdown("**🎯 Plan d'Action:**")
                for i, action in enumerate(actions, 1):
                    st.markdown(f"**{i}.** {action}")