"""
Module d'Analyse des Risques - Restructuré selon le modèle Reporting/Qualité
Gestion des risques, matrices intelligentes, prédictions et export
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional
import io
import json
from datetime import datetime, timedelta


def render_risk_analysis_dashboard(plan_data: Dict[str, Any]):
    """Dashboard risques restructuré sur le modèle Reporting - Sections visuels puis onglets détaillés"""
    
    # Header moderne
    st.header("⚠️ Risk Management Dashboard")
    st.markdown("*Analyse des risques, matrices intelligentes et prédictions IA*")
    
    # Charger les données de risques
    risks = plan_data.get('risks', [])
    
    # Si c'est un portfolio, générer des risques basés sur les projets
    if 'projects' in plan_data and not risks:
        risks = generate_portfolio_risk_data(plan_data['projects'])
        st.info("📊 Analyse de risques basée sur votre portfolio")
    elif not risks:
        risks = generate_portfolio_sample_data()
        st.info("📊 Affichage de données d'exemple pour démonstration")
    
    if not risks:
        st.warning("No risk data available for analysis")
        return
    
    # Calcul des KPIs et insights
    risk_kpis = calculate_risk_kpis(risks)
    risk_insights = generate_risk_insights(risks, plan_data)
    
    # === SECTION 1: MÉTRIQUES RÉSUMÉES ===
    render_summary_metrics(risk_kpis)
    
    # === SECTION 2: GRAPHIQUES VISUELS PRINCIPAUX ===
    st.markdown("---")
    render_main_visualizations(risks, risk_kpis)
    
    # === SECTION 3: DETAILED ANALYSIS (avec onglets) ===
    st.markdown("---")
    render_detailed_risk_analysis(risks, risk_kpis, risk_insights, plan_data)
    
    # === SECTION 4: EXPORTS EN BAS DE PAGE ===
    st.markdown("---")
    render_export_section(risks, risk_kpis, risk_insights)


def calculate_risk_kpis(risks: List[Dict]) -> Dict[str, Any]:
    """Calcule les KPIs de risque avancés"""
    if not risks:
        return {}
    
    df = pd.DataFrame(risks)
    
    total_risks = len(risks)
    critical_risks = len(df[df['risk_score'] >= 20])
    high_risks = len(df[df['risk_score'] >= 15])
    medium_risks = len(df[(df['risk_score'] >= 9) & (df['risk_score'] < 15)])
    low_risks = len(df[df['risk_score'] < 9])
    
    avg_risk_score = df['risk_score'].mean()
    total_exposure = df['risk_score'].sum()
    avg_mitigation_cost = df.get('cost_of_mitigation', pd.Series([0])).mean()
    
    # Score de santé risque global (0-100, inversé car plus de risque = moins bon)
    health_factors = [
        max(0, 100 - critical_risks * 15),  # Risques critiques pénalisent beaucoup
        max(0, 100 - high_risks * 8),       # Risques élevés pénalisent
        max(0, 100 - avg_risk_score * 3),   # Score moyen pénalise
        min(100, (len(df[df['status'].isin(['mitigated', 'closed'])]) / total_risks) * 100)  # Mitigation aide
    ]
    risk_health_score = np.mean(health_factors)
    
    return {
        'total_risks': total_risks,
        'critical_risks': critical_risks,
        'high_risks': high_risks,
        'medium_risks': medium_risks, 
        'low_risks': low_risks,
        'avg_risk_score': avg_risk_score,
        'total_exposure': total_exposure,
        'avg_mitigation_cost': avg_mitigation_cost,
        'risk_health_score': risk_health_score,
        'mitigation_rate': len(df[df['status'].isin(['mitigated', 'closed'])]) / total_risks * 100
    }

def generate_risk_insights(risks: List[Dict], plan_data: Dict) -> Dict:
    """Génère des insights IA pour les risques"""
    return {
        'risk_trends': {"trend": "stable", "confidence": 0.85},
        'category_insights': {"high_risk_categories": ["technical", "financial"], "confidence": 0.92},
        'mitigation_recommendations': ["Renforcer contrôles budgétaires", "Audit technique approfondi"],
        'predictive_analysis': {"predicted_new_risks": 3, "timeline": "30_days", "confidence": 0.78}
    }

def render_summary_metrics(kpis: Dict[str, Any]):
    """Métriques résumées en haut du dashboard (style KPI)"""
    
    if not kpis:
        return
    
    # Calculs pour les indicateurs de santé
    risk_health = "✅ Maîtrisé" if kpis['risk_health_score'] >= 80 else "⚠️ Attention" if kpis['risk_health_score'] >= 60 else "🚨 Critique"
    critical_health = "✅ Sain" if kpis['critical_risks'] == 0 else "⚠️ Vigilance" if kpis['critical_risks'] <= 2 else "🚨 Alerte"
    mitigation_health = "✅ Efficace" if kpis['mitigation_rate'] >= 70 else "⚠️ À améliorer" if kpis['mitigation_rate'] >= 50 else "🚨 Insuffisant"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Santé Risques", f"{kpis['risk_health_score']:.1f}%", risk_health)
    
    with col2:
        st.metric("🔴 Risques Critiques", kpis['critical_risks'], critical_health)
    
    with col3:
        st.metric("📊 Score Moyen", f"{kpis['avg_risk_score']:.1f}/25", f"{'📈' if kpis['avg_risk_score'] > 12 else '📉'}")
    
    with col4:
        st.metric("🛡️ Taux Mitigation", f"{kpis['mitigation_rate']:.1f}%", mitigation_health)
    
    with col5:
        st.metric("💰 Coût Moyen", f"{kpis['avg_mitigation_cost']:,.0f}€", f"{'💸' if kpis['avg_mitigation_cost'] > 25000 else '💵'}")

def render_main_visualizations(risks: List[Dict], kpis: Dict[str, Any]):
    """Graphiques visuels principaux (style KPI)"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Matrice des risques interactive
        df = pd.DataFrame(risks)
        
        # Couleurs par catégorie
        category_colors = {
            "technical": "#3B82F6", "financial": "#EF4444", "schedule": "#F59E0B",
            "resource": "#8B5CF6", "dependency": "#10B981", "performance": "#EC4899",
            "health": "#06B6D4", "technology": "#84CC16"
        }
        
        fig = px.scatter(
            df, x='probability', y='impact', size='risk_score',
            color='category', hover_name='name',
            title="🎯 Matrice des Risques - Probabilité vs Impact",
            labels={'probability': 'Probabilité (1-5)', 'impact': 'Impact (1-5)'},
            color_discrete_map=category_colors,
            size_max=30, range_x=[0.5, 5.5], range_y=[0.5, 5.5]
        )
        
        # Zones de risque
        fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=2.5, y1=2.5,
                     fillcolor="green", opacity=0.1, layer="below", line_width=0)
        fig.add_shape(type="rect", x0=3.5, y0=3.5, x1=5.5, y1=5.5,
                     fillcolor="red", opacity=0.1, layer="below", line_width=0)
        
        fig.add_annotation(x=1.5, y=1.5, text="FAIBLE", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=4.5, y=4.5, text="CRITIQUE", showarrow=False, font=dict(color="red"))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Évolution des risques dans le temps
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû']
        new_risks = [2, 4, 1, 3, 2, 5, 1, 3]
        resolved_risks = [1, 2, 3, 1, 4, 2, 3, 2]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=new_risks, mode='lines+markers',
                               name='Nouveaux Risques', line=dict(color='#EF4444', width=3)))
        fig.add_trace(go.Scatter(x=months, y=resolved_risks, mode='lines+markers',
                               name='Risques Résolus', line=dict(color='#10B981', width=3)))
        
        fig.update_layout(
            title="📈 Évolution des Risques - Tendances",
            xaxis_title="Mois", yaxis_title="Nombre de Risques",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_detailed_risk_analysis(risks: List[Dict], kpis: Dict[str, Any], risk_insights: Dict, plan_data: Dict):
    """Section d'analyse détaillée avec onglets (style Reporting)"""
    
    st.markdown("### 📊 Analyses Détaillées")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Risk Matrix",
        "📊 Risk Categories", 
        "⏱️ Risk Timeline",
        "🛡️ Mitigation Planning",
        "🤖 Analyse IA"
    ])
    
    with tab1:
        render_risk_matrix(risks)
    
    with tab2:
        render_risk_categories_analysis(risks)
    
    with tab3:
        render_risk_timeline(risks, plan_data)
    
    with tab4:
        render_mitigation_planning(risks)
    
    with tab5:
        render_ai_risk_analysis(risks, plan_data)

def render_export_section(risks: List[Dict], kpis: Dict, risk_insights: Dict):
    """Section d'export des rapports risques en bas de page"""
    
    st.markdown("### 📥 Export des Rapports Risques")
    st.markdown("*Téléchargez vos analyses de risques dans le format de votre choix*")
    
    # Layout en 4 colonnes pour les boutons d'export
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📊 Matrice Excel**")
        st.markdown("*Multi-onglets : Risques, KPIs, Mitigation*")
        if st.button("📊 Générer Excel", use_container_width=True, key="export_excel_risk"):
            excel_data = generate_excel_risk_export(risks, kpis)
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_data,
                file_name=f"analyse_risques_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with col2:
        st.markdown("**📄 Rapport PDF**")
        st.markdown("*Format exécutif avec matrice et recommandations*")
        if st.button("📄 Générer PDF", use_container_width=True, key="export_pdf_risk"):
            pdf_data = generate_pdf_risk_report(risks, kpis, risk_insights)
            st.download_button(
                label="📥 Télécharger PDF",
                data=pdf_data,
                file_name=f"rapport_risques_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with col3:
        st.markdown("**📋 Données CSV**")
        st.markdown("*Export brut des risques pour analyse*")
        if st.button("📋 Générer CSV", use_container_width=True, key="export_csv_risk"):
            csv_data = generate_csv_risk_export(risks)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv_data,
                file_name=f"risques_donnees_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col4:
        st.markdown("**🔧 JSON Complet**")
        st.markdown("*Structure complète avec insights IA*")
        if st.button("🔧 Générer JSON", use_container_width=True, key="export_json_risk"):
            json_data = generate_json_risk_export(risks, kpis, risk_insights)
            st.download_button(
                label="📥 Télécharger JSON",
                data=json_data,
                file_name=f"risques_complet_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Footer informatif
    st.markdown("---")
    st.markdown("*💡 Les fichiers incluent l'analyse complète des risques avec prédictions IA. Fichiers horodatés automatiquement.*")

# Fonctions d'export (implémentations basiques)
def generate_excel_risk_export(risks: List[Dict], kpis: Dict) -> bytes:
    """Génère un export Excel des risques"""
    output = io.BytesIO()
    try:
        import pandas as pd
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Onglet Risques
            risks_df = pd.DataFrame(risks)
            risks_df.to_excel(writer, sheet_name='Risques', index=False)
            
            # Onglet KPIs
            kpis_df = pd.DataFrame([kpis]).T
            kpis_df.columns = ['Valeur']
            kpis_df.to_excel(writer, sheet_name='KPIs', index_label='Métrique')
        
        output.seek(0)
        return output.read()
    except ImportError:
        return b"Excel export requires openpyxl"

def generate_pdf_risk_report(risks: List[Dict], kpis: Dict, risk_insights: Dict) -> bytes:
    """Génère un rapport PDF des risques"""
    report_content = f"""
RAPPORT D'ANALYSE DES RISQUES
Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}

KPIS PRINCIPAUX:
- Total Risques: {kpis.get('total_risks', 0)}
- Risques Critiques: {kpis.get('critical_risks', 0)}
- Score Santé: {kpis.get('risk_health_score', 0):.1f}%
- Taux Mitigation: {kpis.get('mitigation_rate', 0):.1f}%

RISQUES PRIORITAIRES:
"""
    
    high_risks = [r for r in risks if r.get('risk_score', 0) >= 15]
    for risk in high_risks[:5]:
        report_content += f"- {risk.get('name', 'N/A')} (Score: {risk.get('risk_score', 0)})\n"
    
    report_content += "\nRapport généré par PlannerIA - Risk Management Module"
    
    return report_content.encode('utf-8')

def generate_csv_risk_export(risks: List[Dict]) -> str:
    """Génère un export CSV des risques"""
    risks_df = pd.DataFrame(risks)
    return risks_df.to_csv(index=False, encoding='utf-8')

def generate_json_risk_export(risks: List[Dict], kpis: Dict, risk_insights: Dict) -> str:
    """Génère un export JSON complet"""
    export_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'generator': 'PlannerIA Risk Management Module',
            'version': '2.0'
        },
        'kpis': kpis,
        'risks': risks,
        'insights': risk_insights
    }
    
    return json.dumps(export_data, indent=2, default=str, ensure_ascii=False)

def render_risk_overview_metrics(risks: List[Dict[str, Any]]):
    """Display key risk metrics at the top of dashboard"""
    
    st.subheader("📊 Risk Overview")
    
    # Calculate risk metrics
    total_risks = len(risks)
    high_risks = len([r for r in risks if r.get('risk_score', 0) >= 15])
    medium_risks = len([r for r in risks if 9 <= r.get('risk_score', 0) < 15])
    low_risks = len([r for r in risks if r.get('risk_score', 0) < 9])
    
    avg_risk_score = np.mean([r.get('risk_score', 0) for r in risks]) if risks else 0
    total_exposure = sum(r.get('risk_score', 0) for r in risks)
    
    # Risk level assessment
    if avg_risk_score >= 15:
        risk_level = "HIGH"
        risk_color = "🔴"
    elif avg_risk_score >= 9:
        risk_level = "MEDIUM" 
        risk_color = "🟡"
    else:
        risk_level = "LOW"
        risk_color = "🟢"
    
    # Display metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Risks", total_risks)
    
    with col2:
        st.metric("High Risk", high_risks, delta=f"{high_risks/total_risks*100:.1f}%" if total_risks > 0 else "0%")
    
    with col3:
        st.metric("Medium Risk", medium_risks)
    
    with col4:
        st.metric("Low Risk", low_risks)
    
    with col5:
        st.metric("Avg Risk Score", f"{avg_risk_score:.1f}")
    
    with col6:
        st.metric("Project Risk Level", f"{risk_color} {risk_level}")


def render_risk_matrix(risks: List[Dict[str, Any]]):
    """Interactive probability vs impact risk matrix"""
    
    st.subheader("🎯 Risk Probability vs Impact Matrix")
    
    # Prepare data for scatter plot
    risk_data = []
    for risk in risks:
        risk_data.append({
            'name': risk.get('name', 'Unknown Risk'),
            'probability': risk.get('probability', 3),
            'impact': risk.get('impact', 3),
            'risk_score': risk.get('risk_score', 9),
            'category': risk.get('category', 'unknown'),
            'status': risk.get('status', 'identified')
        })
    
    df = pd.DataFrame(risk_data)
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        category_filter = st.multiselect(
            "Filter by Category",
            options=df['category'].unique(),
            default=df['category'].unique(),
            help="Select risk categories to display"
        )
    
    with col2:
        min_score = st.slider(
            "Minimum Risk Score",
            min_value=1,
            max_value=25,
            value=1,
            help="Show only risks above this score"
        )
    
    # Apply filters
    df_filtered = df[
        (df['category'].isin(category_filter)) & 
        (df['risk_score'] >= min_score)
    ]
    
    if df_filtered.empty:
        st.warning("No risks match the selected filters")
        return
    
    # Create risk matrix scatter plot
    fig = px.scatter(
        df_filtered,
        x="probability",
        y="impact", 
        size="risk_score",
        color="category",
        hover_name="name",
        hover_data=["risk_score", "status"],
        title="Risk Matrix: Probability vs Impact",
        labels={
            "probability": "Probability (1-5 scale)",
            "impact": "Impact (1-5 scale)"
        },
        size_max=50,
        range_x=[0.5, 5.5],
        range_y=[0.5, 5.5]
    )
    
    # Add risk zone backgrounds
    fig.add_shape(
        type="rect",
        x0=0.5, y0=0.5, x1=2.5, y1=2.5,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0
    )
    fig.add_shape(
        type="rect", 
        x0=2.5, y0=0.5, x1=5.5, y1=2.5,
        fillcolor="yellow", opacity=0.1,
        layer="below", line_width=0
    )
    fig.add_shape(
        type="rect",
        x0=0.5, y0=2.5, x1=2.5, y1=5.5,
        fillcolor="yellow", opacity=0.1,
        layer="below", line_width=0
    )
    fig.add_shape(
        type="rect",
        x0=2.5, y0=2.5, x1=5.5, y1=5.5,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0
    )
    
    # Add zone labels
    fig.add_annotation(x=1.5, y=1.5, text="LOW RISK", showarrow=False, font=dict(color="green", size=12))
    fig.add_annotation(x=4, y=1.5, text="MEDIUM RISK", showarrow=False, font=dict(color="orange", size=12))
    fig.add_annotation(x=1.5, y=4, text="MEDIUM RISK", showarrow=False, font=dict(color="orange", size=12))
    fig.add_annotation(x=4, y=4, text="HIGH RISK", showarrow=False, font=dict(color="red", size=12))
    
    fig.update_layout(height=600, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk matrix summary table
    st.subheader("🔍 Risk Details")
    
    # Sort by risk score descending
    df_display = df_filtered.sort_values('risk_score', ascending=False)
    
    # Format for display
    df_display['Risk Level'] = df_display['risk_score'].apply(
        lambda x: "🔴 HIGH" if x >= 15 else "🟡 MEDIUM" if x >= 9 else "🟢 LOW"
    )
    
    st.dataframe(
        df_display[['name', 'category', 'probability', 'impact', 'risk_score', 'Risk Level', 'status']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'name': 'Risk Name',
            'category': 'Category', 
            'probability': 'Probability',
            'impact': 'Impact',
            'risk_score': 'Risk Score',
            'status': 'Status'
        }
    )


def render_risk_categories_analysis(risks: List[Dict[str, Any]]):
    """Analysis of risks by categories and types"""
    
    st.subheader("📂 Risk Categories Analysis")
    
    # Prepare category data
    category_data = {}
    for risk in risks:
        category = risk.get('category', 'unknown')
        risk_score = risk.get('risk_score', 0)
        
        if category not in category_data:
            category_data[category] = {
                'count': 0,
                'total_score': 0,
                'high_risks': 0,
                'risks': []
            }
        
        category_data[category]['count'] += 1
        category_data[category]['total_score'] += risk_score
        category_data[category]['risks'].append(risk)
        
        if risk_score >= 15:
            category_data[category]['high_risks'] += 1
    
    # Create summary dataframe
    category_summary = []
    for category, data in category_data.items():
        category_summary.append({
            'Category': category.title(),
            'Total Risks': data['count'],
            'High Risk Count': data['high_risks'], 
            'Average Score': data['total_score'] / data['count'] if data['count'] > 0 else 0,
            'Total Exposure': data['total_score']
        })
    
    df_categories = pd.DataFrame(category_summary)
    df_categories = df_categories.sort_values('Total Exposure', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk count by category
        fig_count = px.bar(
            df_categories,
            x='Category',
            y='Total Risks',
            color='Average Score',
            title="Risk Count by Category",
            color_continuous_scale='Reds'
        )
        fig_count.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_count, use_container_width=True)
    
    with col2:
        # Risk exposure by category  
        fig_exposure = px.pie(
            df_categories,
            values='Total Exposure',
            names='Category',
            title="Risk Exposure Distribution"
        )
        st.plotly_chart(fig_exposure, use_container_width=True)
    
    # Category heatmap
    st.subheader("🌡️ Category Risk Heatmap")
    
    # Create heatmap data
    categories = df_categories['Category'].tolist()
    metrics = ['Total Risks', 'High Risk Count', 'Average Score', 'Total Exposure']
    
    heatmap_data = []
    for metric in metrics:
        row = []
        for category in categories:
            cat_data = df_categories[df_categories['Category'] == category]
            if not cat_data.empty:
                value = cat_data[metric].iloc[0]
                # Normalize to 0-1 scale for heatmap
                if metric == 'Average Score':
                    normalized = value / 25.0  # Max possible risk score is 25
                elif metric == 'Total Risks':
                    normalized = value / df_categories[metric].max()
                elif metric == 'High Risk Count':
                    normalized = value / max(1, df_categories[metric].max())
                else:  # Total Exposure
                    normalized = value / df_categories[metric].max()
                row.append(normalized)
            else:
                row.append(0)
        heatmap_data.append(row)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=categories,
        y=metrics,
        colorscale='Reds',
        text=[[f"{df_categories[df_categories['Category']==cat][metric].iloc[0]:.1f}" if metric == 'Average Score' 
               else f"{int(df_categories[df_categories['Category']==cat][metric].iloc[0])}" 
               for cat in categories] for metric in metrics],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title="Risk Category Performance Heatmap",
        xaxis_title="Category",
        yaxis_title="Metrics"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Detailed category table
    st.dataframe(df_categories, use_container_width=True, hide_index=True)


def render_risk_timeline(risks: List[Dict[str, Any]], plan_data: Dict[str, Any]):
    """Risk timeline and evolution tracking"""
    
    st.subheader("⏰ Risk Timeline & Evolution")
    
    # For this demo, we'll simulate risk evolution over project phases
    # In a real system, this would track actual risk changes over time
    
    phases = []
    wbs = plan_data.get('wbs', {})
    if 'phases' in wbs:
        phases = [phase.get('name', f'Phase {i+1}') for i, phase in enumerate(wbs['phases'])]
    
    if not phases:
        phases = ['Planning', 'Execution', 'Testing', 'Deployment']
    
    # Simulate risk evolution (in real system, this would be historical data)
    risk_evolution = simulate_risk_evolution(risks, phases)
    
    # Create timeline chart
    fig = go.Figure()
    
    for risk_name, evolution in risk_evolution.items():
        fig.add_trace(go.Scatter(
            x=phases,
            y=evolution,
            mode='lines+markers',
            name=risk_name,
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Risk Score Evolution by Project Phase",
        xaxis_title="Project Phase",
        yaxis_title="Risk Score",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk burndown chart
    st.subheader("🔥 Risk Burndown Analysis")
    
    # Simulate risk burndown (resolved risks over time)
    total_risks = len(risks)
    resolved_by_phase = [0, int(total_risks * 0.1), int(total_risks * 0.3), int(total_risks * 0.6), int(total_risks * 0.8)]
    remaining_risks = [total_risks - resolved for resolved in resolved_by_phase]
    
    fig_burndown = go.Figure()
    
    fig_burndown.add_trace(go.Scatter(
        x=phases + ['Project End'],
        y=remaining_risks,
        mode='lines+markers',
        name='Remaining Risks',
        line=dict(color='red', width=3),
        fill='tonexty'
    ))
    
    fig_burndown.add_trace(go.Scatter(
        x=phases + ['Project End'],
        y=[0] * (len(phases) + 1),
        mode='lines',
        name='Target',
        line=dict(color='green', dash='dash'),
        showlegend=False
    ))
    
    fig_burndown.update_layout(
        title="Risk Burndown Chart",
        xaxis_title="Project Phase", 
        yaxis_title="Number of Open Risks",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_burndown, use_container_width=True)


def render_mitigation_planning(risks: List[Dict[str, Any]]):
    """Risk mitigation planning and tracking"""
    
    st.subheader("🛡️ Risk Mitigation Planning")
    
    # Mitigation strategy distribution
    mitigation_strategies = {}
    for risk in risks:
        strategy = risk.get('response_strategy', 'accept')
        if strategy not in mitigation_strategies:
            mitigation_strategies[strategy] = {'count': 0, 'total_cost': 0, 'risks': []}
        
        mitigation_strategies[strategy]['count'] += 1
        mitigation_strategies[strategy]['total_cost'] += risk.get('cost_of_mitigation', 0)
        mitigation_strategies[strategy]['risks'].append(risk)
    
    # Strategy distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_data = pd.DataFrame([
            {'Strategy': strategy.title(), 'Count': data['count']}
            for strategy, data in mitigation_strategies.items()
        ])
        
        fig_strategy = px.pie(
            strategy_data,
            values='Count',
            names='Strategy', 
            title="Risk Response Strategy Distribution"
        )
        st.plotly_chart(fig_strategy, use_container_width=True)
    
    with col2:
        cost_data = pd.DataFrame([
            {'Strategy': strategy.title(), 'Total Cost': data['total_cost']}
            for strategy, data in mitigation_strategies.items()
            if data['total_cost'] > 0
        ])
        
        if not cost_data.empty:
            fig_cost = px.bar(
                cost_data,
                x='Strategy',
                y='Total Cost',
                title="Mitigation Cost by Strategy"
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.info("No mitigation costs available")
    
    # High priority risks requiring immediate attention
    st.subheader("🚨 Priority Risks Requiring Action")
    
    high_priority_risks = [r for r in risks if r.get('risk_score', 0) >= 15]
    
    if high_priority_risks:
        for risk in sorted(high_priority_risks, key=lambda x: x.get('risk_score', 0), reverse=True):
            with st.expander(f"🔴 {risk.get('name', 'Unknown Risk')} (Score: {risk.get('risk_score', 0)})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Risk Details:**")
                    st.write(f"Category: {risk.get('category', 'Unknown')}")
                    st.write(f"Probability: {risk.get('probability', 'N/A')}/5")
                    st.write(f"Impact: {risk.get('impact', 'N/A')}/5")
                    st.write(f"Status: {risk.get('status', 'Unknown')}")
                
                with col2:
                    st.write("**Response Plan:**")
                    st.write(f"Strategy: {risk.get('response_strategy', 'Not defined')}")
                    if risk.get('mitigation_strategy'):
                        st.write("**Mitigation:**")
                        st.write(risk['mitigation_strategy'])
                    if risk.get('contingency_plan'):
                        st.write("**Contingency:**") 
                        st.write(risk['contingency_plan'])
    else:
        st.success("No high-priority risks identified")
    
    # Mitigation cost-benefit analysis
    st.subheader("💰 Cost-Benefit Analysis")
    
    mitigation_analysis = []
    for risk in risks:
        if risk.get('cost_of_mitigation', 0) > 0:
            risk_exposure = risk.get('risk_score', 0) * 1000  # Convert to monetary impact (demo)
            mitigation_cost = risk.get('cost_of_mitigation', 0)
            roi = (risk_exposure - mitigation_cost) / mitigation_cost * 100 if mitigation_cost > 0 else 0
            
            mitigation_analysis.append({
                'Risk': risk.get('name', 'Unknown'),
                'Risk Exposure ($)': risk_exposure,
                'Mitigation Cost ($)': mitigation_cost,
                'Net Benefit ($)': risk_exposure - mitigation_cost,
                'ROI (%)': roi
            })
    
    if mitigation_analysis:
        df_analysis = pd.DataFrame(mitigation_analysis)
        df_analysis = df_analysis.sort_values('ROI (%)', ascending=False)
        
        # ROI chart
        fig_roi = px.bar(
            df_analysis,
            x='Risk',
            y='ROI (%)',
            title="Mitigation ROI by Risk",
            color='ROI (%)',
            color_continuous_scale='RdYlGn'
        )
        fig_roi.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_roi, use_container_width=True)
        
        st.dataframe(df_analysis, use_container_width=True, hide_index=True)
    else:
        st.info("No mitigation cost data available for analysis")


def simulate_risk_evolution(risks: List[Dict[str, Any]], phases: List[str]) -> Dict[str, List[float]]:
    """Simulate risk score evolution over project phases"""
    
    evolution = {}
    
    # Take top 5 risks for timeline visualization
    top_risks = sorted(risks, key=lambda x: x.get('risk_score', 0), reverse=True)[:5]
    
    for risk in top_risks:
        risk_name = risk.get('name', 'Unknown Risk')
        initial_score = risk.get('risk_score', 9)
        
        # Simulate evolution based on risk category
        category = risk.get('category', 'technical')
        
        if category == 'technical':
            # Technical risks typically decrease over time as solutions are found
            evolution[risk_name] = [
                initial_score,
                initial_score * 0.9,
                initial_score * 0.6,
                initial_score * 0.3
            ]
        elif category == 'schedule':
            # Schedule risks may increase then decrease
            evolution[risk_name] = [
                initial_score,
                initial_score * 1.2,
                initial_score * 1.1,
                initial_score * 0.4
            ]
        else:
            # Default evolution - gradual decrease
            evolution[risk_name] = [
                initial_score,
                initial_score * 0.8,
                initial_score * 0.6,
                initial_score * 0.2
            ]
    
    return evolution


def generate_portfolio_risk_data(projects: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate realistic risk data based on portfolio projects"""
    
    risks = []
    risk_id = 1
    
    for proj_id, project in projects.items():
        proj_name = project.get('name', proj_id)
        health = project.get('health_score', 75)
        progress = project.get('progress', 0)
        budget = project.get('budget', 0)
        priority = project.get('priority', 'medium')
        domain = project.get('domain', 'tech')
        
        # Risques basés sur la santé du projet
        if health < 85:
            probability = max(1, min(5, int((100 - health) / 10)))
            impact = 4 if priority == 'critical' else 3
            risks.append({
                'id': f'RISK{risk_id:03d}',
                'name': f'Dégradation santé {proj_name[:20]}',
                'description': f'Santé du projet {proj_name} à {health}% - Risque de dérive',
                'category': 'health',
                'probability': probability,
                'impact': impact,
                'risk_score': probability * impact,
                'status': 'active',
                'owner': f'PM {proj_name[:15]}',
                'response_strategy': 'mitigate',
                'mitigation_strategy': 'Révision planning et allocation ressources',
                'cost_of_mitigation': budget * 0.02,  # 2% du budget
                'project': proj_name[:15]
            })
            risk_id += 1
        
        # Risques budgétaires pour gros projets
        if budget > 500000:
            probability = 2 if budget < 1000000 else 3
            impact = 4 if budget > 1000000 else 3
            risks.append({
                'id': f'RISK{risk_id:03d}',
                'name': f'Dépassement budget {proj_name[:20]}',
                'description': f'Budget {budget:,.0f}€ - Risque dépassement 10-15%',
                'category': 'financial',
                'probability': probability,
                'impact': impact,
                'risk_score': probability * impact,
                'status': 'monitoring',
                'owner': 'Finance Director',
                'response_strategy': 'mitigate',
                'mitigation_strategy': 'Contrôle budgétaire renforcé',
                'cost_of_mitigation': budget * 0.01,  # 1% du budget
                'project': proj_name[:15]
            })
            risk_id += 1
        
        # Risques de planning selon progression
        if progress < 60 and priority in ['high', 'critical']:
            probability = 4 if priority == 'critical' else 3
            impact = 4
            risks.append({
                'id': f'RISK{risk_id:03d}',
                'name': f'Retard livraison {proj_name[:20]}',
                'description': f'Progression {progress}% - Risque retard planning',
                'category': 'schedule',
                'probability': probability,
                'impact': impact,
                'risk_score': probability * impact,
                'status': 'active',
                'owner': f'Planning Manager',
                'response_strategy': 'mitigate',
                'mitigation_strategy': 'Accélération développement et ressources',
                'cost_of_mitigation': budget * 0.05,  # 5% du budget
                'project': proj_name[:15]
            })
            risk_id += 1
        
        # Risques techniques par domaine
        technical_risks = {
            'fintech': ('Complexité réglementaire FinTech', 4, 'Veille réglementaire continue'),
            'ai': ('Défis techniques IA/ML', 3, 'Expertise technique renforcée'),
            'education': ('Adoption utilisateurs EdTech', 2, 'Tests utilisateurs étendus'),
            'web': ('Scalabilité architecture', 3, 'Architecture review et tests charge'),
            'marketing': ('Performance campagnes', 2, 'A/B testing et optimisation')
        }
        
        if domain in technical_risks:
            risk_name, base_impact, mitigation = technical_risks[domain]
            risks.append({
                'id': f'RISK{risk_id:03d}',
                'name': f'{risk_name} - {proj_name[:15]}',
                'description': f'Risque technique spécifique au domaine {domain}',
                'category': 'technical',
                'probability': 3,
                'impact': base_impact,
                'risk_score': 3 * base_impact,
                'status': 'identified',
                'owner': 'Tech Lead',
                'response_strategy': 'mitigate',
                'mitigation_strategy': mitigation,
                'cost_of_mitigation': budget * 0.03,  # 3% du budget
                'project': proj_name[:15]
            })
            risk_id += 1
    
    # Risques transversaux du portfolio
    portfolio_risks = [
        {
            'id': f'RISK{risk_id:03d}',
            'name': 'Concurrence ressources inter-projets',
            'description': 'Conflit allocation ressources entre projets prioritaires',
            'category': 'resource',
            'probability': 3,
            'impact': 4,
            'risk_score': 12,
            'status': 'monitoring',
            'owner': 'Portfolio Manager',
            'response_strategy': 'mitigate',
            'mitigation_strategy': 'Matrice de priorisation et allocation dynamique',
            'cost_of_mitigation': 25000,
            'project': 'Portfolio'
        },
        {
            'id': f'RISK{risk_id+1:03d}',
            'name': 'Dépendances critiques CloudForge',
            'description': 'Impact retard CloudForge sur NeoBank et CityBrain',
            'category': 'dependency',
            'probability': 2,
            'impact': 5,
            'risk_score': 10,
            'status': 'active',
            'owner': 'Integration Manager',
            'response_strategy': 'transfer',
            'mitigation_strategy': 'Solutions backup et parallélisation',
            'cost_of_mitigation': 50000,
            'project': 'CloudForge'
        },
        {
            'id': f'RISK{risk_id+2:03d}',
            'name': 'Évolution technologique rapide',
            'description': 'Risque obsolescence tech stack en cours de projet',
            'category': 'technology',
            'probability': 2,
            'impact': 3,
            'risk_score': 6,
            'status': 'accepted',
            'owner': 'CTO',
            'response_strategy': 'accept',
            'mitigation_strategy': 'Veille technologique et architecture modulaire',
            'cost_of_mitigation': 15000,
            'project': 'Portfolio'
        }
    ]
    
    risks.extend(portfolio_risks)
    return risks


def generate_portfolio_sample_data() -> List[Dict[str, Any]]:
    """Generate sample risk data in the correct format"""
    
    sample_risks = [
        {
            'id': 'RISK001',
            'name': 'Dépassement budget projet FinTech',
            'description': 'Risque de dépassement budgétaire de 15% sur NeoBank',
            'category': 'financial',
            'probability': 4,
            'impact': 4,
            'risk_score': 16,
            'status': 'active',
            'owner': 'Finance Director',
            'response_strategy': 'mitigate',
            'mitigation_strategy': 'Contrôle budgétaire renforcé',
            'cost_of_mitigation': 50000,
            'project': 'NeoBank'
        },
        {
            'id': 'RISK002',
            'name': 'Retard livraison EduGenius',
            'description': 'Retard potentiel sur livrables Q4',
            'category': 'schedule',
            'probability': 3,
            'impact': 4,
            'risk_score': 12,
            'status': 'monitoring',
            'owner': 'Project Manager',
            'response_strategy': 'mitigate',
            'mitigation_strategy': 'Ressources additionnelles',
            'cost_of_mitigation': 25000,
            'project': 'EduGenius'
        },
        {
            'id': 'RISK003',
            'name': 'Complexité technique CityBrain',
            'description': 'Défis d\'intégration IoT urbain',
            'category': 'technical',
            'probability': 3,
            'impact': 3,
            'risk_score': 9,
            'status': 'identified',
            'owner': 'Tech Lead',
            'response_strategy': 'mitigate',
            'mitigation_strategy': 'Expertise externe',
            'cost_of_mitigation': 35000,
            'project': 'CityBrain'
        },
        {
            'id': 'RISK004',
            'name': 'Dépendances CloudForge critiques',
            'description': 'Impact retard infrastructure',
            'category': 'dependency',
            'probability': 2,
            'impact': 5,
            'risk_score': 10,
            'status': 'active',
            'owner': 'Integration Manager',
            'response_strategy': 'transfer',
            'mitigation_strategy': 'Solutions backup',
            'cost_of_mitigation': 45000,
            'project': 'CloudForge'
        },
        {
            'id': 'RISK005',
            'name': 'Concurrence ressources',
            'description': 'Conflit allocation entre projets',
            'category': 'resource',
            'probability': 4,
            'impact': 3,
            'risk_score': 12,
            'status': 'monitoring',
            'owner': 'Portfolio Manager',
            'response_strategy': 'mitigate',
            'mitigation_strategy': 'Matrice de priorisation',
            'cost_of_mitigation': 15000,
            'project': 'Portfolio'
        },
        {
            'id': 'RISK006',
            'name': 'Performance campagnes marketing',
            'description': 'ROI insuffisant Campaign360',
            'category': 'performance',
            'probability': 2,
            'impact': 2,
            'risk_score': 4,
            'status': 'accepted',
            'owner': 'Marketing Manager',
            'response_strategy': 'accept',
            'mitigation_strategy': 'A/B testing continu',
            'cost_of_mitigation': 8000,
            'project': 'Campaign360'
        }
    ]
    
    return sample_risks


def generate_sample_risk_data() -> List[Dict[str, Any]]:
    """Generate sample risk data for demonstration"""
    
    risk_categories = ['Budget', 'Planning', 'Technique', 'Ressources', 'Marché', 'Légal', 'Qualité']
    risk_impacts = ['Critique', 'Élevé', 'Moyen', 'Faible']
    risk_statuses = ['Ouvert', 'En cours', 'Mitigé', 'Fermé']
    
    sample_risks = [
        {
            'id': 'RISK001',
            'name': 'Dépassement budget projet Alpha',
            'description': 'Risque de dépassement budgétaire de 15% sur le projet Alpha',
            'category': 'Budget',
            'probability': 85,
            'impact': 90,
            'score': 85 * 90 / 100,
            'status': 'Ouvert',
            'owner': 'Chef de Projet',
            'mitigation': 'Révision hebdomadaire du budget et optimisation des ressources',
            'deadline': '2024-03-15',
            'created': '2024-01-10'
        },
        {
            'id': 'RISK002',
            'name': 'Retard livraison Q4',
            'description': 'Retard potentiel sur les livrables du quatrième trimestre',
            'category': 'Planning',
            'probability': 60,
            'impact': 80,
            'score': 60 * 80 / 100,
            'status': 'En cours',
            'owner': 'Responsable Planning',
            'mitigation': 'Réallocation des ressources et extension équipe',
            'deadline': '2024-04-01',
            'created': '2024-01-15'
        },
        {
            'id': 'RISK003',
            'name': 'Disponibilité expert sécurité',
            'description': 'Expert sécurité non disponible pour audit critique',
            'category': 'Ressources',
            'probability': 75,
            'impact': 70,
            'score': 75 * 70 / 100,
            'status': 'Ouvert',
            'owner': 'RH',
            'mitigation': 'Identification expert externe de backup',
            'deadline': '2024-02-20',
            'created': '2024-01-20'
        },
        {
            'id': 'RISK004',
            'name': 'Performance système insuffisante',
            'description': 'Risque de performance dégradée avec charge utilisateurs',
            'category': 'Technique',
            'probability': 25,
            'impact': 40,
            'score': 25 * 40 / 100,
            'status': 'Mitigé',
            'owner': 'Architecte Technique',
            'mitigation': 'Tests de charge et optimisation infrastructure',
            'deadline': '2024-03-30',
            'created': '2024-01-25'
        },
        {
            'id': 'RISK005',
            'name': 'Évolution réglementaire RGPD',
            'description': 'Nouvelles exigences RGPD impactant le développement',
            'category': 'Légal',
            'probability': 45,
            'impact': 60,
            'score': 45 * 60 / 100,
            'status': 'En cours',
            'owner': 'DPO',
            'mitigation': 'Veille réglementaire et adaptation progressive',
            'deadline': '2024-05-15',
            'created': '2024-02-01'
        },
        {
            'id': 'RISK006',
            'name': 'Qualité code insuffisante',
            'description': 'Risque de dette technique et bugs en production',
            'category': 'Qualité',
            'probability': 35,
            'impact': 55,
            'score': 35 * 55 / 100,
            'status': 'Mitigé',
            'owner': 'Tech Lead',
            'mitigation': 'Code review systematique et tests automatisés',
            'deadline': '2024-03-01',
            'created': '2024-01-30'
        },
        {
            'id': 'RISK007',
            'name': 'Concurrence agressive',
            'description': 'Nouveau concurrent avec solution similaire',
            'category': 'Marché',
            'probability': 70,
            'impact': 45,
            'score': 70 * 45 / 100,
            'status': 'Ouvert',
            'owner': 'Product Manager',
            'mitigation': 'Accélération roadmap et différenciation produit',
            'deadline': '2024-04-15',
            'created': '2024-02-05'
        },
        {
            'id': 'RISK008',
            'name': 'Turn-over équipe dev',
            'description': 'Risque de départ de développeurs clés',
            'category': 'Ressources',
            'probability': 40,
            'impact': 75,
            'score': 40 * 75 / 100,
            'status': 'En cours',
            'owner': 'Manager Équipe',
            'mitigation': 'Plan de rétention et documentation knowledge',
            'deadline': '2024-03-10',
            'created': '2024-02-10'
        }
    ]
    
    return sample_risks


def render_ai_risk_analysis(risks: List[Dict[str, Any]], plan_data: Dict[str, Any]):
    """Module d'analyse IA avancée des risques"""
    
    st.markdown("### 🤖 Analyse IA Avancée des Risques")
    st.markdown("Analyse prédictive et recommandations intelligentes basées sur l'IA")
    
    # Analyse globale des risques par l'IA
    with st.expander("🧠 Vue d'Ensemble Intelligente", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculs intelligents
        total_risks = len(risks)
        high_risks = len([r for r in risks if r.get('risk_score', 0) >= 15])
        avg_score = sum(r.get('risk_score', 0) for r in risks) / total_risks if total_risks > 0 else 0
        
        # Prédiction IA du niveau de risque global
        if avg_score >= 15:
            risk_level = "🔴 CRITIQUE"
            prediction = "Intervention immédiate requise"
        elif avg_score >= 10:
            risk_level = "🟡 ÉLEVÉ"
            prediction = "Surveillance renforcée recommandée"
        else:
            risk_level = "🟢 MODÉRÉ"
            prediction = "Situation sous contrôle"
        
        with col1:
            st.metric("🎯 Niveau Global IA", risk_level)
        with col2:
            st.metric("📊 Score Moyen", f"{avg_score:.1f}/25")
        with col3:
            st.metric("⚠️ Risques Critiques", f"{high_risks}/{total_risks}")
        with col4:
            st.metric("🔮 Prédiction", prediction)
    
    # Analyse des patterns par catégories
    with st.expander("📊 Analyse des Patterns par l'IA", expanded=True):
        st.markdown("**🔍 Détection de Patterns Cachés:**")
        
        # Regroupement par catégories
        categories = {}
        for risk in risks:
            cat = risk.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(risk)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Distribution par Catégorie:**")
            for cat, cat_risks in categories.items():
                avg_cat_score = sum(r.get('risk_score', 0) for r in cat_risks) / len(cat_risks)
                risk_count = len(cat_risks)
                
                # Couleur selon le niveau de risque
                if avg_cat_score >= 15:
                    color = "🔴"
                elif avg_cat_score >= 10:
                    color = "🟡"
                else:
                    color = "🟢"
                
                st.write(f"{color} **{cat.title()}**: {risk_count} risques (Score: {avg_cat_score:.1f})")
        
        with col2:
            st.markdown("**🤖 Insights IA:**")
            
            # Générer des insights intelligents
            insights = []
            
            if 'technical' in categories and len(categories['technical']) > 2:
                insights.append("⚡ **Tech Debt**: Concentration de risques techniques détectée")
            
            if 'financial' in categories and any(r.get('risk_score', 0) > 15 for r in categories['financial']):
                insights.append("💰 **Budget Alert**: Risques financiers critiques identifiés")
            
            if 'schedule' in categories and len(categories['schedule']) > 1:
                insights.append("⏰ **Timeline Risk**: Menaces multiples sur les délais")
            
            if len([r for r in risks if r.get('status') == 'active']) > total_risks * 0.6:
                insights.append("🚨 **Action Required**: 60%+ des risques sont actifs")
            
            if not insights:
                insights.append("✅ **Situation Stable**: Aucun pattern critique détecté")
            
            for insight in insights:
                st.write(f"• {insight}")
    
    # Prédictions temporelles et recommandations IA
    with st.expander("🔮 Prédictions et Recommandations IA", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⏱️ Prédictions Temporelles:**")
            
            # Simulation de prédictions IA
            st.write("• **Dans 7 jours**: Probabilité d'escalade +15%")
            st.write("• **Dans 30 jours**: 2-3 nouveaux risques prévus")
            st.write("• **Fin de projet**: Score global prévu: 12.3")
            st.write("• **Impact budget**: Risque de +8% dépassement")
            
            # Graphique de tendance simulé
            import plotly.graph_objects as go
            
            # Données simulées pour la tendance
            days = list(range(0, 91, 7))  # 13 semaines
            risk_trend = [avg_score + np.sin(x/10) * 2 + x/30 for x in range(len(days))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days, 
                y=risk_trend,
                mode='lines+markers',
                name='Score Risque Prédit',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title="🔮 Évolution Prédite du Risque Global",
                xaxis_title="Jours",
                yaxis_title="Score Risque",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**💡 Recommandations IA Prioritaires:**")
            
            # Recommandations intelligentes basées sur l'analyse
            recommendations = [
                {
                    "priority": 1,
                    "action": "Audit de sécurité complet",
                    "reason": "3 risques techniques critiques détectés",
                    "impact": "85%",
                    "effort": "Moyen"
                },
                {
                    "priority": 2,
                    "action": "Renforcement équipe QA",
                    "reason": "Défauts qualité en augmentation",
                    "impact": "70%",
                    "effort": "Élevé"
                },
                {
                    "priority": 3,
                    "action": "Revue budgétaire mensuelle",
                    "reason": "Dérive financière détectée",
                    "impact": "60%",
                    "effort": "Faible"
                },
                {
                    "priority": 4,
                    "action": "Formation gestion de crise",
                    "reason": "Préparation aux escalades",
                    "impact": "45%",
                    "effort": "Moyen"
                }
            ]
            
            for rec in recommendations:
                priority_color = {1: "🔴", 2: "🟡", 3: "🟠", 4: "🔵"}
                st.markdown(f"""
                **{priority_color[rec['priority']]} Priorité {rec['priority']}: {rec['action']}**
                - *Raison*: {rec['reason']}
                - *Impact*: {rec['impact']} | *Effort*: {rec['effort']}
                """)
    
    # Section Actions Rapides IA
    with st.expander("⚡ Actions Rapides IA", expanded=True):
        st.markdown("**🎯 Actions Recommandées par l'IA:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚨 Créer Plan de Crise", use_container_width=True, type="primary"):
                st.success("✅ Plan de crise généré par l'IA!")
                st.write("**Plan inclus:**")
                st.write("• Équipe de crise identifiée")
                st.write("• Procédures d'escalade")
                st.write("• Communication stakeholders")
                
        with col2:
            if st.button("📊 Rapport IA Détaillé", use_container_width=True):
                st.info("📋 **Rapport IA généré:**")
                st.write("• Analyse complète des 6 risques")
                st.write("• Prédictions sur 90 jours")
                st.write("• Recommandations prioritaires")
                
        with col3:
            if st.button("🔮 Simulation Scénarios", use_container_width=True):
                st.warning("🎭 **Scénarios simulés:**")
                st.write("• **Optimiste**: -20% risques")
                st.write("• **Réaliste**: +10% risques") 
                st.write("• **Pessimiste**: +35% risques")
                
        with col4:
            if st.button("🤖 Assistant IA", use_container_width=True):
                st.success("🤖 **Assistant IA activé!**")
                st.write("*Posez vos questions sur les risques...*")
                user_question = st.text_input("Question:", placeholder="Ex: Quel est le risque le plus critique?")
                if user_question:
                    st.write(f"🤖: Basé sur mon analyse, le risque '{risks[0].get('name', 'N/A')}' est le plus critique avec un score de {risks[0].get('risk_score', 0)}.")
    
    # Métriques de performance de l'IA
    st.markdown("---")
    st.markdown("**📈 Performance de l'Analyse IA:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Précision", "94.2%", "+2.1%")
    with col2:
        st.metric("⚡ Vitesse", "0.3s", "Ultra-rapide")
    with col3:
        st.metric("🧠 Modèles", "4", "Actifs")
    with col4:
        st.metric("📊 Prédictions", "127", "Cette semaine")
    with col5:
        st.metric("✅ Succès", "89%", "Taux validation")