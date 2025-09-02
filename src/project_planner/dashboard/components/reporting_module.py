"""
Module de Reporting Complet - Tableaux de bord exécutifs et analytics avancés
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


class ReportingModule:
    """Module de Reporting et Analytics Avancés"""
    
    def __init__(self):
        self.initialize_data()
    
    def initialize_data(self):
        """Initialise les données simulées pour démonstration"""
        self.sample_data = self.generate_sample_data()
    
    def render_reporting_dashboard(self, project_id: str = "portfolio"):
        """Dashboard reporting restructuré sur le modèle Qualité - Section visuels puis onglets détaillés"""
        
        # Header moderne
        st.header("📄 Rapports Executive Dashboard")
        st.markdown("*Intelligence décisionnelle, analyses multi-projets et communication dirigeants*")
        
        # Charger les données
        analytics_data = self.load_portfolio_analytics_data()
        kpis = self.calculate_executive_kpis(analytics_data)
        reporting_insights = self.generate_reporting_insights(analytics_data, kpis)
        
        # === SECTION 1: MÉTRIQUES RÉSUMÉES ===
        self.render_summary_metrics(kpis)
        
        # === SECTION 2: GRAPHIQUES VISUELS PRINCIPAUX ===
        st.markdown("---")
        self.render_main_visualizations(analytics_data, kpis)
        
        # === SECTION 3: DETAILED ANALYSIS (avec onglets) ===
        st.markdown("---")
        self.render_detailed_reporting_analysis(analytics_data, kpis, reporting_insights)
        
        # === SECTION 4: EXPORTS EN BAS DE PAGE ===
        st.markdown("---")
        self.render_export_section(analytics_data, kpis, reporting_insights)
    
    def render_summary_metrics(self, kpis: Dict[str, Any]):
        """Métriques résumées en haut du dashboard (style KPI)"""
        
        # Calculs pour les indicateurs de santé
        portfolio_health = "✅ Excellent" if kpis['portfolio_health_score'] >= 85 else "⚠️ Attention" if kpis['portfolio_health_score'] >= 70 else "🚨 Critique"
        budget_health = "✅ Sain" if kpis['budget_health'] >= 80 else "⚠️ Attention" if kpis['budget_health'] >= 60 else "🚨 Critique" 
        delivery_health = "✅ Dans les temps" if kpis['on_time_delivery_rate'] >= 85 else "⚠️ Quelques retards" if kpis['on_time_delivery_rate'] >= 70 else "🚨 Retards critiques"
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Santé Portfolio", f"{kpis['portfolio_health_score']:.1f}%", portfolio_health)
        
        with col2:
            st.metric("💰 Budget Utilisé", f"{kpis['budget_consumed']:.1f}%", budget_health)
        
        with col3:
            st.metric("🚀 Projets Livrés", f"{kpis['delivered_projects']}/{kpis['total_projects']}", delivery_health)
        
        with col4:
            st.metric("📈 ROI Moyen", f"{kpis['avg_roi']:.1f}%", f"{'📈' if kpis['avg_roi'] > 15 else '📉'}")
        
        with col5:
            st.metric("⚠️ Risques Actifs", kpis['active_risks'], f"{'✅' if kpis['active_risks'] <= 5 else '🚨'}")

    def render_export_section(self, analytics_data: Dict, kpis: Dict, reporting_insights: Dict):
        """Section d'export des rapports en bas de page"""
        
        st.markdown("### 📥 Export des Rapports")
        st.markdown("*Téléchargez vos données et rapports dans le format de votre choix*")
        
        # Layout en 4 colonnes pour les boutons d'export
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**📊 Format Excel**")
            st.markdown("*Multi-onglets : KPIs, Projets, Finances, Équipes*")
            if st.button("📊 Générer Excel", use_container_width=True, key="export_excel"):
                excel_data = self.generate_excel_export(analytics_data, kpis)
                st.download_button(
                    label="📥 Télécharger Excel",
                    data=excel_data,
                    file_name=f"rapport_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("**📄 Rapport PDF**")
            st.markdown("*Format exécutif avec insights IA et tableaux*")
            if st.button("📄 Générer PDF", use_container_width=True, key="export_pdf"):
                pdf_data = self.generate_pdf_report(analytics_data, kpis, reporting_insights)
                st.download_button(
                    label="📥 Télécharger PDF",
                    data=pdf_data,
                    file_name=f"rapport_executif_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        with col3:
            st.markdown("**📋 Données CSV**")
            st.markdown("*Export brut des projets pour analyse*")
            if st.button("📋 Générer CSV", use_container_width=True, key="export_csv"):
                csv_data = self.generate_csv_export(analytics_data)
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv_data,
                    file_name=f"donnees_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col4:
            st.markdown("**🔧 JSON Complet**")
            st.markdown("*Structure complète avec métadonnées*")
            if st.button("🔧 Générer JSON", use_container_width=True, key="export_json"):
                json_data = self.generate_json_export(analytics_data, kpis, reporting_insights)
                st.download_button(
                    label="📥 Télécharger JSON",
                    data=json_data,
                    file_name=f"portfolio_complet_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Footer informatif
        st.markdown("---")
        st.markdown("*💡 Les fichiers sont générés avec horodatage automatique. Tous les exports incluent les données actuelles du portfolio.*")

    def render_main_visualizations(self, analytics_data: Dict, kpis: Dict[str, Any]):
        """Graphiques visuels principaux (style KPI)"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio Performance Overview
            project_data = analytics_data.get('projects', [])
            if project_data:
                df = pd.DataFrame(project_data)
                
                # Calculer le pourcentage budget consommé
                df['budget_consumed_pct'] = (df['spent'] / df['budget'] * 100).fillna(0)
                
                fig = px.scatter(df, x='budget_consumed_pct', y='progress', size='team_size', 
                               color='status', title="🎯 Performance Portfolio - Budget vs Avancement",
                               hover_data=['name'], color_discrete_map={
                                   'En cours': '#3B82F6', 'Terminé': '#10B981', 'En retard': '#EF4444',
                                   'Completed': '#10B981', 'In Progress': '#3B82F6', 'Delayed': '#EF4444', 'At Risk': '#F59E0B'
                               })
                fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Objectif 80%")
                fig.add_vline(x=75, line_dash="dash", line_color="orange", annotation_text="Seuil Budget")
                fig.update_layout(xaxis_title="Budget Consommé (%)", yaxis_title="Avancement (%)")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # KPIs évolution temporelle
            months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
            delivery_trend = [78, 82, 85, 88, 91, 87]
            budget_trend = [72, 75, 68, 71, 74, 76]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=delivery_trend, mode='lines+markers', 
                                   name='Taux Livraison (%)', line=dict(color='#10B981', width=3)))
            fig.add_trace(go.Scatter(x=months, y=budget_trend, mode='lines+markers', 
                                   name='Respect Budget (%)', line=dict(color='#3B82F6', width=3)))
            
            fig.update_layout(title="📈 Évolution des KPIs Portfolio", xaxis_title="Mois", 
                            yaxis_title="Performance (%)", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    def render_detailed_reporting_analysis(self, analytics_data: Dict, kpis: Dict, reporting_insights: Dict):
        """Section d'analyse détaillée avec onglets (style Qualité)"""
        
        st.markdown("### 📊 Analyses Détaillées")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Dashboard Exécutif",
            "💰 Analyses Financières", 
            "👥 Performance Équipes",
            "⚠️ Gestion des Risques",
            "🤖 IA Reporting"
        ])
        
        with tab1:
            self.render_executive_analysis(analytics_data, kpis)
        
        with tab2:
            self.render_financial_analysis(analytics_data, kpis)
        
        with tab3:
            self.render_team_performance_analysis(analytics_data)
        
        with tab4:
            self.render_risk_management_analysis(analytics_data, kpis)
        
        with tab5:
            self.render_ai_reporting_analysis(analytics_data, kpis, reporting_insights)

    def render_critical_alerts(self, analytics_data: Dict, kpis: Dict):
        """Affiche les alertes critiques"""
        st.markdown("### 🚨 Alertes Critiques")
        
        alerts = []
        
        if kpis['budget_health'] < 60:
            alerts.append(("Budget", f"Santé budgétaire critique: {kpis['budget_health']:.1f}%"))
        
        if kpis['defect_rate'] > 5:
            alerts.append(("Qualité", f"Taux de défauts élevé: {kpis['defect_rate']:.1f}%"))
        
        if len(analytics_data.get('delayed_projects', [])) > 3:
            alerts.append(("Planning", f"{len(analytics_data['delayed_projects'])} projets en retard"))
        
        if alerts:
            for category, message in alerts:
                st.error(f"🚨 **{category}**: {message}")
        else:
            st.success("✅ Aucune alerte critique active")
    
    def render_executive_kpis(self, kpis: Dict):
        """Affiche les KPIs exécutifs"""
        st.markdown("### 📊 Tableau de Bord Exécutif")
        
        # Première ligne de KPIs
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("📁 Projets Total", kpis['total_projects'], delta="+2 ce mois")
        
        with col2:
            st.metric("✅ Taux Succès", f"{kpis['success_rate']:.1f}%", 
                     delta=f"{kpis['success_vs_industry']:+.1f}% vs industrie")
        
        with col3:
            st.metric("💰 Santé Budget", f"{kpis['budget_health']:.1f}%", 
                     delta="+2.3%")
        
        with col4:
            st.metric("🎯 Score Qualité", f"{kpis['quality_health']:.1f}", 
                     delta="📈 Tendance +")
        
        with col5:
            st.metric("📈 ROI Portfolio", f"{kpis['portfolio_roi']:.1f}%", 
                     delta=f"{kpis['roi_vs_industry']:+.1f}% vs industrie")
        
        with col6:
            st.metric("🌟 Santé Globale", f"{kpis['global_health_score']:.1f}/100", 
                     delta="+2.1")
        
        # Deuxième ligne de KPIs
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("👥 Efficacité Équipe", f"{kpis['team_efficiency']:.1f}%", delta="+1.8%")
        
        with col2:
            st.metric("😊 Satisfaction Client", f"{kpis['client_satisfaction']:.1f}/10", delta="📈 +0.3")
        
        with col3:
            st.metric("⚡ Vélocité Moy.", f"{kpis['avg_velocity']:.1f}", delta="Stable")
        
        with col4:
            st.metric("🚀 Freq. Déploiements", f"{kpis['deployment_frequency']:.1f}/mois", delta="+2")
        
        with col5:
            st.metric("🐛 Taux Défauts", f"{kpis['defect_rate']:.1f}%", delta="-0.2%")
        
        with col6:
            st.metric("💰 Valeur/€ Investi", f"{kpis['value_created_per_euro']:.2f}x", delta="+0.15x")
    
    def render_executive_dashboard(self, analytics_data: Dict, kpis: Dict):
        """Affiche le dashboard exécutif complet"""
        st.markdown("#### 📊 Vue d'Ensemble Exécutive")
        
        # Graphique principal
        fig = self.create_executive_overview_chart(analytics_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights automatiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🎯 Points Forts")
            
            strengths = []
            if kpis['portfolio_roi'] > 15:
                strengths.append(f"ROI excellent ({kpis['portfolio_roi']:.1f}%)")
            if kpis['team_efficiency'] > 80:
                strengths.append("Équipe très performante")
            if kpis['defect_rate'] < 3:
                strengths.append("Excellent taux de défauts")
            
            for strength in strengths:
                st.success(f"✅ {strength}")
        
        with col2:
            st.markdown("##### ⚠️ Points d'Attention")
            
            concerns = []
            if kpis['budget_health'] < 70:
                concerns.append("Santé budgétaire à surveiller")
            if kpis['client_satisfaction'] < 8:
                concerns.append("Satisfaction client à améliorer")
            
            for concern in concerns:
                st.warning(f"⚠️ {concern}")
        
        # Résumé exécutif
        with st.expander("📋 Résumé Exécutif Détaillé", expanded=False):
            st.markdown(f"""
            **Santé Globale du Portfolio:** {kpis['global_health_score']:.1f}/100
            
            **Performance vs Industrie:**
            - Taux de succès: {kpis['success_rate']:.1f}% ({kpis['success_vs_industry']:+.1f}% vs industrie)
            - ROI: {kpis['portfolio_roi']:.1f}% ({kpis['roi_vs_industry']:+.1f}% vs industrie)
            
            **État du Portfolio:**
            - {kpis['active_projects']} projets actifs, {kpis['completed_projects']} terminés
            - Valeur totale: {analytics_data['financial_summary']['total_portfolio_value']:,}€
            
            **Équipe & Qualité:**
            - Efficacité équipe: {kpis['team_efficiency']:.1f}%
            - Score qualité: {kpis['quality_health']:.1f}
            - Satisfaction client: {kpis['client_satisfaction']:.1f}/10
            """)
    
    def render_comparative_analysis(self, analytics_data: Dict, kpis: Dict):
        """Analyse comparative avec benchmarks"""
        st.markdown("#### 📊 Analyse Comparative vs Industrie")
        
        # Graphique radar de comparaison
        fig = self.create_benchmark_radar(kpis, analytics_data['industry_benchmarks'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau comparatif
        comparison_df = pd.DataFrame({
            'Métrique': ['Taux de Succès (%)', 'ROI (%)', 'Efficacité Équipe (%)', 'Score Qualité', 'Satisfaction Client'],
            'Notre Performance': [kpis['success_rate'], kpis['portfolio_roi'], kpis['team_efficiency'], 
                                kpis['quality_health'], kpis['client_satisfaction']],
            'Benchmark Industrie': [85, 12, 75, 7.2, 7.8],
            'Écart': [kpis['success_vs_industry'], kpis['roi_vs_industry'], 
                     kpis['team_efficiency'] - 75, kpis['quality_health'] - 7.2, 
                     kpis['client_satisfaction'] - 7.8]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
    
    def render_team_performance(self, analytics_data: Dict):
        """Performance des équipes"""
        st.markdown("#### 👥 Performance des Équipes")
        
        # Graphique de performance par équipe
        team_data = analytics_data.get('team_performance', {})
        
        if team_data:
            fig = px.bar(
                x=list(team_data.keys()),
                y=list(team_data.values()),
                title="Efficacité par Équipe",
                labels={'x': 'Équipe', 'y': 'Efficacité (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Métriques détaillées par équipe
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Équipe Top", "Dev Frontend", "92.5%")
        
        with col2:
            st.metric("📈 Amélioration", "DevOps", "+8.3%")
        
        with col3:
            st.metric("⚡ Vélocité Max", "Backend", "24 pts/sprint")
    
    def render_financial_overview(self, analytics_data: Dict):
        """Vue financière détaillée"""
        st.markdown("#### 💰 Vue Financière")
        
        financial_data = analytics_data['financial_summary']
        
        # Métriques financières
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💼 Budget Total", f"{financial_data['total_budget']:,}€")
        
        with col2:
            st.metric("💸 Dépensé", f"{financial_data['spent']:,}€")
        
        with col3:
            st.metric("💰 Économies", f"{financial_data['savings']:,}€")
        
        with col4:
            st.metric("📊 ROI Réalisé", f"{financial_data['roi_achieved']:.1f}%")
        
        # Graphique d'évolution budgétaire
        fig = self.create_financial_trend_chart(financial_data)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_analysis(self, analytics_data: Dict):
        """Analyse des risques"""
        st.markdown("#### ⚠️ Analyse des Risques")
        
        risks = analytics_data.get('risk_analysis', {})
        
        # Matrice de risques
        fig = self.create_risk_matrix(risks)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top risques
        st.markdown("##### 🚨 Top 5 Risques Critiques")
        
        top_risks = [
            ("Budget", "Dépassement budget projet Alpha", "Élevé", "85%"),
            ("Planning", "Retard livraison Q4", "Moyen", "60%"),
            ("Ressources", "Disponibilité expert sécurité", "Élevé", "75%"),
            ("Technique", "Performance système", "Faible", "25%"),
            ("Marché", "Évolution réglementaire", "Moyen", "45%")
        ]
        
        for i, (category, description, severity, probability) in enumerate(top_risks, 1):
            with st.expander(f"{i}. {category}: {description}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Sévérité:** {severity}")
                with col2:
                    st.write(f"**Probabilité:** {probability}")
                with col3:
                    color = "🔴" if severity == "Élevé" else "🟡" if severity == "Moyen" else "🟢"
                    st.write(f"**Status:** {color}")
    
    def render_stakeholder_analysis(self, analytics_data: Dict):
        """Analyse des parties prenantes"""
        st.markdown("#### 🤝 Analyse des Parties Prenantes")
        
        # Matrice d'influence
        stakeholders = analytics_data.get('stakeholders', {})
        
        fig = self.create_stakeholder_matrix(stakeholders)
        st.plotly_chart(fig, use_container_width=True)
        
        # Engagement des parties prenantes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 👑 Parties Prenantes Clés")
            key_stakeholders = [
                ("CEO", "Champion", "🟢 Très engagé"),
                ("DSI", "Décideur", "🟡 Modéré"),
                ("Utilisateurs", "Bénéficiaires", "🟢 Positif"),
                ("Équipe Projet", "Exécutants", "🟢 Motivé")
            ]
            
            for name, role, status in key_stakeholders:
                st.write(f"**{name}** ({role}): {status}")
        
        with col2:
            st.markdown("##### 📊 Niveau d'Engagement")
            engagement_data = {
                'Très engagé': 35,
                'Modéré': 45,
                'Peu engagé': 15,
                'Résistant': 5
            }
            
            fig_pie = px.pie(
                values=list(engagement_data.values()),
                names=list(engagement_data.keys()),
                title="Répartition Engagement"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def create_executive_overview_chart(self, analytics_data: Dict) -> go.Figure:
        """Crée le graphique de vue d'ensemble exécutive"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Évolution KPIs', 'Répartition Budget', 'Performance Équipes', 'Satisfaction'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Évolution des KPIs dans le temps
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
        roi_data = [12, 14, 16, 15, 17, 18]
        quality_data = [7.2, 7.5, 7.8, 7.6, 8.1, 8.3]
        
        fig.add_trace(
            go.Scatter(x=months, y=roi_data, name="ROI (%)", line=dict(color="blue")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=months, y=quality_data, name="Qualité", line=dict(color="green")),
            row=1, col=1, secondary_y=True
        )
        
        # Répartition budget
        fig.add_trace(
            go.Pie(labels=['Développement', 'Infrastructure', 'Formation', 'Support'],
                   values=[45, 25, 15, 15], name="Budget"),
            row=1, col=2
        )
        
        # Performance équipes
        teams = ['Frontend', 'Backend', 'DevOps', 'QA']
        performance = [92, 88, 85, 90]
        fig.add_trace(
            go.Bar(x=teams, y=performance, name="Performance", marker_color="lightblue"),
            row=2, col=1
        )
        
        # Satisfaction client
        satisfaction_months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
        satisfaction_scores = [7.8, 8.0, 7.9, 8.2, 8.4, 8.5]
        fig.add_trace(
            go.Scatter(x=satisfaction_months, y=satisfaction_scores, 
                      mode='lines+markers', name="Satisfaction",
                      line=dict(color="orange")),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Dashboard Exécutif - Vue d'Ensemble")
        return fig
    
    def create_benchmark_radar(self, kpis: Dict, benchmarks: Dict) -> go.Figure:
        """Crée un graphique radar de comparaison avec benchmarks"""
        categories = ['Succès', 'ROI', 'Efficacité', 'Qualité', 'Satisfaction']
        
        notre_perf = [
            kpis['success_rate'],
            kpis['portfolio_roi'],
            kpis['team_efficiency'],
            kpis['quality_health'] * 10,  # Normalisation
            kpis['client_satisfaction'] * 10
        ]
        
        bench_perf = [85, 12, 75, 72, 78]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=notre_perf,
            theta=categories,
            fill='toself',
            name='Notre Performance',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=bench_perf,
            theta=categories,
            fill='toself',
            name='Benchmark Industrie',
            line_color='red',
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Comparaison vs Benchmarks Industrie"
        )
        
        return fig
    
    def create_financial_trend_chart(self, financial_data: Dict) -> go.Figure:
        """Crée le graphique d'évolution financière"""
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
        budget_planned = [100000, 200000, 300000, 400000, 500000, 600000]
        budget_actual = [95000, 190000, 285000, 390000, 485000, 580000]
        savings = [5000, 15000, 30000, 40000, 55000, 75000]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months, y=budget_planned,
            mode='lines',
            name='Budget Planifié',
            line=dict(color='blue', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=months, y=budget_actual,
            mode='lines+markers',
            name='Budget Réel',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Bar(
            x=months, y=savings,
            name='Économies',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Évolution Budgétaire",
            xaxis_title="Mois",
            yaxis_title="Montant (€)",
            hovermode='x unified'
        )
        
        return fig
    
    def create_risk_matrix(self, risks: Dict) -> go.Figure:
        """Crée la matrice de risques"""
        # Données simulées de risques
        risk_data = pd.DataFrame({
            'Risque': ['Budget Alpha', 'Retard Q4', 'Expert Sécurité', 'Performance', 'Réglementaire'],
            'Probabilité': [85, 60, 75, 25, 45],
            'Impact': [90, 80, 70, 40, 60],
            'Taille': [20, 15, 18, 10, 12]
        })
        
        fig = px.scatter(
            risk_data, 
            x='Probabilité', 
            y='Impact',
            size='Taille',
            color='Impact',
            hover_name='Risque',
            title="Matrice des Risques",
            color_continuous_scale='Reds'
        )
        
        # Ajouter les zones de risque
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=50, y1=50,
            fillcolor="green", opacity=0.1,
            line=dict(width=0)
        )
        fig.add_shape(
            type="rect",
            x0=50, y0=50, x1=100, y1=100,
            fillcolor="red", opacity=0.1,
            line=dict(width=0)
        )
        
        fig.update_layout(
            xaxis_title="Probabilité (%)",
            yaxis_title="Impact (%)"
        )
        
        return fig
    
    def create_stakeholder_matrix(self, stakeholders: Dict) -> go.Figure:
        """Crée la matrice des parties prenantes"""
        stakeholder_data = pd.DataFrame({
            'Partie Prenante': ['CEO', 'DSI', 'Utilisateurs', 'Équipe', 'Finances', 'Juridique'],
            'Influence': [95, 85, 40, 60, 70, 50],
            'Intérêt': [90, 80, 95, 85, 60, 45],
            'Taille': [25, 20, 30, 25, 15, 12]
        })
        
        fig = px.scatter(
            stakeholder_data,
            x='Influence',
            y='Intérêt',
            size='Taille',
            color='Intérêt',
            hover_name='Partie Prenante',
            title="Matrice Influence vs Intérêt",
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_title="Influence (%)",
            yaxis_title="Intérêt (%)"
        )
        
        return fig
    
    def generate_sample_data(self) -> Dict:
        """Génère des données d'exemple réalistes"""
        return {
            'projects': self.generate_project_data(),
            'financial_summary': self.generate_financial_data(),
            'team_performance': self.generate_team_data(),
            'risk_analysis': self.generate_risk_data(),
            'stakeholders': self.generate_stakeholder_data(),
            'industry_benchmarks': self.generate_benchmark_data()
        }
    
    def generate_project_data(self) -> List[Dict]:
        """Génère des données de projets"""
        projects = []
        statuses = ['Completed', 'In Progress', 'Delayed', 'At Risk']
        
        for i in range(15):
            budget = np.random.randint(50000, 500000)
            spent = np.random.randint(30000, min(400000, budget))
            
            project = {
                'id': f'proj_{i:03d}',
                'name': f'Projet {chr(65+i)}',
                'status': np.random.choice(statuses, p=[0.4, 0.3, 0.2, 0.1]),
                'progress': np.random.randint(20, 100),
                'budget': budget,
                'spent': spent,
                'roi': np.random.uniform(8, 25),  # ROI en pourcentage
                'priority': np.random.choice(['Critique', 'Haute', 'Moyenne'], p=[0.2, 0.4, 0.4]),
                'team_size': np.random.randint(3, 12),
                'start_date': datetime.now() - timedelta(days=np.random.randint(30, 365)),
                'end_date': datetime.now() + timedelta(days=np.random.randint(30, 180))
            }
            projects.append(project)
        
        return projects
    
    def generate_financial_data(self) -> Dict:
        """Génère des données financières"""
        return {
            'total_budget': 2500000,
            'spent': 1850000,
            'remaining': 650000,
            'savings': 175000,
            'roi_achieved': 18.5,
            'total_portfolio_value': 4200000
        }
    
    def generate_team_data(self) -> Dict:
        """Génère des données d'équipe"""
        return {
            'Frontend': 92.5,
            'Backend': 88.2,
            'DevOps': 85.7,
            'QA': 90.1,
            'UX/UI': 87.3
        }
    
    def generate_risk_data(self) -> Dict:
        """Génère des données de risques"""
        return {
            'high_risk_count': 3,
            'medium_risk_count': 7,
            'low_risk_count': 12,
            'total_risk_score': 68.5
        }
    
    def generate_stakeholder_data(self) -> Dict:
        """Génère des données de parties prenantes"""
        return {
            'total_stakeholders': 24,
            'engaged_count': 18,
            'neutral_count': 4,
            'resistant_count': 2
        }
    
    def generate_benchmark_data(self) -> Dict:
        """Génère des données de benchmarks"""
        return {
            'avg_project_success_rate': 85,
            'avg_roi': 12,
            'avg_team_utilization': 75,
            'avg_quality_score': 7.2,
            'avg_deployment_frequency': 8.5,
            'avg_delay_days': 15
        }
    
    def load_portfolio_analytics_data(self) -> Dict:
        """Charge les données d'analytics du portfolio"""
        return {
            **self.sample_data,
            'delayed_projects': [p for p in self.sample_data['projects'] if p['status'] == 'Delayed'],
            'high_risk_projects': [p for p in self.sample_data['projects'] if p['status'] == 'At Risk'],
            'industry_benchmarks': self.sample_data['industry_benchmarks']
        }
    
    def calculate_executive_kpis(self, analytics_data: Dict) -> Dict:
        """Calcule les KPIs exécutifs"""
        projects = analytics_data['projects']
        financial = analytics_data['financial_summary']
        
        completed_projects = len([p for p in projects if p['status'] == 'Completed'])
        total_projects = len(projects)
        
        # Calculs KPIs pour la nouvelle structure
        delivered_projects = len([p for p in projects if p['status'] in ['Terminé', 'Completed']])
        on_time_projects = len([p for p in projects if p.get('on_time', True)])
        
        return {
            'total_projects': total_projects,
            'delivered_projects': delivered_projects, 
            'active_projects': len([p for p in projects if p['status'] in ['En cours', 'In Progress']]),
            'portfolio_health_score': np.random.uniform(82, 91),
            'budget_consumed': np.random.uniform(65, 85),
            'budget_health': ((financial['total_budget'] - financial['spent']) / financial['total_budget'] * 100),
            'on_time_delivery_rate': (on_time_projects / total_projects * 100) if total_projects > 0 else 0,
            'avg_roi': financial['roi_achieved'],
            'active_risks': np.random.randint(3, 8),
            'defect_rate': np.random.uniform(1.5, 3.2),
            'team_efficiency': np.mean(list(analytics_data['team_performance'].values())),
            'client_satisfaction': np.random.uniform(8.2, 8.8)
        }

    def generate_reporting_insights(self, analytics_data: Dict, kpis: Dict) -> Dict:
        """Génère des insights IA pour le reporting"""
        return {
            'portfolio_trends': self.analyze_portfolio_trends(analytics_data),
            'risk_predictions': self.predict_project_risks(analytics_data),
            'optimization_recommendations': self.generate_optimization_recommendations(kpis),
            'executive_summary': self.generate_executive_summary(analytics_data, kpis)
        }

    def render_executive_analysis(self, analytics_data: Dict, kpis: Dict):
        """Analyse détaillée du dashboard exécutif"""
        
        st.markdown("### 📈 Dashboard Exécutif - Vue Dirigeants")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio Status Overview
            project_data = analytics_data.get('projects', [])
            if project_data:
                status_counts = {}
                for project in project_data:
                    status = project.get('status', 'Inconnu')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                fig = px.pie(values=list(status_counts.values()), names=list(status_counts.keys()),
                           title="📊 Répartition Statuts Portfolio",
                           color_discrete_map={'En cours': '#3B82F6', 'Terminé': '#10B981', 'En retard': '#EF4444'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Budget vs ROI Analysis
            if project_data:
                df = pd.DataFrame(project_data)
                fig = px.scatter(df, x='budget', y='roi', size='progress',
                               color='priority', title="💰 Budget vs ROI Attendu",
                               hover_data=['name'])
                fig.update_layout(xaxis_title="Budget Total (k€)", yaxis_title="ROI Attendu (%)")
                st.plotly_chart(fig, use_container_width=True)

    def render_financial_analysis(self, analytics_data: Dict, kpis: Dict):
        """Analyse financière détaillée"""
        
        st.markdown("### 💰 Analyses Financières Avancées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Budget Consumption Trend
            months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun']
            budget_planned = [120, 140, 160, 180, 200, 220]
            budget_actual = [125, 138, 175, 185, 195, 235]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=months, y=budget_planned, name='Budget Planifié', 
                               marker_color='#3B82F6'))
            fig.add_trace(go.Bar(x=months, y=budget_actual, name='Budget Consommé', 
                               marker_color='#EF4444'))
            
            fig.update_layout(title="📊 Évolution Budget Planifié vs Réel", 
                            xaxis_title="Mois", yaxis_title="Budget (k€)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROI Analysis by Project Type
            project_types = ['Web App', 'Mobile', 'API', 'Infrastructure', 'IA/ML']
            roi_values = [18.5, 22.1, 15.8, 12.3, 28.7]
            
            fig = px.bar(x=project_types, y=roi_values, title="📈 ROI Moyen par Type de Projet",
                        color=roi_values, color_continuous_scale="RdYlGn")
            fig.update_layout(xaxis_title="Type de Projet", yaxis_title="ROI (%)")
            st.plotly_chart(fig, use_container_width=True)

    def render_team_performance_analysis(self, analytics_data: Dict):
        """Analyse de performance des équipes"""
        
        st.markdown("### 👥 Performance des Équipes")
        
        # Données simulées équipes
        teams_data = [
            {"team": "Frontend", "productivity": 87, "quality": 92, "satisfaction": 4.2, "projects": 8},
            {"team": "Backend", "productivity": 91, "quality": 89, "satisfaction": 4.0, "projects": 12},
            {"team": "DevOps", "productivity": 85, "quality": 94, "satisfaction": 4.3, "projects": 6},
            {"team": "QA", "productivity": 89, "quality": 96, "satisfaction": 4.1, "projects": 10},
            {"team": "IA/ML", "productivity": 82, "quality": 88, "satisfaction": 4.4, "projects": 4}
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance Radar Chart
            df_teams = pd.DataFrame(teams_data)
            
            fig = go.Figure()
            
            for _, team_data in df_teams.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[team_data['productivity'], team_data['quality'], team_data['satisfaction']*20],
                    theta=['Productivité', 'Qualité', 'Satisfaction'],
                    fill='toself',
                    name=team_data['team']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="🎯 Performance Équipes - Vue Radar"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Team Workload
            fig = px.bar(df_teams, x='team', y='projects', title="📊 Charge de Travail par Équipe",
                        color='productivity', color_continuous_scale="Viridis")
            fig.update_layout(xaxis_title="Équipe", yaxis_title="Nombre de Projets")
            st.plotly_chart(fig, use_container_width=True)

    def render_risk_management_analysis(self, analytics_data: Dict, kpis: Dict):
        """Analyse de gestion des risques"""
        
        st.markdown("### ⚠️ Analyse des Risques Portfolio")
        
        # Données risques simulées
        risks_data = [
            {"category": "Technique", "count": 12, "severity": "Haute", "impact": 85},
            {"category": "Budget", "count": 8, "severity": "Moyenne", "impact": 70},
            {"category": "Planning", "count": 15, "severity": "Haute", "impact": 90},
            {"category": "Ressources", "count": 6, "severity": "Critique", "impact": 95},
            {"category": "Externe", "count": 4, "severity": "Moyenne", "impact": 60}
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Distribution
            df_risks = pd.DataFrame(risks_data)
            fig = px.bar(df_risks, x='category', y='count', color='severity',
                        title="📊 Distribution des Risques par Catégorie",
                        color_discrete_map={'Critique': '#EF4444', 'Haute': '#F59E0B', 'Moyenne': '#10B981'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk Impact Matrix
            fig = px.scatter(df_risks, x='count', y='impact', size='count', color='severity',
                           title="🎯 Matrice Impact-Fréquence des Risques", hover_data=['category'])
            fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Seuil Critique")
            st.plotly_chart(fig, use_container_width=True)

    def render_ai_reporting_analysis(self, analytics_data: Dict, kpis: Dict, reporting_insights: Dict):
        """Analyse IA avancée pour le reporting - Similaire au module qualité"""
        
        st.markdown("### 🤖 Intelligence Artificielle - Analyses Prédictives")
        
        with st.expander("🔮 Insights IA Portfolio", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**💡 Analyses Prédictives IA:**")
                
                # Génération d'insights intelligents
                portfolio_insights = []
                
                if kpis['portfolio_health_score'] < 75:
                    portfolio_insights.append("🚨 **Santé Portfolio Critique**: Risque de dégradation majeure détectée")
                else:
                    portfolio_insights.append("✅ **Portfolio Performant**: Trajectoire positive confirmée par l'IA")
                
                if kpis['budget_health'] < 70:
                    portfolio_insights.append("💰 **Dérive Budgétaire**: Surconsommation détectée sur 3+ projets")
                else:
                    portfolio_insights.append("🎯 **Budget Maîtrisé**: Gestion financière optimale")
                
                # Analyse des projets en retard
                delayed_projects = len([p for p in analytics_data.get('projects', []) if p.get('status') == 'En retard'])
                if delayed_projects > 2:
                    portfolio_insights.append(f"⏰ **{delayed_projects} Projets en Retard**: Impact planning significatif")
                
                # Analyse ROI
                if kpis['avg_roi'] < 15:
                    portfolio_insights.append("📉 **ROI Sous-Optimal**: Rentabilité portfolio à améliorer")
                else:
                    portfolio_insights.append("📈 **ROI Excellent**: Performance financière supérieure aux standards")
                
                for insight in portfolio_insights:
                    st.write(f"• {insight}")
                
                # Recommandations IA Reporting
                st.markdown("**🎯 Recommandations Stratégiques IA:**")
                
                reporting_recommendations = []
                
                if kpis['portfolio_health_score'] < 75:
                    critical_projects = [p['name'] for p in analytics_data.get('projects', []) if p.get('status') == 'En retard'][:2]
                    reporting_recommendations.append({
                        "priority": "🔴 URGENT",
                        "action": f"Intervention immédiate sur {', '.join(critical_projects) if critical_projects else 'projets critiques'}",
                        "impact": f"Récupération de {100-kpis['portfolio_health_score']:.1f}% de performance",
                        "confidence": "94%"
                    })
                
                if kpis['budget_health'] < 70:
                    reporting_recommendations.append({
                        "priority": "🟠 HAUTE",
                        "action": "Audit budgétaire et replanification financière",
                        "impact": f"Économie potentielle de {(100-kpis['budget_health'])*2:.0f}k€",
                        "confidence": "87%"
                    })
                
                if kpis['avg_roi'] < 15:
                    reporting_recommendations.append({
                        "priority": "🟡 MOYENNE",
                        "action": "Optimisation portfolio - Focus projets haute valeur",
                        "impact": f"Augmentation ROI potentielle: +{20-kpis['avg_roi']:.1f}%",
                        "confidence": "81%"
                    })
                
                for rec in reporting_recommendations[:3]:
                    st.markdown(f"""
                    **{rec['priority']}** - *Confiance: {rec['confidence']}*
                    - 🎯 **Action**: {rec['action']}
                    - 📈 **Impact**: {rec['impact']}
                    """)
            
            with col2:
                st.markdown("**📊 Prédictions Portfolio IA:**")
                
                # Prédictions futures
                months_future = ['Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
                current_trend = kpis['portfolio_health_score']
                predicted_health = []
                
                for i in range(6):
                    # Simulation prédiction basée sur tendances actuelles
                    trend_factor = 1 + (kpis['budget_health'] - 80) / 1000 if kpis['budget_health'] > 0 else 1  # Ajustement basé budget
                    noise = np.random.normal(0, 2)  # Variabilité
                    predicted_value = min(100, max(0, current_trend * trend_factor + noise))
                    predicted_health.append(predicted_value)
                    current_trend = predicted_value
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=months_future, y=predicted_health, mode='lines+markers',
                                       name='Santé Portfolio Prédite', line=dict(color='#8B5CF6', width=3)))
                
                # Zone de confiance
                upper_bound = [min(100, val + 5) for val in predicted_health]
                lower_bound = [max(0, val - 5) for val in predicted_health]
                
                fig.add_trace(go.Scatter(x=months_future, y=upper_bound, fill=None, mode='lines',
                                       line_color='rgba(139,92,246,0)', showlegend=False))
                fig.add_trace(go.Scatter(x=months_future, y=lower_bound, fill='tonexty', mode='lines',
                                       line_color='rgba(139,92,246,0)', name='Zone de Confiance',
                                       fillcolor='rgba(139,92,246,0.2)'))
                
                fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Objectif Excellence")
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Seuil Attention")
                
                fig.update_layout(title="🔮 Prédiction Santé Portfolio (6 mois)", 
                                xaxis_title="Mois", yaxis_title="Performance (%)",
                                hovermode='x unified')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Score de prédictibilité
                predictability_score = min(95, kpis['portfolio_health_score'] * 1.1)
                st.metric("🎯 Fiabilité Prédictions", f"{predictability_score:.1f}%", 
                         "IA Confiance Élevée" if predictability_score > 85 else "Incertitude Modérée")

    def analyze_portfolio_trends(self, analytics_data: Dict) -> Dict:
        """Analyse des tendances portfolio"""
        return {"trend": "positive", "confidence": 0.87}
    
    def predict_project_risks(self, analytics_data: Dict) -> Dict:
        """Prédictions des risques projets"""
        return {"high_risk_projects": 2, "confidence": 0.91}
    
    def generate_optimization_recommendations(self, kpis: Dict) -> List[str]:
        """Génère des recommandations d'optimisation"""
        return ["Optimiser allocation ressources", "Revoir priorités portfolio"]
    
    def generate_executive_summary(self, analytics_data: Dict, kpis: Dict) -> str:
        """Génère un résumé exécutif IA"""
        return f"Portfolio santé: {kpis['portfolio_health_score']:.1f}% - Performance satisfaisante"

    def generate_excel_export(self, analytics_data: Dict, kpis: Dict) -> bytes:
        """Génère un export Excel complet du portfolio"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Onglet KPIs
            kpis_df = pd.DataFrame([kpis]).T
            kpis_df.columns = ['Valeur']
            kpis_df.to_excel(writer, sheet_name='KPIs Portfolio', index_label='Métrique')
            
            # Onglet Projets
            projects_df = pd.DataFrame(analytics_data['projects'])
            projects_df.to_excel(writer, sheet_name='Projets', index=False)
            
            # Onglet Financier
            financial_df = pd.DataFrame([analytics_data['financial_summary']]).T
            financial_df.columns = ['Valeur']
            financial_df.to_excel(writer, sheet_name='Finances', index_label='Métrique')
            
            # Onglet Performance Équipes
            teams_df = pd.DataFrame.from_dict(analytics_data['team_performance'], orient='index', columns=['Performance'])
            teams_df.to_excel(writer, sheet_name='Équipes', index_label='Équipe')
            
        output.seek(0)
        return output.read()

    def generate_csv_export(self, analytics_data: Dict) -> str:
        """Génère un export CSV des projets"""
        projects_df = pd.DataFrame(analytics_data['projects'])
        return projects_df.to_csv(index=False, encoding='utf-8')

    def generate_json_export(self, analytics_data: Dict, kpis: Dict, reporting_insights: Dict) -> str:
        """Génère un export JSON complet"""
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'PlannerIA Reporting Module',
                'version': '2.0'
            },
            'kpis': kpis,
            'analytics_data': analytics_data,
            'insights': reporting_insights
        }
        
        # Convertir les dates et autres objets non sérialisables
        def json_serial(obj):
            if isinstance(obj, (datetime, np.datetime64)):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        return json.dumps(export_data, indent=2, default=json_serial, ensure_ascii=False)

    def generate_pdf_report(self, analytics_data: Dict, kpis: Dict, reporting_insights: Dict) -> bytes:
        """Génère un rapport PDF executive avec mise en page professionnelle"""
        buffer = io.BytesIO()
        
        try:
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Style personnalisé pour le titre
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            
            # En-tête du rapport
            story.append(Paragraph("RAPPORT EXECUTIVE PORTFOLIO", title_style))
            story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Section KPIs Principaux
            story.append(Paragraph("INDICATEURS CLÉS DE PERFORMANCE", styles['Heading2']))
            
            kpis_data = [
                ['Métrique', 'Valeur', 'Status'],
                ['Santé Portfolio', f"{kpis['portfolio_health_score']:.1f}%", 
                 'Excellent' if kpis['portfolio_health_score'] >= 85 else 'Attention'],
                ['Budget Utilisé', f"{kpis['budget_consumed']:.1f}%", 
                 'Sain' if kpis['budget_health'] >= 80 else 'Critique'],
                ['Projets Livrés', f"{kpis['delivered_projects']}/{kpis['total_projects']}", 
                 'Dans les temps' if kpis['on_time_delivery_rate'] >= 85 else 'Retards'],
                ['ROI Moyen', f"{kpis['avg_roi']:.1f}%", 
                 'Excellent' if kpis['avg_roi'] > 15 else 'À améliorer'],
                ['Risques Actifs', str(kpis['active_risks']), 
                 'Maîtrisés' if kpis['active_risks'] <= 5 else 'Critiques']
            ]
            
            kpis_table = Table(kpis_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            kpis_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(kpis_table)
            story.append(Spacer(1, 20))
            
            # Section Projets Critiques
            story.append(Paragraph("PROJETS PRIORITAIRES", styles['Heading2']))
            
            projects = analytics_data['projects']
            critical_projects = [p for p in projects if p.get('priority') == 'Critique'][:5]
            
            if critical_projects:
                projects_data = [['Projet', 'Statut', 'Budget (k€)', 'Avancement', 'ROI (%)']]
                for project in critical_projects:
                    projects_data.append([
                        project['name'],
                        project['status'], 
                        f"{project['budget']/1000:.0f}",
                        f"{project['progress']}%",
                        f"{project['roi']:.1f}"
                    ])
                
                projects_table = Table(projects_data, colWidths=[2*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
                projects_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(projects_table)
                story.append(Spacer(1, 20))
            
            # Section Insights IA
            story.append(Paragraph("ANALYSES PRÉDICTIVES IA", styles['Heading2']))
            
            # Insights générés par l'IA
            ai_insights = []
            if kpis['portfolio_health_score'] >= 85:
                ai_insights.append("✓ Portfolio en excellente santé - Trajectoire positive confirmée")
            else:
                ai_insights.append("⚠ Santé portfolio à surveiller - Actions correctives recommandées")
                
            if kpis['budget_health'] >= 80:
                ai_insights.append("✓ Gestion budgétaire optimale - Maîtrise des coûts")
            else:
                ai_insights.append("⚠ Dérive budgétaire détectée - Audit recommandé")
            
            for insight in ai_insights:
                story.append(Paragraph(f"• {insight}", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Footer
            story.append(Paragraph("---", styles['Normal']))
            story.append(Paragraph("Rapport généré par PlannerIA - Intelligence Artificielle de Gestion de Projet", styles['Italic']))
            
            # Construction du PDF
            doc.build(story)
            
        except ImportError:
            # Fallback si reportlab n'est pas disponible
            buffer = io.BytesIO()
            simple_report = f"""
RAPPORT EXECUTIVE PORTFOLIO
Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}

INDICATEURS CLÉS:
- Santé Portfolio: {kpis['portfolio_health_score']:.1f}%
- Budget Utilisé: {kpis['budget_consumed']:.1f}%
- Projets Livrés: {kpis['delivered_projects']}/{kpis['total_projects']}
- ROI Moyen: {kpis['avg_roi']:.1f}%
- Risques Actifs: {kpis['active_risks']}

RÉSUMÉ EXÉCUTIF:
{self.generate_executive_summary(analytics_data, kpis)}
"""
            buffer.write(simple_report.encode('utf-8'))
        
        buffer.seek(0)
        return buffer.read()


# Point d'entrée pour l'utilisation
def render_reporting_dashboard(project_id: str = "portfolio"):
    """Point d'entrée principal pour le module reporting"""
    reporting_module = ReportingModule()
    reporting_module.render_reporting_dashboard(project_id)


# Export de la classe pour utilisation externe
__all__ = ['ReportingModule', 'render_reporting_dashboard']