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


class ReportingModule:
    """Module de Reporting et Analytics Avancés"""
    
    def __init__(self):
        self.initialize_data()
    
    def initialize_data(self):
        """Initialise les données simulées pour démonstration"""
        self.sample_data = self.generate_sample_data()
    
    def render_reporting_dashboard(self, project_id: str = "portfolio"):
        """Interface principale du module reporting"""
        st.title("📄 Reporting Complet")
        st.markdown("*Tableaux de bord exécutifs, analyses multi-projets et intelligence décisionnelle*")
        
        # Configuration
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            report_type = st.selectbox(
                "Type de rapport:",
                ["Dashboard Exécutif", "Analyse Comparative", "Performance Équipe", 
                 "Vue Financière", "Analyse des Risques", "Parties Prenantes"],
                key="reporting_type"
            )
        
        with col2:
            time_period = st.selectbox(
                "Période:",
                ["6 derniers mois", "Année courante", "12 derniers mois", "Toutes données"],
                key="reporting_period"
            )
        
        with col3:
            auto_refresh = st.checkbox("Auto-refresh", key="auto_refresh")
            
        with col4:
            if st.button("🔄 Actualiser", use_container_width=True, key="reporting_refresh"):
                st.success("✅ Données actualisées!")
        
        # Chargement des données
        analytics_data = self.load_portfolio_analytics_data()
        kpis = self.calculate_executive_kpis(analytics_data)
        
        # Alertes critiques
        self.render_critical_alerts(analytics_data, kpis)
        
        # KPIs Exécutifs
        self.render_executive_kpis(kpis)
        
        st.divider()
        
        # Contenu selon le type de rapport
        if report_type == "Dashboard Exécutif":
            self.render_executive_dashboard(analytics_data, kpis)
            
        elif report_type == "Analyse Comparative":
            self.render_comparative_analysis(analytics_data, kpis)
            
        elif report_type == "Performance Équipe":
            self.render_team_performance(analytics_data)
            
        elif report_type == "Vue Financière":
            self.render_financial_overview(analytics_data)
            
        elif report_type == "Analyse des Risques":
            self.render_risk_analysis(analytics_data)
            
        elif report_type == "Parties Prenantes":
            self.render_stakeholder_analysis(analytics_data)
    
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
            - Valeur totale: {analytics_data['total_portfolio_value']:,}€
            
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
            project = {
                'id': f'proj_{i:03d}',
                'name': f'Projet {chr(65+i)}',
                'status': np.random.choice(statuses, p=[0.4, 0.3, 0.2, 0.1]),
                'progress': np.random.randint(20, 100),
                'budget': np.random.randint(50000, 500000),
                'spent': np.random.randint(30000, 400000),
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
        
        return {
            'total_projects': total_projects,
            'active_projects': len([p for p in projects if p['status'] == 'In Progress']),
            'completed_projects': completed_projects,
            'success_rate': (completed_projects / total_projects * 100) if total_projects > 0 else 0,
            'success_vs_industry': 4.2,  # vs benchmark
            'budget_health': ((financial['total_budget'] - financial['spent']) / financial['total_budget'] * 100),
            'quality_health': np.random.uniform(7.8, 8.5),
            'portfolio_roi': financial['roi_achieved'],
            'roi_vs_industry': financial['roi_achieved'] - analytics_data['industry_benchmarks']['avg_roi'],
            'global_health_score': np.random.uniform(82, 88),
            'team_efficiency': np.mean(list(analytics_data['team_performance'].values())),
            'client_satisfaction': np.random.uniform(8.2, 8.8),
            'avg_velocity': np.random.uniform(18, 25),
            'deployment_frequency': np.random.uniform(12, 18),
            'defect_rate': np.random.uniform(1.5, 3.2),
            'value_created_per_euro': financial['total_portfolio_value'] / financial['spent'],
            'avg_delay': np.random.uniform(8, 18),
            'delay_vs_industry': -3.2  # vs benchmark
        }


# Point d'entrée pour l'utilisation
def render_reporting_dashboard(project_id: str = "portfolio"):
    """Point d'entrée principal pour le module reporting"""
    reporting_module = ReportingModule()
    reporting_module.render_reporting_dashboard(project_id)


# Export de la classe pour utilisation externe
__all__ = ['ReportingModule', 'render_reporting_dashboard']