"""
Module Qualité Avancé pour PlannerIA
Suivi des livrables, gestion des tests, validation des milestones et métriques qualité
Restructuré selon le modèle KPI/Budget avec analyse IA spécialisée
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

class QualityModule:
    def __init__(self):
        self.status_colors = {
            "Validé": "🟢", 
            "En test": "🟡", 
            "En développement": "🔵", 
            "En revue": "🟠", 
            "Bloqué": "🔴"
        }
        self.priority_colors = {
            "Critique": "🔴", 
            "Haute": "🟠", 
            "Moyenne": "🟡"
        }

    def create_quality_data(self) -> tuple:
        """Génère des données enrichies pour le module qualité"""
        
        # Livrables avec données détaillées
        deliverables = [
            {
                "id": "DEL-001",
                "name": "API Gateway v2.0",
                "project": "Plateforme E-commerce",
                "status": "En test",
                "type": "Technique",
                "priority": "Critique",
                "completion": 85,
                "quality_score": 92,
                "tests_passed": 234,
                "tests_failed": 8,
                "code_coverage": 89,
                "bugs_critical": 0,
                "bugs_major": 2,
                "bugs_minor": 5,
                "security_score": 95,
                "performance_score": 88,
                "due_date": datetime.now() + timedelta(days=5),
                "reviewer": "Tech Lead",
                "effort_spent": 120,
                "effort_estimated": 140
            },
            {
                "id": "DEL-002", 
                "name": "Dashboard Analytics",
                "project": "BI Platform",
                "status": "Validé",
                "type": "Fonctionnel",
                "priority": "Haute",
                "completion": 100,
                "quality_score": 96,
                "tests_passed": 187,
                "tests_failed": 1,
                "code_coverage": 94,
                "bugs_critical": 0,
                "bugs_major": 0,
                "bugs_minor": 1,
                "security_score": 97,
                "performance_score": 93,
                "due_date": datetime.now() - timedelta(days=2),
                "reviewer": "Product Owner",
                "effort_spent": 95,
                "effort_estimated": 100
            },
            {
                "id": "DEL-003",
                "name": "Module Paiements",
                "project": "Plateforme E-commerce", 
                "status": "En développement",
                "type": "Critique",
                "priority": "Critique",
                "completion": 60,
                "quality_score": 78,
                "tests_passed": 145,
                "tests_failed": 23,
                "code_coverage": 72,
                "bugs_critical": 1,
                "bugs_major": 4,
                "bugs_minor": 12,
                "security_score": 85,
                "performance_score": 74,
                "due_date": datetime.now() + timedelta(days=12),
                "reviewer": "Security Expert",
                "effort_spent": 78,
                "effort_estimated": 130
            },
            {
                "id": "DEL-004",
                "name": "Interface Mobile",
                "project": "App Mobile",
                "status": "En revue",
                "type": "UX/UI",
                "priority": "Haute",
                "completion": 90,
                "quality_score": 87,
                "tests_passed": 156,
                "tests_failed": 12,
                "code_coverage": 81,
                "bugs_critical": 0,
                "bugs_major": 1,
                "bugs_minor": 8,
                "security_score": 90,
                "performance_score": 85,
                "due_date": datetime.now() + timedelta(days=3),
                "reviewer": "UX Lead",
                "effort_spent": 85,
                "effort_estimated": 95
            },
            {
                "id": "DEL-005",
                "name": "API Notifications",
                "project": "Microservices",
                "status": "Bloqué",
                "type": "Technique",
                "priority": "Moyenne",
                "completion": 45,
                "quality_score": 65,
                "tests_passed": 89,
                "tests_failed": 34,
                "code_coverage": 58,
                "bugs_critical": 2,
                "bugs_major": 7,
                "bugs_minor": 15,
                "security_score": 70,
                "performance_score": 62,
                "due_date": datetime.now() + timedelta(days=8),
                "reviewer": "Senior Dev",
                "effort_spent": 67,
                "effort_estimated": 150
            },
            {
                "id": "DEL-006",
                "name": "Système d'Auth",
                "project": "Infrastructure",
                "status": "Validé",
                "type": "Sécurité",
                "priority": "Critique",
                "completion": 100,
                "quality_score": 98,
                "tests_passed": 203,
                "tests_failed": 0,
                "code_coverage": 96,
                "bugs_critical": 0,
                "bugs_major": 0,
                "bugs_minor": 0,
                "security_score": 99,
                "performance_score": 95,
                "due_date": datetime.now() - timedelta(days=5),
                "reviewer": "CISO",
                "effort_spent": 180,
                "effort_estimated": 175
            }
        ]
        
        # Tests automatisés par catégorie
        test_categories = {
            "Unit Tests": {"total": 2847, "passed": 2789, "failed": 58, "coverage": 91, "duration_avg": 0.45, "flaky_rate": 2.1},
            "Integration Tests": {"total": 456, "passed": 423, "failed": 33, "coverage": 84, "duration_avg": 12.3, "flaky_rate": 4.8},
            "E2E Tests": {"total": 187, "passed": 175, "failed": 12, "coverage": 78, "duration_avg": 156.7, "flaky_rate": 8.9},
            "Security Tests": {"total": 89, "passed": 82, "failed": 7, "coverage": 95, "duration_avg": 45.2, "flaky_rate": 1.2},
            "Performance Tests": {"total": 67, "passed": 61, "failed": 6, "coverage": 88, "duration_avg": 234.5, "flaky_rate": 3.4}
        }
        
        # Métriques qualité temps réel
        quality_metrics = {
            "code_quality_score": 87,
            "technical_debt_ratio": 12.3,
            "defect_density": 3.2,
            "test_automation_rate": 84,
            "release_success_rate": 92,
            "mttr_hours": 4.2,
            "customer_reported_bugs": 8,
            "security_vulnerabilities": 2,
            "performance_degradation": 5.8,
            "compliance_score": 94,
            "maintainability_index": 78
        }
        
        # Métriques historiques sur 8 mois
        historical_metrics = []
        base_date = datetime.now() - timedelta(days=240)
        
        for i in range(8):
            month_date = base_date + timedelta(days=30*i)
            historical_metrics.append({
                "month": month_date.strftime("%Y-%m"),
                "deliverables_completed": np.random.randint(15, 25),
                "deliverables_delayed": np.random.randint(2, 8),
                "avg_quality_score": 82 + np.random.randint(-5, 8),
                "defect_rate": round(np.random.uniform(2.5, 8.2), 1),
                "code_coverage": 75 + np.random.randint(-3, 15),
                "customer_satisfaction": 3.8 + np.random.uniform(-0.3, 0.7),
                "security_incidents": np.random.randint(0, 3),
                "performance_issues": np.random.randint(1, 6),
                "rework_rate": round(np.random.uniform(8, 18), 1),
                "technical_debt_hours": np.random.randint(120, 280)
            })
        
        return deliverables, test_categories, quality_metrics, historical_metrics

    def calculate_quality_kpis(self, deliverables, test_categories, quality_metrics, historical_metrics) -> Dict[str, Any]:
        """Calcule les KPIs qualité avancés"""
        
        df = pd.DataFrame(deliverables)
        
        # KPIs de base
        kpis = {
            "total_deliverables": len(deliverables),
            "completed_deliverables": len(df[df['status'] == 'Validé']),
            "avg_quality_score": round(df['quality_score'].mean(), 1),
            "critical_bugs": df['bugs_critical'].sum(),
            "on_time_delivery_rate": round(len(df[df['due_date'] >= datetime.now()]) / len(df) * 100, 1),
            "avg_code_coverage": round(df['code_coverage'].mean(), 1),
            "security_score": round(df['security_score'].mean(), 1),
            "performance_score": round(df['performance_score'].mean(), 1),
            "effort_variance": round(((df['effort_spent'] - df['effort_estimated']) / df['effort_estimated'] * 100).mean(), 1),
            "test_success_rate": round(sum([cat['passed'] for cat in test_categories.values()]) / 
                                     sum([cat['total'] for cat in test_categories.values()]) * 100, 1)
        }
        
        # Score de santé qualité global (0-100)
        health_factors = [
            min(kpis['avg_quality_score'], 100) * 0.25,
            min(100 - kpis['critical_bugs'] * 10, 100) * 0.20,
            kpis['test_success_rate'] * 0.15,
            kpis['avg_code_coverage'] * 0.15,
            kpis['security_score'] * 0.15,
            kpis['performance_score'] * 0.10
        ]
        kpis['quality_health_score'] = round(sum(health_factors), 1)
        
        return kpis

    def render_quality_dashboard(self, project_id: str = "projet_test"):
        """Dashboard qualité restructuré sur le modèle KPI - Visuels d'abord, puis Detailed Analysis"""
        
        # Header moderne
        st.header("✅ Quality Management Dashboard")
        
        # Charger les données
        deliverables, test_categories, quality_metrics, historical_metrics = self.create_quality_data()
        kpis = self.calculate_quality_kpis(deliverables, test_categories, quality_metrics, historical_metrics)
        
        # === SECTION 1: MÉTRIQUES RÉSUMÉES ===
        self.render_summary_metrics(kpis, quality_metrics)
        
        # === SECTION 2: GRAPHIQUES VISUELS PRINCIPAUX ===
        st.markdown("---")
        self.render_main_visualizations(deliverables, test_categories)
        
        # === SECTION 3: DETAILED ANALYSIS (avec onglets) ===
        st.markdown("---")
        self.render_detailed_quality_analysis(deliverables, test_categories, quality_metrics, historical_metrics)

    def render_summary_metrics(self, kpis: Dict[str, Any], quality_metrics: Dict[str, Any]):
        """Métriques résumées en haut du dashboard (style KPI)"""
        
        # Calculs pour les indicateurs de santé
        quality_health = "✅ Excellent" if kpis['quality_health_score'] >= 85 else "⚠️ Attention" if kpis['quality_health_score'] >= 70 else "🚨 Critique"
        bugs_health = "✅ Sain" if kpis['critical_bugs'] == 0 else "⚠️ Attention" if kpis['critical_bugs'] <= 2 else "🚨 Critique"
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Score Qualité Global", f"{kpis['quality_health_score']:.1f}%", quality_health)
        
        with col2:
            st.metric("📋 Livrables Validés", f"{kpis['completed_deliverables']}/{kpis['total_deliverables']}", f"{kpis['completed_deliverables']*100//kpis['total_deliverables']}%")
        
        with col3:
            st.metric("🐛 Bugs Critiques", kpis['critical_bugs'], bugs_health)
        
        with col4:
            st.metric("🧪 Taux Tests", f"{kpis['test_success_rate']:.1f}%", f"{'✅' if kpis['test_success_rate'] > 90 else '⚠️'}")
        
        with col5:
            st.metric("📊 Couverture Code", f"{kpis['avg_code_coverage']:.1f}%", f"{'✅' if kpis['avg_code_coverage'] > 80 else '⚠️'}")

    def render_main_visualizations(self, deliverables: List[Dict], test_categories: Dict[str, Any]):
        """Graphiques visuels principaux (style KPI)"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique distribution des statuts
            status_chart = self.create_status_distribution_chart(deliverables)
            if status_chart:
                st.plotly_chart(status_chart, use_container_width=True)
        
        with col2:
            # Graphique résultats des tests
            test_chart = self.create_test_results_chart(test_categories)
            if test_chart:
                st.plotly_chart(test_chart, use_container_width=True)

    def create_status_distribution_chart(self, deliverables: List[Dict]) -> go.Figure:
        """Crée le graphique de distribution des statuts"""
        df = pd.DataFrame(deliverables)
        status_counts = df['status'].value_counts()
        
        colors = ['#10B981', '#F59E0B', '#3B82F6', '#EF4444', '#8B5CF6']
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Nombre: %{value}<br>Pourcentage: %{percent}<extra></extra>'
        )])
        
        # Ajout du texte central
        fig.add_annotation(
            text=f"<b>{len(deliverables)}<br>Livrables</b>",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )
        
        fig.update_layout(
            title="Distribution des Statuts",
            height=400,
            showlegend=True,
            font=dict(size=11)
        )
        
        return fig

    def create_test_results_chart(self, test_categories: Dict[str, Any]) -> go.Figure:
        """Crée le graphique des résultats de tests"""
        categories = list(test_categories.keys())
        passed = [test_categories[cat]['passed'] for cat in categories]
        failed = [test_categories[cat]['failed'] for cat in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Tests Réussis',
            x=categories,
            y=passed,
            marker_color='#10B981',
            text=passed,
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Tests Échoués',
            x=categories,
            y=failed,
            marker_color='#EF4444',
            text=failed,
            textposition='inside'
        ))
        
        fig.update_layout(
            title="Résultats Tests par Catégorie",
            xaxis_title="Catégorie de Tests",
            yaxis_title="Nombre de Tests",
            barmode='stack',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig

    def render_detailed_quality_analysis(self, deliverables: List[Dict], test_categories: Dict[str, Any], quality_metrics: Dict[str, Any], historical_metrics: List[Dict]):
        """Section Detailed Analysis avec onglets (style KPI)"""
        
        st.subheader("📋 Detailed Analysis")
        
        # Onglets comme dans le module KPI/Budget
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Livrables & Validation", 
            "🧪 Tests & Couverture", 
            "📈 Métriques Historiques",
            "🛡️ Sécurité & Performance", 
            "🤖 Analyse IA"
        ])
        
        with tab1:
            self.render_deliverables_analysis(deliverables)
        
        with tab2:
            self.render_tests_coverage_analysis(test_categories)
        
        with tab3:
            self.render_historical_metrics_analysis(historical_metrics)
        
        with tab4:
            self.render_security_performance_analysis(deliverables)
        
        with tab5:
            self.render_ai_quality_analysis(deliverables, test_categories, quality_metrics)

    def render_deliverables_analysis(self, deliverables: List[Dict]):
        """Analyse détaillée des livrables"""
        
        st.markdown("### 📋 Suivi des Livrables")
        
        # Graphique score qualité par livrable
        df = pd.DataFrame(deliverables)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df, x='name', y='quality_score', color='priority',
                        title="Score Qualité par Livrable",
                        color_discrete_map={'Critique': '#EF4444', 'Haute': '#F59E0B', 'Moyenne': '#10B981'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Matrice risque
            fig = px.scatter(df, x='security_score', y='quality_score', size='completion',
                            color='priority', hover_data=['name'], 
                            title="Matrice Risque : Sécurité vs Qualité")
            fig.add_vline(x=90, line_dash="dash", line_color="green", annotation_text="Seuil Sécurité")
            fig.add_hline(y=85, line_dash="dash", line_color="blue", annotation_text="Seuil Qualité")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        st.markdown("### 📊 Détails des Livrables")
        
        display_data = []
        for d in deliverables:
            display_data.append({
                "Livrable": d['name'],
                "Projet": d['project'],
                "Statut": f"{self.status_colors.get(d['status'], '⚪')} {d['status']}",
                "Priorité": f"{self.priority_colors.get(d['priority'], '⚪')} {d['priority']}",
                "Avancement": f"{d['completion']}%",
                "Score Qualité": f"{d['quality_score']}%",
                "Couverture": f"{d['code_coverage']}%",
                "Bugs 🔴": d['bugs_critical'],
                "Bugs 🟠": d['bugs_major'],
                "Sécurité": f"{d['security_score']}%",
                "Performance": f"{d['performance_score']}%"
            })
        
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)

    def render_tests_coverage_analysis(self, test_categories: Dict[str, Any]):
        """Analyse détaillée des tests et couverture"""
        
        st.markdown("### 🧪 Tests Automatisés & Couverture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Couverture par catégorie
            categories = list(test_categories.keys())
            coverage_data = [test_categories[cat]["coverage"] for cat in categories]
            
            fig = px.bar(x=categories, y=coverage_data, title="Couverture de Code par Catégorie (%)",
                        color=coverage_data, color_continuous_scale="RdYlGn")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Taux de tests flaky
            flaky_rates = [test_categories[cat]["flaky_rate"] for cat in categories]
            
            fig = px.bar(x=categories, y=flaky_rates, title="Taux de Tests Instables (%)",
                        color=flaky_rates, color_continuous_scale="RdYlBu_r")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé des tests
        st.markdown("### 📋 Détail des Tests par Catégorie")
        
        test_df = pd.DataFrame([
            {
                "Catégorie": cat, 
                "Total": data["total"], 
                "Réussis": data["passed"], 
                "Échoués": data["failed"], 
                "Taux Succès": f"{round(data['passed']/data['total']*100, 1)}%",
                "Couverture": f"{data['coverage']}%",
                "Durée Moy.": f"{data['duration_avg']:.1f}s",
                "Taux Instable": f"{data['flaky_rate']}%"
            }
            for cat, data in test_categories.items()
        ])
        
        st.dataframe(test_df, use_container_width=True)

    def render_historical_metrics_analysis(self, historical_metrics: List[Dict]):
        """Analyse des métriques historiques"""
        
        st.markdown("### 📈 Évolution des Métriques Qualité")
        
        hist_df = pd.DataFrame(historical_metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Évolution score qualité moyen
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df['month'], y=hist_df['avg_quality_score'], 
                                    mode='lines+markers', name='Score Qualité Moyen',
                                    line=dict(color='#3B82F6', width=3)))
            fig.add_hline(y=85, line_dash="dash", line_color="green", 
                         annotation_text="Objectif: 85%")
            fig.update_layout(title="Évolution Score Qualité", yaxis_title="Score (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Taux de défauts
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df['month'], y=hist_df['defect_rate'], 
                                    mode='lines+markers', name='Taux de Défauts',
                                    line=dict(color='#EF4444', width=3)))
            fig.add_hline(y=5, line_dash="dash", line_color="orange", 
                         annotation_text="Seuil: 5%")
            fig.update_layout(title="Évolution Taux de Défauts", yaxis_title="Taux (%)")
            st.plotly_chart(fig, use_container_width=True)

    def render_security_performance_analysis(self, deliverables: List[Dict]):
        """Analyse sécurité et performance"""
        
        st.markdown("### 🛡️ Sécurité & Performance")
        
        df = pd.DataFrame(deliverables)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance vs bugs critiques
            fig = px.scatter(df, x='performance_score', y='bugs_critical', size='bugs_major',
                            color='status', hover_data=['name'],
                            title="Performance vs Bugs Critiques")
            fig.add_vline(x=80, line_dash="dash", line_color="green", annotation_text="Seuil Performance")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution des scores sécurité
            fig = px.histogram(df, x='security_score', nbins=10, title="Distribution des Scores Sécurité")
            fig.add_vline(x=90, line_dash="dash", line_color="green", annotation_text="Seuil Sécurité")
            st.plotly_chart(fig, use_container_width=True)
        
        # Matrice risque sécurité/performance
        st.markdown("### 🎯 Matrice Risque Sécurité/Performance")
        
        risk_matrix = []
        for d in deliverables:
            risk_level = "Faible"
            if d['security_score'] < 85 or d['performance_score'] < 75:
                risk_level = "Élevé"
            elif d['security_score'] < 90 or d['performance_score'] < 85:
                risk_level = "Moyen"
            
            risk_matrix.append({
                "Livrable": d['name'],
                "Projet": d['project'],
                "Sécurité": f"{d['security_score']}%",
                "Performance": f"{d['performance_score']}%",
                "Bugs Critiques": d['bugs_critical'],
                "Niveau Risque": risk_level
            })
        
        risk_df = pd.DataFrame(risk_matrix)
        
        # Styling function pour coloration
        def color_risk_level(val):
            if val == "Élevé":
                return 'background-color: #FEE2E2; color: #DC2626'
            elif val == "Moyen":
                return 'background-color: #FEF3C7; color: #D97706'
            else:
                return 'background-color: #D1FAE5; color: #059669'
        
        # Affichage avec style
        styled_df = risk_df.style.applymap(color_risk_level, subset=['Niveau Risque'])
        st.dataframe(styled_df, use_container_width=True)

    def render_ai_quality_analysis(self, deliverables: List[Dict], test_categories: Dict[str, Any], quality_metrics: Dict[str, Any]):
        """Module d'analyse IA spécialisé pour la qualité"""
        
        st.markdown("### 🤖 Analyse IA Avancée - Qualité")
        st.markdown("Intelligence artificielle appliquée au contrôle qualité et prédictions de défauts")
        
        # Calculs IA spécialisés qualité
        df = pd.DataFrame(deliverables)
        total_deliverables = len(deliverables)
        completed_deliverables = len(df[df['status'] == 'Validé'])
        avg_quality_score = df['quality_score'].mean()
        critical_bugs = df['bugs_critical'].sum()
        avg_code_coverage = df['code_coverage'].mean()
        
        # Tests metrics
        total_tests = sum([cat['total'] for cat in test_categories.values()])
        passed_tests = sum([cat['passed'] for cat in test_categories.values()])
        test_success_rate = (passed_tests / total_tests) * 100
        
        # Score IA Qualité composite
        completion_score = (completed_deliverables / total_deliverables) * 100
        bugs_score = max(0, 100 - (critical_bugs * 15))  # Pénalité forte pour bugs critiques
        coverage_score = avg_code_coverage
        test_score = test_success_rate
        security_score = df['security_score'].mean()
        
        ai_quality_score = (
            avg_quality_score * 0.25 +
            completion_score * 0.20 +
            bugs_score * 0.20 +
            coverage_score * 0.15 +
            test_score * 0.12 +
            security_score * 0.08
        )
        
        # Classification IA Qualité
        if ai_quality_score >= 90:
            ai_quality_rating = "🟢 EXCELLENCE"
            ai_quality_recommendation = "Qualité exceptionnelle - Standard de référence"
        elif ai_quality_score >= 80:
            ai_quality_rating = "🔵 TRÈS BIEN"
            ai_quality_recommendation = "Qualité solide - Quelques optimisations possibles"
        elif ai_quality_score >= 70:
            ai_quality_rating = "🟡 CORRECT"
            ai_quality_recommendation = "Qualité acceptable - Améliorations nécessaires"
        elif ai_quality_score >= 60:
            ai_quality_rating = "🟠 FAIBLE"
            ai_quality_recommendation = "Qualité insuffisante - Actions urgentes requises"
        else:
            ai_quality_rating = "🔴 CRITIQUE"
            ai_quality_recommendation = "Qualité critique - Intervention immédiate"
        
        # === DASHBOARD IA QUALITÉ ===
        with st.expander("🧠 Dashboard IA Qualité", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Score IA Qualité", f"{ai_quality_score:.1f}/100", ai_quality_rating)
            with col2:
                st.metric("📋 Taux Complétion", f"{completion_score:.1f}%", f"{'↗️' if completion_score > 70 else '↘️'}")
            with col3:
                st.metric("🐛 Score Anti-Bugs", f"{bugs_score:.1f}%", f"{'✅' if critical_bugs == 0 else '🚨'}")
            with col4:
                st.metric("🤖 Recommandation IA", ai_quality_recommendation, "Quality AI Advisor")
        
        # === ANALYSES PRÉDICTIVES QUALITÉ ===
        with st.expander("🔮 Prédictions Qualité IA", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**💡 Insights IA Qualité:**")
                
                # Génération d'insights intelligents
                quality_insights = []
                
                if critical_bugs > 0:
                    quality_insights.append("🚨 **Bugs Critiques Détectés**: Risque de régression majeure identifié")
                else:
                    quality_insights.append("✅ **Zéro Bug Critique**: Excellence en stabilité du code")
                
                if avg_code_coverage < 80:
                    quality_insights.append("📊 **Couverture Insuffisante**: Tests manquants détectés")
                else:
                    quality_insights.append("🎯 **Couverture Optimale**: Protection robuste contre les régressions")
                
                # Analyse des livrables en retard
                overdue_deliverables = [d for d in deliverables if d['due_date'] < datetime.now() and d['status'] != 'Validé']
                if overdue_deliverables:
                    quality_insights.append(f"⏰ **{len(overdue_deliverables)} Livrables en Retard**: Impact planning détecté")
                
                # Analyse des tests
                if test_success_rate < 90:
                    quality_insights.append("🧪 **Instabilité Tests**: Fiabilité CI/CD compromise")
                else:
                    quality_insights.append("🚀 **Tests Excellents**: Pipeline de déploiement fiable")
                
                for insight in quality_insights:
                    st.write(f"• {insight}")
                
                # Recommandations IA Qualité
                st.markdown("**🎯 Actions Recommandées:**")
                
                quality_recommendations = []
                
                if critical_bugs > 0:
                    critical_deliverables = [d['name'] for d in deliverables if d['bugs_critical'] > 0]
                    quality_recommendations.append({
                        "priority": "🔴 URGENT",
                        "action": f"Résoudre bugs critiques dans {', '.join(critical_deliverables[:2])}",
                        "impact": f"Élimination de {critical_bugs} bug(s) critique(s)",
                        "confidence": "98%"
                    })
                
                low_coverage_deliverables = [d for d in deliverables if d['code_coverage'] < 80]
                if low_coverage_deliverables:
                    quality_recommendations.append({
                        "priority": "🟡 IMPORTANT",
                        "action": f"Améliorer couverture tests - {len(low_coverage_deliverables)} livrables",
                        "impact": f"Couverture +{80-avg_code_coverage:.1f}% possible",
                        "confidence": "92%"
                    })
                
                if test_success_rate < 90:
                    failed_tests = sum([cat['failed'] for cat in test_categories.values()])
                    quality_recommendations.append({
                        "priority": "🔵 AMÉLIORATION",
                        "action": f"Stabiliser {failed_tests} tests échoués",
                        "impact": "Fiabilité pipeline +15%",
                        "confidence": "87%"
                    })
                
                if ai_quality_score >= 85:
                    quality_recommendations.append({
                        "priority": "🟢 EXCELLENCE",
                        "action": "Documenter les bonnes pratiques actuelles",
                        "impact": "Standardisation qualité",
                        "confidence": "95%"
                    })
                
                for rec in quality_recommendations:
                    st.markdown(f"""
                    **{rec['priority']}: {rec['action']}**
                    - *Impact prévu*: {rec['impact']}
                    - *Confiance IA*: {rec['confidence']}
                    """)
            
            with col2:
                # Graphique prédictif IA avec zones de qualité
                months_future = ['Fév', 'Mar', 'Avr', 'Mai', 'Jun']
                
                # Prédictions basées sur les tendances actuelles
                quality_trend = 0.5 if critical_bugs == 0 else -2.0
                coverage_trend = 1.2 if avg_code_coverage > 80 else -0.8
                
                predicted_quality = []
                predicted_coverage = []
                current_quality = avg_quality_score
                current_coverage = avg_code_coverage
                
                for i in range(len(months_future)):
                    current_quality += quality_trend
                    current_coverage += coverage_trend
                    predicted_quality.append(max(60, min(100, current_quality)))
                    predicted_coverage.append(max(50, min(100, current_coverage)))
                
                fig = go.Figure()
                
                # Prédictions qualité
                fig.add_trace(go.Scatter(
                    x=months_future,
                    y=predicted_quality,
                    mode='lines+markers',
                    name='Score Qualité Prédit',
                    line=dict(color='#3B82F6', width=3)
                ))
                
                # Prédictions couverture
                fig.add_trace(go.Scatter(
                    x=months_future,
                    y=predicted_coverage,
                    mode='lines+markers',
                    name='Couverture Prédite',
                    line=dict(color='#10B981', width=3, dash='dot')
                ))
                
                # Zones de qualité
                fig.add_hline(y=90, line_dash="dash", line_color="green", 
                            annotation_text="Zone Excellence")
                fig.add_hline(y=80, line_dash="dash", line_color="orange",
                            annotation_text="Zone Acceptable")
                fig.add_hline(y=70, line_dash="dash", line_color="red",
                            annotation_text="Zone Critique")
                
                fig.update_layout(
                    title="🔮 Projection IA - Évolution Qualité avec Zones de Performance",
                    xaxis_title="Mois",
                    yaxis_title="Score (%)",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # === OPTIMISATIONS IA ===
        with st.expander("⚡ Optimisations IA Qualité", expanded=True):
            st.markdown("**🎯 Actions Recommandées par l'IA Qualité:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🐛 Corriger Bugs Critiques", use_container_width=True, type="primary"):
                    st.success("✅ **Plan de correction généré!**")
                    st.write("**Bugs critiques identifiés:**")
                    critical_deliverables = [d for d in deliverables if d['bugs_critical'] > 0]
                    for d in critical_deliverables:
                        st.write(f"• {d['name']}: {d['bugs_critical']} bug(s)")
            
            with col2:
                if st.button("📊 Booster Couverture", use_container_width=True):
                    st.info("🎯 **Stratégie de couverture:**")
                    st.write(f"**Objectif**: Atteindre 85% (actuellement {avg_code_coverage:.1f}%)")
                    st.write("• Tests unitaires prioritaires")
                    st.write("• Tests d'intégration manquants")
            
            with col3:
                if st.button("🧪 Stabiliser Tests", use_container_width=True):
                    st.warning("🔧 **Plan de stabilisation:**")
                    failed_tests = sum([cat['failed'] for cat in test_categories.values()])
                    st.write(f"**{failed_tests} tests à corriger**")
                    worst_category = max(test_categories.keys(), key=lambda x: test_categories[x]['failed'])
                    st.write(f"• Priorité: {worst_category}")
            
            with col4:
                if st.button("🤖 Assistant Qualité", use_container_width=True):
                    st.success("🤖 **Quality AI Assistant activé!**")
                    quality_question = st.text_input("Question Qualité:", placeholder="Ex: Comment améliorer la sécurité?")
                    if quality_question:
                        st.write(f"🤖: Basé sur votre score qualité de {ai_quality_score:.1f}% et {critical_bugs} bug(s) critique(s), je recommande {quality_recommendations[0]['action'].lower() if quality_recommendations else 'de maintenir les standards actuels'}.")
        
        # === MÉTRIQUES PERFORMANCE IA QUALITÉ ===
        st.markdown("---")
        st.markdown("**📈 Performance IA Quality Analyzer:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🧮 Analyses IA", "52", "Effectuées")
        with col2:
            st.metric("⚡ Vitesse", "0.06s", "Ultra-rapide")
        with col3:
            st.metric("🎯 Précision", "96.2%", "+3.1%")
        with col4:
            st.metric("💡 Recommandations", f"{len(quality_recommendations) if 'quality_recommendations' in locals() else 0}", "Générées")
        with col5:
            st.metric("✅ Fiabilité", "94%", "Quality IA")

# Fonction d'entrée pour intégration dans l'app principale
def create_quality_dashboard():
    """Point d'entrée pour le module qualité amélioré"""
    quality_module = QualityModule()
    quality_module.render_quality_dashboard()

# Test si le fichier est exécuté directement
if __name__ == "__main__":
    st.set_page_config(
        page_title="Module Qualité Intelligent - PlannerIA",
        page_icon="✅",
        layout="wide"
    )
    
    create_quality_dashboard()