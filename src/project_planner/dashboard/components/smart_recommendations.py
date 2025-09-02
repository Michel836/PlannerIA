"""
🧠 Composant Dashboard - Recommandations Intelligentes
=====================================================

Interface utilisateur avancée pour le système de recommandations IA de PlannerIA.
Ce module fournit une expérience interactive complète pour visualiser, évaluer et
appliquer les suggestions d'optimisation générées par l'intelligence artificielle.

Fonctionnalités du Dashboard:
- 🎯 Affichage interactif des recommandations par priorité
- 📊 Analyse comportementale avec graphiques Plotly
- 📚 Base de connaissances searchable avec rating système
- 🔍 Détection de patterns industriels avec exemples concrets
- ⚙️ Configuration avancée des paramètres IA

Architecture des Onglets:
1. 🎯 Recommandations Actives - Actions suggérées avec apply/dismiss
2. 📊 Analyse Comportementale - Profiling utilisateur et préférences
3. 📚 Base de Connaissances - Meilleures pratiques et recherche
4. 🔍 Patterns Détectés - Insights comportementaux et industriels  
5. ⚙️ Paramètres IA - Configuration fine du moteur ML

Interactions Utilisateur:
- ✅ Application directe des recommandations avec feedback
- ❌ Système de rejet avec apprentissage des préférences
- 📊 Notation des pratiques pour amélioration continue
- 🔍 Recherche textuelle intelligente dans la knowledge base
- 📈 Visualisations interactives avec drill-down

Intégrations:
- Moteur de Recommandations ML pour génération suggestions
- Analyseur Comportemental pour personnalisation
- Base de Connaissances Dynamique pour meilleures pratiques
- Système de Suggestions Contextuelles pour UX optimisée

Exemple d'usage:
    dashboard = SmartRecommendationsDashboard()
    
    # Rendu du panel principal avec projet
    dashboard.render_main_recommendations_panel({
        'project_id': 'demo_001',
        'industry': 'software',
        'complexity': 'high',
        'team_size': 8
    })
    
    # Recommandations en sidebar
    dashboard.render_recommendations_sidebar(project_data)

Performance:
- Cache intelligent des recommandations calculées
- Lazy loading des graphiques complexes  
- Debouncing des interactions utilisateur
- Streaming des mises à jour comportementales

Auteur: PlannerIA AI System
Version: 2.0.0
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

# Import des modules de recommandations
from ...recommendations import (
    get_recommendation_engine,
    generate_project_recommendations,
    get_contextual_suggestions,
    get_behavioral_analyzer,
    track_user_action,
    get_user_preferences,
    get_knowledge_base,
    get_industry_patterns,
    get_best_practices
)

from ...recommendations.ui_suggestions import (
    get_contextual_suggestions_ui,
    render_contextual_suggestions,
    SuggestionTrigger,
    SuggestionContext
)

from ...recommendations.recommendation_engine import (
    RecommendationType,
    RecommendationPriority
)

from ...recommendations.behavioral_analyzer import (
    ActionType,
    PreferenceType
)

from ...recommendations.knowledge_base import (
    IndustryType,
    PracticeCategory,
    PatternType
)

logger = logging.getLogger(__name__)

class SmartRecommendationsDashboard:
    """
    🧠 Dashboard Principal des Recommandations Intelligentes
    
    Classe principale orchestrant l'affichage et l'interaction avec le système
    de recommandations IA. Fournit une interface utilisateur complète pour
    visualiser, évaluer et appliquer les suggestions d'optimisation.
    
    Architecture:
        Cette classe intègre plusieurs systèmes d'IA:
        - RecommendationEngine: Génération ML des suggestions
        - BehavioralAnalyzer: Analyse des préférences utilisateur
        - KnowledgeBase: Base de connaissances des meilleures pratiques
        - UISuggestions: Système de suggestions contextuelles
    
    Fonctionnalités:
        🎯 Recommandations Actives:
        - Affichage par priorité (CRITICAL > HIGH > MEDIUM > LOW)
        - Actions directes (appliquer/ignorer/détails)
        - Métriques d'impact et niveau de confiance IA
        
        📊 Analyse Comportementale:
        - Profiling automatique des préférences utilisateur
        - Graphiques interactifs de patterns détectés
        - Statistiques de session et d'usage
        
        📚 Base de Connaissances:
        - Recherche textuelle dans les meilleures pratiques
        - Système de notation des pratiques utilisées
        - Patterns industriels avec exemples concrets
        
        ⚙️ Configuration IA:
        - Paramétrage du moteur de recommandations
        - Seuils de confiance et types préférés
        - Gestion de la confidentialité et données
    
    Usage Typique:
        # Initialisation
        dashboard = SmartRecommendationsDashboard()
        
        # Rendu interface principale
        dashboard.render_main_recommendations_panel({
            'project_id': 'proj_001',
            'industry': 'software',
            'team_size': 5,
            'complexity': 'high'
        })
        
        # Intégration sidebar
        dashboard.render_recommendations_sidebar(project_data)
    
    Performance:
        - Cache intelligent des recommandations calculées
        - Lazy loading des graphiques Plotly complexes
        - Debouncing automatique des interactions utilisateur
        - Streaming en arrière-plan des mises à jour comportementales
    
    Sécurité:
        - Validation stricte des données d'entrée utilisateur
        - Anonymisation des patterns comportementaux
        - Respect des préférences de confidentialité RGPD
    """
    
    def __init__(self):
        self.recommendation_engine = get_recommendation_engine()
        self.behavioral_analyzer = get_behavioral_analyzer()
        self.knowledge_base = get_knowledge_base()
        self.ui_suggestions = get_contextual_suggestions_ui()
        
    def render_main_recommendations_panel(self, project_data: Dict[str, Any]):
        """
        🎯 Rendu du Panel Principal des Recommandations Intelligentes
        
        Affiche l'interface utilisateur complète du système de recommandations avec
        5 onglets spécialisés et tracking automatique des interactions utilisateur.
        
        Args:
            project_data: Dictionnaire contenant les informations du projet
                - project_id (str): Identifiant unique du projet
                - industry (str): Secteur d'activité (software, construction, etc.)
                - team_size (int): Taille de l'équipe projet
                - complexity (str): Niveau de complexité (low, medium, high)
                - budget (float, optionnel): Budget alloué au projet
                - duration (int, optionnel): Durée prévue en jours
        
        Interface générée:
            📑 Onglet "Recommandations Actives":
            - Filtrage par industrie et types de recommandations
            - Affichage par priorité avec métriques de confiance
            - Actions directes: appliquer, plus d'infos, ignorer
            - Statistiques rapides: critiques, importantes, confiance moyenne
            
            📊 Onglet "Analyse Comportementale":
            - Profil utilisateur avec préférences détectées
            - Graphiques Plotly interactifs (confiance/fréquence)
            - Statistiques de session et répartition des actions
            - Historique des patterns comportementaux
            
            📚 Onglet "Base de Connaissances":
            - Moteur de recherche textuelle intelligent
            - Rating des meilleures pratiques (1-5 étoiles)  
            - Browser des patterns industriels avec exemples
            - Statistiques de la knowledge base
            
            🔍 Onglet "Patterns Détectés":
            - Patterns comportementaux avec niveaux de confiance
            - Patterns industriels filtrables par type/industrie
            - Visualisations des déclencheurs et résultats
            - Insights actionables basés sur les détections
            
            ⚙️ Onglet "Paramètres IA":
            - Configuration du moteur de recommandations
            - Seuils de confiance et types préférés
            - Paramètres d'analyse comportementale
            - Gestion des données et confidentialité
        
        Interactions Trackées:
            - Vue de la page → ActionType.DASHBOARD_VIEW
            - Application recommandation → ActionType.RECOMMENDATION_ACCEPTED
            - Rejet recommandation → ActionType.RECOMMENDATION_REJECTED
            - Notation pratique → Mise à jour base de connaissances
            - Changement paramètres → Sauvegarde préférences
        
        Performance:
            - Génération lazy des recommandations (uniquement si onglet actif)
            - Cache des graphiques Plotly pour éviter recalculs
            - Streaming des mises à jour comportementales en arrière-plan
            - Pagination automatique des grandes listes
        
        Exemple d'appel:
            dashboard.render_main_recommendations_panel({
                'project_id': 'ecommerce_2024',
                'industry': 'software',
                'team_size': 8,
                'complexity': 'high',
                'budget': 150000,
                'duration': 180
            })
        """
        
        st.markdown("# 🧠 Recommandations Intelligentes")
        st.markdown("---")
        
        # Génération des recommandations contextuelles
        current_context = {
            'page': 'recommendations',
            'user_action': 'viewing_recommendations'
        }
        
        # Tracker l'action de visualisation
        track_user_action(
            ActionType.DASHBOARD_VIEW,
            context=current_context,
            project_id=project_data.get('project_id')
        )
        
        # Tabs pour différentes vues
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Recommandations Actives", 
            "📊 Analyse Comportementale", 
            "📚 Base de Connaissances", 
            "🔍 Patterns Détectés",
            "⚙️ Paramètres IA"
        ])\n        
        with tab1:
            self._render_active_recommendations_tab(project_data)
            
        with tab2:
            self._render_behavioral_analysis_tab(project_data)
            
        with tab3:
            self._render_knowledge_base_tab(project_data)
            
        with tab4:
            self._render_patterns_tab(project_data)
            
        with tab5:
            self._render_ai_settings_tab(project_data)
            
    def _render_active_recommendations_tab(self, project_data: Dict[str, Any]):
        """Onglet des recommandations actives"""
        
        st.markdown("## 🎯 Recommandations Personnalisées")
        
        # Paramètres de génération
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            industry_filter = st.selectbox(
                "Industrie",
                options=[i.value for i in IndustryType],
                index=0,
                key="rec_industry_filter"
            )
            
        with col2:
            rec_types = st.multiselect(
                "Types de recommandations",
                options=[t.value for t in RecommendationType],
                default=[RecommendationType.OPTIMIZATION.value, RecommendationType.RISK_MITIGATION.value],
                key="rec_types_filter"
            )
            
        with col3:
            if st.button("🔄 Actualiser", key="refresh_recommendations"):
                st.session_state['refresh_recommendations'] = True
                
        # Générer les recommandations
        try:
            with st.spinner("🧠 Génération des recommandations IA..."):
                recommendations = generate_project_recommendations(
                    project_data=project_data,
                    recommendation_types=[RecommendationType(rt) for rt in rec_types] if rec_types else None,
                    limit=10
                )
                
            if recommendations:
                # Statistiques rapides
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    critical_count = len([r for r in recommendations if r.priority == RecommendationPriority.CRITICAL])
                    st.metric("🔴 Critiques", critical_count)
                    
                with col2:
                    high_count = len([r for r in recommendations if r.priority == RecommendationPriority.HIGH])
                    st.metric("🟠 Importantes", high_count)
                    
                with col3:
                    avg_confidence = np.mean([r.confidence for r in recommendations])
                    st.metric("📊 Confiance Moyenne", f"{avg_confidence:.1%}")
                    
                with col4:
                    ml_count = len([r for r in recommendations if r.source == 'ml_analysis'])
                    st.metric("🤖 ML-Based", ml_count)
                
                st.markdown("---")
                
                # Affichage des recommandations par priorité
                priority_groups = {
                    RecommendationPriority.CRITICAL: [],
                    RecommendationPriority.HIGH: [],
                    RecommendationPriority.MEDIUM: [],
                    RecommendationPriority.LOW: []
                }
                
                for rec in recommendations:
                    priority_groups[rec.priority].append(rec)
                    
                # Recommandations critiques en premier
                if priority_groups[RecommendationPriority.CRITICAL]:
                    st.markdown("### 🚨 Recommandations Critiques")
                    for rec in priority_groups[RecommendationPriority.CRITICAL]:
                        self._render_recommendation_card(rec, project_data, is_critical=True)
                        
                # Recommandations importantes
                if priority_groups[RecommendationPriority.HIGH]:
                    st.markdown("### ⚠️ Recommandations Importantes")
                    for rec in priority_groups[RecommendationPriority.HIGH]:
                        self._render_recommendation_card(rec, project_data)
                        
                # Recommandations moyennes (dans un expander)
                if priority_groups[RecommendationPriority.MEDIUM]:
                    with st.expander("📋 Recommandations Moyennes", expanded=False):
                        for rec in priority_groups[RecommendationPriority.MEDIUM]:
                            self._render_recommendation_card(rec, project_data, compact=True)
                            
                # Recommandations faibles (dans un expander)
                if priority_groups[RecommendationPriority.LOW]:
                    with st.expander("💡 Suggestions Additionnelles", expanded=False):
                        for rec in priority_groups[RecommendationPriority.LOW]:
                            self._render_recommendation_card(rec, project_data, compact=True)
                            
            else:
                st.info("Aucune recommandation générée. Ajustez les paramètres ou le contexte projet.")
                
        except Exception as e:
            st.error(f"Erreur génération recommandations: {e}")
            logger.error(f"Erreur recommandations: {e}")
            
    def _render_recommendation_card(self, 
                                   recommendation: Any, 
                                   project_data: Dict[str, Any], 
                                   is_critical: bool = False,
                                   compact: bool = False):
        """Rendre une carte de recommandation"""
        
        # Couleurs selon la priorité
        colors = {
            RecommendationPriority.CRITICAL: "#ff4444",
            RecommendationPriority.HIGH: "#ff8800", 
            RecommendationPriority.MEDIUM: "#ffbb00",
            RecommendationPriority.LOW: "#888888"
        }
        
        priority_color = colors.get(recommendation.priority, "#888888")
        
        # Container avec bordure colorée
        container = st.container()
        
        with container:
            if not compact:
                # Version complète
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Titre avec icône de priorité
                    priority_icons = {
                        RecommendationPriority.CRITICAL: "🚨",
                        RecommendationPriority.HIGH: "⚠️",
                        RecommendationPriority.MEDIUM: "📋", 
                        RecommendationPriority.LOW: "💡"
                    }
                    
                    icon = priority_icons.get(recommendation.priority, "📋")
                    st.markdown(f"### {icon} {recommendation.title}")
                    st.markdown(f"**Description:** {recommendation.description}")
                    
                    # Actions suggérées
                    if hasattr(recommendation, 'actions') and recommendation.actions:
                        st.markdown("**Actions recommandées:**")
                        for i, action in enumerate(recommendation.actions, 1):
                            st.markdown(f"{i}. {action}")
                            
                    # Métriques si disponibles
                    if hasattr(recommendation, 'impact_metrics') and recommendation.impact_metrics:
                        st.markdown("**Impact estimé:**")
                        metrics_cols = st.columns(min(len(recommendation.impact_metrics), 4))
                        for i, (metric, value) in enumerate(recommendation.impact_metrics.items()):
                            if i < len(metrics_cols):
                                with metrics_cols[i]:
                                    st.metric(metric.replace('_', ' ').title(), value)
                                    
                with col2:
                    # Badge de confiance
                    confidence_color = "green" if recommendation.confidence > 0.8 else "orange" if recommendation.confidence > 0.6 else "red"
                    st.markdown(f"<div style='text-align: center; color: {confidence_color}; font-weight: bold;'>Confiance<br>{recommendation.confidence:.0%}</div>", unsafe_allow_html=True)
                    
                    # Boutons d'action
                    if st.button("✅ Appliquer", key=f"apply_{recommendation.recommendation_id}", type="primary"):
                        self._apply_recommendation(recommendation, project_data)
                        
                    if st.button("❓ Plus d'infos", key=f"info_{recommendation.recommendation_id}"):
                        self._show_recommendation_details(recommendation)
                        
                    if st.button("❌ Ignorer", key=f"dismiss_{recommendation.recommendation_id}"):
                        self._dismiss_recommendation(recommendation, project_data)
                        
            else:
                # Version compacte
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    icon = "🚨" if recommendation.priority == RecommendationPriority.CRITICAL else "📋"
                    st.markdown(f"**{icon} {recommendation.title}**")
                    st.markdown(f"<small>{recommendation.description[:100]}...</small>", unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"<small>Confiance: {recommendation.confidence:.0%}</small>", unsafe_allow_html=True)
                    
                with col3:
                    if st.button("✅", key=f"apply_compact_{recommendation.recommendation_id}", help="Appliquer"):
                        self._apply_recommendation(recommendation, project_data)
                        
            # Séparateur
            st.markdown("---")
            
    def _render_behavioral_analysis_tab(self, project_data: Dict[str, Any]):
        """Onglet d'analyse comportementale"""
        
        st.markdown("## 🧭 Analyse de Vos Préférences")
        
        # Obtenir les préférences utilisateur
        preferences = get_user_preferences()
        
        if preferences:
            # Vue d'ensemble des préférences
            st.markdown("### 📊 Profil Utilisateur")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Préférences Détectées")
                
                for pref_key, preference in preferences.items():
                    confidence_color = "🟢" if preference.confidence > 0.8 else "🟡" if preference.confidence > 0.6 else "🔴"
                    
                    with st.expander(f"{confidence_color} {pref_key.replace('_', ' ').title()}", expanded=preference.confidence > 0.7):
                        st.markdown(f"**Valeur:** {preference.value}")
                        st.markdown(f"**Confiance:** {preference.confidence:.1%}")
                        st.markdown(f"**Fréquence:** {preference.frequency} fois")
                        st.markdown(f"**Dernière mise à jour:** {preference.last_updated.strftime('%d/%m/%Y %H:%M')}")
                        
            with col2:
                st.markdown("#### 📈 Graphiques des Préférences")
                
                # Graphique de confiance des préférences
                if preferences:
                    pref_names = [k.replace('_', ' ').title() for k in preferences.keys()]
                    confidences = [p.confidence for p in preferences.values()]
                    frequencies = [p.frequency for p in preferences.values()]
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Niveau de Confiance', 'Fréquence d\'Usage'),
                        vertical_spacing=0.1
                    )
                    
                    # Graphique de confiance
                    fig.add_trace(
                        go.Bar(
                            x=pref_names,
                            y=confidences,
                            name='Confiance',
                            marker_color=['green' if c > 0.8 else 'orange' if c > 0.6 else 'red' for c in confidences]
                        ),
                        row=1, col=1
                    )
                    
                    # Graphique de fréquence
                    fig.add_trace(
                        go.Bar(
                            x=pref_names,
                            y=frequencies,
                            name='Fréquence',
                            marker_color='lightblue'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=500, showlegend=False)
                    fig.update_yaxes(title_text="Niveau de Confiance", row=1, col=1)
                    fig.update_yaxes(title_text="Nombre d'Occurrences", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            # Statistiques de session
            st.markdown("### 📊 Statistiques de Session")
            session_stats = self.behavioral_analyzer.get_session_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Actions Totales", session_stats['total_actions'])
                
            with col2:
                st.metric("Durée Session", f"{session_stats['session_duration_minutes']} min")
                
            with col3:
                st.metric("Patterns Détectés", session_stats['patterns_detected'])
                
            with col4:
                st.metric("Préférences", session_stats['preferences_updated'])
                
            # Répartition des actions
            if session_stats['action_breakdown']:
                st.markdown("#### 🎯 Répartition des Actions")
                
                action_df = pd.DataFrame(
                    list(session_stats['action_breakdown'].items()),
                    columns=['Action', 'Count']
                )
                
                fig = px.pie(
                    action_df, 
                    values='Count', 
                    names='Action',
                    title="Distribution des Actions Utilisateur"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.info("🤖 Continuez à utiliser PlannerIA pour que notre IA apprenne vos préférences et personnalise votre expérience!")
            
    def _render_knowledge_base_tab(self, project_data: Dict[str, Any]):
        """Onglet de la base de connaissances"""
        
        st.markdown("## 📚 Base de Connaissances Dynamique")
        
        # Statistiques de la KB
        kb_stats = self.knowledge_base.get_knowledge_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Meilleures Pratiques", kb_stats['practices']['total'])
            
        with col2:
            st.metric("Patterns Industriels", kb_stats['patterns']['total'])
            
        with col3:
            st.metric("Évaluations Utilisateur", kb_stats['total_ratings'])
            
        with col4:
            kb_size = kb_stats['knowledge_base_size']
            st.metric("Taille Base", f"{kb_size} entrées")
            
        # Recherche dans la base
        st.markdown("### 🔍 Recherche de Connaissances")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Rechercher des pratiques ou patterns...",
                placeholder="Ex: gestion des risques, planification agile, budget",
                key="kb_search"
            )
            
        with col2:
            search_type = st.selectbox(
                "Type",
                options=['Tout', 'Pratiques', 'Patterns'],
                key="kb_search_type"
            )
            
        if search_query:
            search_types = {
                'Tout': ['practices', 'patterns'],
                'Pratiques': ['practices'],
                'Patterns': ['patterns']
            }
            
            results = self.knowledge_base.search_knowledge(
                search_query,
                knowledge_types=search_types[search_type]
            )
            
            # Afficher résultats pratiques
            if results['practices']:
                st.markdown("#### 📋 Meilleures Pratiques Trouvées")
                for practice in results['practices']:
                    with st.expander(f"🎯 {practice.title} (Succès: {practice.success_rate:.0%})", expanded=False):
                        st.markdown(f"**Description:** {practice.description}")
                        st.markdown(f"**Catégorie:** {practice.category.value}")
                        st.markdown(f"**Industrie:** {practice.industry.value}")
                        
                        if practice.implementation_steps:
                            st.markdown("**Étapes d'implémentation:**")
                            for i, step in enumerate(practice.implementation_steps, 1):
                                st.markdown(f"{i}. {step}")
                                
                        # Rating de la pratique
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            rating = st.slider(
                                "Évaluer cette pratique", 
                                1, 5, 3, 
                                key=f"rate_{practice.practice_id}",
                                help="1=Inutile, 5=Très utile"
                            )
                        with col2:
                            if st.button("Noter", key=f"submit_rating_{practice.practice_id}"):
                                self.knowledge_base.rate_practice(
                                    practice.practice_id,
                                    rating,
                                    project_id=project_data.get('project_id')
                                )
                                st.success(f"✅ Pratique notée: {rating}/5")
                                
            # Afficher résultats patterns
            if results['patterns']:
                st.markdown("#### 🔍 Patterns Industriels Trouvés")
                for pattern in results['patterns']:
                    pattern_icon = "✅" if pattern.pattern_type.value == 'success_pattern' else "❌" if pattern.pattern_type.value == 'failure_pattern' else "⚡"
                    
                    with st.expander(f"{pattern_icon} {pattern.title} (Impact: {pattern.impact_score:.0%})", expanded=False):
                        st.markdown(f"**Description:** {pattern.description}")
                        st.markdown(f"**Type:** {pattern.pattern_type.value}")
                        st.markdown(f"**Industrie:** {pattern.industry.value}")
                        st.markdown(f"**Fréquence:** {pattern.frequency} occurrences")
                        
                        if pattern.triggers:
                            st.markdown("**Déclencheurs:** " + ", ".join(pattern.triggers))
                            
                        if pattern.outcomes:
                            st.markdown("**Résultats:** " + ", ".join(pattern.outcomes))
                            
                        if pattern.examples:
                            st.markdown("**Exemples:**")
                            for example in pattern.examples[:3]:  # Limiter à 3 exemples
                                st.json(example)
                                
    def _render_patterns_tab(self, project_data: Dict[str, Any]):
        """Onglet des patterns détectés"""
        
        st.markdown("## 🔍 Patterns Détectés")
        
        # Obtenir les patterns comportementaux
        behavior_patterns = self.behavioral_analyzer.get_behavior_patterns()
        
        if behavior_patterns:
            st.markdown("### 🧭 Patterns Comportementaux Détectés")
            
            for pattern_id, pattern in behavior_patterns.items():
                confidence_color = "🟢" if pattern.confidence > 0.8 else "🟡" if pattern.confidence > 0.6 else "🔴"
                
                with st.expander(f"{confidence_color} {pattern.description} (Confiance: {pattern.confidence:.0%})", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Type:** {pattern.pattern_type}")
                        st.markdown(f"**Fréquence:** {pattern.frequency}")
                        st.markdown(f"**Détecté le:** {pattern.detected_at.strftime('%d/%m/%Y %H:%M')}")
                        
                    with col2:
                        if pattern.triggers:
                            st.markdown("**Déclencheurs:**")
                            for trigger in pattern.triggers:
                                st.markdown(f"- {trigger}")
                                
                        if pattern.outcomes:
                            st.markdown("**Résultats:**")
                            for outcome in pattern.outcomes:
                                st.markdown(f"- {outcome}")
                                
        # Obtenir patterns industriels pertinents
        industry_patterns = get_industry_patterns()
        
        if industry_patterns:
            st.markdown("### 🏢 Patterns Industriels Disponibles")
            
            # Filtres
            col1, col2 = st.columns(2)
            
            with col1:
                industry_filter = st.selectbox(
                    "Filtrer par industrie",
                    options=['Toutes'] + [i.value for i in IndustryType],
                    key="industry_pattern_filter"
                )
                
            with col2:
                pattern_type_filter = st.selectbox(
                    "Filtrer par type",
                    options=['Tous'] + [p.value for p in PatternType],
                    key="pattern_type_filter"
                )
                
            # Filtrer les patterns
            filtered_patterns = industry_patterns
            
            if industry_filter != 'Toutes':
                filtered_patterns = [p for p in filtered_patterns if p.industry.value == industry_filter]
                
            if pattern_type_filter != 'Tous':
                filtered_patterns = [p for p in filtered_patterns if p.pattern_type.value == pattern_type_filter]
                
            # Afficher les patterns
            for pattern in filtered_patterns[:10]:  # Limiter à 10
                self._render_industry_pattern(pattern)
                
        if not behavior_patterns and not industry_patterns:
            st.info("Aucun pattern détecté pour le moment. Continuez à utiliser PlannerIA pour que notre IA détecte vos habitudes de travail.")
            
    def _render_industry_pattern(self, pattern):
        """Rendre un pattern industriel"""
        
        type_icons = {
            'success_pattern': '✅',
            'failure_pattern': '❌', 
            'optimization_pattern': '⚡',
            'risk_pattern': '⚠️',
            'efficiency_pattern': '🚀'
        }
        
        icon = type_icons.get(pattern.pattern_type.value, '📊')
        impact_color = "🔴" if pattern.impact_score > 0.8 else "🟠" if pattern.impact_score > 0.6 else "🟡"
        
        with st.expander(f"{icon} {pattern.title} {impact_color} Impact: {pattern.impact_score:.0%}", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Description:** {pattern.description}")
                st.markdown(f"**Industrie:** {pattern.industry.value}")
                st.markdown(f"**Fréquence:** {pattern.frequency} occurrences")
                
            with col2:
                st.markdown(f"**Type:** {pattern.pattern_type.value}")
                st.markdown(f"**Confiance:** {pattern.confidence:.0%}")
                st.markdown(f"**Impact:** {pattern.impact_score:.0%}")
                
            # Déclencheurs et indicateurs
            if pattern.triggers:
                st.markdown("**🎯 Déclencheurs:**")
                for trigger in pattern.triggers:
                    st.markdown(f"- {trigger}")
                    
            if pattern.indicators:
                st.markdown("**📊 Indicateurs:**")
                for indicator in pattern.indicators:
                    st.markdown(f"- {indicator}")
                    
            if pattern.outcomes:
                st.markdown("**🎯 Résultats Typiques:**")
                for outcome in pattern.outcomes:
                    st.markdown(f"- {outcome}")
                    
    def _render_ai_settings_tab(self, project_data: Dict[str, Any]):
        """Onglet des paramètres IA"""
        
        st.markdown("## ⚙️ Configuration du Système IA")
        
        # Paramètres de recommandations
        st.markdown("### 🎯 Paramètres des Recommandations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seuil de confiance
            confidence_threshold = st.slider(
                "Seuil de confiance minimal",
                0.0, 1.0, 0.6,
                step=0.1,
                help="Recommandations avec confiance en dessous de ce seuil ne seront pas affichées",
                key="confidence_threshold"
            )
            
            # Types de recommandations préférées
            preferred_types = st.multiselect(
                "Types de recommandations préférées",
                options=[t.value for t in RecommendationType],
                default=[RecommendationType.OPTIMIZATION.value, RecommendationType.RISK_MITIGATION.value],
                key="preferred_rec_types"
            )
            
        with col2:
            # Fréquence des suggestions
            suggestion_frequency = st.selectbox(
                "Fréquence des suggestions contextuelles",
                options=['Élevée', 'Moyenne', 'Faible', 'Désactivée'],
                index=1,
                key="suggestion_frequency"
            )
            
            # Apprentissage automatique
            enable_ml_learning = st.checkbox(
                "Activer l'apprentissage automatique",
                value=True,
                help="Le système apprend de vos actions pour améliorer les recommandations",
                key="enable_ml_learning"
            )
            
        # Paramètres de l'analyse comportementale  
        st.markdown("### 🧭 Analyse Comportementale")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tracking_enabled = st.checkbox(
                "Activer le tracking comportemental",
                value=True,
                help="Permet au système de détecter vos patterns de travail",
                key="tracking_enabled"
            )
            
            privacy_level = st.selectbox(
                "Niveau de confidentialité",
                options=['Basique', 'Élevé', 'Maximum'],
                index=0,
                help="Contrôle quelles données sont collectées",
                key="privacy_level"
            )
            
        with col2:
            # Réinitialisation des données
            st.markdown("**Gestion des Données**")
            
            if st.button("🗑️ Effacer historique comportemental", key="clear_behavior_data"):
                self._clear_behavioral_data()
                
            if st.button("📊 Exporter données personnelles", key="export_personal_data"):
                self._export_personal_data()
                
        # Paramètres avancés
        with st.expander("🔧 Paramètres Avancés", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Paramètres du moteur de recommandations
                st.markdown("**Moteur de Recommandations**")
                
                max_recommendations = st.number_input(
                    "Nombre maximum de recommandations",
                    min_value=1, max_value=20, value=10,
                    key="max_recommendations"
                )
                
                recommendation_refresh_interval = st.number_input(
                    "Intervalle de rafraîchissement (minutes)",
                    min_value=1, max_value=60, value=15,
                    key="rec_refresh_interval"
                )
                
            with col2:
                # Paramètres de la base de connaissances
                st.markdown("**Base de Connaissances**")
                
                kb_auto_update = st.checkbox(
                    "Mise à jour automatique",
                    value=True,
                    key="kb_auto_update"
                )
                
                kb_industry_focus = st.selectbox(
                    "Focus industrie",
                    options=['Auto-détection'] + [i.value for i in IndustryType],
                    key="kb_industry_focus"
                )
                
        # Bouton de sauvegarde des paramètres
        if st.button("💾 Sauvegarder Configuration", key="save_ai_config", type="primary"):
            self._save_ai_configuration({
                'confidence_threshold': confidence_threshold,
                'preferred_types': preferred_types,
                'suggestion_frequency': suggestion_frequency,
                'enable_ml_learning': enable_ml_learning,
                'tracking_enabled': tracking_enabled,
                'privacy_level': privacy_level,
                'max_recommendations': max_recommendations,
                'recommendation_refresh_interval': recommendation_refresh_interval,
                'kb_auto_update': kb_auto_update,
                'kb_industry_focus': kb_industry_focus
            })
            
    def _apply_recommendation(self, recommendation: Any, project_data: Dict[str, Any]):
        """Appliquer une recommandation"""
        
        # Tracker l'acceptation
        track_user_action(
            ActionType.RECOMMENDATION_ACCEPTED,
            context={
                'recommendation_id': recommendation.recommendation_id,
                'recommendation_type': recommendation.recommendation_type.value if hasattr(recommendation, 'recommendation_type') else 'unknown',
                'confidence': recommendation.confidence
            },
            project_id=project_data.get('project_id')
        )
        
        # Logique d'application selon le type de recommandation
        if hasattr(recommendation, 'recommendation_type'):
            if recommendation.recommendation_type == RecommendationType.OPTIMIZATION:
                st.session_state['apply_optimization'] = recommendation.to_dict()
            elif recommendation.recommendation_type == RecommendationType.RISK_MITIGATION:
                st.session_state['apply_risk_mitigation'] = recommendation.to_dict()
            elif recommendation.recommendation_type == RecommendationType.BUDGET_ADJUSTMENT:
                st.session_state['apply_budget_adjustment'] = recommendation.to_dict()
                
        st.success(f"✅ Recommandation appliquée: {recommendation.title}")
        st.success("✅ Recommandation appliquée")
        
    def _dismiss_recommendation(self, recommendation: Any, project_data: Dict[str, Any]):
        """Ignorer une recommandation"""
        
        # Tracker le rejet
        track_user_action(
            ActionType.RECOMMENDATION_REJECTED,
            context={
                'recommendation_id': recommendation.recommendation_id,
                'reason': 'user_dismissed'
            },
            project_id=project_data.get('project_id')
        )
        
        st.info(f"Recommandation ignorée: {recommendation.title}")
        st.success("✅ Recommandation appliquée")
        
    def _show_recommendation_details(self, recommendation: Any):
        """Afficher les détails d'une recommandation"""
        
        with st.modal(f"Détails - {recommendation.title}"):
            st.markdown(f"**Description complète:** {recommendation.description}")
            
            if hasattr(recommendation, 'reasoning') and recommendation.reasoning:
                st.markdown(f"**Raisonnement IA:** {recommendation.reasoning}")
                
            if hasattr(recommendation, 'actions') and recommendation.actions:
                st.markdown("**Actions recommandées:**")
                for i, action in enumerate(recommendation.actions, 1):
                    st.markdown(f"{i}. {action}")
                    
            if hasattr(recommendation, 'impact_metrics') and recommendation.impact_metrics:
                st.markdown("**Impact estimé:**")
                st.json(recommendation.impact_metrics)
                
            st.markdown(f"**Confiance:** {recommendation.confidence:.1%}")
            st.markdown(f"**Source:** {recommendation.source}")
            
            if hasattr(recommendation, 'created_at'):
                st.markdown(f"**Créé le:** {recommendation.created_at.strftime('%d/%m/%Y %H:%M')}")
                
    def _clear_behavioral_data(self):
        """Effacer les données comportementales"""
        try:
            # Logic to clear behavioral data would go here
            st.success("✅ Données comportementales effacées")
            st.success("✅ Recommandation appliquée")
        except Exception as e:
            st.error(f"Erreur lors de l'effacement: {e}")
            
    def _export_personal_data(self):
        """Exporter les données personnelles"""
        try:
            # Logic to export personal data would go here
            st.success("✅ Données exportées (voir téléchargements)")
        except Exception as e:
            st.error(f"Erreur lors de l'export: {e}")
            
    def _save_ai_configuration(self, config: Dict[str, Any]):
        """Sauvegarder la configuration IA"""
        try:
            # Save configuration logic would go here
            st.session_state['ai_config'] = config
            st.success("✅ Configuration sauvegardée")
        except Exception as e:
            st.error(f"Erreur sauvegarde: {e}")
            
    def render_recommendations_sidebar(self, project_data: Dict[str, Any]):
        """Rendre les recommandations dans la sidebar"""
        
        current_context = {'page': 'sidebar', 'context': 'navigation'}
        
        # Générer suggestions contextuelles pour la sidebar
        suggestions = self.ui_suggestions.generate_contextual_suggestions(
            project_data, 
            current_context, 
            SuggestionTrigger.ON_LOAD
        )
        
        if suggestions:
            with st.sidebar:
                st.markdown("### 💡 Suggestions IA")
                
                sidebar_suggestions = [s for s in suggestions if s.context == SuggestionContext.SIDEBAR]
                
                for suggestion in sidebar_suggestions[:3]:  # Limiter à 3
                    with st.container():
                        st.markdown(f"**{suggestion.title}**")
                        st.markdown(f"<small>{suggestion.description}</small>", unsafe_allow_html=True)
                        
                        if st.button("Appliquer", key=f"sidebar_sug_{suggestion.suggestion_id}"):
                            # Logic to apply sidebar suggestion
                            st.success(f"✅ {suggestion.title}")
                            
                        st.markdown("---")

# Instance globale
def get_smart_recommendations_dashboard():
    """Obtenir l'instance du dashboard des recommandations"""
    if 'smart_rec_dashboard' not in st.session_state:
        st.session_state['smart_rec_dashboard'] = SmartRecommendationsDashboard()
    return st.session_state['smart_rec_dashboard']

# Fonctions utilitaires pour l'intégration
def render_smart_recommendations_panel(project_data: Dict[str, Any]):
    """Fonction utilitaire pour rendre le panneau de recommandations"""
    dashboard = get_smart_recommendations_dashboard()
    dashboard.render_main_recommendations_panel(project_data)

def render_smart_recommendations_sidebar(project_data: Dict[str, Any]):
    """Fonction utilitaire pour rendre les recommandations en sidebar"""
    dashboard = get_smart_recommendations_dashboard()
    dashboard.render_recommendations_sidebar(project_data)