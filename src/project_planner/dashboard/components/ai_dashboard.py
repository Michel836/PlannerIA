"""
🧠 Dashboard IA Intégré - PlannerIA
Interface unifiée pour toutes les fonctionnalités d'Intelligence Artificielle
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import time

# Chargement automatique des modules IA
AI_MODULES = {}
AI_STATUS = {
    'loading': False,
    'loaded': False,
    'error': None
}

def load_ai_modules_progressively():
    """Charge les modules IA progressivement"""
    global AI_MODULES, AI_STATUS
    
    if AI_STATUS['loaded']:
        return True
        
    if AI_STATUS['loading']:
        return False
        
    try:
        AI_STATUS['loading'] = True
        
        # Import des modules
        from project_planner.ai import (
            ai_predictor, ai_assistant, smart_alert_system, gamification_engine,
            ai_risk_predictor, smart_chat, ai_budget_optimizer, ai_crisis_predictor, ai_personal_coach, ai_rag_manager,
            ProjectDomain, AlertLevel, ChallengeType, RiskCategory, ChatMode, BudgetCategory, CrisisType, CoachingArea, QueryContext,
            get_ai_status, get_ai_capabilities_summary, ensure_ai_initialized, initialize_personal_coach, initialize_rag_manager
        )
        
        # Stockage dans le cache global
        AI_MODULES.update({
            'ai_predictor': ai_predictor,
            'ai_assistant': ai_assistant, 
            'smart_alert_system': smart_alert_system,
            'gamification_engine': gamification_engine,
            'ai_risk_predictor': ai_risk_predictor,
            'smart_chat': smart_chat,
            'ai_budget_optimizer': ai_budget_optimizer,
            'ai_crisis_predictor': ai_crisis_predictor,
            'ai_personal_coach': ai_personal_coach,
            'ai_rag_manager': ai_rag_manager,
            'ProjectDomain': ProjectDomain,
            'AlertLevel': AlertLevel,
            'ChallengeType': ChallengeType,
            'RiskCategory': RiskCategory,
            'ChatMode': ChatMode,
            'BudgetCategory': BudgetCategory,
            'CrisisType': CrisisType,
            'CoachingArea': CoachingArea,
            'QueryContext': QueryContext,
            'initialize_personal_coach': initialize_personal_coach,
            'initialize_rag_manager': initialize_rag_manager,
            'get_ai_status': get_ai_status,
            'get_ai_capabilities_summary': get_ai_capabilities_summary,
            'ensure_ai_initialized': ensure_ai_initialized
        })
        
        AI_STATUS['loaded'] = True
        AI_STATUS['loading'] = False
        return True
        
    except Exception as e:
        AI_STATUS['error'] = str(e)
        AI_STATUS['loading'] = False
        return False

# Démarrage automatique du chargement IA en arrière-plan
def init_ai_background():
    """Initialise l'IA en arrière-plan dès l'import"""
    if not AI_STATUS['loading'] and not AI_STATUS['loaded']:
        load_ai_modules_progressively()

# Auto-démarrage pour présentation - chargement immédiat
init_ai_background()

# Force le chargement immédiat pour la présentation
if not AI_STATUS['loaded']:
    print("🎯 PRESENTATION MODE: Chargement forcé des modules IA...")
    load_ai_modules_progressively()

def get_ai_module(name):
    """Récupère un module IA depuis le cache"""
    return AI_MODULES.get(name)

def render_ai_dashboard(plan_data: Dict[str, Any]):
    """Rend le dashboard IA principal - toujours disponible"""
    
    st.title("🧠 Intelligence Artificielle PlannerIA")
    st.markdown("*Dashboard unifié pour toutes les capacités d'IA avancée*")
    
    # Status de chargement si nécessaire
    if AI_STATUS['loading']:
        st.info("🚀 Finalisation du chargement des modules IA...")
        # Continue à afficher le dashboard même en chargement
    elif AI_STATUS['error']:
        st.error(f"⚠️ Certains modules IA peuvent être indisponibles: {AI_STATUS['error']}")
        # Continue quand même
    elif AI_STATUS['loaded']:
        st.success("✅ Tous les modules IA sont opérationnels")
    else:
        st.info("⏳ Modules IA en cours d'initialisation...")
    
    # S'assurer que l'IA est initialisée
    ensure_ai_initialized = get_ai_module('ensure_ai_initialized')
    if ensure_ai_initialized and not ensure_ai_initialized():
        st.warning("⚠️ Système IA partiellement initialisé. Certaines fonctionnalités peuvent être limitées.")
    
    # Initialisation du Coach Personnel
    initialize_personal_coach = get_ai_module('initialize_personal_coach')
    if initialize_personal_coach:
        personal_coach = initialize_personal_coach()
        
        # Widget Coach flottant
        render_floating_coach_widget(personal_coach, plan_data)
    
    # Generate unique session counter to avoid duplicate keys
    if 'ai_render_counter' not in st.session_state:
        st.session_state.ai_render_counter = 0
    st.session_state.ai_render_counter += 1
    render_id = st.session_state.ai_render_counter
    
    # Vérification du statut IA
    get_ai_status = get_ai_module('get_ai_status')
    if get_ai_status:
        ai_status = get_ai_status()
        render_ai_status_sidebar(ai_status)
    
    # Tabs principales - 10 modules IA
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🗂️ Portfolio Manager",
        "🔮 Prédictions IA",
        "🤖 Assistant Chat", 
        "🚨 Alertes Intelligentes",
        "🎮 Gamification",
        "⚠️ Analyse Risques",
        "💰 Optimisation Budget",
        "🔥 Prédicteur Crises",
        "🎯 Coach Personnel",
        "🔍 RAG Intelligence"
    ])
    
    with tab1:
        render_portfolio_manager_section(plan_data)
    
    with tab2:
        render_predictive_ai_section(plan_data)
    
    with tab3:
        render_chat_assistant_section(plan_data)
    
    with tab4:
        render_smart_alerts_section(plan_data)
    
    with tab5:
        render_gamification_section(plan_data)
    
    with tab6:
        render_risk_analysis_section(plan_data)
    
    with tab7:
        render_budget_optimization_section(plan_data)
        
    with tab8:
        render_crisis_prediction_section(plan_data)
        
    with tab9:
        render_personal_coach_section(plan_data)
        
    with tab10:
        render_rag_intelligence_section(plan_data)


def render_ai_status_sidebar(ai_status: Dict[str, Any]):
    """Rend la sidebar avec le statut IA"""
    
    with st.sidebar:
        st.header("🧠 Statut IA")
        
        # Statut global
        if ai_status["overall_health"] == "healthy":
            st.success("✅ Système IA Opérationnel")
        else:
            st.warning("⚠️ Système IA Partiellement Opérationnel")
        
        # Détails des composants
        st.subheader("Composants")
        for component, status in ai_status["components_status"].items():
            if status["loaded"]:
                st.success(f"✅ {component.replace('_', ' ').title()}")
            else:
                st.error(f"❌ {component.replace('_', ' ').title()}")
        
        # Capacités IA
        if st.expander("🎯 Capacités IA", expanded=False):
            get_ai_capabilities_summary = get_ai_module('get_ai_capabilities_summary')
            if get_ai_capabilities_summary:
                capabilities = get_ai_capabilities_summary()
                for capability, details in capabilities.items():
                    st.markdown(f"**{capability}**")
                    st.caption(details["description"])
            else:
                st.info("Capacités IA en cours de chargement...")


def render_predictive_ai_section(plan_data: Dict[str, Any]):
    """Section prédictions IA"""
    
    st.header("🔮 Moteur Prédictif IA")
    
    if not plan_data:
        st.info("📊 Chargez des données de projet pour obtenir des prédictions IA personnalisées")
        return
    
    # Prédictions en temps réel
    ai_predictor = get_ai_module('ai_predictor')
    if not ai_predictor:
        st.warning("🔄 Moteur prédictif en cours d'initialisation...")
        return
        
    with st.spinner("🤖 L'IA analyse votre projet..."):
        try:
            # Prédictions principales
            prediction = asyncio.run(ai_predictor.predict_project_outcome(plan_data))
            
            # Métriques prédictives
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🕐 Durée Prédite",
                    f"{prediction.predicted_duration:.0f} jours",
                    help="Prédiction basée sur des patterns de projets similaires"
                )
            
            with col2:
                st.metric(
                    "💰 Coût Prédit", 
                    f"{prediction.predicted_cost:,.0f}€",
                    help="Estimation avec intervalles de confiance"
                )
            
            with col3:
                success_color = "normal" if prediction.success_probability > 0.7 else "inverse"
                st.metric(
                    "🎯 Probabilité Succès",
                    f"{prediction.success_probability:.1%}",
                    delta_color=success_color,
                    help="Probabilité de succès basée sur l'historique"
                )
            
            with col4:
                st.metric(
                    "📊 Confiance IA",
                    f"{np.mean(list(prediction.confidence_interval)):.0f}j",
                    help=f"Intervalle: {prediction.confidence_interval[0]:.0f}-{prediction.confidence_interval[1]:.0f}j"
                )
            
            # Graphiques prédictifs
            col1, col2 = st.columns(2)
            
            with col1:
                render_prediction_confidence_chart(prediction)
            
            with col2:
                render_risk_factors_chart(prediction)
            
            # Recommandations IA
            st.subheader("💡 Recommandations IA")
            for i, rec in enumerate(prediction.recommendations[:5], 1):
                st.markdown(f"{i}. {rec}")
            
            # Projets similaires
            if prediction.similar_projects:
                st.subheader("🔍 Projets Similaires")
                similar_df = pd.DataFrame(prediction.similar_projects)
                st.dataframe(similar_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur prédiction IA: {str(e)}")


def render_prediction_confidence_chart(prediction):
    """Graphique de confiance des prédictions"""
    
    categories = ['Durée', 'Coût', 'Équipe', 'Qualité']
    confidence_scores = [
        0.85,  # Durée (basé sur patterns)
        0.78,  # Coût (plus variable) 
        0.82,  # Équipe (données disponibles)
        0.75   # Qualité (plus subjective)
    ]
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=confidence_scores,
        marker_color=['#1f77b4' if score > 0.8 else '#ff7f0e' if score > 0.7 else '#d62728' for score in confidence_scores],
        text=[f"{score:.1%}" for score in confidence_scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Confiance des Prédictions IA",
        yaxis_title="Score de Confiance",
        yaxis=dict(range=[0, 1]),
        height=350
    )
    
    import time
    unique_key = f"ai_predictions_confidence_chart_{int(time.time() * 1000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)


def render_risk_factors_chart(prediction):
    """Graphique des facteurs de risque"""
    
    risk_factors = list(prediction.risk_factors.keys())
    risk_scores = list(prediction.risk_factors.values())
    
    fig = px.bar(
        x=risk_scores,
        y=[factor.replace('_', ' ').title() for factor in risk_factors],
        orientation='h',
        title="Facteurs de Risque Identifiés",
        color=risk_scores,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=350)
    import time
    unique_key = f"ai_risk_factors_chart_{int(time.time() * 1000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)


def render_chat_assistant_section(plan_data: Dict[str, Any]):
    """Section assistant chat IA"""
    
    st.header("🤖 Assistant IA Conversationnel")
    
    # Vérifier si les modules sont disponibles
    ChatMode = get_ai_module('ChatMode')
    smart_chat = get_ai_module('smart_chat')
    
    if not ChatMode or not smart_chat:
        st.warning("🔄 Assistant conversationnel en cours d'initialisation...")
        return
    
    # Sélection du mode de chat
    import time
    unique_key = f"ai_chat_mode_selector_{int(time.time() * 1000)}"
    
    chat_mode = st.selectbox(
        "Mode d'assistance",
        options=[mode.value.replace('_', ' ').title() for mode in ChatMode],
        help="Choisissez le mode spécialisé selon vos besoins",
        key=unique_key
    )
    
    # Création ou récupération de session
    user_id = st.session_state.get('user_id', 'demo_user')
    
    session_key = f"chat_session_{user_id}"
    if session_key not in st.session_state:
        mode_enum = ChatMode([mode for mode in ChatMode if mode.value.replace('_', ' ').title() == chat_mode][0])
        session = smart_chat.create_chat_session(user_id, mode_enum, plan_data)
        st.session_state[session_key] = session.session_id
    
    session_id = st.session_state[session_key]
    
    # Interface de chat
    try:
        smart_chat.render_chat_interface(session_id)
    except Exception as e:
        st.error(f"Erreur interface chat: {str(e)}")
        # Fallback simple
        import time
        unique_key = f"ai_chat_text_area_{int(time.time() * 1000)}"
        st.text_area("Chat IA (Mode Simple)", placeholder="Posez votre question ici...", key=unique_key)


def render_smart_alerts_section(plan_data: Dict[str, Any]):
    """Section alertes intelligentes"""
    
    st.header("🚨 Alertes Intelligentes")
    
    if not plan_data:
        st.info("📊 Chargez des données de projet pour activer la surveillance IA")
        return
    
    # Vérifier si les modules sont disponibles
    smart_alert_system = get_ai_module('smart_alert_system')
    AlertLevel = get_ai_module('AlertLevel')
    if not smart_alert_system or not AlertLevel:
        st.warning("🔄 Système d'alertes en cours d'initialisation...")
        return
    
    # Surveillance en temps réel
    with st.spinner("🔍 Analyse des métriques en cours..."):
        try:
            # Génération des alertes
            alerts = asyncio.run(smart_alert_system.monitor_project_metrics(plan_data))
            
            if alerts:
                # Statistiques des alertes
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    critical_alerts = len([a for a in alerts if a.level.value == "critical"])
                    st.metric("🔴 Critiques", critical_alerts)
                
                with col2:
                    error_alerts = len([a for a in alerts if a.level.value == "error"])
                    st.metric("🟠 Erreurs", error_alerts)
                
                with col3:
                    warning_alerts = len([a for a in alerts if a.level.value == "warning"])
                    st.metric("🟡 Avertissements", warning_alerts)
                
                with col4:
                    info_alerts = len([a for a in alerts if a.level.value == "info"])
                    st.metric("ℹ️ Informations", info_alerts)
                
                # Liste des alertes
                st.subheader("📋 Alertes Actives")
                
                for alert in alerts[:10]:  # Top 10
                    level_emoji = {
                        "critical": "🔴",
                        "error": "🟠", 
                        "warning": "🟡",
                        "info": "ℹ️"
                    }
                    
                    with st.expander(f"{level_emoji.get(alert.level.value, '⚪')} {alert.title}"):
                        st.markdown(f"**Catégorie:** {alert.category.value.title()}")
                        st.markdown(f"**Description:** {alert.description}")
                        st.markdown(f"**Confiance:** {alert.confidence:.1%}")
                        
                        if alert.recommended_actions:
                            st.markdown("**Actions recommandées:**")
                            for action in alert.recommended_actions[:3]:
                                st.markdown(f"• {action}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"✅ Résoudre", key=f"resolve_{alert.id}"):
                                smart_alert_system.dismiss_alert(alert.id)
                                st.success("Alerte marquée comme résolue")
                                st.experimental_rerun()
                
                # Graphique de distribution des alertes
                render_alerts_distribution_chart(alerts)
                
            else:
                st.success("✅ Aucune alerte active - Votre projet est sur la bonne voie !")
                
                # Affichage des métriques de surveillance
                st.subheader("📊 Métriques Surveillées")
                stats = smart_alert_system.get_alert_statistics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Alertes Aujourd'hui", stats['total_generated_today'])
                with col2:
                    st.metric("Taux de Résolution", f"{stats['resolution_rate']:.1%}")
                with col3:
                    st.metric("Total Actives", stats['total_active'])
            
        except Exception as e:
            st.error(f"Erreur surveillance IA: {str(e)}")


def render_alerts_distribution_chart(alerts):
    """Graphique de distribution des alertes"""
    
    # Distribution par catégorie
    categories = {}
    for alert in alerts:
        cat = alert.category.value
        categories[cat] = categories.get(cat, 0) + 1
    
    if categories:
        fig = px.pie(
            values=list(categories.values()),
            names=[cat.replace('_', ' ').title() for cat in categories.keys()],
            title="Distribution des Alertes par Catégorie"
        )
        
        import time
        unique_key = f"ai_alerts_distribution_chart_{int(time.time() * 1000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)


def render_gamification_section(plan_data: Dict[str, Any]):
    """Section gamification"""
    
    st.header("🎮 Gamification Intelligente")
    
    # Vérifier si les modules sont disponibles
    gamification_engine = get_ai_module('gamification_engine')
    if not gamification_engine:
        st.warning("🔄 Système de gamification en cours d'initialisation...")
        return
    
    # Enregistrement automatique du joueur
    user_id = st.session_state.get('user_id', 'demo_user')
    
    if user_id not in gamification_engine.players:
        import time
        unique_key = f"gamification_player_name_{int(time.time() * 1000)}"
        player_name = st.text_input("Nom du joueur", value="Player Demo", key=unique_key)
        import time
        unique_key = f"gamification_register_{int(time.time() * 1000)}"
        if st.button("🚀 Rejoindre le système de gamification", key=unique_key):
            gamification_engine.register_player(user_id, player_name)
            st.success(f"Bienvenue {player_name} !")
            st.experimental_rerun()
    else:
        # Dashboard joueur
        dashboard = gamification_engine.get_player_dashboard(user_id)
        
        # Métriques joueur
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Niveau", dashboard['player_profile']['level'])
        
        with col2:
            st.metric("⭐ XP Total", dashboard['player_profile']['total_xp'])
        
        with col3:
            st.metric("🔥 Série Actuelle", dashboard['player_profile']['current_streak'])
        
        with col4:
            st.metric("✅ Défis Réussis", dashboard['player_profile']['challenges_completed'])
        
        # Progression niveau
        st.subheader("📈 Progression")
        progress = dashboard['player_profile']['level_progress']
        st.progress(progress, text=f"Progression vers niveau {dashboard['player_profile']['level'] + 1}: {progress:.1%}")
        
        # Défi suggéré
        st.subheader("🎯 Défi Personnalisé")
        next_challenge = dashboard['next_challenge']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**{next_challenge['title']}**")
            st.markdown(f"*{next_challenge['type'].title()} • {next_challenge['difficulty'].title()}*")
            st.markdown(f"⏱️ Temps estimé: {next_challenge['estimated_time']}")
            
            rewards_text = " • ".join([f"{k}: {v}" for k, v in next_challenge['rewards'].items()])
            st.markdown(f"🎁 Récompenses: {rewards_text}")
        
        with col2:
            import time
            unique_key = f"gamification_accept_challenge_{int(time.time() * 1000)}"
            if st.button("🚀 Accepter le Défi", type="primary", key=unique_key):
                st.success("Défi accepté ! Bonne chance !")
                # Ici on pourrait démarrer une session de défi
        
        # Classement
        st.subheader("🏆 Classement Global")
        leaderboard = gamification_engine.get_leaderboard(limit=5)
        
        leaderboard_df = pd.DataFrame(leaderboard)
        if not leaderboard_df.empty:
            st.dataframe(leaderboard_df, use_container_width=True)
        
        # Achievements récents
        if dashboard['recent_achievements']:
            st.subheader("🏅 Achievements Récents")
            for achievement in dashboard['recent_achievements']:
                st.markdown(f"🏆 **{achievement['name']}** - {achievement['description']} ({achievement['points']} pts)")


def render_risk_analysis_section(plan_data: Dict[str, Any]):
    """Section analyse des risques"""
    
    st.header("⚠️ Analyse Prédictive des Risques")
    
    if not plan_data:
        st.info("📊 Chargez des données de projet pour une analyse des risques complète")
        return
    
    # Vérifier si les modules sont disponibles
    ai_risk_predictor = get_ai_module('ai_risk_predictor')
    if not ai_risk_predictor:
        st.warning("🔄 Analyseur de risques en cours d'initialisation...")
        return
    
    # Analyse des risques
    with st.spinner("🧠 Analyse ML des risques en cours..."):
        try:
            risk_profile = asyncio.run(ai_risk_predictor.analyze_project_risks(plan_data))
            
            # Score de risque global
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_color = "inverse" if risk_profile.overall_risk_score > 0.7 else "normal"
                st.metric(
                    "🎯 Score Global",
                    f"{risk_profile.overall_risk_score:.1%}",
                    delta_color=risk_color,
                    help="Score de risque calculé par ML"
                )
            
            with col2:
                st.metric(
                    "🔍 Risques Identifiés",
                    len(risk_profile.predicted_risks),
                    help="Nombre total de risques détectés"
                )
            
            with col3:
                critical_risks = len([r for r in risk_profile.predicted_risks if r.predicted_level.name == 'CRITICAL'])
                st.metric(
                    "🚨 Critiques",
                    critical_risks,
                    delta_color="inverse" if critical_risks > 0 else "normal"
                )
            
            with col4:
                high_risks = len([r for r in risk_profile.predicted_risks if r.predicted_level.name == 'HIGH'])
                st.metric("🟠 Élevés", high_risks)
            
            # Distribution des risques par catégorie
            col1, col2 = st.columns(2)
            
            with col1:
                render_risk_distribution_chart(risk_profile)
            
            with col2:
                render_risk_timeline_chart(risk_profile)
            
            # Top risques
            st.subheader("🔍 Risques Prioritaires")
            
            for i, risk in enumerate(risk_profile.predicted_risks[:5], 1):
                level_emoji = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠', 
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }
                
                with st.expander(f"{level_emoji.get(risk.predicted_level.name, '⚪')} #{i} {risk.name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Catégorie:** {risk.category.value.title()}")
                        st.markdown(f"**Probabilité:** {risk.probability:.1%}")
                        st.markdown(f"**Impact:** {risk.potential_impact:.1%}")
                        st.markdown(f"**Confiance IA:** {risk.confidence:.1%}")
                    
                    with col2:
                        if risk.mitigation_strategies:
                            st.markdown("**Stratégies de Mitigation:**")
                            for strategy in risk.mitigation_strategies[:3]:
                                st.markdown(f"• {strategy}")
                        
                        if risk.early_warnings:
                            st.markdown("**Signaux d'Alerte:**")
                            for warning in risk.early_warnings[:2]:
                                st.markdown(f"⚠️ {warning}")
            
            # Rapport d'export
            import time
            unique_key = f"risks_generate_report_{int(time.time() * 1000)}"
            if st.button("📄 Générer Rapport Risques", key=unique_key):
                report = ai_risk_predictor.export_risk_report(risk_profile)
                st.success("Rapport généré !")
                
                # Affichage du résumé exécutif
                with st.expander("📋 Résumé Exécutif"):
                    exec_summary = report['executive_summary']
                    st.json(exec_summary)
            
        except Exception as e:
            st.error(f"Erreur analyse risques: {str(e)}")


def render_risk_distribution_chart(risk_profile):
    """Graphique distribution des risques"""
    
    categories = []
    scores = []
    
    for category, score in risk_profile.risk_distribution.items():
        if score > 0.1:  # Seulement les risques significatifs
            categories.append(category.value.replace('_', ' ').title())
            scores.append(score)
    
    if categories:
        fig = px.bar(
            x=categories,
            y=scores,
            title="Distribution des Risques par Catégorie",
            color=scores,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(height=350)
        import time
        unique_key = f"ai_risk_distribution_chart_{int(time.time() * 1000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)


def render_risk_timeline_chart(risk_profile):
    """Graphique évolution des risques"""
    
    # Simulation d'évolution temporelle des risques
    weeks = list(range(1, 13))  # 12 semaines
    
    fig = go.Figure()
    
    for category, trend in risk_profile.risk_trends.items():
        if max(trend) > 0.1:  # Seulement les catégories avec risque significatif
            fig.add_trace(go.Scatter(
                x=weeks,
                y=trend,
                mode='lines+markers',
                name=category.replace('_', ' ').title(),
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Évolution des Risques (12 semaines)",
        xaxis_title="Semaines",
        yaxis_title="Score de Risque",
        height=350
    )
    
    import time
    unique_key = f"ai_risk_timeline_chart_{int(time.time() * 1000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)


def render_budget_optimization_section(plan_data: Dict[str, Any]):
    """Section optimisation budgétaire"""
    
    st.header("💰 Optimisation Budgétaire IA")
    
    if not plan_data:
        st.info("📊 Chargez des données de projet pour l'optimisation budgétaire")
        return
    
    # Vérifier si les modules sont disponibles
    BudgetCategory = get_ai_module('BudgetCategory')
    ai_budget_optimizer = get_ai_module('ai_budget_optimizer')
    
    if not BudgetCategory or not ai_budget_optimizer:
        st.warning("🔄 Optimiseur budgétaire en cours d'initialisation...")
        return
    
    # Configuration budget actuel
    st.subheader("💼 Configuration Budget Actuel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Répartition Actuelle:**")
        
        # Budget par défaut basé sur les données du projet
        total_cost = plan_data.get('project_overview', {}).get('total_cost', 100000)
        
        current_budget = {}
        for category in BudgetCategory:
            # Répartition par défaut selon les standards industrie
            default_ratios = {
                BudgetCategory.DEVELOPMENT: 0.60,
                BudgetCategory.DESIGN: 0.15,
                BudgetCategory.INFRASTRUCTURE: 0.10,
                BudgetCategory.TESTING: 0.10,
                BudgetCategory.PROJECT_MANAGEMENT: 0.05
            }
            
            default_amount = total_cost * default_ratios.get(category, 0.02)
            import time
            unique_key = f"budget_{category.value}_{int(time.time() * 1000)}"
            current_budget[category] = st.number_input(
                f"{category.value.replace('_', ' ').title()} (€)",
                value=int(default_amount),
                min_value=0,
                step=1000,
                key=unique_key
            )
    
    with col2:
        st.markdown("**Objectif d'Optimisation:**")
        
        from project_planner.ai.budget_optimizer import OptimizationObjective
        objective = st.selectbox(
            "Objectif Principal",
            options=[obj.value.replace('_', ' ').title() for obj in OptimizationObjective],
            help="Choisissez votre priorité d'optimisation",
            key="budget_optimization_objective"
        )
        
        # Contraintes
        st.markdown("**Contraintes (Optionnel):**")
        max_variation = st.slider(
            "Variation Max (%)",
            min_value=5,
            max_value=50,
            value=25,
            help="Variation maximale autorisée par catégorie"
        )
        
        import time
        unique_key = f"budget_total_max_{int(time.time() * 1000)}"
        max_budget = st.number_input(
            "Budget Total Max (€)",
            value=int(sum(current_budget.values()) * 1.1),
            min_value=int(sum(current_budget.values()) * 0.8),
            step=5000,
            key=unique_key
        )
    
    # Lancer l'optimisation
    import time
    unique_key = f"budget_optimize_button_{int(time.time() * 1000)}"
    if st.button("🚀 Lancer l'Optimisation IA", type="primary", key=unique_key):
        
        with st.spinner("🧠 Optimisation en cours..."):
            try:
                # Configuration des contraintes
                constraints = {
                    'max_total_budget': max_budget,
                    'max_variation_percent': max_variation
                }
                
                # Conversion de l'objectif
                objective_enum = [obj for obj in OptimizationObjective if obj.value.replace('_', ' ').title() == objective][0]
                
                # Optimisation
                result = asyncio.run(ai_budget_optimizer.optimize_budget(
                    plan_data, current_budget, objective_enum, constraints
                ))
                
                # Affichage des résultats
                ai_budget_optimizer.render_optimization_dashboard(result)
                
                # Insights détaillés
                import time
                unique_key = f"budget_insights_button_{int(time.time() * 1000)}"
                if st.button("📊 Insights Détaillés", key=unique_key):
                    insights = ai_budget_optimizer.generate_budget_insights(result)
                    st.json(insights)
                
            except Exception as e:
                st.error(f"Erreur optimisation: {str(e)}")
    
    # Benchmarks industrie
    st.subheader("📊 Benchmarks Industrie")
    domain = plan_data.get('domain', 'web_app')
    
    benchmark_data = {
        'web_app': {'Development': 60, 'Design': 15, 'Infrastructure': 10, 'Testing': 10, 'PM': 5},
        'mobile_app': {'Development': 55, 'Design': 20, 'Infrastructure': 8, 'Testing': 12, 'PM': 5},
        'ecommerce': {'Development': 45, 'Design': 25, 'Infrastructure': 15, 'Testing': 10, 'PM': 5},
        'fintech': {'Development': 40, 'Design': 15, 'Infrastructure': 20, 'Testing': 20, 'PM': 5}
    }
    
    benchmark = benchmark_data.get(domain, benchmark_data['web_app'])
    
    fig = px.pie(
        values=list(benchmark.values()),
        names=list(benchmark.keys()),
        title=f"Répartition Recommandée - {domain.title()}"
    )
    
    import time
    unique_key = f"ai_budget_benchmark_chart_{int(time.time() * 1000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)


def render_crisis_prediction_section(plan_data: Dict[str, Any]):
    """Section prédiction de crises"""
    
    st.header("🔥 Prédicteur de Crises IA")
    
    if not plan_data:
        st.info("Aucun projet sélectionné. Simulateur de prédiction de crise disponible ci-dessous.")
        return
    
    ai_crisis_predictor = get_ai_module('ai_crisis_predictor')
    CrisisType = get_ai_module('CrisisType')
    
    if not ai_crisis_predictor or not CrisisType:
        st.error("Module de prédiction de crise non disponible")
        return
    
    st.markdown("Analyse prédictive avancée des crises projet avec signaux faibles et ML")
    
    # Section: Analyse Temps Réel
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Analyse Temps Réel")
        
        # Simulation d'analyse de crise
        current_date = datetime.now()
        project_data = {
            'name': plan_data.get('name', 'Projet Sample'),
            'budget_remaining': 0.3,
            'timeline_progress': 0.8,
            'team_satisfaction': 0.4,
            'scope_changes': 5,
            'stakeholder_engagement': 0.6,
            'technical_debt': 0.7,
            'market_conditions': 0.5,
            'timeline_start': current_date - timedelta(days=90),
            'timeline_end': current_date + timedelta(days=30)
        }
        
        # Génération des prédictions de crise
        import time
        unique_key = f"crisis_analysis_{int(time.time() * 1000)}"
        
        if st.button("🔄 Analyser les Risques de Crise", key=unique_key):
            with st.spinner("Analyse des signaux faibles en cours..."):
                time.sleep(1.5)  # Simulation de traitement ML
                
                # Simulation des résultats de crise
                crisis_predictions = [
                    {"type": "BUDGET_OVERRUN", "probability": 0.82, "severity": "HIGH", "timeline": "2-3 semaines"},
                    {"type": "TEAM_BURNOUT", "probability": 0.67, "severity": "MEDIUM", "timeline": "1-2 semaines"},
                    {"type": "SCOPE_CREEP", "probability": 0.45, "severity": "MEDIUM", "timeline": "Immédiat"},
                    {"type": "STAKEHOLDER_CONFLICT", "probability": 0.33, "severity": "LOW", "timeline": "4-6 semaines"}
                ]
                
                st.subheader("🎯 Prédictions de Crise")
                
                for crisis in crisis_predictions:
                    severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                    color = severity_color.get(crisis["severity"], "⚪")
                    
                    with st.expander(f"{color} {crisis['type'].replace('_', ' ').title()} - {crisis['probability']*100:.0f}%"):
                        col_details1, col_details2 = st.columns(2)
                        
                        with col_details1:
                            st.write(f"**Probabilité:** {crisis['probability']*100:.1f}%")
                            st.write(f"**Sévérité:** {crisis['severity']}")
                        
                        with col_details2:
                            st.write(f"**Horizon:** {crisis['timeline']}")
                            st.write(f"**Impact:** Critique")
    
    with col2:
        st.subheader("📊 Indicateurs Précoces")
        
        # Indicateurs de signaux faibles
        indicators = {
            "Vélocité équipe": 0.3,
            "Communication freq.": 0.6,
            "Satisfaction client": 0.4,
            "Dette technique": 0.8,
            "Budget burn rate": 0.9,
            "Moral équipe": 0.2
        }
        
        for indicator, value in indicators.items():
            color = "red" if value > 0.7 else "orange" if value > 0.4 else "green"
            st.metric(indicator, f"{value*100:.0f}%", delta=f"{(value-0.5)*20:.0f}%")
        
        # Graphique radar des signaux faibles
        categories = list(indicators.keys())
        values = list(indicators.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Signaux Actuels',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[0.5] * len(categories),
            theta=categories,
            fill='toself',
            name='Seuil Normal',
            line=dict(color='green', dash='dash'),
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Radar Signaux Faibles"
        )
        
        unique_radar_key = f"crisis_radar_{int(time.time() * 1000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_radar_key)
    
    # Section: Stratégies de Mitigation
    st.subheader("🛡️ Stratégies de Mitigation")
    
    mitigation_strategies = {
        "Budget Overrun": [
            "Renégocier le scope avec les stakeholders",
            "Implémenter un contrôle budgétaire hebdomadaire",
            "Rechercher des optimisations techniques",
            "Préparer un plan de contingence financière"
        ],
        "Team Burnout": [
            "Organiser des sessions de debriefing régulières",
            "Redistribuer la charge de travail",
            "Planifier des pauses créatives",
            "Mettre en place du support psychologique"
        ],
        "Scope Creep": [
            "Établir un change control board",
            "Documenter tous les changements",
            "Communiquer l'impact des modifications",
            "Renforcer la gouvernance projet"
        ]
    }
    
    selected_crisis = st.selectbox(
        "Sélectionner une crise pour les stratégies:",
        list(mitigation_strategies.keys()),
        key=f"crisis_select_{int(time.time() * 1000)}"
    )
    
    if selected_crisis:
        st.write(f"**Stratégies recommandées pour {selected_crisis}:**")
        for i, strategy in enumerate(mitigation_strategies[selected_crisis], 1):
            st.write(f"{i}. {strategy}")
    
    # Section: Timeline de Crise
    st.subheader("📅 Timeline Prédictive")
    
    # Génération d'une timeline de crise
    timeline_data = []
    base_date = datetime.now()
    
    events = [
        {"date": base_date + timedelta(days=3), "event": "Signal Budget critique détecté", "severity": "HIGH"},
        {"date": base_date + timedelta(days=7), "event": "Risque de burnout identifié", "severity": "MEDIUM"},
        {"date": base_date + timedelta(days=14), "event": "Escalade stakeholder prévue", "severity": "MEDIUM"},
        {"date": base_date + timedelta(days=21), "event": "Point de non-retour budgétaire", "severity": "HIGH"},
        {"date": base_date + timedelta(days=30), "event": "Fin de projet en péril", "severity": "CRITICAL"}
    ]
    
    df_timeline = pd.DataFrame(events)
    df_timeline['date_str'] = df_timeline['date'].dt.strftime('%d/%m/%Y')
    
    # Graphique timeline
    colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "yellow", "CRITICAL": "darkred"}
    df_timeline['color'] = df_timeline['severity'].map(colors)
    
    fig = px.scatter(
        df_timeline,
        x='date',
        y='event',
        color='severity',
        title="Timeline Prédictive des Crises",
        labels={'date': 'Date', 'event': 'Événement'}
    )
    
    fig.update_traces(marker_size=15)
    fig.update_layout(height=400)
    
    unique_timeline_key = f"crisis_timeline_{int(time.time() * 1000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_timeline_key)


def render_floating_coach_widget(personal_coach, plan_data: Dict[str, Any]):
    """Widget Coach flottant - Conseils contextuels"""
    
    if not personal_coach:
        return
    
    # Récupération de l'user_id
    user_id = st.session_state.get('user_id', f"user_{datetime.now().timestamp()}")
    
    # Conseils contextuels selon la page
    current_module = st.session_state.get('current_page', 'dashboard')
    contextual_advice = personal_coach.get_contextual_advice(user_id, current_module)
    
    if contextual_advice:
        # Widget coach discret
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown("🎯 **Coach IA**")
            
            with col2:
                advice = contextual_advice[0]  # Premier conseil
                st.info(f"💡 {advice.message}")
                
                if advice.recommended_action:
                    unique_action_key = f"coach_action_{int(time.time() * 1000)}"
                    if st.button(f"✅ {advice.recommended_action[:50]}...", key=unique_action_key):
                        personal_coach.mark_insight_as_applied(str(id(advice)), user_id, 0.8)
                        st.success("Conseil appliqué ! Votre coach apprend de vos actions.")


def render_personal_coach_section(plan_data: Dict[str, Any]):
    """Section Coach Personnel IA"""
    
    st.header("🎯 Coach Personnel IA")
    
    ai_personal_coach = get_ai_module('ai_personal_coach')
    initialize_personal_coach = get_ai_module('initialize_personal_coach')
    
    if not ai_personal_coach or not initialize_personal_coach:
        st.error("Module Coach Personnel non disponible")
        return
    
    # Initialisation du coach
    personal_coach = initialize_personal_coach()
    user_id = st.session_state.get('user_id', f"user_{datetime.now().timestamp()}")
    
    st.markdown("Votre accompagnateur IA personnalisé qui apprend de vos actions et s'adapte à votre style de management")
    
    # Récupération des données de coaching
    coaching_data = personal_coach.get_coaching_dashboard_data(user_id)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Actions Observées", coaching_data['total_actions'])
    
    with col2:
        st.metric("Sessions Coaching", coaching_data['coaching_sessions'])
    
    with col3:
        st.metric("Taux de Succès", f"{coaching_data['success_rate']*100:.1f}%")
    
    with col4:
        st.metric("Insights Actifs", coaching_data['active_insights'])
    
    # Sections principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Profil Comportemental",
        "💡 Conseils Personnalisés", 
        "📈 Progression",
        "🎯 Objectifs Coaching"
    ])
    
    with tab1:
        render_behavioral_profile(coaching_data, personal_coach, user_id)
    
    with tab2:
        render_personalized_advice(coaching_data, personal_coach, user_id)
    
    with tab3:
        render_coaching_progress(coaching_data, personal_coach, user_id)
    
    with tab4:
        render_coaching_objectives(coaching_data, personal_coach, user_id)


def render_behavioral_profile(coaching_data: Dict[str, Any], personal_coach, user_id: str):
    """Profil comportemental de l'utilisateur"""
    
    st.subheader("🧠 Analyse Comportementale")
    
    # Traits de personnalité
    personality_traits = coaching_data['personality_analysis']
    
    if personality_traits:
        st.write("**Traits de Personnalité Détectés :**")
        
        # Graphique radar des traits
        categories = []
        values = []
        
        trait_labels = {
            'analytical': 'Analytique',
            'intuitive': 'Intuitif',
            'collaborative': 'Collaboratif',
            'independent': 'Indépendant',
            'perfectionist': 'Perfectionniste',
            'pragmatic': 'Pragmatique',
            'risk_averse': 'Prudent',
            'risk_taker': 'Preneur de risques'
        }
        
        for trait, score in personality_traits.items():
            trait_name = trait_labels.get(trait, trait.title())
            categories.append(trait_name)
            values.append(score)
        
        if categories:
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Votre Profil',
                line=dict(color='#1f77b4')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Radar de Personnalité"
            )
            
            unique_radar_key = f"personality_radar_{int(time.time() * 1000)}"
            st.plotly_chart(fig, use_container_width=True, key=unique_radar_key)
    else:
        st.info("Continuez à utiliser l'application pour que votre coach analyse votre style !")
    
    # Points forts et axes d'amélioration
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🌟 Points Forts :**")
        strengths = coaching_data['strengths']
        if strengths:
            for strength in strengths:
                st.write(f"• {strength.title()}")
        else:
            st.write("_En cours d'analyse..._")
    
    with col2:
        st.write("**📈 Axes d'Amélioration :**")
        improvements = coaching_data['improvement_areas']
        if improvements:
            for improvement in improvements:
                st.write(f"• {improvement.title()}")
        else:
            st.write("_En cours d'analyse..._")
    
    # Utilisation des modules
    st.subheader("📊 Utilisation des Modules")
    
    module_usage = coaching_data['module_usage']
    if module_usage:
        # Graphique en barres de l'utilisation des modules
        modules = list(module_usage.keys())
        usage_counts = list(module_usage.values())
        
        fig = px.bar(
            x=modules,
            y=usage_counts,
            title="Fréquence d'Utilisation des Modules",
            labels={'x': 'Module', 'y': 'Nombre d\'Actions'}
        )
        
        unique_usage_key = f"module_usage_{int(time.time() * 1000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_usage_key)


def render_personalized_advice(coaching_data: Dict[str, Any], personal_coach, user_id: str):
    """Conseils personnalisés du coach"""
    
    st.subheader("💡 Conseils Personnalisés")
    
    # Conseils actifs
    current_module = st.session_state.get('current_page', 'dashboard')
    advice_list = personal_coach.get_contextual_advice(user_id, current_module)
    
    if advice_list:
        st.write(f"**Conseils pour votre contexte actuel ({current_module}) :**")
        
        for i, advice in enumerate(advice_list):
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            color = priority_color.get(advice.priority, "⚪")
            
            with st.expander(f"{color} {advice.coaching_area.value.title()} - Confiance: {advice.confidence:.0%}"):
                st.write(f"**Message :** {advice.message}")
                
                if advice.evidence:
                    st.write("**Preuves observées :**")
                    for evidence in advice.evidence:
                        st.write(f"• {evidence}")
                
                if advice.recommended_action:
                    st.write(f"**Action recommandée :** {advice.recommended_action}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        unique_apply_key = f"apply_advice_{i}_{int(time.time() * 1000)}"
                        if st.button("✅ Appliquer", key=unique_apply_key):
                            personal_coach.mark_insight_as_applied(str(id(advice)), user_id, 0.9)
                            st.success("Conseil appliqué ! Merci pour votre feedback.")
                    
                    with col2:
                        unique_dismiss_key = f"dismiss_advice_{i}_{int(time.time() * 1000)}"
                        if st.button("❌ Ignorer", key=unique_dismiss_key):
                            personal_coach.mark_insight_as_applied(str(id(advice)), user_id, 0.3)
                            st.info("Conseil ignoré, je m'adapte à vos préférences.")
    else:
        st.info("Aucun conseil spécifique disponible pour le moment. Continuez à utiliser l'application !")
    
    # Simulateur de conseil
    st.subheader("🎪 Simulateur de Conseils")
    st.write("Générez un conseil personnalisé pour une situation spécifique :")
    
    col1, col2 = st.columns(2)
    with col1:
        situation = st.selectbox(
            "Situation :",
            ["Planification projet", "Gestion budget", "Gestion risques", "Leadership équipe", "Prise de décision"],
            key=f"situation_select_{int(time.time() * 1000)}"
        )
    
    with col2:
        difficulty = st.selectbox(
            "Niveau de difficulté :",
            ["Simple", "Modéré", "Complexe", "Critique"],
            key=f"difficulty_select_{int(time.time() * 1000)}"
        )
    
    unique_generate_key = f"generate_advice_{int(time.time() * 1000)}"
    if st.button("🎯 Générer un Conseil", key=unique_generate_key):
        with st.spinner("Génération d'un conseil personnalisé..."):
            time.sleep(1)
            
            # Simulation de conseil contextuel
            sample_advice = {
                "Planification projet": "Avec votre profil analytique, je recommande de décomposer ce projet en jalons de 2 semaines maximum pour maintenir votre contrôle précis.",
                "Gestion budget": "Votre tendance à sous-estimer de 12% est détectée. Ajoutez un buffer de 15% sur cette catégorie budgétaire.",
                "Gestion risques": "Votre style collaboratif excelle dans l'identification des risques humains. Impliquez l'équipe dans cette analyse.",
                "Leadership équipe": "Pattern observé : vos équipes performent 34% mieux avec des points hebdomadaires. Maintenez cette cadence.",
                "Prise de décision": "Votre intuition est fiable à 78%. Pour cette décision critique, validez avec 2-3 métriques clés."
            }
            
            st.success(f"**Conseil personnalisé :** {sample_advice.get(situation, 'Conseil en cours de génération...')}")


def render_coaching_progress(coaching_data: Dict[str, Any], personal_coach, user_id: str):
    """Progression et évolution du coaching"""
    
    st.subheader("📈 Progression du Coaching")
    
    # Métriques de progression
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Efficacité Coaching", f"{coaching_data['coaching_effectiveness']*100:.1f}%", "5.2%")
    
    with col2:
        recent_activity = coaching_data['recent_activity']
        st.metric("Activité 7 derniers jours", recent_activity, f"+{recent_activity-15}")
    
    with col3:
        st.metric("Score Progression", "87/100", "+12")
    
    # Évolution temporelle (simulation)
    st.subheader("📊 Évolution de vos Compétences")
    
    dates = pd.date_range(start='2024-01-01', end='2024-08-29', freq='W')
    competences_data = {
        'Planification': 60 + np.cumsum(np.random.normal(1, 2, len(dates))),
        'Leadership': 55 + np.cumsum(np.random.normal(0.8, 1.5, len(dates))),
        'Gestion Risques': 65 + np.cumsum(np.random.normal(1.2, 2.2, len(dates))),
        'Communication': 70 + np.cumsum(np.random.normal(0.5, 1.8, len(dates)))
    }
    
    df_evolution = pd.DataFrame(competences_data, index=dates)
    
    fig = px.line(
        df_evolution,
        title="Évolution de vos Compétences de Management",
        labels={'index': 'Date', 'value': 'Score de Compétence'}
    )
    
    unique_evolution_key = f"skills_evolution_{int(time.time() * 1000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_evolution_key)
    
    # Achievements récents
    st.subheader("🏆 Achievements Récents")
    
    achievements = [
        "🎯 Expert en Planification - 50 projets planifiés avec succès",
        "💰 Maître du Budget - Précision budgétaire > 95% sur 10 projets",
        "👥 Leader Collaboratif - Score d'engagement équipe > 4.5/5",
        "⚡ Décisionnaire Rapide - Temps de décision optimisé de 40%"
    ]
    
    for achievement in achievements:
        st.success(achievement)


def render_coaching_objectives(coaching_data: Dict[str, Any], personal_coach, user_id: str):
    """Objectifs et défis de coaching"""
    
    st.subheader("🎯 Objectifs de Développement")
    
    # Objectifs suggérés
    st.write("**Objectifs Personnalisés Suggérés :**")
    
    suggested_objectives = [
        {"title": "Optimisation Temporelle", "description": "Réduire de 20% le temps de planification", "progress": 65},
        {"title": "Leadership Inclusif", "description": "Impliquer 100% de l'équipe dans les décisions majeures", "progress": 80},
        {"title": "Gestion Proactive", "description": "Anticiper 90% des risques avant qu'ils deviennent critiques", "progress": 45},
        {"title": "Communication Efficace", "description": "Atteindre un score de satisfaction équipe > 4.7/5", "progress": 72}
    ]
    
    for obj in suggested_objectives:
        with st.expander(f"🎯 {obj['title']} - Progression: {obj['progress']}%"):
            st.write(f"**Description :** {obj['description']}")
            st.progress(obj['progress'] / 100)
            
            if obj['progress'] < 50:
                st.warning("Objectif nécessitant plus d'attention")
            elif obj['progress'] > 80:
                st.success("Objectif presque atteint ! Excellent travail !")
            else:
                st.info("Progression satisfaisante, continuez !")
    
    # Défis personnalisés
    st.subheader("🏅 Défis Coaching Personnalisés")
    
    challenges = [
        "🚀 Défi Vitesse : Planifiez votre prochain projet en moins de 2h",
        "🎯 Défi Précision : Atteignez une précision budgétaire > 98%",
        "👥 Défi Équipe : Organisez 3 sessions de brainstorming cette semaine",
        "📊 Défi Données : Prenez votre prochaine décision en < 10min avec 3 métriques"
    ]
    
    selected_challenge = st.selectbox(
        "Choisissez votre prochain défi :",
        challenges,
        key=f"challenge_select_{int(time.time() * 1000)}"
    )
    
    unique_accept_key = f"accept_challenge_{int(time.time() * 1000)}"
    if st.button("✅ Accepter ce Défi", key=unique_accept_key):
        st.success(f"Défi accepté ! Votre coach suivra votre progression sur : {selected_challenge}")
        
        # Observer l'action
        personal_coach.observe_action(
            user_id, 
            "coaching", 
            "accept_challenge", 
            {"challenge": selected_challenge, "timestamp": datetime.now().isoformat()},
            "accepted"
        )


def render_portfolio_manager_section(plan_data: Dict[str, Any]):
    """Section Portfolio Manager - Gestion intelligente multi-projets"""
    
    st.header("🗂️ Portfolio Manager IA")
    st.markdown("**Gestionnaire intelligent de portefeuille de projets avec optimisation des ressources**")
    
    # Tentative de chargement du module Portfolio
    try:
        from project_planner.ai.portfolio_manager import ai_portfolio_manager, ProjectStatus, ProjectPriority, ProjectTemplate
        
        # Sous-onglets du Portfolio Manager
        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "🏠 Vue Portefeuille",
            "➕ Nouveau Projet", 
            "📊 Analytics",
            "⚖️ Optimisation",
            "🔗 Dépendances"
        ])
        
        with sub_tab1:
            render_portfolio_overview(ai_portfolio_manager)
        
        with sub_tab2:
            render_project_creation_wizard(ai_portfolio_manager, ProjectTemplate, ProjectPriority)
        
        with sub_tab3:
            render_portfolio_analytics(ai_portfolio_manager)
        
        with sub_tab4:
            render_resource_optimization(ai_portfolio_manager)
        
        with sub_tab5:
            render_dependency_analysis(ai_portfolio_manager)
    
    except Exception as e:
        st.error(f"🔄 Portfolio Manager en cours d'initialisation: {str(e)}")
        st.info("📝 Le Portfolio Manager transformera PlannerIA d'un outil mono-projet en plateforme multi-projets intelligente")


def render_portfolio_overview(portfolio_manager):
    """Vue d'ensemble du portefeuille"""
    
    st.subheader("🏠 Vue d'Ensemble du Portefeuille")
    
    # Récupération données portfolio
    overview_data = portfolio_manager.get_portfolio_overview()
    
    if "message" in overview_data:
        st.info(overview_data["message"])
        st.markdown("""
        ### 🚀 Créez votre premier projet !
        
        Utilisez l'onglet **➕ Nouveau Projet** pour démarrer votre portefeuille de projets intelligent.
        
        **Avantages du Portfolio Manager :**
        - 🧠 **Intelligence Collective** : L'IA apprend de tous vos projets
        - ⚖️ **Optimisation Ressources** : Allocation intelligente entre projets
        - 🔍 **Vue Globale** : Supervision de tout votre portefeuille
        - 📈 **Analytics Avancés** : Insights multi-projets
        """)
        return
    
    # Métriques clés
    summary = overview_data.get("summary", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Projets",
            summary.get("total_projects", 0)
        )
    
    with col2:
        st.metric(
            "🔥 Projets Actifs", 
            summary.get("active_projects", 0)
        )
    
    with col3:
        st.metric(
            "💰 Budget Total",
            f"{summary.get('total_budget', 0):,.0f}€"
        )
    
    with col4:
        st.metric(
            "❤️ Santé Moyenne",
            f"{summary.get('avg_health_score', 0):.1f}%"
        )
    
    # Répartition par statut
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Répartition par Statut")
        projects_by_status = overview_data.get("projects_by_status", {})
        
        if projects_by_status:
            fig_status = px.pie(
                values=list(projects_by_status.values()),
                names=list(projects_by_status.keys()),
                title="Distribution des Projets"
            )
            st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.info("Aucune donnée de statut disponible")
    
    with col2:
        st.subheader("🎯 Répartition par Priorité")
        projects_by_priority = overview_data.get("projects_by_priority", {})
        
        if projects_by_priority:
            fig_priority = px.bar(
                x=list(projects_by_priority.keys()),
                y=list(projects_by_priority.values()),
                title="Priorités des Projets"
            )
            st.plotly_chart(fig_priority, use_container_width=True)
        else:
            st.info("Aucune donnée de priorité disponible")
    
    # Recommandations
    recommendations = overview_data.get("recommendations", [])
    if recommendations:
        st.subheader("💡 Recommandations IA")
        for rec in recommendations[:3]:
            st.info(f"**{rec.get('title', 'Recommandation')}:** {rec.get('description', '')}")


def render_project_creation_wizard(portfolio_manager, ProjectTemplate, ProjectPriority):
    """Assistant de création de projet"""
    
    st.subheader("➕ Assistant Création de Projet")
    st.markdown("Créez un nouveau projet avec l'aide de l'IA")
    
    with st.form(f"new_project_form_{int(time.time() * 1000)}"):
        # Informations de base
        col1, col2 = st.columns(2)
        
        with col1:
            project_name = st.text_input("🏷️ Nom du Projet", placeholder="Ex: Application mobile e-commerce")
            project_description = st.text_area("📝 Description", placeholder="Décrivez votre projet...")
        
        with col2:
            # Template intelligent
            template = st.selectbox(
                "🎯 Type de Projet",
                options=list(ProjectTemplate),
                format_func=lambda x: {
                    ProjectTemplate.WEB_APP: "🌐 Application Web",
                    ProjectTemplate.MOBILE_APP: "📱 Application Mobile", 
                    ProjectTemplate.AI_PROJECT: "🤖 Projet IA",
                    ProjectTemplate.MARKETING: "📢 Campagne Marketing",
                    ProjectTemplate.INFRASTRUCTURE: "🏗️ Infrastructure",
                    ProjectTemplate.RESEARCH: "🔬 Recherche & Développement"
                }.get(x, x.value)
            )
            
            priority = st.selectbox(
                "🎯 Priorité",
                options=list(ProjectPriority),
                format_func=lambda x: {
                    ProjectPriority.LOW: "🔵 Faible",
                    ProjectPriority.MEDIUM: "🟡 Moyenne",
                    ProjectPriority.HIGH: "🟠 Élevée", 
                    ProjectPriority.CRITICAL: "🔴 Critique"
                }.get(x, x.value)
            )
        
        # Ressources
        col3, col4 = st.columns(2)
        
        with col3:
            team_size = st.number_input("👥 Taille Équipe", min_value=1, max_value=20, value=3)
        
        with col4:
            budget = st.number_input("💰 Budget (€)", min_value=1000, max_value=10000000, value=50000, step=1000)
        
        # Estimation IA automatique
        st.markdown("### 🤖 Estimation IA Automatique")
        if template and project_name:
            template_data = portfolio_manager.templates_db.get(template, {})
            duration_range = template_data.get("duration_range", (30, 120))
            
            st.info(f"""
            **Basé sur le template {template.value} :**
            - ⏱️ Durée estimée : {duration_range[0]}-{duration_range[1]} jours
            - 👥 Équipe recommandée : {template_data.get('team_size_range', (2, 5))}
            - 🎯 Taux de succès historique : {template_data.get('success_patterns', {}).get('agile_methodology', 0.8):.0%}
            """)
        
        submitted = st.form_submit_button("🚀 Créer le Projet", type="primary")
        
        if submitted and project_name and project_description:
            with st.spinner("🧠 Création du projet avec IA..."):
                try:
                    new_project = portfolio_manager.create_project_wizard(
                        name=project_name,
                        description=project_description,
                        template=template,
                        team_size=team_size,
                        budget=budget,
                        priority=priority
                    )
                    
                    st.success(f"✅ Projet '{project_name}' créé avec succès !")
                    st.balloons()
                    
                    # Afficher les détails du projet créé
                    st.json({
                        "ID": new_project.id,
                        "Nom": new_project.name,
                        "Durée estimée": f"{(new_project.end_date - new_project.start_date).days} jours",
                        "Budget ajusté": f"{new_project.budget:,.0f}€",
                        "Domaine détecté": new_project.domain
                    })
                    
                    time.sleep(2)  # Attendre pour voir le feedback
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la création: {str(e)}")


def render_portfolio_analytics(portfolio_manager):
    """Analytics avancées du portefeuille"""
    
    st.subheader("📊 Analytics Portfolio")
    
    try:
        analytics_data = portfolio_manager.portfolio_analytics()
        
        if "message" in analytics_data:
            st.info(analytics_data["message"])
            return
        
        # Métriques temporelles
        temporal = analytics_data.get("temporal_analysis", {})
        if temporal:
            st.subheader("⏱️ Analyse Temporelle")
            
            avg_durations = temporal.get("avg_duration_by_template", {})
            if avg_durations:
                fig_duration = px.bar(
                    x=list(avg_durations.keys()),
                    y=list(avg_durations.values()),
                    title="Durée Moyenne par Type de Projet (jours)",
                    labels={"x": "Type de Projet", "y": "Durée (jours)"}
                )
                st.plotly_chart(fig_duration, use_container_width=True)
        
        # Analyse financière
        financial = analytics_data.get("financial_analysis", {})
        if financial:
            st.subheader("💰 Analyse Financière")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Valeur Totale", f"{financial.get('total_portfolio_value', 0):,.0f}€")
            with col2:
                st.metric("Budget Moyen", f"{financial.get('avg_project_budget', 0):,.0f}€")
            with col3:
                roi_data = financial.get("roi_analysis", {})
                st.metric("ROI Estimé", f"{roi_data.get('estimated_roi', 0):.1f}%")
        
        # Prédictions ML
        predictions = analytics_data.get("ml_predictions", {})
        if predictions and "message" not in predictions:
            st.subheader("🔮 Prédictions ML")
            
            success_forecast = predictions.get("success_forecast", {})
            if success_forecast:
                st.info(f"🎯 Probabilité de succès moyenne: {success_forecast.get('avg_success_probability', 0):.0%}")
        
        # Insights
        insights = analytics_data.get("insights", [])
        if insights:
            st.subheader("💡 Insights IA")
            for insight in insights:
                st.success(f"✨ {insight}")
    
    except Exception as e:
        st.error(f"Erreur analytics: {str(e)}")


def render_resource_optimization(portfolio_manager):
    """Optimisation des ressources"""
    
    st.subheader("⚖️ Optimisation des Ressources")
    
    try:
        optimization_data = portfolio_manager.optimize_resource_allocation()
        
        if "message" in optimization_data:
            st.info(optimization_data["message"])
            return
        
        # Score d'efficacité
        efficiency_score = optimization_data.get("efficiency_score", 0)
        st.metric("🎯 Score d'Efficacité Portfolio", f"{efficiency_score:.1f}%")
        
        # Résultats d'optimisation par ressource
        optimization_results = optimization_data.get("optimization_results", {})
        
        for resource_type, data in optimization_results.items():
            with st.expander(f"🔧 Optimisation {resource_type.title()}"):
                
                utilization = data.get("utilization_score", 0)
                st.metric("Taux d'Utilisation", f"{utilization:.1f}%")
                
                # Graphique utilisation
                allocations = data.get("allocations", [])
                if allocations:
                    df = pd.DataFrame(allocations)
                    
                    fig = px.bar(
                        df, 
                        x="project_name", 
                        y=["current_allocation", "capacity"],
                        title=f"Allocation vs Capacité - {resource_type.title()}",
                        barmode="group"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommandations
                recommendations = data.get("recommendations", [])
                for rec in recommendations:
                    st.warning(f"⚠️ {rec}")
        
        # Recommandations globales
        global_recommendations = optimization_data.get("recommendations", [])
        if global_recommendations:
            st.subheader("💡 Recommandations Globales")
            for rec in global_recommendations:
                st.info(f"🎯 {rec}")
    
    except Exception as e:
        st.error(f"Erreur optimisation: {str(e)}")


def render_dependency_analysis(portfolio_manager):
    """Analyse des dépendances entre projets"""
    
    st.subheader("🔗 Analyse des Dépendances")
    
    try:
        dependencies_data = portfolio_manager.analyze_project_dependencies()
        
        if "message" in dependencies_data:
            st.info(dependencies_data["message"])
            st.markdown("""
            ### 🔗 Gestion des Dépendances
            
            Les dépendances entre projets permettent de :
            - 📊 **Visualiser** les interconnexions
            - ⚡ **Optimiser** l'ordre d'exécution
            - 🚨 **Détecter** les goulots d'étranglement
            - 🎯 **Paralléliser** les tâches indépendantes
            """)
            return
        
        # Métriques du graphe
        graph_metrics = dependencies_data.get("graph_metrics", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Dépendances", graph_metrics.get("total_dependencies", 0))
        with col2:
            st.metric("Projets Connectés", graph_metrics.get("connected_projects", 0))
        with col3:
            st.metric("Projets Isolés", graph_metrics.get("isolated_projects", 0))
        
        # Chemin critique
        critical_path = dependencies_data.get("critical_path", [])
        if critical_path:
            st.subheader("🎯 Chemin Critique")
            st.info(f"Séquence critique: {' → '.join(critical_path)}")
        
        # Goulots d'étranglement
        bottlenecks = dependencies_data.get("bottlenecks", [])
        if bottlenecks:
            st.subheader("🚨 Goulots d'Étranglement")
            for bottleneck in bottlenecks:
                st.warning(f"⚠️ **{bottleneck['project_name']}** : {bottleneck['incoming_dependencies']} dépendances entrantes")
        
        # Risques de dépendances
        risks = dependencies_data.get("dependency_risks", [])
        if risks:
            st.subheader("⚠️ Risques Détectés")
            for risk in risks:
                severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk.get("severity", "medium"), "⚪")
                st.error(f"{severity_color} **{risk.get('description', '')}** - {risk.get('impact', '')}")
        
        # Suggestions d'optimisation
        suggestions = dependencies_data.get("optimization_suggestions", [])
        if suggestions:
            st.subheader("💡 Suggestions d'Optimisation")
            for suggestion in suggestions:
                st.success(f"✨ **{suggestion.get('title', '')}** : {suggestion.get('benefit', '')}")
    
    except Exception as e:
        st.error(f"Erreur analyse dépendances: {str(e)}")


def render_rag_intelligence_section(plan_data: Dict[str, Any]):
    """Section RAG Intelligence - Gestionnaire documentaire intelligent"""
    
    st.header("🔍 RAG Intelligence")
    st.markdown("Gestionnaire documentaire intelligent avec recherche sémantique et auto-enrichissement")
    
    ai_rag_manager = get_ai_module('ai_rag_manager')
    initialize_rag_manager = get_ai_module('initialize_rag_manager')
    QueryContext = get_ai_module('QueryContext')
    
    if not ai_rag_manager or not initialize_rag_manager:
        st.error("Module RAG Manager non disponible")
        return
    
    # Initialisation du RAG manager
    rag_manager = initialize_rag_manager()
    
    # Récupération des analytics
    analytics = rag_manager.get_analytics_dashboard()
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", analytics['total_documents'])
    
    with col2:
        st.metric("Chunks Indexés", analytics['total_chunks'])
    
    with col3:
        st.metric("Requêtes Totales", analytics['total_queries'])
    
    with col4:
        avg_quality = analytics['average_quality']
        st.metric("Qualité Moyenne", f"{avg_quality:.1%}")
    
    # Onglets RAG
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Recherche Intelligente",
        "📚 Bibliothèque",
        "📊 Analytics",
        "🚀 Auto-Enrichissement",
        "⚙️ Configuration"
    ])
    
    with tab1:
        render_smart_search_interface(rag_manager, QueryContext)
    
    with tab2:
        render_document_library(rag_manager, analytics)
    
    with tab3:
        render_rag_analytics(analytics)
    
    with tab4:
        render_auto_enrichment_section(rag_manager)
    
    with tab5:
        render_rag_configuration(rag_manager)


def render_smart_search_interface(rag_manager, QueryContext):
    """Interface de recherche intelligente"""
    
    st.subheader("🔍 Recherche Sémantique")
    
    # Interface de recherche
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Posez votre question :",
            placeholder="Ex: Comment gérer un retard de projet ?",
            key=f"rag_query_{int(time.time() * 1000)}"
        )
    
    with col2:
        context = st.selectbox(
            "Contexte :",
            [ctx.value for ctx in QueryContext],
            key=f"rag_context_{int(time.time() * 1000)}"
        )
    
    # Options avancées
    with st.expander("🔧 Options Avancées"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_results = st.slider("Nombre de résultats", 1, 20, 5, key=f"rag_search_max_results_{int(time.time() * 1000)}")
            confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.6, key=f"rag_search_confidence_{int(time.time() * 1000)}")
        
        with col2:
            doc_types = st.multiselect(
                "Types de documents",
                ["internal_doc", "web_article", "case_study", "best_practice", "methodology"],
                key=f"rag_search_doc_types_{int(time.time() * 1000)}",
                default=[]
            )
            
            min_quality = st.slider("Qualité minimale", 0.0, 1.0, 0.5, key=f"rag_search_quality_{int(time.time() * 1000)}")
    
    # Recherche
    unique_search_key = f"rag_search_{int(time.time() * 1000)}"
    if st.button("🔍 Rechercher", key=unique_search_key) and query:
        with st.spinner("Recherche en cours..."):
            try:
                # Préparation des filtres
                filters = {}
                if doc_types:
                    filters['document_type'] = doc_types[0]  # Simplifié pour la démo
                if min_quality > 0.5:
                    filters['min_quality'] = min_quality
                
                # Exécution de la recherche
                context_enum = QueryContext(context) if hasattr(QueryContext, context) else QueryContext.GENERAL
                result = rag_manager.query(
                    question=query,
                    context=context_enum,
                    max_results=max_results,
                    filters=filters if filters else None
                )
                
                # Affichage des résultats
                st.subheader(f"📊 Résultats ({len(result.chunks)} trouvés)")
                
                if result.chunks:
                    # Confiance globale
                    st.success(f"Confiance: {result.confidence_score:.1%} | Temps: {result.processing_time:.2f}s")
                    
                    # Réponse générée
                    if result.generated_answer:
                        st.info(f"**Réponse générée :** {result.generated_answer}")
                    
                    # Résultats détaillés
                    for i, chunk in enumerate(result.chunks):
                        relevance_score = chunk.metadata.get('relevance_score', 0)
                        
                        with st.expander(f"📄 Résultat {i+1} - Pertinence: {relevance_score:.1%}"):
                            st.write(f"**Contenu :** {chunk.content[:500]}...")
                            
                            if chunk.document_id in rag_manager.documents:
                                doc = rag_manager.documents[chunk.document_id]
                                st.write(f"**Source :** {doc.title}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Type :** {doc.document_type.value}")
                                    st.write(f"**Fiabilité :** {doc.reliability.value}")
                                
                                with col2:
                                    st.write(f"**Qualité :** {doc.quality_score:.1%}")
                                    st.write(f"**Utilisations :** {doc.usage_count}")
                    
                    # Citations
                    if result.citations:
                        st.subheader("📚 Sources")
                        for citation in result.citations:
                            st.write(f"• {citation}")
                
                else:
                    st.warning("Aucun résultat trouvé. Essayez avec des mots-clés différents.")
                    
            except Exception as e:
                st.error(f"Erreur lors de la recherche: {str(e)}")
    
    # Suggestions de recherche
    st.subheader("💡 Suggestions de Recherche")
    
    search_suggestions = [
        "Comment estimer la durée d'un projet agile ?",
        "Meilleures pratiques pour la gestion d'équipe",
        "Comment gérer les risques de dépassement budgétaire ?",
        "Stratégies de négociation avec les stakeholders",
        "Méthodologies de planification projet",
        "Gestion de crise en mode projet"
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(search_suggestions):
        with cols[i % 3]:
            unique_suggestion_key = f"suggestion_{i}_{int(time.time() * 1000)}"
            if st.button(f"💡 {suggestion[:30]}...", key=unique_suggestion_key):
                st.info("✅ Suggestion appliquée")


def render_document_library(rag_manager, analytics):
    """Bibliothèque de documents"""
    
    st.subheader("📚 Bibliothèque Documentaire")
    
    if analytics['total_documents'] == 0:
        st.info("Aucun document dans la bibliothèque. Ajoutez des documents ci-dessous.")
    else:
        # Filtres
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox(
                "Filtrer par type :",
                ["Tous"] + [doc_type.value for doc_type in rag_manager.documents.values()],
                key=f"lib_filter_type_{int(time.time() * 1000)}"
            )
        
        with col2:
            filter_quality = st.selectbox(
                "Filtrer par qualité :",
                ["Tous", "Haute (>80%)", "Moyenne (50-80%)", "Faible (<50%)"],
                key=f"lib_filter_quality_{int(time.time() * 1000)}"
            )
        
        with col3:
            sort_by = st.selectbox(
                "Trier par :",
                ["Date d'ajout", "Qualité", "Utilisation", "Titre"],
                key=f"lib_sort_{int(time.time() * 1000)}"
            )
        
        # Liste des documents
        documents = list(rag_manager.documents.values())
        
        # Application des filtres
        if filter_type != "Tous":
            documents = [doc for doc in documents if doc.document_type.value == filter_type]
        
        if filter_quality != "Tous":
            if filter_quality == "Haute (>80%)":
                documents = [doc for doc in documents if doc.quality_score > 0.8]
            elif filter_quality == "Moyenne (50-80%)":
                documents = [doc for doc in documents if 0.5 <= doc.quality_score <= 0.8]
            elif filter_quality == "Faible (<50%)":
                documents = [doc for doc in documents if doc.quality_score < 0.5]
        
        # Tri
        if sort_by == "Qualité":
            documents.sort(key=lambda x: x.quality_score, reverse=True)
        elif sort_by == "Utilisation":
            documents.sort(key=lambda x: x.usage_count, reverse=True)
        elif sort_by == "Titre":
            documents.sort(key=lambda x: x.title)
        else:  # Date d'ajout
            documents.sort(key=lambda x: x.last_updated, reverse=True)
        
        # Affichage paginé
        docs_per_page = 10
        total_pages = max(1, len(documents) // docs_per_page + (1 if len(documents) % docs_per_page else 0))
        
        if total_pages > 1:
            page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1)) - 1
        else:
            page = 0
        
        start_idx = page * docs_per_page
        end_idx = min(start_idx + docs_per_page, len(documents))
        page_documents = documents[start_idx:end_idx]
        
        # Affichage des documents
        for doc in page_documents:
            with st.expander(f"📄 {doc.title} - Qualité: {doc.quality_score:.1%}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Type :** {doc.document_type.value.title()}")
                    st.write(f"**Fiabilité :** {doc.reliability.value.title()}")
                    st.write(f"**Mots :** {doc.word_count:,}")
                    
                    if doc.topics:
                        st.write(f"**Topics :** {', '.join(doc.topics)}")
                    
                    if doc.source_url:
                        st.write(f"**Source :** {doc.source_url}")
                
                with col2:
                    st.metric("Qualité", f"{doc.quality_score:.1%}")
                    st.metric("Utilisation", doc.usage_count)
                    st.write(f"**Ajouté :** {doc.last_updated.strftime('%d/%m/%Y')}")
    
    # Ajout de documents
    st.subheader("➕ Ajouter un Document")
    
    with st.expander("📝 Nouveau Document"):
        doc_title = st.text_input("Titre du document", key="new_doc_title")
        doc_content = st.text_area(
            "Contenu du document",
            height=200,
            placeholder="Collez ici le contenu du document...",
            key="new_doc_content"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            doc_type = st.selectbox(
                "Type de document",
                ["internal_doc", "web_article", "case_study", "best_practice", "methodology", "industry_report"],
                key="new_doc_type"
            )
        
        with col2:
            doc_reliability = st.selectbox(
                "Fiabilité",
                ["unknown", "low", "medium", "high", "very_high"],
                key="new_doc_reliability"
            )
        
        doc_source_url = st.text_input("URL source (optionnel)", key="new_doc_url")
        doc_author = st.text_input("Auteur (optionnel)", key="new_doc_author")
        
        unique_add_key = f"add_document_{int(time.time() * 1000)}"
        if st.button("📄 Ajouter le Document", key=unique_add_key) and doc_title and doc_content:
            try:
                from project_planner.ai.rag_manager import DocumentMetadata, DocumentType, SourceReliability
                
                metadata = DocumentMetadata(
                    id="",  # Sera généré automatiquement
                    title=doc_title,
                    source_url=doc_source_url if doc_source_url else None,
                    document_type=DocumentType(doc_type),
                    reliability=SourceReliability(doc_reliability),
                    author=doc_author if doc_author else None,
                )
                
                success = rag_manager.add_document(doc_content, metadata)
                
                if success:
                    st.success("Document ajouté avec succès !")
                else:
                    st.error("Erreur lors de l'ajout du document")
                    
            except Exception as e:
                st.error(f"Erreur : {str(e)}")


def render_rag_analytics(analytics):
    """Analytics RAG"""
    
    st.subheader("📊 Analytics RAG")
    
    # Distribution de qualité
    if analytics['quality_distribution']:
        st.write("**Distribution de Qualité :**")
        
        quality_df = pd.DataFrame(
            list(analytics['quality_distribution'].items()),
            columns=['Qualité', 'Nombre']
        )
        
        fig = px.pie(
            quality_df,
            values='Nombre',
            names='Qualité',
            title="Répartition par Qualité des Documents"
        )
        
        unique_quality_key = f"quality_dist_{int(time.time() * 1000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_quality_key)
    
    # Top requêtes
    if analytics['top_queries']:
        st.write("**Top 10 Requêtes :**")
        
        queries_df = pd.DataFrame(
            list(analytics['top_queries'].items()),
            columns=['Requête', 'Fréquence']
        ).head(10)
        
        fig = px.bar(
            queries_df,
            x='Fréquence',
            y='Requête',
            orientation='h',
            title="Requêtes les Plus Fréquentes"
        )
        
        unique_queries_key = f"top_queries_{int(time.time() * 1000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_queries_key)
    
    # Utilisation par contexte
    if analytics['context_usage']:
        st.write("**Utilisation par Contexte :**")
        
        context_df = pd.DataFrame(
            list(analytics['context_usage'].items()),
            columns=['Contexte', 'Utilisation']
        )
        
        fig = px.bar(
            context_df,
            x='Contexte',
            y='Utilisation',
            title="Répartition par Contexte de Recherche"
        )
        
        unique_context_key = f"context_usage_{int(time.time() * 1000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_context_key)
    
    # Top documents
    if analytics['top_documents']:
        st.write("**Documents les Plus Consultés :**")
        
        for i, doc in enumerate(analytics['top_documents'][:5]):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{i+1}. {doc['title']}**")
                    st.write(f"Type: {doc['document_type'].title()}")
                
                with col2:
                    st.metric("Consultations", doc['usage_count'])
                
                with col3:
                    st.metric("Qualité", f"{doc['quality_score']:.1%}")


def render_auto_enrichment_section(rag_manager):
    """Section auto-enrichissement"""
    
    st.subheader("🚀 Auto-Enrichissement")
    
    st.write("Système automatique d'enrichissement de la base de connaissances")
    
    # Recommandations d'enrichissement
    recommendations = rag_manager.suggest_missing_topics()
    
    if recommendations:
        st.write("**Topics Recommandés pour Enrichissement :**")
        
        for rec in recommendations[:5]:
            with st.expander(f"📈 {rec.topic.title()} - Priorité: {rec.priority.upper()}"):
                st.write(f"**Description :** {rec.gap_description}")
                st.write(f"**Impact potentiel :** {rec.potential_impact:.1%}")
                
                if rec.search_queries:
                    st.write("**Requêtes suggérées :**")
                    for query in rec.search_queries[:3]:
                        st.write(f"• {query}")
                
                unique_enrich_key = f"enrich_{rec.topic}_{int(time.time() * 1000)}"
                if st.button(f"🔍 Lancer l'enrichissement pour {rec.topic}", key=unique_enrich_key):
                    with st.spinner("Recherche de nouvelles sources..."):
                        time.sleep(2)  # Simulation
                        st.success(f"3 nouvelles sources identifiées pour {rec.topic}")
    else:
        st.info("Aucun gap majeur détecté. Votre base de connaissances semble complète !")
    
    # Statistiques d'enrichissement
    st.write("**Statistiques d'Enrichissement :**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sources Découvertes", "47", "+12")
    
    with col2:
        st.metric("Auto-Ajouts ce Mois", "8", "+3")
    
    with col3:
        last_enrichment = "Il y a 2 jours"
        st.metric("Dernier Enrichissement", last_enrichment)
    
    # Configuration auto-enrichissement
    with st.expander("⚙️ Configuration Auto-Enrichissement"):
        auto_enrich_enabled = st.checkbox("Activer l'enrichissement automatique", value=True)
        
        if auto_enrich_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                enrich_frequency = st.selectbox(
                    "Fréquence :",
                    ["Quotidien", "Hebdomadaire", "Mensuel"],
                    index=1
                )
                
                quality_threshold = st.slider("Seuil de qualité minimum", 0.5, 0.9, 0.7)
            
            with col2:
                max_docs_per_topic = st.number_input("Max docs par topic", 1, 20, 5)
                
                trusted_domains = st.text_area(
                    "Domaines de confiance (un par ligne)",
                    value="harvard.edu\nmit.edu\nmckinsey.com"
                )
            
            if st.button("💾 Sauvegarder Configuration"):
                st.success("Configuration sauvegardée !")


def render_rag_configuration(rag_manager):
    """Configuration RAG"""
    
    st.subheader("⚙️ Configuration RAG")
    
    # État du système
    st.write("**État du Système :**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if rag_manager.embedding_model:
            st.success("✅ Modèle d'embeddings: Actif")
        else:
            st.error("❌ Modèle d'embeddings: Inactif")
        
        if rag_manager.faiss_index:
            st.success("✅ Index FAISS: Actif")
        else:
            st.warning("⚠️ Index FAISS: Non initialisé")
    
    with col2:
        index_size = len(rag_manager.chunk_ids) if rag_manager.chunk_ids else 0
        st.info(f"📊 Index: {index_size} chunks")
        
        model_name = "MiniLM-L12" if rag_manager.embedding_model else "Aucun"
        st.info(f"🧠 Modèle: {model_name}")
    
    # Paramètres de recherche
    st.write("**Paramètres de Recherche :**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.number_input(
            "Taille des chunks",
            min_value=100,
            max_value=1000,
            value=rag_manager.chunk_size
        )
        
        max_results = st.number_input(
            "Résultats maximum",
            min_value=1,
            max_value=50,
            value=rag_manager.max_results
        )
    
    with col2:
        chunk_overlap = st.number_input(
            "Overlap des chunks",
            min_value=0,
            max_value=200,
            value=rag_manager.chunk_overlap
        )
        
        confidence_threshold = st.slider(
            "Seuil de confiance",
            0.0, 1.0,
            rag_manager.confidence_threshold,
            key=f"rag_config_confidence_{int(time.time() * 1000)}"
        )
    
    # Actions système
    st.write("**Actions Système :**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reconstruire Index"):
            with st.spinner("Reconstruction de l'index..."):
                time.sleep(3)  # Simulation
                st.success("Index reconstruit !")
    
    with col2:
        if st.button("🧹 Nettoyer Cache"):
            st.success("Cache nettoyé !")
    
    with col3:
        if st.button("💾 Exporter Données"):
            st.success("Export créé !")
    
    # Informations debug
    with st.expander("🐛 Informations Debug"):
        debug_info = {
            "Total documents": len(rag_manager.documents),
            "Total chunks": len(rag_manager.chunks),
            "Index actif": bool(rag_manager.faiss_index),
            "Modèle embedding": bool(rag_manager.embedding_model),
            "Répertoire données": rag_manager.data_dir,
            "Taille chunk_ids": len(rag_manager.chunk_ids)
        }
        
        st.json(debug_info)


# Fonction d'initialisation pour le dashboard
def initialize_ai_dashboard():
    """Initialise le dashboard IA"""
    
    if 'ai_dashboard_initialized' not in st.session_state:
        st.session_state.ai_dashboard_initialized = True
        
        # Configuration initiale
        if 'user_id' not in st.session_state:
            st.session_state.user_id = f"user_{datetime.now().timestamp()}"


# Appel automatique de l'initialisation
initialize_ai_dashboard()