"""
Contextual UI Suggestions for Smart Recommendations
Interface de suggestions contextuelles intelligentes pour le dashboard
"""

import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json

from .recommendation_engine import (
    SmartRecommendationEngine, 
    Recommendation, 
    RecommendationType,
    RecommendationPriority,
    get_recommendation_engine
)
from .behavioral_analyzer import (
    BehavioralAnalyzer,
    ActionType,
    get_behavioral_analyzer,
    track_user_action
)
from .knowledge_base import (
    DynamicKnowledgeBase,
    get_knowledge_base,
    IndustryType,
    PracticeCategory
)

logger = logging.getLogger(__name__)

class SuggestionContext(Enum):
    """Contextes d'affichage des suggestions"""
    SIDEBAR = "sidebar"
    MAIN_PANEL = "main_panel"
    MODAL = "modal"
    FLOATING = "floating"
    NOTIFICATION = "notification"

class SuggestionTrigger(Enum):
    """Déclencheurs de suggestions"""
    ON_LOAD = "on_load"
    ON_PLAN_GENERATED = "on_plan_generated"
    ON_PARAMETER_CHANGE = "on_parameter_change"
    ON_ERROR = "on_error"
    ON_IDLE = "on_idle"
    ON_USER_REQUEST = "on_user_request"

@dataclass
class UISuggestion:
    """Suggestion d'interface utilisateur"""
    suggestion_id: str
    title: str
    description: str
    action_type: str
    context: SuggestionContext
    priority: RecommendationPriority
    parameters: Dict[str, Any]
    expires_at: datetime
    callback: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'suggestion_id': self.suggestion_id,
            'title': self.title,
            'description': self.description,
            'action_type': self.action_type,
            'context': self.context.value,
            'priority': self.priority.value,
            'parameters': self.parameters,
            'expires_at': self.expires_at.isoformat()
        }

class ContextualSuggestionsUI:
    """Gestionnaire de suggestions contextuelles pour l'UI"""
    
    def __init__(self):
        self.recommendation_engine = get_recommendation_engine()
        self.behavioral_analyzer = get_behavioral_analyzer()
        self.knowledge_base = get_knowledge_base()
        self.active_suggestions = {}
        self.suggestion_history = []
        self.user_preferences = {}
        
    def generate_contextual_suggestions(self, 
                                      project_data: Dict[str, Any],
                                      current_context: Dict[str, Any],
                                      trigger: SuggestionTrigger) -> List[UISuggestion]:
        """Générer des suggestions contextuelles pour l'UI"""
        
        suggestions = []
        
        # Obtenir les préférences utilisateur
        self.user_preferences = self.behavioral_analyzer.get_user_preferences()
        
        # Générer suggestions selon le déclencheur
        if trigger == SuggestionTrigger.ON_LOAD:
            suggestions.extend(self._generate_welcome_suggestions(project_data))
            
        elif trigger == SuggestionTrigger.ON_PLAN_GENERATED:
            suggestions.extend(self._generate_post_plan_suggestions(project_data))
            
        elif trigger == SuggestionTrigger.ON_PARAMETER_CHANGE:
            suggestions.extend(self._generate_parameter_suggestions(project_data, current_context))
            
        elif trigger == SuggestionTrigger.ON_ERROR:
            suggestions.extend(self._generate_error_recovery_suggestions(current_context))
            
        elif trigger == SuggestionTrigger.ON_IDLE:
            suggestions.extend(self._generate_productivity_suggestions(project_data))
            
        # Filtrer et prioriser
        filtered_suggestions = self._filter_and_prioritize(suggestions)
        
        # Tracker la génération de suggestions
        track_user_action(
            ActionType.DASHBOARD_VIEW,
            context={'suggestions_generated': len(filtered_suggestions), 'trigger': trigger.value},
            project_id=project_data.get('project_id')
        )
        
        return filtered_suggestions
        
    def _generate_welcome_suggestions(self, project_data: Dict[str, Any]) -> List[UISuggestion]:
        """Suggestions d'accueil"""
        suggestions = []
        
        # Suggestion de génération de plan si pas de plan actuel
        if not project_data.get('has_active_plan', False):
            suggestions.append(UISuggestion(
                suggestion_id="welcome_generate_plan",
                title="🎯 Générer un nouveau plan",
                description="Commencez par créer un plan de projet détaillé avec nos agents IA",
                action_type="generate_plan",
                context=SuggestionContext.MAIN_PANEL,
                priority=RecommendationPriority.HIGH,
                parameters={'show_wizard': True},
                expires_at=datetime.now() + timedelta(hours=1)
            ))
            
        # Suggestion basée sur l'historique
        recent_projects = project_data.get('recent_projects', [])
        if recent_projects:
            suggestions.append(UISuggestion(
                suggestion_id="welcome_load_recent",
                title="📂 Reprendre un projet récent",
                description="Continuez où vous vous êtes arrêté",
                action_type="load_recent_project",
                context=SuggestionContext.SIDEBAR,
                priority=RecommendationPriority.MEDIUM,
                parameters={'recent_projects': recent_projects[:3]},
                expires_at=datetime.now() + timedelta(hours=2)
            ))
            
        # Suggestion d'exploration des fonctionnalités
        if not self.user_preferences:  # Nouvel utilisateur
            suggestions.append(UISuggestion(
                suggestion_id="welcome_tour",
                title="🗺️ Découvrir PlannerIA",
                description="Visite guidée des fonctionnalités principales",
                action_type="start_tour",
                context=SuggestionContext.NOTIFICATION,
                priority=RecommendationPriority.LOW,
                parameters={'tour_type': 'full'},
                expires_at=datetime.now() + timedelta(days=7)
            ))
            
        return suggestions
        
    def _generate_post_plan_suggestions(self, project_data: Dict[str, Any]) -> List[UISuggestion]:
        """Suggestions après génération de plan"""
        suggestions = []
        
        plan_data = project_data.get('current_plan', {})
        
        # Suggestion d'analyse What-If
        suggestions.append(UISuggestion(
            suggestion_id="post_plan_what_if",
            title="🔍 Analyser des scénarios What-If",
            description="Explorez différents scénarios pour optimiser votre plan",
            action_type="open_what_if",
            context=SuggestionContext.SIDEBAR,
            priority=RecommendationPriority.HIGH,
            parameters={'focus_metrics': ['duration', 'budget']},
            expires_at=datetime.now() + timedelta(hours=4)
        ))
        
        # Suggestion d'export
        suggestions.append(UISuggestion(
            suggestion_id="post_plan_export",
            title="📄 Exporter le plan",
            description="Générez des rapports PDF, CSV ou slides de présentation",
            action_type="show_export_options",
            context=SuggestionContext.MAIN_PANEL,
            priority=RecommendationPriority.MEDIUM,
            parameters={'formats': ['pdf', 'csv', 'slides']},
            expires_at=datetime.now() + timedelta(hours=2)
        ))
        
        # Suggestion de validation des risques si risques détectés
        risks = plan_data.get('risks', [])
        if risks:
            high_risks = [r for r in risks if r.get('severity', 0) > 7]
            if high_risks:
                suggestions.append(UISuggestion(
                    suggestion_id="post_plan_risk_review",
                    title="⚠️ Réviser les risques critiques",
                    description=f"{len(high_risks)} risque(s) critique(s) identifié(s)",
                    action_type="review_risks",
                    context=SuggestionContext.NOTIFICATION,
                    priority=RecommendationPriority.HIGH,
                    parameters={'high_risk_count': len(high_risks)},
                    expires_at=datetime.now() + timedelta(hours=1)
                ))
                
        # Suggestion de partage si équipe détectée
        team_size = plan_data.get('team_size', 0)
        if team_size > 1:
            suggestions.append(UISuggestion(
                suggestion_id="post_plan_share",
                title="👥 Partager avec l'équipe",
                description="Partagez le plan avec les membres de votre équipe",
                action_type="share_plan",
                context=SuggestionContext.SIDEBAR,
                priority=RecommendationPriority.MEDIUM,
                parameters={'team_size': team_size},
                expires_at=datetime.now() + timedelta(hours=6)
            ))
            
        return suggestions
        
    def _generate_parameter_suggestions(self, 
                                       project_data: Dict[str, Any], 
                                       current_context: Dict[str, Any]) -> List[UISuggestion]:
        """Suggestions lors de changement de paramètres"""
        suggestions = []
        
        changed_param = current_context.get('changed_parameter')
        param_value = current_context.get('parameter_value')
        
        # Suggestions spécifiques selon le paramètre modifié
        if changed_param == 'budget':
            if isinstance(param_value, (int, float)) and param_value < 10000:
                suggestions.append(UISuggestion(
                    suggestion_id="param_budget_low",
                    title="💰 Budget optimisé détecté",
                    description="Voir nos recommandations pour les projets à budget serré",
                    action_type="show_budget_recommendations",
                    context=SuggestionContext.MODAL,
                    priority=RecommendationPriority.MEDIUM,
                    parameters={'budget_range': 'low'},
                    expires_at=datetime.now() + timedelta(minutes=30)
                ))
                
        elif changed_param == 'duration':
            if isinstance(param_value, (int, float)) and param_value > 180:  # Plus de 6 mois
                suggestions.append(UISuggestion(
                    suggestion_id="param_duration_long",
                    title="⏳ Projet long terme détecté",
                    description="Considérez une planification par phases",
                    action_type="suggest_phased_approach",
                    context=SuggestionContext.SIDEBAR,
                    priority=RecommendationPriority.MEDIUM,
                    parameters={'duration_months': param_value // 30},
                    expires_at=datetime.now() + timedelta(minutes=45)
                ))
                
        elif changed_param == 'team_size':
            if isinstance(param_value, int) and param_value > 10:
                suggestions.append(UISuggestion(
                    suggestion_id="param_team_large",
                    title="👥 Grande équipe détectée",
                    description="Recommandations spéciales pour la gestion d'équipes importantes",
                    action_type="show_large_team_practices",
                    context=SuggestionContext.SIDEBAR,
                    priority=RecommendationPriority.HIGH,
                    parameters={'team_size': param_value},
                    expires_at=datetime.now() + timedelta(hours=1)
                ))
                
        return suggestions
        
    def _generate_error_recovery_suggestions(self, current_context: Dict[str, Any]) -> List[UISuggestion]:
        """Suggestions de récupération d'erreur"""
        suggestions = []
        
        error_type = current_context.get('error_type', 'unknown')
        
        if error_type == 'generation_failed':
            suggestions.append(UISuggestion(
                suggestion_id="error_regenerate",
                title="🔄 Régénérer le plan",
                description="La génération a échoué. Essayez avec des paramètres ajustés",
                action_type="regenerate_with_fallback",
                context=SuggestionContext.NOTIFICATION,
                priority=RecommendationPriority.HIGH,
                parameters={'retry_strategy': 'simplified'},
                expires_at=datetime.now() + timedelta(minutes=10)
            ))
            
        elif error_type == 'validation_failed':
            suggestions.append(UISuggestion(
                suggestion_id="error_fix_validation",
                title="✏️ Corriger les paramètres",
                description="Certains paramètres semblent invalides",
                action_type="highlight_invalid_params",
                context=SuggestionContext.MAIN_PANEL,
                priority=RecommendationPriority.HIGH,
                parameters={},
                expires_at=datetime.now() + timedelta(minutes=15)
            ))
            
        elif error_type == 'timeout':
            suggestions.append(UISuggestion(
                suggestion_id="error_simplify",
                title="⚡ Simplifier la demande",
                description="Le traitement prend trop de temps. Essayez une approche plus simple",
                action_type="suggest_simpler_approach",
                context=SuggestionContext.MODAL,
                priority=RecommendationPriority.MEDIUM,
                parameters={'simplification_options': ['reduce_scope', 'fewer_agents']},
                expires_at=datetime.now() + timedelta(minutes=20)
            ))
            
        return suggestions
        
    def _generate_productivity_suggestions(self, project_data: Dict[str, Any]) -> List[UISuggestion]:
        """Suggestions de productivité pendant les moments d'inactivité"""
        suggestions = []
        
        # Analyser l'activité récente
        last_activity = project_data.get('last_activity_minutes', 0)
        
        if last_activity > 5:  # Plus de 5 minutes d'inactivité
            
            # Suggestion de sauvegarde
            if project_data.get('has_unsaved_changes', False):
                suggestions.append(UISuggestion(
                    suggestion_id="idle_save_work",
                    title="💾 Sauvegarder le travail",
                    description="Vous avez des modifications non sauvegardées",
                    action_type="save_current_work",
                    context=SuggestionContext.NOTIFICATION,
                    priority=RecommendationPriority.HIGH,
                    parameters={},
                    expires_at=datetime.now() + timedelta(minutes=2)
                ))
                
            # Suggestion d'exploration des données
            if project_data.get('has_active_plan', False):
                suggestions.append(UISuggestion(
                    suggestion_id="idle_explore_analytics",
                    title="📊 Explorer les analyses",
                    description="Découvrez les métriques détaillées de votre projet",
                    action_type="open_analytics_panel",
                    context=SuggestionContext.SIDEBAR,
                    priority=RecommendationPriority.LOW,
                    parameters={},
                    expires_at=datetime.now() + timedelta(minutes=30)
                ))
                
            # Suggestion d'apprentissage
            suggestions.append(UISuggestion(
                suggestion_id="idle_learn_tips",
                title="💡 Conseils et astuces",
                description="Découvrez des fonctionnalités que vous ne connaissez peut-être pas",
                action_type="show_tips_panel",
                context=SuggestionContext.FLOATING,
                priority=RecommendationPriority.LOW,
                parameters={'tip_category': 'productivity'},
                expires_at=datetime.now() + timedelta(hours=1)
            ))
            
        return suggestions
        
    def _filter_and_prioritize(self, suggestions: List[UISuggestion]) -> List[UISuggestion]:
        """Filtrer et prioriser les suggestions"""
        
        # Supprimer les suggestions expirées
        current_time = datetime.now()
        valid_suggestions = [s for s in suggestions if s.expires_at > current_time]
        
        # Supprimer les doublons par suggestion_id
        seen_ids = set()
        unique_suggestions = []
        for suggestion in valid_suggestions:
            if suggestion.suggestion_id not in seen_ids:
                unique_suggestions.append(suggestion)
                seen_ids.add(suggestion.suggestion_id)
                
        # Trier par priorité et limiter le nombre
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3
        }
        
        unique_suggestions.sort(key=lambda s: priority_order.get(s.priority, 999))
        
        # Limiter selon le contexte
        context_limits = {
            SuggestionContext.NOTIFICATION: 2,
            SuggestionContext.MODAL: 1,
            SuggestionContext.SIDEBAR: 5,
            SuggestionContext.MAIN_PANEL: 3,
            SuggestionContext.FLOATING: 2
        }
        
        filtered_by_context = {}
        for suggestion in unique_suggestions:
            context = suggestion.context
            if context not in filtered_by_context:
                filtered_by_context[context] = []
                
            limit = context_limits.get(context, 3)
            if len(filtered_by_context[context]) < limit:
                filtered_by_context[context].append(suggestion)
                
        # Réassembler la liste finale
        final_suggestions = []
        for context_suggestions in filtered_by_context.values():
            final_suggestions.extend(context_suggestions)
            
        return final_suggestions
        
    def render_suggestions_sidebar(self, suggestions: List[UISuggestion]):
        """Rendre les suggestions dans la barre latérale"""
        
        sidebar_suggestions = [s for s in suggestions if s.context == SuggestionContext.SIDEBAR]
        
        if not sidebar_suggestions:
            return
            
        st.sidebar.markdown("### 💡 Suggestions")
        
        for suggestion in sidebar_suggestions:
            with st.sidebar.container():
                # Badge de priorité
                priority_colors = {
                    RecommendationPriority.CRITICAL: "🔴",
                    RecommendationPriority.HIGH: "🟠", 
                    RecommendationPriority.MEDIUM: "🟡",
                    RecommendationPriority.LOW: "⚪"
                }
                
                priority_badge = priority_colors.get(suggestion.priority, "⚪")
                
                # Afficher la suggestion
                st.markdown(f"{priority_badge} **{suggestion.title}**")
                st.markdown(f"<small>{suggestion.description}</small>", unsafe_allow_html=True)
                
                # Bouton d'action
                if st.button(
                    "Appliquer", 
                    key=f"sidebar_{suggestion.suggestion_id}",
                    help="Cliquer pour appliquer cette suggestion"
                ):
                    self._handle_suggestion_action(suggestion)
                    
                st.markdown("---")
                
    def render_suggestions_main_panel(self, suggestions: List[UISuggestion]):
        """Rendre les suggestions dans le panneau principal"""
        
        main_suggestions = [s for s in suggestions if s.context == SuggestionContext.MAIN_PANEL]
        
        if not main_suggestions:
            return
            
        st.markdown("## 🎯 Recommandations")
        
        for suggestion in main_suggestions:
            with st.expander(suggestion.title, expanded=(suggestion.priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH])):
                st.markdown(suggestion.description)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if st.button(
                        "✅ Appliquer", 
                        key=f"main_{suggestion.suggestion_id}_apply",
                        type="primary"
                    ):
                        self._handle_suggestion_action(suggestion)
                        
                with col2:
                    if st.button(
                        "ℹ️ Plus d'infos", 
                        key=f"main_{suggestion.suggestion_id}_info"
                    ):
                        self._show_suggestion_details(suggestion)
                        
                with col3:
                    if st.button(
                        "❌ Ignorer", 
                        key=f"main_{suggestion.suggestion_id}_dismiss"
                    ):
                        self._dismiss_suggestion(suggestion)
                        
    def render_suggestions_notifications(self, suggestions: List[UISuggestion]):
        """Rendre les suggestions comme notifications"""
        
        notification_suggestions = [s for s in suggestions if s.context == SuggestionContext.NOTIFICATION]
        
        for suggestion in notification_suggestions:
            if suggestion.priority == RecommendationPriority.CRITICAL:
                st.error(f"🚨 **{suggestion.title}** - {suggestion.description}")
            elif suggestion.priority == RecommendationPriority.HIGH:
                st.warning(f"⚠️ **{suggestion.title}** - {suggestion.description}")
            else:
                st.info(f"💡 **{suggestion.title}** - {suggestion.description}")
                
            # Boutons d'action inline
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(
                    "Appliquer", 
                    key=f"notif_{suggestion.suggestion_id}_apply",
                    type="secondary"
                ):
                    self._handle_suggestion_action(suggestion)
                    
            with col2:
                if st.button(
                    "Fermer", 
                    key=f"notif_{suggestion.suggestion_id}_close"
                ):
                    self._dismiss_suggestion(suggestion)
                    
    def _handle_suggestion_action(self, suggestion: UISuggestion):
        """Gérer l'action d'une suggestion"""
        
        # Tracker l'acceptation de la suggestion
        track_user_action(
            ActionType.RECOMMENDATION_ACCEPTED,
            context={
                'suggestion_id': suggestion.suggestion_id,
                'action_type': suggestion.action_type,
                'priority': suggestion.priority.value
            }
        )
        
        # Exécuter l'action selon le type
        if suggestion.action_type == "generate_plan":
            st.session_state['trigger_plan_generation'] = True
            st.session_state['plan_wizard'] = suggestion.parameters.get('show_wizard', False)
            
        elif suggestion.action_type == "open_what_if":
            st.session_state['show_what_if'] = True
            st.session_state['what_if_focus'] = suggestion.parameters.get('focus_metrics', [])
            
        elif suggestion.action_type == "show_export_options":
            st.session_state['show_export_panel'] = True
            
        elif suggestion.action_type == "review_risks":
            st.session_state['highlight_risks'] = True
            
        elif suggestion.action_type == "share_plan":
            st.session_state['show_share_dialog'] = True
            
        # Callback personnalisé si défini
        if suggestion.callback:
            suggestion.callback(suggestion.parameters)
            
        # Marquer comme traitée
        self.active_suggestions.pop(suggestion.suggestion_id, None)
        
        st.success(f"✅ Suggestion appliquée: {suggestion.title}")
        st.rerun()
        
    def _dismiss_suggestion(self, suggestion: UISuggestion):
        """Ignorer une suggestion"""
        
        # Tracker le rejet
        track_user_action(
            ActionType.RECOMMENDATION_REJECTED,
            context={
                'suggestion_id': suggestion.suggestion_id,
                'reason': 'user_dismissed'
            }
        )
        
        # Retirer de la liste active
        self.active_suggestions.pop(suggestion.suggestion_id, None)
        
        st.info(f"Suggestion ignorée: {suggestion.title}")
        st.rerun()
        
    def _show_suggestion_details(self, suggestion: UISuggestion):
        """Afficher les détails d'une suggestion"""
        
        with st.modal(f"Détails - {suggestion.title}"):
            st.markdown(f"**Description:** {suggestion.description}")
            st.markdown(f"**Priorité:** {suggestion.priority.value}")
            st.markdown(f"**Type:** {suggestion.action_type}")
            
            if suggestion.parameters:
                st.markdown("**Paramètres:**")
                st.json(suggestion.parameters)
                
            st.markdown(f"**Expire le:** {suggestion.expires_at.strftime('%d/%m/%Y à %H:%M')}")

# Instance globale
global_suggestions_ui: Optional[ContextualSuggestionsUI] = None

def get_contextual_suggestions_ui() -> ContextualSuggestionsUI:
    """Obtenir l'instance globale des suggestions contextuelles"""
    global global_suggestions_ui
    
    if global_suggestions_ui is None:
        global_suggestions_ui = ContextualSuggestionsUI()
        
    return global_suggestions_ui

def render_contextual_suggestions(project_data: Dict[str, Any], 
                                current_context: Dict[str, Any],
                                trigger: SuggestionTrigger = SuggestionTrigger.ON_LOAD):
    """Fonction utilitaire pour rendre les suggestions contextuelles"""
    
    ui = get_contextual_suggestions_ui()
    suggestions = ui.generate_contextual_suggestions(project_data, current_context, trigger)
    
    # Rendre dans les différents contextes
    ui.render_suggestions_notifications(suggestions)
    ui.render_suggestions_sidebar(suggestions)
    ui.render_suggestions_main_panel(suggestions)
    
    return suggestions