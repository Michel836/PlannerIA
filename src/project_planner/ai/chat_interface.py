"""
💬 Interface Conversationnelle avec Chat IA - PlannerIA
Chat intelligent intégré avec NLP et analyse contextuelle
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import re
import asyncio
from .conversational_assistant import ai_assistant, ConversationContext
from .predictive_engine import ai_predictor
from .smart_alerts import smart_alert_system
from .gamification_engine import gamification_engine
from .risk_predictor import ai_risk_predictor


class ChatMessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    SUGGESTION = "suggestion"


class ChatMode(Enum):
    GENERAL = "general"
    PROJECT_ANALYSIS = "project_analysis"
    OPTIMIZATION = "optimization"
    RISK_ASSESSMENT = "risk_assessment"
    GAMIFICATION = "gamification"


@dataclass
class ChatMessage:
    """Message de chat avec métadonnées"""
    message_id: str
    type: ChatMessageType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass 
class ChatSession:
    """Session de chat avec historique"""
    session_id: str
    user_id: str
    mode: ChatMode
    messages: List[ChatMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    active_project_data: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


class SmartChatInterface:
    """Interface de chat intelligente pour PlannerIA"""
    
    def __init__(self):
        self.sessions = {}
        self.quick_actions = self._initialize_quick_actions()
        self.smart_suggestions = SmartSuggestionEngine()
        self.context_analyzer = ContextAnalyzer()
        self.response_personalizer = ResponsePersonalizer()
        
    def _initialize_quick_actions(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les actions rapides du chat"""
        
        return {
            "analyze_project": {
                "label": "🔍 Analyser mon projet",
                "description": "Analyse complète avec prédictions IA",
                "icon": "🔍",
                "category": "analysis",
                "quick_prompt": "Analyse mon projet avec toutes ses données"
            },
            "optimize_budget": {
                "label": "💰 Optimiser le budget",
                "description": "Suggestions d'optimisation budgétaire",
                "icon": "💰", 
                "category": "optimization",
                "quick_prompt": "Comment optimiser mon budget projet ?"
            },
            "assess_risks": {
                "label": "⚠️ Évaluer les risques",
                "description": "Analyse prédictive des risques",
                "icon": "⚠️",
                "category": "risk",
                "quick_prompt": "Quels sont les risques de mon projet ?"
            },
            "team_planning": {
                "label": "👥 Planifier l'équipe",
                "description": "Optimisation des ressources humaines",
                "icon": "👥",
                "category": "team",
                "quick_prompt": "Aide-moi à organiser mon équipe"
            },
            "timeline_optimization": {
                "label": "⏱️ Optimiser planning",
                "description": "Amélioration du calendrier projet",
                "icon": "⏱️",
                "category": "timeline",
                "quick_prompt": "Comment améliorer mon planning ?"
            },
            "quality_check": {
                "label": "✅ Check qualité",
                "description": "Vérification des standards qualité",
                "icon": "✅",
                "category": "quality", 
                "quick_prompt": "Vérifie la qualité de mon projet"
            },
            "get_suggestions": {
                "label": "💡 Suggestions IA",
                "description": "Recommandations personnalisées",
                "icon": "💡",
                "category": "suggestions",
                "quick_prompt": "Donne-moi tes meilleures suggestions"
            },
            "challenge_me": {
                "label": "🎮 Défie-moi !",
                "description": "Défi gamifié personnalisé",
                "icon": "🎮",
                "category": "gamification",
                "quick_prompt": "Propose-moi un défi adapté à mon niveau"
            }
        }
    
    def create_chat_session(self, user_id: str, mode: ChatMode = ChatMode.GENERAL, 
                          project_data: Optional[Dict[str, Any]] = None) -> ChatSession:
        """Crée une nouvelle session de chat"""
        
        session_id = f"chat_{user_id}_{datetime.now().timestamp()}"
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            active_project_data=project_data
        )
        
        # Message d'accueil personnalisé
        welcome_message = self._generate_welcome_message(user_id, mode, project_data)
        
        session.messages.append(ChatMessage(
            message_id=f"msg_{datetime.now().timestamp()}",
            type=ChatMessageType.ASSISTANT,
            content=welcome_message,
            timestamp=datetime.now(),
            suggestions=self._get_initial_suggestions(mode)
        ))
        
        self.sessions[session_id] = session
        return session
    
    def _generate_welcome_message(self, user_id: str, mode: ChatMode, 
                                project_data: Optional[Dict[str, Any]]) -> str:
        """Génère un message d'accueil personnalisé"""
        
        base_welcome = "👋 Bonjour ! Je suis votre assistant IA PlannerIA."
        
        if mode == ChatMode.PROJECT_ANALYSIS and project_data:
            project_name = project_data.get('project_overview', {}).get('name', 'votre projet')
            return f"{base_welcome}\n\n🎯 Je vois que vous travaillez sur **{project_name}**. Je peux vous aider à l'analyser, l'optimiser et anticiper les risques !\n\nQue souhaitez-vous explorer ?"
        
        elif mode == ChatMode.OPTIMIZATION:
            return f"{base_welcome}\n\n⚡ Mode Optimisation activé ! Je suis spécialisé dans l'amélioration de vos projets : budget, timeline, équipe, processus...\n\nQuel aspect voulez-vous optimiser ?"
        
        elif mode == ChatMode.RISK_ASSESSMENT:
            return f"{base_welcome}\n\n🛡️ Mode Analyse des Risques activé ! Je vais identifier les risques potentiels de votre projet et vous proposer des stratégies de mitigation.\n\nParlez-moi de votre projet ou de vos préoccupations."
        
        elif mode == ChatMode.GAMIFICATION:
            return f"{base_welcome}\n\n🎮 Mode Gamification activé ! Prêt pour des défis stimulants qui amélioreront vos compétences en gestion de projet ?\n\nQuel défi vous intéresse ?"
        
        else:  # GENERAL
            return f"""{base_welcome}\n\n✨ Je peux vous aider avec :
• 🔍 **Analyse de projets** - Prédictions et insights
• ⚡ **Optimisation** - Budget, timeline, équipe
• ⚠️ **Gestion des risques** - Détection et prévention
• 🎮 **Défis gamifiés** - Améliorez vos compétences
• 💬 **Conseils personnalisés** - Questions ouvertes

Que voulez-vous explorer aujourd'hui ?"""
    
    def _get_initial_suggestions(self, mode: ChatMode) -> List[str]:
        """Retourne les suggestions initiales selon le mode"""
        
        suggestions_map = {
            ChatMode.GENERAL: [
                "Analyser mon projet actuel",
                "Comment optimiser mon budget ?",
                "Quels sont les risques à surveiller ?",
                "Propose-moi un défi !"
            ],
            ChatMode.PROJECT_ANALYSIS: [
                "Analyse complète avec prédictions",
                "Probabilité de succès ?",
                "Projets similaires ?",
                "Points d'amélioration ?"
            ],
            ChatMode.OPTIMIZATION: [
                "Réduire les coûts",
                "Accélérer le développement",
                "Optimiser l'équipe",
                "Améliorer la qualité"
            ],
            ChatMode.RISK_ASSESSMENT: [
                "Identifier tous les risques",
                "Stratégies de prévention",
                "Plan de contingence",
                "Monitoring des risques"
            ],
            ChatMode.GAMIFICATION: [
                "Défi estimation",
                "Challenge optimisation",
                "Quiz gestion de projet", 
                "Voir mes achievements"
            ]
        }
        
        return suggestions_map.get(mode, [])
    
    async def process_message(self, session_id: str, user_message: str) -> ChatMessage:
        """Traite un message utilisateur et génère une réponse"""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} non trouvée")
        
        session = self.sessions[session_id]
        session.last_activity = datetime.now()
        
        # Ajout du message utilisateur
        user_msg = ChatMessage(
            message_id=f"msg_{datetime.now().timestamp()}",
            type=ChatMessageType.USER,
            content=user_message,
            timestamp=datetime.now()
        )
        session.messages.append(user_msg)
        
        # Analyse contextuelle du message
        context_analysis = await self.context_analyzer.analyze_message(user_message, session)
        
        # Détection d'intentions et entités
        intent_analysis = await self._analyze_user_intent(user_message, session)
        
        # Génération de la réponse selon le mode et l'intention
        ai_response = await self._generate_ai_response(user_message, session, context_analysis, intent_analysis)
        
        # Personnalisation de la réponse
        personalized_response = await self.response_personalizer.personalize_response(
            ai_response, session.user_id, context_analysis
        )
        
        # Génération de suggestions intelligentes
        smart_suggestions = await self.smart_suggestions.generate_suggestions(
            user_message, session, context_analysis
        )
        
        # Création du message de réponse
        response_msg = ChatMessage(
            message_id=f"msg_{datetime.now().timestamp()}",
            type=ChatMessageType.ASSISTANT,
            content=personalized_response,
            timestamp=datetime.now(),
            metadata={
                "intent": intent_analysis.get("intent"),
                "confidence": intent_analysis.get("confidence", 0.8),
                "context_analysis": context_analysis
            },
            suggestions=smart_suggestions
        )
        
        session.messages.append(response_msg)
        
        return response_msg
    
    async def _analyze_user_intent(self, message: str, session: ChatSession) -> Dict[str, Any]:
        """Analyse l'intention de l'utilisateur"""
        
        # Patterns d'intention avancés
        intent_patterns = {
            "analyze": [
                r"analys[er|e]", r"évaluer?", r"examiner", r"étudier", 
                r"que penses-tu", r"ton avis", r"diagnostic"
            ],
            "optimize": [
                r"optimis[er|ation]", r"améliorer", r"réduire", r"accélérer",
                r"plus efficace", r"moins cher", r"plus rapide"
            ],
            "predict": [
                r"prédi[re|ction]", r"estimer?", r"combien de temps", r"ça va coûter",
                r"chances de", r"probabilité", r"réussir?"
            ],
            "help": [
                r"aide", r"comment", r"besoin d'aide", r"je ne sais pas",
                r"peux-tu", r"pourrais-tu"
            ],
            "compare": [
                r"comparer", r"différence", r"mieux que", r"vs", r"versus",
                r"alternative", r"option"
            ],
            "risk": [
                r"risque", r"danger", r"problème", r"qui peut mal",
                r"attention", r"précaution"
            ],
            "gamification": [
                r"défi", r"challenge", r"jeu", r"quiz", r"test",
                r"niveau", r"score", r"achievement"
            ]
        }
        
        # Calcul des scores d'intention
        intent_scores = {}
        message_lower = message.lower()
        
        for intent, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, message_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score / len(patterns)  # Normalisation
        
        # Intention avec le score le plus élevé
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
        else:
            best_intent = "general"
            confidence = 0.5
        
        # Ajustement selon le mode de session
        mode_intent_boost = {
            ChatMode.PROJECT_ANALYSIS: "analyze",
            ChatMode.OPTIMIZATION: "optimize", 
            ChatMode.RISK_ASSESSMENT: "risk",
            ChatMode.GAMIFICATION: "gamification"
        }
        
        boosted_intent = mode_intent_boost.get(session.mode)
        if boosted_intent and boosted_intent in intent_scores:
            intent_scores[boosted_intent] *= 1.5  # Boost de 50%
            if intent_scores[boosted_intent] > confidence:
                best_intent = boosted_intent
                confidence = intent_scores[boosted_intent]
        
        return {
            "intent": best_intent,
            "confidence": min(1.0, confidence),
            "all_scores": intent_scores,
            "entities": self._extract_entities_from_message(message)
        }
    
    def _extract_entities_from_message(self, message: str) -> Dict[str, Any]:
        """Extrait les entités du message utilisateur"""
        
        entities = {}
        
        # Extraction de nombres (budget, durée, équipe)
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', message)
        if numbers:
            entities['numbers'] = [float(n.replace(',', '.')) for n in numbers]
        
        # Technologies mentionnées
        tech_keywords = [
            'react', 'vue', 'angular', 'node', 'python', 'java', 'php',
            'mysql', 'postgres', 'mongodb', 'redis', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'api', 'rest', 'graphql'
        ]
        
        mentioned_techs = []
        message_lower = message.lower()
        for tech in tech_keywords:
            if tech in message_lower:
                mentioned_techs.append(tech)
        
        if mentioned_techs:
            entities['technologies'] = mentioned_techs
        
        # Domaines de projet
        domain_keywords = {
            'ecommerce': ['e-commerce', 'boutique', 'vente', 'shop', 'marketplace'],
            'fintech': ['banque', 'finance', 'paiement', 'crypto', 'bancaire'],
            'healthcare': ['santé', 'médical', 'hôpital', 'patient', 'docteur'],
            'education': ['éducation', 'école', 'université', 'formation'],
            'mobile': ['mobile', 'app', 'ios', 'android', 'smartphone']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                entities['domain'] = domain
                break
        
        # Urgence/priorité
        urgency_keywords = {
            'urgent': ['urgent', 'rapidement', 'vite', 'pressé', 'deadline'],
            'normal': ['normal', 'standard', 'habituel'],
            'flexible': ['flexible', 'pas pressé', 'quand possible']
        }
        
        for urgency, keywords in urgency_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                entities['urgency'] = urgency
                break
        
        return entities
    
    async def _generate_ai_response(self, message: str, session: ChatSession,
                                  context_analysis: Dict[str, Any],
                                  intent_analysis: Dict[str, Any]) -> str:
        """Génère une réponse IA basée sur l'intention et le contexte"""
        
        intent = intent_analysis.get("intent", "general")
        entities = intent_analysis.get("entities", {})
        
        # Routage selon l'intention
        if intent == "analyze":
            return await self._handle_analyze_request(message, session, entities)
        elif intent == "optimize":
            return await self._handle_optimize_request(message, session, entities)
        elif intent == "predict":
            return await self._handle_predict_request(message, session, entities)
        elif intent == "risk":
            return await self._handle_risk_request(message, session, entities)
        elif intent == "gamification":
            return await self._handle_gamification_request(message, session, entities)
        elif intent == "compare":
            return await self._handle_compare_request(message, session, entities)
        else:
            return await self._handle_general_request(message, session, context_analysis)
    
    async def _handle_analyze_request(self, message: str, session: ChatSession, 
                                    entities: Dict[str, Any]) -> str:
        """Gère les demandes d'analyse"""
        
        if not session.active_project_data:
            return """🔍 **Analyse de Projet**
            
Pour analyser votre projet, j'aurais besoin de quelques informations :

• **Type de projet** (web app, mobile, e-commerce, etc.)
• **Taille d'équipe** estimée
• **Budget** approximatif
• **Durée** prévue
• **Technologies** envisagées

Vous pouvez me donner ces infos ou charger directement vos données de projet dans PlannerIA !"""
        
        # Utilisation de l'assistant conversationnel
        conv_context = ConversationContext(
            user_id=session.user_id,
            session_id=session.session_id,
            project_context=session.active_project_data
        )
        
        response = await ai_assistant.process_message(message, conv_context)
        
        return f"🔍 **Analyse IA de votre projet**\n\n{response}"
    
    async def _handle_optimize_request(self, message: str, session: ChatSession,
                                     entities: Dict[str, Any]) -> str:
        """Gère les demandes d'optimisation"""
        
        if not session.active_project_data:
            return """⚡ **Optimisation Intelligente**
            
Je peux vous aider à optimiser plusieurs aspects :

• 💰 **Budget** - Réduction des coûts sans compromis qualité
• ⏱️ **Timeline** - Accélération intelligente du développement  
• 👥 **Équipe** - Allocation optimale des ressources
• 🔧 **Processus** - Amélioration des workflows
• 📈 **Performance** - Optimisation technique

Quel aspect vous intéresse le plus ?"""
        
        # Analyse d'optimisation avec l'IA prédictive
        try:
            optimization_analysis = await ai_predictor.predict_project_outcome(session.active_project_data)
            
            response = "⚡ **Suggestions d'optimisation IA**\n\n"
            
            for suggestion in optimization_analysis.optimization_suggestions[:5]:
                response += f"• {suggestion}\n"
            
            if optimization_analysis.success_probability < 0.7:
                response += f"\n⚠️ **Attention** : Probabilité de succès actuelle: {optimization_analysis.success_probability:.1%}"
                response += "\nJe recommande fortement d'appliquer ces optimisations !"
            
            return response
            
        except Exception as e:
            return "⚡ **Optimisation**\n\nJe peux analyser votre projet pour identifier les opportunités d'optimisation. Partagez-moi vos données de projet ou décrivez votre situation actuelle !"
    
    async def _handle_predict_request(self, message: str, session: ChatSession,
                                    entities: Dict[str, Any]) -> str:
        """Gère les demandes de prédiction"""
        
        if not session.active_project_data:
            return """🔮 **Prédictions IA**
            
Je peux prédire pour votre projet :

• ⏰ **Durée réelle** vs estimation initiale
• 💰 **Coût final** avec intervalles de confiance  
• 📊 **Probabilité de succès** basée sur les caractéristiques
• ⚠️ **Risques potentiels** et leur impact
• 🎯 **Jalons critiques** à surveiller

Donnez-moi les détails de votre projet pour des prédictions précises !"""
        
        try:
            predictions = await ai_predictor.predict_project_outcome(session.active_project_data)
            
            response = f"""🔮 **Prédictions IA pour votre projet**

📊 **Métriques Prédites:**
• **Durée**: {predictions.predicted_duration:.0f} jours
• **Coût**: {predictions.predicted_cost:,.0f}€ 
• **Succès**: {predictions.success_probability:.1%} de probabilité

🎯 **Intervalle de confiance**: {predictions.confidence_interval[0]:.0f} - {predictions.confidence_interval[1]:.0f} jours

⚠️ **Facteurs de risque principaux:**"""
            
            for risk, score in predictions.risk_factors.items():
                if score > 0.6:
                    response += f"\n• {risk.replace('_', ' ').title()}: {score:.1%}"
            
            return response
            
        except Exception:
            return "🔮 **Prédictions IA**\n\nPour des prédictions précises, j'aurais besoin des données de votre projet. Décrivez-moi votre projet ou chargez vos données !"
    
    async def _handle_risk_request(self, message: str, session: ChatSession,
                                 entities: Dict[str, Any]) -> str:
        """Gère les demandes d'analyse de risques"""
        
        if not session.active_project_data:
            return """⚠️ **Analyse des Risques IA**
            
Je peux identifier et analyser :

• 🔧 **Risques techniques** - Complexité, intégrations, performance
• 💰 **Risques financiers** - Dépassements, coûts cachés
• 👥 **Risques d'équipe** - Turnover, compétences, communication  
• ⏱️ **Risques planning** - Retards, dépendances critiques
• 🏢 **Risques business** - Marché, concurrence, réglementation

Partagez vos données projet pour une analyse complète !"""
        
        try:
            risk_profile = await ai_risk_predictor.analyze_project_risks(session.active_project_data)
            
            response = f"""⚠️ **Analyse des Risques IA**

🎯 **Score de risque global**: {risk_profile.overall_risk_score:.1%} 

📊 **Répartition par catégorie:**"""
            
            for category, score in risk_profile.risk_distribution.items():
                if score > 0.3:  # Seulement les risques significatifs
                    emoji = "🔴" if score > 0.7 else "🟡" if score > 0.5 else "🟢"
                    response += f"\n{emoji} {category.value.title()}: {score:.1%}"
            
            response += f"\n\n🚨 **Top 3 risques critiques:**"
            for risk in risk_profile.predicted_risks[:3]:
                response += f"\n• **{risk.name}** ({risk.predicted_level.name})"
                if risk.mitigation_strategies:
                    response += f"\n  → {risk.mitigation_strategies[0]}"
            
            return response
            
        except Exception:
            return "⚠️ **Analyse des Risques**\n\nPour identifier les risques spécifiques, décrivez-moi votre projet : type, équipe, budget, timeline, technologies..."
    
    async def _handle_gamification_request(self, message: str, session: ChatSession,
                                         entities: Dict[str, Any]) -> str:
        """Gère les demandes de gamification"""
        
        try:
            # Vérifier si l'utilisateur est enregistré dans le système de gamification
            if session.user_id not in gamification_engine.players:
                # Enregistrer automatiquement
                player_name = f"Player_{session.user_id}"
                gamification_engine.register_player(session.user_id, player_name)
            
            # Génération d'un défi personnalisé
            challenge = gamification_engine.get_personalized_challenge(session.user_id)
            
            response = f"""🎮 **Défi Personnalisé pour Vous !**

🏆 **{challenge.title}**

📝 **Description:**
{challenge.description}

🎯 **Objectifs:**"""
            
            for i, objective in enumerate(challenge.objectives, 1):
                response += f"\n{i}. {objective}"
            
            response += f"\n\n⏱️ **Temps alloué**: {challenge.time_limit}"
            response += f"\n💎 **Récompenses**: {challenge.rewards.get('xp', 0)} XP"
            
            if challenge.hints:
                response += f"\n\n💡 **Hints:**"
                for hint in challenge.hints[:2]:  # Premiers indices seulement
                    response += f"\n• {hint}"
            
            response += f"\n\n🚀 **Prêt à relever le défi ?** Dites-moi \"Je commence\" pour démarrer !"
            
            return response
            
        except Exception:
            return """🎮 **Système de Gamification**
            
Prêt pour des défis stimulants ?

• 🎯 **Défis d'estimation** - Testez votre précision
• ⚡ **Challenges d'optimisation** - Trouvez les meilleures solutions  
• 🛡️ **Gestion des risques** - Identifiez et mitigez
• 👥 **Team building** - Organisez des équipes parfaites
• 💡 **Innovation** - Proposez des solutions créatives

Quel type de défi vous intéresse ?"""
    
    async def _handle_compare_request(self, message: str, session: ChatSession,
                                    entities: Dict[str, Any]) -> str:
        """Gère les demandes de comparaison"""
        
        return """🔄 **Comparaisons Intelligentes**

Je peux comparer pour vous :

• 🏗️ **Architectures** - Microservices vs Monolithique
• 💻 **Technologies** - React vs Vue, AWS vs Azure
• 📊 **Méthodologies** - Agile vs Waterfall  
• 💰 **Scénarios budgétaires** - Différentes allocations
• 👥 **Compositions d'équipe** - Optimisations possibles
• ⏱️ **Plannings** - Timeline conservative vs agressive

Que souhaitez-vous comparer spécifiquement ?"""
    
    async def _handle_general_request(self, message: str, session: ChatSession,
                                    context_analysis: Dict[str, Any]) -> str:
        """Gère les demandes générales"""
        
        # Utilisation de l'assistant conversationnel
        conv_context = ConversationContext(
            user_id=session.user_id,
            session_id=session.session_id,
            project_context=session.active_project_data
        )
        
        return await ai_assistant.process_message(message, conv_context)
    
    def render_chat_interface(self, session_id: str):
        """Rend l'interface de chat dans Streamlit"""
        
        if session_id not in self.sessions:
            st.error("Session de chat non trouvée")
            return
        
        session = self.sessions[session_id]
        
        # En-tête du chat
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"💬 Chat IA - Mode: {session.mode.value.title()}")
        
        with col2:
            # Bouton changement de mode
            new_mode = st.selectbox(
                "Mode",
                options=[mode.value for mode in ChatMode],
                index=list(ChatMode).index(session.mode),
                key=f"mode_selector_{session_id}"
            )
            if new_mode != session.mode.value:
                session.mode = ChatMode(new_mode)
                st.experimental_rerun()
        
        with col3:
            # Statistiques de session
            st.metric("Messages", len(session.messages))
        
        # Actions rapides
        if st.expander("🚀 Actions rapides", expanded=False):
            self._render_quick_actions(session_id)
        
        # Zone de chat
        chat_container = st.container()
        
        with chat_container:
            # Affichage des messages
            for message in session.messages[-20:]:  # 20 derniers messages
                self._render_chat_message(message)
        
        # Zone de saisie
        self._render_chat_input(session_id)
    
    def _render_quick_actions(self, session_id: str):
        """Rend les actions rapides"""
        
        session = self.sessions[session_id]
        
        # Organisation en colonnes
        cols = st.columns(4)
        
        actions_list = list(self.quick_actions.items())
        
        for i, (action_key, action_data) in enumerate(actions_list):
            col_idx = i % 4
            
            with cols[col_idx]:
                if st.button(
                    f"{action_data['icon']} {action_data['label']}", 
                    key=f"quick_action_{action_key}_{session_id}",
                    help=action_data['description']
                ):
                    # Traitement de l'action rapide
                    asyncio.create_task(
                        self._handle_quick_action(session_id, action_key, action_data)
                    )
                    st.experimental_rerun()
    
    async def _handle_quick_action(self, session_id: str, action_key: str, action_data: Dict[str, Any]):
        """Traite une action rapide"""
        
        quick_prompt = action_data.get('quick_prompt', 'Action rapide')
        
        # Simule un message utilisateur avec l'action rapide
        await self.process_message(session_id, quick_prompt)
    
    def _render_chat_message(self, message: ChatMessage):
        """Rend un message de chat"""
        
        if message.type == ChatMessageType.USER:
            # Message utilisateur (aligné à droite)
            st.markdown(f"""
            <div style="text-align: right; margin: 10px 0;">
                <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 15px; display: inline-block; max-width: 70%;">
                    {message.content}
                </div>
                <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                    {message.timestamp.strftime('%H:%M')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif message.type == ChatMessageType.ASSISTANT:
            # Message assistant (aligné à gauche)
            st.markdown(f"""
            <div style="text-align: left; margin: 10px 0;">
                <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 15px; border-radius: 15px; display: inline-block; max-width: 85%;">
                    {message.content}
                </div>
                <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                    🤖 Assistant IA • {message.timestamp.strftime('%H:%M')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Suggestions si disponibles
            if message.suggestions:
                st.markdown("**💡 Suggestions:**")
                cols = st.columns(min(3, len(message.suggestions)))
                
                for i, suggestion in enumerate(message.suggestions[:3]):
                    with cols[i % 3]:
                        if st.button(
                            suggestion, 
                            key=f"suggestion_{message.message_id}_{i}",
                            type="secondary"
                        ):
                            # Traitement de la suggestion
                            st.session_state[f'suggested_message'] = suggestion
                            st.experimental_rerun()
    
    def _render_chat_input(self, session_id: str):
        """Rend la zone de saisie du chat"""
        
        # Vérifier si une suggestion a été sélectionnée
        suggested_message = st.session_state.get('suggested_message', '')
        if suggested_message:
            st.session_state['suggested_message'] = ''  # Reset
            default_text = suggested_message
        else:
            default_text = ''
        
        # Zone de saisie
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Votre message:",
                value=default_text,
                key=f"chat_input_{session_id}",
                placeholder="Tapez votre message ici..."
            )
        
        with col2:
            send_button = st.button("📤 Envoyer", key=f"send_button_{session_id}")
        
        # Traitement du message
        if (send_button or user_input) and user_input.strip():
            with st.spinner("🤖 L'IA réfléchit..."):
                try:
                    # Traitement asynchrone simulé
                    response_message = asyncio.run(
                        self.process_message(session_id, user_input)
                    )
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Erreur de traitement: {str(e)}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Retourne un résumé de la session de chat"""
        
        if session_id not in self.sessions:
            return {"error": "Session non trouvée"}
        
        session = self.sessions[session_id]
        
        # Analyse des messages
        user_messages = [m for m in session.messages if m.type == ChatMessageType.USER]
        assistant_messages = [m for m in session.messages if m.type == ChatMessageType.ASSISTANT]
        
        # Extraction des intentions principales
        intents = []
        for msg in assistant_messages:
            if 'intent' in msg.metadata:
                intents.append(msg.metadata['intent'])
        
        intent_counts = {intent: intents.count(intent) for intent in set(intents)}
        
        return {
            'session_info': {
                'session_id': session_id,
                'user_id': session.user_id,
                'mode': session.mode.value,
                'duration': (datetime.now() - session.created_at).total_seconds() / 60,  # en minutes
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat()
            },
            'message_stats': {
                'total_messages': len(session.messages),
                'user_messages': len(user_messages),
                'assistant_messages': len(assistant_messages)
            },
            'interaction_analysis': {
                'main_intents': intent_counts,
                'has_project_data': session.active_project_data is not None,
                'mode_changes': 1  # À améliorer avec historique
            },
            'context': session.context
        }


class ContextAnalyzer:
    """Analyseur de contexte conversationnel"""
    
    async def analyze_message(self, message: str, session: ChatSession) -> Dict[str, Any]:
        """Analyse le contexte d'un message"""
        
        context = {
            'message_length': len(message),
            'has_questions': '?' in message,
            'politeness_level': self._assess_politeness(message),
            'technical_complexity': self._assess_technical_complexity(message),
            'urgency_level': self._assess_urgency(message),
            'conversation_phase': self._determine_conversation_phase(session),
            'user_engagement': self._assess_user_engagement(session)
        }
        
        return context
    
    def _assess_politeness(self, message: str) -> float:
        """Évalue le niveau de politesse du message"""
        
        polite_indicators = ['merci', 'sil vous plaît', 'bonjour', 'bonsoir', 'excusez-moi', 'pardon']
        message_lower = message.lower()
        
        politeness_score = sum(1 for indicator in polite_indicators if indicator in message_lower)
        return min(1.0, politeness_score / 3)  # Normalisation
    
    def _assess_technical_complexity(self, message: str) -> float:
        """Évalue la complexité technique du message"""
        
        technical_terms = [
            'api', 'database', 'architecture', 'microservices', 'kubernetes',
            'react', 'vue', 'angular', 'node', 'python', 'java', 'docker',
            'aws', 'azure', 'gcp', 'devops', 'ci/cd', 'monitoring'
        ]
        
        message_lower = message.lower()
        tech_score = sum(1 for term in technical_terms if term in message_lower)
        
        return min(1.0, tech_score / 5)  # Normalisation
    
    def _assess_urgency(self, message: str) -> float:
        """Évalue le niveau d'urgence du message"""
        
        urgent_indicators = ['urgent', 'rapidement', 'vite', 'pressé', 'deadline', 'asap', 'maintenant']
        message_lower = message.lower()
        
        urgency_score = sum(1 for indicator in urgent_indicators if indicator in message_lower)
        return min(1.0, urgency_score / 2)  # Normalisation
    
    def _determine_conversation_phase(self, session: ChatSession) -> str:
        """Détermine la phase de conversation"""
        
        message_count = len(session.messages)
        
        if message_count <= 3:
            return "introduction"
        elif message_count <= 10:
            return "exploration"
        elif message_count <= 20:
            return "deepening"
        else:
            return "advanced"
    
    def _assess_user_engagement(self, session: ChatSession) -> float:
        """Évalue l'engagement de l'utilisateur"""
        
        user_messages = [m for m in session.messages if m.type == ChatMessageType.USER]
        
        if not user_messages:
            return 0.0
        
        # Facteurs d'engagement
        avg_message_length = np.mean([len(m.content) for m in user_messages])
        question_ratio = sum(1 for m in user_messages if '?' in m.content) / len(user_messages)
        
        # Score composite
        engagement = (avg_message_length / 100) * 0.6 + question_ratio * 0.4
        return min(1.0, engagement)


class SmartSuggestionEngine:
    """Moteur de suggestions intelligentes"""
    
    async def generate_suggestions(self, user_message: str, session: ChatSession,
                                 context_analysis: Dict[str, Any]) -> List[str]:
        """Génère des suggestions contextuelles"""
        
        suggestions = []
        
        # Suggestions basées sur le contexte
        conversation_phase = context_analysis.get('conversation_phase', 'exploration')
        
        if conversation_phase == "introduction":
            suggestions.extend([
                "Analyser mon projet en détail",
                "Quelles sont les meilleures pratiques ?",
                "Montrer un exemple concret"
            ])
        
        elif conversation_phase == "exploration":
            suggestions.extend([
                "Aller plus loin dans l'analyse",
                "Voir les alternatives possibles", 
                "Quels sont les risques ?"
            ])
        
        # Suggestions basées sur le mode
        mode_suggestions = {
            ChatMode.PROJECT_ANALYSIS: [
                "Projets similaires ?",
                "Probabilité de succès ?",
                "Points d'amélioration ?"
            ],
            ChatMode.OPTIMIZATION: [
                "Autres optimisations possibles ?",
                "Impact sur la qualité ?",
                "ROI de ces améliorations ?"
            ],
            ChatMode.RISK_ASSESSMENT: [
                "Plans de mitigation ?",
                "Signaux d'alerte précoce ?",
                "Risques cachés ?"
            ]
        }
        
        mode_suggs = mode_suggestions.get(session.mode, [])
        suggestions.extend(mode_suggs)
        
        # Suggestions basées sur le contenu du message
        if any(word in user_message.lower() for word in ['budget', 'coût', 'prix']):
            suggestions.append("Optimiser le budget davantage")
        
        if any(word in user_message.lower() for word in ['équipe', 'team', 'développeur']):
            suggestions.append("Optimiser la composition de l'équipe")
        
        if any(word in user_message.lower() for word in ['risque', 'problème', 'danger']):
            suggestions.append("Stratégies de prévention")
        
        # Retourner les suggestions uniques, limitées à 4
        unique_suggestions = list(dict.fromkeys(suggestions))  # Supprime doublons
        return unique_suggestions[:4]


class ResponsePersonalizer:
    """Personnalisateur de réponses IA"""
    
    def __init__(self):
        self.user_preferences = {}  # À charger depuis une base de données
    
    async def personalize_response(self, response: str, user_id: str,
                                 context_analysis: Dict[str, Any]) -> str:
        """Personnalise une réponse selon l'utilisateur"""
        
        # Ajustements selon la politesse de l'utilisateur
        politeness_level = context_analysis.get('politeness_level', 0.5)
        
        if politeness_level > 0.7:
            response = self._add_polite_touches(response)
        
        # Ajustements selon la complexité technique
        tech_complexity = context_analysis.get('technical_complexity', 0.5)
        
        if tech_complexity > 0.7:
            response = self._add_technical_depth(response)
        elif tech_complexity < 0.3:
            response = self._simplify_technical_language(response)
        
        # Ajustements selon l'urgence
        urgency_level = context_analysis.get('urgency_level', 0.3)
        
        if urgency_level > 0.7:
            response = self._add_urgency_acknowledgment(response)
        
        return response
    
    def _add_polite_touches(self, response: str) -> str:
        """Ajoute des touches polies à la réponse"""
        
        polite_starters = [
            "Je vous remercie pour cette question intéressante ! ",
            "C'est un plaisir de vous aider. ",
            "Excellente question ! "
        ]
        
        if not response.startswith(tuple(['Merci', 'Excellent', 'C\'est un plaisir'])):
            response = np.random.choice(polite_starters) + response
        
        return response
    
    def _add_technical_depth(self, response: str) -> str:
        """Ajoute de la profondeur technique"""
        
        # Logique pour enrichir techniquement la réponse
        return response
    
    def _simplify_technical_language(self, response: str) -> str:
        """Simplifie le langage technique"""
        
        # Remplacements de termes techniques par des équivalents plus simples
        simplifications = {
            'architecture microservices': 'architecture en petits services',
            'API REST': 'interface de communication',
            'DevOps': 'automatisation du développement',
            'CI/CD': 'déploiement automatique'
        }
        
        for tech_term, simple_term in simplifications.items():
            response = response.replace(tech_term, simple_term)
        
        return response
    
    def _add_urgency_acknowledgment(self, response: str) -> str:
        """Reconnaît l'urgence dans la réponse"""
        
        urgency_acknowledgments = [
            "Je comprends l'urgence de votre situation. ",
            "Réponse rapide pour votre demande urgente : ",
            "Compte tenu de l'urgence, voici l'essentiel : "
        ]
        
        return np.random.choice(urgency_acknowledgments) + response


# Instance globale
smart_chat = SmartChatInterface()