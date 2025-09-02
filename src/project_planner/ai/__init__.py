"""
🧠 Module IA PlannerIA - Intelligence Artificielle Intégrée
Hub central pour tous les composants d'IA avancée
"""

# Importation des modules IA principaux
from .predictive_engine import ai_predictor, AIProjectPredictor
from .conversational_assistant import ai_assistant, AIConversationalAssistant
from .smart_alerts import smart_alert_system, SmartAlertSystem
from .gamification_engine import gamification_engine, IntelligentGamificationEngine
from .risk_predictor import ai_risk_predictor, AIRiskPredictor
from .chat_interface import smart_chat, SmartChatInterface
from .budget_optimizer import ai_budget_optimizer, AIBudgetOptimizer
from .crisis_predictor import ai_crisis_predictor, AICrisisPredictor
from .personal_coach import ai_personal_coach, AIPersonalCoach, coach_observe, initialize_personal_coach
from .rag_manager import ai_rag_manager, RAGManagerIntelligent, initialize_rag_manager
from .portfolio_manager import ai_portfolio_manager, AIPortfolioManager, initialize_portfolio_manager

# Imports des classes et enums utiles
from .predictive_engine import ProjectDomain, PredictionResult
from .smart_alerts import AlertLevel, AlertCategory, Alert
from .gamification_engine import ChallengeType, DifficultyLevel, Achievement
from .risk_predictor import RiskCategory, RiskLevel, PredictedRisk
from .chat_interface import ChatMessageType, ChatMode
from .budget_optimizer import BudgetCategory, OptimizationObjective
from .crisis_predictor import CrisisType, CrisisSeverity, CrisisProfile
from .personal_coach import CoachingArea, PersonalityTrait, CoachingStyle
from .rag_manager import DocumentType, SourceReliability, QueryContext
from .portfolio_manager import ProjectStatus, ProjectPriority, ProjectTemplate, ResourceType

# Version du module IA
__version__ = "1.0.0"

# Instances globales disponibles
__all__ = [
    # Instances principales
    'ai_predictor',
    'ai_assistant', 
    'smart_alert_system',
    'gamification_engine',
    'ai_risk_predictor',
    'smart_chat',
    'ai_budget_optimizer',
    'ai_crisis_predictor',
    'ai_personal_coach',
    'ai_rag_manager',
    'ai_portfolio_manager',
    
    # Classes pour instanciation personnalisée
    'AIProjectPredictor',
    'AIConversationalAssistant',
    'SmartAlertSystem',
    'IntelligentGamificationEngine', 
    'AIRiskPredictor',
    'SmartChatInterface',
    'AIBudgetOptimizer',
    'AICrisisPredictor',
    'AIPersonalCoach',
    'RAGManagerIntelligent',
    'AIPortfolioManager',
    
    # Enums et types
    'ProjectDomain',
    'AlertLevel', 
    'AlertCategory',
    'ChallengeType',
    'DifficultyLevel',
    'RiskCategory',
    'RiskLevel',
    'ChatMessageType',
    'ChatMode',
    'BudgetCategory',
    'OptimizationObjective',
    'CrisisType',
    'CrisisSeverity',
    'CoachingArea',
    'PersonalityTrait',
    'CoachingStyle',
    'DocumentType',
    'SourceReliability',
    'QueryContext',
    'ProjectStatus',
    'ProjectPriority',
    'ProjectTemplate',
    'ResourceType',
    
    # Classes de données
    'PredictionResult',
    'Alert',
    'Achievement',
    'PredictedRisk',
    'CrisisProfile',
    
    # Fonctions utiles
    'coach_observe',
    'initialize_personal_coach',
    'initialize_rag_manager',
    'initialize_portfolio_manager'
]

# Métadonnées du module
AI_MODULE_INFO = {
    "name": "PlannerIA AI Engine",
    "version": __version__,
    "description": "Intelligence artificielle avancée pour gestion de projets",
    "components": {
        "portfolio_manager": "Gestionnaire intelligent de portefeuille multi-projets",
        "predictive_engine": "Prédictions et patterns de projets",
        "conversational_assistant": "Assistant IA conversationnel", 
        "smart_alerts": "Alertes intelligentes avec détection précoce",
        "gamification_engine": "Gamification adaptative avec défis IA",
        "risk_predictor": "Analyse prédictive des risques avec ML",
        "chat_interface": "Interface chat intelligente", 
        "budget_optimizer": "Optimisation budgétaire avec allocation IA",
        "crisis_predictor": "Prédicteur de crises avec signaux faibles",
        "personal_coach": "Coach personnel IA adaptatif et transversal",
        "rag_manager": "Gestionnaire RAG intelligent avec auto-enrichissement"
    },
    "capabilities": [
        "Gestion intelligente de portefeuille multi-projets avec optimisation",
        "Prédiction de durée et coût avec ML",
        "Analyse conversationnelle en langage naturel",
        "Détection d'anomalies et alertes précoces", 
        "Défis gamifiés adaptatifs personnalisés",
        "Prédiction de risques avec patterns",
        "Chat contextuel multimodal",
        "Optimisation budgétaire avec contraintes",
        "Détection de signaux faibles et prédiction de crises",
        "Coaching personnalisé adaptatif avec analyse comportementale",
        "Gestion documentaire intelligente avec RAG et auto-enrichissement"
    ]
}

def get_ai_status():
    """Retourne le statut de tous les modules IA"""
    
    status = {
        "module_info": AI_MODULE_INFO,
        "components_status": {},
        "overall_health": "healthy"
    }
    
    # Vérification de chaque composant
    components = {
        "portfolio_manager": ai_portfolio_manager,
        "predictive_engine": ai_predictor,
        "conversational_assistant": ai_assistant,
        "smart_alerts": smart_alert_system, 
        "gamification_engine": gamification_engine,
        "risk_predictor": ai_risk_predictor,
        "chat_interface": smart_chat,
        "budget_optimizer": ai_budget_optimizer,
        "crisis_predictor": ai_crisis_predictor,
        "personal_coach": ai_personal_coach,
        "rag_manager": ai_rag_manager
    }
    
    for name, component in components.items():
        try:
            # Test basique de sanité du composant
            component_status = {
                "loaded": True,
                "class": component.__class__.__name__,
                "methods": len([method for method in dir(component) if not method.startswith('_')]),
                "status": "operational"
            }
            
            # Tests spécifiques par composant
            if hasattr(component, 'trained'):
                component_status["trained"] = getattr(component, 'trained', False)
                
            if hasattr(component, 'patterns_db'):
                component_status["patterns_loaded"] = len(getattr(component, 'patterns_db', {}))
                
            if hasattr(component, 'sessions'):
                component_status["active_sessions"] = len(getattr(component, 'sessions', {}))
                
        except Exception as e:
            component_status = {
                "loaded": False,
                "status": "error",
                "error": str(e)
            }
            status["overall_health"] = "degraded"
        
        status["components_status"][name] = component_status
    
    return status

def initialize_ai_system():
    """Initialise le système IA complet"""
    
    print("[AI] Initialisation du système IA PlannerIA...")
    
    initialization_results = {
        "portfolio_manager": False,
        "predictive_engine": False,
        "conversational_assistant": False, 
        "smart_alerts": False,
        "gamification_engine": False,
        "risk_predictor": False,
        "chat_interface": False,
        "budget_optimizer": False,
        "crisis_predictor": False,
        "personal_coach": False,
        "rag_manager": False
    }
    
    try:
        # Initialisation du Portfolio Manager
        print("[PORTFOLIO] Chargement du gestionnaire de portefeuille...")
        if hasattr(ai_portfolio_manager, 'projects'):
            initialization_results["portfolio_manager"] = True
            print("  [OK] Portfolio Manager: Système multi-projets initialisé")
        
        # Initialisation du moteur prédictif
        print("[PREDICT] Chargement du moteur prédictif...")
        if hasattr(ai_predictor, 'patterns_db'):
            initialization_results["predictive_engine"] = True
            print("  [OK] Moteur prédictif: Patterns chargés")
        
        # Initialisation de l'assistant conversationnel
        print("[ASSISTANT] Chargement de l'assistant conversationnel...")
        if hasattr(ai_assistant, 'intent_patterns'):
            initialization_results["conversational_assistant"] = True
            print("  [OK] Assistant conversationnel: Patterns d'intention chargés")
        
        # Initialisation du système d'alertes
        print("[ALERTS] Chargement du système d'alertes...")
        if hasattr(smart_alert_system, 'alert_rules'):
            initialization_results["smart_alerts"] = True
            print(f"  [OK] Alertes intelligentes: {len(smart_alert_system.alert_rules)} règles chargées")
        
        # Initialisation du moteur de gamification
        print("[GAMIFY] Chargement du moteur de gamification...")
        if hasattr(gamification_engine, 'challenges_pool'):
            challenges_count = sum(len(challenges) for challenges in gamification_engine.challenges_pool.values())
            initialization_results["gamification_engine"] = True
            print(f"  [OK] Gamification: {challenges_count} défis disponibles")
        
        # Initialisation du prédicteur de risques
        print("[RISK] Chargement du prédicteur de risques...")
        if hasattr(ai_risk_predictor, 'models') and ai_risk_predictor.trained:
            initialization_results["risk_predictor"] = True
            print("  [OK] Prédicteur de risques: Modèles ML entraînés")
        
        # Initialisation de l'interface chat
        print("[CHAT] Chargement de l'interface chat...")
        if hasattr(smart_chat, 'quick_actions'):
            initialization_results["chat_interface"] = True
            print(f"  [OK] Chat intelligent: {len(smart_chat.quick_actions)} actions rapides")
        
        # Initialisation de l'optimiseur budgétaire
        print("[BUDGET] Chargement de l'optimiseur budgétaire...")
        if hasattr(ai_budget_optimizer, 'optimization_models'):
            initialization_results["budget_optimizer"] = True
            print("  [OK] Optimiseur budgétaire: Modèles d'optimisation chargés")
        
        # Initialisation du prédicteur de crises
        print("[CRISIS] Chargement du prédicteur de crises...")
        if hasattr(ai_crisis_predictor, 'models') and ai_crisis_predictor.trained:
            initialization_results["crisis_predictor"] = True
            print("  [OK] Prédicteur de crises: Modèles ML et patterns chargés")
        
        # Initialisation du coach personnel
        print("[COACH] Chargement du coach personnel...")
        if hasattr(ai_personal_coach, 'user_profiles'):
            initialization_results["personal_coach"] = True
            print(f"  [OK] Coach personnel: Système d'observation et {len(ai_personal_coach.user_profiles)} profils chargés")
        
        # Initialisation du RAG manager
        print("[RAG] Chargement du gestionnaire RAG...")
        if hasattr(ai_rag_manager, 'documents') and hasattr(ai_rag_manager, 'chunks'):
            initialization_results["rag_manager"] = True
            print(f"  [OK] RAG Manager: {len(ai_rag_manager.documents)} documents, {len(ai_rag_manager.chunks)} chunks indexés")
        
        success_count = sum(initialization_results.values())
        total_count = len(initialization_results)
        
        print(f"\n[RESULT] Initialisation terminée: {success_count}/{total_count} modules opérationnels")
        
        if success_count == total_count:
            print("[SUCCESS] Système IA PlannerIA entièrement opérationnel !")
            return True
        else:
            print(f"[WARNING] Système partiellement opérationnel ({success_count}/{total_count} modules)")
            return False
            
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'initialisation: {str(e)}")
        return False

def get_ai_capabilities_summary():
    """Retourne un résumé des capacités IA disponibles"""
    
    return {
        "🔮 Prédiction Intelligente": {
            "description": "Prédictions de durée, coût et succès avec IA",
            "features": [
                "Patterns de projets par domaine",
                "Prédictions ML avec intervalles de confiance", 
                "Recommandations contextuelles",
                "Projets similaires et benchmarks"
            ]
        },
        "🤖 Assistant Conversationnel": {
            "description": "Assistant IA en langage naturel",
            "features": [
                "Compréhension d'intentions avancée",
                "Analyse contextuelle de projets",
                "Recommandations personnalisées",
                "Support multilingue"
            ]
        },
        "🚨 Alertes Intelligentes": {
            "description": "Détection précoce et alertes contextuelles",
            "features": [
                "Détection d'anomalies en temps réel",
                "Prédiction de problèmes futurs",
                "Alertes personnalisées par règles",
                "Corrélations entre métriques"
            ]
        },
        "🎮 Gamification Adaptative": {
            "description": "Défis personnalisés et progression gamifiée",
            "features": [
                "Défis adaptatifs par niveau",
                "Système d'achievements évolutif",
                "Coach IA personnel",
                "Progression et classements"
            ]
        },
        "⚠️ Analyse Prédictive des Risques": {
            "description": "ML avancé pour prédiction et mitigation des risques",
            "features": [
                "Modèles ML entrainés sur 1000+ projets",
                "Détection de patterns de risque",
                "Stratégies de mitigation automatiques",
                "Corrélations et cascades de risques"
            ]
        },
        "💬 Chat Intelligent": {
            "description": "Interface conversationnelle contextuelle",
            "features": [
                "Modes spécialisés par contexte",
                "Actions rapides intelligentes",
                "Personnalisation des réponses",
                "Suggestions contextuelles"
            ]
        },
        "💰 Optimisation Budgétaire": {
            "description": "Allocation intelligente et optimisation des coûts",
            "features": [
                "Optimisation mathématique avec contraintes",
                "Benchmarks industrie par domaine",
                "Stratégies d'économies personnalisées",
                "Analyse ROI multi-critères"
            ]
        }
    }

# Initialisation automatique à l'import
_ai_initialized = False

def ensure_ai_initialized():
    """S'assure que le système IA est initialisé"""
    global _ai_initialized
    
    if not _ai_initialized:
        _ai_initialized = initialize_ai_system()
    
    return _ai_initialized

# Initialisation complète pour présentation
ensure_ai_initialized()