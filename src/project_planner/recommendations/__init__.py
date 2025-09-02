"""
Smart Recommendation System for PlannerIA
Système de recommandations intelligentes avec ML prédictif
"""

from .recommendation_engine import (
    SmartRecommendationEngine,
    Recommendation,
    RecommendationType,
    RecommendationPriority,
    get_recommendation_engine,
    generate_project_recommendations,
    get_contextual_suggestions
)

from .behavioral_analyzer import (
    BehavioralAnalyzer,
    UserPreference,
    BehaviorPattern,
    get_behavioral_analyzer,
    track_user_action,
    get_user_preferences
)

from .knowledge_base import (
    DynamicKnowledgeBase,
    BestPractice,
    IndustryPattern,
    get_knowledge_base,
    get_industry_patterns,
    get_best_practices
)

__all__ = [
    'SmartRecommendationEngine',
    'Recommendation',
    'RecommendationType',
    'RecommendationPriority',
    'BehavioralAnalyzer',
    'UserPreference',
    'BehaviorPattern',
    'DynamicKnowledgeBase',
    'BestPractice',
    'IndustryPattern',
    'get_recommendation_engine',
    'generate_project_recommendations',
    'get_contextual_suggestions',
    'get_behavioral_analyzer',
    'track_user_action',
    'get_user_preferences',
    'get_knowledge_base',
    'get_industry_patterns',
    'get_best_practices'
]