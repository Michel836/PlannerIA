"""
Dynamic Knowledge Base for Smart Recommendations
Base de connaissances dynamique avec meilleures pratiques et patterns industriels
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class IndustryType(Enum):
    """Types d'industries supportés"""
    SOFTWARE = "software"
    CONSTRUCTION = "construction"
    MARKETING = "marketing"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    MANUFACTURING = "manufacturing"
    CONSULTING = "consulting"
    RESEARCH = "research"
    GENERAL = "general"

class PracticeCategory(Enum):
    """Catégories de meilleures pratiques"""
    PLANNING = "planning"
    RISK_MANAGEMENT = "risk_management"
    TEAM_MANAGEMENT = "team_management"
    BUDGET_CONTROL = "budget_control"
    QUALITY_ASSURANCE = "quality_assurance"
    COMMUNICATION = "communication"
    TOOL_USAGE = "tool_usage"
    METHODOLOGY = "methodology"

class PatternType(Enum):
    """Types de patterns industriels"""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    OPTIMIZATION_PATTERN = "optimization_pattern"
    RISK_PATTERN = "risk_pattern"
    EFFICIENCY_PATTERN = "efficiency_pattern"

@dataclass
class BestPractice:
    """Meilleure pratique avec contexte"""
    practice_id: str
    title: str
    description: str
    category: PracticeCategory
    industry: IndustryType
    success_rate: float
    application_contexts: List[str]
    prerequisites: List[str]
    implementation_steps: List[str]
    metrics: Dict[str, Any]
    source: str
    confidence: float
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'practice_id': self.practice_id,
            'title': self.title,
            'description': self.description,
            'category': self.category.value,
            'industry': self.industry.value,
            'success_rate': self.success_rate,
            'application_contexts': self.application_contexts,
            'prerequisites': self.prerequisites,
            'implementation_steps': self.implementation_steps,
            'metrics': self.metrics,
            'source': self.source,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat()
        }

@dataclass  
class IndustryPattern:
    """Pattern spécifique à une industrie"""
    pattern_id: str
    pattern_type: PatternType
    industry: IndustryType
    title: str
    description: str
    triggers: List[str]
    indicators: List[str]
    outcomes: List[str]
    frequency: int
    impact_score: float
    confidence: float
    examples: List[Dict[str, Any]]
    detected_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'industry': self.industry.value,
            'title': self.title,
            'description': self.description,
            'triggers': self.triggers,
            'indicators': self.indicators,
            'outcomes': self.outcomes,
            'frequency': self.frequency,
            'impact_score': self.impact_score,
            'confidence': self.confidence,
            'examples': self.examples,
            'detected_at': self.detected_at.isoformat()
        }

class DynamicKnowledgeBase:
    """Base de connaissances dynamique avec apprentissage"""
    
    def __init__(self, db_path: str = "data/knowledge_base.db"):
        self.db_path = db_path
        self.best_practices = {}
        self.industry_patterns = {}
        self.practice_ratings = defaultdict(list)
        self.pattern_validations = defaultdict(list)
        self.knowledge_cache = {}
        
        self._init_database()
        self._load_initial_knowledge()
        
    def _init_database(self):
        """Initialiser la base de données de connaissances"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table des meilleures pratiques
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS best_practices (
                        practice_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        category TEXT NOT NULL,
                        industry TEXT NOT NULL,
                        success_rate REAL NOT NULL,
                        application_contexts TEXT NOT NULL,
                        prerequisites TEXT NOT NULL,
                        implementation_steps TEXT NOT NULL,
                        metrics TEXT NOT NULL,
                        source TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        created_at DATETIME NOT NULL
                    )
                """)
                
                # Table des patterns industriels
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS industry_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        pattern_type TEXT NOT NULL,
                        industry TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        triggers TEXT NOT NULL,
                        indicators TEXT NOT NULL,
                        outcomes TEXT NOT NULL,
                        frequency INTEGER NOT NULL,
                        impact_score REAL NOT NULL,
                        confidence REAL NOT NULL,
                        examples TEXT NOT NULL,
                        detected_at DATETIME NOT NULL
                    )
                """)
                
                # Table des évaluations de pratiques
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS practice_ratings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        practice_id TEXT NOT NULL,
                        project_id TEXT,
                        rating INTEGER NOT NULL,
                        feedback TEXT,
                        context TEXT,
                        rated_at DATETIME NOT NULL,
                        FOREIGN KEY (practice_id) REFERENCES best_practices (practice_id)
                    )
                """)
                
                # Table des validations de patterns
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_validations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT NOT NULL,
                        project_id TEXT,
                        validated BOOLEAN NOT NULL,
                        outcome_match REAL,
                        feedback TEXT,
                        context TEXT,
                        validated_at DATETIME NOT NULL,
                        FOREIGN KEY (pattern_id) REFERENCES industry_patterns (pattern_id)
                    )
                """)
                
                conn.commit()
                logger.info("✅ Base de données de connaissances initialisée")
                
        except Exception as e:
            logger.error(f"Erreur initialisation DB connaissances: {e}")
            
    def _load_initial_knowledge(self):
        """Charger les connaissances initiales"""
        self._load_builtin_practices()
        self._load_builtin_patterns()
        self._load_stored_knowledge()
        
    def _load_builtin_practices(self):
        """Charger les pratiques pré-définies"""
        
        builtin_practices = [
            {
                'practice_id': 'agile_sprint_planning',
                'title': 'Planification par sprints Agile',
                'description': 'Division du projet en sprints courts avec objectifs clairs',
                'category': PracticeCategory.PLANNING,
                'industry': IndustryType.SOFTWARE,
                'success_rate': 0.85,
                'application_contexts': ['développement_logiciel', 'projets_itératifs'],
                'prerequisites': ['équipe_formée_agile', 'product_owner_disponible'],
                'implementation_steps': [
                    'Définir la durée du sprint (1-4 semaines)',
                    'Organiser la planification de sprint',
                    'Estimer les user stories',
                    'Planifier la revue et rétrospective'
                ],
                'metrics': {'velocity': 'story_points_per_sprint', 'burndown': 'daily_progress'},
                'source': 'builtin_agile_knowledge',
                'confidence': 0.9
            },
            {
                'practice_id': 'risk_register_maintenance',
                'title': 'Tenue d\'un registre des risques',
                'description': 'Documentation et suivi continu des risques projet',
                'category': PracticeCategory.RISK_MANAGEMENT,
                'industry': IndustryType.GENERAL,
                'success_rate': 0.78,
                'application_contexts': ['projets_complexes', 'environnement_incertain'],
                'prerequisites': ['identification_risques_initiale'],
                'implementation_steps': [
                    'Créer le registre des risques',
                    'Évaluer probabilité et impact',
                    'Définir les stratégies de mitigation',
                    'Révision régulière (weekly/monthly)'
                ],
                'metrics': {'risk_reduction': 'percentage', 'mitigation_success': 'boolean'},
                'source': 'builtin_pmi_knowledge',
                'confidence': 0.85
            },
            {
                'practice_id': 'daily_standups',
                'title': 'Réunions quotidiennes d\'équipe',
                'description': 'Points quotidiens courts pour synchronisation équipe',
                'category': PracticeCategory.COMMUNICATION,
                'industry': IndustryType.SOFTWARE,
                'success_rate': 0.82,
                'application_contexts': ['équipes_distribuées', 'projets_agiles'],
                'prerequisites': ['équipe_stable', 'engagement_participation'],
                'implementation_steps': [
                    'Fixer heure et lieu récurrents',
                    'Limiter à 15 minutes maximum',
                    'Format: fait hier, prévu aujourd\'hui, blockers',
                    'Résolution des blockers en post-standup'
                ],
                'metrics': {'attendance_rate': 'percentage', 'blocker_resolution_time': 'hours'},
                'source': 'builtin_scrum_knowledge',
                'confidence': 0.88
            },
            {
                'practice_id': 'budget_buffer_allocation',
                'title': 'Allocation de buffer budgétaire',
                'description': 'Réservation d\'une marge de sécurité dans le budget',
                'category': PracticeCategory.BUDGET_CONTROL,
                'industry': IndustryType.GENERAL,
                'success_rate': 0.75,
                'application_contexts': ['projets_incertains', 'premiers_projets'],
                'prerequisites': ['estimation_budget_base'],
                'implementation_steps': [
                    'Calculer le budget base du projet',
                    'Ajouter 10-20% selon la complexité',
                    'Séparer budget opérationnel et buffer',
                    'Approval process pour utilisation buffer'
                ],
                'metrics': {'budget_variance': 'percentage', 'buffer_usage': 'percentage'},
                'source': 'builtin_pm_knowledge',
                'confidence': 0.80
            },
            {
                'practice_id': 'code_review_mandatory',
                'title': 'Revue de code obligatoire',
                'description': 'Processus systématique de revue du code avant intégration',
                'category': PracticeCategory.QUALITY_ASSURANCE,
                'industry': IndustryType.SOFTWARE,
                'success_rate': 0.89,
                'application_contexts': ['développement_logiciel', 'projets_critiques'],
                'prerequisites': ['outil_de_revue', 'guidelines_établies'],
                'implementation_steps': [
                    'Configurer l\'outil de revue (GitHub, GitLab, etc.)',
                    'Définir les critères de revue',
                    'Assigner des reviewers compétents',
                    'Bloquer la merge sans approval'
                ],
                'metrics': {'defect_detection_rate': 'percentage', 'review_coverage': 'percentage'},
                'source': 'builtin_devops_knowledge',
                'confidence': 0.92
            }
        ]
        
        for practice_data in builtin_practices:
            practice = BestPractice(
                practice_id=practice_data['practice_id'],
                title=practice_data['title'],
                description=practice_data['description'],
                category=practice_data['category'],
                industry=practice_data['industry'],
                success_rate=practice_data['success_rate'],
                application_contexts=practice_data['application_contexts'],
                prerequisites=practice_data['prerequisites'],
                implementation_steps=practice_data['implementation_steps'],
                metrics=practice_data['metrics'],
                source=practice_data['source'],
                confidence=practice_data['confidence'],
                created_at=datetime.now()
            )
            
            self.best_practices[practice.practice_id] = practice
            self._save_best_practice(practice)
            
        logger.info(f"✅ {len(builtin_practices)} meilleures pratiques pré-définies chargées")
        
    def _load_builtin_patterns(self):
        """Charger les patterns pré-définis"""
        
        builtin_patterns = [
            {
                'pattern_id': 'software_scope_creep',
                'pattern_type': PatternType.FAILURE_PATTERN,
                'industry': IndustryType.SOFTWARE,
                'title': 'Dérive des exigences logiciel',
                'description': 'Ajouts non contrôlés de fonctionnalités pendant développement',
                'triggers': ['demandes_client_fréquentes', 'spécifications_floues'],
                'indicators': ['budget_exceeded', 'timeline_delayed', 'team_stressed'],
                'outcomes': ['project_failure', 'customer_dissatisfaction'],
                'frequency': 45,
                'impact_score': 0.8,
                'confidence': 0.85,
                'examples': [
                    {'project': 'e-commerce', 'scope_increase': '40%', 'delay': '3_months'},
                    {'project': 'mobile_app', 'scope_increase': '25%', 'delay': '6_weeks'}
                ]
            },
            {
                'pattern_id': 'agile_velocity_stabilization',
                'pattern_type': PatternType.SUCCESS_PATTERN,
                'industry': IndustryType.SOFTWARE,
                'title': 'Stabilisation de la vélocité Agile',
                'description': 'Amélioration progressive et stabilisation de la vélocité équipe',
                'triggers': ['sprint_planning_régulier', 'rétrospectives_actionables'],
                'indicators': ['velocity_trend_positive', 'predictability_improved'],
                'outcomes': ['delivery_predictable', 'team_satisfaction_high'],
                'frequency': 78,
                'impact_score': 0.7,
                'confidence': 0.82,
                'examples': [
                    {'project': 'saas_platform', 'velocity_improvement': '35%', 'time_to_stable': '4_sprints'},
                    {'project': 'api_development', 'velocity_improvement': '28%', 'time_to_stable': '3_sprints'}
                ]
            },
            {
                'pattern_id': 'budget_overrun_cascade',
                'pattern_type': PatternType.FAILURE_PATTERN,
                'industry': IndustryType.GENERAL,
                'title': 'Cascade de dépassements budgétaires',
                'description': 'Dépassements qui s\'amplifient dans les phases suivantes',
                'triggers': ['estimation_optimiste', 'risques_sous_estimés'],
                'indicators': ['early_budget_variance', 'frequent_change_requests'],
                'outcomes': ['major_budget_overrun', 'project_cancellation_risk'],
                'frequency': 32,
                'impact_score': 0.9,
                'confidence': 0.78,
                'examples': [
                    {'project': 'construction', 'initial_overrun': '15%', 'final_overrun': '45%'},
                    {'project': 'it_migration', 'initial_overrun': '12%', 'final_overrun': '38%'}
                ]
            },
            {
                'pattern_id': 'early_risk_identification_success',
                'pattern_type': PatternType.SUCCESS_PATTERN,
                'industry': IndustryType.GENERAL,
                'title': 'Succès par identification précoce des risques',
                'description': 'Projets réussis grâce à identification et mitigation précoces',
                'triggers': ['risk_workshops_early', 'stakeholder_engagement_high'],
                'indicators': ['risk_register_complete', 'mitigation_plans_ready'],
                'outcomes': ['smooth_execution', 'minimal_surprises'],
                'frequency': 67,
                'impact_score': 0.75,
                'confidence': 0.80,
                'examples': [
                    {'project': 'system_integration', 'risks_identified': 'week_1', 'success_rate': '95%'},
                    {'project': 'product_launch', 'risks_identified': 'planning_phase', 'success_rate': '88%'}
                ]
            }
        ]
        
        for pattern_data in builtin_patterns:
            pattern = IndustryPattern(
                pattern_id=pattern_data['pattern_id'],
                pattern_type=pattern_data['pattern_type'],
                industry=pattern_data['industry'],
                title=pattern_data['title'],
                description=pattern_data['description'],
                triggers=pattern_data['triggers'],
                indicators=pattern_data['indicators'],
                outcomes=pattern_data['outcomes'],
                frequency=pattern_data['frequency'],
                impact_score=pattern_data['impact_score'],
                confidence=pattern_data['confidence'],
                examples=pattern_data['examples'],
                detected_at=datetime.now()
            )
            
            self.industry_patterns[pattern.pattern_id] = pattern
            self._save_industry_pattern(pattern)
            
        logger.info(f"✅ {len(builtin_patterns)} patterns industriels pré-définis chargés")
        
    def _load_stored_knowledge(self):
        """Charger les connaissances stockées en base"""
        try:
            # Charger les pratiques depuis la DB
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM best_practices")
                
                for row in cursor.fetchall():
                    try:
                        practice = self._row_to_best_practice(row)
                        if practice.practice_id not in self.best_practices:
                            self.best_practices[practice.practice_id] = practice
                    except Exception as e:
                        logger.warning(f"Erreur chargement pratique: {e}")
                        
                # Charger les patterns depuis la DB
                cursor.execute("SELECT * FROM industry_patterns")
                
                for row in cursor.fetchall():
                    try:
                        pattern = self._row_to_industry_pattern(row)
                        if pattern.pattern_id not in self.industry_patterns:
                            self.industry_patterns[pattern.pattern_id] = pattern
                    except Exception as e:
                        logger.warning(f"Erreur chargement pattern: {e}")
                        
                logger.info(f"✅ Connaissances stockées chargées")
                
        except Exception as e:
            logger.error(f"Erreur chargement connaissances stockées: {e}")
            
    def _row_to_best_practice(self, row) -> BestPractice:
        """Convertir une ligne DB en BestPractice"""
        (practice_id, title, description, category, industry, success_rate,
         application_contexts, prerequisites, implementation_steps, metrics,
         source, confidence, created_at) = row
         
        return BestPractice(
            practice_id=practice_id,
            title=title,
            description=description,
            category=PracticeCategory(category),
            industry=IndustryType(industry),
            success_rate=success_rate,
            application_contexts=json.loads(application_contexts),
            prerequisites=json.loads(prerequisites),
            implementation_steps=json.loads(implementation_steps),
            metrics=json.loads(metrics),
            source=source,
            confidence=confidence,
            created_at=datetime.fromisoformat(created_at)
        )
        
    def _row_to_industry_pattern(self, row) -> IndustryPattern:
        """Convertir une ligne DB en IndustryPattern"""
        (pattern_id, pattern_type, industry, title, description,
         triggers, indicators, outcomes, frequency, impact_score,
         confidence, examples, detected_at) = row
         
        return IndustryPattern(
            pattern_id=pattern_id,
            pattern_type=PatternType(pattern_type),
            industry=IndustryType(industry),
            title=title,
            description=description,
            triggers=json.loads(triggers),
            indicators=json.loads(indicators),
            outcomes=json.loads(outcomes),
            frequency=frequency,
            impact_score=impact_score,
            confidence=confidence,
            examples=json.loads(examples),
            detected_at=datetime.fromisoformat(detected_at)
        )
        
    def _save_best_practice(self, practice: BestPractice):
        """Sauvegarder une meilleure pratique"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO best_practices
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    practice.practice_id,
                    practice.title,
                    practice.description,
                    practice.category.value,
                    practice.industry.value,
                    practice.success_rate,
                    json.dumps(practice.application_contexts),
                    json.dumps(practice.prerequisites),
                    json.dumps(practice.implementation_steps),
                    json.dumps(practice.metrics),
                    practice.source,
                    practice.confidence,
                    practice.created_at
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde pratique: {e}")
            
    def _save_industry_pattern(self, pattern: IndustryPattern):
        """Sauvegarder un pattern industriel"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO industry_patterns
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_type.value,
                    pattern.industry.value,
                    pattern.title,
                    pattern.description,
                    json.dumps(pattern.triggers),
                    json.dumps(pattern.indicators),
                    json.dumps(pattern.outcomes),
                    pattern.frequency,
                    pattern.impact_score,
                    pattern.confidence,
                    json.dumps(pattern.examples),
                    pattern.detected_at
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde pattern: {e}")
            
    def get_relevant_practices(self, 
                             project_context: Dict[str, Any],
                             limit: int = 10) -> List[BestPractice]:
        """Obtenir les meilleures pratiques pertinentes pour un projet"""
        
        industry = project_context.get('industry', IndustryType.GENERAL)
        project_type = project_context.get('project_type', '')
        team_size = project_context.get('team_size', 0)
        complexity = project_context.get('complexity', 'medium')
        
        # Scorer les pratiques selon la pertinence
        scored_practices = []
        
        for practice in self.best_practices.values():
            score = 0.0
            
            # Score par industrie
            if practice.industry == industry:
                score += 0.4
            elif practice.industry == IndustryType.GENERAL:
                score += 0.2
                
            # Score par contexte d'application
            context_keywords = project_type.lower().split()
            for context in practice.application_contexts:
                for keyword in context_keywords:
                    if keyword in context.lower():
                        score += 0.1
                        
            # Score par complexité
            if complexity == 'high' and 'complex' in practice.description.lower():
                score += 0.1
            elif complexity == 'low' and 'simple' in practice.description.lower():
                score += 0.1
                
            # Score par taille équipe
            if team_size > 10 and 'équipe' in practice.description.lower():
                score += 0.05
                
            # Score par taux de succès et confiance
            score += practice.success_rate * 0.2
            score += practice.confidence * 0.15
            
            if score > 0.3:  # Seuil de pertinence
                scored_practices.append((practice, score))
                
        # Trier par score et retourner les meilleures
        scored_practices.sort(key=lambda x: x[1], reverse=True)
        return [practice for practice, _ in scored_practices[:limit]]
        
    def get_relevant_patterns(self,
                            project_context: Dict[str, Any],
                            pattern_types: List[PatternType] = None) -> List[IndustryPattern]:
        """Obtenir les patterns pertinents pour un projet"""
        
        industry = project_context.get('industry', IndustryType.GENERAL)
        current_phase = project_context.get('phase', 'planning')
        risk_indicators = project_context.get('risk_indicators', [])
        
        relevant_patterns = []
        
        for pattern in self.industry_patterns.values():
            # Filtrer par type de pattern si spécifié
            if pattern_types and pattern.pattern_type not in pattern_types:
                continue
                
            # Filtrer par industrie
            if pattern.industry != industry and pattern.industry != IndustryType.GENERAL:
                continue
                
            # Vérifier les déclencheurs
            triggers_match = False
            for trigger in pattern.triggers:
                if trigger.lower() in current_phase.lower():
                    triggers_match = True
                    break
                    
            # Vérifier les indicateurs de risque
            indicators_match = False
            for indicator in pattern.indicators:
                if any(indicator.lower() in risk.lower() for risk in risk_indicators):
                    indicators_match = True
                    break
                    
            if triggers_match or indicators_match:
                relevant_patterns.append(pattern)
                
        # Trier par impact et confiance
        relevant_patterns.sort(
            key=lambda p: (p.impact_score * p.confidence, p.frequency),
            reverse=True
        )
        
        return relevant_patterns
        
    def add_best_practice(self, practice: BestPractice):
        """Ajouter une nouvelle meilleure pratique"""
        self.best_practices[practice.practice_id] = practice
        self._save_best_practice(practice)
        logger.info(f"✅ Nouvelle pratique ajoutée: {practice.title}")
        
    def rate_practice(self, 
                     practice_id: str,
                     rating: int,
                     feedback: str = None,
                     project_id: str = None,
                     context: Dict[str, Any] = None):
        """Noter une pratique utilisée"""
        
        if practice_id not in self.best_practices:
            logger.warning(f"Pratique inconnue: {practice_id}")
            return
            
        # Enregistrer la note
        rating_data = {
            'practice_id': practice_id,
            'project_id': project_id,
            'rating': max(1, min(5, rating)),  # Limiter entre 1-5
            'feedback': feedback,
            'context': json.dumps(context or {}),
            'rated_at': datetime.now()
        }
        
        self.practice_ratings[practice_id].append(rating_data)
        
        # Sauvegarder en DB
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO practice_ratings
                    (practice_id, project_id, rating, feedback, context, rated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    rating_data['practice_id'],
                    rating_data['project_id'],
                    rating_data['rating'],
                    rating_data['feedback'],
                    rating_data['context'],
                    rating_data['rated_at']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde note pratique: {e}")
            
        # Mettre à jour le taux de succès de la pratique
        self._update_practice_success_rate(practice_id)
        
    def _update_practice_success_rate(self, practice_id: str):
        """Mettre à jour le taux de succès d'une pratique"""
        ratings = self.practice_ratings.get(practice_id, [])
        
        if len(ratings) < 3:  # Pas assez de données
            return
            
        # Calculer le nouveau taux basé sur les notes récentes
        recent_ratings = [r['rating'] for r in ratings[-10:]]  # 10 dernières notes
        avg_rating = sum(recent_ratings) / len(recent_ratings)
        
        # Convertir note 1-5 en taux de succès 0.0-1.0
        new_success_rate = (avg_rating - 1) / 4
        
        # Mise à jour progressive (pondération avec ancien taux)
        practice = self.best_practices[practice_id]
        practice.success_rate = (practice.success_rate * 0.7) + (new_success_rate * 0.3)
        
        # Sauvegarder
        self._save_best_practice(practice)
        
        logger.info(f"📊 Taux de succès mis à jour pour {practice.title}: {practice.success_rate:.2f}")
        
    def search_knowledge(self, 
                        query: str,
                        knowledge_types: List[str] = None) -> Dict[str, List[Any]]:
        """Rechercher dans la base de connaissances"""
        
        knowledge_types = knowledge_types or ['practices', 'patterns']
        results = {
            'practices': [],
            'patterns': []
        }
        
        query_lower = query.lower()
        query_terms = re.findall(r'\b\w+\b', query_lower)
        
        # Recherche dans les pratiques
        if 'practices' in knowledge_types:
            for practice in self.best_practices.values():
                score = 0
                searchable_text = (
                    practice.title + " " + practice.description + " " +
                    " ".join(practice.application_contexts) + " " +
                    " ".join(practice.implementation_steps)
                ).lower()
                
                for term in query_terms:
                    if term in searchable_text:
                        score += 1
                        
                if score > 0:
                    results['practices'].append((practice, score))
                    
            results['practices'].sort(key=lambda x: x[1], reverse=True)
            results['practices'] = [p for p, _ in results['practices'][:10]]
            
        # Recherche dans les patterns
        if 'patterns' in knowledge_types:
            for pattern in self.industry_patterns.values():
                score = 0
                searchable_text = (
                    pattern.title + " " + pattern.description + " " +
                    " ".join(pattern.triggers) + " " + " ".join(pattern.indicators)
                ).lower()
                
                for term in query_terms:
                    if term in searchable_text:
                        score += 1
                        
                if score > 0:
                    results['patterns'].append((pattern, score))
                    
            results['patterns'].sort(key=lambda x: x[1], reverse=True)
            results['patterns'] = [p for p, _ in results['patterns'][:10]]
            
        return results
        
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de la base de connaissances"""
        
        practice_stats = {
            'total': len(self.best_practices),
            'by_category': Counter([p.category.value for p in self.best_practices.values()]),
            'by_industry': Counter([p.industry.value for p in self.best_practices.values()]),
            'avg_success_rate': sum(p.success_rate for p in self.best_practices.values()) / len(self.best_practices) if self.best_practices else 0
        }
        
        pattern_stats = {
            'total': len(self.industry_patterns),
            'by_type': Counter([p.pattern_type.value for p in self.industry_patterns.values()]),
            'by_industry': Counter([p.industry.value for p in self.industry_patterns.values()]),
            'avg_impact': sum(p.impact_score for p in self.industry_patterns.values()) / len(self.industry_patterns) if self.industry_patterns else 0
        }
        
        return {
            'practices': practice_stats,
            'patterns': pattern_stats,
            'total_ratings': sum(len(ratings) for ratings in self.practice_ratings.values()),
            'knowledge_base_size': len(self.best_practices) + len(self.industry_patterns)
        }

# Instance globale
global_knowledge_base: Optional[DynamicKnowledgeBase] = None

def get_knowledge_base(db_path: str = "data/knowledge_base.db") -> DynamicKnowledgeBase:
    """Obtenir l'instance globale de la base de connaissances"""
    global global_knowledge_base
    
    if global_knowledge_base is None:
        global_knowledge_base = DynamicKnowledgeBase(db_path)
        
    return global_knowledge_base

def get_industry_patterns(industry: IndustryType = None) -> List[IndustryPattern]:
    """Obtenir les patterns d'une industrie"""
    kb = get_knowledge_base()
    
    if industry:
        return [p for p in kb.industry_patterns.values() if p.industry == industry or p.industry == IndustryType.GENERAL]
    else:
        return list(kb.industry_patterns.values())

def get_best_practices(category: PracticeCategory = None, industry: IndustryType = None) -> List[BestPractice]:
    """Obtenir les meilleures pratiques selon critères"""
    kb = get_knowledge_base()
    practices = list(kb.best_practices.values())
    
    if category:
        practices = [p for p in practices if p.category == category]
        
    if industry:
        practices = [p for p in practices if p.industry == industry or p.industry == IndustryType.GENERAL]
        
    return practices