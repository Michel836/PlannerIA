"""
🔍 Analyseur Prédictif de Risques - PlannerIA
Machine Learning avancé pour détection et prédiction de risques projets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class RiskCategory(Enum):
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    LEGAL = "legal"
    MARKET = "market"
    TEAM = "team"


class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RiskFactor:
    """Facteur de risque identifié"""
    factor_id: str
    name: str
    category: RiskCategory
    description: str
    impact_score: float  # 0-1
    probability: float   # 0-1
    detectability: float # 0-1 (1 = facile à détecter)
    risk_priority_number: float  # impact * probability * (1-detectability)
    

@dataclass
class PredictedRisk:
    """Risque prédit par l'IA"""
    risk_id: str
    name: str
    category: RiskCategory
    predicted_level: RiskLevel
    probability: float
    potential_impact: float
    confidence: float
    triggers: List[str]
    early_warnings: List[str]
    mitigation_strategies: List[str]
    similar_cases: List[Dict[str, Any]]
    predicted_timeline: Optional[timedelta]
    prevention_actions: List[str]


@dataclass
class RiskProfile:
    """Profil de risque complet d'un projet"""
    project_id: str
    overall_risk_score: float
    risk_distribution: Dict[RiskCategory, float]
    predicted_risks: List[PredictedRisk]
    risk_trends: Dict[str, List[float]]
    critical_path_risks: List[str]
    mitigation_effectiveness: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    generated_at: datetime


class AIRiskPredictor:
    """Système IA de prédiction et analyse des risques"""
    
    def __init__(self):
        self.models = self._initialize_ml_models()
        self.risk_patterns = self._load_risk_patterns()
        self.historical_data = self._generate_synthetic_training_data()
        self.feature_encoders = {}
        self.scalers = {}
        self.trained = False
        
        # Entraînement complet pour présentation
        self._train_initial_models()
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialise les modèles ML pour prédiction de risques"""
        
        return {
            # Classification des niveaux de risque
            'risk_classifier': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            
            # Prédiction d'impact financier
            'impact_predictor': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            
            # Prédiction de probabilité d'occurrence
            'probability_predictor': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.15,
                max_depth=6,
                random_state=42
            ),
            
            # Clustering pour identification de patterns
            'pattern_detector': DBSCAN(
                eps=0.5,
                min_samples=3,
                metric='euclidean'
            ),
            
            # Détecteur d'anomalies
            'anomaly_detector': {
                'type': 'isolation_forest',
                'contamination': 0.1
            }
        }
    
    def _load_risk_patterns(self) -> Dict[str, Any]:
        """Charge les patterns de risques connus"""
        
        return {
            # Patterns basés sur les caractéristiques du projet
            'project_patterns': {
                'large_team_communication': {
                    'trigger_conditions': {'team_size': ('>', 10)},
                    'risk_factors': ['Communication overhead', 'Coordination complexity'],
                    'historical_probability': 0.7,
                    'avg_impact': 0.6
                },
                'aggressive_timeline': {
                    'trigger_conditions': {'timeline_pressure': ('>', 0.8)},
                    'risk_factors': ['Quality compromise', 'Team burnout', 'Technical debt'],
                    'historical_probability': 0.8,
                    'avg_impact': 0.8
                },
                'new_technology_stack': {
                    'trigger_conditions': {'new_tech_ratio': ('>', 0.5)},
                    'risk_factors': ['Learning curve', 'Integration issues', 'Support limitations'],
                    'historical_probability': 0.6,
                    'avg_impact': 0.7
                },
                'distributed_team': {
                    'trigger_conditions': {'geographic_distribution': ('>', 3)},
                    'risk_factors': ['Time zone conflicts', 'Cultural differences', 'Communication delays'],
                    'historical_probability': 0.5,
                    'avg_impact': 0.5
                }
            },
            
            # Patterns temporels
            'temporal_patterns': {
                'end_of_sprint_rush': {
                    'timing': 'sprint_end',
                    'risk_increase': 0.3,
                    'affected_categories': [RiskCategory.TECHNICAL, RiskCategory.OPERATIONAL]
                },
                'holiday_periods': {
                    'timing': 'holidays',
                    'risk_increase': 0.4,
                    'affected_categories': [RiskCategory.TEAM, RiskCategory.OPERATIONAL]
                },
                'budget_review_periods': {
                    'timing': 'budget_review',
                    'risk_increase': 0.2,
                    'affected_categories': [RiskCategory.FINANCIAL, RiskCategory.STRATEGIC]
                }
            },
            
            # Corrélations entre risques
            'risk_correlations': {
                ('technical_debt', 'quality_issues'): 0.8,
                ('budget_overrun', 'timeline_delay'): 0.7,
                ('team_turnover', 'knowledge_loss'): 0.9,
                ('scope_creep', 'budget_overrun'): 0.6,
                ('poor_communication', 'requirement_changes'): 0.7
            },
            
            # Facteurs d'amplification
            'amplification_factors': {
                'domain_complexity': {
                    'fintech': 1.5,
                    'healthcare': 1.4,
                    'aerospace': 1.6,
                    'ecommerce': 1.1,
                    'web_app': 1.0
                },
                'team_experience': {
                    'junior': 1.4,
                    'mixed': 1.2,
                    'senior': 1.0,
                    'expert': 0.8
                }
            }
        }
    
    def _generate_synthetic_training_data(self) -> pd.DataFrame:
        """Génère des données d'entraînement synthétiques basées sur des projets réels"""
        
        np.random.seed(42)
        n_projects = 1000
        
        # Caractéristiques des projets
        data = []
        
        for i in range(n_projects):
            # Caractéristiques de base
            team_size = np.random.randint(3, 20)
            project_duration = np.random.randint(30, 365)
            budget = np.random.uniform(10000, 500000)
            complexity = np.random.choice(['low', 'medium', 'high', 'very_high'])
            domain = np.random.choice(['web_app', 'mobile', 'enterprise', 'fintech', 'ecommerce'])
            team_experience = np.random.choice(['junior', 'mixed', 'senior', 'expert'])
            
            # Variables dérivées
            team_size_factor = min(2.0, team_size / 10)
            timeline_pressure = np.random.uniform(0.2, 1.0)
            requirements_clarity = np.random.uniform(0.3, 1.0)
            stakeholder_engagement = np.random.uniform(0.2, 0.9)
            
            # Génération des risques basée sur les caractéristiques
            technical_risk = self._calculate_synthetic_risk(
                complexity, team_experience, timeline_pressure, risk_type='technical'
            )
            financial_risk = self._calculate_synthetic_risk(
                budget/100000, team_size_factor, timeline_pressure, risk_type='financial'
            )
            operational_risk = self._calculate_synthetic_risk(
                team_size_factor, requirements_clarity, stakeholder_engagement, risk_type='operational'
            )
            team_risk = self._calculate_synthetic_risk(
                team_size_factor, team_experience, timeline_pressure, risk_type='team'
            )
            
            # Classification du niveau de risque global
            overall_risk = np.mean([technical_risk, financial_risk, operational_risk, team_risk])
            if overall_risk < 0.3:
                risk_level = 1  # LOW
            elif overall_risk < 0.6:
                risk_level = 2  # MEDIUM
            elif overall_risk < 0.8:
                risk_level = 3  # HIGH
            else:
                risk_level = 4  # CRITICAL
            
            # Simulation d'impact réel (pour l'entraînement)
            actual_impact = overall_risk + np.random.normal(0, 0.1)
            actual_impact = np.clip(actual_impact, 0, 1)
            
            data.append({
                'team_size': team_size,
                'project_duration': project_duration,
                'budget': budget,
                'complexity': complexity,
                'domain': domain,
                'team_experience': team_experience,
                'timeline_pressure': timeline_pressure,
                'requirements_clarity': requirements_clarity,
                'stakeholder_engagement': stakeholder_engagement,
                'technical_risk': technical_risk,
                'financial_risk': financial_risk,
                'operational_risk': operational_risk,
                'team_risk': team_risk,
                'overall_risk_score': overall_risk,
                'risk_level': risk_level,
                'actual_impact': actual_impact
            })
        
        return pd.DataFrame(data)
    
    def _calculate_synthetic_risk(self, *factors, risk_type: str) -> float:
        """Calcule un score de risque synthétique"""
        
        # Conversion des facteurs catégoriels
        processed_factors = []
        for factor in factors:
            if isinstance(factor, str):
                if factor in ['low', 'junior']:
                    processed_factors.append(0.3)
                elif factor in ['medium', 'mixed']:
                    processed_factors.append(0.6)
                elif factor in ['high', 'senior']:
                    processed_factors.append(0.8)
                else:  # very_high, expert
                    processed_factors.append(1.0)
            else:
                processed_factors.append(float(factor))
        
        # Calcul pondéré selon le type de risque
        weights = {
            'technical': [0.4, 0.3, 0.3],
            'financial': [0.5, 0.3, 0.2],
            'operational': [0.3, 0.4, 0.3],
            'team': [0.3, 0.4, 0.3]
        }
        
        risk_weights = weights.get(risk_type, [1/len(processed_factors)] * len(processed_factors))
        
        # Assurer la compatibilité des tailles
        min_len = min(len(processed_factors), len(risk_weights))
        risk_score = np.average(processed_factors[:min_len], weights=risk_weights[:min_len])
        
        # Ajout de bruit réaliste
        risk_score += np.random.normal(0, 0.05)
        return np.clip(risk_score, 0, 1)
    
    def _train_initial_models(self):
        """Entraîne les modèles initiaux avec les données synthétiques"""
        
        df = self.historical_data
        
        # Préparation des features
        feature_columns = [
            'team_size', 'project_duration', 'budget', 'timeline_pressure',
            'requirements_clarity', 'stakeholder_engagement'
        ]
        
        # Encodage des variables catégorielles
        categorical_columns = ['complexity', 'domain', 'team_experience']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.feature_encoders[col] = le
            feature_columns.append(f'{col}_encoded')
        
        # Normalisation des features
        X = df[feature_columns].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        # Entraînement du classificateur de niveau de risque
        y_risk_level = df['risk_level'].values
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_risk_level, test_size=0.2, random_state=42)
        
        self.models['risk_classifier'].fit(X_train, y_train)
        
        # Entraînement du prédicteur d'impact
        y_impact = df['actual_impact'].values
        X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_scaled, y_impact, test_size=0.2, random_state=42)
        
        self.models['impact_predictor'].fit(X_train_imp, y_train_imp)
        
        # Entraînement du prédicteur de probabilité
        y_prob = df['overall_risk_score'].values
        X_train_prob, X_test_prob, y_train_prob, y_test_prob = train_test_split(X_scaled, y_prob, test_size=0.2, random_state=42)
        self.models['probability_predictor'].fit(X_train_prob, y_train_prob)
        
        self.trained = True
        
        print(f"[OK] Modèles entraînés avec {len(df)} projets synthétiques")
        print(f"[STATS] Accuracy classificateur: {self.models['risk_classifier'].score(X_test, y_test):.3f}")
        print(f"[METRICS] RMSE prédicteur impact: {np.sqrt(mean_squared_error(y_test_imp, self.models['impact_predictor'].predict(X_test_imp))):.3f}")
    
    async def analyze_project_risks(self, project_data: Dict[str, Any]) -> RiskProfile:
        """Analyse complète des risques d'un projet"""
        
        if not self.trained:
            raise Exception("Modèles non entraînés. Appelez _train_initial_models() d'abord.")
        
        # Extraction et préparation des features
        features = self._extract_project_features(project_data)
        
        # Prédictions ML
        ml_predictions = await self._run_ml_risk_predictions(features)
        
        # Analyse des patterns
        pattern_risks = self._analyze_risk_patterns(features)
        
        # Détection d'anomalies
        anomaly_risks = self._detect_risk_anomalies(features)
        
        # Combinaison des analyses
        predicted_risks = self._combine_risk_analyses(ml_predictions, pattern_risks, anomaly_risks)
        
        # Génération du profil complet
        risk_profile = await self._generate_risk_profile(project_data, predicted_risks, features)
        
        return risk_profile
    
    def _extract_project_features(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les features pertinentes du projet pour l'analyse de risques"""
        
        overview = project_data.get('project_overview', {})
        resources = project_data.get('resources', [])
        phases = project_data.get('phases', [])
        tasks = project_data.get('tasks', [])
        
        # Features de base
        features = {
            'team_size': len(resources),
            'project_duration': overview.get('total_duration', 90),
            'budget': overview.get('total_cost', 100000),
            'complexity': project_data.get('complexity', 'medium'),
            'domain': project_data.get('domain', 'web_app'),
            'team_experience': project_data.get('team_experience', 'mixed'),
            'timeline_pressure': project_data.get('timeline_pressure', 0.5),
            'requirements_clarity': project_data.get('requirements_clarity', 0.7),
            'stakeholder_engagement': project_data.get('stakeholder_engagement', 0.6)
        }
        
        # Features dérivées
        features.update({
            'phase_count': len(phases),
            'task_count': len(tasks),
            'avg_phase_duration': np.mean([p.get('duration', 30) for p in phases]) if phases else 30,
            'budget_per_person': features['budget'] / max(1, features['team_size']),
            'task_complexity_avg': np.mean([self._estimate_task_complexity(task) for task in tasks]) if tasks else 0.5
        })
        
        # Features contextuelles
        features.update({
            'has_external_dependencies': len(project_data.get('integrations', [])) > 0,
            'regulatory_requirements': 'fintech' in features['domain'] or 'healthcare' in features['domain'],
            'new_technology_ratio': project_data.get('new_tech_ratio', 0.3),
            'geographic_distribution': project_data.get('geographic_distribution', 1)
        })
        
        return features
    
    def _estimate_task_complexity(self, task: Dict[str, Any]) -> float:
        """Estime la complexité d'une tâche"""
        
        complexity_indicators = {
            'integration': 0.8,
            'api': 0.7,
            'database': 0.6,
            'security': 0.9,
            'performance': 0.8,
            'ui/ux': 0.5,
            'testing': 0.4
        }
        
        task_description = task.get('description', '').lower()
        complexity_scores = []
        
        for indicator, score in complexity_indicators.items():
            if indicator in task_description:
                complexity_scores.append(score)
        
        return np.mean(complexity_scores) if complexity_scores else 0.5
    
    async def _run_ml_risk_predictions(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute les prédictions ML sur les features"""
        
        # Préparation des features pour les modèles
        feature_vector = self._prepare_feature_vector(features)
        
        # Prédictions
        risk_level_pred = self.models['risk_classifier'].predict([feature_vector])[0]
        risk_level_proba = self.models['risk_classifier'].predict_proba([feature_vector])[0]
        
        impact_pred = self.models['impact_predictor'].predict([feature_vector])[0]
        probability_pred = self.models['probability_predictor'].predict([feature_vector])[0]
        
        # Prédictions par catégorie (simulées - à améliorer avec modèles spécialisés)
        category_predictions = {}
        for category in RiskCategory:
            # Ajustement basé sur les caractéristiques du projet
            base_score = probability_pred
            
            if category == RiskCategory.TECHNICAL:
                base_score *= (1 + features.get('task_complexity_avg', 0.5))
            elif category == RiskCategory.FINANCIAL:
                base_score *= (1 + features.get('timeline_pressure', 0.5))
            elif category == RiskCategory.TEAM:
                team_factor = min(2.0, features.get('team_size', 5) / 10)
                base_score *= (1 + team_factor - 1)
            elif category == RiskCategory.OPERATIONAL:
                base_score *= (1 + (1 - features.get('requirements_clarity', 0.7)))
            
            category_predictions[category] = np.clip(base_score, 0, 1)
        
        return {
            'overall_risk_level': risk_level_pred,
            'risk_level_probabilities': risk_level_proba,
            'predicted_impact': impact_pred,
            'predicted_probability': probability_pred,
            'category_predictions': category_predictions,
            'confidence': np.max(risk_level_proba)  # Confiance = probabilité max
        }
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prépare le vecteur de features pour les modèles ML"""
        
        # Features numériques
        numeric_features = [
            features.get('team_size', 5),
            features.get('project_duration', 90),
            features.get('budget', 100000),
            features.get('timeline_pressure', 0.5),
            features.get('requirements_clarity', 0.7),
            features.get('stakeholder_engagement', 0.6)
        ]
        
        # Encodage des features catégorielles
        categorical_features = []
        for cat_feature in ['complexity', 'domain', 'team_experience']:
            if cat_feature in self.feature_encoders:
                try:
                    encoded_value = self.feature_encoders[cat_feature].transform([features.get(cat_feature, 'medium')])[0]
                    categorical_features.append(encoded_value)
                except ValueError:
                    # Valeur inconnue, utiliser la valeur moyenne
                    categorical_features.append(0)
        
        # Combinaison
        feature_vector = np.array(numeric_features + categorical_features)
        
        # Normalisation
        if 'features' in self.scalers:
            feature_vector = self.scalers['features'].transform([feature_vector])[0]
        
        return feature_vector
    
    def _analyze_risk_patterns(self, features: Dict[str, Any]) -> List[PredictedRisk]:
        """Analyse les risques basés sur les patterns connus"""
        
        pattern_risks = []
        project_patterns = self.risk_patterns['project_patterns']
        
        for pattern_name, pattern_info in project_patterns.items():
            # Vérification des conditions de déclenchement
            triggers_met = []
            
            for condition_key, (operator, threshold) in pattern_info['trigger_conditions'].items():
                feature_value = features.get(condition_key, 0)
                
                if operator == '>' and feature_value > threshold:
                    triggers_met.append(f"{condition_key} > {threshold}")
                elif operator == '<' and feature_value < threshold:
                    triggers_met.append(f"{condition_key} < {threshold}")
                elif operator == '=' and feature_value == threshold:
                    triggers_met.append(f"{condition_key} = {threshold}")
            
            # Si des triggers sont activés, créer le risque prédit
            if triggers_met:
                risk_id = f"pattern_{pattern_name}_{datetime.now().timestamp()}"
                
                # Calcul du niveau de risque basé sur la probabilité historique
                historical_prob = pattern_info['historical_probability']
                avg_impact = pattern_info['avg_impact']
                
                if historical_prob * avg_impact > 0.7:
                    risk_level = RiskLevel.CRITICAL
                elif historical_prob * avg_impact > 0.5:
                    risk_level = RiskLevel.HIGH
                elif historical_prob * avg_impact > 0.3:
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.LOW
                
                # Détermination de la catégorie principale
                category = self._determine_risk_category(pattern_info['risk_factors'])
                
                # Génération des stratégies de mitigation
                mitigation_strategies = self._generate_mitigation_strategies(pattern_name, pattern_info)
                
                pattern_risk = PredictedRisk(
                    risk_id=risk_id,
                    name=pattern_name.replace('_', ' ').title(),
                    category=category,
                    predicted_level=risk_level,
                    probability=historical_prob,
                    potential_impact=avg_impact,
                    confidence=0.8,  # Haute confiance pour les patterns établis
                    triggers=triggers_met,
                    early_warnings=self._generate_early_warnings(pattern_name),
                    mitigation_strategies=mitigation_strategies,
                    similar_cases=[],  # À implémenter avec vraie base de données
                    predicted_timeline=None,
                    prevention_actions=self._generate_prevention_actions(pattern_name)
                )
                
                pattern_risks.append(pattern_risk)
        
        return pattern_risks
    
    def _determine_risk_category(self, risk_factors: List[str]) -> RiskCategory:
        """Détermine la catégorie principale d'un risque basé sur ses facteurs"""
        
        category_keywords = {
            RiskCategory.TECHNICAL: ['technical', 'integration', 'code', 'architecture', 'performance'],
            RiskCategory.TEAM: ['team', 'communication', 'coordination', 'turnover', 'skill'],
            RiskCategory.FINANCIAL: ['budget', 'cost', 'funding', 'financial'],
            RiskCategory.OPERATIONAL: ['process', 'workflow', 'operational', 'resource'],
            RiskCategory.STRATEGIC: ['strategic', 'business', 'market', 'competition']
        }
        
        category_scores = {}
        
        for category, keywords in category_keywords.items():
            score = 0
            for factor in risk_factors:
                factor_lower = factor.lower()
                for keyword in keywords:
                    if keyword in factor_lower:
                        score += 1
            category_scores[category] = score
        
        # Retourner la catégorie avec le score le plus élevé
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return RiskCategory.OPERATIONAL  # Défaut
    
    def _generate_mitigation_strategies(self, pattern_name: str, pattern_info: Dict[str, Any]) -> List[str]:
        """Génère des stratégies de mitigation spécifiques"""
        
        mitigation_templates = {
            'large_team_communication': [
                "Mettre en place des équipes autonomes de 5-7 personnes maximum",
                "Utiliser des outils de communication asynchrone (Slack, Teams)",
                "Organiser des daily standups courts et focalisés",
                "Créer une documentation partagée et maintenue à jour"
            ],
            'aggressive_timeline': [
                "Négocier un scope réduit avec les fonctionnalités critiques seulement",
                "Implémenter une approche MVP (Minimum Viable Product)",
                "Augmenter temporairement les ressources sur les tâches critiques",
                "Mettre en place des processus de développement parallèle"
            ],
            'new_technology_stack': [
                "Prévoir du temps pour formation et montée en compétences",
                "Créer des POCs (Proof of Concepts) pour valider l'approche",
                "Mettre en place un mentorat avec des experts externes",
                "Avoir un plan de fallback vers des technologies connues"
            ],
            'distributed_team': [
                "Définir des créneaux de chevauchement obligatoires",
                "Utiliser des outils de collaboration temps réel",
                "Mettre en place des processus de handover structurés",
                "Organiser des réunions d'équipe régulières par vidéoconférence"
            ]
        }
        
        return mitigation_templates.get(pattern_name, [
            "Surveiller de près l'évolution de cette situation",
            "Mettre en place des métriques de suivi spécifiques",
            "Préparer un plan de contingence",
            "Communiquer régulièrement avec les stakeholders"
        ])
    
    def _generate_early_warnings(self, pattern_name: str) -> List[str]:
        """Génère les signaux d'alerte précoce"""
        
        warning_templates = {
            'large_team_communication': [
                "Augmentation du temps passé en réunions",
                "Baisse de la vélocité des sprints",
                "Augmentation des conflits de merge",
                "Retards dans les livrables inter-équipes"
            ],
            'aggressive_timeline': [
                "Accumulation de dette technique",
                "Augmentation du nombre de bugs",
                "Baisse de la couverture de tests",
                "Signaux de stress dans l'équipe"
            ],
            'new_technology_stack': [
                "Temps de développement plus long que prévu",
                "Multiplication des questions techniques",
                "Difficultés d'intégration",
                "Besoin fréquent de support externe"
            ],
            'distributed_team': [
                "Retards dans la communication d'informations critiques",
                "Duplication de travail entre équipes",
                "Problèmes de cohérence dans le code",
                "Frustration exprimée par les membres de l'équipe"
            ]
        }
        
        return warning_templates.get(pattern_name, [
            "Dérive des métriques clés du projet",
            "Feedback négatif des stakeholders",
            "Augmentation des incidents"
        ])
    
    def _generate_prevention_actions(self, pattern_name: str) -> List[str]:
        """Génère les actions de prévention"""
        
        prevention_templates = {
            'large_team_communication': [
                "Organiser des ateliers de team building",
                "Définir clairement les rôles et responsabilités",
                "Mettre en place une matrice RACI",
                "Former les leads aux techniques de facilitation"
            ],
            'aggressive_timeline': [
                "Valider le scope avec toutes les parties prenantes",
                "Établir des jalons intermédiaires avec validation",
                "Prévoir des buffers de temps pour les imprévus",
                "Mettre en place un monitoring de la vélocité"
            ],
            'new_technology_stack': [
                "Faire un audit des compétences disponibles",
                "Organiser des sessions de formation préventives",
                "Créer une veille technologique active",
                "Établir des partenariats avec des experts"
            ],
            'distributed_team': [
                "Définir des standards de communication",
                "Mettre en place des outils collaboratifs performants",
                "Organiser des rencontres physiques régulières",
                "Créer une culture d'équipe forte"
            ]
        }
        
        return prevention_templates.get(pattern_name, [
            "Effectuer des audits préventifs réguliers",
            "Maintenir une communication proactive",
            "Surveiller les indicateurs de performance"
        ])
    
    def _detect_risk_anomalies(self, features: Dict[str, Any]) -> List[PredictedRisk]:
        """Détecte les anomalies qui pourraient indiquer des risques cachés"""
        
        anomaly_risks = []
        
        # Détection d'anomalies dans les ratios
        budget_per_person = features.get('budget_per_person', 20000)
        if budget_per_person < 10000:
            anomaly_risks.append(self._create_anomaly_risk(
                "Budget per person trop faible",
                RiskCategory.FINANCIAL,
                f"Budget par personne: {budget_per_person:,.0f}€ (recommandé: >20k€)",
                0.7
            ))
        
        timeline_pressure = features.get('timeline_pressure', 0.5)
        team_size = features.get('team_size', 5)
        if timeline_pressure > 0.8 and team_size < 5:
            anomaly_risks.append(self._create_anomaly_risk(
                "Équipe insuffisante pour timeline agressive",
                RiskCategory.OPERATIONAL,
                f"Pression temporelle {timeline_pressure:.1%} avec seulement {team_size} personnes",
                0.8
            ))
        
        # Anomalie de complexité vs expérience
        complexity = features.get('complexity', 'medium')
        team_experience = features.get('team_experience', 'mixed')
        
        high_complexity = complexity in ['high', 'very_high']
        low_experience = team_experience in ['junior', 'mixed']
        
        if high_complexity and low_experience:
            anomaly_risks.append(self._create_anomaly_risk(
                "Complexité élevée avec équipe inexpérimentée",
                RiskCategory.TEAM,
                f"Projet {complexity} avec équipe {team_experience}",
                0.9
            ))
        
        return anomaly_risks
    
    def _create_anomaly_risk(self, name: str, category: RiskCategory, 
                           description: str, severity: float) -> PredictedRisk:
        """Crée un risque détecté par anomalie"""
        
        if severity > 0.8:
            risk_level = RiskLevel.HIGH
        elif severity > 0.6:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return PredictedRisk(
            risk_id=f"anomaly_{datetime.now().timestamp()}",
            name=name,
            category=category,
            predicted_level=risk_level,
            probability=severity,
            potential_impact=severity,
            confidence=0.6,  # Moyenne confiance pour anomalies
            triggers=[f"Détection automatique: {description}"],
            early_warnings=["Métriques inhabituelles détectées"],
            mitigation_strategies=[
                "Analyser les causes de cette anomalie",
                "Ajuster les paramètres du projet si nécessaire",
                "Surveiller de près l'évolution"
            ],
            similar_cases=[],
            predicted_timeline=None,
            prevention_actions=["Validation des métriques de projet"]
        )
    
    def _combine_risk_analyses(self, ml_predictions: Dict[str, Any], 
                             pattern_risks: List[PredictedRisk],
                             anomaly_risks: List[PredictedRisk]) -> List[PredictedRisk]:
        """Combine les différentes analyses de risque"""
        
        combined_risks = pattern_risks + anomaly_risks
        
        # Ajout des risques génériques basés sur les prédictions ML
        category_predictions = ml_predictions['category_predictions']
        
        for category, prediction_score in category_predictions.items():
            if prediction_score > 0.5:  # Seuil de pertinence
                
                if prediction_score > 0.8:
                    level = RiskLevel.HIGH
                elif prediction_score > 0.6:
                    level = RiskLevel.MEDIUM
                else:
                    level = RiskLevel.LOW
                
                generic_risk = PredictedRisk(
                    risk_id=f"ml_{category.value}_{datetime.now().timestamp()}",
                    name=f"Risque {category.value.replace('_', ' ').title()}",
                    category=category,
                    predicted_level=level,
                    probability=prediction_score,
                    potential_impact=ml_predictions['predicted_impact'],
                    confidence=ml_predictions['confidence'],
                    triggers=["Prédiction basée sur modèle ML"],
                    early_warnings=self._get_generic_warnings(category),
                    mitigation_strategies=self._get_generic_mitigations(category),
                    similar_cases=[],
                    predicted_timeline=None,
                    prevention_actions=self._get_generic_preventions(category)
                )
                
                combined_risks.append(generic_risk)
        
        # Déduplication et tri par priorité
        unique_risks = self._deduplicate_risks(combined_risks)
        sorted_risks = sorted(unique_risks, key=lambda r: r.probability * r.potential_impact, reverse=True)
        
        return sorted_risks[:20]  # Limite à 20 risques principaux
    
    def _get_generic_warnings(self, category: RiskCategory) -> List[str]:
        """Retourne les signaux d'alerte génériques par catégorie"""
        
        warnings_map = {
            RiskCategory.TECHNICAL: [
                "Augmentation du temps de build",
                "Multiplication des bugs",
                "Difficultés d'intégration"
            ],
            RiskCategory.FINANCIAL: [
                "Dépassement budgétaire partiel",
                "Coûts cachés émergents",
                "Demandes de ressources supplémentaires"
            ],
            RiskCategory.TEAM: [
                "Baisse de moral de l'équipe",
                "Turnover en augmentation",
                "Conflits interpersonnels"
            ],
            RiskCategory.OPERATIONAL: [
                "Processus inefficaces",
                "Goulots d'étranglement récurrents",
                "Retards dans les livrables"
            ]
        }
        
        return warnings_map.get(category, ["Signaux d'alerte à surveiller"])
    
    def _get_generic_mitigations(self, category: RiskCategory) -> List[str]:
        """Retourne les mitigations génériques par catégorie"""
        
        mitigations_map = {
            RiskCategory.TECHNICAL: [
                "Renforcer les revues de code",
                "Améliorer la couverture de tests",
                "Mettre en place une intégration continue robuste"
            ],
            RiskCategory.FINANCIAL: [
                "Surveiller le budget hebdomadairement",
                "Négocier avec les fournisseurs",
                "Optimiser l'allocation des ressources"
            ],
            RiskCategory.TEAM: [
                "Organiser des sessions de team building",
                "Améliorer la communication interne",
                "Mettre en place un système de feedback"
            ],
            RiskCategory.OPERATIONAL: [
                "Optimiser les processus existants",
                "Automatiser les tâches répétitives",
                "Améliorer la coordination inter-équipes"
            ]
        }
        
        return mitigations_map.get(category, ["Stratégies de mitigation à définir"])
    
    def _get_generic_preventions(self, category: RiskCategory) -> List[str]:
        """Retourne les actions préventives génériques par catégorie"""
        
        preventions_map = {
            RiskCategory.TECHNICAL: [
                "Effectuer des audits techniques réguliers",
                "Maintenir une documentation technique à jour",
                "Organiser des formations techniques"
            ],
            RiskCategory.FINANCIAL: [
                "Établir un suivi budgétaire rigoureux",
                "Prévoir des contingences financières",
                "Valider tous les coûts avec la direction"
            ],
            RiskCategory.TEAM: [
                "Maintenir une communication ouverte",
                "Organiser des one-to-one réguliers",
                "Surveiller la charge de travail"
            ],
            RiskCategory.OPERATIONAL: [
                "Documenter tous les processus",
                "Effectuer des rétrospectives régulières",
                "Optimiser continuellement les workflows"
            ]
        }
        
        return preventions_map.get(category, ["Actions préventives à définir"])
    
    def _deduplicate_risks(self, risks: List[PredictedRisk]) -> List[PredictedRisk]:
        """Supprime les risques en doublon"""
        
        seen_signatures = set()
        unique_risks = []
        
        for risk in risks:
            # Signature basée sur nom + catégorie
            signature = f"{risk.name.lower()}_{risk.category.value}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_risks.append(risk)
            else:
                # Si risque similaire existe, garder celui avec la plus haute probabilité
                for i, existing_risk in enumerate(unique_risks):
                    existing_signature = f"{existing_risk.name.lower()}_{existing_risk.category.value}"
                    if existing_signature == signature:
                        if risk.probability > existing_risk.probability:
                            unique_risks[i] = risk
                        break
        
        return unique_risks
    
    async def _generate_risk_profile(self, project_data: Dict[str, Any], 
                                   predicted_risks: List[PredictedRisk],
                                   features: Dict[str, Any]) -> RiskProfile:
        """Génère le profil de risque complet"""
        
        # Score de risque global
        if predicted_risks:
            risk_scores = [r.probability * r.potential_impact for r in predicted_risks]
            overall_risk_score = np.mean(risk_scores)
        else:
            overall_risk_score = 0.3  # Risque de base
        
        # Distribution par catégorie
        risk_distribution = {}
        for category in RiskCategory:
            category_risks = [r for r in predicted_risks if r.category == category]
            if category_risks:
                avg_score = np.mean([r.probability * r.potential_impact for r in category_risks])
                risk_distribution[category] = avg_score
            else:
                risk_distribution[category] = 0.0
        
        # Tendances de risque (simulées - à améliorer avec données historiques)
        risk_trends = {}
        for category in RiskCategory:
            # Simulation d'évolution sur 12 semaines
            base_value = risk_distribution[category]
            trend = [base_value + np.random.normal(0, 0.05) for _ in range(12)]
            risk_trends[category.value] = [max(0, min(1, value)) for value in trend]
        
        # Risques sur le chemin critique (simplifiés)
        critical_path_risks = [
            risk.name for risk in predicted_risks[:5]
            if risk.predicted_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        
        # Efficacité des mitigations (simulée)
        mitigation_effectiveness = {}
        for risk in predicted_risks[:10]:
            effectiveness = np.random.uniform(0.6, 0.9)  # À remplacer par données réelles
            mitigation_effectiveness[risk.name] = effectiveness
        
        # Intervalles de confiance
        confidence_intervals = {}
        for category in RiskCategory:
            base_score = risk_distribution[category]
            margin = 0.1 * (1 - features.get('requirements_clarity', 0.7))  # Plus d'incertitude si requirements flous
            confidence_intervals[category.value] = (
                max(0, base_score - margin),
                min(1, base_score + margin)
            )
        
        return RiskProfile(
            project_id=project_data.get('project_id', 'unknown'),
            overall_risk_score=overall_risk_score,
            risk_distribution=risk_distribution,
            predicted_risks=predicted_risks,
            risk_trends=risk_trends,
            critical_path_risks=critical_path_risks,
            mitigation_effectiveness=mitigation_effectiveness,
            confidence_intervals=confidence_intervals,
            generated_at=datetime.now()
        )
    
    def update_model_with_feedback(self, project_id: str, actual_risks: List[Dict[str, Any]]):
        """Met à jour les modèles avec les retours d'expérience réels"""
        
        # Cette méthode permettrait d'améliorer les modèles avec les données réelles
        # À implémenter avec un système de feedback structuré
        pass
    
    def export_risk_report(self, risk_profile: RiskProfile) -> Dict[str, Any]:
        """Exporte un rapport de risques détaillé"""
        
        report = {
            'executive_summary': {
                'project_id': risk_profile.project_id,
                'overall_risk_level': self._get_risk_level_text(risk_profile.overall_risk_score),
                'total_risks_identified': len(risk_profile.predicted_risks),
                'critical_risks_count': len([r for r in risk_profile.predicted_risks if r.predicted_level == RiskLevel.CRITICAL]),
                'generated_at': risk_profile.generated_at.isoformat()
            },
            'risk_breakdown': {
                category.value: {
                    'score': score,
                    'level': self._get_risk_level_text(score),
                    'confidence_interval': risk_profile.confidence_intervals.get(category.value, (0, 1))
                }
                for category, score in risk_profile.risk_distribution.items()
            },
            'top_risks': [
                {
                    'name': risk.name,
                    'category': risk.category.value,
                    'level': risk.predicted_level.name,
                    'probability': risk.probability,
                    'impact': risk.potential_impact,
                    'confidence': risk.confidence,
                    'mitigation_strategies': risk.mitigation_strategies
                }
                for risk in risk_profile.predicted_risks[:10]
            ],
            'recommendations': self._generate_strategic_recommendations(risk_profile),
            'action_plan': self._generate_action_plan(risk_profile)
        }
        
        return report
    
    def _get_risk_level_text(self, risk_score: float) -> str:
        """Convertit un score de risque en texte descriptif"""
        
        if risk_score >= 0.8:
            return "CRITIQUE"
        elif risk_score >= 0.6:
            return "ÉLEVÉ"
        elif risk_score >= 0.4:
            return "MODÉRÉ"
        elif risk_score >= 0.2:
            return "FAIBLE"
        else:
            return "TRÈS FAIBLE"
    
    def _generate_strategic_recommendations(self, risk_profile: RiskProfile) -> List[str]:
        """Génère des recommandations stratégiques basées sur le profil de risque"""
        
        recommendations = []
        
        # Recommandations basées sur le score global
        if risk_profile.overall_risk_score > 0.7:
            recommendations.append("🚨 URGENT: Revoir immédiatement la faisabilité du projet")
            recommendations.append("📋 Considérer une réduction du scope pour limiter l'exposition aux risques")
            
        # Recommandations par catégorie dominante
        max_risk_category = max(risk_profile.risk_distribution.items(), key=lambda x: x[1])
        
        if max_risk_category[1] > 0.6:
            category_recommendations = {
                RiskCategory.TECHNICAL: "🔧 Renforcer l'expertise technique et prévoir plus de temps pour R&D",
                RiskCategory.FINANCIAL: "💰 Réviser le budget et négocier des marges de sécurité",
                RiskCategory.TEAM: "👥 Investir dans la cohésion d'équipe et la formation",
                RiskCategory.OPERATIONAL: "⚙️ Optimiser les processus et améliorer la coordination"
            }
            
            rec = category_recommendations.get(max_risk_category[0])
            if rec:
                recommendations.append(rec)
        
        # Recommandations sur les risques critiques
        critical_risks = [r for r in risk_profile.predicted_risks if r.predicted_level == RiskLevel.CRITICAL]
        if critical_risks:
            recommendations.append(f"⚠️ Traiter en priorité les {len(critical_risks)} risques critiques identifiés")
        
        return recommendations[:5]  # Top 5 recommandations
    
    def _generate_action_plan(self, risk_profile: RiskProfile) -> List[Dict[str, Any]]:
        """Génère un plan d'action structuré"""
        
        action_plan = []
        
        # Actions immédiates (risques critiques)
        critical_risks = [r for r in risk_profile.predicted_risks if r.predicted_level == RiskLevel.CRITICAL]
        
        for risk in critical_risks[:3]:  # Top 3 risques critiques
            action_plan.append({
                'priority': 'URGENT',
                'action': f"Mitiger le risque: {risk.name}",
                'strategies': risk.mitigation_strategies,
                'timeline': '1-2 semaines',
                'responsible': 'Chef de projet + équipe concernée'
            })
        
        # Actions à court terme (risques élevés)
        high_risks = [r for r in risk_profile.predicted_risks if r.predicted_level == RiskLevel.HIGH]
        
        for risk in high_risks[:5]:  # Top 5 risques élevés
            action_plan.append({
                'priority': 'HAUTE',
                'action': f"Prévenir le risque: {risk.name}",
                'strategies': risk.prevention_actions,
                'timeline': '2-4 semaines',
                'responsible': 'Équipe projet'
            })
        
        # Actions de surveillance (tous les autres)
        action_plan.append({
            'priority': 'CONTINUE',
            'action': 'Surveiller l\'évolution des risques identifiés',
            'strategies': ['Monitoring hebdomadaire', 'Mise à jour des métriques', 'Réunions de suivi'],
            'timeline': 'Tout au long du projet',
            'responsible': 'Project Manager'
        })
        
        return action_plan


# Instance globale
ai_risk_predictor = AIRiskPredictor()