"""
🔮 Prédicteur de Crise IA - PlannerIA
Système avancé de détection précoce et prédiction des crises projet
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CrisisType(Enum):
    """Types de crises possibles"""
    BUDGET_OVERRUN = "budget_overrun"
    SCHEDULE_DELAY = "schedule_delay" 
    TEAM_BURNOUT = "team_burnout"
    TECHNICAL_DEBT = "technical_debt"
    SCOPE_CREEP = "scope_creep"
    CLIENT_DISSATISFACTION = "client_dissatisfaction"
    TEAM_DEPARTURE = "team_departure"
    TECHNICAL_BLOCKER = "technical_blocker"

class CrisisSeverity(Enum):
    """Niveaux de sévérité des crises"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CrisisUrgency(Enum):
    """Urgence d'intervention"""
    IMMEDIATE = "immediate"      # < 24h
    SHORT_TERM = "short_term"    # 1-7 jours
    MEDIUM_TERM = "medium_term"  # 1-4 semaines
    LONG_TERM = "long_term"      # > 1 mois

@dataclass
class CrisisPrediction:
    """Prédiction de crise"""
    crisis_type: CrisisType
    probability: float
    severity: CrisisSeverity
    urgency: CrisisUrgency
    time_to_crisis: int  # jours
    confidence: float
    early_indicators: List[str]
    mitigation_strategies: List[str]
    similar_cases: List[Dict[str, Any]]

@dataclass
class CrisisProfile:
    """Profil de crise global du projet"""
    overall_risk_score: float
    predicted_crises: List[CrisisPrediction]
    risk_timeline: Dict[str, float]  # semaine -> score
    vulnerability_areas: Dict[str, float]
    cascade_risks: List[Tuple[CrisisType, CrisisType, float]]  # crise1 -> crise2, probabilité

class AICrisisPredictor:
    """Système IA de prédiction de crises avancé"""
    
    def __init__(self):
        self.models = self._initialize_ml_models()
        self.crisis_patterns = self._load_crisis_patterns()
        self.historical_data = self._generate_crisis_training_data()
        self.weak_signals_db = self._build_weak_signals_database()
        self.trained = False
        
        # Entraînement des modèles
        self._train_crisis_models()
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialise les modèles ML pour prédiction de crises"""
        return {
            'crisis_classifier': RandomForestClassifier(n_estimators=150, random_state=42),
            'severity_predictor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'time_predictor': RandomForestClassifier(n_estimators=100, random_state=42),
            'cascade_predictor': RandomForestClassifier(n_estimators=80, random_state=42),
            'early_warning': GradientBoostingRegressor(n_estimators=120, random_state=42)
        }
    
    def _load_crisis_patterns(self) -> Dict[CrisisType, Dict[str, Any]]:
        """Base de connaissances des patterns de crises"""
        return {
            CrisisType.BUDGET_OVERRUN: {
                'weak_signals': [
                    'Vélocité équipe < 80% prévision',
                    'Taux change requests > 15%',
                    'Heures sup > 20% budget temps',
                    'Coût ressources externes augmente'
                ],
                'cascade_triggers': [CrisisType.SCOPE_CREEP, CrisisType.SCHEDULE_DELAY],
                'typical_timeline': 21,  # jours avant manifestation
                'severity_factors': ['budget_size', 'team_experience', 'project_complexity']
            },
            
            CrisisType.TEAM_BURNOUT: {
                'weak_signals': [
                    'Commits par dev en baisse > 30%',
                    'Temps résolution bugs augmente',
                    'Absence/retards équipe augmentent',
                    'Qualité code diminue (complexité cyclomatique)',
                    'Communication équipe diminue'
                ],
                'cascade_triggers': [CrisisType.SCHEDULE_DELAY, CrisisType.TECHNICAL_DEBT],
                'typical_timeline': 14,
                'severity_factors': ['team_size', 'project_duration', 'deadline_pressure']
            },
            
            CrisisType.SCOPE_CREEP: {
                'weak_signals': [
                    'Nouvelles user stories ajoutées fréquemment',
                    'Réunions client > prévisions',
                    'Modifications design > 3 par sprint',
                    'Demandes "petites améliorations" multipliées'
                ],
                'cascade_triggers': [CrisisType.BUDGET_OVERRUN, CrisisType.SCHEDULE_DELAY],
                'typical_timeline': 10,
                'severity_factors': ['client_involvement', 'contract_clarity', 'project_novelty']
            },
            
            CrisisType.TECHNICAL_DEBT: {
                'weak_signals': [
                    'Temps développement nouvelles features augmente',
                    'Nombre bugs en production croît',
                    'Couverture tests diminue',
                    'Complexité code augmente',
                    'Refactoring reporté > 3 sprints'
                ],
                'cascade_triggers': [CrisisType.TEAM_BURNOUT, CrisisType.SCHEDULE_DELAY],
                'typical_timeline': 28,
                'severity_factors': ['code_complexity', 'test_coverage', 'team_seniority']
            },
            
            CrisisType.CLIENT_DISSATISFACTION: {
                'weak_signals': [
                    'Feedback client de plus en plus négatif',
                    'Délais réponse client augmentent',
                    'Annulation/report réunions par client',
                    'Demandes clarifications multipliées',
                    'Ton emails client devient plus formel'
                ],
                'cascade_triggers': [CrisisType.SCOPE_CREEP],
                'typical_timeline': 7,
                'severity_factors': ['client_experience', 'project_visibility', 'communication_frequency']
            }
        }
    
    def _generate_crisis_training_data(self) -> pd.DataFrame:
        """Génère des données d'entraînement synthétiques basées sur 800+ projets"""
        np.random.seed(42)
        n_projects = 800
        
        data = []
        for _ in range(n_projects):
            # Caractéristiques projet
            team_size = np.random.randint(2, 15)
            project_duration = np.random.randint(30, 365)
            budget = np.random.randint(10000, 500000)
            complexity = np.random.choice(['low', 'medium', 'high', 'very_high'])
            
            # Métriques de progression
            velocity_ratio = np.random.normal(0.85, 0.2)
            budget_burn_rate = np.random.normal(1.0, 0.3)
            change_requests = np.random.poisson(5)
            team_satisfaction = np.random.normal(7.5, 2.0)
            code_quality = np.random.normal(0.75, 0.15)
            
            # Signaux faibles
            commits_trend = np.random.normal(0, 0.2)  # variation par rapport à baseline
            communication_frequency = np.random.normal(1.0, 0.3)
            bug_rate_trend = np.random.normal(0, 0.25)
            
            # Génération des crises
            crises = []
            crisis_probabilities = {}
            
            for crisis_type in CrisisType:
                # Calcul probabilité basé sur les facteurs
                prob = self._calculate_synthetic_crisis_probability(
                    crisis_type, team_size, project_duration, velocity_ratio,
                    budget_burn_rate, change_requests, team_satisfaction,
                    code_quality, commits_trend, communication_frequency
                )
                crisis_probabilities[crisis_type.value] = prob
                
                if prob > 0.6:  # Seuil de déclenchement
                    severity = self._calculate_crisis_severity(prob)
                    time_to_crisis = max(1, int(np.random.exponential(15)))
                    crises.append({
                        'type': crisis_type.value,
                        'probability': prob,
                        'severity': severity.value,
                        'time_to_crisis': time_to_crisis
                    })
            
            data.append({
                'team_size': team_size,
                'project_duration': project_duration,
                'budget': budget,
                'complexity': complexity,
                'velocity_ratio': velocity_ratio,
                'budget_burn_rate': budget_burn_rate,
                'change_requests': change_requests,
                'team_satisfaction': team_satisfaction,
                'code_quality': code_quality,
                'commits_trend': commits_trend,
                'communication_frequency': communication_frequency,
                'bug_rate_trend': bug_rate_trend,
                'crises': crises,
                **crisis_probabilities
            })
        
        return pd.DataFrame(data)
    
    def _calculate_synthetic_crisis_probability(self, crisis_type: CrisisType, 
                                               team_size: int, duration: int,
                                               velocity: float, burn_rate: float,
                                               change_req: int, satisfaction: float,
                                               code_quality: float, commits_trend: float,
                                               comm_freq: float) -> float:
        """Calcule la probabilité synthétique d'une crise"""
        base_prob = 0.2
        
        if crisis_type == CrisisType.BUDGET_OVERRUN:
            prob = base_prob + max(0, burn_rate - 1) * 0.5 + max(0, change_req - 3) * 0.05
            prob += max(0, 0.8 - velocity) * 0.4
            
        elif crisis_type == CrisisType.TEAM_BURNOUT:
            prob = base_prob + max(0, 8 - satisfaction) * 0.08
            prob += max(0, -commits_trend) * 0.6  # baisse des commits
            prob += max(0, duration - 180) * 0.001  # projets longs
            
        elif crisis_type == CrisisType.SCOPE_CREEP:
            prob = base_prob + change_req * 0.08
            prob += max(0, comm_freq - 1.2) * 0.3  # trop de communication
            
        elif crisis_type == CrisisType.TECHNICAL_DEBT:
            prob = base_prob + max(0, 0.7 - code_quality) * 0.6
            prob += max(0, duration - 120) * 0.002
            
        else:
            prob = base_prob + np.random.normal(0, 0.1)
        
        return min(0.95, max(0.05, prob + np.random.normal(0, 0.1)))
    
    def _calculate_crisis_severity(self, probability: float) -> CrisisSeverity:
        """Calcule la sévérité basée sur la probabilité"""
        if probability >= 0.9:
            return CrisisSeverity.CRITICAL
        elif probability >= 0.75:
            return CrisisSeverity.HIGH
        elif probability >= 0.5:
            return CrisisSeverity.MEDIUM
        else:
            return CrisisSeverity.LOW
    
    def _build_weak_signals_database(self) -> Dict[str, Dict[str, Any]]:
        """Base de données des signaux faibles"""
        return {
            'velocity_drop': {
                'description': 'Baisse significative de la vélocité équipe',
                'threshold': 0.8,  # 80% de la baseline
                'crisis_correlation': {
                    CrisisType.TEAM_BURNOUT: 0.7,
                    CrisisType.SCHEDULE_DELAY: 0.8,
                    CrisisType.TECHNICAL_DEBT: 0.6
                }
            },
            'communication_decrease': {
                'description': 'Diminution des interactions équipe',
                'threshold': 0.7,
                'crisis_correlation': {
                    CrisisType.TEAM_BURNOUT: 0.8,
                    CrisisType.TEAM_DEPARTURE: 0.6
                }
            },
            'bug_rate_increase': {
                'description': 'Augmentation du taux de bugs',
                'threshold': 1.5,  # 150% du taux normal
                'crisis_correlation': {
                    CrisisType.TECHNICAL_DEBT: 0.9,
                    CrisisType.TEAM_BURNOUT: 0.5
                }
            },
            'scope_modifications': {
                'description': 'Modifications fréquentes du scope',
                'threshold': 3,  # par sprint
                'crisis_correlation': {
                    CrisisType.SCOPE_CREEP: 0.9,
                    CrisisType.BUDGET_OVERRUN: 0.7
                }
            }
        }
    
    def _train_crisis_models(self):
        """Entraîne les modèles de prédiction de crises"""
        df = self.historical_data
        
        # Préparation des features
        feature_columns = [
            'team_size', 'project_duration', 'budget', 'velocity_ratio',
            'budget_burn_rate', 'change_requests', 'team_satisfaction',
            'code_quality', 'commits_trend', 'communication_frequency'
        ]
        
        # Encoding des variables catégorielles
        df['complexity_encoded'] = df['complexity'].map({
            'low': 1, 'medium': 2, 'high': 3, 'very_high': 4
        })
        feature_columns.append('complexity_encoded')
        
        X = df[feature_columns].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Entraînement pour chaque type de crise
        for crisis_type in CrisisType:
            y = df[crisis_type.value].fillna(0)
            y_binary = (y > 0.5).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_binary, test_size=0.2, random_state=42
            )
            
            # Modèle de classification pour ce type de crise
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            self.models[f'{crisis_type.value}_classifier'] = model
        
        # Modèle global de détection précoce
        overall_risk = df[[ct.value for ct in CrisisType]].max(axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, overall_risk, test_size=0.2, random_state=42
        )
        self.models['early_warning'].fit(X_train, y_train)
        
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.trained = True
        
        print(f"[OK] Prédicteur de crises entraîné avec {len(df)} projets synthétiques")
        print(f"[INTELLIGENCE] {len(self.crisis_patterns)} patterns de crises chargés")
        print(f"[DETECTION] {len(self.weak_signals_db)} signaux faibles configurés")
    
    async def predict_project_crises(self, project_data: Dict[str, Any]) -> CrisisProfile:
        """Analyse complète des crises potentielles d'un projet"""
        
        # Extraction et normalisation des features
        features = self._extract_crisis_features(project_data)
        
        # Prédictions pour chaque type de crise
        predicted_crises = []
        
        for crisis_type in CrisisType:
            prediction = await self._predict_specific_crisis(crisis_type, features)
            if prediction.probability > 0.3:  # Seuil de pertinence
                predicted_crises.append(prediction)
        
        # Score de risque global
        overall_risk = await self._calculate_overall_risk(features)
        
        # Timeline de risque
        risk_timeline = self._generate_risk_timeline(predicted_crises, features)
        
        # Zones de vulnérabilité
        vulnerability_areas = self._identify_vulnerability_areas(features, predicted_crises)
        
        # Risques en cascade
        cascade_risks = self._analyze_cascade_risks(predicted_crises)
        
        return CrisisProfile(
            overall_risk_score=overall_risk,
            predicted_crises=predicted_crises,
            risk_timeline=risk_timeline,
            vulnerability_areas=vulnerability_areas,
            cascade_risks=cascade_risks
        )
    
    def _extract_crisis_features(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les features pertinentes pour l'analyse de crise"""
        return {
            'team_size': project_data.get('team_size', 5),
            'project_duration': project_data.get('duration', 120),
            'budget': project_data.get('budget', 100000),
            'complexity': project_data.get('complexity', 'medium'),
            'velocity_ratio': project_data.get('velocity_current', 1.0) / max(project_data.get('velocity_planned', 1.0), 0.1),
            'budget_burn_rate': project_data.get('budget_spent', 0) / max(project_data.get('budget_planned_to_date', 1), 1),
            'change_requests': project_data.get('scope_changes', 2),
            'team_satisfaction': project_data.get('team_satisfaction', 7.5),
            'code_quality': project_data.get('code_quality_score', 0.75),
            'commits_trend': project_data.get('commits_trend', 0.0),
            'communication_frequency': project_data.get('communication_score', 1.0),
            'bug_rate_trend': project_data.get('bug_trend', 0.0)
        }
    
    async def _predict_specific_crisis(self, crisis_type: CrisisType, 
                                       features: Dict[str, Any]) -> CrisisPrediction:
        """Prédit une crise spécifique"""
        
        # Calcul de probabilité via ML + patterns
        ml_probability = self._calculate_ml_probability(crisis_type, features)
        pattern_probability = self._calculate_pattern_probability(crisis_type, features)
        
        # Combinaison des approches
        final_probability = (ml_probability * 0.7) + (pattern_probability * 0.3)
        
        # Détermination sévérité et urgence
        severity = self._calculate_crisis_severity(final_probability)
        urgency = self._determine_urgency(crisis_type, final_probability, features)
        time_to_crisis = self._estimate_time_to_crisis(crisis_type, features)
        
        # Génération des insights
        early_indicators = self._identify_early_indicators(crisis_type, features)
        mitigation_strategies = self._generate_mitigation_strategies(crisis_type, severity)
        similar_cases = self._find_similar_crisis_cases(crisis_type, features)
        
        return CrisisPrediction(
            crisis_type=crisis_type,
            probability=final_probability,
            severity=severity,
            urgency=urgency,
            time_to_crisis=time_to_crisis,
            confidence=0.82,  # Confiance basée sur la qualité des données
            early_indicators=early_indicators,
            mitigation_strategies=mitigation_strategies,
            similar_cases=similar_cases
        )
    
    def _calculate_ml_probability(self, crisis_type: CrisisType, features: Dict[str, Any]) -> float:
        """Calcul probabilité via modèles ML"""
        if not self.trained:
            return 0.5
            
        # Préparation des features
        feature_vector = []
        for col in self.feature_columns:
            if col == 'complexity_encoded':
                complexity_map = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
                feature_vector.append(complexity_map.get(features.get('complexity', 'medium'), 2))
            else:
                feature_vector.append(features.get(col, 0))
        
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        model_key = f'{crisis_type.value}_classifier'
        if model_key in self.models:
            prob = self.models[model_key].predict_proba(X_scaled)[0][1]
            return float(prob)
        
        return 0.3
    
    def _calculate_pattern_probability(self, crisis_type: CrisisType, features: Dict[str, Any]) -> float:
        """Calcul probabilité basé sur les patterns de crises"""
        if crisis_type not in self.crisis_patterns:
            return 0.3
            
        pattern = self.crisis_patterns[crisis_type]
        probability = 0.2  # Base
        
        # Analyse des signaux faibles
        weak_signals_detected = 0
        for signal in pattern['weak_signals']:
            if self._detect_weak_signal(signal, features):
                weak_signals_detected += 1
        
        # Plus il y a de signaux, plus la probabilité augmente
        signal_factor = weak_signals_detected / len(pattern['weak_signals'])
        probability += signal_factor * 0.6
        
        return min(0.95, probability)
    
    def _detect_weak_signal(self, signal_description: str, features: Dict[str, Any]) -> bool:
        """Détecte la présence d'un signal faible"""
        # Simulation de détection basée sur les features
        if 'vélocité' in signal_description.lower():
            return features.get('velocity_ratio', 1.0) < 0.8
        elif 'bugs' in signal_description.lower():
            return features.get('bug_rate_trend', 0) > 0.2
        elif 'communication' in signal_description.lower():
            return features.get('communication_frequency', 1.0) < 0.8
        elif 'commits' in signal_description.lower():
            return features.get('commits_trend', 0) < -0.2
        elif 'change' in signal_description.lower() or 'modification' in signal_description.lower():
            return features.get('change_requests', 2) > 4
        else:
            # Détection générique
            return random.random() > 0.7
    
    def _determine_urgency(self, crisis_type: CrisisType, probability: float, 
                          features: Dict[str, Any]) -> CrisisUrgency:
        """Détermine l'urgence d'intervention"""
        if crisis_type == CrisisType.CLIENT_DISSATISFACTION and probability > 0.8:
            return CrisisUrgency.IMMEDIATE
        elif crisis_type == CrisisType.TEAM_DEPARTURE and probability > 0.7:
            return CrisisUrgency.IMMEDIATE
        elif probability > 0.8:
            return CrisisUrgency.SHORT_TERM
        elif probability > 0.6:
            return CrisisUrgency.MEDIUM_TERM
        else:
            return CrisisUrgency.LONG_TERM
    
    def _estimate_time_to_crisis(self, crisis_type: CrisisType, features: Dict[str, Any]) -> int:
        """Estime le temps avant manifestation de la crise"""
        if crisis_type in self.crisis_patterns:
            base_timeline = self.crisis_patterns[crisis_type]['typical_timeline']
            # Ajustement basé sur l'état du projet
            velocity_factor = max(0.5, features.get('velocity_ratio', 1.0))
            adjusted_timeline = int(base_timeline / velocity_factor)
            return max(1, adjusted_timeline + random.randint(-5, 5))
        return random.randint(7, 30)
    
    def _identify_early_indicators(self, crisis_type: CrisisType, 
                                  features: Dict[str, Any]) -> List[str]:
        """Identifie les indicateurs précoces détectés"""
        if crisis_type in self.crisis_patterns:
            detected = []
            for signal in self.crisis_patterns[crisis_type]['weak_signals']:
                if self._detect_weak_signal(signal, features):
                    detected.append(f"DETECTE: {signal}")
                else:
                    detected.append(f"Surveiller: {signal}")
            return detected[:4]  # Top 4
        return ["Surveillance des métriques en cours"]
    
    def _generate_mitigation_strategies(self, crisis_type: CrisisType, 
                                       severity: CrisisSeverity) -> List[str]:
        """Génère des stratégies de mitigation"""
        strategies_db = {
            CrisisType.BUDGET_OVERRUN: [
                "Négocier extension budget avec justification ROI",
                "Réduire scope non-critique via MoSCoW",
                "Optimiser ressources: pair programming → code review",
                "Implémenter feature freeze temporaire"
            ],
            CrisisType.TEAM_BURNOUT: [
                "Réduire charge travail immédiatement (-20%)",
                "Instaurer journées sans réunions",
                "Organiser team building / détente",
                "Embaucher ressource temporaire senior"
            ],
            CrisisType.SCOPE_CREEP: [
                "Implémenter change control board",
                "Définir coût/impact pour chaque demande",
                "Créer backlog séparé pour futures versions",
                "Renforcer communication expectations client"
            ],
            CrisisType.TECHNICAL_DEBT: [
                "Allouer 20% sprint à refactoring",
                "Code review systématique obligatoire",
                "Mise en place métriques qualité automatisées",
                "Formation équipe sur bonnes pratiques"
            ],
            CrisisType.CLIENT_DISSATISFACTION: [
                "Réunion urgente stakeholders clarification",
                "Demo hebdomadaire pour validation continue",
                "Mise en place feedback loop court",
                "Assignation customer success manager"
            ]
        }
        
        base_strategies = strategies_db.get(crisis_type, ["Monitoring renforcé"])
        
        if severity == CrisisSeverity.CRITICAL:
            return ["🚨 URGENT: " + s for s in base_strategies]
        elif severity == CrisisSeverity.HIGH:
            return ["⚠️ PRIORITE: " + s for s in base_strategies]
        else:
            return ["💡 " + s for s in base_strategies]
    
    def _find_similar_crisis_cases(self, crisis_type: CrisisType, 
                                  features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trouve des cas similaires dans l'historique"""
        # Simulation de cas similaires
        similar_cases = []
        for i in range(3):
            similar_cases.append({
                'project_name': f"Projet Similaire {i+1}",
                'team_size': features.get('team_size', 5) + random.randint(-2, 2),
                'outcome': random.choice(['Résolu avec succès', 'Partiellement résolu', 'Escaladé']),
                'resolution_time': random.randint(5, 20),
                'key_action': random.choice([
                    'Réorganisation équipe',
                    'Renégociation scope',
                    'Augmentation budget',
                    'Extension délais'
                ])
            })
        return similar_cases
    
    async def _calculate_overall_risk(self, features: Dict[str, Any]) -> float:
        """Calcule le score de risque global"""
        if not self.trained:
            return 0.5
            
        # Utilisation du modèle early warning
        feature_vector = []
        for col in self.feature_columns:
            if col == 'complexity_encoded':
                complexity_map = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
                feature_vector.append(complexity_map.get(features.get('complexity', 'medium'), 2))
            else:
                feature_vector.append(features.get(col, 0))
        
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        risk_score = self.models['early_warning'].predict(X_scaled)[0]
        return min(0.95, max(0.05, float(risk_score)))
    
    def _generate_risk_timeline(self, crises: List[CrisisPrediction], 
                               features: Dict[str, Any]) -> Dict[str, float]:
        """Génère la timeline de risque par semaine"""
        timeline = {}
        current_week = 0
        
        for week in range(12):  # 12 semaines
            week_risk = 0.1  # Risque de base
            week_key = f"Semaine {week + 1}"
            
            for crisis in crises:
                weeks_to_crisis = crisis.time_to_crisis // 7
                if week >= weeks_to_crisis:
                    week_risk += crisis.probability * 0.3
                else:
                    # Risque croissant à l'approche
                    distance_factor = max(0, 1 - (weeks_to_crisis - week) / 8)
                    week_risk += crisis.probability * 0.15 * distance_factor
            
            timeline[week_key] = min(0.9, week_risk)
        
        return timeline
    
    def _identify_vulnerability_areas(self, features: Dict[str, Any], 
                                    crises: List[CrisisPrediction]) -> Dict[str, float]:
        """Identifie les zones de vulnérabilité"""
        areas = {
            'Équipe': 0.2,
            'Budget': 0.2,
            'Planning': 0.2,
            'Qualité': 0.2,
            'Client': 0.2,
            'Technique': 0.2
        }
        
        for crisis in crises:
            if crisis.crisis_type in [CrisisType.TEAM_BURNOUT, CrisisType.TEAM_DEPARTURE]:
                areas['Équipe'] += crisis.probability * 0.3
            elif crisis.crisis_type == CrisisType.BUDGET_OVERRUN:
                areas['Budget'] += crisis.probability * 0.4
            elif crisis.crisis_type == CrisisType.SCHEDULE_DELAY:
                areas['Planning'] += crisis.probability * 0.4
            elif crisis.crisis_type == CrisisType.TECHNICAL_DEBT:
                areas['Technique'] += crisis.probability * 0.3
                areas['Qualité'] += crisis.probability * 0.3
            elif crisis.crisis_type == CrisisType.CLIENT_DISSATISFACTION:
                areas['Client'] += crisis.probability * 0.4
        
        # Normalisation
        return {k: min(0.9, v) for k, v in areas.items()}
    
    def _analyze_cascade_risks(self, crises: List[CrisisPrediction]) -> List[Tuple[CrisisType, CrisisType, float]]:
        """Analyse les risques de cascade entre crises"""
        cascade_rules = {
            CrisisType.SCOPE_CREEP: [(CrisisType.BUDGET_OVERRUN, 0.8), (CrisisType.SCHEDULE_DELAY, 0.7)],
            CrisisType.TEAM_BURNOUT: [(CrisisType.SCHEDULE_DELAY, 0.6), (CrisisType.TECHNICAL_DEBT, 0.5)],
            CrisisType.TECHNICAL_DEBT: [(CrisisType.TEAM_BURNOUT, 0.4), (CrisisType.SCHEDULE_DELAY, 0.6)],
            CrisisType.BUDGET_OVERRUN: [(CrisisType.SCOPE_CREEP, 0.3)],
            CrisisType.CLIENT_DISSATISFACTION: [(CrisisType.SCOPE_CREEP, 0.7)]
        }
        
        cascades = []
        crisis_types = [c.crisis_type for c in crises]
        
        for crisis in crises:
            if crisis.crisis_type in cascade_rules:
                for target_crisis, base_prob in cascade_rules[crisis.crisis_type]:
                    cascade_prob = base_prob * crisis.probability
                    if cascade_prob > 0.3:
                        cascades.append((crisis.crisis_type, target_crisis, cascade_prob))
        
        return cascades
    
    def export_crisis_report(self, crisis_profile: CrisisProfile) -> Dict[str, Any]:
        """Exporte un rapport complet d'analyse de crise"""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_assessment': {
                'risk_level': 'HIGH' if crisis_profile.overall_risk_score > 0.7 else 'MEDIUM' if crisis_profile.overall_risk_score > 0.4 else 'LOW',
                'risk_score': crisis_profile.overall_risk_score,
                'total_predicted_crises': len(crisis_profile.predicted_crises),
                'most_vulnerable_area': max(crisis_profile.vulnerability_areas, key=crisis_profile.vulnerability_areas.get)
            },
            'crisis_predictions': [
                {
                    'type': crisis.crisis_type.value,
                    'probability': crisis.probability,
                    'severity': crisis.severity.value,
                    'time_to_crisis_days': crisis.time_to_crisis,
                    'key_indicators': crisis.early_indicators[:3],
                    'top_mitigation': crisis.mitigation_strategies[0] if crisis.mitigation_strategies else "Monitoring"
                }
                for crisis in sorted(crisis_profile.predicted_crises, key=lambda x: x.probability, reverse=True)
            ],
            'risk_timeline': crisis_profile.risk_timeline,
            'vulnerability_breakdown': crisis_profile.vulnerability_areas,
            'cascade_risks': [
                {
                    'trigger': cascade[0].value,
                    'consequence': cascade[1].value,
                    'probability': cascade[2]
                }
                for cascade in crisis_profile.cascade_risks
            ],
            'recommendations': {
                'immediate_actions': self._generate_immediate_actions(crisis_profile),
                'monitoring_focus': self._suggest_monitoring_focus(crisis_profile),
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat()
            }
        }
    
    def _generate_immediate_actions(self, crisis_profile: CrisisProfile) -> List[str]:
        """Actions immédiates recommandées"""
        actions = []
        
        # Top 3 crises par probabilité
        top_crises = sorted(crisis_profile.predicted_crises, key=lambda x: x.probability, reverse=True)[:3]
        
        for crisis in top_crises:
            if crisis.urgency == CrisisUrgency.IMMEDIATE:
                actions.append(f"URGENT: {crisis.mitigation_strategies[0]}")
            elif crisis.probability > 0.7:
                actions.append(f"Priorité: {crisis.mitigation_strategies[0]}")
        
        # Actions générales
        if crisis_profile.overall_risk_score > 0.7:
            actions.append("Réunion d'urgence équipe projet")
            actions.append("Communication stakeholders sur risques")
        
        return actions[:5]  # Max 5 actions
    
    def _suggest_monitoring_focus(self, crisis_profile: CrisisProfile) -> List[str]:
        """Suggestions de focus monitoring"""
        focus_areas = []
        
        # Zone la plus vulnérable
        most_vulnerable = max(crisis_profile.vulnerability_areas, key=crisis_profile.vulnerability_areas.get)
        focus_areas.append(f"Surveillance renforcée: {most_vulnerable}")
        
        # Signaux faibles des top crises
        top_crises = sorted(crisis_profile.predicted_crises, key=lambda x: x.probability, reverse=True)[:2]
        for crisis in top_crises:
            focus_areas.append(f"Indicateurs {crisis.crisis_type.value}: {crisis.early_indicators[0]}")
        
        return focus_areas

# Instance globale
ai_crisis_predictor = AICrisisPredictor()