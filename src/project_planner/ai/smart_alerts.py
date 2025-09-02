"""
🚨 Système d'Alertes Intelligentes - PlannerIA
Détection précoce des problèmes et alertes contextuelles avec IA
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
from collections import deque
import warnings


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    TIMELINE = "timeline"
    BUDGET = "budget"
    QUALITY = "quality"
    TEAM = "team"
    RISK = "risk"
    PERFORMANCE = "performance"
    BUSINESS = "business"


@dataclass
class Alert:
    """Alerte intelligente"""
    id: str
    level: AlertLevel
    category: AlertCategory
    title: str
    description: str
    recommended_actions: List[str]
    data_points: Dict[str, Any]
    confidence: float
    created_at: datetime
    expires_at: Optional[datetime] = None
    dismissed: bool = False
    auto_generated: bool = True


@dataclass
class AlertRule:
    """Règle d'alerte personnalisable"""
    rule_id: str
    name: str
    category: AlertCategory
    condition: str  # Expression à évaluer
    threshold_values: Dict[str, float]
    alert_level: AlertLevel
    cooldown_minutes: int = 60
    enabled: bool = True


class SmartAlertSystem:
    """Système d'alertes intelligentes avec IA prédictive"""
    
    def __init__(self):
        self.alerts_history = deque(maxlen=1000)
        self.active_alerts = {}
        self.alert_rules = self._initialize_alert_rules()
        self.metrics_history = deque(maxlen=100)  # Historique des métriques
        self.ai_models = self._initialize_ai_models()
        self.alert_patterns = self._load_alert_patterns()
        
    def _initialize_alert_rules(self) -> Dict[str, AlertRule]:
        """Initialise les règles d'alerte par défaut"""
        rules = {
            'budget_overrun': AlertRule(
                rule_id='budget_overrun',
                name='Dépassement budgétaire',
                category=AlertCategory.BUDGET,
                condition='current_cost > budget * 0.9',
                threshold_values={'budget_ratio': 0.9},
                alert_level=AlertLevel.WARNING
            ),
            'timeline_delay': AlertRule(
                rule_id='timeline_delay',
                name='Retard sur planning',
                category=AlertCategory.TIMELINE,
                condition='progress_ratio < expected_progress_ratio * 0.8',
                threshold_values={'progress_threshold': 0.8},
                alert_level=AlertLevel.WARNING
            ),
            'critical_budget': AlertRule(
                rule_id='critical_budget',
                name='Budget critique dépassé',
                category=AlertCategory.BUDGET,
                condition='current_cost > budget * 1.1',
                threshold_values={'budget_ratio': 1.1},
                alert_level=AlertLevel.CRITICAL,
                cooldown_minutes=30
            ),
            'team_velocity_drop': AlertRule(
                rule_id='team_velocity_drop',
                name='Chute de vélocité équipe',
                category=AlertCategory.TEAM,
                condition='current_velocity < avg_velocity * 0.7',
                threshold_values={'velocity_threshold': 0.7},
                alert_level=AlertLevel.WARNING
            ),
            'quality_degradation': AlertRule(
                rule_id='quality_degradation',
                name='Dégradation qualité code',
                category=AlertCategory.QUALITY,
                condition='code_quality_score < 0.7',
                threshold_values={'quality_threshold': 0.7},
                alert_level=AlertLevel.WARNING
            ),
            'high_risk_detected': AlertRule(
                rule_id='high_risk_detected',
                name='Risque élevé détecté',
                category=AlertCategory.RISK,
                condition='risk_score > 0.8',
                threshold_values={'risk_threshold': 0.8},
                alert_level=AlertLevel.ERROR
            ),
            'performance_degradation': AlertRule(
                rule_id='performance_degradation',
                name='Dégradation performance',
                category=AlertCategory.PERFORMANCE,
                condition='response_time > 2000 or error_rate > 0.05',
                threshold_values={'response_time_ms': 2000, 'error_rate': 0.05},
                alert_level=AlertLevel.WARNING
            ),
            'stakeholder_satisfaction': AlertRule(
                rule_id='stakeholder_satisfaction',
                name='Satisfaction stakeholder faible',
                category=AlertCategory.BUSINESS,
                condition='satisfaction_score < 0.6',
                threshold_values={'satisfaction_threshold': 0.6},
                alert_level=AlertLevel.WARNING
            )
        }
        return rules
    
    def _initialize_ai_models(self) -> Dict[str, Any]:
        """Initialise les modèles IA pour prédiction d'alertes"""
        # Modèles simulés pour la démo - à remplacer par de vrais modèles ML
        return {
            'anomaly_detector': {'type': 'isolation_forest', 'threshold': 0.1},
            'trend_predictor': {'type': 'lstm', 'lookback_days': 7},
            'risk_classifier': {'type': 'random_forest', 'confidence_threshold': 0.8}
        }
    
    def _load_alert_patterns(self) -> Dict[str, Any]:
        """Charge les patterns d'alertes historiques"""
        return {
            'cascade_patterns': {
                'budget_timeline': 'Budget overrun souvent suivi de retards',
                'team_quality': 'Problèmes équipe impactent la qualité',
                'performance_user': 'Problèmes perfs créent insatisfaction utilisateur'
            },
            'seasonal_patterns': {
                'monday_deployments': 'Plus d\'erreurs les lundis',
                'end_of_sprint': 'Baisse qualité en fin de sprint',
                'vacation_periods': 'Ralentissement pendant congés'
            },
            'correlation_patterns': {
                'team_size_communication': 'Grandes équipes = + problèmes communication',
                'complexity_bugs': 'Complexité élevée = + bugs',
                'deadline_pressure_quality': 'Pression délais = - qualité'
            }
        }
    
    async def monitor_project_metrics(self, project_data: Dict[str, Any]) -> List[Alert]:
        """Surveille les métriques et génère des alertes"""
        
        # Collecte des métriques actuelles
        current_metrics = self._extract_current_metrics(project_data)
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': current_metrics
        })
        
        # Évaluation des règles d'alerte
        rule_alerts = await self._evaluate_alert_rules(current_metrics)
        
        # Détection d'anomalies par IA
        ai_alerts = await self._detect_ai_anomalies(current_metrics)
        
        # Prédictions d'alertes futures
        predictive_alerts = await self._predict_future_issues(current_metrics)
        
        # Combinaison et déduplication
        all_alerts = rule_alerts + ai_alerts + predictive_alerts
        unique_alerts = self._deduplicate_alerts(all_alerts)
        
        # Mise à jour du système
        self._update_active_alerts(unique_alerts)
        
        return unique_alerts
    
    def _extract_current_metrics(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les métriques actuelles du projet"""
        
        # Métriques de base
        overview = project_data.get('project_overview', {})
        phases = project_data.get('phases', [])
        tasks = project_data.get('tasks', [])
        risks = project_data.get('risks', [])
        
        # Calculs de progression
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.get('status') == 'completed'])
        progress_ratio = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        # Calculs budgétaires
        budget = overview.get('total_cost', 100000)
        spent_budget = sum([p.get('cost', 0) for p in phases])
        budget_ratio = spent_budget / budget if budget > 0 else 0
        
        # Calculs temporels
        project_duration = overview.get('total_duration', 90)
        elapsed_days = 30  # Simulé - à calculer réellement
        time_ratio = elapsed_days / project_duration if project_duration > 0 else 0
        
        expected_progress_ratio = time_ratio  # Progression attendue = temps écoulé
        
        # Calculs d'équipe et vélocité
        team_size = len(project_data.get('resources', []))
        current_velocity = completed_tasks / elapsed_days if elapsed_days > 0 else 0
        
        # Calculs de qualité (simulés)
        code_quality_score = np.random.uniform(0.5, 0.9)  # À remplacer par vraies métriques
        
        # Calculs de risque
        risk_scores = [r.get('risk_score', 0) for r in risks]
        avg_risk_score = np.mean(risk_scores) if risk_scores else 0
        
        # Métriques de performance (simulées)
        response_time = np.random.uniform(500, 3000)
        error_rate = np.random.uniform(0.01, 0.1)
        
        # Satisfaction stakeholder (simulée)
        satisfaction_score = np.random.uniform(0.4, 0.9)
        
        return {
            # Métriques temporelles
            'progress_ratio': progress_ratio,
            'expected_progress_ratio': expected_progress_ratio,
            'time_ratio': time_ratio,
            'elapsed_days': elapsed_days,
            'project_duration': project_duration,
            
            # Métriques budgétaires
            'current_cost': spent_budget,
            'budget': budget,
            'budget_ratio': budget_ratio,
            
            # Métriques équipe
            'team_size': team_size,
            'current_velocity': current_velocity,
            'avg_velocity': current_velocity,  # Simulé
            'completed_tasks': completed_tasks,
            'total_tasks': total_tasks,
            
            # Métriques qualité
            'code_quality_score': code_quality_score,
            
            # Métriques risque
            'risk_score': avg_risk_score,
            
            # Métriques performance
            'response_time': response_time,
            'error_rate': error_rate,
            
            # Métriques business
            'satisfaction_score': satisfaction_score,
            
            # Méta
            'timestamp': datetime.now()
        }
    
    async def _evaluate_alert_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Évalue les règles d'alerte configurées"""
        
        alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
                
            # Vérification cooldown
            if self._is_rule_in_cooldown(rule_id):
                continue
            
            try:
                # Évaluation de la condition
                if self._evaluate_rule_condition(rule.condition, metrics, rule.threshold_values):
                    alert = self._create_rule_alert(rule, metrics)
                    alerts.append(alert)
                    
            except Exception as e:
                # Log de l'erreur d'évaluation de règle
                print(f"Erreur évaluation règle {rule_id}: {e}")
        
        return alerts
    
    def _is_rule_in_cooldown(self, rule_id: str) -> bool:
        """Vérifie si une règle est en cooldown"""
        if rule_id not in self.active_alerts:
            return False
            
        last_alert_time = self.active_alerts[rule_id].created_at
        cooldown_duration = timedelta(minutes=self.alert_rules[rule_id].cooldown_minutes)
        
        return datetime.now() - last_alert_time < cooldown_duration
    
    def _evaluate_rule_condition(self, condition: str, metrics: Dict[str, Any], 
                                thresholds: Dict[str, float]) -> bool:
        """Évalue une condition d'alerte"""
        
        # Construction du contexte d'évaluation
        eval_context = {**metrics, **thresholds}
        
        try:
            # Évaluation sécurisée de l'expression
            result = eval(condition, {"__builtins__": {}}, eval_context)
            return bool(result)
        except:
            return False
    
    def _create_rule_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> Alert:
        """Crée une alerte basée sur une règle"""
        
        # Génération des actions recommandées
        recommended_actions = self._generate_rule_recommendations(rule, metrics)
        
        # Extraction des points de données pertinents
        relevant_data = self._extract_relevant_data(rule.category, metrics)
        
        return Alert(
            id=f"{rule.rule_id}_{datetime.now().timestamp()}",
            level=rule.alert_level,
            category=rule.category,
            title=rule.name,
            description=self._generate_alert_description(rule, metrics),
            recommended_actions=recommended_actions,
            data_points=relevant_data,
            confidence=0.9,  # Haute confiance pour les règles
            created_at=datetime.now(),
            auto_generated=True
        )
    
    def _generate_rule_recommendations(self, rule: AlertRule, metrics: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur la règle déclenchée"""
        
        recommendations = {
            'budget_overrun': [
                "Réviser le scope du projet pour rester dans le budget",
                "Négocier un budget supplémentaire avec les stakeholders",
                "Optimiser l'allocation des ressources sur les tâches critiques"
            ],
            'timeline_delay': [
                "Reprioriser les tâches selon leur valeur business",
                "Augmenter temporairement les ressources sur le projet",
                "Négocier un report de deadline avec le client"
            ],
            'critical_budget': [
                "ARRÊTER immédiatement les dépenses non-critiques",
                "Organiser une réunion d'urgence avec la direction",
                "Évaluer la faisabilité de continuer le projet"
            ],
            'team_velocity_drop': [
                "Identifier les blocages et obstacles de l'équipe",
                "Organiser une rétrospective pour améliorer les processus",
                "Vérifier la charge de travail et répartir si nécessaire"
            ],
            'quality_degradation': [
                "Renforcer les revues de code et tests automatisés",
                "Prévoir du temps pour refactoring technique",
                "Former l'équipe aux meilleures pratiques"
            ],
            'high_risk_detected': [
                "Activer immédiatement les plans de mitigation",
                "Informer les stakeholders des risques identifiés",
                "Revoir et mettre à jour les stratégies de risque"
            ]
        }
        
        return recommendations.get(rule.rule_id, ["Analyser les causes et ajuster la stratégie"])
    
    def _extract_relevant_data(self, category: AlertCategory, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les données pertinentes pour une catégorie d'alerte"""
        
        category_mappings = {
            AlertCategory.BUDGET: ['current_cost', 'budget', 'budget_ratio'],
            AlertCategory.TIMELINE: ['progress_ratio', 'expected_progress_ratio', 'elapsed_days'],
            AlertCategory.TEAM: ['team_size', 'current_velocity', 'completed_tasks'],
            AlertCategory.QUALITY: ['code_quality_score'],
            AlertCategory.RISK: ['risk_score'],
            AlertCategory.PERFORMANCE: ['response_time', 'error_rate'],
            AlertCategory.BUSINESS: ['satisfaction_score']
        }
        
        relevant_keys = category_mappings.get(category, [])
        return {key: metrics.get(key) for key in relevant_keys if key in metrics}
    
    def _generate_alert_description(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Génère une description détaillée de l'alerte"""
        
        descriptions = {
            'budget_overrun': f"Le budget actuel ({metrics.get('current_cost', 0):,.0f}€) représente {metrics.get('budget_ratio', 0):.1%} du budget total.",
            'timeline_delay': f"Progression actuelle: {metrics.get('progress_ratio', 0):.1%}, attendue: {metrics.get('expected_progress_ratio', 0):.1%}",
            'critical_budget': f"CRITIQUE: Budget dépassé de {(metrics.get('budget_ratio', 0) - 1):.1%}",
            'team_velocity_drop': f"Vélocité actuelle: {metrics.get('current_velocity', 0):.2f} tâches/jour",
            'quality_degradation': f"Score qualité code: {metrics.get('code_quality_score', 0):.1%}",
            'high_risk_detected': f"Score de risque critique: {metrics.get('risk_score', 0):.2f}/10"
        }
        
        return descriptions.get(rule.rule_id, f"Condition déclenchée: {rule.condition}")
    
    async def _detect_ai_anomalies(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Détecte des anomalies via IA (algorithmes avancés)"""
        
        alerts = []
        
        # Détection d'anomalies temporelles
        if len(self.metrics_history) >= 5:
            anomalies = self._detect_time_series_anomalies(metrics)
            alerts.extend(anomalies)
        
        # Détection de patterns inhabituels
        pattern_anomalies = self._detect_pattern_anomalies(metrics)
        alerts.extend(pattern_anomalies)
        
        # Corrélations anormales
        correlation_alerts = self._detect_correlation_anomalies(metrics)
        alerts.extend(correlation_alerts)
        
        return alerts
    
    def _detect_time_series_anomalies(self, current_metrics: Dict[str, Any]) -> List[Alert]:
        """Détecte les anomalies dans les séries temporelles"""
        
        alerts = []
        
        # Analyse des métriques clés sur l'historique
        key_metrics = ['progress_ratio', 'budget_ratio', 'current_velocity', 'code_quality_score']
        
        for metric_name in key_metrics:
            if metric_name not in current_metrics:
                continue
                
            # Extraction de l'historique
            historical_values = [
                entry['metrics'].get(metric_name, 0) 
                for entry in self.metrics_history
                if metric_name in entry['metrics']
            ]
            
            if len(historical_values) < 3:
                continue
            
            current_value = current_metrics[metric_name]
            historical_mean = np.mean(historical_values)
            historical_std = np.std(historical_values)
            
            # Détection d'anomalie (> 2 écarts-types)
            if historical_std > 0:
                z_score = abs(current_value - historical_mean) / historical_std
                
                if z_score > 2.0:
                    alert = Alert(
                        id=f"anomaly_{metric_name}_{datetime.now().timestamp()}",
                        level=AlertLevel.WARNING,
                        category=AlertCategory.PERFORMANCE,
                        title=f"Anomalie détectée: {metric_name}",
                        description=f"Valeur inhabituelle pour {metric_name}: {current_value:.3f} (Z-score: {z_score:.2f})",
                        recommended_actions=[
                            "Analyser les causes de cette variation inhabituelle",
                            "Vérifier s'il s'agit d'une évolution normale ou d'un problème",
                            "Surveiller de près cette métrique dans les prochaines mesures"
                        ],
                        data_points={
                            'current_value': current_value,
                            'historical_mean': historical_mean,
                            'z_score': z_score,
                            'metric_name': metric_name
                        },
                        confidence=min(0.9, z_score / 3.0),
                        created_at=datetime.now()
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _detect_pattern_anomalies(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Détecte des patterns inhabituels dans les métriques"""
        
        alerts = []
        
        # Pattern 1: Progression sans dépenses (suspect)
        if (metrics.get('progress_ratio', 0) > 0.1 and 
            metrics.get('budget_ratio', 0) < 0.05):
            
            alerts.append(Alert(
                id=f"pattern_progress_no_cost_{datetime.now().timestamp()}",
                level=AlertLevel.INFO,
                category=AlertCategory.BUSINESS,
                title="Pattern inhabituel: Progression sans coûts",
                description="Le projet progresse mais les coûts n'augmentent pas proportionnellement",
                recommended_actions=[
                    "Vérifier la saisie des coûts dans le système",
                    "S'assurer que tous les coûts sont bien comptabilisés",
                    "Valider les métriques de progression"
                ],
                data_points=metrics,
                confidence=0.7,
                created_at=datetime.now()
            ))
        
        # Pattern 2: Coûts élevés sans progression (problématique)
        if (metrics.get('budget_ratio', 0) > 0.3 and 
            metrics.get('progress_ratio', 0) < 0.1):
            
            alerts.append(Alert(
                id=f"pattern_cost_no_progress_{datetime.now().timestamp()}",
                level=AlertLevel.ERROR,
                category=AlertCategory.BUDGET,
                title="Pattern problématique: Coûts sans progression",
                description="Les coûts augmentent mais la progression reste faible",
                recommended_actions=[
                    "Analyser l'utilisation des ressources",
                    "Identifier les blocages empêchant la progression",
                    "Revoir l'allocation budgétaire et les priorités"
                ],
                data_points=metrics,
                confidence=0.8,
                created_at=datetime.now()
            ))
        
        return alerts
    
    def _detect_correlation_anomalies(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Détecte des corrélations anormales entre métriques"""
        
        alerts = []
        
        # Corrélation attendue: Plus d'équipe = plus de vélocité
        team_size = metrics.get('team_size', 0)
        velocity = metrics.get('current_velocity', 0)
        
        if team_size > 5 and velocity < 0.5:
            alerts.append(Alert(
                id=f"correlation_team_velocity_{datetime.now().timestamp()}",
                level=AlertLevel.WARNING,
                category=AlertCategory.TEAM,
                title="Corrélation anormale: Grande équipe, faible vélocité",
                description=f"Équipe de {team_size} personnes mais vélocité de seulement {velocity:.2f} tâches/jour",
                recommended_actions=[
                    "Analyser les processus et l'organisation de l'équipe",
                    "Identifier les goulots d'étranglement",
                    "Améliorer la coordination et la communication"
                ],
                data_points={'team_size': team_size, 'velocity': velocity},
                confidence=0.6,
                created_at=datetime.now()
            ))
        
        return alerts
    
    async def _predict_future_issues(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Prédit les problèmes futurs basés sur les tendances"""
        
        alerts = []
        
        # Prédiction 1: Dépassement budgétaire imminent
        budget_ratio = metrics.get('budget_ratio', 0)
        progress_ratio = metrics.get('progress_ratio', 0)
        
        if progress_ratio > 0.1:  # Éviter division par zéro
            projected_budget_ratio = budget_ratio / progress_ratio
            
            if projected_budget_ratio > 1.2:  # Dépassement prévu de 20%
                alerts.append(Alert(
                    id=f"prediction_budget_overrun_{datetime.now().timestamp()}",
                    level=AlertLevel.WARNING,
                    category=AlertCategory.BUDGET,
                    title="Prédiction: Dépassement budgétaire imminent",
                    description=f"À ce rythme, le budget sera dépassé de {(projected_budget_ratio-1)*100:.1f}%",
                    recommended_actions=[
                        "Revoir immédiatement l'allocation budgétaire",
                        "Négocier un budget supplémentaire ou réduire le scope",
                        "Optimiser les processus pour réduire les coûts"
                    ],
                    data_points={
                        'current_budget_ratio': budget_ratio,
                        'projected_budget_ratio': projected_budget_ratio,
                        'overrun_prediction': (projected_budget_ratio - 1) * 100
                    },
                    confidence=0.8,
                    created_at=datetime.now()
                ))
        
        # Prédiction 2: Retard de livraison
        time_ratio = metrics.get('time_ratio', 0)
        
        if time_ratio > 0.1 and progress_ratio > 0.1:
            projected_duration_ratio = time_ratio / progress_ratio
            
            if projected_duration_ratio > 1.3:  # Retard prévu de 30%
                alerts.append(Alert(
                    id=f"prediction_timeline_delay_{datetime.now().timestamp()}",
                    level=AlertLevel.WARNING,
                    category=AlertCategory.TIMELINE,
                    title="Prédiction: Retard de livraison probable",
                    description=f"À ce rythme, le projet aura {(projected_duration_ratio-1)*100:.1f}% de retard",
                    recommended_actions=[
                        "Accélérer le développement des fonctionnalités critiques",
                        "Augmenter temporairement les ressources",
                        "Négocier un report de deadline avec le client"
                    ],
                    data_points={
                        'time_ratio': time_ratio,
                        'progress_ratio': progress_ratio,
                        'delay_prediction': (projected_duration_ratio - 1) * 100
                    },
                    confidence=0.7,
                    created_at=datetime.now()
                ))
        
        return alerts
    
    def _deduplicate_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Supprime les alertes en doublon"""
        
        seen_signatures = set()
        unique_alerts = []
        
        for alert in alerts:
            # Signature basée sur catégorie + niveau + mots-clés du titre
            signature = f"{alert.category.value}_{alert.level.value}_{alert.title[:20]}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_alerts.append(alert)
        
        return unique_alerts
    
    def _update_active_alerts(self, alerts: List[Alert]):
        """Met à jour les alertes actives"""
        
        # Ajouter les nouvelles alertes
        for alert in alerts:
            self.active_alerts[alert.id] = alert
            self.alerts_history.append(alert)
        
        # Nettoyer les alertes expirées
        current_time = datetime.now()
        expired_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.expires_at and current_time > alert.expires_at
        ]
        
        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]
    
    def get_active_alerts(self, category: Optional[AlertCategory] = None, 
                         level: Optional[AlertLevel] = None) -> List[Alert]:
        """Récupère les alertes actives avec filtres optionnels"""
        
        alerts = list(self.active_alerts.values())
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        # Trier par niveau de criticité et date
        level_priority = {
            AlertLevel.CRITICAL: 0,
            AlertLevel.ERROR: 1, 
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 3
        }
        
        alerts.sort(key=lambda a: (level_priority[a.level], -a.created_at.timestamp()))
        
        return alerts
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Marque une alerte comme fermée"""
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].dismissed = True
            del self.active_alerts[alert_id]
            return True
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les alertes"""
        
        active_alerts = list(self.active_alerts.values())
        
        stats = {
            'total_active': len(active_alerts),
            'by_level': {level.value: 0 for level in AlertLevel},
            'by_category': {cat.value: 0 for cat in AlertCategory},
            'total_generated_today': 0,
            'resolution_rate': 0.0
        }
        
        # Comptage par niveau et catégorie
        for alert in active_alerts:
            stats['by_level'][alert.level.value] += 1
            stats['by_category'][alert.category.value] += 1
        
        # Alertes générées aujourd'hui
        today = datetime.now().date()
        today_alerts = [
            alert for alert in self.alerts_history
            if alert.created_at.date() == today
        ]
        stats['total_generated_today'] = len(today_alerts)
        
        # Taux de résolution (alertes fermées vs générées)
        if len(self.alerts_history) > 0:
            dismissed_count = len([a for a in self.alerts_history if a.dismissed])
            stats['resolution_rate'] = dismissed_count / len(self.alerts_history)
        
        return stats
    
    def add_custom_rule(self, rule: AlertRule) -> bool:
        """Ajoute une règle d'alerte personnalisée"""
        
        try:
            # Validation de la règle
            if not rule.rule_id or not rule.name or not rule.condition:
                return False
            
            # Test de la condition (syntaxe)
            test_metrics = {'test': 1, 'value': 0.5}
            try:
                eval(rule.condition, {"__builtins__": {}}, test_metrics)
            except:
                return False
            
            self.alert_rules[rule.rule_id] = rule
            return True
            
        except Exception:
            return False
    
    def export_alerts_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Génère un rapport d'alertes pour une période"""
        
        period_alerts = [
            alert for alert in self.alerts_history
            if start_date <= alert.created_at <= end_date
        ]
        
        return {
            'period': {'start': start_date.isoformat(), 'end': end_date.isoformat()},
            'total_alerts': len(period_alerts),
            'by_level': {
                level.value: len([a for a in period_alerts if a.level == level])
                for level in AlertLevel
            },
            'by_category': {
                cat.value: len([a for a in period_alerts if a.category == cat])
                for cat in AlertCategory
            },
            'top_alerts': [
                {
                    'title': alert.title,
                    'level': alert.level.value,
                    'category': alert.category.value,
                    'created_at': alert.created_at.isoformat(),
                    'confidence': alert.confidence
                }
                for alert in sorted(period_alerts, 
                                  key=lambda a: (AlertLevel.CRITICAL.value == a.level.value, 
                                               -a.confidence))[:10]
            ]
        }


# Instance globale
smart_alert_system = SmartAlertSystem()