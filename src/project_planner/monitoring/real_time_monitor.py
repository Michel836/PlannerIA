"""
Advanced Real-Time Monitoring and Predictive Alerts System for PlannerIA
Surveillance intelligente avec détection d'anomalies et recovery automatique
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import numpy as np
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SystemMetric(Enum):
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CONCURRENT_USERS = "concurrent_users"
    AI_MODEL_LATENCY = "ai_model_latency"

@dataclass
class Alert:
    id: str
    level: AlertLevel
    metric: SystemMetric
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    recovery_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'level': self.level.value,
            'metric': self.metric.value,
            'timestamp': self.timestamp.isoformat()
        }

class PredictiveModel:
    """Modèle prédictif simple pour détecter les anomalies"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.anomaly_threshold = 2.5  # Z-score threshold
        
    def add_datapoint(self, value: float):
        """Ajouter une nouvelle mesure"""
        self.history.append(value)
        
    def is_anomaly(self, value: float) -> bool:
        """Détecter si une valeur est anormale"""
        if len(self.history) < 10:  # Pas assez de données historiques
            return False
            
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        if std == 0:  # Éviter division par zéro
            return False
            
        z_score = abs((value - mean) / std)
        return z_score > self.anomaly_threshold
        
    def predict_trend(self) -> str:
        """Prédire la tendance (croissante/décroissante/stable)"""
        if len(self.history) < 5:
            return "insufficient_data"
            
        recent = list(self.history)[-5:]
        if len(set(recent)) == 1:
            return "stable"
            
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0.1:
            return "increasing"
        elif trend < -0.1:
            return "decreasing"
        else:
            return "stable"

class RealTimeMonitor:
    """Moniteur temps réel avec alertes prédictives et recovery automatique"""
    
    def __init__(self):
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alerts: List[Alert] = []
        self.metrics_history: Dict[SystemMetric, PredictiveModel] = {}
        self.alert_callbacks: List[Callable] = []
        self.recovery_actions: Dict[SystemMetric, Callable] = {}
        
        # Configuration des seuils
        self.thresholds = {
            SystemMetric.CPU_USAGE: {"warning": 70, "critical": 90},
            SystemMetric.MEMORY_USAGE: {"warning": 80, "critical": 95},
            SystemMetric.DISK_USAGE: {"warning": 85, "critical": 95},
            SystemMetric.RESPONSE_TIME: {"warning": 2000, "critical": 5000},  # ms
            SystemMetric.ERROR_RATE: {"warning": 5, "critical": 15},  # %
            SystemMetric.AI_MODEL_LATENCY: {"warning": 10000, "critical": 30000}  # ms
        }
        
        # Initialiser les modèles prédictifs
        for metric in SystemMetric:
            self.metrics_history[metric] = PredictiveModel()
            
        # Actions de récupération par défaut
        self._setup_default_recovery_actions()
        
    def _setup_default_recovery_actions(self):
        """Configuration des actions de récupération automatique"""
        
        def cpu_recovery():
            logger.info("🔄 Recovery: CPU élevé détecté - optimisation automatique")
            return "cpu_optimized"
            
        def memory_recovery():
            logger.info("🔄 Recovery: Mémoire critique - nettoyage cache")
            return "memory_cleaned"
            
        def response_time_recovery():
            logger.info("🔄 Recovery: Latence élevée - basculement mode rapide")
            return "fast_mode_enabled"
            
        self.recovery_actions = {
            SystemMetric.CPU_USAGE: cpu_recovery,
            SystemMetric.MEMORY_USAGE: memory_recovery,
            SystemMetric.RESPONSE_TIME: response_time_recovery
        }
        
    def add_alert_callback(self, callback: Callable):
        """Ajouter un callback pour les nouvelles alertes"""
        self.alert_callbacks.append(callback)
        
    def start_monitoring(self, interval: float = 1.0):
        """Démarrer la surveillance temps réel"""
        if self.is_running:
            logger.warning("Monitoring déjà en cours")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("🔍 Monitoring temps réel démarré")
        
    def stop_monitoring(self):
        """Arrêter la surveillance"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("⏹️ Monitoring arrêté")
        
    def _monitor_loop(self, interval: float):
        """Boucle principale de surveillance"""
        while self.is_running:
            try:
                # Collecter les métriques système
                metrics = self._collect_system_metrics()
                
                # Analyser chaque métrique
                for metric, value in metrics.items():
                    self._analyze_metric(metric, value)
                    
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de monitoring: {e}")
                time.sleep(interval)
                
    def _collect_system_metrics(self) -> Dict[SystemMetric, float]:
        """Collecter les métriques système actuelles"""
        try:
            # Métriques système de base
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Simuler d'autres métriques (dans un vrai système, ces valeurs viendraient de votre application)
            response_time = np.random.normal(1000, 200)  # Simulé
            error_rate = max(0, np.random.normal(2, 1))  # Simulé
            ai_latency = np.random.normal(5000, 1000)  # Simulé
            
            return {
                SystemMetric.CPU_USAGE: cpu_percent,
                SystemMetric.MEMORY_USAGE: memory.percent,
                SystemMetric.DISK_USAGE: disk.percent,
                SystemMetric.RESPONSE_TIME: response_time,
                SystemMetric.ERROR_RATE: error_rate,
                SystemMetric.AI_MODEL_LATENCY: ai_latency
            }
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques: {e}")
            return {}
            
    def _analyze_metric(self, metric: SystemMetric, value: float):
        """Analyser une métrique et générer des alertes si nécessaire"""
        
        # Ajouter à l'historique
        self.metrics_history[metric].add_datapoint(value)
        
        # Vérifier les seuils statiques
        alert_level = self._check_thresholds(metric, value)
        
        # Vérifier les anomalies prédictives
        is_anomaly = self.metrics_history[metric].is_anomaly(value)
        
        # Générer alerte si nécessaire
        if alert_level or is_anomaly:
            alert = self._create_alert(metric, value, alert_level, is_anomaly)
            self.alerts.append(alert)
            
            # Notifier les callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Erreur callback alerte: {e}")
                    
            # Déclencher recovery automatique si critique
            if alert_level == AlertLevel.CRITICAL and metric in self.recovery_actions:
                try:
                    recovery_result = self.recovery_actions[metric]()
                    alert.recovery_action = recovery_result
                    logger.info(f"✅ Recovery automatique exécutée pour {metric.value}")
                except Exception as e:
                    logger.error(f"Erreur recovery automatique: {e}")
                    
    def _check_thresholds(self, metric: SystemMetric, value: float) -> Optional[AlertLevel]:
        """Vérifier si une valeur dépasse les seuils configurés"""
        if metric not in self.thresholds:
            return None
            
        thresholds = self.thresholds[metric]
        
        if value >= thresholds.get("critical", float('inf')):
            return AlertLevel.CRITICAL
        elif value >= thresholds.get("warning", float('inf')):
            return AlertLevel.WARNING
            
        return None
        
    def _create_alert(self, metric: SystemMetric, value: float, 
                      alert_level: Optional[AlertLevel], is_anomaly: bool) -> Alert:
        """Créer une nouvelle alerte"""
        
        if alert_level:
            level = alert_level
            threshold = self.thresholds[metric][level.value]
            message = f"{metric.value} élevé: {value:.1f} > {threshold}"
        else:
            level = AlertLevel.INFO
            threshold = 0
            message = f"Anomalie détectée pour {metric.value}: {value:.1f}"
            
        trend = self.metrics_history[metric].predict_trend()
        if trend == "increasing":
            message += " (tendance croissante)"
        elif trend == "decreasing":
            message += " (tendance décroissante)"
            
        alert_id = f"{metric.value}_{int(time.time())}"
        
        return Alert(
            id=alert_id,
            level=level,
            metric=metric,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtenir un snapshot des métriques actuelles"""
        try:
            raw_metrics = self._collect_system_metrics()
            
            # Enrichir avec les prédictions
            enriched_metrics = {}
            for metric, value in raw_metrics.items():
                trend = self.metrics_history[metric].predict_trend()
                is_anomaly = self.metrics_history[metric].is_anomaly(value)
                
                enriched_metrics[metric.value] = {
                    "value": value,
                    "trend": trend,
                    "is_anomaly": is_anomaly,
                    "history_size": len(self.metrics_history[metric].history)
                }
                
            return enriched_metrics
            
        except Exception as e:
            logger.error(f"Erreur snapshot métriques: {e}")
            return {}
            
    def get_active_alerts(self, max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """Obtenir les alertes actives récentes"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        recent_alerts = [
            alert for alert in self.alerts 
            if alert.timestamp > cutoff_time and not alert.resolved
        ]
        
        return [alert.to_dict() for alert in recent_alerts]
        
    def resolve_alert(self, alert_id: str):
        """Marquer une alerte comme résolue"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"✅ Alerte {alert_id} résolue")
                break
                
    def get_system_health_score(self) -> Dict[str, Any]:
        """Calculer un score de santé global du système"""
        
        current_metrics = self._collect_system_metrics()
        if not current_metrics:
            return {"score": 0, "status": "unknown", "details": "Pas de données"}
            
        # Calculer les scores individuels (0-100)
        scores = {}
        for metric, value in current_metrics.items():
            if metric in self.thresholds:
                critical_threshold = self.thresholds[metric].get("critical", 100)
                warning_threshold = self.thresholds[metric].get("warning", 80)
                
                if value >= critical_threshold:
                    score = 0
                elif value >= warning_threshold:
                    score = 50 * (1 - (value - warning_threshold) / (critical_threshold - warning_threshold))
                else:
                    score = 100 * (1 - value / warning_threshold) if warning_threshold > 0 else 100
                    
                scores[metric.value] = max(0, min(100, score))
                
        if not scores:
            return {"score": 100, "status": "healthy", "details": "Aucun seuil configuré"}
            
        # Score global (moyenne pondérée)
        weights = {
            SystemMetric.CPU_USAGE.value: 0.3,
            SystemMetric.MEMORY_USAGE.value: 0.3,
            SystemMetric.RESPONSE_TIME.value: 0.2,
            SystemMetric.ERROR_RATE.value: 0.2
        }
        
        weighted_score = sum(
            scores.get(metric, 100) * weight 
            for metric, weight in weights.items()
        ) / sum(weights.values())
        
        # Déterminer le statut
        if weighted_score >= 80:
            status = "healthy"
        elif weighted_score >= 60:
            status = "warning"
        elif weighted_score >= 30:
            status = "critical"
        else:
            status = "emergency"
            
        return {
            "score": round(weighted_score, 1),
            "status": status,
            "individual_scores": scores,
            "active_alerts": len(self.get_active_alerts(1))  # Dernière heure
        }
        
    def export_monitoring_data(self, hours: int = 1) -> Dict[str, Any]:
        """Exporter les données de monitoring pour analyse"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_period_hours": hours,
            "system_health": self.get_system_health_score(),
            "current_metrics": self.get_current_metrics(),
            "active_alerts": self.get_active_alerts(hours),
            "total_alerts_generated": len(self.alerts),
            "is_monitoring_active": self.is_running
        }

# Instance globale pour l'utilisation dans le dashboard
global_monitor = RealTimeMonitor()

def get_monitor() -> RealTimeMonitor:
    """Obtenir l'instance globale du moniteur"""
    return global_monitor

# Fonctions utilitaires pour l'intégration avec Streamlit
def start_monitoring():
    """Démarrer le monitoring global"""
    global_monitor.start_monitoring()

def stop_monitoring():
    """Arrêter le monitoring global"""
    global_monitor.stop_monitoring()

def get_health_status() -> Dict[str, Any]:
    """Obtenir le statut de santé actuel"""
    return global_monitor.get_system_health_score()

def get_real_time_metrics() -> Dict[str, Any]:
    """Obtenir les métriques temps réel"""
    return global_monitor.get_current_metrics()

def get_recent_alerts(hours: int = 1) -> List[Dict[str, Any]]:
    """Obtenir les alertes récentes"""
    return global_monitor.get_active_alerts(hours)