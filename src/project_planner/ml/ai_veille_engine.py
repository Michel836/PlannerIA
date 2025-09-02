#!/usr/bin/env python3
"""
🤖 MOTEUR IA DE VEILLE - PlannerIA
Surveillance intelligente et prédictive des projets
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import logging
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Alerte:
    """Structure d'une alerte IA"""
    id: str
    type: str  # 'budget', 'delai', 'risque', 'qualite', 'anomalie'
    criticite: str  # 'critique', 'elevee', 'moyenne', 'faible'
    score_confiance: float  # 0.0 - 1.0
    message: str
    recommandation: str
    donnees: Dict[str, Any]
    timestamp: datetime
    action_requise: bool = True
    module_cible: str = None

@dataclass
class Prediction:
    """Structure d'une prédiction"""
    metrique: str
    valeur_predite: float
    valeur_actuelle: float
    horizon_jours: int
    score_confiance: float
    tendance: str  # 'amelioration', 'degradation', 'stable'
    seuil_alerte: float
    timestamp: datetime

class AIVeilleEngine:
    """
    🧠 Moteur IA de Veille Intelligente
    
    Fonctionnalités:
    - Prédictions multi-horizons (3j, 7j, 14j, 30j)
    - Détection d'anomalies temps réel
    - Système d'alertes intelligent avec prioritisation
    - Apprentissage continu sur historique projet
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
        
        # Modèles ML initialisés
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        self.predictors = {
            'budget_usage': RandomForestRegressor(n_estimators=50, random_state=42),
            'completion_rate': RandomForestRegressor(n_estimators=50, random_state=42),
            'quality_score': RandomForestRegressor(n_estimators=50, random_state=42),
            'risk_score': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        self.scalers = {
            key: StandardScaler() for key in self.predictors.keys()
        }
        
        # État de la veille
        self.historique_metriques = pd.DataFrame()
        self.alertes_actives = []
        self.predictions_cache = {}
        self.modeles_entraines = False
        
        self.logger.info("🤖 Moteur IA Veille initialisé")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'seuils_alertes': {
                'budget': {'critique': 0.95, 'elevee': 0.85, 'moyenne': 0.75},
                'delai': {'critique': 1.0, 'elevee': 0.8, 'moyenne': 0.6},
                'qualite': {'critique': 0.3, 'elevee': 0.5, 'moyenne': 0.7},
                'risque': {'critique': 0.8, 'elevee': 0.6, 'moyenne': 0.4}
            },
            'horizons_prediction': [3, 7, 14, 30],
            'fenetre_historique': 90,  # jours
            'freq_refresh': 300,  # secondes
            'min_confiance_alerte': 0.7,
            'clustering_eps': 0.5,
            'clustering_min_samples': 3
        }
    
    def ingerer_donnees(self, metriques: Dict[str, float], timestamp: Optional[datetime] = None) -> bool:
        """
        📊 Ingère nouvelles données de métriques projet
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Validation des données
            metriques_requises = ['budget_usage', 'completion_rate', 'quality_score', 'risk_score']
            if not all(key in metriques for key in metriques_requises):
                missing = [k for k in metriques_requises if k not in metriques]
                self.logger.warning(f"Métriques manquantes: {missing}")
                return False
            
            # Ajout à l'historique
            nouvelle_ligne = {
                'timestamp': timestamp,
                **metriques
            }
            
            df_nouveau = pd.DataFrame([nouvelle_ligne])
            self.historique_metriques = pd.concat([self.historique_metriques, df_nouveau], ignore_index=True)
            
            # Nettoyage historique (garder seulement fenêtre configurée)
            fenetre = self.config['fenetre_historique']
            date_limite = timestamp - timedelta(days=fenetre)
            self.historique_metriques = self.historique_metriques[
                self.historique_metriques['timestamp'] > date_limite
            ]
            
            self.logger.debug(f"Données ingérées: {len(self.historique_metriques)} points historiques")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur ingestion données: {e}")
            return False
    
    def detecter_anomalies(self) -> List[Alerte]:
        """
        🎯 Détection d'anomalies avec Isolation Forest
        """
        alertes = []
        
        try:
            if len(self.historique_metriques) < 10:
                self.logger.warning("Pas assez de données pour détection anomalies")
                return alertes
            
            # Préparation des features
            features = ['budget_usage', 'completion_rate', 'quality_score', 'risk_score']
            X = self.historique_metriques[features].fillna(0)
            
            # Détection anomalies
            anomalies = self.anomaly_detector.fit_predict(X)
            scores = self.anomaly_detector.score_samples(X)
            
            # Identifier les anomalies récentes
            indices_anomalies = np.where(anomalies == -1)[0]
            
            for idx in indices_anomalies[-5:]:  # 5 anomalies les plus récentes
                score_confiance = abs(scores[idx])
                donnees_point = self.historique_metriques.iloc[idx]
                
                if score_confiance > self.config['min_confiance_alerte']:
                    alerte = Alerte(
                        id=f"anomalie_{idx}_{int(datetime.now().timestamp())}",
                        type='anomalie',
                        criticite=self._evaluer_criticite_anomalie(score_confiance),
                        score_confiance=float(score_confiance),
                        message=f"Anomalie détectée dans les métriques projet",
                        recommandation=self._generer_recommandation_anomalie(donnees_point),
                        donnees={'point_anormal': donnees_point.to_dict()},
                        timestamp=donnees_point['timestamp'],
                        module_cible='📊 Dashboard Principal'
                    )
                    alertes.append(alerte)
            
            self.logger.info(f"🎯 {len(alertes)} anomalies détectées")
            return alertes
            
        except Exception as e:
            self.logger.error(f"Erreur détection anomalies: {e}")
            return alertes
    
    def predire_metriques(self, horizons: Optional[List[int]] = None) -> Dict[str, List[Prediction]]:
        """
        📈 Prédictions multi-horizons des métriques clés
        """
        if horizons is None:
            horizons = self.config['horizons_prediction']
        
        predictions = {}
        
        try:
            if len(self.historique_metriques) < 20:
                self.logger.warning("Pas assez de données pour prédictions")
                return predictions
            
            # Préparer les données pour ML
            df = self.historique_metriques.copy()
            df = df.sort_values('timestamp')
            
            metriques = ['budget_usage', 'completion_rate', 'quality_score', 'risk_score']
            
            for metrique in metriques:
                predictions[metrique] = []
                
                try:
                    # Features temporelles
                    X = self._creer_features_temporelles(df, metrique)
                    y = df[metrique].fillna(0).values
                    
                    if len(X) < 10:
                        continue
                    
                    # Entraînement si nécessaire
                    if not self.modeles_entraines:
                        X_scaled = self.scalers[metrique].fit_transform(X)
                        self.predictors[metrique].fit(X_scaled, y[-len(X):])
                    
                    # Prédictions pour chaque horizon
                    for horizon in horizons:
                        pred = self._predire_horizon(metrique, horizon, X, y)
                        if pred:
                            predictions[metrique].append(pred)
                
                except Exception as e:
                    self.logger.warning(f"Erreur prédiction {metrique}: {e}")
                    continue
            
            self.modeles_entraines = True
            self.predictions_cache = predictions
            
            self.logger.info(f"📈 Prédictions générées pour {len(predictions)} métriques")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erreur prédictions: {e}")
            return predictions
    
    def generer_alertes_intelligentes(self) -> List[Alerte]:
        """
        🚨 Génération d'alertes avec prioritisation IA
        """
        alertes = []
        
        # Anomalies temps réel
        alertes.extend(self.detecter_anomalies())
        
        # Alertes prédictives
        predictions = self.predire_metriques()
        for metrique, preds in predictions.items():
            for pred in preds:
                if pred.score_confiance > self.config['min_confiance_alerte']:
                    alerte = self._creer_alerte_predictive(metrique, pred)
                    if alerte:
                        alertes.append(alerte)
        
        # Alertes sur seuils actuels
        if len(self.historique_metriques) > 0:
            alertes.extend(self._detecter_alertes_seuils())
        
        # Prioritisation finale
        alertes = self._prioriser_alertes(alertes)
        
        self.alertes_actives = alertes
        self.logger.info(f"🚨 {len(alertes)} alertes générées")
        
        return alertes
    
    def _creer_features_temporelles(self, df: pd.DataFrame, metrique: str) -> np.ndarray:
        """Crée features temporelles pour prédictions"""
        features = []
        
        values = df[metrique].fillna(0).values
        
        for i in range(5, len(values)):
            # Fenêtre glissante 5 points
            window = values[i-5:i]
            
            feature_row = [
                np.mean(window),  # moyenne
                np.std(window),   # écart-type
                values[i-1],      # valeur précédente
                np.max(window) - np.min(window),  # amplitude
                len(window)       # taille fenêtre
            ]
            features.append(feature_row)
        
        return np.array(features) if features else np.array([]).reshape(0, 5)
    
    def _predire_horizon(self, metrique: str, horizon: int, X: np.ndarray, y: np.ndarray) -> Optional[Prediction]:
        """Prédit une métrique à un horizon donné"""
        try:
            if len(X) == 0:
                return None
            
            # Prédiction
            X_scaled = self.scalers[metrique].transform(X[-1:])
            valeur_predite = self.predictors[metrique].predict(X_scaled)[0]
            
            # Calcul confiance (basé sur variance modèle)
            scores = self.predictors[metrique].predict(self.scalers[metrique].transform(X))
            mse = np.mean((scores - y[-len(scores):]) ** 2)
            score_confiance = max(0.1, 1.0 - min(mse, 1.0))
            
            # Évaluation tendance
            valeur_actuelle = y[-1] if len(y) > 0 else 0
            diff = valeur_predite - valeur_actuelle
            
            if abs(diff) < 0.05:
                tendance = 'stable'
            elif diff > 0:
                tendance = 'amelioration' if metrique in ['completion_rate', 'quality_score'] else 'degradation'
            else:
                tendance = 'degradation' if metrique in ['completion_rate', 'quality_score'] else 'amelioration'
            
            return Prediction(
                metrique=metrique,
                valeur_predite=float(valeur_predite),
                valeur_actuelle=float(valeur_actuelle),
                horizon_jours=horizon,
                score_confiance=float(score_confiance),
                tendance=tendance,
                seuil_alerte=self._get_seuil_metrique(metrique),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.warning(f"Erreur prédiction {metrique} H{horizon}: {e}")
            return None
    
    def _get_seuil_metrique(self, metrique: str) -> float:
        """Récupère seuil d'alerte pour une métrique"""
        mapping = {
            'budget_usage': 'budget',
            'completion_rate': 'delai', 
            'quality_score': 'qualite',
            'risk_score': 'risque'
        }
        
        type_metrique = mapping.get(metrique, 'budget')
        if type_metrique in self.config['seuils_alertes']:
            return self.config['seuils_alertes'][type_metrique]['elevee']
        return 0.7
    
    def _evaluer_criticite_anomalie(self, score: float) -> str:
        """Évalue criticité d'une anomalie"""
        if score > 0.8:
            return 'critique'
        elif score > 0.6:
            return 'elevee'
        elif score > 0.4:
            return 'moyenne'
        return 'faible'
    
    def _generer_recommandation_anomalie(self, donnees: pd.Series) -> str:
        """Génère recommandation pour anomalie"""
        recommendations = []
        
        if donnees.get('budget_usage', 0) > 0.9:
            recommendations.append("Vérifier les dépenses exceptionnelles")
        if donnees.get('quality_score', 1) < 0.5:
            recommendations.append("Auditer les processus qualité")
        if donnees.get('risk_score', 0) > 0.7:
            recommendations.append("Activer plan de contingence")
            
        return " • ".join(recommendations) if recommendations else "Analyser les métriques détaillées"
    
    def _creer_alerte_predictive(self, metrique: str, prediction: Prediction) -> Optional[Alerte]:
        """Crée alerte basée sur prédiction"""
        try:
            seuil = self._get_seuil_metrique(metrique)
            
            # Vérifier si prédiction dépasse seuil
            alerte_requise = False
            if metrique in ['budget_usage', 'risk_score']:
                alerte_requise = prediction.valeur_predite > seuil
            else:  # completion_rate, quality_score
                alerte_requise = prediction.valeur_predite < seuil
            
            if not alerte_requise:
                return None
            
            criticite = self._evaluer_criticite_prediction(prediction, seuil)
            
            return Alerte(
                id=f"pred_{metrique}_{prediction.horizon_jours}j_{int(datetime.now().timestamp())}",
                type='prediction',
                criticite=criticite,
                score_confiance=prediction.score_confiance,
                message=f"Prédiction {prediction.tendance} pour {metrique} dans {prediction.horizon_jours}j",
                recommandation=self._generer_recommandation_predictive(metrique, prediction),
                donnees={'prediction': prediction.__dict__},
                timestamp=datetime.now(),
                module_cible=self._get_module_metrique(metrique)
            )
            
        except Exception as e:
            self.logger.warning(f"Erreur création alerte prédictive: {e}")
            return None
    
    def _evaluer_criticite_prediction(self, pred: Prediction, seuil: float) -> str:
        """Évalue criticité d'une prédiction"""
        ecart = abs(pred.valeur_predite - seuil) / seuil if seuil > 0 else 0
        
        if ecart > 0.3 and pred.score_confiance > 0.8:
            return 'critique'
        elif ecart > 0.2 and pred.score_confiance > 0.6:
            return 'elevee'
        elif ecart > 0.1:
            return 'moyenne'
        return 'faible'
    
    def _generer_recommandation_predictive(self, metrique: str, pred: Prediction) -> str:
        """Génère recommandation pour prédiction"""
        recommendations = {
            'budget_usage': f"Réviser allocation budgétaire avant dépassement",
            'completion_rate': f"Accélérer livraisons ou réajuster planning",
            'quality_score': f"Renforcer contrôles qualité immédiatement",
            'risk_score': f"Activer mesures de mitigation des risques"
        }
        
        base_rec = recommendations.get(metrique, "Surveiller évolution métrique")
        return f"{base_rec} (confiance: {pred.score_confiance:.1%})"
    
    def _get_module_metrique(self, metrique: str) -> str:
        """Retourne module cible pour une métrique"""
        mapping = {
            'budget_usage': '💰 Budget',
            'completion_rate': '📋 Planification', 
            'quality_score': '✅ Qualité',
            'risk_score': '⚠️ Analyse des Risques'
        }
        return mapping.get(metrique, '📊 Dashboard Principal')
    
    def _detecter_alertes_seuils(self) -> List[Alerte]:
        """Détecte alertes sur seuils actuels"""
        alertes = []
        
        try:
            derniere_ligne = self.historique_metriques.iloc[-1]
            
            mapping = {
                'budget_usage': 'budget',
                'completion_rate': 'delai',
                'quality_score': 'qualite', 
                'risk_score': 'risque'
            }
            
            for metrique in ['budget_usage', 'completion_rate', 'quality_score', 'risk_score']:
                valeur = derniere_ligne.get(metrique, 0)
                type_metrique = mapping.get(metrique, 'budget')
                seuils = self.config['seuils_alertes'][type_metrique]
                
                criticite = None
                if metrique in ['budget_usage', 'risk_score']:
                    # Plus c'est haut, plus c'est critique
                    if valeur >= seuils['critique']:
                        criticite = 'critique'
                    elif valeur >= seuils['elevee']:
                        criticite = 'elevee'
                    elif valeur >= seuils['moyenne']:
                        criticite = 'moyenne'
                else:
                    # Plus c'est bas, plus c'est critique
                    if valeur <= seuils['critique']:
                        criticite = 'critique'
                    elif valeur <= seuils['elevee']:
                        criticite = 'elevee'
                    elif valeur <= seuils['moyenne']:
                        criticite = 'moyenne'
                
                if criticite:
                    alerte = Alerte(
                        id=f"seuil_{metrique}_{int(datetime.now().timestamp())}",
                        type='seuil',
                        criticite=criticite,
                        score_confiance=0.95,  # Haute confiance pour seuils
                        message=f"Seuil {criticite} dépassé pour {metrique}: {valeur:.2%}",
                        recommandation=f"Action immédiate requise sur {metrique}",
                        donnees={'valeur_actuelle': valeur, 'seuil': seuils[criticite]},
                        timestamp=datetime.now(),
                        module_cible=self._get_module_metrique(metrique)
                    )
                    alertes.append(alerte)
        
        except Exception as e:
            self.logger.warning(f"Erreur détection seuils: {e}")
        
        return alertes
    
    def _prioriser_alertes(self, alertes: List[Alerte]) -> List[Alerte]:
        """Priorise les alertes par criticité et confiance"""
        def score_priorite(alerte: Alerte) -> float:
            weights = {
                'critique': 4.0,
                'elevee': 3.0, 
                'moyenne': 2.0,
                'faible': 1.0
            }
            return weights.get(alerte.criticite, 1.0) * alerte.score_confiance
        
        return sorted(alertes, key=score_priorite, reverse=True)
    
    def obtenir_score_sante(self) -> Dict[str, Any]:
        """
        💚 Calcule score de santé global du projet (0-100)
        """
        try:
            if len(self.historique_metriques) == 0:
                return {'score': 50, 'status': 'inconnu', 'details': 'Pas de données'}
            
            derniere_ligne = self.historique_metriques.iloc[-1]
            
            # Pondération des métriques
            scores = {
                'budget': max(0, 100 - (derniere_ligne.get('budget_usage', 0.5) * 100)),
                'completion': derniere_ligne.get('completion_rate', 0.5) * 100,
                'quality': derniere_ligne.get('quality_score', 0.7) * 100,
                'risk': max(0, 100 - (derniere_ligne.get('risk_score', 0.3) * 100))
            }
            
            # Score pondéré
            poids = {'budget': 0.25, 'completion': 0.30, 'quality': 0.25, 'risk': 0.20}
            score_global = sum(scores[k] * poids[k] for k in scores.keys())
            
            # Status
            if score_global >= 80:
                status = 'excellent'
            elif score_global >= 60:
                status = 'bon'
            elif score_global >= 40:
                status = 'moyen'
            else:
                status = 'critique'
            
            return {
                'score': round(score_global, 1),
                'status': status,
                'details': scores,
                'timestamp': datetime.now(),
                'nb_alertes_actives': len(self.alertes_actives)
            }
            
        except Exception as e:
            self.logger.error(f"Erreur calcul score santé: {e}")
            return {'score': 0, 'status': 'erreur', 'details': str(e)}
    
    def obtenir_resume_veille(self) -> Dict[str, Any]:
        """
        📋 Retourne résumé complet de la veille
        """
        try:
            score_sante = self.obtenir_score_sante()
            alertes = self.generer_alertes_intelligentes()
            
            # Statistiques alertes
            stats_alertes = {
                'total': len(alertes),
                'critique': len([a for a in alertes if a.criticite == 'critique']),
                'elevee': len([a for a in alertes if a.criticite == 'elevee']),
                'moyenne': len([a for a in alertes if a.criticite == 'moyenne']),
                'faible': len([a for a in alertes if a.criticite == 'faible'])
            }
            
            # Top alertes (5 plus critiques)
            top_alertes = alertes[:5]
            
            # Prédictions récentes
            predictions_resume = {}
            if self.predictions_cache:
                for metrique, preds in self.predictions_cache.items():
                    if preds:
                        # Prendre prédiction 7j
                        pred_7j = next((p for p in preds if p.horizon_jours == 7), preds[0])
                        predictions_resume[metrique] = {
                            'valeur_predite': pred_7j.valeur_predite,
                            'tendance': pred_7j.tendance,
                            'confiance': pred_7j.score_confiance
                        }
            
            return {
                'timestamp': datetime.now(),
                'score_sante': score_sante,
                'alertes': {
                    'statistiques': stats_alertes,
                    'top_alertes': [a.__dict__ for a in top_alertes]
                },
                'predictions': predictions_resume,
                'etat_systeme': {
                    'nb_points_historique': len(self.historique_metriques),
                    'modeles_entraines': self.modeles_entraines,
                    'derniere_maj': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Erreur résumé veille: {e}")
            return {'erreur': str(e), 'timestamp': datetime.now()}

# Test rapide si exécuté directement
if __name__ == "__main__":
    import random
    
    # Créer moteur de test
    moteur = AIVeilleEngine()
    
    # Générer données de test
    for i in range(30):
        metriques_test = {
            'budget_usage': random.uniform(0.3, 0.95),
            'completion_rate': random.uniform(0.4, 0.9),
            'quality_score': random.uniform(0.5, 0.95),
            'risk_score': random.uniform(0.1, 0.8)
        }
        
        timestamp_test = datetime.now() - timedelta(days=30-i)
        moteur.ingerer_donnees(metriques_test, timestamp_test)
    
    # Test des fonctionnalités
    print("Test Moteur IA Veille")
    print(f"Points historique: {len(moteur.historique_metriques)}")
    
    # Score santé
    score = moteur.obtenir_score_sante()
    print(f"Score sante: {score['score']}/100 ({score['status']})")
    
    # Alertes
    alertes = moteur.generer_alertes_intelligentes()
    print(f"Alertes generees: {len(alertes)}")
    
    # Prédictions
    predictions = moteur.predire_metriques()
    print(f"Predictions: {len(predictions)} metriques")
    
    print("Test termine avec succes")