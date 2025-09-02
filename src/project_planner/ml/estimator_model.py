"""
EstimatorModel optimisé pour PlannerIA
Modèle ML avancé pour l'estimation de durée et coût des tâches
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
import re

# ML imports avec gestion d'absence
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Métriques de performance du modèle"""
    mae: float
    mse: float
    r2: float
    cv_score: float


class AdvancedFeatureExtractor:
    """Extracteur de caractéristiques avancé"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.label_encoders = {}
        self.complexity_weights = {
            'simple': 0.5, 'low': 0.7, 'medium': 1.0, 
            'high': 1.5, 'complex': 1.8, 'very_high': 2.2, 'critical': 2.5
        }
        self.priority_weights = {
            'low': 0.8, 'medium': 1.0, 'high': 1.3, 'critical': 1.6, 'urgent': 2.0
        }
        
    def extract_text_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Extrait les caractéristiques textuelles avec TF-IDF"""
        if not SKLEARN_AVAILABLE:
            return np.zeros((len(texts), 1))
            
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
            return self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            return self.tfidf_vectorizer.transform(texts).toarray()
    
    def extract_task_keywords(self, text: str) -> Dict[str, int]:
        """Extrait des mots-clés techniques spécifiques"""
        keywords = {
            # Domaines techniques
            'frontend': ['ui', 'interface', 'css', 'html', 'react', 'vue', 'angular'],
            'backend': ['api', 'server', 'database', 'sql', 'endpoint', 'service'],
            'mobile': ['ios', 'android', 'mobile', 'app', 'native'],
            'devops': ['deploy', 'docker', 'kubernetes', 'ci/cd', 'infrastructure'],
            'testing': ['test', 'qa', 'unit', 'integration', 'e2e'],
            'security': ['auth', 'security', 'encryption', 'ssl', 'oauth'],
            'ai_ml': ['ai', 'ml', 'machine learning', 'neural', 'model'],
        }
        
        text_lower = text.lower()
        features = {}
        
        for category, words in keywords.items():
            count = sum(1 for word in words if word in text_lower)
            features[f'{category}_keywords'] = count
            
        return features
    
    def extract_complexity_indicators(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Analyse la complexité basée sur différents indicateurs"""
        indicators = {}
        
        # Analyse textuelle
        text = f"{task.get('name', '')} {task.get('description', '')}"
        
        # Mots indiquant la complexité
        complex_words = ['complex', 'advanced', 'sophisticated', 'enterprise', 'scalable']
        simple_words = ['simple', 'basic', 'quick', 'easy', 'straightforward']
        
        indicators['complexity_score'] = (
            sum(1 for word in complex_words if word in text.lower()) -
            sum(1 for word in simple_words if word in text.lower())
        )
        
        # Longueur et détail
        indicators['description_complexity'] = min(3.0, len(text.split()) / 20)
        indicators['technical_depth'] = len(re.findall(r'\b[A-Z]{2,}\b', text)) / 10
        
        return indicators


class EstimatorModel:
    """Modèle d'estimation ML avancé pour PlannerIA"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path("data/models") if model_path is None else Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Modèles
        self.duration_model = None
        self.cost_model = None
        
        # Preprocessing
        self.feature_extractor = AdvancedFeatureExtractor()
        self.scaler = None
        self.feature_names = []
        
        # Métadonnées
        self.training_date = None
        self.model_version = "1.0"
        self.performance_metrics = {}
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn non disponible. Estimation heuristique activée.")
        
        # Chargement automatique si modèle existant
        self._try_load_latest_model()
    
    def _try_load_latest_model(self) -> bool:
        """Tente de charger le dernier modèle sauvegardé"""
        latest_model = self.model_path / "estimator_latest.pkl"
        if latest_model.exists():
            return self.load_model(str(latest_model))
        return False
    
    def extract_features(self, tasks: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extraction complète des caractéristiques"""
        features_data = []
        task_texts = []
        
        for task in tasks:
            # Texte pour TF-IDF
            task_text = f"{task.get('name', '')} {task.get('description', '')}"
            task_texts.append(task_text)
            
            # Caractéristiques de base
            features = {
                'name_length': len(task.get('name', '')),
                'description_length': len(task.get('description', '')),
                'word_count': len(task.get('description', '').split()),
                
                # Complexité encodée
                'complexity_numeric': self.feature_extractor.complexity_weights.get(
                    task.get('complexity_level', 'medium'), 1.0
                ),
                'priority_numeric': self.feature_extractor.priority_weights.get(
                    task.get('priority', 'medium'), 1.0
                ),
                
                # Ressources et dépendances
                'team_size': task.get('team_size', len(task.get('assigned_resources', [1]))),
                'dependencies_count': len(task.get('dependencies', [])),
                'deliverables_count': len(task.get('deliverables', [])),
                'has_dependencies': 1 if task.get('dependencies') else 0,
                
                # Caractéristiques catégorielles
                'task_type': task.get('task_type', 'backend'),
                'priority': task.get('priority', 'medium'),
                'complexity_level': task.get('complexity_level', 'medium')
            }
            
            # Ajout des mots-clés techniques
            keyword_features = self.feature_extractor.extract_task_keywords(task_text)
            features.update(keyword_features)
            
            # Ajout des indicateurs de complexité
            complexity_indicators = self.feature_extractor.extract_complexity_indicators(task)
            features.update(complexity_indicators)
            
            features_data.append(features)
        
        # Création du DataFrame
        df = pd.DataFrame(features_data)
        
        # Ajout des caractéristiques TF-IDF si disponible
        if SKLEARN_AVAILABLE and task_texts:
            try:
                tfidf_features = self.feature_extractor.extract_text_features(
                    task_texts, fit=(self.duration_model is None)
                )
                
                # Ajout des colonnes TF-IDF
                for i in range(tfidf_features.shape[1]):
                    df[f'tfidf_{i}'] = tfidf_features[:, i]
                    
            except Exception as e:
                logger.warning(f"Erreur TF-IDF: {e}")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prépare les données pour l'entraînement"""
        # Colonnes catégorielles et numériques
        self.categorical_cols = ['task_type', 'priority', 'complexity_level']
        self.numeric_cols = [col for col in df.columns 
                            if col not in self.categorical_cols + ['duration', 'cost']]
        
        # Preprocessing pipeline
        if SKLEARN_AVAILABLE:
            self.preprocessor = ColumnTransformer([
                ('num', StandardScaler(), self.numeric_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_cols)
            ])
            
            X = self.preprocessor.fit_transform(df[self.numeric_cols + self.categorical_cols])
            
            # Noms des caractéristiques après preprocessing
            cat_feature_names = []
            if hasattr(self.preprocessor.named_transformers_['cat'], 'categories_'):
                for col, categories in zip(self.categorical_cols, 
                                         self.preprocessor.named_transformers_['cat'].categories_):
                    for cat in categories[1:]:  # Skip first due to drop='first'
                        cat_feature_names.append(f"{col}_{cat}")
            
            feature_names = self.numeric_cols + cat_feature_names
            self.scaler = self.preprocessor
            
        else:
            # Fallback simple - utilise seulement les colonnes numériques
            X = df[self.numeric_cols].fillna(0).values
            feature_names = self.numeric_cols
            self.scaler = None
            
        return X, feature_names
    
    def train_model_ensemble(self, X: np.ndarray, y: np.ndarray, 
                           model_type: str = 'duration') -> Pipeline:
        """Entraîne un ensemble de modèles avec optimisation d'hyperparamètres"""
        if not SKLEARN_AVAILABLE:
            return None
            
        # Définition des modèles candidats
        models = {
            'rf': RandomForestRegressor(random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(random_state=42),
            'lr': LinearRegression()
        }
        
        # Grilles de paramètres
        param_grids = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gb': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'lr': {}
        }
        
        best_model = None
        best_score = -np.inf
        
        # Test de chaque modèle
        for name, model in models.items():
            try:
                if param_grids[name]:  # Grid search si paramètres définis
                    grid_search = GridSearchCV(
                        model, param_grids[name], 
                        cv=5, scoring='r2', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X, y)
                    current_model = grid_search.best_estimator_
                    score = grid_search.best_score_
                else:
                    current_model = model
                    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    score = scores.mean()
                    current_model.fit(X, y)
                
                logger.info(f"{model_type.title()} {name}: R² = {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = current_model
                    
            except Exception as e:
                logger.warning(f"Erreur avec {name}: {e}")
        
        return best_model, best_score
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Entraîne les modèles de durée et coût"""
        if not SKLEARN_AVAILABLE:
            logger.error("Impossible d'entraîner: scikit-learn non disponible")
            return self._create_fallback_response()
        
        logger.info(f"Démarrage entraînement avec {len(training_data)} échantillons")
        
        try:
            # Extraction des caractéristiques
            df = self.extract_features(training_data)
            
            # Ajout des targets
            df['duration'] = [task.get('duration', 0) for task in training_data]
            df['cost'] = [task.get('cost', 0) for task in training_data]
            
            # Filtrage des données valides
            valid_mask = (df['duration'] > 0) & (df['cost'] > 0)
            df = df[valid_mask]
            
            logger.info(f"Données valides après filtrage: {len(df)}")
            
            if len(df) < 10:
                raise ValueError("Données d'entraînement insuffisantes")
            
            # Préparation des données
            X, feature_names = self.prepare_training_data(df)
            y_duration = df['duration'].values
            y_cost = df['cost'].values
            
            self.feature_names = feature_names
            
            # Entraînement des modèles
            self.duration_model, duration_score = self.train_model_ensemble(X, y_duration, 'duration')
            self.cost_model, cost_score = self.train_model_ensemble(X, y_cost, 'cost')
            
            # Évaluation finale
            duration_metrics = self._evaluate_model(self.duration_model, X, y_duration)
            cost_metrics = self._evaluate_model(self.cost_model, X, y_cost)
            
            # Sauvegarde des métriques
            self.performance_metrics = {
                'duration': duration_metrics.__dict__,
                'cost': cost_metrics.__dict__
            }
            
            self.training_date = datetime.now().isoformat()
            
            # Sauvegarde du modèle
            self.save_model()
            
            return {
                'status': 'success',
                'training_date': self.training_date,
                'samples_used': len(df),
                'performance_metrics': self.performance_metrics,
                'feature_count': len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Évalue les performances d'un modèle"""
        y_pred = model.predict(X)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        return ModelMetrics(
            mae=mean_absolute_error(y, y_pred),
            mse=mean_squared_error(y, y_pred),
            r2=r2_score(y, y_pred),
            cv_score=cv_scores.mean()
        )
    
    def predict_task_estimates(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Prédit durée et coût pour une tâche"""
        return self.predict_multiple_tasks([task])[0]
    
    def predict_multiple_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prédit durée et coût pour plusieurs tâches"""
        if not self.duration_model or not self.cost_model or not SKLEARN_AVAILABLE:
            return self._fallback_predictions(tasks)
        
        try:
            # Extraction des caractéristiques
            df = self.extract_features(tasks)
            
            # Assurer la présence de toutes les colonnes nécessaires
            # Colonnes numériques manquantes -> 0
            for col in self.numeric_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # Colonnes catégorielles manquantes -> valeur par défaut
            if not hasattr(self, 'categorical_cols'):
                self.categorical_cols = ['task_type', 'priority', 'complexity_level']
                
            for col in self.categorical_cols:
                if col not in df.columns:
                    default_values = {
                        'task_type': 'backend',
                        'priority': 'medium', 
                        'complexity_level': 'medium'
                    }
                    df[col] = default_values.get(col, 'medium')
            
            # Réorganiser dans l'ordre correct
            all_cols = self.numeric_cols + self.categorical_cols
            df_ordered = df[all_cols].copy()
            
            # Preprocessing
            if self.scaler and hasattr(self, 'preprocessor'):
                X = self.preprocessor.transform(df_ordered)
            else:
                # Fallback vers colonnes numériques seulement
                numeric_data = df[self.numeric_cols].fillna(0)
                if self.scaler:
                    X = self.scaler.transform(numeric_data)
                else:
                    X = numeric_data.values
            
            # Prédictions
            duration_pred = self.duration_model.predict(X)
            cost_pred = self.cost_model.predict(X)
            
            # Construction des résultats
            results = []
            for i, task in enumerate(tasks):
                results.append({
                    'task_id': task.get('id', f'task_{i}'),
                    'duration': max(0.5, float(duration_pred[i])),
                    'cost': max(100.0, float(cost_pred[i])),
                    'confidence_duration': 0.8,
                    'confidence_cost': 0.8,
                    'method': 'ml_model'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur de prédiction ML: {e}")
            return self._fallback_predictions(tasks)
    
    def _fallback_predictions(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prédictions heuristiques de secours"""
        results = []
        
        for i, task in enumerate(tasks):
            # Heuristiques simples basées sur la complexité
            complexity = task.get('complexity_level', 'medium')
            priority = task.get('priority', 'medium')
            team_size = task.get('team_size', 1)
            
            base_durations = {
                'simple': 1.5, 'low': 2.0, 'medium': 4.0, 
                'high': 7.0, 'complex': 10.0, 'very_high': 15.0
            }
            
            priority_multipliers = {
                'low': 0.8, 'medium': 1.0, 'high': 1.2, 'critical': 1.5
            }
            
            base_duration = base_durations.get(complexity, 4.0)
            priority_mult = priority_multipliers.get(priority, 1.0)
            
            duration = base_duration * priority_mult
            cost = duration * team_size * 8 * 100  # 8h/jour * 100€/h
            
            results.append({
                'task_id': task.get('id', f'task_{i}'),
                'duration': duration,
                'cost': cost,
                'confidence_duration': 0.6,
                'confidence_cost': 0.6,
                'method': 'heuristic'
            })
        
        return results
    
    def save_model(self) -> str:
        """Sauvegarde le modèle entraîné"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_data = {
            'duration_model': self.duration_model,
            'cost_model': self.cost_model,
            'scaler': self.scaler,
            'preprocessor': getattr(self, 'preprocessor', None),
            'feature_extractor': self.feature_extractor,
            'feature_names': self.feature_names,
            'numeric_cols': getattr(self, 'numeric_cols', []),
            'categorical_cols': getattr(self, 'categorical_cols', []),
            'performance_metrics': self.performance_metrics,
            'training_date': self.training_date,
            'model_version': self.model_version
        }
        
        # Sauvegarde avec timestamp
        timestamped_path = self.model_path / f"estimator_{timestamp}.pkl"
        with open(timestamped_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Sauvegarde latest
        latest_path = self.model_path / "estimator_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modèle sauvegardé: {timestamped_path}")
        return str(timestamped_path)
    
    def load_model(self, model_path: str) -> bool:
        """Charge un modèle pré-entraîné"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.duration_model = model_data.get('duration_model')
            self.cost_model = model_data.get('cost_model')
            self.scaler = model_data.get('scaler')
            self.preprocessor = model_data.get('preprocessor')  # Nouveau
            self.feature_extractor = model_data.get('feature_extractor', AdvancedFeatureExtractor())
            self.feature_names = model_data.get('feature_names', [])
            self.numeric_cols = model_data.get('numeric_cols', [])  # Nouveau
            self.categorical_cols = model_data.get('categorical_cols', [])  # Nouveau
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.training_date = model_data.get('training_date')
            
            logger.info(f"Modèle chargé depuis {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Retourne l'importance des caractéristiques"""
        importance = {}
        
        if hasattr(self.duration_model, 'feature_importances_') and self.feature_names:
            importance['duration_model'] = dict(zip(
                self.feature_names, 
                self.duration_model.feature_importances_
            ))
        
        if hasattr(self.cost_model, 'feature_importances_') and self.feature_names:
            importance['cost_model'] = dict(zip(
                self.feature_names, 
                self.cost_model.feature_importances_
            ))
        
        return importance
    
    def export_predictions_csv(self, tasks: List[Dict[str, Any]], filename: str):
        """Exporte les prédictions vers CSV"""
        predictions = self.predict_multiple_tasks(tasks)
        
        # Combinaison tâches + prédictions
        export_data = []
        for task, pred in zip(tasks, predictions):
            export_data.append({
                'task_id': pred['task_id'],
                'task_name': task.get('name', ''),
                'predicted_duration': pred['duration'],
                'predicted_cost': pred['cost'],
                'confidence_duration': pred['confidence_duration'],
                'confidence_cost': pred['confidence_cost'],
                'method': pred['method']
            })
        
        df = pd.DataFrame(export_data)
        output_path = self.model_path.parent / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Prédictions exportées vers {output_path}")
    
    def _create_fallback_response(self) -> Dict[str, Any]:
        """Crée une réponse de fallback quand ML n'est pas disponible"""
        return {
            'status': 'fallback',
            'training_date': datetime.now().isoformat(),
            'samples_used': 0,
            'performance_metrics': {
                'duration': {'r2': 0.6, 'mae': 2.0},
                'cost': {'r2': 0.6, 'mae': 1000.0}
            },
            'message': 'Utilisation heuristiques de secours'
        }