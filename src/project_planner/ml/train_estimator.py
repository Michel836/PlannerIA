"""
Script d'entraînement final pour les modèles d'estimation PlannerIA
Compatible avec EstimatorModel optimisé
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.project_planner.ml.estimator_model import EstimatorModel
except ImportError as e:
    logger.error(f"Impossible d'importer EstimatorModel: {e}")
    logger.error("Vérifiez que le module est dans le bon répertoire")
    sys.exit(1)


class TrainingDataManager:
    """Gestionnaire des données d'entraînement"""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.training_dir = data_dir / "training"
        self.runs_dir = data_dir / "runs"
        
        # Création des répertoires
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_training_data(self) -> List[Dict[str, Any]]:
        """Charge toutes les données d'entraînement disponibles"""
        all_data = []
        
        # 1. Données historiques des runs
        historical_data = self._load_historical_runs()
        if historical_data:
            logger.info(f"Chargé {len(historical_data)} tâches depuis l'historique des runs")
            all_data.extend(historical_data)
        
        # 2. Données d'entraînement explicites
        training_data = self._load_training_files()
        if training_data:
            logger.info(f"Chargé {len(training_data)} tâches depuis les fichiers d'entraînement")
            all_data.extend(training_data)
        
        # 3. Déduplication basique
        all_data = self._deduplicate_tasks(all_data)
        
        logger.info(f"Total après déduplication: {len(all_data)} tâches")
        return all_data
    
    def _load_historical_runs(self) -> List[Dict[str, Any]]:
        """Charge les tâches depuis l'historique des runs PlannerIA"""
        historical_tasks = []
        
        for run_dir in self.runs_dir.glob("run_*"):
            plan_file = run_dir / "plan.json"
            if plan_file.exists():
                try:
                    with open(plan_file, 'r', encoding='utf-8') as f:
                        plan_data = json.load(f)
                    
                    # Extraction des tâches avec métadonnées du run
                    tasks = plan_data.get('tasks', [])
                    for task in tasks:
                        # Ajout métadonnées contextuelles
                        task['source_run'] = run_dir.name
                        task['project_context'] = {
                            'total_cost': plan_data.get('total_cost', 0),
                            'total_duration': plan_data.get('total_duration', 0),
                            'project_type': plan_data.get('project_type', 'unknown')
                        }
                        historical_tasks.append(task)
                        
                except Exception as e:
                    logger.warning(f"Erreur lecture {plan_file}: {e}")
        
        return historical_tasks
    
    def _load_training_files(self) -> List[Dict[str, Any]]:
        """Charge les fichiers d'entraînement explicites"""
        training_data = []
        
        for file_path in self.training_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Différents formats supportés
                if isinstance(data, list):
                    training_data.extend(data)
                elif isinstance(data, dict):
                    if 'tasks' in data:
                        training_data.extend(data['tasks'])
                    elif 'data' in data:
                        training_data.extend(data['data'])
                    else:
                        training_data.append(data)
                        
            except Exception as e:
                logger.warning(f"Erreur lecture {file_path}: {e}")
        
        return training_data
    
    def _deduplicate_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Déduplique les tâches basé sur nom et description"""
        seen = set()
        unique_tasks = []
        
        for task in tasks:
            # Clé de déduplication
            key = (
                task.get('name', '').strip().lower(),
                task.get('description', '').strip().lower()[:100]  # Premiers 100 chars
            )
            
            if key not in seen:
                seen.add(key)
                unique_tasks.append(task)
        
        return unique_tasks
    
    def generate_synthetic_data(self, num_samples: int = 500) -> List[Dict[str, Any]]:
        """Génère des données synthétiques réalistes"""
        logger.info(f"Génération de {num_samples} échantillons synthétiques")
        
        # Import du générateur depuis le fichier précédent (adapté)
        from .synthetic_generator import EnhancedSyntheticGenerator
        
        generator = EnhancedSyntheticGenerator()
        synthetic_data = generator.generate_realistic_tasks(num_samples)
        
        return synthetic_data
    
    def save_training_dataset(self, data: List[Dict[str, Any]], name: str = None):
        """Sauvegarde le dataset d'entraînement"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_dataset_{timestamp}.json" if name is None else f"{name}_{timestamp}.json"
        
        output_path = self.training_dir / filename
        
        dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(data),
                'version': '1.0'
            },
            'data': data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Dataset sauvegardé: {output_path}")
        return output_path


class ModelTrainingPipeline:
    """Pipeline complet d'entraînement des modèles"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.data_manager = TrainingDataManager()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'min_samples_required': 50,
            'synthetic_samples': 1000,
            'test_split_ratio': 0.2,
            'enable_feature_selection': True,
            'save_predictions_sample': True
        }
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Exécute le pipeline complet d'entraînement"""
        logger.info("=== DÉMARRAGE PIPELINE D'ENTRAÎNEMENT ===")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'steps_completed': []
        }
        
        try:
            # Étape 1: Chargement des données
            logger.info("Étape 1: Chargement des données")
            all_data = self.data_manager.load_all_training_data()
            pipeline_results['data_loaded'] = len(all_data)
            pipeline_results['steps_completed'].append('data_loading')
            
            # Étape 2: Augmentation avec données synthétiques si nécessaire
            if len(all_data) < self.config['min_samples_required']:
                logger.info("Étape 2: Génération de données synthétiques")
                synthetic_data = self.data_manager.generate_synthetic_data(
                    self.config['synthetic_samples']
                )
                all_data.extend(synthetic_data)
                pipeline_results['synthetic_generated'] = len(synthetic_data)
            
            pipeline_results['total_samples'] = len(all_data)
            pipeline_results['steps_completed'].append('data_augmentation')
            
            # Étape 3: Sauvegarde du dataset
            logger.info("Étape 3: Sauvegarde du dataset")
            self.data_manager.save_training_dataset(all_data, 'pipeline_run')
            pipeline_results['steps_completed'].append('data_saving')
            
            # Étape 4: Entraînement du modèle
            logger.info("Étape 4: Entraînement des modèles ML")
            model = EstimatorModel()
            training_results = model.train(all_data)
            pipeline_results['model_training'] = training_results
            pipeline_results['steps_completed'].append('model_training')
            
            # Étape 5: Validation et tests
            logger.info("Étape 5: Validation du modèle")
            validation_results = self._validate_model(model, all_data)
            pipeline_results['validation'] = validation_results
            pipeline_results['steps_completed'].append('model_validation')
            
            # Étape 6: Export d'exemples de prédictions
            if self.config['save_predictions_sample']:
                logger.info("Étape 6: Export d'exemples de prédictions")
                sample_tasks = all_data[:20]  # Premier 20 tâches
                model.export_predictions_csv(sample_tasks, "training_predictions_sample.csv")
                pipeline_results['steps_completed'].append('predictions_export')
            
            # Finalisation
            pipeline_results['status'] = 'completed'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            logger.info("=== PIPELINE TERMINÉ AVEC SUCCÈS ===")
            self._print_summary(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results['status'] = 'error'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            logger.error(f"Erreur dans le pipeline: {e}")
            raise
    
    def _validate_model(self, model: EstimatorModel, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Valide le modèle entraîné"""
        validation_results = {}
        
        try:
            # Test sur échantillon
            sample_size = min(50, len(data) // 4)
            sample_data = data[-sample_size:]  # Dernières tâches pour test
            
            predictions = model.predict_multiple_tasks(sample_data)
            
            # Calcul d'erreurs simples
            duration_errors = []
            cost_errors = []
            
            for pred, actual in zip(predictions, sample_data):
                if actual.get('duration', 0) > 0:
                    error = abs(pred['duration'] - actual['duration']) / actual['duration']
                    duration_errors.append(error)
                
                if actual.get('cost', 0) > 0:
                    error = abs(pred['cost'] - actual['cost']) / actual['cost']
                    cost_errors.append(error)
            
            validation_results = {
                'sample_size': len(predictions),
                'duration_mean_error': sum(duration_errors) / len(duration_errors) if duration_errors else 0,
                'cost_mean_error': sum(cost_errors) / len(cost_errors) if cost_errors else 0,
                'predictions_successful': len(predictions)
            }
            
        except Exception as e:
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Affiche un résumé des résultats"""
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'ENTRAÎNEMENT")
        print("="*60)
        
        print(f"Statut: {results['status'].upper()}")
        print(f"Échantillons utilisés: {results.get('total_samples', 'N/A')}")
        
        if 'model_training' in results:
            training = results['model_training']
            if 'performance_metrics' in training:
                metrics = training['performance_metrics']
                if 'duration' in metrics:
                    print(f"R² Durée: {metrics['duration'].get('r2', 'N/A'):.3f}")
                if 'cost' in metrics:
                    print(f"R² Coût: {metrics['cost'].get('r2', 'N/A'):.3f}")
        
        if 'validation' in results:
            val = results['validation']
            if 'duration_mean_error' in val:
                print(f"Erreur moyenne durée: {val['duration_mean_error']:.1%}")
            if 'cost_mean_error' in val:
                print(f"Erreur moyenne coût: {val['cost_mean_error']:.1%}")
        
        print(f"Étapes complétées: {len(results.get('steps_completed', []))}")
        print("="*60)


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Entraînement des modèles PlannerIA')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Nombre d\'échantillons synthétiques à générer')
    parser.add_argument('--min-samples', type=int, default=50,
                       help='Nombre minimum d\'échantillons requis')
    parser.add_argument('--config', type=str,
                       help='Chemin vers fichier de configuration JSON')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'synthetic_samples': args.samples,
        'min_samples_required': args.min_samples,
        'test_split_ratio': 0.2,
        'save_predictions_sample': True
    }
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Exécution du pipeline
    try:
        pipeline = ModelTrainingPipeline(config)
        results = pipeline.run_full_pipeline()
        
        # Sauvegarde des résultats
        results_file = Path("data/training") / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nRésultats sauvegardés: {results_file}")
        
        return 0 if results['status'] == 'completed' else 1
        
    except KeyboardInterrupt:
        logger.info("Entraînement interrompu par l'utilisateur")
        return 1
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Création des répertoires nécessaires
    Path("logs").mkdir(exist_ok=True)
    Path("data/training").mkdir(parents=True, exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    sys.exit(main())