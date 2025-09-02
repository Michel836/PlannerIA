"""
Test complet du système ML PlannerIA
Teste tous les modèles, génération de données et intégration
"""

import sys
import os
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# Ajouter le chemin ML au PYTHONPATH
ml_path = Path(__file__).parent / "ml"
sys.path.insert(0, str(ml_path))

def test_imports():
    """Test des imports de tous les modules ML"""
    print("=" * 60)
    print("TEST 1: IMPORTS DES MODULES ML")
    print("=" * 60)
    
    modules_to_test = [
        'estimator_model',
        'risk_model', 
        'synthetic_generator',
        'train_estimator',
        'train_risk',
        'test_ml_system'
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✓ {module_name}: OK")
            results[module_name] = "SUCCESS"
        except ImportError as e:
            print(f"✗ {module_name}: ERREUR - {e}")
            results[module_name] = f"IMPORT_ERROR: {e}"
        except Exception as e:
            print(f"⚠ {module_name}: ERREUR GÉNÉRALE - {e}")
            results[module_name] = f"ERROR: {e}"
    
    return results

def test_estimator_model():
    """Test du modèle d'estimation"""
    print("\n" + "=" * 60)
    print("TEST 2: MODÈLE D'ESTIMATION")
    print("=" * 60)
    
    try:
        from estimator_model import EstimatorModel
        
        print("Initialisation du modèle d'estimation...")
        estimator = EstimatorModel()
        
        # Test avec données d'exemple
        test_data = {
            'complexity': 7,
            'team_size': 5,
            'experience_level': 3,
            'similar_projects': 2,
            'requirements_clarity': 8
        }
        
        print(f"Test avec données: {test_data}")
        
        # Test de prédiction
        prediction = estimator.predict(test_data)
        print(f"✓ Prédiction: {prediction}")
        
        # Test de confiance si disponible
        if hasattr(estimator, 'predict_with_confidence'):
            prediction, confidence = estimator.predict_with_confidence(test_data)
            print(f"✓ Prédiction avec confiance: {prediction} (confiance: {confidence})")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"✗ Erreur dans le modèle d'estimation: {e}")
        traceback.print_exc()
        return f"ERROR: {e}"

def test_risk_model():
    """Test du modèle de risques"""
    print("\n" + "=" * 60)
    print("TEST 3: MODÈLE DE RISQUES")
    print("=" * 60)
    
    try:
        from risk_model import RiskModel
        
        print("Initialisation du modèle de risques...")
        risk_model = RiskModel()
        
        # Test avec données de projet
        project_data = {
            'budget_variance': 0.15,
            'schedule_variance': -0.08,
            'team_turnover': 0.12,
            'complexity_score': 7.5,
            'stakeholder_involvement': 6
        }
        
        print(f"Test avec données projet: {project_data}")
        
        # Test d'évaluation des risques
        risk_assessment = risk_model.assess_risk(project_data)
        print(f"✓ Évaluation des risques: {risk_assessment}")
        
        # Test des risques par catégorie si disponible
        if hasattr(risk_model, 'categorize_risks'):
            risk_categories = risk_model.categorize_risks(project_data)
            print(f"✓ Risques par catégorie: {risk_categories}")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"✗ Erreur dans le modèle de risques: {e}")
        traceback.print_exc()
        return f"ERROR: {e}"

def test_synthetic_generator():
    """Test du générateur de données synthétiques"""
    print("\n" + "=" * 60)
    print("TEST 4: GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES")
    print("=" * 60)
    
    try:
        from synthetic_generator import SyntheticGenerator
        
        print("Initialisation du générateur...")
        generator = SyntheticGenerator()
        
        # Test de génération de projets
        print("Génération de projets synthétiques...")
        synthetic_projects = generator.generate_projects(n_projects=5)
        print(f"✓ {len(synthetic_projects)} projets générés")
        
        # Afficher un exemple
        if synthetic_projects:
            print(f"✓ Exemple de projet: {synthetic_projects[0]}")
        
        # Test de génération de tâches si disponible
        if hasattr(generator, 'generate_tasks'):
            synthetic_tasks = generator.generate_tasks(n_tasks=10)
            print(f"✓ {len(synthetic_tasks)} tâches générées")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"✗ Erreur dans le générateur: {e}")
        traceback.print_exc()
        return f"ERROR: {e}"

def test_model_files():
    """Test de la présence des fichiers de modèles entraînés"""
    print("\n" + "=" * 60)
    print("TEST 5: FICHIERS DE MODÈLES ENTRAÎNÉS")
    print("=" * 60)
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("✗ Dossier 'models' non trouvé")
        return "NO_MODELS_DIR"
    
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        print("⚠ Aucun fichier .pkl trouvé dans le dossier models")
        return "NO_MODELS"
    
    print(f"✓ {len(model_files)} fichiers de modèles trouvés:")
    for model_file in model_files:
        file_size = model_file.stat().st_size / 1024  # KB
        print(f"  - {model_file.name} ({file_size:.1f} KB)")
    
    # Test de chargement d'un modèle
    try:
        import pickle
        latest_model = model_files[0]
        with open(latest_model, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Modèle {latest_model.name} chargé avec succès")
        return "SUCCESS"
    except Exception as e:
        print(f"✗ Erreur de chargement du modèle: {e}")
        return f"LOAD_ERROR: {e}"

def test_training_scripts():
    """Test des scripts d'entraînement"""
    print("\n" + "=" * 60)
    print("TEST 6: SCRIPTS D'ENTRAÎNEMENT")
    print("=" * 60)
    
    results = {}
    
    # Test train_estimator
    try:
        from train_estimator import main as train_estimator_main
        print("✓ Script train_estimator importé")
        results['train_estimator'] = "IMPORT_OK"
    except Exception as e:
        print(f"✗ Erreur import train_estimator: {e}")
        results['train_estimator'] = f"IMPORT_ERROR: {e}"
    
    # Test train_risk
    try:
        from train_risk import main as train_risk_main
        print("✓ Script train_risk importé")
        results['train_risk'] = "IMPORT_OK"
    except Exception as e:
        print(f"✗ Erreur import train_risk: {e}")
        results['train_risk'] = f"IMPORT_ERROR: {e}"
    
    return results

def test_data_flow():
    """Test du flux de données complet"""
    print("\n" + "=" * 60)
    print("TEST 7: FLUX DE DONNÉES COMPLET")
    print("=" * 60)
    
    try:
        # 1. Générer des données synthétiques
        from synthetic_generator import SyntheticGenerator
        generator = SyntheticGenerator()
        project_data = generator.generate_projects(n_projects=1)[0]
        print("✓ Données synthétiques générées")
        
        # 2. Tester l'estimation
        from estimator_model import EstimatorModel
        estimator = EstimatorModel()
        estimation = estimator.predict(project_data)
        print(f"✓ Estimation calculée: {estimation}")
        
        # 3. Tester l'évaluation des risques
        from risk_model import RiskModel
        risk_model = RiskModel()
        risk_score = risk_model.assess_risk(project_data)
        print(f"✓ Risque évalué: {risk_score}")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"✗ Erreur dans le flux de données: {e}")
        traceback.print_exc()
        return f"ERROR: {e}"

def test_performance():
    """Test de performance des modèles"""
    print("\n" + "=" * 60)
    print("TEST 8: PERFORMANCE DES MODÈLES")
    print("=" * 60)
    
    try:
        import time
        
        # Test de performance de l'estimateur
        from estimator_model import EstimatorModel
        estimator = EstimatorModel()
        
        test_data = {'complexity': 5, 'team_size': 4}
        
        # Test de vitesse
        start_time = time.time()
        for _ in range(100):
            estimator.predict(test_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # en ms
        print(f"✓ Performance estimateur: {avg_time:.2f} ms par prédiction")
        
        if avg_time < 50:  # Moins de 50ms
            print("✓ Performance excellente")
        elif avg_time < 200:
            print("⚠ Performance correcte")
        else:
            print("✗ Performance lente")
        
        return "SUCCESS"
        
    except Exception as e:
        print(f"✗ Erreur test performance: {e}")
        return f"ERROR: {e}"

def run_all_tests():
    """Lance tous les tests"""
    print("DÉBUT DES TESTS ML PLANNERIA")
    print("=" * 80)
    
    test_results = {}
    
    # Lancer tous les tests
    test_results['imports'] = test_imports()
    test_results['estimator'] = test_estimator_model()
    test_results['risk_model'] = test_risk_model()
    test_results['synthetic'] = test_synthetic_generator()
    test_results['model_files'] = test_model_files()
    test_results['training_scripts'] = test_training_scripts()
    test_results['data_flow'] = test_data_flow()
    test_results['performance'] = test_performance()
    
    # Résumé final
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES TESTS")
    print("=" * 80)
    
    success_count = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            # Pour les tests avec sous-résultats
            sub_success = sum(1 for v in result.values() if 'SUCCESS' in str(v))
            sub_total = len(result)
            print(f"{test_name:20}: {sub_success}/{sub_total} sous-tests réussis")
            if sub_success == sub_total:
                success_count += 1
        else:
            status = "✓ RÉUSSI" if "SUCCESS" in str(result) else "✗ ÉCHOUÉ"
            print(f"{test_name:20}: {status}")
            if "SUCCESS" in str(result):
                success_count += 1
    
    print("\n" + "=" * 80)
    print(f"RÉSULTAT GLOBAL: {success_count}/{total_tests} tests réussis")
    
    if success_count == total_tests:
        print("🎉 TOUS LES TESTS SONT PASSÉS! Le système ML est opérationnel.")
    elif success_count >= total_tests * 0.8:
        print("⚠ La plupart des tests passent. Quelques ajustements nécessaires.")
    else:
        print("❌ Plusieurs tests échouent. Vérification approfondie requise.")
    
    return test_results

if __name__ == "__main__":
    # Changer vers le répertoire du projet
    try:
        os.chdir(Path(__file__).parent)
        print(f"Répertoire de travail: {os.getcwd()}")
    except Exception as e:
        print(f"Erreur changement répertoire: {e}")
    
    # Lancer tous les tests
    results = run_all_tests()