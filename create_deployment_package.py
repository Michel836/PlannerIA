#!/usr/bin/env python3
"""
Script de création d'un package de déploiement PlannerIA
Crée un ZIP optimisé avec tout le nécessaire pour installation
"""

import os
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

def create_deployment_zip():
    """Crée un package ZIP de déploiement complet"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"PlannerIA_Deployment_{timestamp}.zip"
    
    print(f"Creation du package de deploiement: {zip_filename}")
    
    # Fichiers et dossiers essentiels à inclure
    essential_files = [
        "requirements.txt",
        "README.md", 
        "crew.py",
        ".env.example",
        ".gitignore"
    ]
    
    essential_dirs = [
        "src/project_planner/",
        "config/",
        "schema/", 
        "tests/",
        "data/models/",  # Modèles ML pré-entraînés
        "data/examples/",
    ]
    
    # Créer les dossiers de données vides nécessaires
    required_empty_dirs = [
        "data/runs/",
        "data/reports/", 
        "data/feedback/",
        "data/training/",
        "logs/"
    ]
    
    # Scripts d'installation à inclure
    install_scripts = [
        "install_plannerai_updated.bat",
        "install_plannerai.bat"
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Ajouter les fichiers essentiels
        for file in essential_files:
            if Path(file).exists():
                zipf.write(file, file)
                print(f"Ajoute: {file}")
        
        # Ajouter les dossiers essentiels
        for dir_path in essential_dirs:
            if Path(dir_path).exists():
                for root, dirs, files in os.walk(dir_path):
                    # Exclure les caches et fichiers temporaires
                    dirs[:] = [d for d in dirs if not d.startswith(('.', '__pycache__'))]
                    
                    for file in files:
                        if not file.endswith(('.pyc', '.pyo', '.log')):
                            file_path = Path(root) / file
                            arcname = str(file_path).replace('\\', '/')
                            zipf.write(file_path, arcname)
                
                print(f"Ajoute dossier: {dir_path}")
        
        # Créer les dossiers vides nécessaires
        for empty_dir in required_empty_dirs:
            Path(empty_dir).mkdir(parents=True, exist_ok=True)
            # Ajouter un fichier .gitkeep pour conserver le dossier
            gitkeep_path = Path(empty_dir) / ".gitkeep"
            gitkeep_path.touch()
            zipf.write(gitkeep_path, str(gitkeep_path).replace('\\', '/'))
            print(f"Cree dossier vide: {empty_dir}")
        
        # Ajouter les scripts d'installation s'ils existent
        for script in install_scripts:
            if Path(script).exists():
                zipf.write(script, script)
                print(f"Ajoute script: {script}")
        
        # Ajouter le guide d'installation
        install_guide = create_installation_guide()
        with open("INSTALLATION_GUIDE.md", "w", encoding="utf-8") as f:
            f.write(install_guide)
        zipf.write("INSTALLATION_GUIDE.md", "INSTALLATION_GUIDE.md")
        print("Guide d'installation cree")
        
        # Ajouter le script de démarrage rapide
        quick_start = create_quick_start_script()
        with open("quick_start.bat", "w", encoding="utf-8") as f:
            f.write(quick_start)
        zipf.write("quick_start.bat", "quick_start.bat")
        print("Script demarrage rapide cree")
    
    # Nettoyer les fichiers temporaires
    for temp_file in ["INSTALLATION_GUIDE.md", "quick_start.bat"]:
        if Path(temp_file).exists():
            Path(temp_file).unlink()
    
    for empty_dir in required_empty_dirs:
        gitkeep_path = Path(empty_dir) / ".gitkeep"
        if gitkeep_path.exists():
            gitkeep_path.unlink()
    
    file_size = Path(zip_filename).stat().st_size / (1024*1024)
    print(f"Package cree: {zip_filename} ({file_size:.1f} MB)")
    
    return zip_filename

def create_installation_guide():
    """Crée le guide d'installation complet"""
    
    guide = """# 🚀 Guide d'Installation PlannerIA

## 📋 Prérequis Système

### Logiciels Requis
- **Python 3.11** ou supérieur
- **Git** (optionnel, pour les mises à jour)
- **Ollama** (pour la génération de plans IA)

### Système d'Exploitation
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux Ubuntu 18.04+

---

## ⚡ Installation Rapide (Windows)

### 1. Extraction du Package
```bash
# Extraire PlannerIA_Deployment_XXXXXXXX_XXXXXX.zip
# dans le dossier de votre choix (ex: C:\PlannerIA)
```

### 2. Installation Automatique
```bash
# Double-cliquer sur quick_start.bat
# OU exécuter dans un terminal :
quick_start.bat
```

### 3. Démarrage
```bash
# Le dashboard s'ouvre automatiquement sur :
http://localhost:8521
```

---

## 🔧 Installation Manuelle Détaillée

### Étape 1: Installation Python
1. Télécharger Python 3.11+ depuis https://python.org
2. ⚠️ IMPORTANT: Cocher "Add Python to PATH" lors de l'installation
3. Vérifier l'installation:
```bash
python --version
pip --version
```

### Étape 2: Installation Ollama (pour IA complète)
1. Télécharger Ollama depuis https://ollama.ai
2. Installer et redémarrer l'ordinateur
3. Ouvrir un terminal et exécuter:
```bash
ollama pull llama3.2:latest
ollama serve
```
4. ⚠️ Laisser le terminal Ollama ouvert

### Étape 3: Installation PlannerIA
```bash
# Naviguer vers le dossier extrait
cd C:\chemin\vers\PlannerIA

# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521
```

---

## 🎯 Modes de Fonctionnement

### Mode Dashboard (Recommandé)
✅ **Fonctionne TOUJOURS sans prérequis**
```bash
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521
```
**Accès**: http://localhost:8521

**Fonctionnalités disponibles**:
- 11 modules IA opérationnels
- Visualisations interactives
- Rapports PDF/CSV
- Système vocal
- Analytics temps réel

### Mode Génération IA (Nécessite Ollama)
⚠️ **Requiert Ollama actif**
```bash
# Terminal 1: Ollama
ollama serve

# Terminal 2: PlannerIA CLI
python crew.py "Votre brief de projet"
```

### Mode API (Nécessite Ollama)
⚠️ **Requiert Ollama actif**
```bash
# Terminal 1: Ollama  
ollama serve

# Terminal 2: API
python -m uvicorn src.project_planner.api.main:app --reload
```
**Accès**: http://localhost:8000

---

## 🧪 Tests de Fonctionnement

### Test Rapide Dashboard
```bash
# Lancer le dashboard
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521

# Vérifier dans les logs:
# "11/11 modules opérationnels" = ✅ Succès
```

### Test Modèles ML
```bash
python -c "
from src.project_planner.ml.estimator_model import EstimatorModel
em = EstimatorModel()
print('✅ Modèles ML fonctionnels')
"
```

### Test Complet
```bash
python test_ml_complete.py
```

---

## 🔧 Dépannage

### Erreur "Module not found"
```bash
# Solution:
pip install -r requirements.txt
```

### Erreur Ollama "Connection refused"
```bash
# Solution:
ollama serve
# Puis relancer PlannerIA
```

### Erreur Streamlit
```bash
# Solution:
pip install streamlit --upgrade
```

### Port 8521 occupé
```bash
# Solution: utiliser un autre port
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8522
```

---

## 📁 Structure Post-Installation

```
PlannerIA/
├── quick_start.bat           # Script démarrage rapide
├── requirements.txt          # Dépendances Python
├── crew.py                   # CLI génération plans
├── src/                      # Code source
│   └── project_planner/      # Modules principaux
├── data/                     # Données et modèles
│   ├── models/              # Modèles ML pré-entraînés
│   ├── runs/                # Plans générés (créé auto)
│   └── reports/             # Rapports exportés (créé auto)
├── config/                   # Configurations
└── logs/                     # Logs système (créé auto)
```

---

## 🚀 Commandes Essentielles

### Démarrage Dashboard
```bash
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521
```

### Génération Plan (avec Ollama)
```bash
python crew.py "Développer une application mobile de gestion de tâches"
```

### Tests Performance
```bash
python benchmark_complete.py
```

### Mise à Jour
```bash
git pull origin main  # Si installé via Git
pip install -r requirements.txt --upgrade
```

---

## 📞 Support

- 🐛 **Issues**: https://github.com/Michel836/PlannerIA/issues
- 📚 **Documentation**: Voir README.md
- 💡 **Suggestions**: Pull requests bienvenues

---

## ✅ Checklist Installation Réussie

- [ ] Python 3.11+ installé et dans le PATH
- [ ] Dépendances pip installées sans erreur
- [ ] Dashboard accessible sur http://localhost:8521  
- [ ] Message "11/11 modules opérationnels" visible
- [ ] Exports PDF/CSV fonctionnels
- [ ] (Optionnel) Ollama installé et opérationnel
- [ ] (Optionnel) API accessible sur http://localhost:8000

**Installation réussie = Dashboard fonctionnel !**

---

*Version: 1.0.0-deployment | Date: $(date '+%Y-%m-%d')*
"""
    
    return guide

def create_quick_start_script():
    """Crée le script de démarrage rapide Windows"""
    
    script = """@echo off
echo ==========================================
echo    PlannerIA - Demarrage Rapide
echo ==========================================
echo.

REM Verification Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    echo Veuillez installer Python 3.11+ depuis https://python.org
    echo IMPORTANT: Cocher "Add Python to PATH" lors de l'installation
    pause
    exit /b 1
)

echo ✅ Python detecte
echo.

REM Installation des dependances
echo 📦 Installation des dependances...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERREUR: Installation des dependances echouee
    pause
    exit /b 1
)

echo ✅ Dependances installees
echo.

REM Creation des dossiers necessaires
if not exist "data\\runs" mkdir "data\\runs"
if not exist "data\\reports" mkdir "data\\reports" 
if not exist "logs" mkdir "logs"

echo ✅ Dossiers crees
echo.

REM Test rapide des modeles ML
echo 🧠 Test des modeles ML...
python -c "from src.project_planner.ml.estimator_model import EstimatorModel; EstimatorModel(); print('✅ Modeles ML operationnels')" 2>nul
if errorlevel 1 (
    echo ⚠️  Modeles ML non charges (normal au premier lancement)
) else (
    echo ✅ Modeles ML operationnels
)

echo.
echo ==========================================
echo    LANCEMENT DU DASHBOARD
echo ==========================================
echo.
echo 🚀 Lancement du dashboard PlannerIA...
echo 📱 Le dashboard va s'ouvrir sur: http://localhost:8521
echo.
echo ⏹️  Pour arreter: Ctrl+C dans ce terminal
echo ℹ️  Si Ollama est installe, vous aurez toutes les fonctionnalites IA
echo ℹ️  Sinon, le dashboard reste entierement fonctionnel
echo.

REM Lancer Streamlit
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521

echo.
echo Merci d'avoir utilise PlannerIA !
pause
"""
    
    return script

if __name__ == "__main__":
    zip_file = create_deployment_zip()
    print(f"\nPackage de deploiement pret: {zip_file}")
    print("\nContenu du package:")
    print("  Code source complet")
    print("  Modeles ML pre-entraines") 
    print("  Guide d'installation detaille")
    print("  Script de demarrage rapide")
    print("  Dossiers de donnees pre-configures")
    print(f"\nPret a deployer sur GitHub !")