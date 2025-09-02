# 📦 Guide de Transfert PlannerIA

## 🎯 Package de Déploiement Complet

### Fichier Principal
- **`PlannerIA_Deployment_20250902_XXXXXX.zip`** (2.9 MB)
- ✅ **Code source complet** avec tous les modules
- ✅ **Modèles ML pré-entraînés** prêts à l'emploi
- ✅ **Guide d'installation automatique** (INSTALLATION_GUIDE.md)
- ✅ **Script de démarrage rapide** (quick_start.bat)

---

## 🚀 Options de Transfert

### Option 1: Téléchargement Direct (Recommandé)
```bash
# Sur le nouveau système :
1. Télécharger PlannerIA_Deployment_XXXXXX.zip depuis GitHub
2. Extraire dans un dossier (ex: C:\PlannerIA)  
3. Double-cliquer sur quick_start.bat
4. ✅ PlannerIA est opérationnel !
```

### Option 2: Clone Git + ZIP
```bash
# Cloner le repository
git clone https://github.com/Michel836/PlannerIA.git
cd PlannerIA

# Télécharger le ZIP de déploiement pour les modèles ML
# Extraire uniquement le dossier data/models/ du ZIP
```

### Option 3: USB/Clé
```bash
# Copier PlannerIA_Deployment_XXXXXX.zip sur clé USB
# Sur le nouveau système :
1. Copier le ZIP depuis la clé
2. Extraire et suivre les instructions
```

---

## ⚡ Installation Automatique

### Windows (Recommandé)
```bash
# Après extraction du ZIP :
quick_start.bat
# Le script fait TOUT automatiquement :
# - Vérifie Python
# - Installe les dépendances  
# - Lance le dashboard
```

### Manuel (Tous OS)
```bash
# Prérequis : Python 3.11+
pip install -r requirements.txt

# Lancer le dashboard
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521

# Accès : http://localhost:8521
```

---

## 📋 Contenu du Package

### Fichiers Principaux
- `quick_start.bat` - Script de lancement automatique
- `INSTALLATION_GUIDE.md` - Guide complet 150+ lignes
- `requirements.txt` - Dépendances Python
- `crew.py` - CLI de génération de plans
- `README.md` - Documentation technique complète

### Code Source
- `src/project_planner/` - Modules principaux
  - `dashboard/mvp_v1.py` - Interface Streamlit
  - `ml/` - Modèles ML pré-entraînés
  - `ai/` - 11 modules IA opérationnels
  - `reports/` - Génération PDF/CSV
  - `voice/` - Système vocal complet

### Données Pré-configurées
- `data/models/` - Modèles ML entraînés (EstimatorModel, RiskModel)
- `data/runs/` - Dossier pour plans générés
- `data/reports/` - Dossier pour exports
- `config/` - Configurations par défaut

---

## ✅ Tests de Validation

### Test Immédiat (30 secondes)
```bash
# Après installation :
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521

# Vérifier :
# 1. Dashboard s'ouvre sur http://localhost:8521
# 2. Message "11/11 modules opérationnels" dans les logs  
# 3. Visualisations Gantt/Sunburst s'affichent
# 4. Exports PDF/CSV fonctionnent
```

### Test Avancé (avec Ollama)
```bash
# Si Ollama installé :
ollama serve

# Dans un autre terminal :
python crew.py "Développer une app mobile"
# ✅ Plan généré automatiquement
```

---

## 🔧 Configuration Post-Installation

### Fonctionnel Immédiatement
- ✅ Dashboard Streamlit complet
- ✅ 11 modules IA opérationnels
- ✅ Modèles ML pré-entraînés
- ✅ Système vocal (TTS + reconnaissance)
- ✅ Rapports PDF/CSV professionnels
- ✅ Analytics temps réel

### Optionnel (pour IA complète)
- Ollama + Llama3.2 (génération de plans)
- API FastAPI (endpoints REST)

---

## 📊 Tailles et Performances

### Package
- **Taille** : 2.9 MB (optimisé)
- **Installation** : 2-5 minutes
- **Démarrage** : 10-30 secondes

### Ressources Système
- **RAM** : 500MB minimum (dashboard)
- **Stockage** : 50MB après installation
- **CPU** : Tout processeur moderne
- **OS** : Windows 10+, macOS 10.15+, Linux Ubuntu 18+

---

## 🎯 Modes d'Utilisation

### Mode 1: Dashboard Standalone ✅
```bash
# Fonctionne TOUJOURS, zero configuration
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521
```
**Fonctionnalités** : Visualisations, Analytics, Rapports, IA, Voice

### Mode 2: Génération IA ⚠️ 
```bash  
# Nécessite Ollama
ollama serve
python crew.py "Brief projet"
```
**Fonctionnalités** : + Génération automatique de plans

### Mode 3: API REST ⚠️
```bash
# Nécessite Ollama 
python -m uvicorn src.project_planner.api.main:app --reload
```
**Fonctionnalités** : + Endpoints API complets

---

## 🆘 Dépannage Rapide

### Erreur "Python not found"
```bash
# Solution :
# Installer Python 3.11+ depuis https://python.org
# ✅ COCHER "Add Python to PATH"
```

### Erreur "Module not found"  
```bash
# Solution :
pip install -r requirements.txt
```

### Port 8521 occupé
```bash
# Solution :
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8522
```

### Dashboard ne démarre pas
```bash
# Solution :
# 1. Vérifier Python 3.11+
# 2. pip install streamlit --upgrade  
# 3. Vérifier les logs d'erreur
```

---

## 📞 Support et Ressources

- 📚 **Guide complet** : INSTALLATION_GUIDE.md (dans le ZIP)
- 🐛 **Issues GitHub** : https://github.com/Michel836/PlannerIA/issues
- 📖 **Documentation** : README.md (technique détaillé)
- ⚡ **Démarrage rapide** : quick_start.bat (automatique)

---

## 🎉 Checklist Installation Réussie

- [ ] ZIP extrait dans un dossier dédié
- [ ] Python 3.11+ installé et accessible  
- [ ] `pip install -r requirements.txt` exécuté avec succès
- [ ] Dashboard accessible via http://localhost:8521
- [ ] Message "11/11 modules opérationnels" visible
- [ ] Visualisations s'affichent correctement
- [ ] Exports PDF/CSV fonctionnent
- [ ] (Optionnel) Ollama installé pour génération IA

**✅ Installation réussie = Dashboard opérationnel en moins de 5 minutes !**

---

*Package créé le : 2 septembre 2025*  
*Version : 1.0.0-deployment*  
*Taille optimisée : 2.9 MB*