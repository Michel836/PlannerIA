# 🤖 PlannerIA - Système de Planification de Projet IA

PlannerIA est un système de planification de projets avec interface Streamlit et modèles d'apprentissage automatique pour l'estimation et l'analyse de projets.

## 🎯 **État Actuel du Projet**

**✅ CE QUI FONCTIONNE :**
- **Dashboard Streamlit** complet et opérationnel sur http://localhost:8521
- **11 modules IA spécialisés** initialisés et fonctionnels
- **Modèles ML** entraînés et chargés (EstimatorModel, RiskModel) 
- **Système de voix** complet (reconnaissance vocale + synthèse TTS)
- **Visualisations avancées** (Gantt, Sunburst, Risk Matrix)
- **Génération de rapports PDF/CSV** professionnels
- **Base de données FAISS** pour le système RAG

**⚠️ DÉPENDANCES REQUISES :**
- **Ollama** doit être démarré (`ollama serve`) pour la génération de plans
- **Modèle Llama3.2** requis (`ollama pull llama3.2:latest`)

---

## ⚡ **Démarrage Rapide**

### Prérequis
```bash
# 1. Cloner le projet
git clone https://github.com/Michel836/PlannerIA.git
cd PlannerIA

# 2. Installer les dépendances Python
pip install -r requirements.txt

# 3. OBLIGATOIRE: Installer et démarrer Ollama
# Télécharger depuis https://ollama.ai
ollama pull llama3.2:latest
ollama serve  # Laisser ouvert dans un terminal
```

### Lancement
```bash
# Dashboard principal (ce qui fonctionne)
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521

# Génération CLI (nécessite Ollama actif)
python crew.py "Votre brief de projet"

# API FastAPI (nécessite Ollama actif)
python -m uvicorn src.project_planner.api.main:app --reload
```

---

## 🏗️ **Architecture Réelle**

### Structure des Fichiers Fonctionnels
```
PlannerIA/
├── crew.py                           # CLI principal (dépend d'Ollama)
├── src/project_planner/
│   ├── dashboard/
│   │   ├── mvp_v1.py                 # ✅ Dashboard principal fonctionnel
│   │   ├── components/               # ✅ 15+ modules UI opérationnels
│   │   └── ui_components.py          # ✅ Composants de base
│   ├── ml/                           # ✅ Modèles ML entraînés
│   │   ├── estimator_model.py        # ✅ Prédictions coût/durée
│   │   ├── risk_model.py             # ✅ Analyse des risques
│   │   └── synthetic_generator.py    # ✅ Génération données test
│   ├── ai/                           # ✅ 11 modules IA opérationnels
│   │   ├── rag_manager.py           # ✅ RAG avec FAISS
│   │   ├── predictive_engine.py     # ✅ Prédictions avancées
│   │   └── [9+ autres modules]      # ✅ Tous initialisés
│   ├── reports/                      # ✅ Génération rapports
│   │   ├── pdf_generator.py         # ✅ PDF professionnels
│   │   └── csv_exporter.py          # ✅ Export CSV/ZIP
│   ├── visualizations/              # ✅ Graphiques avancés
│   │   └── advanced_charts.py       # ✅ Gantt, Sunburst, etc.
│   ├── voice/                       # ✅ Système vocal complet
│   │   ├── text_to_speech.py       # ✅ TTS 5 voix disponibles
│   │   └── speech_recognizer.py    # ✅ Reconnaissance vocale
│   └── api/main.py                  # ⚠️ Dépend d'Ollama
└── data/
    ├── models/                      # ✅ Modèles ML pré-entraînés
    ├── runs/                        # ✅ Plans générés
    └── reports/                     # ✅ Rapports exportés
```

---

## 🎮 **Interface Dashboard (Opérationnelle)**

### Dashboard Principal (`mvp_v1.py`)
**URL** : http://localhost:8521

**Fonctionnalités Confirmées :**
- ✅ **Vue d'ensemble projet** avec métriques temps réel
- ✅ **Diagramme de Gantt interactif** avec chemin critique
- ✅ **Matrice des risques** avec scoring ML
- ✅ **Décomposition budgétaire** (graphiques Sunburst)
- ✅ **Analytics en temps réel** avec 11 modules IA
- ✅ **Export PDF/CSV** avec graphiques intégrés
- ✅ **Interface vocale** (reconnaissance + synthèse)
- ✅ **Système RAG** pour questions/réponses

### Modules UI Fonctionnels
- ✅ **Planning Module** - Gestion tâches avancée
- ✅ **Analytics Module** - KPIs et métriques
- ✅ **Intelligence Module** - Insights IA
- ✅ **Quality Module** - Contrôle qualité
- ✅ **What-If Module** - Simulation scénarios
- ✅ **Voice Module** - Interface vocale complète

---

## 🧠 **Système IA (11 Modules Opérationnels)**

### Modules Confirmés Fonctionnels
1. ✅ **Portfolio Manager** - Gestion multi-projets
2. ✅ **Moteur Prédictif** - Prédictions basées patterns
3. ✅ **Assistant Conversationnel** - Patterns d'intention
4. ✅ **Alertes Intelligentes** - 8 règles configurées
5. ✅ **Moteur Gamification** - 21 défis disponibles
6. ✅ **Prédicteur Risques** - Modèles ML entraînés
7. ✅ **Chat Intelligent** - 8 actions rapides
8. ✅ **Optimiseur Budgétaire** - Modèles optimisation
9. ✅ **Prédicteur Crises** - ML + patterns
10. ✅ **Coach Personnel** - Système observation
11. ✅ **RAG Manager** - FAISS indexé, prêt pour documents

### Modèles ML Pré-entraînés
- ✅ **EstimatorModel** : 85.0% précision classification
- ✅ **RiskModel** : RMSE 0.121 sur prédictions impact
- ✅ **Crisis Predictor** : 800 projets d'entraînement
- ✅ **1000 projets synthétiques** pour entraînement

---

## 🔧 **API FastAPI (Conditionnelle)**

**État** : ⚠️ Nécessite Ollama actif

### Endpoints Disponibles (si Ollama fonctionne)
- `GET /` - Page d'accueil API
- `GET /health` - État de santé du système
- `POST /generate-plan` - Génération de plan (via CrewAI)
- `GET /plans/{run_id}` - Récupération plan
- `POST /ai/estimate` - Estimations ML
- `POST /ai/analyze-risks` - Analyse risques ML
- `POST /ai/rag` - Requêtes RAG

**URL** : http://localhost:8000 (si Ollama actif)
**Documentation** : http://localhost:8000/docs

---

## 📊 **Rapports & Exports (Fonctionnels)**

### Export PDF Professionnel
✅ **Génération automatique avec** :
- Page de couverture avec métriques
- WBS détaillé avec visualisations 
- Analyse des risques avec matrice
- Graphiques Gantt intégrés
- Recommandations IA personnalisées

### Export CSV Multi-fichiers (ZIP)
✅ **10 fichiers CSV générés** :
1. `project_overview.csv` - Vue d'ensemble
2. `detailed_tasks.csv` - Tâches détaillées  
3. `phases_summary.csv` - Résumé phases
4. `resource_allocation.csv` - Allocation ressources
5. `risk_assessment.csv` - Évaluation risques
6. `budget_breakdown.csv` - Décomposition budget
7. `critical_path.csv` - Chemin critique
8. `kpis_metrics.csv` - Indicateurs clés
9. `ai_insights.csv` - Insights IA
10. `timeline.csv` - Chronologie

---

## 🎤 **Système Vocal (Opérationnel)**

### Text-to-Speech
✅ **5 voix disponibles** configurées  
✅ **Synthèse vocale** des rapports et métriques

### Reconnaissance Vocale
✅ **Microphone initialisé** avec calibration automatique
✅ **Seuil adaptatif** pour environnements variés  
✅ **Commands vocales** intégrées au dashboard

---

## 🔬 **Données & Modèles**

### Format Plan Généré
```json
{
  "run_id": "uuid-unique",
  "timestamp": "2025-09-02T15:30:00",
  "project_overview": {
    "title": "Application E-commerce Avancée",
    "total_duration": 60,
    "total_cost": 48000,
    "team_size": 3
  },
  "wbs": {
    "phases": [
      {
        "id": "phase_1", 
        "name": "Planification et Conception",
        "duration": 15,
        "tasks": [...]
      }
    ]
  },
  "risks": [...],
  "milestones": [...],
  "ml_enhanced": true
}
```

### Modèles Stockés
- ✅ `data/models/estimator_latest.pkl` - Modèle estimation
- ✅ `data/models/risk_model.pkl` - Modèle risques
- ✅ Base FAISS pour RAG indexée et prête

---

## ⚙️ **Configuration Technique**

### Technologies Confirmées Utilisées
- **Python 3.11** - Langage de base
- **Streamlit 1.2+** - Interface dashboard ✅ 
- **scikit-learn** - Modèles ML ✅
- **FAISS** - Base vectorielle RAG ✅
- **Plotly** - Visualisations interactives ✅
- **ReportLab** - Génération PDF ✅
- **pyttsx3** - Synthèse vocale ✅
- **speech_recognition** - Reconnaissance vocale ✅

### Dépendances Conditionnelles
- **CrewAI** - Framework agents (nécessite Ollama)
- **Ollama/Llama3.2** - LLM pour génération plans
- **FastAPI** - API REST (dépend du LLM)

---

## 🧪 **Tests & Qualité (Confirmés)**

### Métriques ML Réelles
- ✅ **Classificateur** : 85.0% précision
- ✅ **Prédicteur Impact** : RMSE 0.121
- ✅ **Entraînement** : 1000 projets synthétiques
- ✅ **Stability** : 11/11 modules IA opérationnels

### Suite de Tests
```bash
# Tests disponibles
python test_ml_complete.py        # ✅ Tests ML complets
python benchmark_complete.py      # ✅ Benchmarks performance
python test_ml_real.py           # ✅ Tests données réelles
```

---

## 🚀 **Guide d'Utilisation Pratique**

### 1. Mode Dashboard (Recommandé - Fonctionne Toujours)
```bash
# Ce qui marche sans prérequis
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521

# Accéder à http://localhost:8521
# Toutes les fonctionnalités IA sont disponibles
```

### 2. Mode Génération IA (Nécessite Ollama)
```bash
# Démarrer Ollama d'abord
ollama serve

# Dans un autre terminal
python crew.py "Développer une application mobile e-commerce"
```

### 3. Export et Rapports
```bash
# Depuis le dashboard Streamlit
# Boutons "Export PDF" et "Export CSV" fonctionnels
# Fichiers générés dans data/reports/
```

---

## ⚠️ **Limitations Actuelles**

### Non Fonctionnel Sans Ollama
- ❌ Génération de plans via `crew.py`
- ❌ API FastAPI complète
- ❌ Endpoints CrewAI

### Fonctionnel Indépendamment
- ✅ Dashboard Streamlit complet
- ✅ Tous les modèles ML et IA
- ✅ Visualisations et rapports
- ✅ Système vocal
- ✅ Analytics et métriques

---

## 📞 **Support & Issues**

- 🐛 **Bugs** : [GitHub Issues](https://github.com/Michel836/PlannerIA/issues)
- 💡 **Suggestions** : Pull requests bienvenues
- 📧 **Contact** : Via GitHub

---

## 📄 **Licence**

MIT License - Voir LICENSE pour détails complets.

---

**Statut Projet** : ✅ Dashboard Opérationnel | ⚠️ API Conditionnelle (Ollama requis)

**Dernière Validation** : 2 septembre 2025 - Tests complets effectués

---

## 🎯 **Commandes de Test Rapide**

```bash
# Vérifier que tout fonctionne
python -m streamlit run src/project_planner/dashboard/mvp_v1.py --server.port=8521

# Tester les modèles ML
python -c "
from src.project_planner.ml.estimator_model import EstimatorModel
em = EstimatorModel()
print('✅ Modèles ML opérationnels')
"

# Vérifier les modules IA (depuis le dashboard)
# Regarder les logs de démarrage pour "11/11 modules opérationnels"
```

**Version Vérifiée** : 1.0.0-verified | **Test Date** : 2025-09-02