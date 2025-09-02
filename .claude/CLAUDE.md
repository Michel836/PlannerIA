# 🎯 Contexte & Objectif

PlannerIA = pipeline **multi-agents** de planification (RAG), **API FastAPI**, **Dashboard Streamlit**, **optimisation chemin critique**, briques **ML**.
Cible: livrable soutenable (code propre, docs, tests, perf).

**Livrables clés**

- `data/runs/<id>/plan.json`, `rapport.md`, `slides.pdf`, `plan.csv`.

**Stack & périmètre**

- Python 3.11 — Windows 10 — VS Code — RTX 3090.
- Arbo: `src/project_planner/**`, `schema/`, `data/`, `tests/`, `config/`.
- Agents: Supervisor → Planner (WBS) → Estimator (coûts/durées) → Risk → Documentation.
- Validation: Pydantic + `schema/plan.schema.json`.

---

## 🧭 Règles d’interaction (toujours AVANT d’écrire du code)

1. **Plan en étapes**: architecture proposée, fichiers impactés, risques/impacts.
2. **Diff minimal**: proposer un patch strictement nécessaire.
3. **Justification**: robustesse, performance, testabilité, impacts.
4. **Tests Pytest**: fournir systématiquement des tests couvrant la modif.
5. **Commandes**: donner les commandes pour exécuter tests/serveurs.

---

## ✅ Checklists de qualité (gates)

### A. Gate “PRÉ-DIFF”

- [ ] Les entrées sont **validées** (types/valeurs), cas d’erreurs prévus.
- [ ] Architecture cohérente (pas de couplage inutile, DI légère).
- [ ] I/O évitables identifiées (préférer in-memory ou pipelines).
- [ ] Tests envisagés (unitaires + mocks, pas d’accès réseau réel).
- [ ] Impacts performance: O(n)/mémoire, profiling rapide si nécessaire.

### B. Gate “POST-DIFF”

- [ ] **Tests fournis** (Pytest) + commandes d’exécution.
- [ ] **Logs structurés** (niveau, message clair, contexte; pas de secrets).
- [ ] **Docstrings** (Google/NumPy) + commentaires pédagogiques.
- [ ] **Conformité PEP8** (ruff/black) + **typage** (mypy si pertinent).
- [ ] **Benchmark rapide** si hot-path (ex. boucle critique).

---

## 5 critères (définition opérationnelle)

1. **Lisibilité**
   - Docstrings + commentaires pédagogiques, noms explicites, early return.
2. **Modularité**
   - Fonctions pures quand possible; séparation I/O vs logique; DI légère.
3. **Robustesse**
   - `try/except` ciblés; erreurs explicites; validations Pydantic; logs utiles.
4. **Performance**
   - Éviter I/O sync bloquants, vectoriser si pertinent; pas de copies inutiles.
5. **Testabilité**
   - Tests unitaires isolés (mocks), données de test minimales, seed fixe.

**Cibles pratiques**

- Couverture sur les fichiers modifiés ≥ **80% lignes** / **100% branches critiques**.
- Latence endpoints “santé” < **20ms** local; calcul chemin critique < **100ms** sur jeu test.
- RAG: temps d’embed/page < **200ms** (local) et top-k citations exactes.

---

## 🔐 Sécurité & Confidentialité

- Ne **jamais** lire/afficher: `.env`, `secrets/**`, `*.key`, `*credentials*`.
- Périmètre d’édition: **uniquement** sous `src/**`, `tests/**`, `schema/**`, `config/**`.
- Pas d’upload externe non demandé. Éviter chemins absolus et secrets dans logs.

---

## 🧩 Mémos architecture

- **Multi-agents**: Supervisor → Planner → Estimator → Risk → Doc.
- **Validation**: Pydantic + `schema/plan.schema.json` (fail fast + messages clairs).
- **Optimisation**: chemin critique (graph DAG) + export KPIs.
- **Outputs**: `data/runs/<id>/plan.json`, `logs.json`, `rapport.md`.

---

## 🧪 Tests & Qualité (politique)

- **Pytest**: tests unitaires + mocks (pas d’appels réseau/LLM réels).
- Générer fixtures courtes; nommer `test_<module>_*.py`.
- Qualité: `ruff`, `black`, `mypy` (si annotations ajoutées).
- À chaque diff, fournir:
  - tests,
  - **commande**: `pytest -q`,
  - **résultat attendu** (ex. “7 passed”).

---

## 📡 API FastAPI — rappels

Endpoints à conserver / étendre:

- `POST /generate_plan`
- `GET /get_run/{id}`
- `POST /predict_estimates`
- `POST /predict_risks`
- `POST /feedback`
- _(à ajouter)_ `GET /health`, `GET /health/full`

---

## 🧰 Recettes (prompts prêts à l’emploi)

### 1) Endpoints santé (FastAPI)

> **Objectif**: ajouter `GET /health` et `GET /health/full`  
> **Demande**:

- Propose l’architecture (fichiers impactés).
- Affiche un **diff minimal**.
- Ajoute **tests Pytest** (mocks).
- Explique les choix (robustesse/testabilité/perf).
- Donne les **commandes**: `uvicorn …`, `pytest -q`.

### 2) Abstraction LLM (Ollama/vLLM)

> **Objectif**: `LLMClient` (sync/async, timeouts, retries, streaming) + adaptateur Ollama HTTP.  
> **Exigences**:

- Interface claire (protocol/ABC), injection par config.
- Tests unitaires avec mocks; exemple d’usage.
- **Diff minimal** aux bons emplacements (`src/project_planner/llm/…`).

### 3) RAG local (FAISS → fallback numpy)

> **Objectif**: ingestion `data/docs/` → embeddings `data/embeddings/` + citations.  
> **Exigences**:

- Fonction `ingest(path)` + `query(text, k=5)`;
- Fallback numpy si FAISS absent;
- Tests unitaires rapides sur mini-corpus;
- Exposer `rag_citations` dans `plan.json`.

### 4) Dashboard Streamlit (What-If)

> **Objectif**: module `whatif.py` avec sliders (durée/coût), recalcul chemin critique.  
> **Exigences**:

- Logique isolée testable;
- UI simple + commande lancement;
- Tests unitaires pour logique.

---

## 📦 Commandes utiles

- API: `python -m uvicorn src.project_planner.api.main:app --reload`
- Dashboard: `python -m streamlit run src/project_planner/dashboard/app.py`
- Tests: `pytest -q`
- Format/qualité: `ruff check . && black .`

---

## 📝 Style commit

`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:` — message court + impact (perf/robustesse/tests).

---

## 📌 Règles d’application des diffs

- Toujours **proposer** un diff, **attendre validation**, puis appliquer.
- Lister les **fichiers impactés** + **commandes** pour tester localement.
