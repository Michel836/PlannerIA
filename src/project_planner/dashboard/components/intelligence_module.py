# Module RAG + ML pour PlannerIA
# Assistant IA intelligent pour la gestion de projet avec apprentissage continu

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import plotly.graph_objects as go
import plotly.express as px

# Imports RAG réels avec fallback local
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import ollama
    RAG_AVAILABLE = True
    print("RAG complet disponible")
except ImportError:
    RAG_AVAILABLE = False
    print("RAG partiel - sentence-transformers ou chromadb manquant")
    # On peut quand même faire du RAG basique
    import ollama

class LocalRAGSystem:
    """Système RAG local avec Ollama + embeddings locaux"""
    
    def __init__(self):
        self.vector_db = None
        self.embedder = None
        self.ollama_model = "llama3.2:latest"
        self._init_rag()
    
    def _init_rag(self):
        """Initialise le système RAG"""
        try:
            if RAG_AVAILABLE:
                # ChromaDB pour le stockage vectoriel
                self.vector_db = chromadb.Client()
                try:
                    self.collection = self.vector_db.get_collection("project_knowledge")
                except:
                    self.collection = self.vector_db.create_collection("project_knowledge")
                
                # Modèle d'embeddings local
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("RAG systeme initialise avec ChromaDB")
            else:
                print("⚠️ RAG basique - utilisation Ollama seul")
        except Exception as e:
            print(f"Erreur RAG init: {e}")
    
    def add_knowledge(self, text: str, metadata: dict, doc_id: str):
        """Ajoute de la connaissance au système RAG"""
        if not RAG_AVAILABLE or not self.embedder:
            return False
            
        try:
            # Génération d'embedding
            embedding = self.embedder.encode([text])[0].tolist()
            
            # Ajout à ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"Erreur ajout connaissance: {e}")
            return False
    
    def search_knowledge(self, query: str, n_results: int = 3) -> List[Dict]:
        """Recherche dans la base de connaissances"""
        if not RAG_AVAILABLE or not self.embedder:
            return self._fallback_search(query)
        
        try:
            # Embedding de la requête
            query_embedding = self.embedder.encode([query])[0].tolist()
            
            # Recherche vectorielle
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format des résultats
            knowledge = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            )):
                knowledge.append({
                    "text": doc,
                    "metadata": metadata,
                    "relevance": 1 - distance,  # Score de pertinence
                    "rank": i + 1
                })
            
            return knowledge
        except Exception as e:
            print(f"Erreur recherche: {e}")
            return self._fallback_search(query)
    
    def _fallback_search(self, query: str) -> List[Dict]:
        """Recherche de fallback sans RAG complet"""
        # Simulation basique basée sur mots-clés
        fallback_knowledge = [
            {
                "text": "Pour les projets e-commerce, prévoir 20% de temps supplémentaire pour les tests de paiement.",
                "metadata": {"domain": "e-commerce", "type": "best_practice"},
                "relevance": 0.8,
                "rank": 1
            },
            {
                "text": "Les projets avec IA nécessitent généralement 30% plus de ressources que prévu.",
                "metadata": {"domain": "ai", "type": "estimation"},
                "relevance": 0.7,
                "rank": 2
            }
        ]
        return fallback_knowledge[:3]
    
    def generate_rag_response(self, question: str, context_docs: List[str]) -> str:
        """Génère une réponse RAG avec Ollama"""
        try:
            # Construction du prompt RAG
            context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)])
            
            prompt = f"""Tu es un expert en gestion de projet. Réponds à la question en utilisant uniquement les informations contextuelles fournies.

CONTEXTE:
{context}

QUESTION: {question}

RÉPONSE (basée uniquement sur le contexte fourni):"""

            # Appel à Ollama
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={"temperature": 0.1}  # Réponse plus factuelle
            )
            
            return response['response']
            
        except Exception as e:
            return f"Erreur génération RAG: {e}"

@dataclass
class ProjectKnowledge:
    """Structure de connaissance projet pour le RAG"""
    project_id: str
    title: str
    domain: str
    team_size: int
    duration_weeks: int
    technologies: List[str]
    final_quality_score: float
    delivered_on_time: bool
    budget_respected: bool
    lessons_learned: str
    success_factors: List[str]
    failure_points: List[str]
    mitigation_actions: List[str]

@dataclass
class MLPrediction:
    """Résultat de prédiction ML"""
    prediction_type: str
    value: float
    confidence: float
    factors: Dict[str, float]
    recommendations: List[str]

class PlannerRAGSystem:
    """Système RAG pour l'assistance IA en gestion de projet"""
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        # Initialisation RAG réel
        self.rag_system = LocalRAGSystem()
        self._populate_rag_with_knowledge()
        print("Assistant IA avec RAG initialise")
        
    def _load_knowledge_base(self) -> List[ProjectKnowledge]:
        """Simule une base de connaissances de projets passés"""
        return [
            ProjectKnowledge(
                project_id="proj_001",
                title="Application Mobile Banking",
                domain="Finance",
                team_size=8,
                duration_weeks=24,
                technologies=["React Native", "Node.js", "PostgreSQL"],
                final_quality_score=92,
                delivered_on_time=True,
                budget_respected=True,
                lessons_learned="Les tests automatisés intensifs ont été cruciaux. L'implication continue du client a évité les retours majeurs.",
                success_factors=["Tests automatisés dès le début", "Reviews de code strictes", "Communication client hebdomadaire"],
                failure_points=["Retards initiaux sur l'architecture", "Sous-estimation des tests de sécurité"],
                mitigation_actions=["Prototypage architecture en amont", "Audit sécurité externe précoce"]
            ),
            ProjectKnowledge(
                project_id="proj_002", 
                title="Plateforme E-commerce",
                domain="Retail",
                team_size=12,
                duration_weeks=36,
                technologies=["Java", "Spring", "MySQL", "Redis"],
                final_quality_score=78,
                delivered_on_time=False,
                budget_respected=False,
                lessons_learned="Complexité sous-estimée. Manque d'expertise sur certaines technos. Communication équipe difficile avec 2 sites.",
                success_factors=["Architecture microservices bien pensée", "DevOps robuste"],
                failure_points=["Estimation initiale trop optimiste", "Équipe distribuée mal coordonnée", "Tests de charge tardifs"],
                mitigation_actions=["Formation équipe en amont", "Daily meetings cross-sites", "Tests perf en continu"]
            ),
            ProjectKnowledge(
                project_id="proj_003",
                title="API Gateway Enterprise", 
                domain="Infrastructure",
                team_size=6,
                duration_weeks=16,
                technologies=["Go", "Docker", "Kubernetes", "MongoDB"],
                final_quality_score=95,
                delivered_on_time=True,
                budget_respected=True,
                lessons_learned="Équipe senior experte. Scope bien défini. Architecture simple et robuste.",
                success_factors=["Équipe experte homogène", "Architecture simple", "Tests de performance dès le début"],
                failure_points=["Documentation utilisateur retardée"],
                mitigation_actions=["Writer technique dédié plus tôt"]
            ),
            ProjectKnowledge(
                project_id="proj_004",
                title="CRM SaaS Multi-tenant",
                domain="Enterprise Software",
                team_size=15,
                duration_weeks=48,
                technologies=["Python", "Django", "React", "PostgreSQL", "Redis"],
                final_quality_score=85,
                delivered_on_time=True,
                budget_respected=False,
                lessons_learned="Projet complexe réussi grâce à une architecture bien pensée et des tests rigoureux. Budget dépassé par les coûts d'infrastructure cloud.",
                success_factors=["Architecture multi-tenant bien conçue", "Tests automatisés complets", "Monitoring avancé"],
                failure_points=["Coûts cloud sous-estimés", "Optimisations performance tardives"],
                mitigation_actions=["Audit coûts cloud mensuel", "Tests de performance dès les premiers sprints"]
            ),
            ProjectKnowledge(
                project_id="proj_005",
                title="Système IoT Industriel",
                domain="IoT/Industry",
                team_size=10,
                duration_weeks=32,
                technologies=["C++", "Python", "MQTT", "InfluxDB", "Grafana"],
                final_quality_score=88,
                delivered_on_time=False,
                budget_respected=True,
                lessons_learned="Challenges hardware et protocoles complexes. Intégration terrain plus complexe que prévu.",
                success_factors=["Expertise IoT solide", "Tests en environnement réel", "Protocoles robustes"],
                failure_points=["Intégration terrain sous-estimée", "Validation hardware tardive"],
                mitigation_actions=["Prototypage hardware early", "Tests terrain fréquents"]
            )
        ]
    
    def _populate_rag_with_knowledge(self):
        """Peuple le système RAG avec les connaissances"""
        for i, project in enumerate(self.knowledge_base):
            # Création du texte de connaissance
            knowledge_text = f"""
Projet: {project.title}
Domaine: {project.domain}
Équipe: {project.team_size} personnes
Durée: {project.duration_weeks} semaines
Technologies: {', '.join(project.technologies)}
Score qualité: {project.final_quality_score}%
Livré à temps: {'Oui' if project.delivered_on_time else 'Non'}
Budget respecté: {'Oui' if project.budget_respected else 'Non'}

Leçons apprises: {project.lessons_learned}

Facteurs de succès:
{chr(10).join([f'- {factor}' for factor in project.success_factors])}

Points de défaillance:
{chr(10).join([f'- {point}' for point in project.failure_points])}

Actions de mitigation:
{chr(10).join([f'- {action}' for action in project.mitigation_actions])}
"""
            
            # Métadonnées pour la recherche
            metadata = {
                "project_id": project.project_id,
                "domain": project.domain,
                "team_size": project.team_size,
                "duration_weeks": project.duration_weeks,
                "technologies": ','.join(project.technologies),
                "quality_score": project.final_quality_score,
                "on_time": project.delivered_on_time,
                "on_budget": project.budget_respected
            }
            
            # Ajout au RAG
            success = self.rag_system.add_knowledge(
                text=knowledge_text,
                metadata=metadata,
                doc_id=f"project_{i}"
            )
            
            if success and i == 0:  # Log seulement le premier pour éviter le spam
                print("Base de connaissances chargee dans RAG")
    
    def initialize_vector_store(self):
        """Initialise le store vectoriel - Déjà fait dans _populate_rag_with_knowledge"""
        pass
        
    def ask_question(self, question: str, project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Interface principale pour poser des questions à l'IA avec RAG réel"""
        
        # 🧠 RECHERCHE RAG RÉELLE
        relevant_docs = self.rag_system.search_knowledge(question, n_results=3)
        
        # Extraction des textes pour la génération
        context_texts = [doc["text"] for doc in relevant_docs]
        
        # 🤖 GÉNÉRATION RAG avec Ollama
        rag_answer = self.rag_system.generate_rag_response(question, context_texts)
        
        # Analyse de la question pour classification
        question_type = self._classify_question(question)
        
        response = {
            "question": question,
            "question_type": question_type,
            "answer": rag_answer,  # Réponse RAG réelle
            "confidence": sum([doc["relevance"] for doc in relevant_docs]) / len(relevant_docs) if relevant_docs else 0.0,
            "sources": [
                {
                    "text": doc["text"][:200] + "...",
                    "metadata": doc["metadata"],
                    "relevance": doc["relevance"]
                } for doc in relevant_docs
            ],
            "recommendations": [],
            "rag_used": True,
            "knowledge_sources": len(relevant_docs)
        }
        
        # Trouve les projets similaires pour l'analyse approfondie
        similar_projects = self._find_similar_projects(project_context)
        
        if question_type == "risk_analysis":
            response.update(self._analyze_risks(project_context, similar_projects))
        elif question_type == "timeline_optimization":
            response.update(self._suggest_timeline_optimization(project_context, similar_projects))
        elif question_type == "quality_prediction":
            response.update(self._predict_quality_outcome(project_context, similar_projects))
        elif question_type == "resource_optimization":
            response.update(self._optimize_resources(project_context, similar_projects))
        else:
            response.update(self._general_consultation(question, project_context, relevant_docs))
        
        return response
    
    def _find_similar_projects(self, project_context: Dict[str, Any]) -> List[ProjectKnowledge]:
        """Trouve les projets similaires dans la base de connaissances"""
        current_domain = project_context.get("domain", "")
        current_team_size = project_context.get("team_size", 0)
        current_technologies = set(project_context.get("technologies", []))
        
        similarities = []
        for project in self.knowledge_base:
            similarity_score = 0.0
            
            # Similarité domaine
            if project.domain.lower() == current_domain.lower():
                similarity_score += 0.3
            
            # Similarité taille équipe  
            team_diff = abs(project.team_size - current_team_size)
            if team_diff <= 2:
                similarity_score += 0.2
            elif team_diff <= 5:
                similarity_score += 0.1
            
            # Similarité technologies
            project_tech = set(project.technologies)
            tech_overlap = len(current_technologies.intersection(project_tech))
            if tech_overlap > 0:
                similarity_score += 0.3 * (tech_overlap / len(current_technologies.union(project_tech)))
            
            # Bonus pour les projets réussis
            if project.delivered_on_time and project.final_quality_score > 85:
                similarity_score += 0.2
            
            similarities.append((similarity_score, project))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [project for _, project in similarities]
    
    def _classify_question(self, question: str) -> str:
        """Classifie le type de question posée"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["risque", "problème", "échec", "retard", "danger"]):
            return "risk_analysis"
        elif any(word in question_lower for word in ["timeline", "planning", "optimis", "délai", "chemin critique", "réduire"]):
            return "timeline_optimization" 
        elif any(word in question_lower for word in ["qualité", "défaut", "test", "couverture", "livrable"]):
            return "quality_prediction"
        elif any(word in question_lower for word in ["ressource", "équipe", "allocation", "compétence"]):
            return "resource_optimization"
        else:
            return "general"
    
    def _analyze_risks(self, project_context: Dict, similar_projects: List[ProjectKnowledge]) -> Dict:
        """Analyse les risques basée sur les projets similaires"""
        risks = []
        mitigations = []
        
        for project in similar_projects[:3]:
            if not project.delivered_on_time or project.final_quality_score < 80:
                risks.extend(project.failure_points)
                mitigations.extend(project.mitigation_actions)
        
        # Déduplication et scoring
        unique_risks = list(set(risks))
        unique_mitigations = list(set(mitigations))
        
        return {
            "answer": f"Basé sur l'analyse de {len(similar_projects)} projets similaires, j'ai identifié {len(unique_risks)} risques potentiels majeurs. Ces risques sont extraits de projets ayant rencontré des difficultés similaires à votre contexte.",
            "confidence": min(0.9, len(similar_projects) * 0.3),
            "risks": unique_risks[:5],
            "mitigations": unique_mitigations[:5],
            "recommendations": [
                "Mettre en place un suivi hebdomadaire des risques identifiés avec des KPIs précis",
                "Prévoir des plans de contingence détaillés pour les 3 risques les plus critiques",
                "Organiser des retours d'expérience avec les équipes ayant géré des projets similaires",
                "Implémenter un système d'alerte précoce pour détecter les signaux faibles"
            ]
        }
    
    def _suggest_timeline_optimization(self, project_context: Dict, similar_projects: List[ProjectKnowledge]) -> Dict:
        """Suggestions d'optimisation de timeline"""
        successful_projects = [p for p in similar_projects if p.delivered_on_time]
        
        optimizations = []
        for project in successful_projects:
            optimizations.extend(project.success_factors)
        
        # Suggestions spécifiques pour réduire le chemin critique
        critical_path_recommendations = [
            "Identifier et paralléliser les tâches indépendantes du chemin critique",
            "Réduire les dépendances entre équipes par des interfaces bien définies",
            "Implémenter des livraisons incrémentales pour réduire la complexité",
            "Automatiser les tâches répétitives et chronophages",
            "Mettre en place des daily standups focalisés sur les blocages du chemin critique"
        ]
        
        return {
            "answer": f"Basé sur l'analyse de {len(successful_projects)} projets livrés à temps avec des contextes similaires, voici mes recommandations pour optimiser votre timeline et réduire le chemin critique.",
            "confidence": 0.85,
            "optimizations": list(set(optimizations))[:6],
            "critical_path_optimizations": critical_path_recommendations,
            "recommendations": [
                "Réaliser un audit du chemin critique actuel avec identification précise des goulots d'étranglement",
                "Implémenter les pratiques des projets les plus réussis dans votre domaine",
                "Mettre en place des jalons de validation hebdomadaires pour détecter les dérives tôt",
                "Organiser des sessions de résolution de problèmes collaboratives sur les blocages identifiés"
            ]
        }
    
    def _predict_quality_outcome(self, project_context: Dict, similar_projects: List[ProjectKnowledge]) -> Dict:
        """Prédiction de la qualité finale"""
        if not similar_projects:
            return {"answer": "Données insuffisantes pour une prédiction fiable de la qualité", "confidence": 0.1}
        
        avg_quality = np.mean([p.final_quality_score for p in similar_projects[:5]])
        quality_variance = np.std([p.final_quality_score for p in similar_projects[:5]])
        
        # Ajustements basés sur le contexte actuel
        current_coverage = project_context.get("current_test_coverage", 0)
        if current_coverage > 85:
            avg_quality += 8
        elif current_coverage < 70:
            avg_quality -= 10
        
        quality_recommendations = [
            f"Cibler une couverture de tests supérieure à {max(85, current_coverage + 15)}% pour optimiser la qualité",
            "Mettre en place des code reviews systématiques avec checklist qualité",
            "Implémenter des audits qualité à chaque jalon majeur",
            "Automatiser les tests de régression et d'intégration"
        ]
        
        return {
            "answer": f"Prédiction qualité finale basée sur {len(similar_projects)} projets similaires: {avg_quality:.1f}/100 (±{quality_variance:.1f}). Cette estimation intègre votre couverture de tests actuelle de {current_coverage}%.",
            "confidence": 0.75,
            "predicted_score": avg_quality,
            "variance": quality_variance,
            "recommendations": quality_recommendations
        }
    
    def _optimize_resources(self, project_context: Dict, similar_projects: List[ProjectKnowledge]) -> Dict:
        """Optimisation des ressources"""
        resource_insights = []
        
        for project in similar_projects[:3]:
            if project.final_quality_score > 85:
                resource_insights.extend(project.success_factors)
        
        return {
            "answer": f"Recommandations d'optimisation des ressources basées sur {len(similar_projects)} projets similaires réussis.",
            "confidence": 0.8,
            "resource_optimizations": list(set(resource_insights))[:6],
            "recommendations": [
                "Allouer les ressources senior sur les modules critiques du chemin critique",
                "Mettre en place du pair programming sur les composants complexes",
                "Organiser des formations croisées pour réduire les dépendances personnelles",
                "Implémenter une matrice de compétences pour optimiser l'allocation"
            ]
        }
    
    def _general_consultation(self, question: str, project_context: Dict, similar_projects: List[ProjectKnowledge]) -> Dict:
        """Consultation générale"""
        return {
            "answer": f"Basé sur l'analyse de {len(similar_projects)} projets similaires, voici mes insights pour votre question: '{question}'. L'analyse porte sur des projets dans le domaine {project_context.get('domain', 'non spécifié')} avec des équipes de taille comparable.",
            "confidence": 0.7,
            "recommendations": [
                "Analyser les retours d'expérience des projets similaires",
                "Adapter les bonnes pratiques à votre contexte spécifique",
                "Mettre en place un suivi régulier des métriques clés"
            ]
        }

class PlannerMLEngine:
    """Moteur ML pour prédictions et optimisations de projet"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialise les modèles ML"""
        self.models = {
            "delay_predictor": RandomForestRegressor(n_estimators=100, random_state=42),
            "quality_predictor": RandomForestRegressor(n_estimators=100, random_state=42),
            "risk_classifier": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        self.scalers = {
            "delay_predictor": StandardScaler(),
            "quality_predictor": StandardScaler(), 
            "risk_classifier": StandardScaler()
        }
    
    def generate_training_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Génère des données d'entraînement simulées"""
        np.random.seed(42)
        n_projects = 500
        
        # Features
        data = {
            "team_size": np.random.randint(3, 15, n_projects),
            "duration_weeks": np.random.randint(8, 52, n_projects),
            "complexity_score": np.random.uniform(1, 10, n_projects),
            "team_experience": np.random.uniform(1, 5, n_projects),
            "test_coverage": np.random.uniform(40, 95, n_projects),
            "code_review_rate": np.random.uniform(0.5, 1.0, n_projects),
            "defects_found": np.random.poisson(8, n_projects),
            "critical_defects": np.random.poisson(2, n_projects),
            "velocity_consistency": np.random.uniform(0.6, 1.0, n_projects),
            "stakeholder_changes": np.random.poisson(5, n_projects)
        }
        
        df = pd.DataFrame(data)
        
        # Targets (avec logique business)
        delay_days = []
        quality_scores = []
        risk_levels = []
        
        for i in range(n_projects):
            # Delay prediction (logique business)
            base_delay = 0
            if df.iloc[i]["team_experience"] < 2:
                base_delay += np.random.exponential(5)
            if df.iloc[i]["complexity_score"] > 7:
                base_delay += np.random.exponential(8)
            if df.iloc[i]["stakeholder_changes"] > 8:
                base_delay += np.random.exponential(10)
            if df.iloc[i]["velocity_consistency"] < 0.8:
                base_delay += np.random.exponential(6)
            
            delay_days.append(max(0, base_delay))
            
            # Quality score (logique business)
            base_quality = 85
            base_quality += (df.iloc[i]["test_coverage"] - 70) * 0.3
            base_quality += (df.iloc[i]["code_review_rate"] - 0.7) * 20
            base_quality -= df.iloc[i]["critical_defects"] * 5
            base_quality += (df.iloc[i]["team_experience"] - 2.5) * 4
            base_quality -= (df.iloc[i]["complexity_score"] - 5) * 2
            
            quality_scores.append(max(50, min(100, base_quality + np.random.normal(0, 5))))
            
            # Risk level (0=Low, 1=Medium, 2=High)
            risk_score = 0
            if delay_days[-1] > 10:
                risk_score += 1
            if quality_scores[-1] < 75:
                risk_score += 1
            if df.iloc[i]["critical_defects"] > 3:
                risk_score += 1
            
            risk_levels.append(min(2, risk_score))
        
        targets = {
            "delay_days": pd.Series(delay_days),
            "quality_score": pd.Series(quality_scores),
            "risk_level": pd.Series(risk_levels)
        }
        
        return df, targets
    
    def train_models(self) -> Dict[str, float]:
        """Entraîne tous les modèles ML"""
        X, y = self.generate_training_data()
        
        results = {}
        
        # Train delay predictor
        X_scaled = self.scalers["delay_predictor"].fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y["delay_days"], test_size=0.2, random_state=42)
        
        self.models["delay_predictor"].fit(X_train, y_train)
        predictions = self.models["delay_predictor"].predict(X_test)
        results["delay_mse"] = mean_squared_error(y_test, predictions)
        
        # Train quality predictor  
        X_scaled = self.scalers["quality_predictor"].fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y["quality_score"], test_size=0.2, random_state=42)
        
        self.models["quality_predictor"].fit(X_train, y_train)
        predictions = self.models["quality_predictor"].predict(X_test)
        results["quality_mse"] = mean_squared_error(y_test, predictions)
        
        # Train risk classifier
        X_scaled = self.scalers["risk_classifier"].fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y["risk_level"], test_size=0.2, random_state=42)
        
        self.models["risk_classifier"].fit(X_train, y_train)
        predictions = self.models["risk_classifier"].predict(X_test)
        results["risk_accuracy"] = accuracy_score(y_test, predictions)
        
        self.is_trained = True
        return results
    
    def predict_project_outcomes(self, project_features: Dict[str, float]) -> Dict[str, MLPrediction]:
        """Prédit les résultats du projet"""
        if not self.is_trained:
            self.train_models()
        
        # Préparer les features
        feature_vector = [
            project_features.get("team_size", 6),
            project_features.get("duration_weeks", 20), 
            project_features.get("complexity_score", 5),
            project_features.get("team_experience", 3),
            project_features.get("test_coverage", 75),
            project_features.get("code_review_rate", 0.8),
            project_features.get("defects_found", 8),
            project_features.get("critical_defects", 2),
            project_features.get("velocity_consistency", 0.85),
            project_features.get("stakeholder_changes", 5)
        ]
        
        predictions = {}
        
        # Delay prediction
        X_scaled = self.scalers["delay_predictor"].transform([feature_vector])
        delay_pred = self.models["delay_predictor"].predict(X_scaled)[0]
        delay_confidence = min(0.9, 1.0 - (delay_pred / 30))  # Moins de confiance pour gros retards
        
        predictions["delay"] = MLPrediction(
            prediction_type="delay_days",
            value=delay_pred,
            confidence=delay_confidence,
            factors=self._get_feature_importance("delay_predictor", feature_vector),
            recommendations=self._get_delay_recommendations(delay_pred, project_features)
        )
        
        # Quality prediction
        X_scaled = self.scalers["quality_predictor"].transform([feature_vector])
        quality_pred = self.models["quality_predictor"].predict(X_scaled)[0]
        quality_confidence = min(0.9, quality_pred / 100)
        
        predictions["quality"] = MLPrediction(
            prediction_type="quality_score",
            value=quality_pred,
            confidence=quality_confidence,
            factors=self._get_feature_importance("quality_predictor", feature_vector),
            recommendations=self._get_quality_recommendations(quality_pred, project_features)
        )
        
        # Risk prediction
        X_scaled = self.scalers["risk_classifier"].transform([feature_vector])
        risk_proba = self.models["risk_classifier"].predict_proba(X_scaled)[0]
        risk_level = np.argmax(risk_proba)
        
        predictions["risk"] = MLPrediction(
            prediction_type="risk_level",
            value=float(risk_level),
            confidence=float(max(risk_proba)),
            factors=self._get_feature_importance("risk_classifier", feature_vector),
            recommendations=self._get_risk_recommendations(risk_level, project_features)
        )
        
        return predictions
    
    def _get_feature_importance(self, model_name: str, feature_vector: List[float]) -> Dict[str, float]:
        """Retourne l'importance des features pour une prédiction"""
        feature_names = [
            "team_size", "duration_weeks", "complexity_score", "team_experience",
            "test_coverage", "code_review_rate", "defects_found", "critical_defects",
            "velocity_consistency", "stakeholder_changes"
        ]
        
        importance = self.models[model_name].feature_importances_
        return dict(zip(feature_names, importance))
    
    def _get_delay_recommendations(self, predicted_delay: float, features: Dict) -> List[str]:
        """Recommandations pour réduire les retards"""
        recommendations = []
        
        if predicted_delay > 14:
            recommendations.append("Risque de retard majeur - Revoir le scope du projet et prioriser les fonctionnalités critiques")
            
        if features.get("team_experience", 3) < 2.5:
            recommendations.append("Renforcer l'équipe avec des développeurs seniors expérimentés")
            
        if features.get("velocity_consistency", 0.85) < 0.8:
            recommendations.append("Améliorer la prédictibilité des sprints par de meilleures estimations")
            
        if features.get("stakeholder_changes", 5) > 8:
            recommendations.append("Limiter les changements de périmètre par un gel fonctionnel")
            
        if predicted_delay > 7:
            recommendations.append("Paralléliser les tâches sur le chemin critique pour accélérer la livraison")
            
        return recommendations
    
    def _get_quality_recommendations(self, predicted_quality: float, features: Dict) -> List[str]:
        """Recommandations pour améliorer la qualité"""
        recommendations = []
        
        if predicted_quality < 80:
            recommendations.append("Objectif qualité en danger - Plan d'action urgent avec audit qualité")
            
        if features.get("test_coverage", 75) < 80:
            recommendations.append("Augmenter la couverture de tests à 85%+ avec focus sur les paths critiques")
            
        if features.get("code_review_rate", 0.8) < 0.9:
            recommendations.append("Systématiser les code reviews avec checklist qualité obligatoire")
            
        if features.get("critical_defects", 2) > 3:
            recommendations.append("Focus sur la réduction des défauts critiques par des tests supplémentaires")
            
        return recommendations
    
    def _get_risk_recommendations(self, risk_level: int, features: Dict) -> List[str]:
        """Recommandations selon le niveau de risque"""
        risk_names = ["Faible", "Moyen", "Élevé"]
        recommendations = [f"Niveau de risque: {risk_names[risk_level]}"]
        
        if risk_level >= 1:
            recommendations.append("Mettre en place un suivi hebdomadaire renforcé avec reporting détaillé")
            
        if risk_level == 2:
            recommendations.append("Alerter les stakeholders - Plan de contingence requis immédiatement")
            recommendations.append("Envisager du renfort d'équipe ou une réduction de scope")
            
        return recommendations

class PlannerIntelligenceModule:
    """Module principal d'intelligence artificielle pour PlannerIA"""
    
    def __init__(self):
        self.rag_system = PlannerRAGSystem()
        self.ml_engine = PlannerMLEngine()
        self.conversation_history = []
    
    def render_intelligence_dashboard(self, project_context: Dict[str, Any]):
        """Interface principale du module IA"""
        st.markdown("### Assistant IA PlannerIA")
        st.markdown("*RAG + ML pour optimisation intelligente de projets*")
        
        # Configuration
        col1, col2 = st.columns([3, 1])
        
        with col1:
            intelligence_mode = st.selectbox(
                "Mode IA:",
                ["Assistant Chat", "Prédictions ML", "Analyse Historique", "Optimisations"],
                key="intelligence_mode_select"
            )
        
        with col2:
            if st.button("Entraîner ML", key="train_ml_button", use_container_width=True):
                with st.spinner("Entraînement des modèles..."):
                    results = self.ml_engine.train_models()
                    st.success(f"Modèles entraînés! Précision: {results.get('risk_accuracy', 0):.2f}")
        
        st.divider()
        
        if intelligence_mode == "Assistant Chat":
            self._render_chat_interface(project_context)
            
        elif intelligence_mode == "Prédictions ML":
            self._render_ml_predictions(project_context)
            
        elif intelligence_mode == "Analyse Historique":
            self._render_historical_analysis()
            
        elif intelligence_mode == "Optimisations":
            self._render_optimization_suggestions(project_context)
    
    def _render_chat_interface(self, project_context: Dict[str, Any]):
        """Interface de chat avec l'assistant IA"""
        st.subheader("Posez une question à l'IA")
        
        # Exemples de questions
        with st.expander("Exemples de questions"):
            example_questions = [
                "Quels sont les risques principaux de ce projet ?",
                "Comment optimiser ma timeline actuelle ?", 
                "Prédis la qualité finale de mes livrables",
                "Quelles ressources allouer en priorité ?",
                "Comment faire pour réduire le chemin critique ?"
            ]
            for i, q in enumerate(example_questions):
                if st.button(q, key=f"example_question_{i}"):
                    st.session_state["selected_question_intel"] = q
        
        # Input utilisateur
        user_question = st.text_input(
            "Votre question:",
            value=st.session_state.get("selected_question_intel", ""),
            key="user_question_intel_input",
            placeholder="Ex: Quels sont les risques de ce projet ?"
        )
        
        if st.button("Poser la question", key="ask_question_button", type="primary") and user_question:
            with st.spinner("L'IA analyse votre question..."):
                try:
                    response = self.rag_system.ask_question(user_question, project_context)
                    
                    # Affichage de la réponse
                    st.markdown("#### Réponse de l'IA")
                    st.write(response["answer"])
                    
                    # Confiance
                    confidence_color = "green" if response["confidence"] > 0.7 else "orange" if response["confidence"] > 0.4 else "red"
                    st.markdown(f"**Confiance:** :{confidence_color}[{response['confidence']:.1%}]")
                    
                    # Recommandations
                    if response.get("recommendations"):
                        st.markdown("#### Recommandations")
                        for rec in response["recommendations"]:
                            st.write(f"• {rec}")
                    
                    # Optimisations spécifiques pour le chemin critique
                    if response.get("critical_path_optimizations"):
                        st.markdown("#### Optimisations Chemin Critique")
                        for opt in response["critical_path_optimizations"]:
                            st.write(f"• {opt}")
                    
                    # Projets similaires
                    if response.get("similar_projects"):
                        st.markdown("#### Basé sur des projets similaires")
                        for proj in response["similar_projects"][:3]:
                            with st.expander(f"{proj.title} - Score: {proj.final_quality_score}/100"):
                                st.write(f"**Domaine:** {proj.domain}")
                                st.write(f"**Équipe:** {proj.team_size} personnes")
                                st.write(f"**Technologies:** {', '.join(proj.technologies)}")
                                st.write(f"**Retours:** {proj.lessons_learned}")
                    
                    # Stockage dans l'historique
                    self.conversation_history.append({
                        "question": user_question,
                        "response": response,
                        "timestamp": datetime.now()
                    })
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement de la question: {str(e)}")
        
        # Nettoyage de la question sélectionnée
        if "selected_question_intel" in st.session_state and user_question != st.session_state["selected_question_intel"]:
            del st.session_state["selected_question_intel"]
    
    def _render_ml_predictions(self, project_context: Dict[str, Any]):
        """Interface des prédictions ML"""
        st.subheader("Prédictions ML")
        
        # Configuration des features du projet
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Configuration Projet**")
            team_size = st.slider("Taille équipe", 3, 15, 6, key="ml_team_size")
            duration_weeks = st.slider("Durée (semaines)", 8, 52, 20, key="ml_duration")
            complexity_score = st.slider("Complexité (1-10)", 1.0, 10.0, 5.0, 0.5, key="ml_complexity")
            team_experience = st.slider("Expérience équipe (1-5)", 1.0, 5.0, 3.0, 0.5, key="ml_experience")
            stakeholder_changes = st.slider("Changements stakeholders", 0, 20, 5, key="ml_changes")
        
        with col2:
            st.markdown("**Métriques Qualité Actuelles**")
            test_coverage = st.slider("Couverture tests (%)", 0, 100, 75, key="ml_coverage")
            code_review_rate = st.slider("Taux code review", 0.0, 1.0, 0.8, 0.05, key="ml_review_rate")
            defects_found = st.slider("Défauts trouvés", 0, 50, 8, key="ml_defects")
            critical_defects = st.slider("Défauts critiques", 0, 10, 2, key="ml_critical")
            velocity_consistency = st.slider("Consistance vélocité", 0.0, 1.0, 0.85, 0.05, key="ml_velocity")
        
        # Prédictions
        if st.button("Générer Prédictions", key="generate_predictions_button", type="primary"):
            features = {
                "team_size": team_size,
                "duration_weeks": duration_weeks,
                "complexity_score": complexity_score,
                "team_experience": team_experience,
                "test_coverage": test_coverage,
                "code_review_rate": code_review_rate,
                "defects_found": defects_found,
                "critical_defects": critical_defects,
                "velocity_consistency": velocity_consistency,
                "stakeholder_changes": stakeholder_changes
            }
            
            with st.spinner("Calcul des prédictions..."):
                predictions = self.ml_engine.predict_project_outcomes(features)
            
            # Affichage des résultats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                delay_pred = predictions["delay"]
                delay_color = "green" if delay_pred.value < 7 else "orange" if delay_pred.value < 14 else "red"
                st.metric(
                    "Retard Prévu", 
                    f"{delay_pred.value:.1f} jours",
                    delta=f"Confiance: {delay_pred.confidence:.1%}"
                )
                st.markdown(f":{delay_color}[Niveau de retard]")
            
            with col2:
                quality_pred = predictions["quality"]
                quality_color = "green" if quality_pred.value > 85 else "orange" if quality_pred.value > 75 else "red"
                st.metric(
                    "Score Qualité",
                    f"{quality_pred.value:.1f}/100",
                    delta=f"Confiance: {quality_pred.confidence:.1%}"
                )
                st.markdown(f":{quality_color}[Niveau qualité]")
            
            with col3:
                risk_pred = predictions["risk"]
                risk_levels = ["Faible", "Moyen", "Élevé"]
                risk_color = ["green", "orange", "red"][int(risk_pred.value)]
                st.metric(
                    "Niveau Risque",
                    risk_levels[int(risk_pred.value)],
                    delta=f"Confiance: {risk_pred.confidence:.1%}"
                )
            
            # Recommandations détaillées
            st.markdown("### Recommandations IA")
            
            tabs = st.tabs(["Délais", "Qualité", "Risques"])
            
            with tabs[0]:
                for rec in delay_pred.recommendations:
                    st.write(f"• {rec}")
            
            with tabs[1]: 
                for rec in quality_pred.recommendations:
                    st.write(f"• {rec}")
            
            with tabs[2]:
                for rec in risk_pred.recommendations:
                    st.write(f"• {rec}")
    
    def _render_historical_analysis(self):
        """Analyse des tendances historiques"""
        st.subheader("Analyse Historique")
        
        # Simulation de données historiques
        projects_data = []
        for project in self.rag_system.knowledge_base:
            projects_data.append({
                "Projet": project.title,
                "Domaine": project.domain,
                "Équipe": project.team_size,
                "Durée": project.duration_weeks,
                "Qualité": project.final_quality_score,
                "À temps": "✅" if project.delivered_on_time else "❌",
                "Budget": "✅" if project.budget_respected else "❌"
            })
        
        df = pd.DataFrame(projects_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique qualité par domaine
            fig_quality = px.box(df, x="Domaine", y="Qualité", 
                               title="Distribution Qualité par Domaine")
            st.plotly_chart(fig_quality, use_container_width=True)
        
        with col2:
            # Relation taille équipe / durée
            fig_team = px.scatter(df, x="Équipe", y="Durée", color="Qualité",
                                title="Taille Équipe vs Durée",
                                hover_data=["Projet"])
            st.plotly_chart(fig_team, use_container_width=True)
        
        # Tableau des projets
        st.markdown("#### Base de Connaissances Projets")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Insights automatiques
        st.markdown("#### Insights Automatiques")
        avg_quality = df["Qualité"].mean()
        success_rate = len(df[df["À temps"] == "✅"]) / len(df) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Qualité Moyenne", f"{avg_quality:.1f}/100")
        with col2:
            st.metric("Taux Livraison", f"{success_rate:.1f}%")
        with col3:
            best_domain = df.groupby("Domaine")["Qualité"].mean().idxmax()
            st.metric("Meilleur Domaine", best_domain)
    
    def _render_optimization_suggestions(self, project_context: Dict[str, Any]):
        """Suggestions d'optimisation intelligentes"""
        st.subheader("Optimisations Intelligentes")
        
        optimization_type = st.selectbox(
            "Type d'optimisation:",
            ["Timeline", "Ressources", "Qualité", "Risques"],
            key="optimization_type_select"
        )
        
        if optimization_type == "Timeline":
            st.markdown("#### Optimisation Timeline")
            
            # Simulation d'analyse du chemin critique
            critical_tasks = [
                {"task": "API Backend Core", "current_duration": 5, "optimized_duration": 3, "method": "Parallélisation"},
                {"task": "Tests E2E", "current_duration": 3, "optimized_duration": 2, "method": "Automatisation"},
                {"task": "Documentation", "current_duration": 2, "optimized_duration": 1, "method": "Templates"},
                {"task": "Intégration Mobile", "current_duration": 4, "optimized_duration": 2, "method": "Framework unifié"},
                {"task": "Tests de Performance", "current_duration": 3, "optimized_duration": 1, "method": "Tests continus"}
            ]
            
            total_gain = sum(t["current_duration"] - t["optimized_duration"] for t in critical_tasks)
            st.success(f"Gain potentiel: {total_gain} jours sur le chemin critique")
            
            for task in critical_tasks:
                with st.expander(f"{task['task']} - Gain: {task['current_duration'] - task['optimized_duration']}j"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Durée actuelle:** {task['current_duration']} jours")
                    with col2:
                        st.write(f"**Durée optimisée:** {task['optimized_duration']} jours")  
                    with col3:
                        st.write(f"**Méthode:** {task['method']}")
        
        elif optimization_type == "Ressources":
            st.markdown("#### Optimisation Ressources")
            
            # Matrice compétences simulée
            skills_matrix = {
                "Alice Martin": ["React", "Node.js", "Tests"],
                "Bob Dupont": ["Java", "Spring", "Architecture"], 
                "Diana Prince": ["UI/UX", "Mobile", "Design"],
                "Charlie Moreau": ["DevOps", "Docker", "CI/CD"],
                "Eva Schmidt": ["Python", "Data", "ML"],
                "Frank Liu": ["Go", "Microservices", "Kubernetes"]
            }
            
            st.write("**Allocation Optimale Recommandée:**")
            
            for person, skills in skills_matrix.items():
                with st.expander(f"{person} - Compétences: {', '.join(skills)}"):
                    # Recommandations d'allocation
                    if "React" in skills:
                        st.success("Recommandé pour: Interface Utilisateur Mobile")
                    elif "Java" in skills:
                        st.success("Recommandé pour: API Backend Core") 
                    elif "UI/UX" in skills:
                        st.success("Recommandé pour: Design System")
                    elif "DevOps" in skills:
                        st.success("Recommandé pour: Pipeline CI/CD")
                    elif "Python" in skills:
                        st.success("Recommandé pour: Analytics et ML")
                    elif "Go" in skills:
                        st.success("Recommandé pour: Services haute performance")
        
        elif optimization_type == "Qualité":
            st.markdown("#### Optimisation Qualité")
            
            quality_actions = [
                {"action": "Augmenter couverture tests", "impact": "+8 points qualité", "effort": "2 semaines", "priority": "Haute"},
                {"action": "Code review systématique", "impact": "+5 points qualité", "effort": "Continue", "priority": "Haute"},
                {"action": "Audit sécurité", "impact": "+12 points qualité", "effort": "1 semaine", "priority": "Critique"},
                {"action": "Refactoring technique", "impact": "+6 points qualité", "effort": "3 semaines", "priority": "Moyenne"},
                {"action": "Tests de performance", "impact": "+7 points qualité", "effort": "1 semaine", "priority": "Haute"},
                {"action": "Documentation API", "impact": "+4 points qualité", "effort": "1 semaine", "priority": "Moyenne"}
            ]
            
            for action in quality_actions:
                priority_color = {"Critique": "red", "Haute": "orange", "Moyenne": "blue"}.get(action["priority"], "gray")
                with st.expander(f":{priority_color}[{action['priority']}] {action['action']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Impact:** {action['impact']}")
                    with col2:
                        st.write(f"**Effort:** {action['effort']}")
        
        elif optimization_type == "Risques":
            st.markdown("#### Mitigation des Risques")
            
            risks = [
                {"risk": "Retard sur API Backend", "probability": 0.7, "impact": "Élevé", "mitigation": "Renfort technique senior"},
                {"risk": "Défauts sécurité", "probability": 0.4, "impact": "Critique", "mitigation": "Audit externe ASAP"},
                {"risk": "Changements périmètre", "probability": 0.6, "impact": "Moyen", "mitigation": "Gel fonctionnel"},
                {"risk": "Performance insuffisante", "probability": 0.5, "impact": "Élevé", "mitigation": "Tests charge continus"},
                {"risk": "Intégration complexe", "probability": 0.3, "impact": "Moyen", "mitigation": "POC intégration early"}
            ]
            
            for risk in risks:
                risk_score = risk["probability"] * {"Faible": 1, "Moyen": 2, "Élevé": 3, "Critique": 4}[risk["impact"]]
                color = "red" if risk_score > 2.5 else "orange" if risk_score > 1.5 else "green"
                
                with st.expander(f":{color}[Score: {risk_score:.1f}] {risk['risk']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Probabilité:** {risk['probability']:.0%}")
                    with col2:
                        st.write(f"**Impact:** {risk['impact']}")
                    with col3:
                        st.write(f"**Mitigation:** {risk['mitigation']}")

def show_intelligence_module(project_context: Dict[str, Any]):
    """Point d'entrée pour le module d'intelligence artificielle"""
    intelligence_module = PlannerIntelligenceModule()
    intelligence_module.render_intelligence_dashboard(project_context)

if __name__ == "__main__":
    st.set_page_config(page_title="PlannerIA - Module Intelligence", layout="wide")
    
    # Contexte de test
    test_context = {
        "project_id": "projet_test",
        "domain": "Finance",
        "team_size": 8,
        "technologies": ["React", "Node.js", "PostgreSQL"],
        "current_test_coverage": 76
    }
    
    show_intelligence_module(test_context)