"""
🗂️ Portfolio Manager IA - Gestionnaire intelligent de portefeuille de projets
Orchestration multi-projets avec intelligence collective et optimisation de ressources
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx

# Types et énumérations
class ProjectStatus(Enum):
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ProjectPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResourceType(Enum):
    HUMAN = "human"
    BUDGET = "budget"
    TECHNOLOGY = "technology"
    TIME = "time"

class ProjectTemplate(Enum):
    WEB_APP = "web_application"
    MOBILE_APP = "mobile_application"
    AI_PROJECT = "ai_project"
    MARKETING = "marketing_campaign"
    INFRASTRUCTURE = "infrastructure"
    RESEARCH = "research_development"

@dataclass
class Project:
    id: str
    name: str
    description: str
    status: ProjectStatus
    priority: ProjectPriority
    start_date: datetime
    end_date: Optional[datetime]
    budget: float
    team_size: int
    domain: str
    template: ProjectTemplate
    progress: float = 0.0
    health_score: float = 100.0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ResourceAllocation:
    project_id: str
    resource_type: ResourceType
    allocated: float
    capacity: float
    efficiency: float = 1.0

@dataclass
class ProjectDependency:
    from_project: str
    to_project: str
    dependency_type: str
    strength: float
    description: str

@dataclass
class PortfolioInsight:
    type: str
    severity: str
    title: str
    description: str
    affected_projects: List[str]
    recommendation: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AIPortfolioManager:
    """Gestionnaire IA intelligent pour portefeuille de projets"""
    
    def __init__(self):
        self.projects: Dict[str, Project] = {}
        self.resource_allocations: Dict[str, List[ResourceAllocation]] = {}
        self.dependencies: List[ProjectDependency] = []
        self.insights_history: List[PortfolioInsight] = []
        self.templates_db = self._initialize_templates()
        self.optimization_cache = {}
        self.analytics_cache = {}
        
        # Configuration ML
        self.scaler = StandardScaler()
        self.cluster_model = KMeans(n_clusters=3, random_state=42)
        
        # Métriques portfolio
        self.portfolio_metrics = {
            "total_budget": 0.0,
            "total_projects": 0,
            "avg_health_score": 0.0,
            "resource_efficiency": 0.0,
            "on_time_delivery": 0.0
        }
        
        print("[PORTFOLIO] Portfolio Manager IA initialisé")
    
    def _initialize_templates(self) -> Dict[ProjectTemplate, Dict]:
        """Initialise les templates de projets avec patterns IA"""
        
        return {
            ProjectTemplate.WEB_APP: {
                "phases": ["Analyse", "Design", "Développement", "Test", "Déploiement"],
                "duration_range": (30, 120),
                "team_size_range": (2, 8),
                "budget_multiplier": 1.0,
                "risk_factors": ["Scope creep", "Technology changes", "User feedback"],
                "success_patterns": {
                    "agile_methodology": 0.85,
                    "experienced_team": 0.9,
                    "clear_requirements": 0.8
                }
            },
            ProjectTemplate.MOBILE_APP: {
                "phases": ["Conception", "UI/UX", "Développement", "Test", "Store"],
                "duration_range": (45, 180),
                "team_size_range": (3, 10),
                "budget_multiplier": 1.3,
                "risk_factors": ["Platform compatibility", "App store approval", "Performance"],
                "success_patterns": {
                    "native_development": 0.88,
                    "user_testing": 0.92,
                    "performance_optimization": 0.85
                }
            },
            ProjectTemplate.AI_PROJECT: {
                "phases": ["Research", "Data", "Modelling", "Training", "Deployment"],
                "duration_range": (60, 240),
                "team_size_range": (2, 6),
                "budget_multiplier": 1.5,
                "risk_factors": ["Data quality", "Model performance", "Compute resources"],
                "success_patterns": {
                    "quality_data": 0.95,
                    "iterative_approach": 0.88,
                    "domain_expertise": 0.9
                }
            },
            ProjectTemplate.MARKETING: {
                "phases": ["Stratégie", "Création", "Lancement", "Optimisation", "Analyse"],
                "duration_range": (15, 90),
                "team_size_range": (2, 12),
                "budget_multiplier": 0.8,
                "risk_factors": ["Market response", "Budget allocation", "Creative quality"],
                "success_patterns": {
                    "data_driven": 0.85,
                    "multi_channel": 0.8,
                    "continuous_optimization": 0.9
                }
            }
        }
    
    def create_project_wizard(self, name: str, description: str, template: ProjectTemplate, 
                            team_size: int, budget: float, priority: ProjectPriority = ProjectPriority.MEDIUM) -> Project:
        """Assistant IA pour création de nouveau projet"""
        
        project_id = f"proj_{int(time.time() * 1000)}_{len(self.projects)}"
        template_data = self.templates_db[template]
        
        # Estimation intelligente basée sur template
        duration_days = np.random.randint(*template_data["duration_range"])
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        # Ajustement budget selon template
        adjusted_budget = budget * template_data["budget_multiplier"]
        
        # Détection du domaine
        domain = self._detect_domain(name, description, template)
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            status=ProjectStatus.PLANNING,
            priority=priority,
            start_date=start_date,
            end_date=end_date,
            budget=adjusted_budget,
            team_size=team_size,
            domain=domain,
            template=template
        )
        
        # Enregistrement projet
        self.projects[project_id] = project
        
        # Allocation initiale des ressources
        self._initialize_project_resources(project_id, template_data)
        
        # Génération insights initial
        self._generate_project_creation_insights(project)
        
        print(f"[PORTFOLIO] Nouveau projet créé: {name} (ID: {project_id})")
        return project
    
    def _detect_domain(self, name: str, description: str, template: ProjectTemplate) -> str:
        """Détection intelligente du domaine projet"""
        
        domain_keywords = {
            "e-commerce": ["shop", "store", "e-commerce", "vente", "boutique"],
            "fintech": ["finance", "bank", "payment", "crypto", "fintech"],
            "healthcare": ["health", "medical", "patient", "hospital", "santé"],
            "education": ["education", "learning", "school", "formation"],
            "entertainment": ["game", "media", "entertainment", "streaming"],
            "productivity": ["productivity", "task", "management", "planning"]
        }
        
        text = (name + " " + description).lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        # Fallback basé sur template
        template_domains = {
            ProjectTemplate.WEB_APP: "web",
            ProjectTemplate.MOBILE_APP: "mobile",
            ProjectTemplate.AI_PROJECT: "ai",
            ProjectTemplate.MARKETING: "marketing"
        }
        
        return template_domains.get(template, "general")
    
    def _initialize_project_resources(self, project_id: str, template_data: Dict):
        """Initialisation intelligente des ressources projet"""
        
        project = self.projects[project_id]
        allocations = []
        
        # Allocation humaine
        allocations.append(ResourceAllocation(
            project_id=project_id,
            resource_type=ResourceType.HUMAN,
            allocated=project.team_size,
            capacity=project.team_size * 1.2,  # 20% marge
            efficiency=0.85
        ))
        
        # Allocation budgétaire
        allocations.append(ResourceAllocation(
            project_id=project_id,
            resource_type=ResourceType.BUDGET,
            allocated=project.budget,
            capacity=project.budget,
            efficiency=1.0
        ))
        
        # Allocation temps
        duration_days = (project.end_date - project.start_date).days
        allocations.append(ResourceAllocation(
            project_id=project_id,
            resource_type=ResourceType.TIME,
            allocated=duration_days,
            capacity=duration_days,
            efficiency=0.9
        ))
        
        self.resource_allocations[project_id] = allocations
    
    def get_portfolio_overview(self) -> Dict[str, Any]:
        """Vue d'ensemble intelligente du portefeuille"""
        
        if not self.projects:
            return {"message": "Aucun projet dans le portefeuille"}
        
        # Mise à jour métriques
        self._update_portfolio_metrics()
        
        # Classification des projets
        project_clusters = self._cluster_projects()
        
        # Analyse de santé
        health_analysis = self._analyze_portfolio_health()
        
        # Recommandations
        recommendations = self._generate_portfolio_recommendations()
        
        return {
            "summary": {
                "total_projects": len(self.projects),
                "active_projects": len([p for p in self.projects.values() if p.status == ProjectStatus.IN_PROGRESS]),
                "total_budget": sum(p.budget for p in self.projects.values()),
                "avg_health_score": np.mean([p.health_score for p in self.projects.values()]),
                "portfolio_value": self._calculate_portfolio_value()
            },
            "projects_by_status": self._group_projects_by_status(),
            "projects_by_priority": self._group_projects_by_priority(),
            "resource_utilization": self._calculate_resource_utilization(),
            "health_analysis": health_analysis,
            "project_clusters": project_clusters,
            "recommendations": recommendations,
            "timeline": self._generate_portfolio_timeline()
        }
    
    def _cluster_projects(self) -> Dict[str, List[str]]:
        """Clustering intelligent des projets"""
        
        if len(self.projects) < 2:
            return {"all": list(self.projects.keys())}
        
        # Features pour clustering
        features = []
        project_ids = []
        
        for project_id, project in self.projects.items():
            features.append([
                project.budget,
                project.team_size,
                (project.end_date - project.start_date).days if project.end_date else 30,
                project.priority.value.__hash__() % 10,
                project.progress,
                project.health_score
            ])
            project_ids.append(project_id)
        
        features = np.array(features)
        features_scaled = self.scaler.fit_transform(features)
        
        # Clustering
        n_clusters = min(3, len(self.projects))
        self.cluster_model.n_clusters = n_clusters
        clusters = self.cluster_model.fit_predict(features_scaled)
        
        # Regroupement
        cluster_groups = {}
        cluster_names = ["Projets Stratégiques", "Projets Opérationnels", "Projets Exploratoires"]
        
        for i, cluster_id in enumerate(clusters):
            cluster_name = cluster_names[cluster_id] if cluster_id < len(cluster_names) else f"Cluster {cluster_id}"
            if cluster_name not in cluster_groups:
                cluster_groups[cluster_name] = []
            cluster_groups[cluster_name].append(project_ids[i])
        
        return cluster_groups
    
    def _analyze_portfolio_health(self) -> Dict[str, Any]:
        """Analyse intelligente de la santé du portefeuille"""
        
        projects_data = list(self.projects.values())
        
        # Métriques de santé
        health_scores = [p.health_score for p in projects_data]
        progress_scores = [p.progress for p in projects_data]
        
        # Détection problèmes
        problematic_projects = [p for p in projects_data if p.health_score < 70]
        stalled_projects = [p for p in projects_data if p.progress < 10 and p.status == ProjectStatus.IN_PROGRESS]
        
        # Analyse des risques
        risk_analysis = self._analyze_portfolio_risks()
        
        return {
            "overall_health": np.mean(health_scores) if health_scores else 0,
            "avg_progress": np.mean(progress_scores) if progress_scores else 0,
            "health_distribution": {
                "excellent": len([s for s in health_scores if s >= 90]),
                "good": len([s for s in health_scores if 70 <= s < 90]),
                "warning": len([s for s in health_scores if 50 <= s < 70]),
                "critical": len([s for s in health_scores if s < 50])
            },
            "problematic_projects": [{"id": p.id, "name": p.name, "health": p.health_score} for p in problematic_projects],
            "stalled_projects": [{"id": p.id, "name": p.name, "progress": p.progress} for p in stalled_projects],
            "risk_analysis": risk_analysis
        }
    
    def _analyze_portfolio_risks(self) -> Dict[str, Any]:
        """Analyse des risques transversaux du portefeuille"""
        
        risks = []
        
        # Risque de sur-allocation ressources
        total_team_demand = sum(p.team_size for p in self.projects.values() if p.status == ProjectStatus.IN_PROGRESS)
        if total_team_demand > 20:  # Seuil arbitraire
            risks.append({
                "type": "resource_overallocation",
                "severity": "high",
                "description": f"Sur-allocation équipes: {total_team_demand} personnes requises",
                "impact": "Retards potentiels sur tous les projets actifs"
            })
        
        # Risque budgétaire
        total_budget = sum(p.budget for p in self.projects.values() if p.status != ProjectStatus.CANCELLED)
        if total_budget > 1000000:  # 1M threshold
            risks.append({
                "type": "budget_concentration",
                "severity": "medium",
                "description": f"Concentration budgétaire élevée: {total_budget:,.0f}€",
                "impact": "Risque financier concentré"
            })
        
        # Risque temporel
        concurrent_projects = len([p for p in self.projects.values() if p.status == ProjectStatus.IN_PROGRESS])
        if concurrent_projects > 5:
            risks.append({
                "type": "parallel_execution",
                "severity": "medium", 
                "description": f"Trop de projets en parallèle: {concurrent_projects}",
                "impact": "Dilution de l'attention et des ressources"
            })
        
        return {
            "total_risks": len(risks),
            "risk_level": "high" if any(r["severity"] == "high" for r in risks) else "medium" if risks else "low",
            "risks": risks
        }
    
    def _generate_portfolio_recommendations(self) -> List[Dict[str, str]]:
        """Génération de recommandations intelligentes"""
        
        recommendations = []
        
        # Analyse des projets
        active_projects = [p for p in self.projects.values() if p.status == ProjectStatus.IN_PROGRESS]
        
        # Recommandation priorisation
        if len(active_projects) > 3:
            recommendations.append({
                "type": "prioritization",
                "title": "Réduire le nombre de projets actifs",
                "description": f"Vous avez {len(active_projects)} projets actifs. Concentrez-vous sur les 3 plus critiques.",
                "action": "Mettre en attente les projets moins prioritaires"
            })
        
        # Recommandation ressources
        total_budget = sum(p.budget for p in self.projects.values())
        if total_budget > 0:
            avg_budget = total_budget / len(self.projects)
            expensive_projects = [p for p in self.projects.values() if p.budget > avg_budget * 2]
            
            if expensive_projects:
                recommendations.append({
                    "type": "budget_optimization",
                    "title": "Optimiser l'allocation budgétaire",
                    "description": f"{len(expensive_projects)} projets consomment plus que la moyenne",
                    "action": "Réviser la répartition budgétaire"
                })
        
        # Recommandation innovation
        ai_projects = [p for p in self.projects.values() if p.template == ProjectTemplate.AI_PROJECT]
        if not ai_projects and len(self.projects) > 2:
            recommendations.append({
                "type": "innovation",
                "title": "Intégrer l'intelligence artificielle",
                "description": "Aucun projet IA dans votre portefeuille",
                "action": "Considérer l'ajout d'un projet d'innovation IA"
            })
        
        return recommendations
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimisation intelligente des ressources avec contraintes"""
        
        if not self.projects:
            return {"message": "Aucun projet à optimiser"}
        
        # Récupération des projets actifs
        active_projects = {pid: p for pid, p in self.projects.items() 
                          if p.status in [ProjectStatus.PLANNING, ProjectStatus.IN_PROGRESS]}
        
        if not active_projects:
            return {"message": "Aucun projet actif à optimiser"}
        
        # Matrice d'optimisation
        optimization_results = {}
        
        # Optimisation par type de ressource
        for resource_type in ResourceType:
            optimization_results[resource_type.value] = self._optimize_resource_type(active_projects, resource_type)
        
        # Recommandations globales
        global_recommendations = self._generate_optimization_recommendations(optimization_results)
        
        # Score d'efficacité
        efficiency_score = self._calculate_portfolio_efficiency()
        
        return {
            "optimization_results": optimization_results,
            "efficiency_score": efficiency_score,
            "recommendations": global_recommendations,
            "resource_conflicts": self._detect_resource_conflicts(),
            "optimal_allocation": self._calculate_optimal_allocation(active_projects)
        }
    
    def _optimize_resource_type(self, projects: Dict[str, Project], resource_type: ResourceType) -> Dict[str, Any]:
        """Optimisation spécifique par type de ressource"""
        
        allocations = []
        total_demand = 0
        
        for project_id, project in projects.items():
            if project_id in self.resource_allocations:
                for allocation in self.resource_allocations[project_id]:
                    if allocation.resource_type == resource_type:
                        allocations.append({
                            "project_id": project_id,
                            "project_name": project.name,
                            "current_allocation": allocation.allocated,
                            "capacity": allocation.capacity,
                            "efficiency": allocation.efficiency,
                            "priority_weight": self._get_priority_weight(project.priority)
                        })
                        total_demand += allocation.allocated
        
        # Calcul recommandations
        recommendations = []
        if total_demand > sum(a["capacity"] for a in allocations):
            recommendations.append("Capacité insuffisante - Augmenter les ressources ou étaler les projets")
        
        # Score d'utilisation
        utilization_score = min(100, (total_demand / max(sum(a["capacity"] for a in allocations), 1)) * 100)
        
        return {
            "allocations": allocations,
            "total_demand": total_demand,
            "total_capacity": sum(a["capacity"] for a in allocations),
            "utilization_score": utilization_score,
            "recommendations": recommendations
        }
    
    def _get_priority_weight(self, priority: ProjectPriority) -> float:
        """Conversion priorité en poids numérique"""
        weights = {
            ProjectPriority.LOW: 1.0,
            ProjectPriority.MEDIUM: 2.0,
            ProjectPriority.HIGH: 3.0,
            ProjectPriority.CRITICAL: 5.0
        }
        return weights.get(priority, 1.0)
    
    def analyze_project_dependencies(self) -> Dict[str, Any]:
        """Analyse intelligente des interdépendances projets"""
        
        if not self.dependencies:
            return {"message": "Aucune dépendance configurée"}
        
        # Création du graphe de dépendances
        G = nx.DiGraph()
        
        # Ajout des nœuds (projets)
        for project_id, project in self.projects.items():
            G.add_node(project_id, name=project.name, status=project.status.value)
        
        # Ajout des arêtes (dépendances)
        for dep in self.dependencies:
            G.add_edge(dep.from_project, dep.to_project, 
                      type=dep.dependency_type, strength=dep.strength)
        
        # Analyses du graphe
        analysis = {
            "graph_metrics": {
                "total_dependencies": len(self.dependencies),
                "connected_projects": len([n for n in G.nodes() if G.degree(n) > 0]),
                "isolated_projects": len([n for n in G.nodes() if G.degree(n) == 0])
            },
            "critical_path": self._find_critical_path(G),
            "bottlenecks": self._identify_bottlenecks(G),
            "dependency_risks": self._analyze_dependency_risks(G),
            "optimization_suggestions": self._suggest_dependency_optimizations(G)
        }
        
        return analysis
    
    def _find_critical_path(self, graph: nx.DiGraph) -> List[str]:
        """Identification du chemin critique"""
        try:
            # Algorithme de chemin le plus long (critique)
            if nx.is_directed_acyclic_graph(graph):
                longest_path = nx.dag_longest_path(graph)
                return longest_path
        except:
            pass
        
        return []
    
    def _identify_bottlenecks(self, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """Identification des goulots d'étranglement"""
        
        bottlenecks = []
        
        # Projets avec le plus de dépendances entrantes
        in_degrees = dict(graph.in_degree())
        max_in_degree = max(in_degrees.values()) if in_degrees else 0
        
        for project_id, in_degree in in_degrees.items():
            if in_degree >= max(2, max_in_degree * 0.8):  # 80% du maximum
                project = self.projects.get(project_id)
                bottlenecks.append({
                    "project_id": project_id,
                    "project_name": project.name if project else "Unknown",
                    "incoming_dependencies": in_degree,
                    "risk_level": "high" if in_degree >= 3 else "medium"
                })
        
        return bottlenecks
    
    def _analyze_dependency_risks(self, graph: nx.DiGraph) -> List[Dict[str, str]]:
        """Analyse des risques liés aux dépendances"""
        
        risks = []
        
        # Cycles dans les dépendances
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                risks.append({
                    "type": "circular_dependency",
                    "severity": "high",
                    "description": f"Dépendances circulaires détectées: {len(cycles)} cycles",
                    "impact": "Blocage potentiel de l'exécution"
                })
        except:
            pass
        
        # Projets critiques bloqués
        blocked_projects = []
        for project_id, project in self.projects.items():
            if project.status == ProjectStatus.ON_HOLD:
                # Vérifier impact sur autres projets
                dependents = list(graph.successors(project_id))
                if dependents:
                    blocked_projects.extend(dependents)
        
        if blocked_projects:
            risks.append({
                "type": "blocked_cascade",
                "severity": "medium",
                "description": f"{len(set(blocked_projects))} projets impactés par des blocages",
                "impact": "Effet domino sur le portefeuille"
            })
        
        return risks
    
    def _suggest_dependency_optimizations(self, graph: nx.DiGraph) -> List[Dict[str, str]]:
        """Suggestions d'optimisation des dépendances"""
        
        suggestions = []
        
        # Parallélisation possible
        independent_projects = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if len(independent_projects) > 1:
            suggestions.append({
                "type": "parallelization",
                "title": "Exécution en parallèle possible",
                "description": f"{len(independent_projects)} projets peuvent démarrer simultanément",
                "benefit": "Réduction du temps total de réalisation"
            })
        
        # Simplification des dépendances
        complex_dependencies = [n for n in graph.nodes() if graph.degree(n) > 4]
        if complex_dependencies:
            suggestions.append({
                "type": "simplification",
                "title": "Simplifier les dépendances complexes",
                "description": f"{len(complex_dependencies)} projets ont des dépendances complexes",
                "benefit": "Réduction des risques et amélioration de l'agilité"
            })
        
        return suggestions
    
    def portfolio_analytics(self) -> Dict[str, Any]:
        """Analytics avancées du portefeuille avec ML"""
        
        if not self.projects:
            return {"message": "Aucune donnée pour l'analyse"}
        
        # Analytics temporelles
        temporal_analysis = self._analyze_temporal_patterns()
        
        # Analytics financières  
        financial_analysis = self._analyze_financial_patterns()
        
        # Analytics de performance
        performance_analysis = self._analyze_performance_patterns()
        
        # Prédictions ML
        ml_predictions = self._generate_ml_predictions()
        
        # Benchmarking
        benchmarks = self._generate_portfolio_benchmarks()
        
        return {
            "temporal_analysis": temporal_analysis,
            "financial_analysis": financial_analysis,
            "performance_analysis": performance_analysis,
            "ml_predictions": ml_predictions,
            "benchmarks": benchmarks,
            "insights": self._generate_analytics_insights(),
            "trends": self._identify_portfolio_trends()
        }
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyse des patterns temporels"""
        
        projects = list(self.projects.values())
        
        # Durées par template
        duration_by_template = {}
        for project in projects:
            if project.end_date:
                duration = (project.end_date - project.start_date).days
                template_name = project.template.value
                
                if template_name not in duration_by_template:
                    duration_by_template[template_name] = []
                duration_by_template[template_name].append(duration)
        
        # Statistiques temporelles
        avg_durations = {template: np.mean(durations) for template, durations in duration_by_template.items()}
        
        # Charge temporelle
        temporal_load = self._calculate_temporal_load()
        
        return {
            "avg_duration_by_template": avg_durations,
            "temporal_load": temporal_load,
            "project_timeline": self._generate_portfolio_timeline(),
            "seasonal_patterns": self._detect_seasonal_patterns()
        }
    
    def _analyze_financial_patterns(self) -> Dict[str, Any]:
        """Analyse des patterns financiers"""
        
        projects = list(self.projects.values())
        
        # Distribution budgétaire
        budgets = [p.budget for p in projects]
        
        financial_metrics = {
            "total_portfolio_value": sum(budgets),
            "avg_project_budget": np.mean(budgets) if budgets else 0,
            "budget_distribution": {
                "min": min(budgets) if budgets else 0,
                "max": max(budgets) if budgets else 0,
                "std": np.std(budgets) if budgets else 0
            },
            "budget_by_template": self._calculate_budget_by_template(),
            "roi_analysis": self._analyze_portfolio_roi()
        }
        
        return financial_metrics
    
    def _generate_ml_predictions(self) -> Dict[str, Any]:
        """Prédictions ML pour le portefeuille"""
        
        if len(self.projects) < 3:
            return {"message": "Données insuffisantes pour les prédictions ML"}
        
        predictions = {}
        
        # Prédiction succès projets
        success_predictions = self._predict_project_success()
        predictions["success_forecast"] = success_predictions
        
        # Prédiction charge future
        load_predictions = self._predict_future_load()
        predictions["load_forecast"] = load_predictions
        
        # Prédiction budget
        budget_predictions = self._predict_budget_needs()
        predictions["budget_forecast"] = budget_predictions
        
        return predictions
    
    def get_portfolio_dashboard_data(self) -> Dict[str, Any]:
        """Données formatées pour le dashboard"""
        
        # Vue d'ensemble
        overview = self.get_portfolio_overview()
        
        # Métriques clés
        key_metrics = {
            "total_projects": len(self.projects),
            "active_projects": len([p for p in self.projects.values() if p.status == ProjectStatus.IN_PROGRESS]),
            "total_budget": f"{sum(p.budget for p in self.projects.values()):,.0f}€",
            "avg_health": f"{np.mean([p.health_score for p in self.projects.values()]):.1f}%" if self.projects else "0%",
            "completion_rate": f"{len([p for p in self.projects.values() if p.status == ProjectStatus.COMPLETED]) / max(len(self.projects), 1) * 100:.1f}%"
        }
        
        # Projets par statut (pour graphiques)
        status_distribution = {}
        for project in self.projects.values():
            status = project.status.value
            status_distribution[status] = status_distribution.get(status, 0) + 1
        
        # Timeline projets
        timeline_data = []
        for project in self.projects.values():
            if project.end_date:
                timeline_data.append({
                    "name": project.name,
                    "start": project.start_date.strftime("%Y-%m-%d"),
                    "end": project.end_date.strftime("%Y-%m-%d"),
                    "status": project.status.value,
                    "progress": project.progress
                })
        
        return {
            "overview": overview,
            "key_metrics": key_metrics,
            "status_distribution": status_distribution,
            "timeline_data": timeline_data,
            "recent_insights": self.insights_history[-5:] if self.insights_history else [],
            "resource_alerts": self._get_resource_alerts()
        }
    
    def _update_portfolio_metrics(self):
        """Mise à jour des métriques portfolio"""
        
        if not self.projects:
            return
        
        projects = list(self.projects.values())
        
        self.portfolio_metrics = {
            "total_budget": sum(p.budget for p in projects),
            "total_projects": len(projects),
            "avg_health_score": np.mean([p.health_score for p in projects]),
            "resource_efficiency": self._calculate_portfolio_efficiency(),
            "on_time_delivery": self._calculate_on_time_rate()
        }
    
    def _calculate_portfolio_efficiency(self) -> float:
        """Calcul de l'efficacité globale du portefeuille"""
        
        if not self.resource_allocations:
            return 0.0
        
        total_efficiency = 0.0
        total_weight = 0.0
        
        for project_id, allocations in self.resource_allocations.items():
            project = self.projects.get(project_id)
            if project:
                project_weight = self._get_priority_weight(project.priority)
                for allocation in allocations:
                    total_efficiency += allocation.efficiency * project_weight
                    total_weight += project_weight
        
        return (total_efficiency / max(total_weight, 1)) * 100
    
    # Méthodes utilitaires diverses...
    def _group_projects_by_status(self) -> Dict[str, int]:
        """Regroupement projets par statut"""
        groups = {}
        for project in self.projects.values():
            status = project.status.value
            groups[status] = groups.get(status, 0) + 1
        return groups
    
    def _group_projects_by_priority(self) -> Dict[str, int]:
        """Regroupement projets par priorité"""
        groups = {}
        for project in self.projects.values():
            priority = project.priority.value
            groups[priority] = groups.get(priority, 0) + 1
        return groups
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calcul utilisation des ressources"""
        utilization = {}
        
        for resource_type in ResourceType:
            total_allocated = 0.0
            total_capacity = 0.0
            
            for allocations in self.resource_allocations.values():
                for allocation in allocations:
                    if allocation.resource_type == resource_type:
                        total_allocated += allocation.allocated
                        total_capacity += allocation.capacity
            
            utilization[resource_type.value] = (total_allocated / max(total_capacity, 1)) * 100
        
        return utilization
    
    def _calculate_portfolio_value(self) -> float:
        """Calcul de la valeur totale du portefeuille"""
        
        total_value = 0.0
        
        for project in self.projects.values():
            # Valeur = Budget * Health Score * Priority Weight
            priority_multiplier = self._get_priority_weight(project.priority) / 2.0
            health_multiplier = project.health_score / 100.0
            
            project_value = project.budget * health_multiplier * priority_multiplier
            total_value += project_value
        
        return total_value
    
    def _generate_portfolio_timeline(self) -> List[Dict[str, Any]]:
        """Génération timeline du portefeuille"""
        
        timeline = []
        
        for project in self.projects.values():
            if project.end_date:
                timeline.append({
                    "project_id": project.id,
                    "name": project.name,
                    "start": project.start_date,
                    "end": project.end_date,
                    "status": project.status.value,
                    "priority": project.priority.value,
                    "progress": project.progress
                })
        
        # Tri par date de début
        timeline.sort(key=lambda x: x["start"])
        
        return timeline
    
    def _generate_project_creation_insights(self, project: Project):
        """Génération d'insights lors de la création projet"""
        
        insights = []
        
        # Insight sur la charge
        active_projects = len([p for p in self.projects.values() if p.status == ProjectStatus.IN_PROGRESS])
        if active_projects > 3:
            insights.append(PortfolioInsight(
                type="workload",
                severity="warning",
                title="Charge de travail élevée",
                description=f"Vous avez maintenant {active_projects + 1} projets actifs",
                affected_projects=[project.id],
                recommendation="Considérez la priorisation ou l'étalement dans le temps"
            ))
        
        # Insight sur le budget
        total_budget = sum(p.budget for p in self.projects.values())
        if total_budget > 500000:  # 500k threshold
            insights.append(PortfolioInsight(
                type="budget",
                severity="info",
                title="Budget portfolio conséquent",
                description=f"Budget total portefeuille: {total_budget:,.0f}€",
                affected_projects=[project.id],
                recommendation="Assurez-vous d'avoir les ressources financières appropriées"
            ))
        
        self.insights_history.extend(insights)
    
    def _get_resource_alerts(self) -> List[Dict[str, str]]:
        """Génération d'alertes ressources"""
        
        alerts = []
        
        # Vérification sur-allocation
        utilization = self._calculate_resource_utilization()
        
        for resource_type, util_rate in utilization.items():
            if util_rate > 90:
                alerts.append({
                    "type": "overallocation",
                    "resource": resource_type,
                    "message": f"Sur-allocation {resource_type}: {util_rate:.1f}%",
                    "severity": "high"
                })
            elif util_rate > 75:
                alerts.append({
                    "type": "warning",
                    "resource": resource_type,
                    "message": f"Utilisation élevée {resource_type}: {util_rate:.1f}%",
                    "severity": "medium"
                })
        
        return alerts
    
    # Méthodes de stubs pour fonctionnalités avancées
    def _detect_seasonal_patterns(self) -> Dict[str, Any]:
        return {"message": "Analyse saisonnière non disponible avec peu de données"}
    
    def _calculate_budget_by_template(self) -> Dict[str, float]:
        budget_by_template = {}
        for project in self.projects.values():
            template = project.template.value
            if template not in budget_by_template:
                budget_by_template[template] = 0.0
            budget_by_template[template] += project.budget
        return budget_by_template
    
    def _analyze_portfolio_roi(self) -> Dict[str, float]:
        return {"estimated_roi": 15.5, "roi_range": "10-25%"}
    
    def _predict_project_success(self) -> Dict[str, Any]:
        return {"avg_success_probability": 0.78, "high_risk_projects": 1}
    
    def _predict_future_load(self) -> Dict[str, Any]:
        return {"next_month_load": "Medium", "peak_period": "Q2 2024"}
    
    def _predict_budget_needs(self) -> Dict[str, Any]:
        return {"predicted_budget_next_quarter": 150000, "variance": 20}
    
    def _calculate_temporal_load(self) -> Dict[str, int]:
        return {"current_month": 3, "next_month": 2, "following_month": 1}
    
    def _analyze_performance_patterns(self) -> Dict[str, Any]:
        return {"avg_delivery_time": "85% on time", "quality_score": 8.2}
    
    def _generate_portfolio_benchmarks(self) -> Dict[str, Any]:
        return {"industry_average_success": 0.72, "your_portfolio": 0.78}
    
    def _generate_analytics_insights(self) -> List[str]:
        return [
            "Vos projets IA ont un taux de succès 15% supérieur à la moyenne",
            "La charge optimale semble être de 3 projets simultanés",
            "Les projets avec équipes < 5 personnes livrent plus rapidement"
        ]
    
    def _identify_portfolio_trends(self) -> Dict[str, str]:
        return {
            "budget_trend": "increasing",
            "team_size_trend": "stable", 
            "success_rate_trend": "improving"
        }
    
    def _calculate_on_time_rate(self) -> float:
        completed_projects = [p for p in self.projects.values() if p.status == ProjectStatus.COMPLETED]
        if not completed_projects:
            return 0.0
        
        # Simulation - dans la vraie vie, on comparerait les dates réelles vs prévues
        return 85.0  # 85% des projets livrés à temps
    
    def _generate_optimization_recommendations(self, results: Dict) -> List[str]:
        recommendations = []
        
        for resource_type, data in results.items():
            if data.get("utilization_score", 0) > 90:
                recommendations.append(f"Réduire la charge sur les ressources {resource_type}")
            elif data.get("utilization_score", 0) < 50:
                recommendations.append(f"Opportunité d'augmenter l'utilisation des ressources {resource_type}")
        
        return recommendations
    
    def _detect_resource_conflicts(self) -> List[Dict[str, str]]:
        conflicts = []
        
        # Simulation de détection de conflits
        active_projects = [p for p in self.projects.values() if p.status == ProjectStatus.IN_PROGRESS]
        if len(active_projects) > 3:
            conflicts.append({
                "type": "team_overlap",
                "description": "Potentiel conflit d'allocation équipe entre projets actifs",
                "affected_projects": [p.name for p in active_projects[:2]]
            })
        
        return conflicts
    
    def _calculate_optimal_allocation(self, projects: Dict[str, Project]) -> Dict[str, Any]:
        """Calcul de l'allocation optimale théorique"""
        
        total_priority_weight = sum(self._get_priority_weight(p.priority) for p in projects.values())
        
        optimal_allocation = {}
        for project_id, project in projects.items():
            weight = self._get_priority_weight(project.priority)
            allocation_ratio = weight / total_priority_weight if total_priority_weight > 0 else 0
            
            optimal_allocation[project_id] = {
                "project_name": project.name,
                "recommended_allocation": allocation_ratio,
                "current_priority": project.priority.value
            }
        
        return optimal_allocation


# Instance globale
ai_portfolio_manager = AIPortfolioManager()

# Fonctions d'initialisation
def initialize_portfolio_manager():
    """Initialisation du gestionnaire de portefeuille"""
    print("[PORTFOLIO] Initialisation du Portfolio Manager IA...")
    return ai_portfolio_manager

def get_portfolio_manager():
    """Récupération de l'instance du gestionnaire"""
    return ai_portfolio_manager