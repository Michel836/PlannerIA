from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PlannerIA API", 
    version="0.1.0",
    description="API pour la génération automatique de plans de projet avec IA",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ajouter le répertoire racine du projet au Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Initialisation sécurisée de PlannerIA
planner_ai = None
initialization_error = None

try:
    from crew import PlannerIA
    planner_ai = PlannerIA()
    logger.info("PlannerIA initialized successfully")
except Exception as e:
    initialization_error = str(e)
    logger.error(f"PlannerIA initialization failed: {e}")

# Modèles Pydantic pour les requêtes et réponses
class PlanRequest(BaseModel):
    brief: str = Field(..., description="Description textuelle du projet à planifier", min_length=10)
    config_path: str = Field(default="config/default.yaml", description="Chemin vers le fichier de configuration")

class PlanResponse(BaseModel):
    success: bool
    run_id: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class PlanSummary(BaseModel):
    run_id: str
    timestamp: str

# Nouveaux modèles pour endpoints IA
class AIEstimationRequest(BaseModel):
    tasks: List[Dict[str, Any]] = Field(..., description="Liste des tâches à estimer")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Contexte du projet")

class AIRiskRequest(BaseModel):
    risks: List[Dict[str, Any]] = Field(..., description="Liste des risques à analyser")
    project_context: Optional[Dict[str, Any]] = Field(default=None, description="Contexte du projet")

class AISuggestRequest(BaseModel):
    current_plan: Dict[str, Any] = Field(..., description="Plan actuel")
    optimization_goal: str = Field(default="time", description="Objectif d'optimisation: time, cost, quality")

class AIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class RAGRequest(BaseModel):
    question: str = Field(..., description="Question à poser au système RAG")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Contexte du projet")
    brief: str
    status: str
    title: Optional[str] = None

# Events
@app.on_event("startup")
async def startup_event():
    if planner_ai:
        logger.info("PlannerIA API started successfully")
        # Créer les dossiers nécessaires
        Path("data/runs").mkdir(parents=True, exist_ok=True)
        Path("data/feedback").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
    else:
        logger.warning("PlannerIA API started with initialization errors")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("PlannerIA API shutting down")

# Endpoints de base
@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    if initialization_error:
        return {
            "status": "error",
            "message": "PlannerIA failed to initialize",
            "error": initialization_error
        }
    return {
        "message": "PlannerIA API running", 
        "status": "ok",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Vérification de l'état de santé de l'API"""
    return {
        "status": "healthy" if planner_ai else "degraded",
        "planner_ready": planner_ai is not None,
        "initialization_error": initialization_error,
        "timestamp": Path("data/runs").exists()
    }

@app.get("/debug")
async def debug_info():
    """Endpoint de debug pour diagnostiquer les problèmes de chemin"""
    return {
        "python_path": sys.path[:3],  # Limiter la sortie
        "project_root": str(project_root),
        "current_file": str(__file__),
        "planner_available": planner_ai is not None,
        "error": initialization_error,
        "data_dirs_exist": {
            "runs": Path("data/runs").exists(),
            "feedback": Path("data/feedback").exists(),
            "logs": Path("logs").exists()
        }
    }

# Endpoints principaux pour la génération de plans
@app.post("/generate-plan", response_model=PlanResponse)
async def generate_plan(request: PlanRequest):
    """Génère un plan de projet complet à partir d'un brief"""
    if not planner_ai:
        raise HTTPException(status_code=503, detail="PlannerIA not initialized")
    
    try:
        logger.info(f"Generating plan for brief: {request.brief[:50]}...")
        plan = planner_ai.generate_plan(request.brief)
        
        return PlanResponse(
            success=True,
            run_id=plan.get("run_id"),
            plan=plan
        )
    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        return PlanResponse(
            success=False,
            error=str(e)
        )

@app.get("/plans/{run_id}")
async def get_plan(run_id: str):
    """Récupère un plan sauvegardé par son ID"""
    plan_path = Path(f"data/runs/{run_id}/plan.json")
    
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="Plan not found")
    
    try:
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan = json.load(f)
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading plan: {str(e)}")

@app.get("/plans", response_model=List[PlanSummary])
async def list_plans():
    """Liste tous les plans générés"""
    runs_dir = Path("data/runs")
    if not runs_dir.exists():
        return []
    
    plans = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            plan_file = run_dir / "plan.json"
            if plan_file.exists():
                try:
                    with open(plan_file, 'r', encoding='utf-8') as f:
                        plan_data = json.load(f)
                    
                    overview = plan_data.get("project_overview", {})
                    plans.append(PlanSummary(
                        run_id=run_dir.name,
                        timestamp=plan_data.get("timestamp", ""),
                        brief=plan_data.get("brief", "")[:100] + "..." if len(plan_data.get("brief", "")) > 100 else plan_data.get("brief", ""),
                        status=plan_data.get("status", "completed"),
                        title=overview.get("title")
                    ))
                except Exception as e:
                    logger.warning(f"Error reading plan {run_dir.name}: {e}")
                    continue
    
    return sorted(plans, key=lambda x: x.timestamp, reverse=True)

@app.delete("/plans/{run_id}")
async def delete_plan(run_id: str):
    """Supprime un plan et ses fichiers associés"""
    plan_dir = Path(f"data/runs/{run_id}")
    
    if not plan_dir.exists():
        raise HTTPException(status_code=404, detail="Plan not found")
    
    try:
        import shutil
        shutil.rmtree(plan_dir)
        return {"message": f"Plan {run_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting plan: {str(e)}")

@app.get("/plans/{run_id}/export/{format}")
async def export_plan(run_id: str, format: str):
    """Exporte un plan dans différents formats (json, csv, md)"""
    if format not in ["json", "csv", "md"]:
        raise HTTPException(status_code=400, detail="Format not supported. Use: json, csv, md")
    
    plan_dir = Path(f"data/runs/{run_id}")
    if not plan_dir.exists():
        raise HTTPException(status_code=404, detail="Plan not found")
    
    if format == "json":
        file_path = plan_dir / "plan.json"
    elif format == "csv":
        file_path = plan_dir / "plan.csv"
    elif format == "md":
        file_path = plan_dir / "rapport.md"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Export format {format} not available for this plan")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=f"plan_{run_id}.{format}",
        media_type="application/octet-stream"
    )

# Endpoints utilitaires
@app.get("/stats")
async def get_stats():
    """Statistiques sur l'utilisation de l'API"""
    runs_dir = Path("data/runs")
    if not runs_dir.exists():
        return {"total_plans": 0, "successful_plans": 0, "failed_plans": 0}
    
    total = 0
    successful = 0
    failed = 0
    
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir():
            total += 1
            plan_file = run_dir / "plan.json"
            if plan_file.exists():
                try:
                    with open(plan_file, 'r', encoding='utf-8') as f:
                        plan_data = json.load(f)
                    if plan_data.get("status") == "failed":
                        failed += 1
                    else:
                        successful += 1
                except:
                    failed += 1
    
    return {
        "total_plans": total,
        "successful_plans": successful,
        "failed_plans": failed,
        "success_rate": round((successful / total * 100) if total > 0 else 0, 2)
    }

# ============================================================================
# 🧠 NOUVEAUX ENDPOINTS IA
# ============================================================================

@app.post("/ai/estimate", response_model=AIResponse)
async def ai_estimate_tasks(request: AIEstimationRequest):
    """Estimation intelligente des tâches avec ML"""
    if not planner_ai:
        raise HTTPException(status_code=503, detail="PlannerIA not initialized")
    
    try:
        from src.project_planner.ml.estimator_model import EstimatorModel
        estimator = EstimatorModel()
        
        predictions = estimator.predict_multiple_tasks(request.tasks)
        
        return AIResponse(
            success=True,
            data={
                "predictions": predictions,
                "context": request.context,
                "method": "ml_estimation"
            }
        )
    except Exception as e:
        return AIResponse(
            success=False,
            error=str(e)
        )

@app.post("/ai/analyze-risks", response_model=AIResponse)
async def ai_analyze_risks(request: AIRiskRequest):
    """Analyse intelligente des risques avec ML"""
    if not planner_ai:
        raise HTTPException(status_code=503, detail="PlannerIA not initialized")
    
    try:
        from src.project_planner.ml.risk_model import RiskAssessmentModel
        risk_model = RiskAssessmentModel()
        
        risk_assessment = risk_model.predict_risk_assessment(
            request.risks, 
            request.project_context
        )
        
        return AIResponse(
            success=True,
            data={
                "risk_assessment": risk_assessment,
                "context": request.project_context,
                "method": "ml_risk_analysis"
            }
        )
    except Exception as e:
        return AIResponse(
            success=False,
            error=str(e)
        )

@app.post("/ai/optimize", response_model=AIResponse)
async def ai_optimize_plan(request: AISuggestRequest):
    """Optimisation intelligente des plans"""
    if not planner_ai:
        raise HTTPException(status_code=503, detail="PlannerIA not initialized")
    
    try:
        from src.project_planner.core.optimizer import ProjectOptimizer
        optimizer = ProjectOptimizer()
        
        optimized_plan = optimizer.optimize_plan(request.current_plan)
        
        return AIResponse(
            success=True,
            data={
                "optimized_plan": optimized_plan,
                "optimization_goal": request.optimization_goal,
                "method": "graph_optimization"
            }
        )
    except Exception as e:
        return AIResponse(
            success=False,
            error=str(e)
        )

@app.post("/ai/generate-plan", response_model=AIResponse)
async def ai_generate_complete_plan(request: PlanRequest):
    """Génération complète de plan avec LLM + ML"""
    if not planner_ai:
        raise HTTPException(status_code=503, detail="PlannerIA not initialized")
    
    try:
        # Génération avec LLM
        plan_data = planner_ai.generate_plan(request.brief)
        
        return AIResponse(
            success=True,
            data={
                "plan": plan_data,
                "brief": request.brief,
                "run_id": plan_data.get("run_id"),
                "method": "llm_generation"
            }
        )
    except Exception as e:
        return AIResponse(
            success=False,
            error=str(e)
        )

@app.get("/ai/models/status")
async def ai_models_status():
    """Status des modèles IA disponibles"""
    status = {
        "llm_available": planner_ai is not None,
        "models": {}
    }
    
    # Test EstimatorModel
    try:
        from src.project_planner.ml.estimator_model import EstimatorModel
        EstimatorModel()
        status["models"]["estimator"] = "available"
    except Exception as e:
        status["models"]["estimator"] = f"error: {str(e)}"
    
    # Test RiskModel
    try:
        from src.project_planner.ml.risk_model import RiskAssessmentModel
        RiskAssessmentModel()
        status["models"]["risk"] = "available"
    except Exception as e:
        status["models"]["risk"] = f"error: {str(e)}"
    
    # Test MonteCarloEstimator
    try:
        from src.project_planner.ml.monte_carlo_estimator import MonteCarloEstimator
        MonteCarloEstimator()
        status["models"]["monte_carlo"] = "available"
    except Exception as e:
        status["models"]["monte_carlo"] = f"error: {str(e)}"
    
    return status

@app.post("/ai/rag", response_model=AIResponse)
async def ai_rag_query(request: RAGRequest):
    """Requête RAG - Question/Réponse avec base de connaissances"""
    if not planner_ai:
        raise HTTPException(status_code=503, detail="PlannerIA not initialized")
    
    try:
        from src.project_planner.dashboard.components.intelligence_module import PlannerRAGSystem
        rag_system = PlannerRAGSystem()
        
        # Question RAG
        rag_response = rag_system.ask_question(request.question, request.context or {})
        
        return AIResponse(
            success=True,
            data={
                "rag_response": rag_response,
                "method": "retrieval_augmented_generation"
            }
        )
    except Exception as e:
        return AIResponse(
            success=False,
            error=str(e)
        )