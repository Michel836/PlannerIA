"""
Plan Validator - Validates and corrects project plans using Pydantic and JSON Schema
Optimized: Pydantic v2-compatible, logging, helpers, defensive coding.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union, Literal
from pathlib import Path
from datetime import datetime, date

from pydantic import BaseModel, Field, model_validator

import jsonschema

# Configure logging (caller can reconfigure)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --- Helpers -----------------------------------------------------------------
def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        v = int(value)
    except Exception:
        return default
    return max(minimum, min(v, maximum))


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


# --- Domain models -----------------------------------------------------------
StatusStr = Literal["valid", "corrected", "failed"]


class ProjectOverview(BaseModel):
    title: str = Field(..., max_length=200)
    description: str = Field(..., max_length=2000)
    objectives: Optional[List[str]] = Field(default_factory=list)
    success_criteria: Optional[List[str]] = Field(default_factory=list)
    total_duration: Optional[float] = Field(None, ge=0)
    total_cost: Optional[float] = Field(None, ge=0)
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class Task(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    duration: float = Field(..., ge=0)
    duration_unit: str = Field(default="days", pattern="^(hours|days|weeks)$")
    effort: Optional[float] = Field(None, ge=0)
    cost: Optional[float] = Field(None, ge=0)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: str = Field(default="not_started", pattern="^(not_started|in_progress|completed|blocked)$")
    priority: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    assigned_resources: Optional[List[str]] = Field(default_factory=list)
    deliverables: Optional[List[str]] = Field(default_factory=list)


class Phase(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    tasks: List[Task] = Field(default_factory=list)


class WBS(BaseModel):
    phases: List[Phase] = Field(default_factory=list)


class Dependency(BaseModel):
    predecessor: str
    successor: str
    type: str = Field(default="finish_to_start", pattern="^(finish_to_start|start_to_start|finish_to_finish|start_to_finish)$")
    lag: float = Field(default=0.0)


class Risk(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    category: str = Field(default="technical", pattern="^(technical|schedule|budget|resource|external|quality)$")
    probability: int = Field(..., ge=1, le=5)
    impact: int = Field(..., ge=1, le=5)
    risk_score: Optional[int] = Field(None, ge=1, le=25)
    mitigation_strategy: Optional[str] = ""
    contingency_plan: Optional[str] = ""
    owner: Optional[str] = ""
    status: str = Field(default="identified", pattern="^(identified|assessed|mitigated|closed)$")

    @model_validator(mode="before")
    def compute_and_check(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pydantic v2 pre-validator.

        - If the caller provided an explicit (possibly invalid) probability/impact:
            -> raise ValueError when out of range (keeps direct-instantiation tests happy)
        - If fields missing: set sensible defaults (3) and compute risk_score if absent.
        Note: PlanValidator.validate_risks clamps values BEFORE creating Risk, so
        that path will not raise.
        """
        if not isinstance(values, dict):
            return values or {}

        # handle probability
        raw_prob = values.get("probability", None)
        if raw_prob is None:
            prob = 3
        else:
            try:
                prob = int(raw_prob)
            except Exception:
                raise ValueError("probability must be an integer between 1 and 5")
            # if user explicitly passed an out-of-range value -> raise (test expects this)
            if prob < 1 or prob > 5:
                raise ValueError("probability must be between 1 and 5")

        # handle impact
        raw_imp = values.get("impact", None)
        if raw_imp is None:
            imp = 3
        else:
            try:
                imp = int(raw_imp)
            except Exception:
                raise ValueError("impact must be an integer between 1 and 5")
            if imp < 1 or imp > 5:
                raise ValueError("impact must be between 1 and 5")

        values["probability"] = prob
        values["impact"] = imp

        # compute risk_score if not present or falsy
        if not values.get("risk_score"):
            values["risk_score"] = prob * imp

        return values


class Resource(BaseModel):
    id: str
    name: str
    type: str = Field(..., pattern="^(human|equipment|material|facility)$")
    cost_per_unit: Optional[float] = Field(None, ge=0)
    availability: Optional[float] = Field(None, ge=0, le=1)
    skills: Optional[List[str]] = Field(default_factory=list)


class Milestone(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    date: date
    deliverables: Optional[List[str]] = Field(default_factory=list)
    criteria: Optional[List[str]] = Field(default_factory=list)


class RAGCitation(BaseModel):
    source: str
    content: str
    relevance_score: Optional[float] = Field(None, ge=0, le=1)


class Documentation(BaseModel):
    report: Optional[str] = ""
    slides: Optional[str] = ""
    executive_summary: Optional[str] = ""


class Metadata(BaseModel):
    model_used: Optional[str] = ""
    processing_time: Optional[float] = None
    validation_status: StatusStr = Field(default="valid", pattern="^(valid|corrected|failed)$")


class ProjectPlan(BaseModel):
    run_id: str
    timestamp: datetime
    brief: Optional[str] = ""
    project_overview: ProjectOverview
    wbs: WBS = Field(default_factory=WBS)
    tasks: Optional[List[Task]] = Field(default_factory=list)
    dependencies: Optional[List[Dependency]] = Field(default_factory=list)
    risks: Optional[List[Risk]] = Field(default_factory=list)
    resources: Optional[List[Resource]] = Field(default_factory=list)
    milestones: Optional[List[Milestone]] = Field(default_factory=list)
    critical_path: Optional[List[str]] = Field(default_factory=list)
    rag_citations: Optional[List[RAGCitation]] = Field(default_factory=list)
    documentation: Optional[Documentation] = Field(default_factory=Documentation)
    metadata: Optional[Metadata] = Field(default_factory=Metadata)


# --- Validator ----------------------------------------------------------------
class PlanValidator:
    """Validates and corrects project plans using Pydantic models and JSON Schema"""

    def __init__(self, schema_source: Optional[Union[str, Path, Dict[str, Any]]] = "schema/plan.schema.json"):
        """
        schema_source can be:
          - a path (str or Path) to a JSON schema file, or
          - a dict already containing a schema, or
          - None (no schema validation)
        """
        self.schema = {}
        if isinstance(schema_source, dict):
            self.schema = schema_source
        elif schema_source:
            self.schema_path = Path(schema_source)
            self.schema = self._load_schema(self.schema_path)

    def _load_schema(self, path: Path) -> Dict[str, Any]:
        """Load JSON schema from file path (returns {} if missing/invalid)"""
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load schema %s: %s", path, exc)
        return {}

    @staticmethod
    def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
        """Convert pydantic model to dict with compatibility for v1/v2"""
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def validate_plan(self, plan_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate and correct a project plan.

        - Raises TypeError when plan_data is None (keeps previous tests behaviour).
        - If plan_data is not dict, returns a minimal failed plan.
        """
        if plan_data is None:
            raise TypeError("plan_data must be a dict, got None")

        if not isinstance(plan_data, dict):
            logger.info("Received non-dict plan_data, returning minimal failed plan.")
            return self._create_minimal_valid_plan({})

        # Work on shallow copy to avoid mutating caller's dict
        data = plan_data.copy()
        original_empty = len(data) == 0
        made_corrections = False

        # JSON Schema validation & auto-corrections
        if self.schema:
            try:
                jsonschema.validate(data, self.schema)
            except jsonschema.ValidationError as e:
                logger.warning("JSON Schema validation warning: %s", e.message)
                data = self._apply_schema_corrections(data, e)
                made_corrections = True

        # Sanitize metadata.validation_status (remove invalid values)
        if isinstance(data.get("metadata"), dict):
            status = data["metadata"].get("validation_status")
            if status is not None and status not in {"valid", "corrected", "failed"}:
                # remove invalid value so Pydantic will apply default
                data = data.copy()
                data["metadata"] = data["metadata"].copy()
                data["metadata"].pop("validation_status", None)

        # Pydantic validation (main)
        try:
            validated = ProjectPlan(**data)
            result = self._model_to_dict(validated)

            result.setdefault("metadata", {})
            if original_empty:
                # explicit policy: completely empty input -> failed
                result["metadata"]["validation_status"] = "failed"
            else:
                result["metadata"]["validation_status"] = "corrected" if made_corrections else "valid"

            return result

        except Exception as e:
            logger.warning("Pydantic validation error: %s", e)
            return self._apply_fallback_corrections(data, str(e))

    def _apply_schema_corrections(self, plan_data: Dict[str, Any], error: jsonschema.ValidationError) -> Dict[str, Any]:
        """Apply defensive corrections for common schema validation errors"""
        corrected = plan_data.copy() if isinstance(plan_data, dict) else {}

        corrected.setdefault("run_id", str(datetime.now().timestamp()))
        corrected.setdefault("timestamp", datetime.now().isoformat())
        corrected.setdefault("project_overview", {
            "title": corrected.get("brief", "Generated Project Plan"),
            "description": corrected.get("brief", "No description provided"),
        })
        corrected.setdefault("wbs", {"phases": []})

        # sanitize numeric fields in overview when present as strings
        overview = _ensure_dict(corrected.get("project_overview"))
        if isinstance(overview.get("total_duration"), str):
            try:
                overview["total_duration"] = float(overview["total_duration"])
            except Exception:
                overview["total_duration"] = 0.0
        if isinstance(overview.get("total_cost"), str):
            try:
                overview["total_cost"] = float(overview["total_cost"])
            except Exception:
                overview["total_cost"] = 0.0
        corrected["project_overview"] = overview

        corrected.setdefault("metadata", {})
        corrected["metadata"]["validation_status"] = "corrected"

        return corrected

    def _apply_fallback_corrections(self, plan_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Return a minimal plan when Pydantic cannot validate the provided data"""
        corrected = self._create_minimal_valid_plan(plan_data if isinstance(plan_data, dict) else {})
        corrected.setdefault("metadata", {})
        corrected["metadata"]["validation_status"] = "failed"
        corrected["metadata"]["validation_error"] = error_msg
        return corrected

    def _create_minimal_valid_plan(self, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a minimal valid plan dictionary from any input (defensive)"""
        if not isinstance(original_data, dict):
            original_data = {}

        po = _ensure_dict(original_data.get("project_overview"))
        wbs = _ensure_dict(original_data.get("wbs"))

        return {
            "run_id": original_data.get("run_id", str(datetime.now().timestamp())),
            "timestamp": original_data.get("timestamp", datetime.now().isoformat()),
            "brief": original_data.get("brief", ""),
            "project_overview": {
                "title": po.get("title", "Generated Project Plan"),
                "description": po.get("description", original_data.get("brief", "No description provided")),
            },
            "wbs": {"phases": wbs.get("phases", [])},
            "tasks": original_data.get("tasks", []),
            "dependencies": original_data.get("dependencies", []),
            "risks": original_data.get("risks", []),
            "resources": original_data.get("resources", []),
            "milestones": original_data.get("milestones", []),
            "critical_path": original_data.get("critical_path", []),
            "rag_citations": original_data.get("rag_citations", []),
            "documentation": original_data.get("documentation", {}),
            "metadata": {
                "model_used": _ensure_dict(original_data.get("metadata")).get("model_used", ""),
                "processing_time": _ensure_dict(original_data.get("metadata")).get("processing_time"),
                "validation_status": "failed",
            },
        }

    def validate_tasks(self, tasks: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Validate and correct individual tasks"""
        if not tasks:
            return []

        result: List[Dict[str, Any]] = []
        for i, raw in enumerate(tasks):
            item = _ensure_dict(raw)
            item.setdefault("id", f"task_{i+1}")
            item.setdefault("name", f"Task {i+1}")
            item.setdefault("duration", 1.0)

            try:
                t = Task(**item)
                result.append(self._model_to_dict(t))
            except Exception as e:
                logger.warning("Task validation error: %s", e)
                result.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "duration": float(item.get("duration", 1.0)),
                    "duration_unit": item.get("duration_unit", "days"),
                    "status": item.get("status", "not_started"),
                    "priority": item.get("priority", "medium"),
                })
        return result

    def validate_risks(self, risks: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Validate and correct risk data"""
        if not risks:
            return []

        validated: List[Dict[str, Any]] = []
        for i, raw in enumerate(risks):
            item = _ensure_dict(raw)
            item.setdefault("id", f"risk_{i+1}")
            item.setdefault("name", f"Risk {i+1}")

            # clamp before instantiating Risk model -> won't raise
            prob = _clamp_int(item.get("probability", 3), 1, 5, 3)
            imp = _clamp_int(item.get("impact", 3), 1, 5, 3)
            item["probability"] = prob
            item["impact"] = imp
            item["risk_score"] = prob * imp

            try:
                r = Risk(**item)
                validated.append(self._model_to_dict(r))
            except Exception as e:
                logger.warning("Risk validation error: %s", e)
                validated.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "probability": prob,
                    "impact": imp,
                    "risk_score": prob * imp,
                    "category": item.get("category", "technical"),
                    "status": item.get("status", "identified"),
                })
        return validated
