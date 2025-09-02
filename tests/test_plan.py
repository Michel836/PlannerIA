#!/usr/bin/env python3
"""
Test rapide de PlannerIA.generate_plan()
"""

import json
from pathlib import Path
import importlib.util
import sys

# Charger crew.py comme module
crew_path = Path(__file__).resolve().parent.parent / "crew.py"
spec = importlib.util.spec_from_file_location("crew", crew_path)
crew = importlib.util.module_from_spec(spec)
sys.modules["crew"] = crew
spec.loader.exec_module(crew)

PlannerIA = crew.PlannerIA


def test_generate_plan():
    print("=== TEST generate_plan() ===")

    planner = PlannerIA()
    brief = "Créer un site web simple avec une page d’accueil et un formulaire de contact."
    plan = planner.generate_plan(brief)

    assert isinstance(plan, dict), "Le plan doit être un dictionnaire"
    assert "project_overview" in plan, "Le plan doit contenir un project_overview"

    out_path = Path("tests/output_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False, default=str)

    print("✅ Test terminé avec succès")
    print(f"📄 Résultat sauvegardé dans {out_path}")


if __name__ == "__main__":
    test_generate_plan()
