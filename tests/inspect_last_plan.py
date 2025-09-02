#!/usr/bin/env python3
import sys
import json
from pathlib import Path

def load_last_plan(path: Path):
    if not path.exists():
        print(f"❌ Fichier introuvable : {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_tree(data, indent=0):
    """Affiche récursivement les clés/valeurs avec indentation"""
    space = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"{space}- {k}:")
            print_tree(v, indent + 1)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            print(f"{space}- [{i}]")
            print_tree(v, indent + 1)
    else:
        print(f"{space}{data}")

if __name__ == "__main__":
    # chemin passé en argument OU fallback automatique
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data/last_plan.json")
    plan = load_last_plan(path)
    if plan:
        print("✅ Plan chargé avec succès\n")
        print_tree(plan)
