"""
Entry point for running PlannerIA as a module
Usage: python -m src.project_planner "Your project brief here"
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from crew import main

if __name__ == "__main__":
    main()
