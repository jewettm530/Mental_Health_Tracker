"""04_analysis.py

Useful statistical analysis for the Health Tracker project.

This script creates interpretable statistics rather than one generic correlation dump:
- days with vs without specific factors
- mood-focused correlations
- filtered correlations with unhelpful derived/count relationships removed
- top positive and negative associations with lowest mood

Run from the project root:
    python3 scripts/04_analysis.py
"""

from utils.analysis import run_useful_analysis
from utils.paths import ensure_project_folders


def main() -> None:
    ensure_project_folders()
    print("=== 04 Useful Analysis ===")
    run_useful_analysis()
    print("Finished 04_analysis.py")


if __name__ == "__main__":
    main()
