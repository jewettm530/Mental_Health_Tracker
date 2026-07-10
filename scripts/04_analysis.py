"""04_analysis.py

Runs both the existing useful association analysis and the extended insight analyses
used by the Streamlit Summary, Predictions, Lagged Effects, Consistency,
Personal Baselines, and Things to Watch tabs.
"""
from utils.analysis import run_useful_analysis
from utils.insights import run_extended_analysis
from utils.paths import ensure_project_folders


def main() -> None:
    ensure_project_folders()
    print("=== 04 Useful + Extended Analysis ===")
    run_useful_analysis()
    run_extended_analysis()
    print("Finished 04_analysis.py")


if __name__ == "__main__":
    main()
