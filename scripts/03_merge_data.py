"""03_merge_data.py

Creates merged daily setup files for later analysis and visualization.

Outputs:
- data/processed/merged/master_daily.csv
- data/processed/merged/correlation_ready_daily.csv
- data/processed/merged/merge_inventory.json
"""

from utils.merge import build_master_daily_dataset
from utils.paths import MERGED_DIR, ensure_project_folders


def main() -> None:
    ensure_project_folders()

    print("=== 03 Merge Data ===")
    build_master_daily_dataset(MERGED_DIR)
    print("Finished 03_merge_data.py")


if __name__ == "__main__":
    main()
