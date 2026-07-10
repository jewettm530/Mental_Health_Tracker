"""02_process_data.py

Turns imported CSVs into separated clean daily datasets.

Outputs include:
- data/processed/stoic/clean/stoic_daily_mood.csv
- data/processed/stoic/clean/stoic_answers_long.csv
- data/processed/stoic/clean/stoic_daily_wide.csv
- data/processed/apple_health/clean/apple_daily_sleep.csv
- data/processed/apple_health/clean/apple_daily_activity.csv
- data/processed/apple_health/clean/apple_daily_heart.csv
- data/processed/apple_health/clean/apple_daily_body.csv
- data/processed/apple_health/clean/apple_daily_respiratory.csv
- data/processed/apple_health/clean/apple_daily_workouts.csv
- data/processed/apple_health/clean/apple_activity_summary.csv
"""

from utils.apple_health import process_apple_health
from utils.paths import APPLE_CLEAN_DIR, APPLE_IMPORTED_DIR, STOIC_CLEAN_DIR, STOIC_IMPORTED_DIR, ensure_project_folders
from utils.stoic import process_stoic


def main() -> None:
    ensure_project_folders()

    print("=== 02 Process Data ===")

    process_stoic(STOIC_IMPORTED_DIR, STOIC_CLEAN_DIR)
    print("Stoic processing complete.")

    process_apple_health(APPLE_IMPORTED_DIR, APPLE_CLEAN_DIR)
    print("Apple Health processing complete.")

    print("Finished 02_process_data.py")


if __name__ == "__main__":
    main()
