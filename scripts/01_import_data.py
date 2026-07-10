"""01_import_data.py

Reads the two raw zip files:
- data/raw/stoic.zip
- data/raw/health.zip

Outputs imported/raw-ish CSVs into:
- data/processed/stoic/imported/
- data/processed/apple_health/imported/
"""

from utils.apple_health import import_apple_health_zip
from utils.paths import APPLE_HEALTH_RAW, APPLE_IMPORTED_DIR, STOIC_IMPORTED_DIR, STOIC_RAW, ensure_project_folders
from utils.stoic import import_stoic_zip


def main() -> None:
    ensure_project_folders()

    print("=== 01 Import Data ===")
    print(f"Stoic zip: {STOIC_RAW}")
    print(f"Apple Health zip: {APPLE_HEALTH_RAW}")

    import_stoic_zip(STOIC_RAW, STOIC_IMPORTED_DIR)
    print("Stoic import complete.")

    import_apple_health_zip(APPLE_HEALTH_RAW, APPLE_IMPORTED_DIR)
    print("Apple Health import complete.")

    print("Finished 01_import_data.py")


if __name__ == "__main__":
    main()
