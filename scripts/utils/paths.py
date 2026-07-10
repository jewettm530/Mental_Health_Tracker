"""Central project paths for Health_Tracker."""

from pathlib import Path

PROJECT_DIR = Path("/Users/maddiemac/My_Projects/Health_Tracker")

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

# Raw exports. Keep these exact names when replacing exports.
STOIC_RAW = RAW_DIR / "stoic.zip"
APPLE_HEALTH_RAW = RAW_DIR / "health.zip"

# Imported/raw-ish CSV folders
STOIC_IMPORTED_DIR = PROCESSED_DIR / "stoic" / "imported"
APPLE_IMPORTED_DIR = PROCESSED_DIR / "apple_health" / "imported"

# Clean separated datasets
STOIC_CLEAN_DIR = PROCESSED_DIR / "stoic" / "clean"
APPLE_CLEAN_DIR = PROCESSED_DIR / "apple_health" / "clean"
MERGED_DIR = PROCESSED_DIR / "merged"

# Future output folders
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
DASHBOARDS_DIR = OUTPUT_DIR / "dashboards"


def ensure_project_folders() -> None:
    """Create the standard project folders if they do not already exist."""
    folders = [
        RAW_DIR,
        STOIC_IMPORTED_DIR,
        APPLE_IMPORTED_DIR,
        STOIC_CLEAN_DIR,
        APPLE_CLEAN_DIR,
        MERGED_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        DASHBOARDS_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
