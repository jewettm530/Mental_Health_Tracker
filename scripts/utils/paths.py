"""Central project paths for Health_Tracker."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
CONFIG_DIR = PROJECT_DIR / "config"

# Raw exports. Keep these exact names when replacing exports.
STOIC_RAW = RAW_DIR / "stoic.zip"
STOIC_TXT_RAW = RAW_DIR / "stoic.txt"
APPLE_HEALTH_RAW = RAW_DIR / "health.zip"
SCREEN_TIME_RAW = RAW_DIR / "screen_time_daily.csv"
CONTEXT_CONFIG = CONFIG_DIR / "context.json"

# Imported/raw-ish CSV folders
STOIC_IMPORTED_DIR = PROCESSED_DIR / "stoic" / "imported"
APPLE_IMPORTED_DIR = PROCESSED_DIR / "apple_health" / "imported"

# Clean separated datasets
STOIC_CLEAN_DIR = PROCESSED_DIR / "stoic" / "clean"
APPLE_CLEAN_DIR = PROCESSED_DIR / "apple_health" / "clean"
CONTEXT_CLEAN_DIR = PROCESSED_DIR / "context" / "clean"
MERGED_DIR = PROCESSED_DIR / "merged"

# Future output folders
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
DASHBOARDS_DIR = OUTPUT_DIR / "dashboards"


def ensure_project_folders() -> None:
    """Create the standard project folders if they do not already exist."""
    folders = [
        RAW_DIR,
        CONFIG_DIR,
        STOIC_IMPORTED_DIR,
        APPLE_IMPORTED_DIR,
        STOIC_CLEAN_DIR,
        APPLE_CLEAN_DIR,
        CONTEXT_CLEAN_DIR,
        MERGED_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        DASHBOARDS_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
