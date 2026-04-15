"""Repository root for resolving data files and .env when cwd varies."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
SHARKS_CSV = DATA_DIR / "sharks_cleaned.csv"
INTEGRATED_DATA_FULL_CSV = DATA_DIR / "integrated_data_full.csv"
