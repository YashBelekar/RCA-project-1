from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_PATH = DATA_DIR / "Consumer_Shopping_Trends_2026.csv"

DATASET_URL = "https://raw.githubusercontent.com/SahilYadav-op/Consumer-shopping-trends-analysis/main/Consumer_Shopping_Trends_2026%20%286%29.csv"

# Set this to the exact spending column name after inspecting df.columns if auto-detection fails.
TARGET_COLUMN = None

RANDOM_STATE = 42
TEST_SIZE = 0.20
