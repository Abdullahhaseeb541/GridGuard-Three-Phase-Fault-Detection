
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

def load_raw():
    possible_names = [
        "detect dataset.csv",
        "detect_dataset.csv",
    ]

    for filename in possible_names:
        path = DATA_DIR / filename
        if path.exists():
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"CSV file not found in: {DATA_DIR}"
    )

if __name__ == "__main__":
    df = load_raw()

    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string())