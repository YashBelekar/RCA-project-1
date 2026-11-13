"""Download the public Consumer Shopping Trends CSV into data/."""
from pathlib import Path
import requests
from src.config import DATA_PATH, DATASET_URL

def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the public Consumer Shopping Trends dataset...")
    try:
        response = requests.get(DATASET_URL, timeout=90)
        response.raise_for_status()
        if len(response.content) < 10_000 or b"," not in response.content[:5000]:
            raise ValueError("The downloaded response does not look like a CSV file.")
        DATA_PATH.write_bytes(response.content)
        print(f"Saved: {DATA_PATH} ({len(response.content):,} bytes)")
    except Exception as exc:
        print("\nDataset download failed. Check your internet connection and the source URL.")
        print(f"Reason: {exc}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
