"""
Genzon — Dataset Download Script
Downloads all datasets needed for the fake review detector.

Datasets:
  1. Fake Reviews Dataset (Kaggle/mexwell) — Primary labeled corpus (~40K reviews)
     → Requires manual download (no API key needed via browser)
  2. Deceptive Opinion Spam Corpus v1.4 (Ott et al.) — Cross-domain benchmark (~1,600 reviews)
     → Direct download from myleott.com
  3. McAuley Amazon Review Data (UCSD) — Unlabeled Amazon reviews for domain adaptation
     → Direct download from McAuley Labs

Usage:
    python data/download.py              # download all available datasets
    python data/download.py --only ott   # download only Ott corpus
    python data/download.py --only amazon # download only McAuley Amazon data
"""

import argparse
import io
import zipfile
from pathlib import Path

import requests
import pandas as pd

from data.config import RAW_DIR, PROCESSED_DIR

# ─── Constants ──────────────────────────────────────────────

OTT_CORPUS_URL = "https://myleott.com/op_spam_v1.4.zip"
OTT_CORPUS_ZIP = RAW_DIR / "op_spam_v1.4.zip"
OTT_CORPUS_DIR = RAW_DIR / "op_spam_v1.4"

# McAuley Amazon Reviews — using the smaller "Electronics" category for domain adaptation
# Full catalog: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
MCAULEY_ELECTRONICS_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/"
    "amazon_2023/raw/review_categories/Electronics.jsonl.gz"
)
MCAULEY_FILE = RAW_DIR / "amazon_electronics_reviews.jsonl.gz"

KAGGLE_DATASET_NAME = "mexwell/fake-reviews-dataset"
KAGGLE_EXPECTED_FILE = RAW_DIR / "fake reviews dataset.csv"


# ─── Download helpers ───────────────────────────────────────


def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file with progress indication."""
    if dest.exists():
        print(f"  ✓ {description} already exists at {dest}")
        return True

    print(f"  ↓ Downloading {description}...")
    print(f"    URL: {url}")

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8192

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / (1024 * 1024)
                    print(f"\r    {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

        print(f"\n  ✓ Saved to {dest}")
        return True

    except requests.RequestException as e:
        print(f"\n  ✗ Failed to download: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ─── Dataset 1: Kaggle Fake Reviews ────────────────────────


def download_kaggle_dataset() -> bool:
    """
    Kaggle dataset requires manual download (browser login).
    This function checks if it's present and guides the user.
    """
    print("\n📦 Dataset 1: Fake Reviews Dataset (Kaggle — primary training data)")
    print("   ~40,000 labeled reviews (CG = Computer Generated/Fake, OR = Original/Real)")

    # Check multiple possible filenames (Kaggle CSVs vary)
    possible_files = [
        RAW_DIR / "fake reviews dataset.csv",
        RAW_DIR / "fake_reviews_dataset.csv",
        RAW_DIR / "Fake Reviews Dataset.csv",
    ]

    for f in possible_files:
        if f.exists():
            print(f"  ✓ Found at {f}")
            return True

    print("  ⚠ Not found. Please download manually:")
    print(f"    1. Go to: https://www.kaggle.com/datasets/{KAGGLE_DATASET_NAME}")
    print("    2. Click 'Download' (requires free Kaggle account)")
    print(f"    3. Extract the CSV to: {RAW_DIR}/")
    print(f"    4. Expected file: {KAGGLE_EXPECTED_FILE}")
    print("    5. Re-run this script to verify.")
    return False


# ─── Dataset 2: Ott Deceptive Opinion Spam ─────────────────


def download_ott_corpus() -> bool:
    """Download the Deceptive Opinion Spam Corpus v1.4 from myleott.com."""
    print("\n📦 Dataset 2: Deceptive Opinion Spam Corpus v1.4 (Ott et al.)")
    print("   ~1,600 hotel reviews (800 truthful + 800 deceptive)")

    if OTT_CORPUS_DIR.exists() and any(OTT_CORPUS_DIR.rglob("*.txt")):
        print(f"  ✓ Already extracted at {OTT_CORPUS_DIR}")
        return True

    success = download_file(OTT_CORPUS_URL, OTT_CORPUS_ZIP, "Ott corpus zip")
    if not success:
        return False

    # Extract
    print("  ↓ Extracting zip...")
    try:
        with zipfile.ZipFile(OTT_CORPUS_ZIP, "r") as z:
            z.extractall(RAW_DIR)
        print(f"  ✓ Extracted to {OTT_CORPUS_DIR}")

        # Clean up zip
        OTT_CORPUS_ZIP.unlink()
        return True

    except zipfile.BadZipFile:
        print("  ✗ Corrupt zip file. Deleting and retry later.")
        OTT_CORPUS_ZIP.unlink()
        return False


def parse_ott_corpus() -> pd.DataFrame | None:
    """
    Parse the Ott corpus directory structure into a flat DataFrame.
    Structure: op_spam_v1.4/{negative,positive}_polarity/{deceptive,truthful}_from_*/{fold1..5}/*.txt
    """
    if not OTT_CORPUS_DIR.exists():
        return None

    rows = []
    for txt_file in OTT_CORPUS_DIR.rglob("*.txt"):
        parts = txt_file.relative_to(OTT_CORPUS_DIR).parts

        if len(parts) < 4:
            continue

        polarity = "positive" if "positive" in parts[0] else "negative"
        is_deceptive = "deceptive" in parts[1]
        fold = parts[2]  # fold1, fold2, ..., fold5

        text = txt_file.read_text(encoding="utf-8", errors="replace").strip()

        rows.append({
            "text": text,
            "label": "fake" if is_deceptive else "genuine",
            "polarity": polarity,
            "fold": fold,
            "source": "ott_corpus",
            "file": str(txt_file.name),
        })

    if not rows:
        print("  ⚠ No review text files found in Ott corpus.")
        return None

    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "ott_corpus_parsed.csv"
    df.to_csv(out, index=False)
    print(f"  ✓ Parsed {len(df)} reviews → {out}")
    print(f"    Fake: {(df['label'] == 'fake').sum()}, Genuine: {(df['label'] == 'genuine').sum()}")
    return df


# ─── Dataset 3: McAuley Amazon Reviews ─────────────────────


def download_mcauley_amazon() -> bool:
    """
    Download a subset of McAuley Amazon review data for BERT domain adaptation.
    Using Electronics category (~20M reviews, compressed ~3GB).
    NOTE: This is a large file. We'll download it and then sample during preprocessing.
    """
    print("\n📦 Dataset 3: McAuley Amazon Reviews (Electronics — domain adaptation)")
    print("   Unlabeled Amazon reviews for BERT pre-training / language adaptation")
    print("   ⚠ This is a large download (~3GB compressed). Skip if bandwidth is limited.")

    if MCAULEY_FILE.exists():
        print(f"  ✓ Already downloaded at {MCAULEY_FILE}")
        return True

    success = download_file(MCAULEY_ELECTRONICS_URL, MCAULEY_FILE, "Amazon Electronics reviews")
    return success


# ─── Main ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Download datasets for Genzon")
    parser.add_argument(
        "--only",
        choices=["kaggle", "ott", "amazon"],
        help="Download only a specific dataset",
    )
    parser.add_argument(
        "--skip-amazon",
        action="store_true",
        help="Skip the large McAuley Amazon download",
    )
    args = parser.parse_args()

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Genzon — Dataset Downloader")
    print("=" * 60)

    results = {}

    if args.only is None or args.only == "kaggle":
        results["kaggle"] = download_kaggle_dataset()

    if args.only is None or args.only == "ott":
        results["ott"] = download_ott_corpus()
        if results.get("ott"):
            parse_ott_corpus()

    if (args.only is None and not args.skip_amazon) or args.only == "amazon":
        results["amazon"] = download_mcauley_amazon()

    # Summary
    print("\n" + "=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}")

    if all(results.values()):
        print("\n  All datasets ready! Next step: python data/preprocess.py")
    else:
        print("\n  Some datasets missing — see instructions above.")


if __name__ == "__main__":
    main()