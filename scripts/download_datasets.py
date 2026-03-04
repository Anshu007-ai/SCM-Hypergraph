#!/usr/bin/env python3
"""
HT-HGNN v2.0 -- Dataset Download Helper
=========================================
CLI utility to download / locate all datasets used by the supply-chain
risk-analysis pipeline.

Usage examples:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset dataco
    python scripts/download_datasets.py --dataset maintenance
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so `src.*` imports work from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "Data set"

# ---- Dataset-specific helpers ---------------------------------------------

def _ensure_dir(directory: Path) -> None:
    """Create the target directory if it does not already exist."""
    directory.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] Target directory ready: {directory}")


def download_dataco() -> None:
    """Print instructions for downloading the DataCo SMART Supply Chain dataset."""
    print("\n" + "=" * 70)
    print("DATASET: DataCo SMART Supply Chain")
    print("=" * 70)
    target = DATA_ROOT / "DataCo"
    _ensure_dir(target)
    print("""
  Source  : Mendeley Data
  URL     : https://data.mendeley.com/datasets/8gx2fvg2k6/5
  License : CC BY 4.0

  Steps:
    1. Visit the URL above and click 'Download' (ZIP, ~180 MB).
    2. Extract the ZIP so the following file exists:
         Data set/DataCo/DataCoSupplyChainDataset.csv
    3. (Optional) Also place DescriptionDataCoSupplyChain.csv in the
       same folder for column documentation.
""")


def download_bom() -> None:
    """BOM data ships with the repository -- just confirm it."""
    print("\n" + "=" * 70)
    print("DATASET: Bill of Materials (BOM)")
    print("=" * 70)
    target = DATA_ROOT / "BOM"
    _ensure_dir(target)

    train_ok = (target / "train_set.csv").exists()
    test_ok = (target / "test_set.csv").exists()

    if train_ok and test_ok:
        print("  [OK] BOM data is included in the repository and already present.")
        print(f"       train_set.csv : {target / 'train_set.csv'}")
        print(f"       test_set.csv  : {target / 'test_set.csv'}")
    else:
        print("  [WARN] BOM CSV files not found.  They should be included in the repo.")
        print("         Expected location:")
        print(f"           {target / 'train_set.csv'}")
        print(f"           {target / 'test_set.csv'}")
        print("         Try running `git checkout -- 'Data set/BOM/'` to restore them.")


def download_ports() -> None:
    """Print instructions for downloading IMF PortWatch disruption data."""
    print("\n" + "=" * 70)
    print("DATASET: Port Disruption (IMF PortWatch)")
    print("=" * 70)
    target = DATA_ROOT / "Ports"
    _ensure_dir(target)
    print("""
  Source  : IMF PortWatch
  URL     : https://portwatch.imf.org/
  API docs: https://portwatch.imf.org/portal/home/pages/api

  Steps:
    1. Visit the PortWatch website and register for API access.
    2. Download the port-level disruption CSV through the portal or API.
    3. Place the resulting CSV(s) in:
         Data set/Ports/
    4. Expected columns (may vary by export):
         port_id, port_name, country, latitude, longitude,
         disruption_start, disruption_end, severity, cause

  Note: If API access is unavailable, you can manually compile port
        disruption events from the PortWatch interactive map.
""")


def download_maintenance() -> None:
    """Attempt to download the UCI AI4I 2020 maintenance dataset."""
    print("\n" + "=" * 70)
    print("DATASET: Predictive Maintenance (AI4I 2020)")
    print("=" * 70)
    target = DATA_ROOT / "Maintenance"
    _ensure_dir(target)

    csv_path = target / "ai4i2020.csv"
    if csv_path.exists():
        print(f"  [OK] Dataset already present: {csv_path}")
        return

    # Try downloading from UCI
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00601/ai4i2020.csv"
    )
    print(f"  Attempting download from UCI: {url}")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(csv_path))
        print(f"  [OK] Downloaded to {csv_path}")
    except Exception as exc:
        print(f"  [WARN] Automatic download failed: {exc}")
        print("""
  Manual download instructions:
    URL     : https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
    Steps   :
      1. Visit the URL above and download the CSV.
      2. Place it at:
           Data set/Maintenance/ai4i2020.csv
""")


def download_retail() -> None:
    """Print Kaggle API download instructions for the Online Retail dataset."""
    print("\n" + "=" * 70)
    print("DATASET: Online Retail (Kaggle)")
    print("=" * 70)
    target = DATA_ROOT / "Retail"
    _ensure_dir(target)
    print("""
  Source  : Kaggle -- Online Retail II dataset
  URL     : https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

  Option A -- Kaggle CLI (recommended):
    1. Install the Kaggle CLI:
         pip install kaggle
    2. Place your kaggle.json API token in ~/.kaggle/kaggle.json
    3. Run:
         kaggle datasets download -d mashlyn/online-retail-ii-uci \\
             -p "Data set/Retail/" --unzip

  Option B -- Manual download:
    1. Visit the Kaggle URL above and click 'Download'.
    2. Extract the ZIP into:
         Data set/Retail/
    3. Expected file: online_retail_II.xlsx  or  online_retail_II.csv
""")


# ---- Dispatcher -----------------------------------------------------------

DATASET_HANDLERS = {
    "dataco":      download_dataco,
    "bom":         download_bom,
    "ports":       download_ports,
    "maintenance": download_maintenance,
    "retail":      download_retail,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HT-HGNN v2.0 -- Download / locate datasets",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show download instructions for every dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_HANDLERS.keys()),
        help="Show instructions for a single dataset.",
    )

    args = parser.parse_args()

    if not args.all and args.dataset is None:
        parser.print_help()
        sys.exit(1)

    print("HT-HGNN v2.0 -- Dataset Download Helper")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data root    : {DATA_ROOT}")

    if args.all:
        for handler in DATASET_HANDLERS.values():
            handler()
    else:
        DATASET_HANDLERS[args.dataset]()

    print("\n" + "=" * 70)
    print("Done.  Re-run with --all to see every dataset.")
    print("=" * 70)


if __name__ == "__main__":
    main()
