#!/usr/bin/env python3
"""
Build data/sharks_cleaned.csv from data/integrated_data_full.csv (track + metadata columns only).

Run from repository root:
  python3 scripts/export_sharks_cleaned.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTEGRATED = ROOT / "data" / "integrated_data_full.csv"
OUT = ROOT / "data" / "sharks_cleaned.csv"

COLUMNS = [
    "active",
    "datetime",
    "id",
    "latitude",
    "longitude",
    "name",
    "gender",
    "species",
    "weight",
    "length",
    "tagDate",
    "dist_total",
    "foraging_behavior",
]


def main() -> None:
    if not INTEGRATED.is_file():
        raise SystemExit(f"Missing {INTEGRATED} — add integrated data first.")
    df = pd.read_csv(INTEGRATED)
    use = [c for c in COLUMNS if c in df.columns]
    if not {"datetime", "latitude", "longitude", "name"}.issubset(set(use)):
        raise SystemExit(f"integrated file missing required columns; have: {use}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[use].to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT}")


if __name__ == "__main__":
    main()
