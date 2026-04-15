# Data directory

Place large or sensitive datasets here (this folder is mostly ignored by Git).

Expected files for the full stack:

- `integrated_data_full.csv` — integrated training table (used by the API and Docker mount).
- `sharks_cleaned.csv` — cleaned shark tracks (ML pipeline + Docker). Regenerate from integrated data anytime:

  ```bash
  python3 scripts/export_sharks_cleaned.py
  ```

Copy or generate integrated data from your preprocessing pipeline; paths in code and Compose point here.
