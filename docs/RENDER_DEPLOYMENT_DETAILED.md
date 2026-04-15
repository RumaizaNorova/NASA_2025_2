# Render deployment (detailed, recommended)

This repo is a **React frontend** + **FastAPI backend**. The backend requires:

- Model artifacts:
  - `results_retrained/models/gradientboosting_model.pkl`
  - `results_retrained/models/feature_names.pkl`
  - `results_retrained/model_performance.json`
- Dataset:
  - `data/integrated_data_full.csv`

These are intentionally **not committed** to Git. Instead, we host them as **GitHub Release assets** and let Render download them at startup.

## 0) One-time prep: create a GitHub Release with assets (you do this)

1. In GitHub, open your repo.
2. Go to **Releases** → **Draft a new release**.
3. Choose a tag name, for example: `artifacts-v1`
4. Upload these files as **release assets** with **exact filenames**:
   - `integrated_data_full.csv` (from your local `data/integrated_data_full.csv`)
   - `gradientboosting_model.pkl` (from `results_retrained/models/gradientboosting_model.pkl`)
   - `feature_names.pkl` (from `results_retrained/models/feature_names.pkl`)
   - `model_performance.json` (from `results_retrained/model_performance.json`)
5. Publish the release.

After publishing, your base download URL will be:

`https://github.com/<owner>/<repo>/releases/download/<tag>`

Example:

`https://github.com/RumaizaNorova/NASA_2025_2/releases/download/artifacts-v1`

That full string is what you’ll set as `ARTIFACT_BASE_URL` on Render.

## 1) Deploy on Render using the Blueprint (you do this)

1. Log into Render.
2. Click **New** → **Blueprint**.
3. Connect your GitHub repo and pick this repository.
4. Render will detect `render.yaml` and propose two services:
   - `shark-habitat-backend` (web service)
   - `shark-habitat-frontend` (static site)
5. Click **Apply** / **Create** (Render wording varies).

## 2) Configure backend env vars (you do this)

Open the backend service → **Environment** (or **Settings → Environment Variables**) and add:

- `ARTIFACT_BASE_URL` = `https://github.com/<owner>/<repo>/releases/download/<tag>`
- `MAPBOX_PUBLIC_TOKEN` = your Mapbox token

Optional:

- `OPENAI_API_KEY` (only needed if you use AI endpoints)
- Earthdata vars (only needed if you download NASA data at runtime):
  - `EARTHDATA_TOKEN`
  - `EARTHDATA_USERNAME`
  - `EARTHDATA_PASSWORD`

Then redeploy (or Render will auto-redeploy after env var changes).

## 3) Configure frontend env vars (you do this)

Open the frontend static site → **Environment Variables** and set:

- `REACT_APP_API_URL` = your backend public URL, for example:
  - `https://shark-habitat-backend.onrender.com`
- `REACT_APP_MAPBOX_TOKEN` = same Mapbox token

Redeploy the frontend.

## 4) Verify

- Backend health: open `https://<backend-host>/health`
- Backend docs: `https://<backend-host>/docs`
- Frontend: open the static site URL and click the map; check that predictions return.

## Troubleshooting

### Backend fails on startup with “file not found”
This means one of the artifact URLs is wrong.

1. Re-check `ARTIFACT_BASE_URL` is correct and does not end with a double slash.
2. Confirm these URLs work in a browser (or `curl`):
   - `${ARTIFACT_BASE_URL}/integrated_data_full.csv`
   - `${ARTIFACT_BASE_URL}/gradientboosting_model.pkl`
   - `${ARTIFACT_BASE_URL}/feature_names.pkl`
   - `${ARTIFACT_BASE_URL}/model_performance.json`

### CORS / frontend can’t call backend
If the frontend is hosted on a Render domain that isn’t covered by CORS in `backend/app.py`, add the frontend origin to `allow_origins` (or use a stricter regex) and redeploy the backend.

