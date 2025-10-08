# 🚂 Railway Deployment Guide - Easy Way!

Since Railway CLI requires browser interaction, let's use Railway's web interface instead!

## ⚠️ CRITICAL FIX APPLIED (Latest Commit)

**Problem Fixed:** Model loading was crashing due to path resolution bug in `backend/app.py`
- **Bug:** Line 197 had `if not os.path.exists(perf_path):` which would crash if `perf_path` was `None`
- **Fixed:** Changed to `if perf_path is None:` for proper null checking
- **Added:** Detailed logging to show which paths are being checked and why they fail

**Files That Were Force-Committed (despite .gitignore):**
✅ `results_retrained/models/gradientboosting_model.pkl` (34MB)
✅ `results_retrained/models/feature_names.pkl`
✅ `results_retrained/model_performance.json`
✅ `integrated_data_full.csv` (34MB)

**Latest Deployment Commit:** `c81869e` - "Fix critical path resolution bug in model loading"

## Step 1: Go to Railway Dashboard

1. Open browser and go to: **https://railway.app/dashboard**
2. Login with your GitHub account (rumaizanorova@gmail.com)

## Step 2: Deploy Backend

1. Click **"New Project"**
2. Click **"Deploy from GitHub repo"**
3. Select your `NASA_2025_2` repository
4. Railway will detect it's a Python project

### Backend Configuration:

**Root Directory:** `backend`

**Build Command:** (Railway auto-detects)
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

**Environment Variables:**
Click "Variables" tab and add these (get values from your .env file):
```
OPENAI_API_KEY=<your_openai_key_from_env_file>

MAPBOX_PUBLIC_TOKEN=<your_mapbox_token_from_env_file>
```

**Important:** After deployment, copy your backend URL (looks like: `https://your-app-name.up.railway.app`)

## Step 3: Deploy Frontend

1. Click **"New Project"** again
2. Click **"Deploy from GitHub repo"**
3. Select your `NASA_2025_2` repository again

### Frontend Configuration:

**Root Directory:** `frontend`

**Build Command:**
```bash
npm install && npm run build
```

**Start Command:**
```bash
npx serve -s build -l $PORT
```

**Environment Variables:**
Click "Variables" tab and add:
```
REACT_APP_API_URL=<PASTE YOUR BACKEND URL HERE>

REACT_APP_MAPBOX_TOKEN=<your_mapbox_token_from_env_file>
```

**Replace `<PASTE YOUR BACKEND URL HERE>`** with the backend URL from Step 2!

## Step 4: Test Your Deployment

Once both are deployed:
1. Click on your frontend service
2. Click "Settings" → "Networking" → "Generate Domain"
3. Visit your domain - your app is LIVE! 🎉

## Troubleshooting

### Backend Issues:
- Check logs: Click backend service → "Deployments" → View logs
- Verify model files are included (they should auto-deploy from repo)

**What to Look for in Logs:**
1. ✅ `"Current working directory: /app/backend"`
2. ✅ `"Checking for model at: ../results_retrained/models/gradientboosting_model.pkl - exists: True"`
3. ✅ `"Successfully loaded retrained GradientBoosting model"`
4. ✅ `"Successfully loaded retrained model performance data"`
5. ✅ `"Successfully loaded X shark track records"`
6. ✅ `"Startup completed successfully - Model and data loaded!"`

**If Model Loading Fails:**
- The logs will show which paths were checked and which one succeeded/failed
- Working directory should be `/app/backend`
- Model files should be at `/app/results_retrained/models/` (accessible via `../results_retrained/models/`)

### Frontend Issues:
- Make sure `REACT_APP_API_URL` points to your BACKEND URL
- Check frontend logs for any build errors

### CORS Issues:
If frontend can't connect to backend:
1. Go to backend service
2. The CORS settings in `backend/app.py` already allow Railway domains
3. Just redeploy backend if needed

### Model Path Issues (FIXED):
The critical bug in model loading has been fixed in commit `c81869e`. If you still see path errors:
1. Check that Railway is using the latest commit
2. Verify the logs show the path checking messages
3. Check that model files are in the repository (they were force-committed)

## Your Services Will Be:

- **Frontend**: `https://your-frontend-name.up.railway.app` (Your main app)
- **Backend API**: `https://your-backend-name.up.railway.app` (Your API)

## Free Tier Limits

Railway free tier includes:
- 500 hours/month execution time
- $5 free credit/month
- Enough for your shark prediction dashboard!

## Done! 🚀

Your shark habitat prediction dashboard is now live and accessible worldwide!

