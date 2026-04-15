# 🦈 Shark Habitat Prediction Dashboard

An interactive web application for predicting shark foraging behavior using NASA satellite data and machine learning models.

## 🌐 Live Demo

- **Dashboard**: https://shark-habitat-frontend.onrender.com/

## ⚠️ Current Status (Known Limitations)

This project is an active work-in-progress. The current model can produce **poor / unrealistic predictions** (e.g., predicting high shark presence on land). We expect to correct this with improved feature engineering, stricter geospatial constraints, and better validation.

Planned improvements include experimenting with **alternative modeling approaches**, potentially including **physics-based methods** and more **specialized domain models**, in addition to further ML iteration.

## 🌟 Features

- **Interactive Map**: Real-time habitat prediction with Mapbox integration
- **AI-Powered Insights**: OpenAI integration for natural language queries and analysis
- **Analytics Dashboard**: Comprehensive data visualization and model performance metrics
- **Temporal Analysis**: Time-based patterns in shark behavior
- **Feature Importance**: Understanding which environmental factors matter most
- **Species Distribution**: Analysis across different shark species

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Node.js 16+ (for local development)
- Python 3.9+ (for local development)

### Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Fill in your API keys in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
MAPBOX_PUBLIC_TOKEN=your_mapbox_token_here
EARTHDATA_TOKEN=your_earthdata_token_here
EARTHDATA_USERNAME=your_email@example.com
EARTHDATA_PASSWORD=your_password_here
```

3. **Data files** for Docker / the API: put `integrated_data_full.csv` and `sharks_cleaned.csv` in the **`data/`** folder (see `data/README.md`). The backend still sees them as `/app/integrated_data_full.csv` inside the container.

### Running with Docker (Recommended)

```bash
# Build and start all services (from repository root; needs Docker Compose v2.20+ for `include`)
docker compose up --build

# Older Compose: use the file under deploy/ directly
docker compose -f deploy/docker-compose.yml up --build

# Access the application
open http://localhost:80
```

### Local Development

#### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend
```bash
cd frontend
npm install
npm start
```

#### Dev helper (backend + frontend)
From the repository root:
```bash
./scripts/start.sh        # macOS / Linux
scripts\start.bat         # Windows
```

#### ML / data pipeline (optional)
Also from the repository root, after `pip install -r requirements.txt`:
```bash
python -m ml.main                      # integration + modeling demo
python -m ml.run_data_preprocessing    # full preprocessing orchestration
python -m ml.scale_full_dataset        # scale to full dataset (example)
```
Other runnable modules live under `ml/` and support the same `python -m ml.<module>` pattern.

## 📊 Model Performance

- **Best Model**: GradientBoosting Classifier
- **AUC Score**: 0.983 (Excellent)
- **Accuracy**: 94.8%
- **Training Samples**: 64,942 shark tracking records
- **Features**: 28 environmental and temporal features

## 🛠️ Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with Pydantic models
- **ML Models**: Scikit-learn (GradientBoosting, RandomForest, LogisticRegression)
- **Data Processing**: Pandas, NumPy
- **AI Integration**: OpenAI GPT-4 for insights and Q&A

### Frontend (React)
- **Framework**: React 18 with Hooks
- **Styling**: Tailwind CSS with custom ocean theme
- **Maps**: Mapbox GL JS with React Map GL
- **Charts**: Plotly.js with React Plotly
- **Animations**: Framer Motion
- **State Management**: React Context API

### Data Sources
- **Shark Tracking**: Tagged shark movement data
- **Satellite Data**: NASA MODIS-Aqua (SST, Chlorophyll-a)
- **Environmental**: Sea surface height, primary productivity

## 🐳 Docker Services

- **Backend**: FastAPI application on port 8000
- **Frontend**: React app served by Nginx on port 80
- **Nginx**: Reverse proxy for production deployment

## 📁 Project Structure

```
├── backend/                 # FastAPI API (Docker / production)
├── frontend/                # React app (Mapbox, dashboard, charts)
├── ml/                      # Offline NASA integration, preprocessing, training scripts
├── tests/                   # Smoke / setup checks (e.g. test_setup.py)
├── scripts/                 # start.sh, start.bat, production helpers
├── deploy/                  # Docker Compose stacks + prod nginx (canonical definitions)
├── docs/                    # Extra deployment and data notes (WEBAPP_README, Railway, etc.)
├── data/                    # Large CSVs (integrated + shark tracks); see data/README.md
├── outputs/                 # Plots (outputs/figures) and submission PDFs (outputs/submission)
├── docker-compose.yml       # Thin wrapper: includes deploy/docker-compose.yml (Compose 2.20+)
├── docker-compose.prod.yml  # Thin wrapper: includes deploy/docker-compose.prod.yml
├── requirements.txt         # Python deps for the ml/ pipeline
└── results_retrained/       # Trained models + metrics (used by deployed API)
```

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Habitat prediction
- `GET /shark-tracks` - Shark tracking data
- `GET /model-performance` - Model metrics
- `GET /species` - Available species
- `GET /stats` - Dataset statistics

### AI Endpoints
- `POST /generate-insights` - AI-powered insights
- `POST /ask-question` - Natural language Q&A
- `POST /generate-report` - Comprehensive analysis

## 🎯 Usage

1. **Map Interaction**: Click anywhere on the map to get habitat predictions
2. **Species Filtering**: Use the controls panel to filter by shark species
3. **Analytics**: Switch to the dashboard for detailed analysis
4. **AI Assistant**: Ask questions about the data and predictions
5. **Temporal Analysis**: Explore time-based patterns in shark behavior

## 🔒 Security

- Environment variables for sensitive API keys
- CORS configuration for cross-origin requests
- Input validation with Pydantic models
- Non-root user in Docker containers

## 📈 Performance

- **Backend**: FastAPI with async/await for high concurrency
- **Frontend**: React with optimized re-renders
- **Caching**: Model and data caching for faster predictions
- **Lazy Loading**: Components loaded on demand

## 🚀 Deployment

### Production Deployment
```bash
# Build for production (root file includes deploy/docker-compose.prod.yml)
docker compose -f docker-compose.prod.yml up --build

# Or call the canonical file
docker compose -f deploy/docker-compose.prod.yml up --build

# Scale services
docker compose up --scale backend=3
```

### Environment Variables for Production
- Set `REACT_APP_API_URL` to your production API URL
- Configure proper CORS origins
- Use production-grade API keys
- Set up SSL/TLS certificates

### Publishing online (public URL)

This project is **React + FastAPI**. It is **not** a Streamlit app; Streamlit would mean rebuilding the whole UI.

| Piece | Good options | Notes |
|--------|----------------|--------|
| **API + model** | [Render](https://render.com), [Railway](https://railway.app), [Fly.io](https://fly.io) | Long‑running Python process, load `results_retrained/` and `data/integrated_data_full.csv` (or attach storage). Repo hints: `render.yaml`, `railway.json`, `Procfile`, `nixpacks.toml`. |
| **Static React app** | [Netlify](https://www.netlify.com), [Vercel](https://vercel.com) | Build `frontend/` → publish `frontend/build`. Set `REACT_APP_API_URL` to your **public API base URL** and `REACT_APP_MAPBOX_TOKEN`. Repo hint: `netlify.toml`. |

**Vercel alone** is a weak fit for this **backend** (big model + pandas in one process). Typical pattern: **Vercel or Netlify for the site**, **Render/Railway/Fly for the API**.

**Checklist**

1. Deploy **backend**: e.g. `cd backend && pip install -r requirements.txt` then `uvicorn app:app --host 0.0.0.0 --port $PORT`.
2. Provide **env vars** on the host: `OPENAI_API_KEY`, `MAPBOX_PUBLIC_TOKEN`, Earthdata vars if used, and **CORS** allowing your frontend origin (see `backend/app.py`).
3. Ship **model + metrics** (`results_full/` or `results_retrained/`, matching what `app.py` expects) and **`data/integrated_data_full.csv`** on the server or volume.
4. Deploy **frontend** with `npm install && npm run build` (you need a valid **`frontend/package.json`** in the repo for CI).
5. Point **`REACT_APP_API_URL`** at the deployed API (HTTPS). Mapbox token in build env for the map tiles.

See also `docs/DEPLOYMENT.md` and `docs/RAILWAY_DEPLOYMENT_GUIDE.md` for host-specific steps.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA for satellite data access
- Shark researchers for tracking data
- OpenAI for AI capabilities
- Mapbox for mapping services

## 📞 Support

For questions or issues, please open a GitHub issue or contact the development team.

---

**🦈 Built with ❤️ for marine conservation and shark research**