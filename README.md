# 🦈 Shark Habitat Prediction Dashboard

An interactive web application for predicting shark foraging behavior using NASA satellite data and machine learning models.

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

### Running with Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

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

## 📊 Model Performance

- **Best Model**: GradientBoosting Classifier
- **AUC Score**: 0.972 (Excellent)
- **Accuracy**: 92.9%
- **Training Samples**: 64,942 shark tracking records
- **Features**: 27 environmental and temporal features

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
├── backend/                 # FastAPI backend
│   ├── app.py              # Main application
│   ├── openai_service.py   # AI integration
│   ├── Dockerfile          # Backend container
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── context/        # State management
│   │   ├── services/       # API services
│   │   └── App.js          # Main app component
│   ├── Dockerfile          # Frontend container
│   └── package.json        # Node dependencies
├── results_full/           # Trained models and results
├── docker-compose.yml      # Multi-container setup
└── nginx.conf              # Reverse proxy config
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
# Build for production
docker-compose -f docker-compose.prod.yml up --build

# Scale services
docker-compose up --scale backend=3
```

### Environment Variables for Production
- Set `REACT_APP_API_URL` to your production API URL
- Configure proper CORS origins
- Use production-grade API keys
- Set up SSL/TLS certificates

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