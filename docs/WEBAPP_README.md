# Shark Habitat Prediction Dashboard

A modern, interactive web application for predicting shark foraging behavior using NASA satellite data and machine learning models.

## Features

- **Interactive Map**: Click anywhere on the map to get real-time habitat predictions
- **Shark Tracking**: Visualize shark movement patterns and foraging behavior
- **Analytics Dashboard**: Comprehensive analysis of model performance and data insights
- **AI Assistant**: OpenAI-powered chat interface for natural language queries
- **Real-time Predictions**: Get instant foraging probability predictions for any location
- **Species Filtering**: Filter shark tracks by species for focused analysis

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Scikit-learn**: Machine learning models
- **Pandas**: Data processing
- **OpenAI API**: AI-powered insights and natural language processing

### Frontend
- **React 18**: Modern React with hooks
- **Mapbox GL JS**: Interactive mapping
- **Framer Motion**: Smooth animations
- **Tailwind CSS**: Utility-first styling
- **Axios**: HTTP client

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.9+
- Docker (optional, for production)

### 1. Environment Setup

Copy the environment template and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
MAPBOX_PUBLIC_TOKEN=your_mapbox_token_here

# Optional (for real-time satellite data)
EARTHDATA_TOKEN=your_earthdata_token_here
EARTHDATA_USERNAME=your_earthdata_username
EARTHDATA_PASSWORD=your_earthdata_password
```

### 2. Development Mode

#### Option A: Using the startup script (Windows)
```bash
start_dev.bat
```

#### Option B: Manual setup
```bash
# Terminal 1: Start backend
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend
npm install
npm start
```

### 3. Production Mode

#### Using Docker Compose
```bash
start_prod.bat
```

#### Manual Docker setup
```bash
docker-compose up --build -d
```

## API Endpoints

### Core Endpoints
- `GET /` - API information and status
- `GET /health` - Health check
- `POST /predict` - Get habitat prediction for a location
- `GET /shark-tracks` - Get shark tracking data
- `GET /model-performance` - Get model performance metrics
- `GET /species` - Get list of shark species
- `GET /stats` - Get dataset statistics

### AI-Powered Endpoints
- `POST /generate-insights` - Generate AI insights about predictions
- `POST /ask-question` - Ask natural language questions
- `POST /generate-report` - Generate comprehensive analysis reports

## Usage

### Interactive Map
1. Click anywhere on the map to get a habitat prediction
2. Use the controls panel to filter by species
3. Toggle prediction overlays and animations
4. Click on shark tracks for detailed information

### Analytics Dashboard
1. View comprehensive statistics and model performance
2. Analyze species distribution and foraging patterns
3. Explore temporal trends in shark behavior

### AI Assistant
1. Click the "AI Assistant" button in the navigation
2. Ask questions about shark behavior, model performance, or data insights
3. Generate reports and get scientific interpretations

## Model Information

- **Model Type**: GradientBoosting Classifier
- **AUC Score**: 0.972 (Excellent)
- **Training Samples**: 64,942 records
- **Features**: 27 environmental and temporal features
- **Data Source**: NASA MODIS-Aqua satellite data

## Data Sources

- **Shark Tracking**: Satellite-tagged shark movement data
- **Environmental**: Sea surface temperature, chlorophyll-a, primary productivity
- **Temporal**: Hour, month, day of year, seasonal patterns
- **Spatial**: Latitude, longitude, distance to coast

## Troubleshooting

### Common Issues

1. **Mapbox Token Missing**
   - Ensure `REACT_APP_MAPBOX_TOKEN` is set in your environment
   - Get a free token from [Mapbox](https://www.mapbox.com/)

2. **Backend Connection Failed**
   - Check if backend is running on port 8000
   - Verify CORS settings in backend configuration

3. **Model Loading Errors**
   - Backend will use mock data if model files are missing
   - Ensure model files are in the correct directory structure

4. **OpenAI API Errors**
   - Verify your OpenAI API key is valid
   - Check API usage limits and billing

### Development Tips

- Use browser developer tools to debug API calls
- Check backend logs for detailed error messages
- Use the health check endpoint to verify service status
- Monitor network requests in the browser console

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the NASA "Sharks from Space" challenge solution.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check backend logs for detailed error messages
4. Ensure all environment variables are properly configured

