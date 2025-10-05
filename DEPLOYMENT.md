# 🚀 Deployment Guide

This guide covers deploying the Shark Habitat Prediction Dashboard to various platforms.

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- Mapbox API token
- OpenAI API key
- Trained model files in `results_full/models/`

## 🏠 Local Development

### Quick Start
```bash
# Using the startup script
./start.sh  # Linux/Mac
start.bat   # Windows

# Or manually
cd backend && python run.py
cd frontend && npm start
```

### Environment Setup
1. Copy `.env` file and update with your API keys
2. Install backend dependencies: `cd backend && pip install -r requirements.txt`
3. Install frontend dependencies: `cd frontend && npm install`

## ☁️ Cloud Deployment

### Option 1: Railway (Recommended)

#### Backend Deployment
1. Connect your GitHub repository to Railway
2. Set environment variables:
   - `OPENAI_API_KEY`
   - `MAPBOX_PUBLIC_TOKEN`
3. Deploy from `backend/` directory
4. Railway will automatically detect FastAPI and install dependencies

#### Frontend Deployment
1. Create a new Railway service for the frontend
2. Set build command: `npm run build`
3. Set start command: `npx serve -s build`
4. Set environment variables:
   - `REACT_APP_API_URL=https://your-backend-url.railway.app`

### Option 2: Render

#### Backend Deployment
1. Connect GitHub repository to Render
2. Create a new Web Service
3. Set build command: `cd backend && pip install -r requirements.txt`
4. Set start command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables

#### Frontend Deployment
1. Create a new Static Site
2. Set build command: `cd frontend && npm install && npm run build`
3. Set publish directory: `frontend/build`
4. Add environment variables

### Option 3: Heroku

#### Backend Deployment
1. Create `Procfile` in backend directory:
   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
2. Create `runtime.txt`:
   ```
   python-3.9.7
   ```
3. Deploy using Heroku CLI or GitHub integration

#### Frontend Deployment
1. Use Heroku Buildpack for static sites
2. Set build command in `package.json`:
   ```json
   {
     "scripts": {
       "heroku-postbuild": "npm install && npm run build"
     }
   }
   ```

## 🐳 Docker Deployment

### Backend Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .
COPY results_full/ ./results_full/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile
```dockerfile
FROM node:16-alpine

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MAPBOX_PUBLIC_TOKEN=${MAPBOX_PUBLIC_TOKEN}
    volumes:
      - ./results_full:/app/results_full

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - backend
```

## 🔧 Environment Variables

### Backend
```env
OPENAI_API_KEY=your_openai_key
MAPBOX_PUBLIC_TOKEN=your_mapbox_token
```

### Frontend
```env
REACT_APP_API_URL=http://localhost:8000
```

## 📊 Performance Optimization

### Backend
- Use Gunicorn for production: `gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker`
- Enable Redis caching for predictions
- Use CDN for static assets

### Frontend
- Enable gzip compression
- Use service workers for offline functionality
- Optimize bundle size with code splitting

## 🔒 Security Considerations

- Use HTTPS in production
- Validate all API inputs
- Rate limit API endpoints
- Secure environment variables
- Enable CORS properly

## 📈 Monitoring

### Health Checks
- Backend: `GET /health`
- Frontend: Static file serving

### Logging
- Backend: Structured logging with Python logging
- Frontend: Error boundary for React errors

### Metrics
- API response times
- Prediction accuracy
- User engagement

## 🚨 Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure `results_full/models/` directory exists
   - Check file permissions

2. **API connection errors**
   - Verify environment variables
   - Check CORS configuration
   - Ensure backend is running

3. **Map not loading**
   - Verify Mapbox token
   - Check network connectivity
   - Ensure token has correct permissions

### Debug Mode
```bash
# Backend
cd backend && python run.py --reload --log-level debug

# Frontend
cd frontend && npm start
```

## 📚 Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [React Deployment](https://create-react-app.dev/docs/deployment/)
- [Mapbox GL JS](https://docs.mapbox.com/mapbox-gl-js/)
- [OpenAI API](https://platform.openai.com/docs)

## 🆘 Support

For deployment issues:
1. Check the logs
2. Verify environment variables
3. Test locally first
4. Open an issue with deployment details

