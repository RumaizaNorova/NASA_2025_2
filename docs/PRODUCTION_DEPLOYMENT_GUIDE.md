# 🐋 Shark Habitat Prediction Dashboard - Production Deployment Guide

## 🚀 Quick Start

Your production setup is now fixed and ready! Here's how to deploy:

### Windows (PowerShell)
```powershell
.\start_production.bat
```

### Linux/Mac (Bash)
```bash
./start_production.sh
```

### Manual Docker Commands
```bash
# Stop any existing containers
docker-compose down

# Build and start services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## 🔧 What Was Fixed

### 1. **Docker Configuration Issues**
- ✅ Fixed backend Dockerfile copy paths
- ✅ Fixed frontend Dockerfile build context
- ✅ Added proper health checks with retries
- ✅ Fixed environment variable handling

### 2. **Backend API Issues**
- ✅ Fixed model loading with fallback paths
- ✅ Fixed data file loading with fallback paths
- ✅ Added proper error handling
- ✅ Fixed OpenAI service integration

### 3. **Frontend Integration Issues**
- ✅ Fixed Mapbox token handling
- ✅ Fixed API service configuration
- ✅ Added proper error handling
- ✅ Fixed Docker environment variables

### 4. **Deployment Scripts**
- ✅ Created Windows batch script (`start_production.bat`)
- ✅ Created Linux/Mac shell script (`start_production.sh`)
- ✅ Added comprehensive error checking
- ✅ Added health monitoring

## 🌐 Access Your Application

Once deployed, your application will be available at:

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔍 Testing Your Setup

Run the test script to verify everything is working:

```bash
python tests/test_setup.py
```

This will check:
- ✅ Required files exist
- ✅ Environment variables are set
- ✅ Backend is responding
- ✅ Frontend is accessible
- ✅ API endpoints are working

## 📋 Required Environment Variables

Make sure your `.env` file contains:

```env
OPENAI_API_KEY=your_openai_api_key_here
MAPBOX_PUBLIC_TOKEN=your_mapbox_token_here
EARTHDATA_TOKEN=your_earthdata_token_here
EARTHDATA_USERNAME=your_earthdata_username
EARTHDATA_PASSWORD=your_earthdata_password
```

## 🐳 Docker Services

Your setup includes:

1. **Backend Service** (Port 8000)
   - FastAPI application
   - ML model serving
   - OpenAI integration
   - Health checks

2. **Frontend Service** (Port 3000)
   - React dashboard
   - Mapbox integration
   - API communication
   - Nginx serving

3. **Nginx Proxy** (Port 80)
   - Reverse proxy
   - Load balancing
   - SSL termination (if configured)

## 🔧 Troubleshooting

### Backend Not Starting
```bash
# Check backend logs
docker-compose logs backend

# Check if model files exist
ls -la results_full/models/
```

### Frontend Not Loading
```bash
# Check frontend logs
docker-compose logs frontend

# Verify Mapbox token
echo $REACT_APP_MAPBOX_TOKEN
```

### API Connection Issues
```bash
# Test backend health
curl http://localhost:8000/health

# Check API endpoints
curl http://localhost:8000/
```

## 📊 Monitoring

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Check Service Status
```bash
docker-compose ps
```

### Resource Usage
```bash
docker stats
```

## 🚀 Production Considerations

### Security
- [ ] Change default API keys
- [ ] Set up SSL certificates
- [ ] Configure firewall rules
- [ ] Enable rate limiting

### Performance
- [ ] Set up Redis for caching
- [ ] Configure database connection pooling
- [ ] Enable gzip compression
- [ ] Set up CDN for static assets

### Monitoring
- [ ] Set up log aggregation
- [ ] Configure health check alerts
- [ ] Monitor resource usage
- [ ] Set up backup procedures

## 🆘 Support

If you encounter issues:

1. **Check the logs**: `docker-compose logs -f`
2. **Run the test script**: `python tests/test_setup.py`
3. **Verify environment variables**: Check your `.env` file
4. **Check file permissions**: Ensure all files are readable
5. **Restart services**: `docker-compose restart`

## 🎉 Success!

Your Shark Habitat Prediction Dashboard is now ready for production! The application provides:

- 🗺️ Interactive map visualization with Mapbox
- 🤖 AI-powered insights with OpenAI
- 📊 Real-time predictions using your trained ML model
- 📈 Performance analytics and monitoring
- 🔍 Natural language queries about shark behavior

Enjoy exploring shark habitat predictions with your production-ready dashboard!


