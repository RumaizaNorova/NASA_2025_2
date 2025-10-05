#!/bin/bash

# Production startup script for Shark Habitat Prediction Dashboard

echo "🐋 Starting Shark Habitat Prediction Dashboard in Production Mode..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with the required environment variables:"
    echo "OPENAI_API_KEY=your_openai_key"
    echo "MAPBOX_PUBLIC_TOKEN=your_mapbox_token"
    echo "EARTHDATA_TOKEN=your_earthdata_token"
    echo "EARTHDATA_USERNAME=your_earthdata_username"
    echo "EARTHDATA_PASSWORD=your_earthdata_password"
    exit 1
fi

# Check if required files exist
if [ ! -f "results_full/models/gradientboosting_model.pkl" ]; then
    echo "❌ Error: Model file not found!"
    echo "Please ensure the model files are in the results_full directory."
    exit 1
fi

if [ ! -f "integrated_data_full.csv" ]; then
    echo "❌ Error: Data file not found!"
    echo "Please ensure integrated_data_full.csv exists in the root directory."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
docker-compose ps

# Show logs
echo "📋 Showing recent logs..."
docker-compose logs --tail=50

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📊 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
echo ""

