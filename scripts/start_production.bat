@echo off
cd /d "%~dp0\.."
REM Production startup script for Shark Habitat Prediction Dashboard (Windows)

echo 🐋 Starting Shark Habitat Prediction Dashboard in Production Mode...

REM Check if .env file exists
if not exist .env (
    echo ❌ Error: .env file not found!
    echo Please create a .env file with the required environment variables:
    echo OPENAI_API_KEY=your_openai_key
    echo MAPBOX_PUBLIC_TOKEN=your_mapbox_token
    echo EARTHDATA_TOKEN=your_earthdata_token
    echo EARTHDATA_USERNAME=your_earthdata_username
    echo EARTHDATA_PASSWORD=your_earthdata_password
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "results_full\models\gradientboosting_model.pkl" (
    echo ❌ Error: Model file not found!
    echo Please ensure the model files are in the results_full directory.
    pause
    exit /b 1
)

if not exist "data\integrated_data_full.csv" (
    echo ❌ Error: Data file not found!
    echo Please ensure data\integrated_data_full.csv exists (see data\README.md).
    pause
    exit /b 1
)

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Build and start services
echo 🔨 Building and starting services...
docker-compose up --build -d

REM Wait for services to be healthy
echo ⏳ Waiting for services to be healthy...
timeout /t 10 /nobreak > nul

REM Check service health
echo 🏥 Checking service health...
docker-compose ps

REM Show logs
echo 📋 Showing recent logs...
docker-compose logs --tail=50

echo.
echo ✅ Services started successfully!
echo.
echo 🌐 Access your application:
echo    Frontend: http://localhost:3000
echo    Backend API: http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo.
echo 📊 To view logs: docker-compose logs -f
echo 🛑 To stop: docker-compose down
echo.
pause


