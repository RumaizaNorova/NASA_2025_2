@echo off
echo Starting Shark Habitat Prediction Dashboard in Production Mode...
echo.

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found!
    echo Please copy .env.example to .env and configure your API keys.
    pause
    exit /b 1
)

REM Build and start with Docker Compose
echo Building and starting services with Docker Compose...
docker-compose up --build -d

echo.
echo Services are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo To view logs: docker-compose logs -f
echo To stop services: docker-compose down
echo.
pause

