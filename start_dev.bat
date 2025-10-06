@echo off
echo Starting Shark Habitat Prediction Dashboard in Development Mode...
echo.

REM Check if .env file exists
if not exist .env (
    echo Warning: .env file not found!
    echo Please copy .env.example to .env and configure your API keys.
    echo.
)

REM Start backend
echo Starting backend server...
start "Backend Server" cmd /k "cd backend && python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend
echo Starting frontend server...
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo Both servers are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul

