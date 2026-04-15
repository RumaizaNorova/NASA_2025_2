@echo off
cd /d "%~dp0\.."
echo 🦈 Starting Shark Habitat Prediction Dashboard...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 16+ and try again.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found. Creating from template...
    echo MAPBOX_PUBLIC_TOKEN=your_mapbox_public_token_here > .env
    echo OPENAI_API_KEY=your_openai_key_here >> .env
    echo 📝 Please update .env file with your API keys before running again.
    pause
    exit /b 1
)

echo 🚀 Starting FastAPI backend...
cd backend
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt
start "Backend" cmd /k "python run.py"
cd ..

timeout /t 3 /nobreak >nul

echo 🌐 Starting React frontend...
cd frontend
if not exist node_modules (
    echo 📦 Installing npm dependencies...
    npm install
)
start "Frontend" cmd /k "npm start"
cd ..

echo.
echo 🎉 Dashboard is starting up!
echo 📊 Backend: http://localhost:8000
echo 🌐 Frontend: http://localhost:3000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit...
pause >nul

