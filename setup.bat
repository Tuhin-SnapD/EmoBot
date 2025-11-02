@echo off
echo ========================================
echo Emotion Detection Chatbot - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/7] Checking Python version...
python --version
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [2/7] Virtual environment already exists. Skipping creation.
) else (
    echo [2/7] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
)
echo.

REM Activate virtual environment
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo.

REM Upgrade pip
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded!
echo.

REM Install requirements
echo [5/7] Installing Python packages from requirements.txt...
echo This may take several minutes...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements!
    pause
    exit /b 1
)
echo.
echo All packages installed successfully!
echo.

REM Download NLTK data and pre-download transformer model
echo [6/7] Downloading NLTK data and pre-loading transformer model...
echo This may take a few minutes (downloading ~500MB model)...
python setup_helper.py
if errorlevel 1 (
    echo [WARNING] Some downloads may have failed. The application will download missing resources on first run.
)
echo.

REM Check/create necessary directories
echo [7/7] Creating necessary directories...
if not exist "models" mkdir models
if not exist "backend\static" mkdir backend\static
echo Directories ready!
echo.

REM Verify data files exist
echo Verifying data files...
if exist "data\emotion_responses.json" (
    echo [OK] emotion_responses.json found
) else (
    echo [WARNING] emotion_responses.json not found. Application may still work with defaults.
)
echo.

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run start_app.bat to start the application
echo 2. Open your browser and go to http://localhost:5000
echo.
pause

