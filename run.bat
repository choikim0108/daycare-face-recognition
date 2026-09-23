@echo off
echo ==========================================
echo Starting Daycare Face Recognition Setup...
echo ==========================================

if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Is Python installed?
        pause
        exit /b
    )
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing required packages...
pip install --default-timeout=1000 -r requirements.txt

echo Launching application...
python app.py

pause
