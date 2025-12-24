@echo off
REM ============================================
REM KI-TURB 3D Launcher (Windows)
REM ============================================
REM This script starts the Streamlit dashboard.
REM The dashboard will open automatically in your default web browser.
REM Close this window to stop the dashboard.

echo.
echo ============================================
echo   KI-TURB 3D
echo   Turbulence Visualization and Analysis Suite
echo ============================================
echo.
echo Starting dashboard...
echo The dashboard will open in your browser at: http://localhost:8501
echo.
echo To stop the dashboard, close this window or press Ctrl+C
echo.

REM Change to script directory
cd /d "%~dp0"

REM Set paths to virtual environment
set "VENV_PYTHON=%~dp0myenv\Scripts\python.exe"
set "VENV_STREAMLIT=%~dp0myenv\Scripts\streamlit.exe"

REM Check if virtual environment exists
if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please create the virtual environment first:
    echo   python -m venv myenv
    echo   myenv\Scripts\pip.exe install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Check if Streamlit executable exists
if not exist "%VENV_STREAMLIT%" (
    echo ERROR: Streamlit executable not found!
    echo.
    echo Please install dependencies first:
    echo   %VENV_PYTHON% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Run the dashboard using the virtual environment's Streamlit
"%VENV_STREAMLIT%" run app.py

pause
