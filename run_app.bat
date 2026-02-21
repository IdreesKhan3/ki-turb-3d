@echo off
REM ============================================
REM LBM Turbulence Dashboard Launcher (Windows)
REM ============================================
REM This script starts the Streamlit dashboard
REM The dashboard will open automatically in your default web browser
REM Close this window to stop the dashboard

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
set VENV_PYTHON=%~dp0myenv\Scripts\python.exe
set VENV_STREAMLIT=%~dp0myenv\Scripts\streamlit.exe

REM Check if virtual environment exists
if not exist "%~dp0myenv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo.
    echo Please create the virtual environment first:
    echo   python -m venv myenv
    echo   myenv\Scripts\pip.exe install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Check if streamlit executable exists
if not exist "%VENV_STREAMLIT%" (
    echo ERROR: Streamlit executable not found!
    echo.
    echo Please install dependencies first:
    echo   %VENV_PYTHON% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM ============================================
REM API Key Configuration
REM ============================================
REM Set your Gemini API key below to enable Gemini chatbot
REM Get your API key from: https://makersuite.google.com/app/apikey
REM IMPORTANT: Never commit your actual API key! Use .env file instead.
REM set GOOGLE_API_KEY=your-api-key-here

REM ============================================
REM Ollama Model Configuration
REM ============================================
REM Pull any model: qwen2.5-coder:32b, mistral:7b, llama2:7b, etc.
REM To use Qwen:
REM set OLLAMA_MODEL=qwen2.5-coder:32b

@REM set OLLAMA_MODEL=mistral:7b
set OLLAMA_MODEL=qwen2.5-coder:32b

REM Check if Ollama is running (optional - for chatbot feature)
"%VENV_PYTHON%" -c "import requests; requests.get('http://localhost:11434/api/tags', timeout=2)" 2>nul
if errorlevel 1 (
    echo.
    echo NOTE: Ollama is not running. Chatbot feature will use Gemini if available.
    echo To use Ollama locally, install it from: https://ollama.com
    echo Then start it with: ollama serve
    echo.
) else (
    echo Ollama is running - chatbot will use local model: %OLLAMA_MODEL%
    echo.
)

REM Run the dashboard using the virtual environment's streamlit
"%VENV_STREAMLIT%" run app.py

pause

