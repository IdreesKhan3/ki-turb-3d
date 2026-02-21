#!/bin/bash
# ============================================
# LBM Turbulence Dashboard Launcher (Linux/Mac)
# ============================================
# This script starts the Streamlit dashboard
# The dashboard will open automatically in your default web browser
# Press Ctrl+C to stop the dashboard

# AI Assistant (optional):
# - Cloud: set GOOGLE_API_KEY to enable Gemini
# - Local: set OLLAMA_MODEL (e.g., mistral:7b or qwen2.5-coder:32b) and run Ollama on localhost:11434
# Core turbulence analysis works without AI.

# Get script directory and ensure we're in the right place
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo "  KI-TURB 3D"
echo "  Turbulence Visualization & Analysis Suite"
echo "============================================"
echo ""
echo "Starting dashboard..."
echo "The dashboard will open in your browser at: http://localhost:8501"
echo ""
echo "To stop the dashboard, press Ctrl+C"
echo ""

# Activate virtual environment if it exists
if [ -d "myenv" ]; then
    source myenv/bin/activate
    echo "Activated virtual environment: myenv"
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "ERROR: Streamlit is not installed!"
    echo ""
    echo "Please install dependencies first:"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

# ============================================
# API Key Configuration
# ============================================
# Set your Gemini API key below to enable Gemini chatbot
# Get your API key from: https://makersuite.google.com/app/apikey

# export GOOGLE_API_KEY="your-api-key-here" #

# ============================================
# Ollama Model Configuration
# ============================================
# Pull any model: qwen2.5-coder:32b, mistral:7b, llama2:7b, etc.
# To use Qwen:
# set OLLAMA_MODEL=qwen2.5-coder:32b

# export OLLAMA_MODEL="mistral:7b"
export OLLAMA_MODEL="qwen2.5-coder:32b"

# GPU for Ollama. Set OLLAMA_GPU=0,1,2,3 etc. to use a specific GPU. Leave unset for default.
if [ -n "${OLLAMA_GPU}" ]; then
    export CUDA_VISIBLE_DEVICES="$OLLAMA_GPU"
fi

# Auto-start Ollama for AI Assistant (if local installation exists in project)
# Place Ollama binary at: <project_dir>/local_tools/ollama/ollama
# Or install system-wide from https://ollama.com and run: ollama serve
OLLAMA_DIR="$SCRIPT_DIR/local_tools/ollama"
if [ -f "$OLLAMA_DIR/ollama" ]; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        if [ "${OLLAMA_FORCE_RESTART:-0}" = "1" ]; then
            echo "Restarting Ollama..."
            pkill -f "ollama serve" 2>/dev/null || true
            sleep 2
        else
            echo "Ollama already running. To restart: OLLAMA_FORCE_RESTART=1 ./run_app.sh"
        fi
    fi
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Starting Ollama server..."
        export OLLAMA_MODELS="${OLLAMA_MODELS:-$OLLAMA_DIR/models}"
        (cd "$OLLAMA_DIR" && OLLAMA_MODELS="$OLLAMA_DIR/models" ./ollama serve > /dev/null 2>&1 &)
        sleep 3
    fi
fi

# Run the dashboard (we're already in SCRIPT_DIR)
streamlit run app.py


# # Find process using port 8501
# lsof -ti:8501
# # Or see full details
# ps aux | grep streamlit | grep -v grep
# # Or check what's using the port
# lsof -i:8501

