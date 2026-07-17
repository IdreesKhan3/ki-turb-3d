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
# Gemini: https://makersuite.google.com/app/apikey
# export GOOGLE_API_KEY="your-gemini-api-key"
#

# DeepSeek: https://platform.deepseek.com/api_keys
# export DEEPSEEK_API_KEY="your-deepseek-api-key"
# export DEEPSEEK_MODEL="deepseek-v4-pro"   # or deepseek-v4-flash

# ============================================
# Ollama Model Configuration
# ============================================
# Pull any model: qwen2.5-coder:32b, mistral:7b, llama2:7b, etc.
# To use Qwen:
# set OLLAMA_MODEL=qwen2.5-coder:32b

# export OLLAMA_MODEL="mistral:7b"
export OLLAMA_MODEL="qwen2.5-coder:32b"

# Start system Ollama on the selected GPU (default 3)
OLLAMA_GPU="${OLLAMA_GPU:-3}"
if command -v ollama >/dev/null 2>&1; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        if [ "${OLLAMA_FORCE_RESTART:-0}" = "1" ]; then
            echo "Restarting Ollama on GPU $OLLAMA_GPU..."
            pkill -u "$USER" -f "ollama serve" 2>/dev/null || true
            sleep 2
        else
            echo "Ollama already running. To use GPU $OLLAMA_GPU: OLLAMA_FORCE_RESTART=1 ./run_app.sh"
        fi
    fi
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Starting Ollama server on GPU $OLLAMA_GPU..."
        (CUDA_VISIBLE_DEVICES="$OLLAMA_GPU" ollama serve > /dev/null 2>&1 &)
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

