#!/bin/bash
# ============================================
# KI-TURB 3D Launcher (Linux/macOS)
# ============================================
# This script starts the Streamlit dashboard.
# The dashboard will open in your default web browser.
# Press Ctrl+C to stop the dashboard.

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
  # shellcheck disable=SC1091
  source myenv/bin/activate
  echo "Activated virtual environment: myenv"
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
  echo "ERROR: Streamlit is not installed!"
  echo ""
  echo "Please install dependencies first:"
  echo "  pip install -r requirements.txt"
  echo ""
  exit 1
fi

# Run the dashboard
exec streamlit run app.py
