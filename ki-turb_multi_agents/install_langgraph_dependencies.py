"""Install the required KI-TURB LangChain/LangGraph runtime."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

requirements = Path(__file__).with_name("requirements-langgraph.txt")
raise SystemExit(subprocess.call([sys.executable, "-m", "pip", "install", "-r", str(requirements)]))
