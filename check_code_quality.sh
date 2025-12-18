#!/usr/bin/env bash
# Code Quality Check Script for AI Assistant Scripts
# Run this before publishing to GitHub
#
# Notes:
# - Uses `python -m ...` for robustness (works inside venvs, avoids PATH issues).
# - Produces clear PASS/FAIL signals and exits non-zero if any check fails.
# - Keeps output readable and capped for terminal friendliness.

set -Eeuo pipefail

echo "=========================================="
echo "Code Quality Checks for AI Assistant Scripts"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Target directory or file (override via env: TARGET_DIR=... ./quality_check.sh)
# Can also pass as first argument: ./check_code_quality.sh path/to/file.py
if [ $# -gt 0 ]; then
  TARGET="${1}"
else
  TARGET="${TARGET_DIR:-utils/ai_assist}"
fi

# Check if it's a file or directory
if [ -f "$TARGET" ]; then
  TARGET_TYPE="file"
  TARGET_DIR="$(dirname "$TARGET")"
  TARGET_FILE="$(basename "$TARGET")"
  echo "Target: Single file - $TARGET"
elif [ -d "$TARGET" ]; then
  TARGET_TYPE="directory"
  TARGET_DIR="$TARGET"
  TARGET_FILE=""
  echo "Target: Directory - $TARGET"
else
  echo -e "${RED}Error:${NC} '$TARGET' is neither a file nor a directory"
  exit 1
fi

# Optional: where to write reports
BANDIT_JSON="${BANDIT_JSON:-/tmp/bandit_report.json}"

# Output limits
HEAD_LINES="${HEAD_LINES:-60}"

# Failure accumulator
FAILED=0

have_python_module() {
  # Checks whether a Python module is importable in the current interpreter
  python - "$1" <<'PY' >/dev/null 2>&1
import importlib, sys
mod = sys.argv[1]
importlib.import_module(mod)
PY
}

run_check() {
  # Usage: run_check "Label" "python_module" -- command args...
  local label="$1"
  local module="$2"
  shift 2

  echo -e "${YELLOW}${label}${NC}"

  if ! have_python_module "$module"; then
    echo -e "  ${YELLOW}SKIP:${NC} Python module '${module}' not installed in this environment."
    echo -e "  Install with: pip install ${module}\n"
    return 0
  fi

  # Run and capture output; cap displayed lines
  set +e
  local out
  out="$("$@" 2>&1)"
  local code=$?
  set -e

  if [ $code -eq 0 ]; then
    echo -e "  ${GREEN}PASS${NC}"
    if [ -n "$out" ]; then
      echo "$out" | head -n "$HEAD_LINES"
    fi
    echo ""
    return 0
  else
    echo -e "  ${RED}FAIL${NC} (exit code: $code)"
    if [ -n "$out" ]; then
      echo "$out" | head -n "$HEAD_LINES"
    fi
    echo ""
    FAILED=1
    return 0
  fi
}

# Check if target exists
if [ "$TARGET_TYPE" = "file" ]; then
  if [ ! -f "$TARGET" ]; then
    echo -e "${RED}Error:${NC} File '$TARGET' not found"
    exit 1
  fi
  echo "Checking file: $TARGET"
  CHECK_TARGET="$TARGET"
else
  if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${RED}Error:${NC} Directory '$TARGET_DIR' not found"
    exit 1
  fi
  echo "Checking directory: $TARGET_DIR"
  CHECK_TARGET="$TARGET_DIR"
fi
echo ""

# 0) Interpreter info (helps when multiple Pythons exist)
echo "Python interpreter: $(command -v python)"
python --version
echo ""

# 1) Dead code and unused imports (vulture)
if [ "$TARGET_TYPE" = "file" ]; then
  run_check "[1/8] Dead code and unused imports (vulture)..." "vulture" \
    python -m vulture "$CHECK_TARGET" --min-confidence 80
else
  run_check "[1/8] Dead code and unused imports (vulture)..." "vulture" \
    python -m vulture "$CHECK_TARGET" --min-confidence 80
fi

# 2) Unused imports and variables (autoflake)
# Note: autoflake may be silent on success; this wrapper prints PASS/FAIL explicitly.
if [ "$TARGET_TYPE" = "file" ]; then
  run_check "[2/8] Unused imports/vars (autoflake --check)..." "autoflake" \
    python -m autoflake --check \
      --remove-all-unused-imports --remove-unused-variables \
      "$CHECK_TARGET"
else
  run_check "[2/8] Unused imports/vars (autoflake --check)..." "autoflake" \
    python -m autoflake --check --recursive \
      --remove-all-unused-imports --remove-unused-variables \
      "$CHECK_TARGET"
fi

# 3) Linting (pylint)
# Keeps output readable; disables style/refactor noise but still catches real issues.
run_check "[3/8] Linting (pylint)..." "pylint" \
  python -m pylint "$CHECK_TARGET" \
    --disable=C,R \
    --max-line-length=120 \
    --output-format=text

# 4) Type checking (mypy)
run_check "[4/8] Type checking (mypy)..." "mypy" \
  python -m mypy "$CHECK_TARGET" \
    --ignore-missing-imports \
    --no-strict-optional

# 5) Security checks (bandit)
# Produces both console output and a JSON report (if desired).
echo -e "${YELLOW}[5/8] Security checks (bandit)...${NC}"
if have_python_module "bandit"; then
  set +e
  if [ "$TARGET_TYPE" = "file" ]; then
    bandit_out="$(python -m bandit "$CHECK_TARGET" -ll 2>&1)"
  else
    bandit_out="$(python -m bandit -r "$CHECK_TARGET" -ll 2>&1)"
  fi
  bandit_code=$?
  set -e

  if [ $bandit_code -eq 0 ]; then
    echo -e "  ${GREEN}PASS${NC}"
  else
    echo -e "  ${RED}FAIL${NC} (exit code: $bandit_code)"
    FAILED=1
  fi

  echo "$bandit_out" | head -n "$HEAD_LINES"
  echo ""

  # Always attempt to write JSON report for later inspection (non-fatal if it fails)
  set +e
  if [ "$TARGET_TYPE" = "file" ]; then
    python -m bandit "$CHECK_TARGET" -f json -o "$BANDIT_JSON" >/dev/null 2>&1
  else
    python -m bandit -r "$CHECK_TARGET" -f json -o "$BANDIT_JSON" >/dev/null 2>&1
  fi
  set -e
  echo "  Bandit JSON report (if generated): $BANDIT_JSON"
  echo ""
else
  echo -e "  ${YELLOW}SKIP:${NC} Python module 'bandit' not installed."
  echo -e "  Install with: pip install bandit\n"
fi

# 6) Code complexity (radon)
echo -e "${YELLOW}[6/8] Code complexity (radon)...${NC}"
if have_python_module "radon"; then
  # Cyclomatic complexity
  set +e
  cc_out="$(python -m radon cc "$CHECK_TARGET" --min B 2>&1)"
  cc_code=$?
  set -e

  if [ $cc_code -eq 0 ]; then
    echo -e "  ${GREEN}PASS${NC} (cyclomatic complexity scan ran)"
  else
    echo -e "  ${RED}FAIL${NC} (exit code: $cc_code)"
    FAILED=1
  fi
  echo "  Cyclomatic Complexity (min grade B):"
  echo "$cc_out" | head -n "$HEAD_LINES"
  echo ""

  # Maintainability index
  set +e
  mi_out="$(python -m radon mi "$CHECK_TARGET" --min B 2>&1)"
  mi_code=$?
  set -e

  if [ $mi_code -eq 0 ]; then
    echo -e "  ${GREEN}PASS${NC} (maintainability index scan ran)"
  else
    echo -e "  ${RED}FAIL${NC} (exit code: $mi_code)"
    FAILED=1
  fi
  echo "  Maintainability Index (min grade B):"
  echo "$mi_out" | head -n "$HEAD_LINES"
  echo ""
else
  echo -e "  ${YELLOW}SKIP:${NC} Python module 'radon' not installed."
  echo -e "  Install with: pip install radon\n"
fi

# 7) Common issues: syntax/import errors (pyflakes)
run_check "[7/8] Common issues (pyflakes)..." "pyflakes" \
  python -m pyflakes "$CHECK_TARGET"

# 8) Optional: Ruff (recommended; fast linting + import rules)
# Enable by installing ruff. This can replace several checks over time.
echo -e "${YELLOW}[8/8] Optional: Ruff (recommended)...${NC}"
if have_python_module "ruff"; then
  set +e
  ruff_out="$(python -m ruff check "$CHECK_TARGET" 2>&1)"
  ruff_code=$?
  set -e

  if [ $ruff_code -eq 0 ]; then
    echo -e "  ${GREEN}PASS${NC}"
  else
    echo -e "  ${RED}FAIL${NC} (exit code: $ruff_code)"
    FAILED=1
  fi
  echo "$ruff_out" | head -n "$HEAD_LINES"
  echo ""
else
  echo -e "  ${YELLOW}SKIP:${NC} Python module 'ruff' not installed."
  echo -e "  Install with: pip install ruff\n"
fi

echo "=========================================="
if [ "$FAILED" -eq 0 ]; then
  echo -e "${GREEN}All code quality checks passed.${NC}"
else
  echo -e "${RED}One or more checks failed.${NC}"
fi
echo "=========================================="
echo ""

echo "Recommended tools to install:"
echo "  pip install vulture autoflake pylint mypy bandit radon pyflakes ruff"
echo ""
echo "To fix issues automatically (where possible):"
if [ "$TARGET_TYPE" = "file" ]; then
  echo "  python -m autoflake --in-place --remove-all-unused-imports --remove-unused-variables $CHECK_TARGET"
  echo "  python -m ruff check --fix $CHECK_TARGET"
else
  echo "  python -m autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables $CHECK_TARGET"
  echo "  python -m ruff check --fix $CHECK_TARGET"
fi
echo ""
echo "Usage examples:"
echo "  ./check_code_quality.sh                    # Check default directory (utils/ai_assist)"
echo "  ./check_code_quality.sh path/to/file.py    # Check single file"
echo "  ./check_code_quality.sh path/to/dir/       # Check directory"
echo ""

exit "$FAILED"
