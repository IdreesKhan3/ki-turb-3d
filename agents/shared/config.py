"""
Configuration constants for AI Assistant module
Centralized timeout and limit values
"""

# ==========================================================
# Timeouts (seconds)
# ==========================================================
OLLAMA_CHECK_TIMEOUT = 2
GIT_OPERATION_TIMEOUT = 10
GIT_PUSH_PULL_TIMEOUT = 30
GIT_LOG_DEFAULT_LIMIT = 10
WEB_SEARCH_TIMEOUT = 10
WEB_BROWSE_TIMEOUT = 15
FILE_DOWNLOAD_TIMEOUT = 30
LLM_REQUEST_TIMEOUT = 120
OLLAMA_DEFAULT_TIMEOUT = 300
GEMINI_DEFAULT_TIMEOUT = 120
SHELL_COMMAND_TIMEOUT = 120

# ==========================================================
# Search and Result Limits
# ==========================================================
WEB_SEARCH_DEFAULT_RESULTS = 10
CODEBASE_SEARCH_MAX_RESULTS = 50
CODEBASE_SEARCH_DISPLAY_LIMIT = 20
TEXT_SEARCH_MAX_RESULTS = 50
RESOLVER_MAX_CANDIDATES = 10
CONTEXT_FILES_PREVIEW_LIMIT = 3
FILE_SEARCH_DISPLAY_LIMIT = 20
FILE_TREE_MAX_LINES = 500

# ==========================================================
# File Processing Limits
# ==========================================================
MAX_FILE_CHARS = 10_000_000  # 10MB limit - effectively no truncation for normal files
CONTEXT_FILE_LINES_THRESHOLD = 2000  # Files larger than this get truncated preview
CONTEXT_FIRST_LINES = 1000
CONTEXT_LAST_LINES = 200

# ==========================================================
# Chat History Limits
# ==========================================================
CHAT_HISTORY_PREVIEW_LINES = 200
CHAT_HISTORY_RECENT_MESSAGES = 3
WRITING_HISTORY_RECENT_MESSAGES = 5

