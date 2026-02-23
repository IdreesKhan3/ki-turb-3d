# Privacy Policy for KI-TURB 3D

**First Published:** February 21, 2026 (KI-TURB 3D v2 with AI Assistant)  
**Last Updated:** February 23, 2026  
**Effective Date:** February 23, 2026

## Overview

KI-TURB 3D is a turbulence analysis and visualization suite with an optional AI assistant (Autonomous Lab). This privacy policy explains how data is handled in each usage mode.

### Two Usage Modes

KI-TURB 3D can be used in two distinct modes:

1. **Fully Private Mode (Manual)**: Use pages 01–13 directly (Overview, Theory, Energy Spectra, etc.). Fully local—all processing runs on your machine; no AI, agents, or external connections.

2. **Agent Mode (Autonomous Lab)**: Use the 00 Autonomous Lab page with the 5-agent system. Choose one of two LLM backends:
   - **Cloud (Google Gemini)**: Chat messages and tool results (e.g. from `read_file`, overview summaries) are sent to Google. Plots and raw data stay local. When agents use web search, queries go to DuckDuckGo.
   - **Local (Ollama)**: LLM runs on your machine (Mistral, Qwen Coder, etc.). When agents use web search, queries go to DuckDuckGo; otherwise processing stays local.

Agents can use web search (DuckDuckGo) with either backend when helpful—web search sends queries externally. For maximum privacy, use Manual mode.

## Data Collection and Usage

### 1. Local Data Processing (Manual Mode and Core Features)
- **Simulation Data Files**: All scientific data files are processed on your local machine; none are transmitted to external servers.
- **Visualization and Analysis**: All computational analysis, plotting, and visualization run locally.
- **Manual Mode**: Using pages 01–13 exclusively keeps all processing local; data remains on your machine.

### 2. AI Assistant – Session Data (Agent Mode Only)
When you use the AI Assistant (Autonomous Lab, page 00):

- **Chat History**: Stored temporarily in memory; cleared when the session ends.
- **User Queries**: Questions and prompts you submit are sent to the configured LLM provider (Gemini or Ollama; see below).
- **File Operations**: The AI assistant can read, modify, create, or delete files in your workspace with explicit user confirmation for destructive operations. All operations occur locally.

### 3. External Services and Third-Party APIs (Agent Mode Only)

The AI Assistant (Autonomous Lab) may communicate with external services based on your configuration:

#### **LLM Providers** (required for Agent mode):
- **Ollama (Local)**: Runs on your machine (`localhost:11434` by default). LLM data remains local unless you use a remote Ollama instance. Web search queries (when used) go to DuckDuckGo.
- **Google Gemini (Cloud)**: If configured, your prompts, chat context, and uploaded images are sent to Google's Gemini API servers. Subject to [Google's Privacy Policy](https://policies.google.com/privacy) and [Google AI Terms of Service](https://ai.google.dev/terms).

#### **Web Search** (when agents use it):
- **DuckDuckGo**: Used for web searches. Queries are sent to DuckDuckGo servers. Agents may use web search with either Gemini or Ollama when the task requires it. Subject to [DuckDuckGo's Privacy Policy](https://duckduckgo.com/privacy).

#### **Web Browsing**:
- The AI assistant can fetch content from URLs you specify. HTTP requests are made directly from your machine to the target websites.

### 4. Data Transmitted to External Services (Agent Mode)

**With Cloud (Gemini)**: Text prompts, chat history, app state, uploaded images, and tool results (e.g. file content from `read_file`, overview summaries) are sent to Google. When agents use web search, queries go to DuckDuckGo.

**With Local (Ollama)**: Only web search queries (when agents use them) are sent to DuckDuckGo. Prompts, chat, and tool results stay on your machine.

**Never transmitted** (any mode): Plots, tables, and raw turbulence data files (spectrum files, binaries) remain on disk; they are not sent externally.

**Note**: With Gemini, file content is transmitted when the agent uses `read_file` or `search_codebase`. `search_codebase` can match any file type; matching lines from CSV, DAT, or other data files may be sent to Google if the agent searches them.

### 5. API Keys and Credentials

- API keys (e.g., `GOOGLE_API_KEY`) are stored as environment variables on your local machine.
- API keys are transmitted only to the intended service provider (e.g., Google for Gemini).
- You are responsible for keeping your API keys secure.

## Data Retention

- **Session Data**: Chat history (Agent mode) and session state are stored in memory; cleared when the session ends.
- **Exported Files**: Files you explicitly export (PNG, PDF, CSV, etc.) are saved to your local machine.
- **Cloud Storage**: User data, chat logs, and files are not stored in cloud storage.

## User Control and Consent

- **Mode Selection**: **Manual mode** (pages 01–13 only) provides fully private operation. **Agent mode** (Autonomous Lab) lets you choose Cloud (Gemini) or Local (Ollama) via AI Assistant settings. Web search (DuckDuckGo) may be used with either backend when the task requires it.
- **Destructive Operations**: File deletion, renaming, and overwriting require explicit user confirmation.
- **Data Export**: You control when and where to export or download files.
- **Telemetry Control**: A Privacy Settings toggle in the app sidebar allows you to enable/disable Streamlit's anonymous usage statistics.

## Third-Party Privacy Policies

When using external services, you are also subject to their privacy policies:

- **Google Gemini**: [Google Privacy Policy](https://policies.google.com/privacy)
- **DuckDuckGo**: [DuckDuckGo Privacy Policy](https://duckduckgo.com/privacy)

## Changes to This Privacy Policy

We may update this privacy policy from time to time. The "Last Updated" date at the top indicates when the policy was last revised. Continued use of KI-TURB 3D after changes constitutes acceptance of the updated policy.

## Open Source and Transparency

KI-TURB 3D is open-source software. You can review the complete source code to verify data handling practices:
- [GitHub Repository](https://github.com/IdreesKhan3/ki-turb-3d)

## Contact Information

For questions or concerns about this privacy policy:
- **GitHub Issues**: [Report an issue](https://github.com/IdreesKhan3/ki-turb-3d/issues)
- **Email**: [Contact via repository]

## Disclaimer

**LOCAL USE ONLY**: KI-TURB 3D is designed for local, single-user operation. If you deploy this application on a server accessible to others, you become the data controller and are responsible for compliance with applicable data protection regulations (GDPR, CCPA, etc.).

## Your Rights

- **Manual mode**: All data remains on your machine; no third-party sharing.
- **Agent mode (Ollama, no web search)**: Processing stays local.
- **Agent mode (Ollama + web search)**: Search queries go to DuckDuckGo. See [DuckDuckGo's Privacy Policy](https://duckduckgo.com/privacy).
- **Agent mode (Gemini)**: Chat, tool results, and (when used) web search queries may be transmitted. See [Google's Privacy Policy](https://policies.google.com/privacy).

---

**Summary**: KI-TURB 3D prioritizes local processing. **Manual mode** is fully private. **Agent mode** transmits data only when using cloud LLM (Gemini) or web search (DuckDuckGo), limited to what is needed for your requests. We do not perform persistent data collection or user tracking.
