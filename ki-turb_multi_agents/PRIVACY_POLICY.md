# Privacy Policy for KI-TURB 3D

**First Published:** February 21, 2026 (KI-TURB 3D with AI Assistant)  
**Last Updated:** July 16, 2026  
**Effective Date:** July 16, 2026  
**Applies to:** KI-TURB 3D **v3.0.0**

## Overview

KI-TURB 3D is a turbulence analysis and visualization suite with an optional multi-agent Autonomous Lab. This policy explains how data is handled in each usage mode.

### Two Usage Modes

1. **Manual mode (fully private)** — Pages **01–13** only. All analysis, plotting, and file I/O run on your machine. No LLM, agents, or outbound network calls from the app for science workflows.

2. **Agent mode (Autonomous Lab, page 00)** — Seven-role LangGraph team (Orchestrator, Steward, Simulation, Analyst, Visualizer, Reviewer, Engineer). You choose an LLM backend:
   - **DeepSeek (cloud, UI default)** — Chat/tool text is sent to DeepSeek.
   - **Google Gemini (cloud)** — Chat/tool text (and multimodal inputs when used) is sent to Google.
   - **Ollama (local)** — LLM runs on your machine (e.g. Qwen Coder).

In Agent mode, specialists may invoke outbound research tools (`web_search`,
`search_research_papers`, `browse_web`, `download_file`) when a task needs live
docs or literature — those requests leave your machine. Local-only code search
tools (`search_codebase`, `semantic_search`, `regex_search`, symbol finders) stay
on disk. For maximum privacy, use **Manual** mode.

## Data Collection and Usage

### 1. Local Data Processing (Manual mode and core features)

- Simulation data, spectra, volumes, and exported figures are processed and stored **locally**.
- Manual mode does not send scientific payloads to LLM or search providers.

### 2. AI Assistant – Session Data (Agent mode only)

- **Chat history / turn memory** — Held in the Streamlit session; cleared when the session ends (not uploaded to KI-TURB servers; there are none).
- **User prompts** — Sent to the selected LLM provider (DeepSeek, Gemini, or local Ollama).
- **Tool results** — Summaries and file excerpts returned by tools (e.g. `read_file`, `search_codebase`, job status text) may be included in later LLM turns when using a **cloud** backend.
- **File & simulation ops** — Run locally under the project. Destructive edits, downloads, and costly simulation steps (build/compile/start/fetch/postprocess, etc.) require **explicit user confirmation**.

### 3. External Services (Agent mode only)

#### LLM providers

| Backend | Where prompts go | Notes |
|---------|------------------|--------|
| **DeepSeek** | DeepSeek API | Subject to DeepSeek’s terms/privacy policy for API use |
| **Google Gemini** | Google Gemini API | [Google Privacy Policy](https://policies.google.com/privacy), [Google AI Terms](https://ai.google.dev/terms) |
| **Ollama** | `localhost` (default) | Stays local unless you point Ollama at a remote host |

#### Outbound research tools (Agent mode, when used)

These tools make network requests from your machine. Returned titles, snippets,
page text, or downloaded bytes may later appear in **cloud LLM** prompts if a
cloud backend is selected.

| Tool | What leaves the machine | Destinations |
|------|-------------------------|--------------|
| **`web_search`** | Search query (+ result metadata back) | Tried in order until useful hits: optional **Tavily** / **Brave** / **SerpAPI** (if you set their API keys), then **Wikipedia** OpenSearch, then **DuckDuckGo** (Lite / Instant Answer / HTML). |
| **`search_research_papers`** | Literature query | **arXiv** Atom API (`export.arxiv.org`) |
| **`browse_web`** | Target URL; page text fetched | Any HTTP(S) URL the agent opens (docs, forums, paper pages, …) |
| **`download_file`** | Target URL; file saved under the project | Any HTTP(S) URL — **requires user confirmation** |

Optional search API keys (only if you configure them): `TAVILY_API_KEY`,
`BRAVE_SEARCH_API_KEY`, `SERPAPI_API_KEY`. Without those keys, `web_search`
falls back to free paths (Wikipedia / DuckDuckGo).

**Local (no egress):** `search_codebase`, `semantic_search`, `regex_search`,
`find_symbol_definitions`, `find_symbol_references`, and related repo inspect
tools read only your project tree.

### 4. What is and is not transmitted

| Category | Manual | Agent + Ollama (no outbound research tools) | Agent + cloud LLM | Agent + `web_search` / arXiv / browse / download |
|----------|--------|---------------------------------------------|-------------------|--------------------------------------------------|
| Raw turbulence files / plot binaries | Local | Local | Local on disk | Local on disk |
| Chat prompts & tool text | N/A | Local LLM | Sent to cloud LLM | As left + research tool text may be re-sent to the LLM |
| Search / arXiv / URL fetch / download | N/A | Not used | Optional | Sent to those services / URLs |

**Never uploaded to a KI-TURB cloud:** the project does not operate a central telemetry or chat archive. Plots and raw volumes remain files on your machine; they are not automatically attached to LLM calls.

**Caution (cloud LLM):** If agents `read_file` or `search_codebase` on data files, matching **text excerpts** can appear in the prompt sent to DeepSeek/Gemini.

### 5. API Keys and Credentials

- Keys such as `DEEPSEEK_API_KEY` and `GOOGLE_API_KEY` live in your environment / launcher scripts.
- They are sent only to the corresponding provider.
- You are responsible for securing them.

## Data Retention

- **Session state** — In-memory for the running Streamlit process; gone when the process ends.
- **Exports / simulation jobs** — Written under your project (e.g. `exports/`, `simulations/job_*`) only when you or the confirmed agent workflow create them.
- **No KI-TURB cloud retention** of chats or datasets.

## User Control and Consent

- Choose **Manual** vs **Agent**, and which LLM backend, in the UI.
- Confirm destructive file ops and expensive simulation lifecycle tools before they run.
- Control exports and downloads explicitly.
- **Telemetry:** Privacy Settings in the sidebar can disable Streamlit’s anonymous usage statistics.

## Third-Party Privacy Policies

When a tool contacts a service, you are also subject to that provider’s policy:

- DeepSeek (API terms / privacy for your account region)
- [Google Privacy Policy](https://policies.google.com/privacy) (Gemini)
- [DuckDuckGo Privacy Policy](https://duckduckgo.com/privacy) (`web_search` fallbacks)
- [Wikipedia / Wikimedia](https://foundation.wikimedia.org/wiki/Policy:Privacy_policy) (`web_search` Wikipedia path)
- Tavily / Brave / SerpAPI privacy terms **if** you enable those API keys for `web_search`
- [arXiv privacy](https://info.arxiv.org/help/policies/privacy_policy.html) (`search_research_papers`)
- Any site contacted via `browse_web` or `download_file`

## Changes to This Privacy Policy

The **Last Updated** date above reflects the latest revision. Continued use after changes constitutes acceptance of the updated policy for that release.

## Open Source and Transparency

Source is public: [github.com/IdreesKhan3/ki-turb-3d](https://github.com/IdreesKhan3/ki-turb-3d).

## Contact

- **GitHub Issues:** [Report an issue](https://github.com/IdreesKhan3/ki-turb-3d/issues)

## Disclaimer

**LOCAL USE BY DEFAULT.** If you deploy KI-TURB 3D on a multi-user or public server, **you** become the data controller and must comply with applicable law (GDPR, CCPA, etc.).

## Your Rights (summary)

- **Manual** — Fully local science workflow.  
- **Agent + Ollama, no outbound research tools** — LLM local; no `web_search` / arXiv / browse / download egress.  
- **Agent + cloud LLM** — Prompts/tool text leave the machine.  
- **Agent + `web_search` / `search_research_papers` / `browse_web` / `download_file`** — Those queries and URL fetches leave the machine (and may later be summarized into a cloud LLM turn).  

---

**Summary:** Manual mode is private. Agent mode (v3) sends data only as needed for your chosen LLM and optional outbound research tools (`web_search`, `search_research_papers`, `browse_web`, `download_file`). KI-TURB does not run persistent cloud collection or user tracking.
