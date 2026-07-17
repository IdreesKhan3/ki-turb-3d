"""Compact, professional renderer for LangGraph agent activity in Streamlit."""
from __future__ import annotations

import html
import json
import re
import time
from collections import OrderedDict
from typing import Any

KI_TURB_PRO_ACTIVITY_UI_V2 = True

_AGENT_LABELS = {
    "orchestrator": "Orchestrator",
    "data_steward": "Data Steward",
    "steward": "Data Steward",
    "analyst": "Analyst",
    "visualizer": "Visualizer",
    "reviewer": "Reviewer",
    "simulation": "Simulation",
    "engineer": "Engineer",
    "agent": "Agent",
}

_STATUS_ICON = {
    "running": "↻",
    "success": "✓",
    "complete": "✓",
    "warning": "!",
    "error": "×",
    "pending": "…",
    "info": "•",
}


class LiveActivityRenderer:
    """Render a compact activity timeline instead of raw tool payloads."""

    def __init__(
        self,
        placeholder: Any,
        *,
        min_interval: float = 0.08,
        max_events: int = 12,
        max_visible_events: int = 6,
        max_detail_chars: int = 900,
    ):
        self.placeholder = placeholder
        self.min_interval = min_interval
        self.max_events = max_events
        self.max_visible_events = max_visible_events
        self.max_detail_chars = max_detail_chars
        self.events: list[dict[str, str]] = []
        self.streams: "OrderedDict[str, dict[str, str]]" = OrderedDict()
        self.seen_agents: "OrderedDict[str, None]" = OrderedDict()
        self.active_progress: dict[str, str] | None = None
        self._last_render = 0.0
        self._dirty = False

    def log(self, message: Any) -> None:
        """Accept structured workflow events, with a fallback for legacy strings."""
        if isinstance(message, dict):
            if message.get("type") == "token":
                self.stream(message)
                return
            if message.get("type") == "simulation_progress":
                self._update_progress(message)
                return
            event = self._normalise_event(message)
        else:
            event = self._legacy_event(str(message or ""))
        if not event:
            return

        agent = event.get("agent", "agent")
        self.seen_agents[agent] = None

        if event.get("type") == "tool_result":
            call_id = event.get("call_id", "")
            tool = event.get("tool", "")
            if tool in {"supervise_simulation", "compile_simulation", "start_simulation"}:
                if event.get("status") == "success":
                    self._clear_progress(complete=True)
                elif event.get("status") == "error":
                    self._clear_progress(complete=False)
            for existing in reversed(self.events):
                if existing.get("type") != "tool_start":
                    continue
                same_call = bool(call_id and existing.get("call_id") == call_id)
                same_tool = bool(
                    tool
                    and existing.get("tool") == tool
                    and existing.get("agent") == agent
                    and existing.get("status") == "running"
                )
                if same_call or same_tool:
                    existing.update(
                        status=event.get("status", "success"),
                        title=event.get("title") or existing.get("title", "Tool finished"),
                        summary=event.get("summary", ""),
                        result_details=event.get("details", ""),
                    )
                    if event.get("details"):
                        existing["details"] = self._merge_details(
                            existing.get("details", ""),
                            event.get("details", ""),
                        )
                    self._dirty = True
                    self.render(force=True)
                    return

        signature = (
            event.get("type"),
            event.get("agent"),
            event.get("title"),
            event.get("summary"),
        )
        for existing in self.events[-3:]:
            existing_signature = (
                existing.get("type"),
                existing.get("agent"),
                existing.get("title"),
                existing.get("summary"),
            )
            if signature == existing_signature:
                return

        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]
        self._dirty = True
        self.render(force=True)

    def stream(self, event: Any) -> None:
        """Merge token deltas and expose only a short live preview."""
        if not isinstance(event, dict):
            return
        delta = str(event.get("content") or "")
        if not delta:
            return
        agent = self._agent_key(event.get("agent") or "agent")
        self.seen_agents[agent] = None
        stream_id = str(event.get("stream_id") or agent)
        entry = self.streams.setdefault(stream_id, {"agent": agent, "text": ""})
        entry["text"] += delta
        if len(entry["text"]) > 1600:
            entry["text"] = entry["text"][-1600:]
        self._dirty = True
        now = time.monotonic()
        if delta.endswith(("\n", ".", "!", "?", ":")) or now - self._last_render >= self.min_interval:
            self.render()

    def snapshot(self) -> str:
        """Return a concise persistent transcript for the chat-history expander."""
        if not self.events and not self.active_progress:
            return ""
        lines = [f"**Agent workflow** · {len(self.events)} updates"]
        if self.active_progress:
            try:
                pct = float(self.active_progress.get("progress") or 0.0)
            except (TypeError, ValueError):
                pct = 0.0
            summary = self._one_line(self.active_progress.get("summary", ""), 180)
            lines.append(f"- ↻ **Simulation** — {pct:.1f}% {summary}".rstrip())
        for event in self.events[-self.max_visible_events :]:
            status = event.get("status", "info")
            icon = _STATUS_ICON.get(status, "•")
            agent = self._agent_label(event.get("agent", "agent"))
            title = event.get("title", "Activity")
            summary = self._one_line(event.get("summary", ""), 140)
            line = f"- {icon} **{agent}** — {title}"
            if summary:
                line += f": {summary}"
            lines.append(line)
        return "\n".join(lines)

    def render(self, *, force: bool = False) -> None:
        if not self._dirty and not force:
            return
        now = time.monotonic()
        if not force and now - self._last_render < self.min_interval:
            return
        try:
            self.placeholder.markdown(self._html(), unsafe_allow_html=True)
        except Exception as exc:
            if type(exc).__name__ not in {"NoSessionContext", "StreamlitAPIException"}:
                raise
        self._last_render = now
        self._dirty = False

    def _html(self) -> str:
        active_agent = self._active_agent()
        chips = "".join(
            self._agent_chip(agent, active=(agent == active_agent))
            for agent in self.seen_agents.keys()
        )
        # Newest first so a height-capped panel stays useful when Streamlit re-renders.
        visible = list(reversed(self.events[-self.max_visible_events :]))
        event_rows = "".join(self._event_html(event) for event in visible)
        live_preview = self._live_preview_html()
        progress_panel = self._progress_html()
        hidden = max(0, len(self.events) - self.max_visible_events)
        more_html = (
            f'<div class="ki-pro-more">{hidden} earlier update(s) not shown</div>'
            if hidden
            else ""
        )
        return f"""
<style>
.ki-pro-shell {{
  border: 1px solid rgba(128,128,128,.24); border-radius: 14px; overflow: hidden;
  background: rgba(128,128,128,.035); margin: .2rem 0 .5rem 0;
  max-height: min(38vh, 22rem); display: flex; flex-direction: column;
}}
.ki-pro-progress {{
  flex: 0 0 auto; padding:.55rem .85rem; border-bottom:1px solid rgba(128,128,128,.18);
  background:rgba(76,139,245,.04);
}}
.ki-pro-progress-label {{
  display:flex; justify-content:space-between; gap:.75rem; align-items:baseline;
  font-size:.78rem; margin-bottom:.35rem;
}}
.ki-pro-progress-title {{font-weight:650;}}
.ki-pro-progress-pct {{font-variant-numeric:tabular-nums; opacity:.72;}}
.ki-pro-progress-track {{
  height:.45rem; border-radius:999px; background:rgba(128,128,128,.16); overflow:hidden;
}}
.ki-pro-progress-fill {{
  height:100%; border-radius:999px; background:linear-gradient(90deg,#4c8bf5,#2ea043);
  transition:width .35s ease;
}}
.ki-pro-progress-fill.error {{background:linear-gradient(90deg,#d1242f,#bf8700);}}
.ki-pro-progress-fill.complete {{background:linear-gradient(90deg,#2ea043,#3fb950);}}
.ki-pro-progress-summary {{font-size:.72rem; opacity:.68; margin-top:.3rem; line-height:1.3;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;}}
.ki-pro-head {{
  flex: 0 0 auto; display:flex; align-items:center; justify-content:space-between; gap:.75rem;
  padding:.55rem .85rem; border-bottom:1px solid rgba(128,128,128,.18);
}}
.ki-pro-title {{font-weight:650; font-size:.9rem; letter-spacing:-.01em;}}
.ki-pro-live {{font-size:.65rem; font-weight:700; letter-spacing:.08em; padding:.15rem .42rem;
  border-radius:999px; background:rgba(46,160,67,.12); color:#2ea043;}}
.ki-pro-agents {{flex: 0 0 auto; display:flex; flex-wrap:wrap; gap:.3rem; padding:.45rem .85rem .1rem .85rem;}}
.ki-agent-chip {{font-size:.7rem; padding:.15rem .42rem; border-radius:999px;
  border:1px solid rgba(128,128,128,.22); opacity:.72;}}
.ki-agent-chip.active {{opacity:1; border-color:rgba(46,160,67,.5); background:rgba(46,160,67,.08);}}
.ki-pro-list {{
  flex: 1 1 auto; min-height: 0; max-height: 11rem; overflow-y: auto; overscroll-behavior: contain;
  padding:.25rem .85rem .45rem .85rem;
}}
.ki-pro-more {{font-size:.68rem; opacity:.55; padding:.15rem 0 .25rem 0;}}
.ki-pro-row {{display:grid; grid-template-columns:1.2rem 1fr; gap:.4rem; padding:.32rem 0;
  border-bottom:1px solid rgba(128,128,128,.12);}}
.ki-pro-row:last-child {{border-bottom:0;}}
.ki-pro-icon {{width:1.05rem; height:1.05rem; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:.68rem; font-weight:700; margin-top:.06rem;
  background:rgba(128,128,128,.12);}}
.ki-pro-icon.success,.ki-pro-icon.complete {{color:#2ea043;background:rgba(46,160,67,.12);}}
.ki-pro-icon.running {{animation:ki-spin 1.2s linear infinite;color:#4c8bf5;background:rgba(76,139,245,.12);}}
.ki-pro-icon.warning {{color:#bf8700;background:rgba(191,135,0,.12);}}
.ki-pro-icon.error {{color:#d1242f;background:rgba(209,36,47,.12);}}
@keyframes ki-spin {{to {{transform:rotate(360deg)}}}}
.ki-pro-meta {{font-size:.64rem; opacity:.58; text-transform:uppercase; letter-spacing:.045em;}}
.ki-pro-event-title {{font-size:.8rem; font-weight:620; margin-top:.01rem;}}
.ki-pro-summary {{font-size:.74rem; opacity:.74; margin-top:.04rem; line-height:1.3;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;}}
.ki-pro-details summary {{font-size:.68rem; opacity:.62; cursor:pointer; margin-top:.12rem;}}
.ki-pro-details pre {{white-space:pre-wrap; overflow-wrap:anywhere; font-size:.66rem; line-height:1.3;
  padding:.4rem; border-radius:8px; background:rgba(128,128,128,.08); max-height:5rem; overflow:auto;}}
.ki-live-copy {{flex: 0 0 auto; margin:.1rem .85rem .55rem .85rem; padding:.45rem .55rem; border-radius:10px;
  background:rgba(76,139,245,.07); border:1px solid rgba(76,139,245,.14);}}
.ki-live-copy .label {{font-size:.64rem; opacity:.62; text-transform:uppercase; letter-spacing:.045em;}}
.ki-live-copy .text {{font-size:.76rem; line-height:1.35; margin-top:.08rem; opacity:.86;
  display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;}}
.ki-cursor {{display:inline-block; width:.42rem; height:.9rem; margin-left:.18rem;
  background:currentColor; opacity:.52; vertical-align:-.08rem; animation:ki-blink 1s step-end infinite;}}
@keyframes ki-blink {{50% {{opacity:0}}}}
</style>
<div class="ki-pro-shell">
  <div class="ki-pro-head"><div class="ki-pro-title">Agent workflow</div><div class="ki-pro-live">LIVE</div></div>
  {progress_panel}
  <div class="ki-pro-agents">{chips or self._agent_chip('orchestrator', active=True)}</div>
  <div class="ki-pro-list">{more_html}{event_rows or self._empty_row()}</div>
  {live_preview}
</div>
"""

    def _progress_html(self) -> str:
        if not self.active_progress:
            return ""
        try:
            pct = float(self.active_progress.get("progress") or 0.0)
        except (TypeError, ValueError):
            pct = 0.0
        pct = max(0.0, min(100.0, pct))
        status = self.active_progress.get("status", "running")
        fill_class = "complete" if status == "success" else "error" if status == "error" else ""
        title = html.escape(self.active_progress.get("title", "Simulation"))
        summary = html.escape(self._one_line(self.active_progress.get("summary", ""), 220))
        job_id = html.escape(self.active_progress.get("job_id", ""))
        summary_html = (
            f'<div class="ki-pro-progress-summary">{summary}</div>'
            if summary
            else ""
        )
        job_html = (
            f'<div class="ki-pro-progress-summary">job: {job_id}</div>'
            if job_id and not summary
            else ""
        )
        return (
            '<div class="ki-pro-progress">'
            '<div class="ki-pro-progress-label">'
            f'<div class="ki-pro-progress-title">{title}</div>'
            f'<div class="ki-pro-progress-pct">{pct:.1f}%</div>'
            "</div>"
            '<div class="ki-pro-progress-track">'
            f'<div class="ki-pro-progress-fill {fill_class}" style="width:{pct:.1f}%;"></div>'
            "</div>"
            f"{summary_html}{job_html}"
            "</div>"
        )

    def _update_progress(self, raw: dict[str, Any]) -> None:
        progress = dict(raw)
        try:
            progress["progress"] = str(max(0.0, min(100.0, float(raw.get("progress") or 0.0))))
        except (TypeError, ValueError):
            progress["progress"] = "0"
        progress["type"] = "simulation_progress"
        progress["agent"] = self._agent_key(raw.get("agent") or "simulation")
        progress["status"] = str(raw.get("status") or "running").lower()
        progress["title"] = str(raw.get("title") or "Simulation")
        progress["summary"] = str(raw.get("summary") or "")
        progress["job_id"] = str(raw.get("job_id") or "")
        self.active_progress = {str(k): str(v) for k, v in progress.items()}
        self.seen_agents[progress["agent"]] = None
        self._dirty = True
        self.render(force=True)

    def _clear_progress(self, *, complete: bool) -> None:
        if not self.active_progress:
            return
        if complete:
            self.active_progress["progress"] = "100"
            self.active_progress["status"] = "success"
        self._dirty = True
        self.render(force=True)
        self.active_progress = None

    def _event_html(self, event: dict[str, str]) -> str:
        status = event.get("status", "info")
        icon = _STATUS_ICON.get(status, "•")
        agent = html.escape(self._agent_label(event.get("agent", "agent")))
        kind = html.escape(event.get("kind") or event.get("type", "activity").replace("_", " "))
        title = html.escape(event.get("title", "Activity"))
        summary = html.escape(self._one_line(event.get("summary", ""), 160))
        details = event.get("details", "")
        details_html = ""
        # Keep Details collapsed; long payloads must not grow the page.
        if details and len(details.strip()) > 40:
            clipped = details[: self.max_detail_chars]
            if len(details) > self.max_detail_chars:
                clipped += "\n…"
            details_html = (
                '<details class="ki-pro-details"><summary>Details</summary>'
                f"<pre>{html.escape(clipped)}</pre></details>"
            )
        summary_html = f'<div class="ki-pro-summary">{summary}</div>' if summary else ""
        return (
            '<div class="ki-pro-row">'
            f'<div class="ki-pro-icon {html.escape(status)}">{html.escape(icon)}</div>'
            '<div>'
            f'<div class="ki-pro-meta">{agent} · {kind}</div>'
            f'<div class="ki-pro-event-title">{title}</div>'
            f"{summary_html}{details_html}</div></div>"
        )

    def _live_preview_html(self) -> str:
        if not self.streams:
            return ""
        entry = next(reversed(self.streams.values()))
        text = self._one_line(entry.get("text", ""), 220)
        if not text:
            return ""
        agent = html.escape(self._agent_label(entry.get("agent", "agent")))
        return (
            '<div class="ki-live-copy">'
            f'<div class="label">{agent} · composing</div>'
            f'<div class="text">{html.escape(text)}<span class="ki-cursor"></span></div>'
            "</div>"
        )

    def _normalise_event(self, raw: dict[str, Any]) -> dict[str, str]:
        event = {str(k): str(v) if v is not None else "" for k, v in raw.items()}
        event["type"] = str(raw.get("type") or "activity")
        event["agent"] = self._agent_key(raw.get("agent") or "agent")
        event["status"] = str(raw.get("status") or "info").lower()
        event["title"] = str(raw.get("title") or "Activity")
        event["summary"] = str(raw.get("summary") or "")
        event["details"] = str(raw.get("details") or "")
        event["tool"] = str(raw.get("tool") or "")
        event["call_id"] = str(raw.get("call_id") or "")
        event["kind"] = str(raw.get("kind") or event["type"].replace("_", " "))
        return event

    def _legacy_event(self, text: str) -> dict[str, str] | None:
        text = text.strip()
        if not text:
            return None
        match = re.match(r"\*\*\[([^\]]+)\]\*\*\s*(.*)", text, flags=re.S)
        if not match:
            return {
                "type": "activity",
                "agent": "agent",
                "status": "info",
                "title": "Agent update",
                "summary": self._one_line(text, 220),
                "details": text if len(text) > 220 else "",
                "kind": "activity",
            }
        label, body = match.groups()
        label_key = label.lower().replace(" ", "_")
        body = body.strip()
        status = "error" if "error" in label_key else "warning" if "warning" in label_key else "info"
        return {
            "type": "activity",
            "agent": self._agent_key(label_key),
            "status": status,
            "title": label.replace("_", " ").title(),
            "summary": self._one_line(body, 220),
            "details": body if len(body) > 220 else "",
            "kind": "activity",
        }

    def _active_agent(self) -> str:
        if self.streams:
            return next(reversed(self.streams.values())).get("agent", "agent")
        for event in reversed(self.events):
            if event.get("status") == "running":
                return event.get("agent", "agent")
        return next(reversed(self.seen_agents), "orchestrator")

    def _agent_chip(self, agent: str, *, active: bool) -> str:
        label = html.escape(self._agent_label(agent))
        cls = "ki-agent-chip active" if active else "ki-agent-chip"
        return f'<span class="{cls}">{label}</span>'

    @staticmethod
    def _empty_row() -> str:
        return (
            '<div class="ki-pro-row"><div class="ki-pro-icon running">↻</div>'
            '<div><div class="ki-pro-meta">Orchestrator · starting</div>'
            '<div class="ki-pro-event-title">Preparing the agent team</div></div></div>'
        )

    @staticmethod
    def _agent_key(value: Any) -> str:
        key = str(value or "agent").strip().lower().replace(" ", "_")
        return key.removesuffix("_agent")

    @staticmethod
    def _agent_label(value: str) -> str:
        key = LiveActivityRenderer._agent_key(value)
        return _AGENT_LABELS.get(key, key.replace("_", " ").title())

    @staticmethod
    def _one_line(value: Any, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) > limit:
            return text[: limit - 1].rstrip() + "…"
        return text

    @staticmethod
    def _merge_details(first: str, second: str) -> str:
        parts = [part.strip() for part in (first, second) if part and part.strip()]
        return "\n\n".join(parts)


__all__ = ["LiveActivityRenderer"]
