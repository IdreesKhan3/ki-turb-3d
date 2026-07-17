"""
Web search, literature lookup, browsing, and download tools.

Backends (tried in order until useful results appear):
  1. Optional paid/API keys: Tavily, Brave, SerpAPI (env vars)
  2. DuckDuckGo Lite HTML (no key; primary free path)
  3. DuckDuckGo Instant Answer JSON
  4. Wikipedia OpenSearch
  5. DuckDuckGo HTML (last resort; often bot-challenged)

arXiv Atom API powers search_research_papers.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

from ...shared.config import (
    FILE_DOWNLOAD_TIMEOUT,
    WEB_BROWSE_CONTENT_CHARS,
    WEB_BROWSE_TIMEOUT,
    WEB_SEARCH_DEFAULT_RESULTS,
    WEB_SEARCH_TIMEOUT,
    WEB_SEARCH_USER_AGENT,
)


def _decode_ddg_redirect(raw_link: str) -> str:
    """Unwrap DuckDuckGo //duckduckgo.com/l/?uddg=… redirects."""
    if not raw_link:
        return ""
    if "uddg=" in raw_link:
        try:
            parsed = urlparse(raw_link if "://" in raw_link else f"https:{raw_link}")
            uddg = parse_qs(parsed.query).get("uddg")
            if uddg:
                return unquote(uddg[0])
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    if raw_link.startswith("//"):
        return "https:" + raw_link
    return raw_link


def _normalize_results(results: List[Dict[str, Any]], num_results: int) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or item.get("href") or item.get("url") or "").strip()
        snippet = str(item.get("snippet") or item.get("body") or item.get("content") or "").strip()
        if not title and not link:
            continue
        key = link or title.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"title": title, "link": link, "snippet": snippet})
        if len(cleaned) >= num_results:
            break
    return cleaned


class WebSearchTools:
    """Tools for web search, research papers, browsing, and downloads."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": WEB_SEARCH_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def web_search(self, query: str, num_results: int = WEB_SEARCH_DEFAULT_RESULTS) -> Dict[str, Any]:
        """Search the web; try multiple backends until results are found."""
        query = (query or "").strip()
        if not query:
            return {"success": False, "error": "empty query", "message": "query required"}

        errors: List[str] = []
        backends = [
            ("tavily", self._tavily_search),
            ("brave", self._brave_search),
            ("serpapi", self._serpapi_search),
            ("wikipedia", self._wikipedia_search),
            ("duckduckgo_lite", self._duckduckgo_lite_search),
            ("duckduckgo_instant", self._duckduckgo_instant_search),
            ("duckduckgo_html", self._duckduckgo_html_search),
        ]
        for name, fn in backends:
            try:
                results = fn(query, num_results)
                results = _normalize_results(results or [], num_results)
                if results:
                    return {
                        "success": True,
                        "query": query,
                        "backend": name,
                        "results": results,
                        "total": len(results),
                    }
            except Exception as exc:  # noqa: BLE001 — collect and continue backends
                errors.append(f"{name}: {exc}")

        return {
            "success": False,
            "query": query,
            "results": [],
            "total": 0,
            "error": "; ".join(errors) if errors else "no results from any backend",
            "message": (
                "Web search returned no results. Configure TAVILY_API_KEY, "
                "BRAVE_SEARCH_API_KEY, or SERPAPI_API_KEY for stronger coverage, "
                "or retry with a more specific query."
            ),
        }

    def _tavily_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        key = (os.environ.get("TAVILY_API_KEY") or "").strip()
        if not key:
            return []
        response = self.session.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": num_results},
            timeout=WEB_SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in data.get("results") or []
        ]

    def _brave_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        key = (os.environ.get("BRAVE_SEARCH_API_KEY") or "").strip()
        if not key:
            return []
        response = self.session.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": num_results},
            headers={"Accept": "application/json", "X-Subscription-Token": key},
            timeout=WEB_SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        web = (data.get("web") or {}).get("results") or []
        return [
            {
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("description", ""),
            }
            for item in web
        ]

    def _serpapi_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        key = (os.environ.get("SERPAPI_API_KEY") or "").strip()
        if not key:
            return []
        response = self.session.get(
            "https://serpapi.com/search",
            params={"engine": "google", "q": query, "api_key": key, "num": num_results},
            timeout=WEB_SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in data.get("organic_results") or []
        ]

    def _duckduckgo_lite_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        response = self.session.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            timeout=WEB_SEARCH_TIMEOUT,
        )
        text = response.text or ""
        if response.status_code >= 400 or response.status_code == 202:
            return []
        if "anomaly" in text.lower() or "captcha" in text.lower():
            return []
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

        soup = BeautifulSoup(text, "html.parser")
        results: List[Dict[str, Any]] = []
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href") or "")
            if "uddg=" not in href:
                continue
            title = anchor.get_text(strip=True)
            if not title or title.lower() in {"cached", "proxied"}:
                continue
            link = _decode_ddg_redirect(href)
            snippet = ""
            parent = anchor.find_parent("tr")
            if parent is not None:
                sibling = parent.find_next_sibling("tr")
                if sibling is not None:
                    snippet = sibling.get_text(" ", strip=True)
            results.append({"title": title, "link": link, "snippet": snippet})
            if len(results) >= num_results:
                break
        return results

    def _duckduckgo_instant_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        response = self.session.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=WEB_SEARCH_TIMEOUT,
        )
        try:
            data = response.json()
        except ValueError:
            return []

        results: List[Dict[str, Any]] = []
        abstract = (data.get("AbstractText") or "").strip()
        abstract_url = (data.get("AbstractURL") or "").strip()
        heading = (data.get("Heading") or query).strip()
        if abstract and abstract_url:
            results.append({"title": heading, "link": abstract_url, "snippet": abstract})

        def _walk(topics: List[Any]) -> None:
            for topic in topics or []:
                if len(results) >= num_results:
                    return
                if isinstance(topic, dict) and "Topics" in topic:
                    _walk(topic.get("Topics") or [])
                    continue
                if not isinstance(topic, dict):
                    continue
                text = str(topic.get("Text") or "").strip()
                url = str(topic.get("FirstURL") or "").strip()
                if text and url:
                    results.append({"title": text.split(" - ")[0][:120], "link": url, "snippet": text})

        _walk(data.get("RelatedTopics") or [])
        for item in data.get("Results") or []:
            if len(results) >= num_results:
                break
            if isinstance(item, dict):
                text = str(item.get("Text") or "").strip()
                url = str(item.get("FirstURL") or "").strip()
                if text and url:
                    results.append({"title": text.split(" - ")[0][:120], "link": url, "snippet": text})
        return results

    def _wikipedia_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """OpenSearch with progressive query shortening when the full phrase misses."""
        candidates = [query]
        words = [w for w in re.split(r"\s+", query.strip()) if w]
        if len(words) >= 2:
            candidates.append(" ".join(words[:2]))
        if words:
            candidates.append(words[0])
        # Prefer turbulence-related reformulations when relevant.
        lower = query.lower()
        if "kolmogorov" in lower and "spectrum" in lower:
            candidates.insert(0, "Energy cascade")
            candidates.insert(1, "Kolmogorov structure function")
        if "lattice" in lower and "boltzmann" in lower:
            candidates.insert(0, "Lattice Boltzmann methods")

        seen_q: set[str] = set()
        for candidate in candidates:
            key = candidate.lower()
            if key in seen_q:
                continue
            seen_q.add(key)
            response = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": candidate,
                    "limit": num_results,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=WEB_SEARCH_TIMEOUT,
            )
            if response.status_code >= 400:
                continue
            data = response.json()
            titles = data[1] if len(data) > 1 else []
            snippets = data[2] if len(data) > 2 else []
            urls = data[3] if len(data) > 3 else []
            if not titles:
                continue
            results = []
            for i, title in enumerate(titles):
                results.append({
                    "title": title,
                    "link": urls[i] if i < len(urls) else "",
                    "snippet": snippets[i] if i < len(snippets) else "",
                })
            return results
        return []

    def _duckduckgo_html_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        response = self.session.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=WEB_SEARCH_TIMEOUT,
        )
        text = response.text or ""
        if response.status_code >= 400 or "anomaly" in text.lower() or "captcha" in text.lower():
            return []
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]

        soup = BeautifulSoup(text, "html.parser")
        results: List[Dict[str, Any]] = []
        for block in soup.find_all("div", class_="result")[: num_results * 2]:
            title_elem = block.find("a", class_="result__a")
            snippet_elem = block.find("a", class_="result__snippet") or block.find(
                "div", class_="result__snippet"
            )
            if not title_elem:
                continue
            results.append({
                "title": title_elem.get_text(strip=True),
                "link": _decode_ddg_redirect(title_elem.get("href", "")),
                "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
            })
            if len(results) >= num_results:
                break
        return results

    def search_research_papers(
        self,
        query: str,
        source: str = "arxiv",
        max_results: int = WEB_SEARCH_DEFAULT_RESULTS,
    ) -> Dict[str, Any]:
        """Search research papers (arXiv)."""
        if source.lower() != "arxiv":
            return {
                "success": False,
                "error": f"Unknown source: {source}. Only 'arxiv' is supported",
            }
        return self._search_arxiv(query, max_results)

    def _search_arxiv(self, query: str, max_results: int) -> Dict[str, Any]:
        try:
            response = self.session.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": query,
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                },
                timeout=WEB_BROWSE_TIMEOUT,
            )
            response.raise_for_status()
            from xml.etree import ElementTree as ET

            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                link = entry.find("atom:id", ns)
                authors = [
                    author.find("atom:name", ns).text
                    for author in entry.findall("atom:author", ns)
                    if author.find("atom:name", ns) is not None
                ]
                url = link.text if link is not None else ""
                papers.append({
                    "title": (title.text or "").strip() if title is not None else "",
                    "authors": authors,
                    "summary": (summary.text or "").strip() if summary is not None else "",
                    "url": url,
                    "pdf_url": url.replace("/abs/", "/pdf/") + ".pdf" if url else "",
                })
            return {
                "success": True,
                "query": query,
                "source": "arxiv",
                "papers": papers,
                "total": len(papers),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": str(exc),
                "message": f"arXiv search failed: {exc}",
            }

    def download_file(self, url: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Download a file from a URL."""
        try:
            response = self.session.get(url, stream=True, timeout=FILE_DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            if not save_path:
                filename = url.split("/")[-1]
                if "?" in filename:
                    filename = filename.split("?")[0]
                content_disp = response.headers.get("Content-Disposition", "")
                if content_disp:
                    match = re.search(r'filename="?([^"]+)"?', content_disp)
                    if match:
                        filename = match.group(1)
                save_path = Path.cwd() / filename
            save_path_p = Path(save_path)
            save_path_p.parent.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            with open(save_path_p, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
                        downloaded += len(chunk)
            return {
                "success": True,
                "url": url,
                "filepath": str(save_path_p),
                "size": downloaded,
                "message": f"Downloaded {save_path_p.name} ({downloaded} bytes)",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": str(exc),
                "message": f"Download failed: {exc}",
            }

    def browse_web(self, url: str) -> Dict[str, Any]:
        """Fetch a page and extract readable text for in-action learning."""
        if url.startswith("file://") or (
            not url.startswith(("http://", "https://")) and Path(url).exists()
        ):
            file_path = url.replace("file://", "") if url.startswith("file://") else url
            return {
                "success": False,
                "error": "file:// URLs are not supported for web browsing",
                "message": (
                    f"Cannot browse file:// URLs. Use 'read_file' instead for: {file_path}"
                ),
            }
        try:
            response = self.session.get(url, timeout=WEB_BROWSE_TIMEOUT)
            response.raise_for_status()
            try:
                from bs4 import BeautifulSoup  # type: ignore[import-untyped]

                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = " ".join(chunk for chunk in chunks if chunk)
                title_el = soup.find("title")
                title_text = title_el.get_text() if title_el else ""
                links = [a.get("href", "") for a in soup.find_all("a", href=True)]
                limit = WEB_BROWSE_CONTENT_CHARS
                return {
                    "success": True,
                    "url": url,
                    "title": title_text,
                    "content": text[:limit],
                    "links": links[:40],
                    "full_content_available": len(text) > limit,
                }
            except ImportError:
                return {
                    "success": True,
                    "url": url,
                    "content": response.text[:WEB_BROWSE_CONTENT_CHARS],
                    "message": "BeautifulSoup4 not installed. Install for better HTML parsing.",
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": str(exc),
                "message": f"Failed to browse {url}: {exc}",
            }
