"""
Web Search and Research Tools Module
Provides Google Search, research paper search, web browsing, and file download capabilities
"""

import os
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time
from urllib.parse import urljoin, urlparse
import re

from ...shared.config import (
    WEB_SEARCH_TIMEOUT,
    WEB_BROWSE_TIMEOUT,
    FILE_DOWNLOAD_TIMEOUT,
    WEB_SEARCH_DEFAULT_RESULTS,
)


class WebSearchTools:
    """Tools for web search, research papers, and file downloads"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def web_search(self, query: str, num_results: int = WEB_SEARCH_DEFAULT_RESULTS) -> Dict[str, Any]:
        """
        Perform web search using DuckDuckGo
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            Dictionary with search results
        """
        try:
            # Use DuckDuckGo (free, no API key needed)
            return self._duckduckgo_search(query, num_results)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Search failed: {str(e)}"
            }
    
    def _duckduckgo_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using DuckDuckGo (free, no API key)"""
        try:
            # Use DuckDuckGo HTML search
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            response = self.session.get(url, params=params, timeout=WEB_SEARCH_TIMEOUT)
            response.raise_for_status()
            
            # Parse HTML results (simple extraction)
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
            from urllib.parse import unquote, parse_qs, urlparse
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result')[:num_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    # Extract and decode the actual URL from DuckDuckGo's redirect
                    raw_link = title_elem.get('href', '')
                    actual_link = raw_link
                    
                    # DuckDuckGo wraps URLs in redirect links like: //duckduckgo.com/l/?uddg=https%3A%2F%2F...
                    if 'uddg=' in raw_link:
                        try:
                            # Extract the uddg parameter which contains the actual URL
                            parsed = urlparse(raw_link)
                            query_params = parse_qs(parsed.query)
                            if 'uddg' in query_params:
                                actual_link = unquote(query_params['uddg'][0])
                        except (ValueError, KeyError, IndexError, TypeError):
                            actual_link = raw_link
                    
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "link": actual_link,
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else ""
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total": len(results)
            }
        except ImportError:
            # If BeautifulSoup not available, use simple regex parsing
            return self._simple_web_search(query, num_results)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _simple_web_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Simple web search fallback"""
        # For now, return a message suggesting to install beautifulsoup4
        return {
            "success": False,
            "error": "BeautifulSoup4 not installed. Install: pip install beautifulsoup4",
            "message": "Web search requires beautifulsoup4. Install it to enable search."
        }
    
    def search_research_papers(self, query: str, source: str = "arxiv", max_results: int = WEB_SEARCH_DEFAULT_RESULTS) -> Dict[str, Any]:
        """
        Search for research papers from arXiv
        
        Args:
            query: Search query
            source: Source to search ('arxiv')
            max_results: Maximum number of results
        
        Returns:
            Dictionary with paper results
        """
        if source.lower() == "arxiv":
            return self._search_arxiv(query, max_results)
        else:
            return {
                "success": False,
                "error": f"Unknown source: {source}. Only 'arxiv' is supported"
            }
    
    def _search_arxiv(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search arXiv for papers"""
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance"
            }
            response = self.session.get(url, params=params, timeout=WEB_BROWSE_TIMEOUT)
            response.raise_for_status()
            
            # Parse Atom XML response
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.text)
            
            # Namespace for Atom
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                link = entry.find('atom:id', ns)
                authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
                
                papers.append({
                    "title": title.text if title is not None else "",
                    "authors": authors,
                    "summary": summary.text if summary is not None else "",
                    "url": link.text if link is not None else "",
                    "pdf_url": link.text.replace('/abs/', '/pdf/') + '.pdf' if link is not None else ""
                })
            
            return {
                "success": True,
                "query": query,
                "source": "arxiv",
                "papers": papers,
                "total": len(papers)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"arXiv search failed: {str(e)}"
            }
    
    def download_file(self, url: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Download a file from URL
        
        Args:
            url: URL to download from
            save_path: Path to save file (optional, auto-generates if not provided)
        
        Returns:
            Dictionary with download result
        """
        try:
            response = self.session.get(url, stream=True, timeout=FILE_DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            
            # Determine save path
            if not save_path:
                # Extract filename from URL or Content-Disposition header
                filename = url.split('/')[-1]
                if '?' in filename:
                    filename = filename.split('?')[0]
                
                # Get from Content-Disposition if available
                content_disp = response.headers.get('Content-Disposition', '')
                if content_disp:
                    match = re.search(r'filename="?([^"]+)"?', content_disp)
                    if match:
                        filename = match.group(1)
                
                save_path = Path.cwd() / filename
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            return {
                "success": True,
                "url": url,
                "filepath": str(save_path),
                "size": downloaded,
                "message": f"Downloaded {save_path.name} ({downloaded} bytes)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Download failed: {str(e)}"
            }
    
    def browse_web(self, url: str) -> Dict[str, Any]:
        """
        Browse a web page and extract content
        
        Args:
            url: URL to browse
        
        Returns:
            Dictionary with page content
        """
        # Check if this is a file:// URL - these should use read_file action instead
        if url.startswith("file://") or (not url.startswith(("http://", "https://")) and Path(url).exists()):
            file_path = url.replace("file://", "") if url.startswith("file://") else url
            return {
                "success": False,
                "error": "file:// URLs are not supported for web browsing",
                "message": f"Cannot browse file:// URLs. Use 'read_file' action instead to read: {file_path}\n\nExample: 'read file {file_path}' or 'open {file_path}'"
            }
        
        try:
            response = self.session.get(url, timeout=WEB_BROWSE_TIMEOUT)
            response.raise_for_status()
            
            # Try to parse HTML
            try:
                from bs4 import BeautifulSoup  # type: ignore[import-untyped]
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract main content
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Get title
                title = soup.find('title')
                title_text = title.get_text() if title else ""
                
                # Get links
                links = [a.get('href', '') for a in soup.find_all('a', href=True)]
                
                return {
                    "success": True,
                    "url": url,
                    "title": title_text,
                    "content": text[:5000],  # Limit content length
                    "links": links[:20],  # Limit links
                    "full_content_available": len(text) > 5000
                }
            except ImportError:
                # If BeautifulSoup not available, return raw text
                return {
                    "success": True,
                    "url": url,
                    "content": response.text[:5000],
                    "message": "BeautifulSoup4 not installed. Install for better HTML parsing."
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to browse {url}: {str(e)}"
            }

