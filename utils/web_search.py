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


class WebSearchTools:
    """Tools for web search, research papers, and file downloads"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def google_search(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Perform Google Search using SerpAPI or direct search
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            Dictionary with search results
        """
        try:
            # Option 1: Use SerpAPI if available (requires API key)
            serp_api_key = os.getenv("SERP_API_KEY")
            if serp_api_key:
                return self._serpapi_search(query, num_results, serp_api_key)
            
            # Option 2: Use DuckDuckGo (free, no API key needed)
            return self._duckduckgo_search(query, num_results)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Search failed: {str(e)}"
            }
    
    def _serpapi_search(self, query: str, num_results: int, api_key: str) -> Dict[str, Any]:
        """Search using SerpAPI (requires API key)"""
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": api_key,
                "num": num_results
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "organic_results" in data:
                for item in data["organic_results"][:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "total": len(results)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _duckduckgo_search(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using DuckDuckGo (free, no API key)"""
        try:
            # Use DuckDuckGo HTML search
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse HTML results (simple extraction)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result')[:num_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "link": title_elem.get('href', ''),
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
    
    def search_research_papers(self, query: str, source: str = "arxiv", max_results: int = 10) -> Dict[str, Any]:
        """
        Search for research papers from various sources
        
        Args:
            query: Search query
            source: Source to search ('arxiv', 'scholar', 'pubmed')
            max_results: Maximum number of results
        
        Returns:
            Dictionary with paper results
        """
        if source.lower() == "arxiv":
            return self._search_arxiv(query, max_results)
        elif source.lower() == "scholar":
            return self._search_google_scholar(query, max_results)
        elif source.lower() == "pubmed":
            return self._search_pubmed(query, max_results)
        else:
            return {
                "success": False,
                "error": f"Unknown source: {source}. Use 'arxiv', 'scholar', or 'pubmed'"
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
            response = self.session.get(url, params=params, timeout=15)
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
    
    def _search_google_scholar(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search Google Scholar (requires SerpAPI or similar)"""
        # Google Scholar requires SerpAPI or scraping
        serp_api_key = os.getenv("SERP_API_KEY")
        if serp_api_key:
            try:
                url = "https://serpapi.com/search"
                params = {
                    "engine": "google_scholar",
                    "q": query,
                    "api_key": serp_api_key,
                    "num": max_results
                }
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                papers = []
                if "organic_results" in data:
                    for item in data["organic_results"]:
                        papers.append({
                            "title": item.get("title", ""),
                            "authors": item.get("publication_info", {}).get("authors", []),
                            "snippet": item.get("snippet", ""),
                            "url": item.get("link", ""),
                            "pdf_url": item.get("resources", [{}])[0].get("link", "") if item.get("resources") else ""
                        })
                
                return {
                    "success": True,
                    "query": query,
                    "source": "google_scholar",
                    "papers": papers,
                    "total": len(papers)
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {
                "success": False,
                "error": "SERP_API_KEY not set",
                "message": "Google Scholar search requires SERP_API_KEY. Set it or use arXiv instead."
            }
    
    def _search_pubmed(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search PubMed for papers"""
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            paper_ids = data.get("esearchresult", {}).get("idlist", [])
            
            if not paper_ids:
                return {
                    "success": True,
                    "query": query,
                    "source": "pubmed",
                    "papers": [],
                    "total": 0
                }
            
            # Get details for each paper
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(paper_ids),
                "retmode": "xml"
            }
            fetch_response = self.session.get(fetch_url, params=fetch_params, timeout=15)
            fetch_response.raise_for_status()
            
            # Parse XML (simplified)
            papers = []
            # For now, return basic info - full parsing would require more XML handling
            for paper_id in paper_ids:
                papers.append({
                    "id": paper_id,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                    "title": f"PubMed ID: {paper_id}"
                })
            
            return {
                "success": True,
                "query": query,
                "source": "pubmed",
                "papers": papers,
                "total": len(papers)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"PubMed search failed: {str(e)}"
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
            response = self.session.get(url, stream=True, timeout=30)
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
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Try to parse HTML
            try:
                from bs4 import BeautifulSoup
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


# Import os for environment variables
import os

