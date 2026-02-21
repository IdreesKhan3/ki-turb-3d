"""
Semantic Code Search
Natural language code search using embeddings (lightweight implementation)
Uses TF-IDF for similarity instead of heavy embedding models to keep it fast
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import math


class SimpleSemanticSearch:
    """
    Lightweight semantic search using TF-IDF
    No external dependencies - pure Python implementation
    """
    
    def __init__(self):
        self.documents = []
        self.doc_metadata = []
        self.vocabulary = set()
        self.idf_scores = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and extract words
        text = text.lower()
        # Split on non-alphanumeric
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        tf = Counter(tokens)
        total = len(tokens)
        return {term: count / total for term, count in tf.items()}
    
    def _compute_idf(self):
        """Compute inverse document frequency"""
        num_docs = len(self.documents)
        doc_freq = Counter()
        
        for doc_tokens in self.documents:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        self.idf_scores = {
            term: math.log(num_docs / (freq + 1))
            for term, freq in doc_freq.items()
        }
    
    def _compute_tfidf(self, tf: Dict[str, float]) -> Dict[str, float]:
        """Compute TF-IDF scores"""
        return {
            term: tf_score * self.idf_scores.get(term, 0)
            for term, tf_score in tf.items()
        }
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two TF-IDF vectors"""
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # Compute dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(score ** 2 for score in vec1.values()))
        mag2 = math.sqrt(sum(score ** 2 for score in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def index_files(self, workspace_root: str, file_pattern: str = "*.py"):
        """Index files for semantic search"""
        self.documents = []
        self.doc_metadata = []
        workspace_path = Path(workspace_root)
        
        files = list(workspace_path.rglob(file_pattern))
        
        for filepath in files:
            if any(part in filepath.parts for part in ['venv', 'myenv', '__pycache__', '.git']):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tokens = self._tokenize(content)
                self.documents.append(tokens)
                self.doc_metadata.append({
                    'file': str(filepath.relative_to(workspace_path)),
                    'content': content
                })
                self.vocabulary.update(tokens)
            
            except (UnicodeDecodeError, PermissionError):
                continue
        
        # Compute IDF scores
        self._compute_idf()
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code files matching the query
        
        Args:
            query: Natural language query
            top_k: Number of results to return
        
        Returns:
            List of matching files with scores
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)
        query_tfidf = self._compute_tfidf(query_tf)
        
        # Compute similarity for each document
        results = []
        for i, doc_tokens in enumerate(self.documents):
            doc_tf = self._compute_tf(doc_tokens)
            doc_tfidf = self._compute_tfidf(doc_tf)
            
            similarity = self._cosine_similarity(query_tfidf, doc_tfidf)
            
            if similarity > 0:
                results.append({
                    'file': self.doc_metadata[i]['file'],
                    'score': similarity,
                    'preview': self._extract_relevant_snippet(
                        self.doc_metadata[i]['content'],
                        query_tokens
                    )
                })
        
        # Sort by similarity and return top K
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _extract_relevant_snippet(self, content: str, query_tokens: List[str], max_lines: int = 5) -> str:
        """Extract snippet most relevant to query"""
        lines = content.splitlines()
        line_scores = []
        
        for line_num, line in enumerate(lines):
            line_tokens = set(self._tokenize(line))
            # Score based on number of query tokens present
            score = sum(1 for qt in query_tokens if qt in line_tokens)
            line_scores.append((line_num, score))
        
        # Sort by score and get top lines
        line_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_lines = sorted([ln for ln, score in line_scores[:max_lines] if score > 0])
        
        if not relevant_lines:
            return lines[0] if lines else ""
        
        # Return snippet around most relevant lines
        start = max(0, relevant_lines[0] - 1)
        end = min(len(lines), relevant_lines[-1] + 2)
        
        return '\n'.join(lines[start:end])


def semantic_search(
    workspace_root: str,
    query: str,
    top_k: int = 10,
    file_pattern: str = "*.py"
) -> Dict[str, Any]:
    """
    Semantic code search - find code matching natural language query
    
    Args:
        workspace_root: Root directory to search
        query: Natural language query (e.g., "authentication logic", "database connection")
        top_k: Number of results to return
        file_pattern: File pattern to search
    
    Returns:
        Dict with search results
    """
    try:
        searcher = SimpleSemanticSearch()
        
        # Index files
        searcher.index_files(workspace_root, file_pattern)
        
        if not searcher.documents:
            return {
                'ok': False,
                'message': f"No files found matching pattern '{file_pattern}'",
                'outputs': {}
            }
        
        # Perform search
        results = searcher.search(query, top_k)
        
        return {
            'ok': True,
            'message': f"Found {len(results)} relevant file(s) for query '{query}'",
            'outputs': {
                'count': len(results),
                'query': query,
                'results': results
            }
        }
    
    except Exception as e:
        return {
            'ok': False,
            'message': f"Semantic search error: {str(e)}",
            'outputs': {}
        }
