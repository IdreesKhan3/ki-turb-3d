"""
Resolver - Cursor-style file finding with multi-signal retrieval
Index-driven, not LLM-first guessing
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import difflib


class Resolver:
    """Cursor-style file resolver with symbol indexing and multi-signal ranking"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.skip_dirs = {'.git', '__pycache__', 'myenv', 'node_modules', '.venv', 'venv', 'local_tools', '.streamlit', '__pycache__'}
        
        # Indexes (built on demand, cached)
        self._symbol_index: Optional[Dict[str, List[str]]] = None  # symbol_name -> [file_paths]
        self._file_index: Optional[List[str]] = None  # all file paths
        self._content_cache: Dict[str, str] = {}  # file_path -> content (for fast search)
        
    def _build_indexes(self):
        """Build symbol and file indexes (lazy, cached)"""
        if self._symbol_index is not None:
            return  # Already built
        
        self._symbol_index = defaultdict(list)
        self._file_index = []
        
        # Patterns for symbol extraction
        class_pattern = re.compile(r'^class\s+(\w+)', re.MULTILINE)
        def_pattern = re.compile(r'^def\s+(\w+)', re.MULTILINE)
        import_pattern = re.compile(r'^from\s+[\w.]+\s+import\s+(\w+)', re.MULTILINE)
        
        for path in self.project_root.rglob("*"):
            if path.is_file() and not any(skip in path.parts for skip in self.skip_dirs):
                if path.name.startswith('.'):
                    continue
                
                rel_path = str(path.relative_to(self.project_root))
                self._file_index.append(rel_path)
                
                # Extract symbols based on file type
                ext = path.suffix.lower()
                if ext in ['.py', '.f90', '.f', '.f95']:
                    try:
                        content = path.read_text(encoding='utf-8', errors='ignore')
                        self._content_cache[rel_path] = content
                        
                        # Extract classes
                        for match in class_pattern.finditer(content):
                            class_name = match.group(1)
                            self._symbol_index[class_name].append(rel_path)
                        
                        # Extract functions
                        for match in def_pattern.finditer(content):
                            func_name = match.group(1)
                            self._symbol_index[func_name].append(rel_path)
                        
                        # Extract imports (what this file imports)
                        for match in import_pattern.finditer(content):
                            import_name = match.group(1)
                            self._symbol_index[import_name].append(rel_path)
                    except Exception:
                        pass
    
    def _text_search(self, query: str, max_results: int = 50) -> List[Tuple[str, int]]:
        """Fast text search across all files (ripgrep-style)"""
        self._build_indexes()
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for rel_path in self._file_index:
            if rel_path in self._content_cache:
                content = self._content_cache[rel_path]
                content_lower = content.lower()
                
                # Count matches (word-level for better ranking)
                matches = sum(1 for word in query_words if word in content_lower)
                if matches > 0:
                    results.append((rel_path, matches))
        
        # Sort by match count (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def _symbol_search(self, symbol_name: str) -> List[str]:
        """Search for symbol (class/function) definition"""
        self._build_indexes()
        symbol_lower = symbol_name.lower()
        
        # Exact match
        if symbol_name in self._symbol_index:
            return self._symbol_index[symbol_name]
        
        # Fuzzy match on symbol names
        all_symbols = list(self._symbol_index.keys())
        close_matches = difflib.get_close_matches(symbol_name, all_symbols, n=5, cutoff=0.6)
        
        results = []
        for match in close_matches:
            results.extend(self._symbol_index[match])
        
        return list(set(results))  # Remove duplicates
    
    def _rank_candidates(self, hint: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rank candidates using multiple signals:
        - Filename similarity
        - Symbol hits
        - Content hits
        - Path heuristics (prefer shorter, common paths)
        """
        self._build_indexes()
        hint_lower = hint.lower()
        hint_words = hint_lower.split()
        hint_parts = Path(hint).parts if '/' in hint or '\\' in hint else [hint]
        
        ranked = []
        
        for candidate in candidates:
            score = 0.0
            candidate_path = Path(candidate)
            candidate_name = candidate_path.name.lower()
            candidate_stem = candidate_path.stem.lower()
            candidate_parts = candidate_path.parts
            
            # Signal 1: Filename similarity (highest weight)
            filename_sim = difflib.SequenceMatcher(None, hint_lower, candidate_name).ratio()
            stem_sim = difflib.SequenceMatcher(None, hint_lower, candidate_stem).ratio()
            score += max(filename_sim, stem_sim) * 3.0
            
            # Signal 2: Symbol hits
            symbol_matches = 0
            for word in hint_words:
                if word in self._symbol_index:
                    if candidate in self._symbol_index[word]:
                        symbol_matches += 1
            score += symbol_matches * 2.0
            
            # Signal 3: Content hits
            if candidate in self._content_cache:
                content_lower = self._content_cache[candidate].lower()
                content_hits = sum(1 for word in hint_words if word in content_lower)
                score += min(content_hits * 0.5, 2.0)  # Cap at 2.0
            
            # Signal 4: Path heuristics
            # Prefer shorter paths (closer to root)
            depth_penalty = len(candidate_parts) * 0.1
            score -= depth_penalty
            
            # Prefer common directories (utils, pages, etc.)
            common_dirs = {'utils', 'pages', 'src', 'scripts'}
            if any(part.lower() in common_dirs for part in candidate_parts):
                score += 0.5
            
            # Signal 5: Path part matches
            hint_part_matches = sum(1 for part in hint_parts if any(part.lower() in p.lower() for p in candidate_parts))
            score += hint_part_matches * 0.3
            
            ranked.append((candidate, score))
        
        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def resolve_hint(self, hint: str, must_be_dir: bool = False, max_candidates: int = 10) -> List[Tuple[str, float, str]]:
        """
        Resolve a hint to file paths using multi-signal retrieval.
        
        Returns:
            List of (file_path, confidence_score, reason) tuples, sorted by confidence
        """
        hint = hint.strip()
        if not hint:
            return []
        
        # Fix C: Normalize leading "/" to be repo-relative, not OS-absolute
        # If path starts with "/" and doesn't exist as absolute, treat as repo-relative
        if hint.startswith("/") and len(hint) > 1:
            test_absolute = Path(hint)
            if not test_absolute.exists():
                # Strip leading "/" and treat as repo-relative
                hint = hint[1:]
        
        self._build_indexes()
        
        # Strategy 1: Try exact path match first
        exact_path = self.project_root / hint
        if exact_path.exists() and (not must_be_dir or exact_path.is_dir()):
            return [(hint, 10.0, "exact_path_match")]
        
        # Strategy 2: Try relative path
        rel_path = self.project_root / hint
        if rel_path.exists() and (not must_be_dir or rel_path.is_dir()):
            return [(hint, 9.5, "relative_path_match")]
        
        # Strategy 3: Filename match
        if not must_be_dir:
            target_name = Path(hint).name
            filename_candidates = [p for p in self._file_index 
                                 if Path(p).name == target_name]
            if filename_candidates:
                ranked = self._rank_candidates(hint, filename_candidates)
                return [(path, score, "filename_match") for path, score in ranked[:max_candidates]]
        
        # Strategy 4: Symbol search (if hint looks like a symbol)
        # Check if hint matches common symbol patterns
        if re.match(r'^[A-Z][a-zA-Z0-9_]*$', hint) or re.match(r'^[a-z_][a-zA-Z0-9_]*$', hint):
            symbol_results = self._symbol_search(hint)
            if symbol_results:
                ranked = self._rank_candidates(hint, symbol_results)
                return [(path, score, "symbol_match") for path, score in ranked[:max_candidates]]
        
        # Strategy 5: Text search (ripgrep-style)
        text_results = self._text_search(hint, max_results=50)
        if text_results:
            candidates = [path for path, _ in text_results]
            ranked = self._rank_candidates(hint, candidates)
            return [(path, score, "content_match") for path, score in ranked[:max_candidates]]
        
        # Strategy 6: Fuzzy path matching (fallback)
        all_candidates = self._file_index if not must_be_dir else [
            p for p in self._file_index if (self.project_root / p).is_dir()
        ]
        fuzzy_matches = difflib.get_close_matches(hint, all_candidates, n=max_candidates, cutoff=0.5)
        if fuzzy_matches:
            ranked = self._rank_candidates(hint, fuzzy_matches)
            return [(path, score, "fuzzy_match") for path, score in ranked[:max_candidates]]
        
        return []
    
    def resolve_with_fallback(self, hint: str, must_be_dir: bool = False, max_attempts: int = 3) -> Optional[Path]:
        """
        Resolve hint with iterative fallback (try #1, then #2, then #3 if first fails).
        Returns the first valid path found.
        """
        candidates = self.resolve_hint(hint, must_be_dir, max_candidates=max_attempts * 2)
        
        for path_str, score, reason in candidates[:max_attempts]:
            full_path = self.project_root / path_str
            if full_path.exists() and (not must_be_dir or full_path.is_dir()):
                return full_path
        
        return None
    
    def get_resolution_info(self, hint: str, must_be_dir: bool = False) -> Dict:
        """
        Get detailed resolution information (for debugging/display).
        Returns candidates with scores and reasons.
        """
        candidates = self.resolve_hint(hint, must_be_dir, max_candidates=10)
        
        return {
            "hint": hint,
            "candidates": [
                {
                    "path": path,
                    "score": round(score, 2),
                    "reason": reason,
                    "full_path": str(self.project_root / path)
                }
                for path, score, reason in candidates
            ],
            "best_match": candidates[0][0] if candidates else None
        }

