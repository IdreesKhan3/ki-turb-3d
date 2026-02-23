"""
Code Analysis Tools - AST-based symbol navigation
Provides code understanding capabilities
"""

import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass


@dataclass
class SymbolReference:
    """A reference to a symbol in code"""
    file: str
    line: int
    col: int
    context: str
    symbol_type: str  # 'function', 'class', 'variable', 'import'


@dataclass
class SymbolDefinition:
    """A symbol definition"""
    name: str
    file: str
    line: int
    col: int
    symbol_type: str
    signature: Optional[str] = None
    docstring: Optional[str] = None


class PythonSymbolAnalyzer(ast.NodeVisitor):
    """AST visitor to extract symbol information"""
    
    def __init__(self, filepath: str, target_symbol: Optional[str] = None):
        self.filepath = filepath
        self.target_symbol = target_symbol
        self.definitions: List[SymbolDefinition] = []
        self.references: List[SymbolReference] = []
        self.current_class: Optional[str] = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions"""
        full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
        
        # Extract signature
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Record definition
        self.definitions.append(SymbolDefinition(
            name=full_name,
            file=self.filepath,
            line=node.lineno,
            col=node.col_offset,
            symbol_type='function',
            signature=signature,
            docstring=docstring
        ))
        
        # Check if this is the symbol we're looking for
        if self.target_symbol and node.name == self.target_symbol:
            self.references.append(SymbolReference(
                file=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                context=signature,
                symbol_type='definition'
            ))
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions"""
        # Record definition
        bases = [self._get_name(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        
        docstring = ast.get_docstring(node)
        
        self.definitions.append(SymbolDefinition(
            name=node.name,
            file=self.filepath,
            line=node.lineno,
            col=node.col_offset,
            symbol_type='class',
            signature=signature,
            docstring=docstring
        ))
        
        if self.target_symbol and node.name == self.target_symbol:
            self.references.append(SymbolReference(
                file=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                context=signature,
                symbol_type='definition'
            ))
        
        # Visit class body with context
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_Name(self, node: ast.Name):
        """Visit name references"""
        if self.target_symbol and node.id == self.target_symbol:
            # Try to get context (the line of code)
            context = node.id  # Simplified - in real usage we'd need the source
            self.references.append(SymbolReference(
                file=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                context=context,
                symbol_type='reference'
            ))
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import):
        """Visit import statements"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.definitions.append(SymbolDefinition(
                name=name,
                file=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                symbol_type='import',
                signature=f"import {alias.name}"
            ))
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from...import statements"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.definitions.append(SymbolDefinition(
                name=name,
                file=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                symbol_type='import',
                signature=f"from {node.module} import {alias.name}"
            ))
        self.generic_visit(node)
    
    def _get_name(self, node):
        """Extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)


def find_symbol_definitions(workspace_root: str, symbol_name: str, file_pattern: str = "*.py") -> Dict[str, Any]:
    """
    Find all definitions of a symbol (function, class, variable)
    
    Args:
        workspace_root: Root directory to search
        symbol_name: Name of the symbol to find
        file_pattern: File pattern to search (default: *.py)
    
    Returns:
        Dict with definitions list and metadata
    """
    definitions = []
    workspace_path = Path(workspace_root)
    
    # Find all Python files
    python_files = list(workspace_path.rglob(file_pattern))
    
    for filepath in python_files:
        # Skip virtual environments and build directories
        if any(part in filepath.parts for part in ['venv', 'myenv', '__pycache__', '.git', 'node_modules']):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(filepath))
            analyzer = PythonSymbolAnalyzer(str(filepath))
            analyzer.visit(tree)
            
            # Filter for target symbol
            for defn in analyzer.definitions:
                if symbol_name in defn.name or defn.name.endswith(f".{symbol_name}"):
                    definitions.append({
                        'name': defn.name,
                        'file': str(filepath.relative_to(workspace_path)),
                        'line': defn.line,
                        'type': defn.symbol_type,
                        'signature': defn.signature,
                        'docstring': defn.docstring[:200] if defn.docstring else None
                    })
        
        except (SyntaxError, UnicodeDecodeError):
            # Skip files with syntax errors or encoding issues
            continue
    
    return {
        'ok': True,
        'message': f"Found {len(definitions)} definition(s) of '{symbol_name}'",
        'outputs': {
            'count': len(definitions),
            'definitions': definitions
        }
    }


def find_symbol_references(workspace_root: str, symbol_name: str, file_pattern: str = "*.py") -> Dict[str, Any]:
    """
    Find all references to a symbol (where it's used)
    
    Args:
        workspace_root: Root directory to search
        symbol_name: Name of the symbol to find
        file_pattern: File pattern to search
    
    Returns:
        Dict with references list and metadata
    """
    references = []
    workspace_path = Path(workspace_root)
    
    python_files = list(workspace_path.rglob(file_pattern))
    
    for filepath in python_files:
        if any(part in filepath.parts for part in ['venv', 'myenv', '__pycache__', '.git']):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(filepath))
            analyzer = PythonSymbolAnalyzer(str(filepath), target_symbol=symbol_name)
            analyzer.visit(tree)
            
            # Get source lines for context
            lines = source.splitlines()
            
            for ref in analyzer.references:
                if ref.line <= len(lines):
                    context = lines[ref.line - 1].strip()
                    references.append({
                        'file': str(filepath.relative_to(workspace_path)),
                        'line': ref.line,
                        'context': context,
                        'type': ref.symbol_type
                    })
        
        except (SyntaxError, UnicodeDecodeError):
            continue
    
    return {
        'ok': True,
        'message': f"Found {len(references)} reference(s) to '{symbol_name}'",
        'outputs': {
            'count': len(references),
            'references': references
        }
    }


def find_class_implementations(workspace_root: str, base_class: str, file_pattern: str = "*.py") -> Dict[str, Any]:
    """
    Find all classes that inherit from a base class
    
    Args:
        workspace_root: Root directory to search
        base_class: Name of the base class
        file_pattern: File pattern to search
    
    Returns:
        Dict with implementations list
    """
    implementations = []
    workspace_path = Path(workspace_root)
    
    python_files = list(workspace_path.rglob(file_pattern))
    
    for filepath in python_files:
        if any(part in filepath.parts for part in ['venv', 'myenv', '__pycache__', '.git']):
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(filepath))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if any base class matches
                    for base in node.bases:
                        base_name = None
                        if isinstance(base, ast.Name):
                            base_name = base.id
                        elif isinstance(base, ast.Attribute):
                            base_name = base.attr
                        
                        if base_name and base_class in base_name:
                            implementations.append({
                                'class': node.name,
                                'file': str(filepath.relative_to(workspace_path)),
                                'line': node.lineno,
                                'base': base_name
                            })
        
        except (SyntaxError, UnicodeDecodeError):
            continue
    
    return {
        'ok': True,
        'message': f"Found {len(implementations)} class(es) implementing '{base_class}'",
        'outputs': {
            'count': len(implementations),
            'implementations': implementations
        }
    }
