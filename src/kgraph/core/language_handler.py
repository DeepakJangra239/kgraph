"""
Base classes for language-specific code extraction handlers.

This module provides the abstract base class and data structures for
language handlers that extract imports, definitions, and references
using tree-sitter queries.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Import:
    """Represents an import statement."""
    symbol: str          # Imported symbol name
    module: str          # Source module/package
    file_path: str       # File containing the import
    line: int            # Line number
    full_import: str = ""  # Full import string (e.g., com.package.Class)


@dataclass
class Definition:
    """Represents a function/class/method definition."""
    name: str
    kind: str           # "FUNCTION", "CLASS", "METHOD", etc.
    file_path: str
    start_line: int
    end_line: int
    code: str
    docstring: str = ""
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Reference:
    """Represents a call or usage of a symbol."""
    name: str
    kind: str           # "CALL", "INSTANTIATION", "JSX_COMPONENT", etc.
    file_path: str
    line: int
    source_id: str      # ID of the calling function/class
    module_hint: str = ""  # Hint from import_map for resolution


# ============================================================================
# Base Handler
# ============================================================================

class LanguageHandler(ABC):
    """
    Abstract base class for language-specific extraction handlers.
    
    Each language (Python, Java, TypeScript, etc.) implements this interface
    to provide language-specific extraction logic using tree-sitter queries.
    """
    
    @abstractmethod
    def get_language(self):
        """Return the tree-sitter Language object."""
        pass
    
    @abstractmethod
    def extract_imports(self, tree, code: str, file_path: str) -> List[Import]:
        """Extract import statements from the syntax tree."""
        pass
    
    @abstractmethod
    def extract_definitions(self, tree, code: str, file_path: str, parent_id: str) -> List[Definition]:
        """Extract function/class definitions from the syntax tree."""
        pass
    
    @abstractmethod
    def extract_references(self, tree, code: str, file_path: str, source_id: str, import_map: Dict[str, str]) -> List[Reference]:
        """Extract calls and usages from the syntax tree."""
        pass
    
    def get_import_map(self, imports: List[Import]) -> Dict[str, str]:
        """Build a mapping from symbol name to module for resolution."""
        return {imp.symbol: imp.module for imp in imports}
