"""
Language handler registry for routing files to appropriate handlers.

This module provides the LanguageRegistry class that manages language
handlers and routes files to the correct handler based on file extension.
"""

from typing import Dict, Optional, List
from pathlib import Path
from .universal_handler import UniversalHandler

class LanguageRegistry:
    """Registry that loads and manages all language configs via UniversalHandler."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize registry by loading all YAML configs.
        
        Args:
            config_dir: Directory containing language YAML files
                       (defaults to src/kgraph/languages/)
        """
        if config_dir is None:
            # Default to languages directory relative to this file
            # src/kgraph/core/language_registry.py -> src/kgraph/languages
            config_dir = Path(__file__).parent.parent / "languages"
        
        self.config_dir = Path(config_dir)
        self.handlers: Dict[str, UniversalHandler] = {}
        self.extension_map: Dict[str, str] = {}
        self.filename_map: Dict[str, str] = {}
        
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all YAML configs from the languages directory."""
        if not self.config_dir.exists():
            print(f"Warning: Config directory not found: {self.config_dir}")
            return
        
        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                handler = UniversalHandler(str(yaml_file))
                lang_name = handler.language_name
                self.handlers[lang_name] = handler
                
                # Map code extensions
                if handler.code_config:
                    for ext in handler.code_config.get('extensions', []):
                        self.extension_map[ext] = lang_name
                
                # Map config files
                for cfg in handler.file_configs:
                    for pattern in cfg.get('files', []):
                        if pattern.startswith('.'):
                            self.extension_map[pattern] = lang_name
                        else:
                            self.filename_map[pattern] = lang_name
                            
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")
    
    def get_handler(self, file_path: str) -> Optional[UniversalHandler]:
        """Get handler for a file based on its extension or filename."""
        path = Path(file_path)
        
        # Check exact filename matches (e.g., Dockerfile, package.json)
        if path.name in self.filename_map:
            return self.handlers.get(self.filename_map[path.name])
        
        # Check extension
        ext = path.suffix
        if ext in self.extension_map:
            return self.handlers.get(self.extension_map[ext])
        
        return None
    
    def list_supported_languages(self) -> List[str]:
        """Return list of all supported languages."""
        return list(self.handlers.keys())
    
    def supports(self, file_path: str) -> bool:
        """Check if a file is supported."""
        return self.get_handler(file_path) is not None
