"""
Universal Language Handler - Config-Driven Code Extraction

This is the ultimate generic solution: ONE handler that works for ALL languages
by reading YAML configuration files.

Supported Languages & Configs:
  - Code: Tree-sitter (Python, Java, TS, Go, Rust, C++, C#, Ruby, PHP)
  - Configs: Regex, JSON, XML, YAML, TOML (package.json, pom.xml, Dockerfile, etc.)
"""

import yaml
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import importlib
import xml.etree.ElementTree as ET

# Try importing toml lib
try:
    import tomllib as toml  # Python 3.11+
except ImportError:
    try:
        import tomli as toml  # Backport
    except ImportError:
        toml = None

from tree_sitter import Language, Parser, Query, QueryCursor

from .language_handler import LanguageHandler, Import, Definition, Reference


class UniversalHandler(LanguageHandler):
    """Universal handler that reads language configs from YAML files."""
    
    def __init__(self, config_path: str):
        """
        Initialize from a YAML config file.
        
        Args:
            config_path: Path to the language YAML config
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.language_name = self.config['language']
        
        # Initialize Code Parser (Tree-sitter)
        self.code_config = self.config.get('code')
        self.language = None
        self.parser = None
        
        if self.code_config:
            parser_module = self.code_config.get('parser')
            factory_name = self.code_config.get('parser_factory', 'language')
            
            if parser_module:
                try:
                    module = importlib.import_module(parser_module)
                    factory = getattr(module, factory_name)
                    self.language = Language(factory())
                    self.parser = Parser(self.language)
                except ImportError:
                    print(f"Warning: Could not load parser {parser_module}")
                except AttributeError:
                    print(f"Warning: Could not find factory {factory_name} in {parser_module}")
        
        # Initialize Config File Handlers
        self.file_configs = self.config.get('configs', [])
    
    def get_language(self):
        """Return the tree-sitter Language object."""
        return self.language
    
    def _get_file_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Find matching config definition for a file path."""
        fname = os.path.basename(file_path)
        
        for cfg in self.file_configs:
            for pattern in cfg.get('files', []):
                if pattern == fname or (pattern.startswith('.') and fname.endswith(pattern)):
                    return cfg
        return None

    def extract_imports(self, tree, code: str, file_path: str) -> List[Import]:
        """Extract imports using either tree-sitter or file-specific parsers."""
        # 1. Check if it's a config file
        file_cfg = self._get_file_config(file_path)
        if file_cfg:
            return self._extract_imports_from_config(file_cfg, code, file_path)
        
        # 2. Fallback to Tree-sitter for code
        if self.language and self.code_config:
            return self._extract_imports_tree_sitter(tree, code, file_path)
            
        return []

    def _extract_imports_tree_sitter(self, tree, code: str, file_path: str) -> List[Import]:
        """Extract imports using Tree-sitter queries."""
        imports = []
        queries = self.code_config.get('queries', {})
        import_query = queries.get('imports')
        
        if not import_query:
            return imports
        
        query = Query(self.language, import_query)
        cursor = QueryCursor(query)
        
        extraction = self.code_config.get('extraction', {}).get('imports', {})
        
        for pattern_idx, captures in cursor.matches(tree.root_node):
            if 'symbol_capture' in extraction:
                symbols = captures.get(extraction['symbol_capture'], [])
                
                # Handle source capture (TS style)
                if 'source_capture' in extraction:
                    sources = captures.get(extraction['source_capture'], [])
                    if sources:
                        module_name = code[sources[0].start_byte:sources[0].end_byte]
                        if extraction.get('strip_quotes'):
                            module_name = module_name.strip("'\"")
                        
                        for sym_node in symbols:
                            symbol = code[sym_node.start_byte:sym_node.end_byte]
                            imports.append(Import(symbol, module_name, file_path, sym_node.start_point[0]+1))
                    continue

                # Handle module capture (Python style)
                if 'module_capture' in extraction:
                    modules = captures.get(extraction['module_capture'], [])
                    if modules:
                        module = code[modules[0].start_byte:modules[0].end_byte]
                        
                        if symbols:
                            for sym_node in symbols:
                                symbol = code[sym_node.start_byte:sym_node.end_byte]
                                if symbol != module:
                                    imports.append(Import(symbol, module, file_path, sym_node.start_point[0]+1))
                        else:
                            # Module-only import (import os)
                            imports.append(Import(module, module, file_path, modules[0].start_point[0]+1))
            
            elif 'full_import_capture' in extraction:
                # Java style
                nodes = captures.get(extraction['full_import_capture'], [])
                split_on = extraction.get('split_on', '.')
                
                for node in nodes:
                    full = code[node.start_byte:node.end_byte]
                    parts = full.split(split_on)
                    symbol = parts[-1]
                    module = split_on.join(parts[:-1])
                    imports.append(Import(symbol, module, file_path, node.start_point[0]+1, full))
                    
        return imports

    def _extract_imports_from_config(self, cfg: Dict, code: str, file_path: str) -> List[Import]:
        """Extract imports/dependencies from config files."""
        parser_type = cfg.get('parser')
        imports = []
        
        if parser_type == 'text_patterns':
            patterns = cfg.get('patterns', {})
            for key, pat in patterns.items():
                regex = pat.get('regex')
                kind = pat.get('kind', 'DEPENDENCY')
                if not regex: continue
                
                for i, match in enumerate(re.finditer(regex, code, re.MULTILINE)):
                    symbol = match.group(pat.get('capture_group', 1))
                    imports.append(Import(
                        symbol=symbol,
                        module=kind,
                        file_path=file_path,
                        line=code.count('\n', 0, match.start()) + 1,
                        full_import=match.group(0)
                    ))
                    
        elif parser_type == 'json':
            try:
                data = json.loads(code)
                extraction = cfg.get('extraction', {})
                for key, rules in extraction.items():
                    path = rules.get('path', '').split('.')
                    kind = rules.get('kind', 'DEPENDENCY')
                    
                    # Navigate JSON
                    curr = data
                    for p in path:
                        if isinstance(curr, dict): curr = curr.get(p, {})
                        else: break
                    
                    if rules.get('extract') == 'keys' and isinstance(curr, dict):
                        for dep in curr.keys():
                            imports.append(Import(dep, kind, file_path, 1))
                    elif rules.get('extract') == 'value' and isinstance(curr, (str, int)):
                         imports.append(Import(str(curr), kind, file_path, 1))
            except: pass

        elif parser_type == 'xml':
            try:
                root = ET.fromstring(code)
                extraction = cfg.get('extraction', {})
                for key, rules in extraction.items():
                    xpath = rules.get('xpath')
                    kind = rules.get('kind', 'DEPENDENCY')
                    if not xpath: continue
                    
                    # Simple XPath handling (ElementTree has limited XPath)
                    # Convert //tag to .//tag for ElementTree
                    et_xpath = xpath.replace('//', './/')
                    if et_xpath.startswith('/'): et_xpath = et_xpath[1:] # Remove leading /
                    
                    for elem in root.findall(et_xpath):
                        if elem.text:
                            imports.append(Import(elem.text, kind, file_path, 1))
            except: pass
            
        elif parser_type == 'yaml':
            try:
                data = yaml.safe_load(code)
                # Reuse JSON extraction logic since YAML loads to dict
                extraction = cfg.get('extraction', {})
                for key, rules in extraction.items():
                    path = rules.get('path', '').split('.')
                    kind = rules.get('kind', 'CONFIG')
                    
                    curr = data
                    for p in path:
                        if p == '*': # Wildcard list handling
                            if isinstance(curr, list):
                                # Flatten list? Complex. Skip for now or handle simple case
                                pass 
                            break
                        if isinstance(curr, dict): curr = curr.get(p, {})
                        else: break
                        
                    if isinstance(curr, dict) and rules.get('extract') == 'keys':
                        for k in curr.keys():
                            imports.append(Import(k, kind, file_path, 1))
            except: pass

        return imports

    def extract_definitions(self, tree, code: str, file_path: str, parent_id: str) -> List[Definition]:
        """Extract definitions using Tree-sitter (only for code)."""
        if self._get_file_config(file_path):
            return [] # Config files usually don't have definitions in the graph sense
            
        if not (self.language and self.code_config):
            return []
            
        definitions = []
        queries = self.code_config.get('queries', {})
        def_query = queries.get('definitions')
        if not def_query: return []
        
        query = Query(self.language, def_query)
        cursor = QueryCursor(query)
        extraction = self.code_config.get('extraction', {}).get('definitions', {})
        
        for pattern_idx, captures in cursor.matches(tree.root_node):
            # Check which definition type this match corresponds to
            for def_type, rules in extraction.items():
                name_cap = rules.get('name_capture')
                node_cap = rules.get('node_capture')
                kind = rules.get('kind', def_type.upper())
                
                # Check if this match has the captures for this definition type
                if name_cap in captures and node_cap in captures:
                    names = captures[name_cap]
                    nodes = captures[node_cap]
                    
                    if names and nodes:
                        name = code[names[0].start_byte:names[0].end_byte]
                        node = nodes[0]
                        definitions.append(Definition(
                            name, kind, file_path, 
                            node.start_point[0]+1, node.end_point[0]+1,
                            code[node.start_byte:node.end_byte]
                        ))
                        # Break after finding a match to avoid duplicates
                        break
                    
        return definitions

    def extract_references(self, tree, code: str, file_path: str, source_id: str, import_map: Dict[str, str]) -> List[Reference]:
        """Extract references using Tree-sitter (only for code)."""
        if self._get_file_config(file_path):
            return []
            
        if not (self.language and self.code_config):
            return []
            
        references = []
        queries = self.code_config.get('queries', {})
        call_query = queries.get('calls')
        if not call_query: return []
        
        query = Query(self.language, call_query)
        cursor = QueryCursor(query)
        extraction = self.code_config.get('extraction', {}).get('calls', {})
        
        for pattern_idx, captures in cursor.matches(tree.root_node):
            for call_type, rules in extraction.items():
                # Handle nested rules (like in Java/TS)
                if isinstance(rules, dict):
                    name_cap = rules.get('name_capture')
                    kind = rules.get('kind', 'CALL')
                    nodes = captures.get(name_cap, [])
                    
                    for node in nodes:
                        name = code[node.start_byte:node.end_byte]
                        if rules.get('strip_generics') and '<' in name:
                            name = name.split('<')[0]
                        if rules.get('filter_uppercase') and not name[0].isupper():
                            continue
                            
                        references.append(Reference(
                            name, kind, file_path, node.start_point[0]+1, source_id,
                            import_map.get(name, "")
                        ))
                else:
                    # Simple case (Python)
                    pass # Python logic handled differently in previous impl, need to align
                    
            # Python simple case handling (if extraction is flat)
            if 'name_capture' in extraction:
                 nodes = captures.get(extraction['name_capture'], [])
                 for node in nodes:
                     name = code[node.start_byte:node.end_byte]
                     references.append(Reference(
                         name, extraction.get('kind', 'CALL'), file_path, 
                         node.start_point[0]+1, source_id, import_map.get(name, "")
                     ))

        return references


class LanguageConfigRegistry:
    """Registry that loads and manages all language configs."""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "languages"
        
        self.config_dir = Path(config_dir)
        self.handlers: Dict[str, UniversalHandler] = {}
        self.extension_map: Dict[str, str] = {}
        self.filename_map: Dict[str, str] = {}
        
        self._load_all_configs()
    
    def _load_all_configs(self):
        if not self.config_dir.exists(): return
        
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
        path = Path(file_path)
        if path.name in self.filename_map:
            return self.handlers.get(self.filename_map[path.name])
        if path.suffix in self.extension_map:
            return self.handlers.get(self.extension_map[path.suffix])
        return None

    def list_supported_languages(self) -> List[str]:
        """Return list of all supported languages."""
        return list(self.handlers.keys())
