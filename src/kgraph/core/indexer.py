import os
import glob
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Iterator, Set

from .store import GraphStore
from .language_registry import LanguageRegistry

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    "node_modules", "bower_components", "jspm_packages", # JavaScript
    "venv", ".venv", "env", ".env", "__pycache__", ".tox", ".pytest_cache", "site-packages", # Python
    "target", "build", ".gradle", ".m2", # Java/Kotlin
    "vendor", # Go/PHP/Ruby
    "bin", "obj", # C#
    ".git", ".svn", ".hg", ".idea", ".vscode", ".settings", # VCS/IDE
    "dist", "out", "coverage", ".next", ".nuxt", # Build/Output
    "gems", ".bundle", # Ruby
    ".opencode" # opencode
}

class CodeIndexer:
    def __init__(self, store: GraphStore):
        self.store = store
        self.registry = LanguageRegistry()
        logger.info(f"Initialized CodeIndexer with languages: {self.registry.list_supported_languages()}")

    def stream_files(self, root_path: str) -> Iterator[str]:
        """Yields absolute file paths to index."""
        root_path = os.path.abspath(root_path)
        # Walk directory tree
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Filter directories
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if self.registry.supports(file_path):
                    yield file_path

    def process_file_stream(self, file_paths: Iterator[str]) -> Iterator[Tuple[List[Dict], List[Dict]]]:
        """
        Parallel processes files and yields (nodes, edges) results as they complete.
        This acts as a transformation stream.
        """
        with ThreadPoolExecutor() as executor:
            # We map futures to file paths for error reporting
            future_to_file = {}
            
            # Submit all tasks
            # Note: For truly massive codebases, we might want to bound this submission 
            # to avoid too many pending futures, but for <100k files it's fine.
            for file_path in file_paths:
                future = executor.submit(self._process_file, file_path)
                future_to_file[future] = file_path
            
            # Yield results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    yield future.result()
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")

    def index_codebase(self, root_path: str):
        """
        Main entry point. Orchestrates the streaming pipeline with a Producer-Consumer Queue
        to ensure Parsing (CPU) and Embedding (ANE) happen in parallel.
        """
        files = list(self.stream_files(root_path))
        total_files = len(files)
        logger.info(f"Indexing {total_files} files...")
        
        start_time = time.time()
        
        # Producer-Consumer pattern
        import queue
        import threading
        
        # Queue to buffer parsed nodes before embedding
        node_queue = queue.Queue(maxsize=200) 
        stop_event = threading.Event()
        
        # Consumer: Background thread for Embedding & DB Write
        def worker():
            batch_nodes = []
            batch_edges = []
            BATCH_SIZE = 256
            
            while not stop_event.is_set() or not node_queue.empty():
                try:
                    item = node_queue.get(timeout=0.1)
                    nodes, edges = item
                    batch_nodes.extend(nodes)
                    batch_edges.extend(edges)
                    
                    if len(batch_nodes) >= BATCH_SIZE:
                        self.store.add_nodes_batch(batch_nodes)
                        for edge in batch_edges:
                            self.store.add_edge(**edge)
                        batch_nodes = []
                        batch_edges = []
                    
                    node_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in worker: {e}")
            
            # Flush remaining
            if batch_nodes or batch_edges:
                self.store.add_nodes_batch(batch_nodes)
                for edge in batch_edges:
                    self.store.add_edge(**edge)

        # Start consumer thread
        consumer_thread = threading.Thread(target=worker, daemon=True)
        consumer_thread.start()
        
        # Producer: Main thread iterates stream and pushes to queue
        completed = 0
        last_log_time = start_time
        last_log_completed = 0
        
        for result in self.process_file_stream(files):
            # Block if queue is full (backpressure)
            node_queue.put(result)
            completed += 1
            
            # Log progress
            if completed % 100 == 0 or completed == total_files:
                current_time = time.time()
                elapsed_total = current_time - start_time
                elapsed_interval = current_time - last_log_time
                
                files_in_interval = completed - last_log_completed
                current_rate = files_in_interval / elapsed_interval if elapsed_interval > 0 else 0
                avg_rate = completed / elapsed_total if elapsed_total > 0 else 0
                
                eta = (total_files - completed) / avg_rate if avg_rate > 0 else 0
                print(f"Progress: {completed}/{total_files} files ({completed*100//total_files}%) | "
                      f"Rate: {current_rate:.1f} files/s (Avg: {avg_rate:.1f}) | ETA: {eta:.0f}s")
                
                last_log_time = current_time
                last_log_completed = completed
        
        # Signal consumer to stop and wait for it
        stop_event.set()
        consumer_thread.join()
        
        # Post-processing: Resolve UNKNOWN edges (Imports/Inheritance)
        print("Linking graph edges...")
        self.store.resolve_unknown_edges()

    def _process_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse file and return nodes/edges without writing to DB."""
        handler = self.registry.get_handler(file_path)
        if not handler:
            return [], []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return [], []

        # Smart Indexing: Check if file has changed
        import hashlib
        file_hash = hashlib.md5(code.encode('utf-8')).hexdigest()
        file_id = f"file:{file_path}"
        
        try:
            existing_node = self.store.get_node(file_id)
            if existing_node:
                props = existing_node.get('properties', {})
                if props.get('hash') == file_hash:
                    return [], []
        except Exception as e:
            logger.warning(f"Error checking hash for {file_path}: {e}")

        nodes = []
        edges = []

        # Create File Node
        nodes.append({
            "node_id": file_id,
            "node_type": "FILE",
            "name": os.path.basename(file_path),
            "file_path": file_path,
            "start_line": 1,
            "end_line": len(code.splitlines()) + 1,
            "properties": {"hash": file_hash},
            "code_content": code
        })

        try:
            # Parse Code if applicable
            tree = None
            if handler.parser:
                tree = handler.parser.parse(bytes(code, "utf8"))
            
            # Extract Imports
            imports = handler.extract_imports(tree, code, file_path)
            import_map = {}
            for imp in imports:
                import_id = f"IMPORT:{file_path}:{imp.symbol}:{imp.line}"
                import_map[imp.symbol] = imp.module
                
                nodes.append({
                    "node_id": import_id,
                    "node_type": "IMPORT",
                    "name": imp.symbol,
                    "file_path": file_path,
                    "start_line": imp.line,
                    "end_line": imp.line, # Approximate
                    "properties": {"from_module": imp.module, "full_import": imp.full_import},
                    "code_content": imp.full_import or f"import {imp.symbol}"
                })
                edges.append({
                    "source_id": file_id,
                    "target_id": import_id,
                    "edge_type": "CONTAINS"
                })
                edges.append({
                    "source_id": import_id,
                    "target_id": f"UNKNOWN:{imp.symbol}",
                    "edge_type": "IMPORTS"
                })

            # Extract Definitions
            definitions = handler.extract_definitions(tree, code, file_path, file_id)
            for definition in definitions:
                def_id = f"{definition.kind}:{file_path}:{definition.name}"
                nodes.append({
                    "node_id": def_id,
                    "node_type": definition.kind,
                    "name": definition.name,
                    "file_path": file_path,
                    "start_line": definition.start_line,
                    "end_line": definition.end_line,
                    "properties": {"docstring": definition.docstring},
                    "code_content": definition.code
                })
                edges.append({
                    "source_id": file_id, # Simplified: Flattened hierarchy for now (TODO: Nested)
                    "target_id": def_id,
                    "edge_type": "DEFINES"
                })
                
                # TODO: Handle inheritance edges if handler supports it (not yet in UniversalHandler base)
            
            # Extract References
            references = handler.extract_references(tree, code, file_path, file_id, import_map)
            for ref in references:
                # Find parent definition for this reference based on line number
                parent_id = file_id
                for definition in definitions:
                    if definition.start_line <= ref.line <= definition.end_line:
                        # Use the most specific (smallest range) definition
                        parent_id = f"{definition.kind}:{file_path}:{definition.name}"
                
                props = {}
                if ref.module_hint:
                    props['module_hint'] = ref.module_hint
                    
                edges.append({
                    "source_id": parent_id,
                    "target_id": f"UNKNOWN:{ref.name}",
                    "edge_type": "CALLS",
                    "properties": props
                })

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            
        return nodes, edges

    def validate_syntax(self, code: str, file_path: str) -> List[str]:
        """
        Checks for syntax errors in the provided code.
        Returns a list of error messages.
        """
        handler = self.registry.get_handler(file_path)
        if not handler or not handler.parser:
            return [] # Cannot validate without parser
            
        tree = handler.parser.parse(bytes(code, "utf8"))
        
        errors = []
        # Traverse tree to find ERROR or MISSING nodes
        cursor = tree.walk()
        
        visited_children = False
        while True:
            if not visited_children:
                if cursor.node.type == "ERROR":
                    start = cursor.node.start_point
                    snippet = code[cursor.node.start_byte:cursor.node.end_byte]
                    errors.append(f"Syntax Error at line {start[0]+1}, col {start[1]}: Unexpected token '{snippet}'")
                elif cursor.node.is_missing:
                    start = cursor.node.start_point
                    errors.append(f"Syntax Error at line {start[0]+1}, col {start[1]}: Missing expected token")
            
            if not visited_children and cursor.goto_first_child():
                visited_children = False
            elif cursor.goto_next_sibling():
                visited_children = False
            elif cursor.goto_parent():
                visited_children = True
            else:
                break
        
        return errors

    def find_undefined_calls(self, code: str, file_path: str) -> set:
        """
        Finds function/method calls in the code that might be undefined.
        Returns a set of called function names.
        """
        handler = self.registry.get_handler(file_path)
        if not handler or not handler.parser:
            return set()
        
        tree = handler.parser.parse(bytes(code, "utf8"))
        
        # Use handler's extract_references to find calls
        # We pass dummy IDs since we only care about names
        references = handler.extract_references(tree, code, file_path, "dummy", {})
        
        return {ref.name for ref in references}

    def extract_definitions_from_text(self, code: str, file_path: str) -> Dict[str, str]:
        """
        Parses code and returns a dict of defined symbol names to their signatures.
        Used for impact analysis.
        """
        handler = self.registry.get_handler(file_path)
        if not handler or not handler.parser:
            return {}
            
        tree = handler.parser.parse(bytes(code, "utf8"))
        
        # Use handler's extract_definitions
        definitions = handler.extract_definitions(tree, code, file_path, "dummy")
        
        # Map name -> code (signature approximation)
        return {d.name: d.code for d in definitions}

    def extract_imports(self, code: str, file_path: str) -> Set[str]:
        """
        Parses code and returns a set of imported symbol names.
        """
        handler = self.registry.get_handler(file_path)
        if not handler:
            return set()
            
        # Use handler's extract_imports
        tree = None
        if handler.parser:
            tree = handler.parser.parse(bytes(code, "utf8"))
            
        imports = handler.extract_imports(tree, code, file_path)
        return {imp.symbol for imp in imports}
