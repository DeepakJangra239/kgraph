import os
import glob
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Iterator
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_java
from tree_sitter import Language, Parser

from .store import GraphStore

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
        self.parsers = {}
        self._init_parsers()

    def _init_parsers(self):
        try:
            PY_LANGUAGE = Language(tree_sitter_python.language())
            JS_LANGUAGE = Language(tree_sitter_javascript.language())
            JAVA_LANGUAGE = Language(tree_sitter_java.language())
            
            import tree_sitter_go
            GO_LANGUAGE = Language(tree_sitter_go.language())
            
            import tree_sitter_typescript
            TS_LANGUAGE = Language(tree_sitter_typescript.language_typescript())
            TSX_LANGUAGE = Language(tree_sitter_typescript.language_tsx())
            
            self.parsers[".py"] = Parser(PY_LANGUAGE)
            self.parsers[".js"] = Parser(JS_LANGUAGE)
            self.parsers[".jsx"] = Parser(JS_LANGUAGE)
            self.parsers[".ts"] = Parser(TS_LANGUAGE)
            self.parsers[".tsx"] = Parser(TSX_LANGUAGE)
            self.parsers[".java"] = Parser(JAVA_LANGUAGE)
            self.parsers[".go"] = Parser(GO_LANGUAGE)
        except Exception as e:
            logger.error(f"Error initializing parsers: {e}")

    def stream_files(self, root_path: str) -> Iterator[str]:
        """Yields absolute file paths to index."""
        root_path = os.path.abspath(root_path)
        for ext in self.parsers.keys():
            pattern = os.path.join(root_path, "**", f"*{ext}")
            for file_path in glob.glob(pattern, recursive=True):
                # Check if any part of the path is in IGNORED_DIRS
                parts = file_path.split(os.sep)
                if any(part in IGNORED_DIRS for part in parts):
                    continue
                yield os.path.abspath(file_path)

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

    def _process_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Parse file and return nodes/edges without writing to DB."""
        ext = os.path.splitext(file_path)[1]
        if ext not in self.parsers:
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
        
        # We need to access store to check hash. 
        # Since we are in a thread, store._get_conn() will create a new connection for this thread.
        # This is safe because we are only reading.
        try:
            existing_node = self.store.get_node(file_id)
            if existing_node:
                props = existing_node.get('properties', {})
                if props.get('hash') == file_hash:
                    # File unchanged, skip parsing and embedding!
                    return [], []
        except Exception as e:
            # If DB lookup fails, just proceed with indexing
            logger.warning(f"Error checking hash for {file_path}: {e}")

        nodes = []
        edges = []

        # Create File Node with Hash
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
            parser = self.parsers[ext]
            tree = parser.parse(bytes(code, "utf8"))
            
            # Extract Definitions
            def_nodes, def_edges = self._extract_definitions_data(tree.root_node, file_id, file_path, code)
            nodes.extend(def_nodes)
            edges.extend(def_edges)
            
            # Extract References
            ref_edges = self._extract_references_data(tree.root_node, file_id, file_path, code)
            edges.extend(ref_edges)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            
        return nodes, edges

    def _extract_definitions_data(self, node, parent_id: str, file_path: str, code: str) -> Tuple[List[Dict], List[Dict]]:
        nodes = []
        edges = []
        
        # Imports (Python, JS, Java)
        if node.type in ["import_statement", "import_from_statement", "import_declaration"]:
            import_text = code[node.start_byte:node.end_byte]
            node_id = f"IMPORT:{file_path}:{node.start_point[0]}"
            
            nodes.append({
                "node_id": node_id,
                "node_type": "IMPORT",
                "name": import_text.split('\n')[0],
                "file_path": file_path,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "code_content": import_text
            })
            edges.append({
                "source_id": parent_id,
                "target_id": node_id,
                "edge_type": "IMPORTS"
            })

        # Python definitions
        if node.type in ["function_definition", "class_definition"]:
            name_node = node.child_by_field_name("name")
            if name_node:
                name = code[name_node.start_byte:name_node.end_byte]
                node_type = "FUNCTION" if node.type == "function_definition" else "CLASS"
                node_id = f"{node_type}:{file_path}:{name}"
                
                docstring = ""
                if node.child_count > 0 and node.children[-1].type == "block":
                    block = node.children[-1]
                    if block.child_count > 0 and block.children[0].type == "expression_statement":
                        expr = block.children[0]
                        if expr.child_count > 0 and expr.children[0].type == "string":
                            docstring = code[expr.children[0].start_byte:expr.children[0].end_byte]

                nodes.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "name": name,
                    "file_path": file_path,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "properties": {"docstring": docstring},
                    "code_content": code[node.start_byte:node.end_byte]
                })

                edges.append({
                    "source_id": parent_id,
                    "target_id": node_id,
                    "edge_type": "DEFINES"
                })
                
                parent_id = node_id

        # JavaScript definitions
        elif node.type in ["function_declaration", "class_declaration", "method_definition"]:
             name_node = node.child_by_field_name("name")
             if name_node:
                name = code[name_node.start_byte:name_node.end_byte]
                node_type = "FUNCTION" if "function" in node.type or "method" in node.type else "CLASS"
                node_id = f"{node_type}:{file_path}:{name}"
                
                nodes.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "name": name,
                    "file_path": file_path,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "code_content": code[node.start_byte:node.end_byte]
                })
                edges.append({
                    "source_id": parent_id,
                    "target_id": node_id,
                    "edge_type": "DEFINES"
                })
                parent_id = node_id

        # Java definitions
        elif node.type in ["class_declaration", "interface_declaration", "enum_declaration", "method_declaration", "constructor_declaration"]:
             name_node = node.child_by_field_name("name")
             if name_node:
                name = code[name_node.start_byte:name_node.end_byte]
                node_type = "FUNCTION" if "method" in node.type or "constructor" in node.type else "CLASS"
                node_id = f"{node_type}:{file_path}:{name}"
                
                nodes.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "name": name,
                    "file_path": file_path,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "code_content": code[node.start_byte:node.end_byte]
                })
                edges.append({
                    "source_id": parent_id,
                    "target_id": node_id,
                    "edge_type": "DEFINES"
                })
                parent_id = node_id

        for child in node.children:
            n, e = self._extract_definitions_data(child, parent_id, file_path, code)
            nodes.extend(n)
            edges.extend(e)
            
        return nodes, edges

    def _extract_references_data(self, node, source_id: str, file_path: str, code: str) -> List[Dict]:
        edges = []
        
        # Python Call (includes both function calls and class instantiation)
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                call_text = code[func_node.start_byte:func_node.end_byte]
                
                # Extract all potential target names
                # For obj.method() -> extract "method"
                # For ClassName() -> extract "ClassName"
                # For module.Class() -> extract "Class"
                call_names = set()
                
                if "." in call_text:
                    # Handle obj.method() or module.Class()
                    parts = call_text.split(".")
                    call_names.add(parts[-1])  # Last part (method/class name)
                    if len(parts) == 2:
                        call_names.add(parts[0])  # First part might be class name
                else:
                    call_names.add(call_text)
                
                for call_name in call_names:
                    if call_name and not call_name.startswith("_"):  # Skip special methods
                        potential_targets = self.store.find_nodes_by_name(call_name)
                        for target in potential_targets:
                            edges.append({
                                "source_id": source_id,
                                "target_id": target['id'],
                                "edge_type": "CALLS"
                            })

        # JavaScript/TypeScript Call (call_expression and new_expression)
        if node.type in ["call_expression", "new_expression"]:
            # Handle both regular calls and 'new ClassName()'
            if node.type == "new_expression":
                # new ClassName()
                class_node = node.child_by_field_name("constructor")
                if class_node:
                    class_name = code[class_node.start_byte:class_node.end_byte]
                    if "." in class_name:
                        class_name = class_name.split(".")[-1]
                    
                    potential_targets = self.store.find_nodes_by_name(class_name)
                    for target in potential_targets:
                        edges.append({
                            "source_id": source_id,
                            "target_id": target['id'],
                            "edge_type": "CALLS"
                        })
            else:
                # Regular function call
                func_node = node.child_by_field_name("function")
                if func_node:
                    call_text = code[func_node.start_byte:func_node.end_byte]
                    call_names = set()
                    
                    if "." in call_text:
                        parts = call_text.split(".")
                        call_names.add(parts[-1])
                    else:
                        call_names.add(call_text)
                    
                    for call_name in call_names:
                        if call_name:
                            potential_targets = self.store.find_nodes_by_name(call_name)
                            for target in potential_targets:
                                edges.append({
                                    "source_id": source_id,
                                    "target_id": target['id'],
                                    "edge_type": "CALLS"
                                })

        # Java Call (method_invocation and object_creation_expression)
        if node.type in ["method_invocation", "object_creation_expression"]:
            if node.type == "object_creation_expression":
                # new ClassName()
                type_node = node.child_by_field_name("type")
                if type_node:
                    class_name = code[type_node.start_byte:type_node.end_byte]
                    # Remove generic types: ClassName<T> -> ClassName
                    if "<" in class_name:
                        class_name = class_name.split("<")[0]
                    
                    potential_targets = self.store.find_nodes_by_name(class_name)
                    for target in potential_targets:
                        edges.append({
                            "source_id": source_id,
                            "target_id": target['id'],
                            "edge_type": "CALLS"
                        })
            else:
                # Regular method call
                name_node = node.child_by_field_name("name")
                if name_node:
                    call_name = code[name_node.start_byte:name_node.end_byte]
                    
                    potential_targets = self.store.find_nodes_by_name(call_name)
                    for target in potential_targets:
                        edges.append({
                            "source_id": source_id,
                            "target_id": target['id'],
                            "edge_type": "CALLS"
                        })
        new_source_id = source_id
        if node.type in ["function_definition", "class_definition", "function_declaration", "method_definition", "class_declaration", "interface_declaration", "enum_declaration", "method_declaration", "constructor_declaration"]:
             name_node = node.child_by_field_name("name")
             if name_node:
                name = code[name_node.start_byte:name_node.end_byte]
                node_type = "FUNCTION" if "function" in node.type or "method" in node.type else "CLASS"
                new_source_id = f"{node_type}:{file_path}:{name}"

        for child in node.children:
            edges.extend(self._extract_references_data(child, new_source_id, file_path, code))
            
        return edges

    def validate_syntax(self, code: str, file_path: str) -> List[str]:
        """
        Checks for syntax errors in the provided code.
        Returns a list of error messages.
        """
        ext = os.path.splitext(file_path)[1]
        if ext not in self.parsers:
            return [f"Unsupported file extension: {ext}"]
            
        parser = self.parsers[ext]
        tree = parser.parse(bytes(code, "utf8"))
        
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
        ext = os.path.splitext(file_path)[1]
        if ext not in self.parsers:
            return set()
        
        parser = self.parsers[ext]
        tree = parser.parse(bytes(code, "utf8"))
        
        called_functions = set()
        
        # Traverse tree to find function calls
        cursor = tree.walk()
        visited_children = False
        
        while True:
            if not visited_children:
                node = cursor.node
                
                # Python call
                if node.type == "call":
                    func_node = node.child_by_field_name("function")
                    if func_node and func_node.type == "identifier":
                        func_name = code[func_node.start_byte:func_node.end_byte]
                        called_functions.add(func_name)
                
                # JavaScript/TypeScript call_expression  
                elif node.type == "call_expression":
                    func_node = node.child_by_field_name("function")
                    if func_node and func_node.type == "identifier":
                        func_name = code[func_node.start_byte:func_node.end_byte]
                        called_functions.add(func_name)
                
                # Java method_invocation
                elif node.type == "method_invocation":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        func_name = code[name_node.start_byte:name_node.end_byte]
                        called_functions.add(func_name)
            
            if not visited_children and cursor.goto_first_child():
                visited_children = False
            elif cursor.goto_next_sibling():
                visited_children = False
            elif cursor.goto_parent():
                visited_children = True
            else:
                break
        
        return called_functions

    def extract_definitions_from_text(self, code: str, file_path: str) -> List[str]:
        """
        Parses code and returns a list of defined symbol names (functions/classes).
        Used for impact analysis.
        """
        ext = os.path.splitext(file_path)[1]
        if ext not in self.parsers:
            return []
            
        parser = self.parsers[ext]
        tree = parser.parse(bytes(code, "utf8"))
        
        definitions = []
        
        def _visit(node):
            if node.type in ["function_definition", "class_definition", "function_declaration", "class_declaration", "method_definition"]:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code[name_node.start_byte:name_node.end_byte]
                    definitions.append(name)
            for child in node.children:
                _visit(child)
                
        _visit(tree.root_node)
        return definitions

