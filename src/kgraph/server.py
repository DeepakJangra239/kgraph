from mcp.server.fastmcp import FastMCP
import os
import sys
import json

# Add src to path so we can import if running directly
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .core.store import GraphStore
from .core.indexer import CodeIndexer

# Initialize Server
mcp = FastMCP("KnowledgeGraph")

# Global store and indexer (will be reinitialized per project)
store = None
indexer = None
observer = None
current_watch_path = None

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("[WARNING] watchdog not installed. Automatic updates disabled.", file=sys.stderr)

class CodeEventHandler(FileSystemEventHandler):
    def __init__(self, indexer_instance):
        self.indexer = indexer_instance
        self.last_events = {}
        
    def _process(self, event):
        if event.is_directory:
            return
            
        # Ignore .kgraph and other hidden directories
        if "/.kgraph/" in event.src_path or "/." in event.src_path:
            return
            
        # Simple debounce/deduplicate
        import time
        current_time = time.time()
        if event.src_path in self.last_events:
            if current_time - self.last_events[event.src_path] < 1.0:
                return
        self.last_events[event.src_path] = current_time
        
        try:
            if event.event_type == 'deleted':
                self.indexer.remove_file(event.src_path)
            elif event.event_type in ['created', 'modified']:
                self.indexer.update_file(event.src_path)
            elif event.event_type == 'moved':
                self.indexer.remove_file(event.src_path)
                self.indexer.update_file(event.dest_path)
        except Exception as e:
            print(f"[ERROR] Error processing file event {event}: {e}", file=sys.stderr)

    def on_modified(self, event):
        self._process(event)
    
    def on_created(self, event):
        self._process(event)
        
    def on_deleted(self, event):
        self._process(event)
        
    def on_moved(self, event):
        self._process(event)

def start_watcher(path, indexer_instance):
    global observer, current_watch_path
    
    if not WATCHDOG_AVAILABLE:
        return
        
    if observer and current_watch_path == path:
        return # Already watching
        
    if observer:
        observer.stop()
        observer.join()
        
    current_watch_path = path
    event_handler = CodeEventHandler(indexer_instance)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"[INFO] Started file watcher for {path}", file=sys.stderr)

def get_or_create_store_for_path(root_path: str) -> tuple[GraphStore, CodeIndexer]:
    """Get or create a GraphStore and CodeIndexer for a specific project path."""
    global store, indexer
    
    # Resolve symlinks to get canonical path
    root_path = os.path.realpath(root_path)
    
    # Create .kgraph directory inside the project
    kgraph_dir = os.path.join(root_path, ".kgraph")
    os.makedirs(kgraph_dir, exist_ok=True)
    
    db_path = os.path.join(kgraph_dir, "graph.db")
    lancedb_path = os.path.join(kgraph_dir, "vectors")
    
    # Reinitialize if path changed or not initialized
    if store is None or store.db_path != db_path:
        print(f"[INFO] Initializing database for {root_path}", file=sys.stderr)
        print(f"[INFO] Database: {db_path}", file=sys.stderr)
        store = GraphStore(db_path=db_path, lancedb_path=lancedb_path)
        indexer = CodeIndexer(store)
        
    # Ensure watcher is running for this path
    start_watcher(root_path, indexer)
    
    return store, indexer

def main():
    """Entry point for the CLI."""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()

@mcp.tool()
def reindex_codebase(root_path: str) -> str:
    """
    Scans and indexes the specified directory to build the knowledge graph.
    This is a heavy operation and should be called when the codebase changes significantly.
    Database will be created in {root_path}/.kgraph/
    """
    if not os.path.exists(root_path):
        return f"Error: Path '{root_path}' does not exist."
    
    root_path = os.path.abspath(root_path)
    
    try:
        # Get or create store for this specific project
        project_store, project_indexer = get_or_create_store_for_path(root_path)
        
        # Index the codebase
        project_indexer.index_codebase(root_path)
        return f"Successfully indexed codebase at {root_path}\nDatabase: {root_path}/.kgraph/"
    except Exception as e:
        import traceback
        import sys
        traceback.print_exc(file=sys.stderr)
        return f"Error indexing codebase: {str(e)}"

@mcp.tool()
def search_code(query: str, limit: int = 5, file_type: str = "", verbose: bool = False) -> str:
    """
    Performs a semantic search for code snippets relevant to the query.
    Note: Requires a project to be indexed first with reindex_codebase.
    
    Parameters:
    - query: Search query
    - limit: Maximum results (default 5)
    - file_type: Optional file extension filter (e.g., "py", "java", "ts")
    - verbose: If True, include technical details like vectors and distances (default False)
    """
    if store is None:
        return "Error: No project indexed yet. Please run reindex_codebase first."

    # Hybrid Search: Semantic + Keyword
    semantic_results = store.search_nodes(query, limit=limit)
    keyword_results = store.search_nodes_keyword(query, limit=limit)
    
    # Deduplicate and Merge
    seen_ids = set()
    final_results = []
    
    # Prioritize keyword matches (exact hits)
    for res in keyword_results:
        if res['id'] not in seen_ids:
            res['search_method'] = 'keyword'
            res['confidence'] = 1.0  # Exact match = 100% confidence
            final_results.append(res)
            seen_ids.add(res['id'])
            
    # Add semantic matches
    for res in semantic_results:
        if res['id'] not in seen_ids:
            res['search_method'] = 'semantic'
            # Convert distance to confidence (0-1 scale)
            # Lower distance = higher confidence
            # Typical distance range is 0-2, so we use 1 / (1 + distance)
            distance = res.get('_distance', 1.0)
            res['confidence'] = round(1.0 / (1.0 + distance), 2)
            final_results.append(res)
            seen_ids.add(res['id'])
    
    # Apply file type filter if specified
    if file_type:
        # Normalize file_type (remove leading dot if present)
        if not file_type.startswith('.'):
            file_type = '.' + file_type
        
        final_results = [
            r for r in final_results 
            if r.get('file_path', '').endswith(file_type)
        ]
            
    # Clean up results based on verbose mode
    clean_results = []
    for r in final_results[:limit]:
        if verbose:
            # Verbose mode: include all data
            clean_results.append(r)
        else:
            # Clean mode (default): remove vector data and internal metrics
            clean = {k: v for k, v in r.items() if k not in ['vector', '_distance']}
            clean_results.append(clean)
    
    return json.dumps(clean_results, indent=2)

@mcp.tool()
def get_structure(file_path: str) -> str:
    """
    Returns the structure (classes, functions) of a specific file.
    Note: Requires the file's project to be indexed first.
    """
    file_path = os.path.abspath(file_path)
    if store is None:
        return "Error: No project indexed yet. Please run reindex_codebase first."

    # Find the file node first
    # This is a simplified lookup. In a real app, we'd query by file_path property.
    # For now, let's assume we can query edges from the file node if we know its ID.
    # Since IDs are deterministic (file:path), we can construct it.
    
    file_id = f"file:{file_path}"
    
    # Fetch Imports
    imports = store.get_related(file_id, edge_type="IMPORTS", direction="out")
    
    # Fetch Definitions (Classes, Functions)
    definitions = store.get_related(file_id, edge_type="DEFINES", direction="out")
    
    if not imports and not definitions:
        return f"No structure found for {file_path}. Is it indexed?"
    
    structure = {
        "imports": [imp['name'] for imp in imports],
        "definitions": []
    }
    
    for node in definitions:
        item = {
            "type": node['type'],
            "name": node['name'],
            "line": node['start_line'],
            "doc": node.get('properties', {}).get('docstring', '')
        }
        
        # If Class, fetch nested methods
        if node['type'] == 'CLASS':
            methods = store.get_related(node['id'], edge_type="DEFINES", direction="out")
            if methods:
                item['methods'] = []
                for m in methods:
                    item['methods'].append({
                        "name": m['name'],
                        "line": m['start_line'],
                        "doc": m.get('properties', {}).get('docstring', '')
                    })
        
        structure["definitions"].append(item)
        
    return json.dumps(structure, indent=2)

@mcp.tool()
def find_definitions(name: str) -> str:
    """
    Finds where a class or function is defined.
    Returns file path, line number, and docstring.
    """
    # 1. Exact match
    nodes = store.find_nodes_by_name(name)
    
    # Filter out IMPORT nodes - prioritize actual definitions
    actual_defs = [n for n in nodes if n['type'] in ['CLASS', 'FUNCTION', 'METHOD']]
    if actual_defs:
        nodes = actual_defs
    
    # 2. Fuzzy match if no exact match
    if not nodes:
        # Get all node names (this might be expensive on huge graphs, but okay for now)
        # Optimization: Use SQL LIKE query
        conn = store._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM nodes WHERE name LIKE ? AND type IN ('CLASS', 'FUNCTION', 'METHOD')", (f"%{name}%",))
        matches = cursor.fetchall()
        fuzzy_names = [m[0] for m in matches]
        
        # Limit to top 5
        fuzzy_names = fuzzy_names[:5]
        
        for fuzzy_name in fuzzy_names:
            fuzzy_nodes = store.find_nodes_by_name(fuzzy_name)
            # Again, filter to actual definitions
            nodes.extend([n for n in fuzzy_nodes if n['type'] in ['CLASS', 'FUNCTION', 'METHOD']])
            
    if not nodes:
        return f"No definitions found for '{name}'.'"
    
    results = []
    for node in nodes:
        results.append({
            "type": node['type'],
            "file": node['file_path'],
            "line": node['start_line'],
            "doc": node.get('properties', {}).get('docstring', '')
        })
    return json.dumps(results, indent=2)

@mcp.tool()
def find_references(name: str) -> str:
    """
    Finds where a symbol is used in the codebase.
    Returns a list of files and lines where the symbol is called.
    """
    if store is None:
        return "Error: No project indexed yet. Please run reindex_codebase first."

    # 1. Find definition nodes for the name
    target_nodes = store.find_nodes_by_name(name)
    if not target_nodes:
        return f"No definitions found for '{name}'. Cannot find references."
    
    references = []
    seen_refs = set()  # Track unique references
    
    # Filter to only actual definitions (CLASS, FUNCTION), not IMPORT nodes
    definition_nodes = [n for n in target_nodes if n['type'] in ['CLASS', 'FUNCTION', 'METHOD']]
    
    if not definition_nodes:
        return f"No definitions found for '{name}'. Only found imports/references."
    
    for target in definition_nodes:
        target_id = target['id']
        # 2. Find incoming CALLS edges
        callers = store.get_related(target_id, edge_type="CALLS", direction="in")
        
        for caller in callers:
            # Use a comprehensive key to avoid any duplicates
            ref_key = (
                caller['file_path'], 
                caller['start_line'], 
                caller['name'],
                target['file_path'],
                target['name']
            )
            if ref_key not in seen_refs:
                seen_refs.add(ref_key)
                references.append({
                    "source_file": caller['file_path'],
                    "source_line": caller['start_line'],
                    "caller_name": caller['name'],
                    "target_file": target['file_path'],
                    "target_name": target['name'],
                    "target_type": target['type'],
                    "reference_type": "CALL"
                })

        # 3. Find incoming IMPORTS edges
        importers = store.get_related(target_id, edge_type="IMPORTS", direction="in")
        for imp in importers:
            ref_key = (imp['file_path'], imp['start_line'], "import", target['file_path'], target['name'])
            if ref_key not in seen_refs:
                seen_refs.add(ref_key)
                references.append({
                    "source_file": imp['file_path'],
                    "source_line": imp['start_line'],
                    "caller_name": "import",
                    "target_file": target['file_path'],
                    "target_name": target['name'],
                    "target_type": target['type'],
                    "reference_type": "IMPORT"
                })

        # 4. Find incoming INHERITS_FROM edges
        inheritors = store.get_related(target_id, edge_type="INHERITS_FROM", direction="in")
        for inh in inheritors:
            ref_key = (inh['file_path'], inh['start_line'], inh['name'], target['file_path'], target['name'])
            if ref_key not in seen_refs:
                seen_refs.add(ref_key)
                references.append({
                    "source_file": inh['file_path'],
                    "source_line": inh['start_line'],
                    "caller_name": inh['name'],
                    "target_file": target['file_path'],
                    "target_name": target['name'],
                    "target_type": target['type'],
                    "reference_type": "INHERITANCE"
                })
            
    if not references:
        return f"No references found for '{name}'."
        
    return json.dumps(references, indent=2)

@mcp.tool()
def get_file_summary(file_path: str) -> str:
    """
    Returns a token-efficient summary of a file: imports, classes, functions, and docstrings.
    Optimized for small context windows.
    """
    file_path = os.path.abspath(file_path)
    file_id = f"file:{file_path}"
    node = store.get_node(file_id)
    if not node:
        return f"File {file_path} not found in index."
    
    related = store.get_related(file_id, edge_type="DEFINES", direction="out")
    
    summary = [f"File: {file_path}"]
    summary.append(f"Lines: {node.get('end_line', '?')}")
    summary.append("Structure:")
    
    for child in related:
        kind = child['type']
        name = child['name']
        line = child['start_line']
        doc = child.get('properties', {}).get('docstring', '')
        doc_snippet = doc[:50] + "..." if len(doc) > 50 else doc
        summary.append(f"  - {kind} {name} (L{line}): {doc_snippet}")
        
    return "\n".join(summary)

@mcp.tool()
def get_usage_context(symbol: str) -> str:
    """
    Retrieves a comprehensive context for a symbol:
    1. Definition (where it is)
    2. References (who calls it)
    3. Calls (who it calls)
    
    This is optimized for small LLMs to get full context in one shot.
    """
    # 1. Find Definition
    nodes = store.find_nodes_by_name(symbol)
    if not nodes:
        return f"Symbol '{symbol}' not found."
    
    # Prefer DEFINITIONS (FUNCTION, CLASS) over IMPORTS
    target_node = next((n for n in nodes if n['type'] in ['FUNCTION', 'CLASS']), nodes[0])
    target_id = target_node['id']
    
    context = [f"Symbol: {symbol} ({target_node['type']})"]
    context.append(f"Defined in: {target_node['file_path']} (L{target_node['start_line']})")
    doc = target_node.get('properties', {}).get('docstring', '')
    if doc:
        context.append(f"Docstring: {doc}")
        
    # 2. Find References (Incoming CALLS)
    callers = store.get_related(target_id, edge_type="CALLS", direction="in")
    context.append(f"Used by ({len(callers)}):")
    for c in callers[:5]:
        name = c.get('name', 'Unknown')
        file_path = c.get('file_path', '')
        fname = os.path.basename(file_path) if file_path else 'Unknown File'
        context.append(f"  - {name} in {fname}")
        
    # 3. Find Outgoing Calls
    calls = store.get_related(target_id, edge_type="CALLS", direction="out")
    context.append(f"Calls ({len(calls)}):")
    for c in calls[:5]:
        name = c.get('name', 'Unknown')
        context.append(f"  - {name}")

    # 4. Find Importers
    importers = store.get_related(target_id, edge_type="IMPORTS", direction="in")
    if importers:
        context.append(f"Imported by ({len(importers)}):")
        for imp in importers[:5]:
            file_path = imp.get('file_path', '')
            fname = os.path.basename(file_path) if file_path else 'Unknown File'
            context.append(f"  - {fname}")
            
    # 5. Find Inheritors
    inheritors = store.get_related(target_id, edge_type="INHERITS_FROM", direction="in")
    if inheritors:
        context.append(f"Inherited by ({len(inheritors)}):")
        for inh in inheritors[:5]:
            name = inh.get('name', 'Unknown')
            file_path = inh.get('file_path', '')
            fname = os.path.basename(file_path) if file_path else 'Unknown File'
            context.append(f"  - {name} in {fname}")
        
    return "\n".join(context)

@mcp.tool()
def validate_edit(file_path: str, new_content: str) -> str:
    """
    Validates a proposed code edit BEFORE applying it.
    Checks for:
    1. Syntax Errors (missing braces, invalid syntax).
    2. Breaking Changes (removing functions/classes that are used by other files).
    
    Use this tool to check your code before writing it to a file.
    """
    file_path = os.path.abspath(file_path)
    
    # 1. Syntax Check
    syntax_errors = indexer.validate_syntax(new_content, file_path)
    if syntax_errors:
        return "❌ SYNTAX ERRORS DETECTED:\n" + "\n".join(syntax_errors)
    
    # 2. Impact Analysis (Breaking Changes)
    current_defs = store.get_nodes_by_file(file_path)
    current_defs = [n for n in current_defs if n['type'] != 'FILE']
    current_def_names = {node['name']: node['id'] for node in current_defs}
    
    new_defs = indexer.extract_definitions_from_text(new_content, file_path)
    new_def_names = set(new_defs.keys())
    removed_names = set(current_def_names.keys()) - new_def_names
    
    breaking_changes = []
    for name in removed_names:
        node_id = current_def_names[name]
        refs = store.get_related(node_id, edge_type="CALLS", direction="in")
        if refs:
            ref_list = [f"{r['name']} in {r['file_path']}" for r in refs]
            breaking_changes.append(f"⚠️ BREAKING CHANGE: You are removing '{name}' which is used by:\n   - " + "\n   - ".join(ref_list))
            
    # Check for signature changes
    try:
        with open(file_path, 'r') as f:
            old_content = f.read()
        old_defs = indexer.extract_definitions_from_text(old_content, file_path)
        
        for name, new_sig in new_defs.items():
            if name in old_defs:
                old_sig = old_defs[name]
                if old_sig != new_sig:
                    node_id = current_def_names.get(name)
                    if node_id:
                        refs = store.get_related(node_id, edge_type="CALLS", direction="in")
                        if refs:
                            ref_list = [f"{r['name']} in {r['file_path']}" for r in refs]
                            breaking_changes.append(f"⚠️ SIGNATURE CHANGE: '{name}' signature changed from '{old_sig}' to '{new_sig}'. Check usage in:\n   - " + "\n   - ".join(ref_list))
    except Exception as e:
        pass
    
    
    # 3. Runtime Validation Warnings (Undefined symbols)
    # Dynamically detect built-in functions instead of hardcoding
    def is_builtin(call_name: str, file_path: str) -> bool:
        """Check if a function is a built-in for the given language."""
        ext = os.path.splitext(file_path)[1]
        
        # Python built-ins
        if ext == '.py':
            import builtins
            return hasattr(builtins, call_name)
        
        # JavaScript/TypeScript built-ins
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            # Common JS global objects and functions
            js_globals = {
                'console', 'window', 'document', 'Array', 'Object', 'String', 
                'Number', 'Boolean', 'Math', 'JSON', 'Date', 'Promise', 'Set', 
                'Map', 'Error', 'setTimeout', 'setInterval', 'fetch', 'parseInt',
                'parseFloat', 'isNaN', 'isFinite', 'encodeURIComponent', 'decodeURIComponent'
            }
            return call_name in js_globals
        
        # Java built-ins (minimal set - most are in imported classes)
        elif ext == '.java':
            # Java doesn't have many true built-ins, mostly in java.lang
            java_implicit = {'System', 'Math', 'String', 'Integer', 'Double', 'Boolean'}
            return call_name in java_implicit
        
        return False
    
    warnings = []
    undefined_calls = indexer.find_undefined_calls(new_content, file_path)
    imported_symbols = indexer.extract_imports(new_content, file_path)
    
    if undefined_calls:
        for call_name in undefined_calls:
            # Skip built-in functions (dynamic check)
            if is_builtin(call_name, file_path):
                continue
            
            # Skip imported symbols
            if call_name in imported_symbols:
                continue
                
            # Check if it exists anywhere in the codebase
            definitions = store.find_nodes_by_name(call_name)
            if not definitions:
                warnings.append(f"⚠️ WARNING: Function '{call_name}' is called but not found in the indexed codebase. Verify:")
                warnings.append(f"   - Is it imported from an external library?")
                warnings.append(f"   - Is it defined elsewhere?")
                warnings.append(f"   - Is the spelling correct?")
    
    # Build response
    response = "✅ Syntax is Valid.\n"
    
    if breaking_changes:
        response += "\n" + "\n".join(breaking_changes)
    
    if warnings:
        response += "\n\n" + "\n".join(warnings)
    
    if not breaking_changes and not warnings:
        response += "\n✅ No breaking changes or warnings detected."
        
    return response

def main():
    """Entry point for the application script"""
    mcp.run(transport="stdio")

# Run the server
if __name__ == "__main__":
    main()

