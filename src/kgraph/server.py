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

def get_or_create_store_for_path(root_path: str) -> tuple[GraphStore, CodeIndexer]:
    """Get or create a GraphStore and CodeIndexer for a specific project path."""
    global store, indexer
    
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
def search_code(query: str, limit: int = 5) -> str:
    """
    Performs a semantic search for code snippets relevant to the query.
    Note: Requires a project to be indexed first with reindex_codebase.
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
            final_results.append(res)
            seen_ids.add(res['id'])
            
    # Add semantic matches
    for res in semantic_results:
        if res['id'] not in seen_ids:
            res['search_method'] = 'semantic'
            final_results.append(res)
            seen_ids.add(res['id'])
            
    return json.dumps(final_results, indent=2)

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
    related = store.get_related(file_id, edge_type="DEFINES", direction="out")
    
    if not related:
        return f"No structure found for {file_path}. Is it indexed?"
    
    structure = []
    for node in related:
        structure.append({
            "type": node['type'],
            "name": node['name'],
            "line": node['start_line']
        })
    
    return json.dumps(structure, indent=2)

@mcp.tool()
def find_definitions(name: str) -> str:
    """
    Finds where a class or function is defined.
    Returns file path, line number, and docstring.
    """
    nodes = store.find_nodes_by_name(name)
    if not nodes:
        return f"No definitions found for '{name}'."
    
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
    for target in target_nodes:
        target_id = target['id']
        # 2. Find incoming CALLS edges
        # We want to know who calls this target
        callers = store.get_related(target_id, edge_type="CALLS", direction="in")
        
        for caller in callers:
            references.append({
                "source_file": caller['file_path'],
                "source_line": caller['start_line'],
                "caller_name": caller['name'],
                "target_file": target['file_path'],
                "target_type": target['type']
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
    
    target_node = nodes[0] # Take the first match for now
    target_id = target_node['id']
    
    context = [f"Symbol: {symbol} ({target_node['type']})"]
    context.append(f"Defined in: {target_node['file_path']} (L{target_node['start_line']})")
    doc = target_node.get('properties', {}).get('docstring', '')
    if doc:
        context.append(f"Docstring: {doc}")
        
    # 2. Find References (Incoming CALLS)
    # We need to query edges where target is target_id and type is CALLS
    refs = store.get_related(target_id, edge_type="CALLS", direction="in")
    context.append(f"\nUsed by ({len(refs)}):")
    for ref in refs:
        context.append(f" - {ref['name']} in {ref['file_path']}")
        
    # 3. Find Calls (Outgoing CALLS)
    calls = store.get_related(target_id, edge_type="CALLS", direction="out")
    context.append(f"\nCalls ({len(calls)}):")
    for call in calls:
        context.append(f" - {call['name']}")
        
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
    # Get current definitions in the graph for this file (recursive, all nodes in file)
    current_defs = store.get_nodes_by_file(file_path)
    # Filter out the file node itself
    current_defs = [n for n in current_defs if n['type'] != 'FILE']
    current_def_names = {node['name']: node['id'] for node in current_defs}
    
    # Get new definitions from proposed content
    new_def_names = set(indexer.extract_definitions_from_text(new_content, file_path))
    
    # Find removed definitions
    removed_names = set(current_def_names.keys()) - new_def_names
    
    breaking_changes = []
    for name in removed_names:
        node_id = current_def_names[name]
        # Check if this node is used by others
        refs = store.get_related(node_id, edge_type="CALLS", direction="in")
        if refs:
            ref_list = [f"{r['name']} in {r['file_path']}" for r in refs]
            breaking_changes.append(f"⚠️ BREAKING CHANGE: You are removing '{name}' which is used by:\n   - " + "\n   - ".join(ref_list))
            
    if breaking_changes:
        return "✅ Syntax is Valid.\n\n" + "\n".join(breaking_changes)
        
    return "✅ Edit looks safe. Syntax is valid and no breaking changes detected."

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
