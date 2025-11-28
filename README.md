# kgraph 🧠
**Knowledge Graph for Codebases**

`kgraph` is a powerful Model Context Protocol (MCP) server that indexes your codebase into a knowledge graph, enabling semantic search, precise code navigation, and impact analysis for LLM agents.

Unlike simple text search or grep, `kgraph` understands the *structure* of your code—classes, functions, imports, and calls—allowing agents to answer complex questions like "Who calls this function?" or "What happens if I change this class?".

## ✨ Features

- **Semantic Search**: Find code by meaning ("auth middleware") not just keywords. Uses LanceDB for vector search.
- **Precise Navigation**: Jump to definitions and find references with 100% accuracy using Tree-sitter parsing.
- **Structure Analysis**: Understand file outlines (classes, methods, hooks) instantly.
- **Impact Analysis**: `validate_edit` tool checks for syntax errors and breaking changes *before* you apply edits.
- **React/TypeScript Support**: First-class support for modern web frameworks (Components, Hooks, JSX).
- **Multi-Language**: Supports Python, JavaScript, TypeScript, Java, and more.

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- `uv` (recommended) or `pip`

### Install with uv (Recommended)
```bash
# Install directly from source
uv tool install --force --editable .
```

### Install with pip
```bash
pip install .
```

## ⚡ Quick Start

1. **Start the Server**:
   Run the server using an MCP client (see below) or directly for testing:
   ```bash
   kgraph
   ```

2. **Index Your Codebase**:
   The first time you use `kgraph` on a project, you must index it:
   - Use the `reindex_codebase` tool with the path to your project.
   - Example: `reindex_codebase(root_path="/path/to/my/project")`

## 🛠️ MCP Tools

`kgraph` exposes the following tools to MCP clients:

### 🔍 Search & Navigation
- **`search_code(query, limit=5, file_type="")`**
  - Performs a hybrid search (Semantic + Keyword) to find relevant code snippets.
  - Great for: "Find the user authentication logic" or "Where is the payment processed?"

- **`find_definitions(name)`**
  - Locates where a class, function, or component is defined.
  - Returns file path, line number, and docstring.

- **`find_references(name)`**
  - Finds all usages of a symbol (calls, imports, inheritance).
  - Essential for refactoring and understanding dependencies.

### 📄 Code Understanding
- **`get_structure(file_path)`**
  - Returns a structured outline of a file: imports, classes, functions, and methods.
  - Supports React components, hooks, and event handlers.

- **`get_usage_context(symbol)`**
  - Retrieves a 360° view of a symbol: Definition + References + Outgoing Calls.
  - Optimized for LLMs to get full context in one shot.

- **`get_file_summary(file_path)`**
  - Returns a token-efficient summary of a file (signatures and docstrings only).

### 🛡️ Validation
- **`validate_edit(file_path, new_content)`**
  - Validates a proposed code edit *before* applying it.
  - Checks for:
    1. **Syntax Errors**: Ensures code is valid (supports JSX/TSX).
    2. **Breaking Changes**: Warns if you remove a function used by other files.
    3. **Signature Changes**: Warns if you change a function signature used elsewhere.

### ⚙️ Management
- **`reindex_codebase(root_path)`**
  - Scans and indexes the codebase. Run this initially and after major changes.
  - Creates a `.kgraph` directory in your project root.

## 🔌 MCP Client Configuration

### Claude Desktop
Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kgraph": {
      "command": "uv",
      "args": [
        "tool",
        "run",
        "kgraph"
      ]
    }
  }
}
```

### Cursor / Other Clients
If your client supports stdio MCP servers, use:
- **Command**: `uv`
- **Args**: `tool run kgraph`

## 🏗️ Architecture

`kgraph` uses a hybrid storage approach:
- **SQLite**: Stores the structural graph (Nodes: Files, Functions, Classes; Edges: IMPORTS, CALLS, DEFINES).
- **LanceDB**: Stores vector embeddings of code snippets for semantic search.
- **Tree-sitter**: Used for robust, error-tolerant parsing of source code.

## 📝 License

MIT
