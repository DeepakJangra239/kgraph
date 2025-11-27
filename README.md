# Knowledge Graph MCP Server

A high-performance, lightweight Knowledge Graph MCP server for code analysis, optimized for Apple Silicon.

## Features

- **Fast Indexing**: Optimized for Apple Silicon (MLX) and multi-core CPUs
- **Smart Incremental Indexing**: Content hashing skips unchanged files (instant re-indexing)
- **Smart Scope**: Automatically ignores dependencies (node_modules, venv, target, etc.)
- **Hybrid Search**: Semantic + exact keyword search
- **Graph Analysis**: Understand code relationships (definitions, references, calls)
- **Multi-Language**: Python, JavaScript, TypeScript, Java, Go

## Installation

```bash
# Run directly with uv
uvx --from . kgraph

# Or install via pip
pip install .
kgraph
```

## MCP Client Configuration

Add to your MCP client (e.g., Claude Desktop) configuration:

```json
{
  "mcpServers": {
    "kgraph": {
      "command": "uvx",
      "args": ["--from", "/path/to/kgraph", "kgraph"]
    }
  }
}
```

## Available Tools

The server exposes 8 tools that LLMs can use for code analysis:

### 1. `reindex_codebase`
**Purpose**: Index a codebase to enable all other tools.

**Parameters**:
- `project_root_path` (string): Absolute or relative path to the project directory

**LLM Usage**:
```
"Index the project at /path/to/my-app"
"Reindex the codebase to pick up my latest changes"
```

**Returns**: Indexing statistics (files processed, nodes/edges created, time taken)

---

### 2. `search_code`
**Purpose**: Search for code using hybrid semantic + keyword search with optional filtering.

**Parameters**:
- `query` (string): Search query (concept or exact name)
- `limit` (int, default=5): Maximum results to return
- `file_type` (string, optional): Filter by file extension (e.g., "py", "java", "ts")

**LLM Usage**:
```
"Search for authentication logic"
"Find code related to payment processing"
"Search for getUserById function"
"Search for React components in TypeScript files" (with file_type="tsx")
```

**Returns**: List of matching code snippets with:
- File path and line numbers
- Code content
- Match type (semantic or keyword)
- **Confidence score** (0.0-1.0, where 1.0 = 100% match)

---

### 3. `get_structure`
**Purpose**: Get the structure of a file (imports, classes, functions, methods).

**Parameters**:
- `file_path` (string): Path to the file

**LLM Usage**:
```
"Show me the structure of src/auth/login.py"
"What classes and methods are in UserService.java?"
```

**Returns**: Structured JSON with:
- `imports`: List of import statements
- `definitions`: Classes and functions with:
  - Nested methods (for classes)
  - Line numbers
  - Docstrings

---

### 4. `find_definitions`
**Purpose**: Find where a symbol (function, class, variable) is defined.

**Parameters**:
- `symbol_name` (string): Name of the symbol to find

**LLM Usage**:
```
"Where is the User class defined?"
"Find the definition of authenticate"
```

**Returns**: List of definitions with file paths, line numbers, and code snippets

---

### 5. `find_references`
**Purpose**: Find where a symbol is called/used in the codebase.

**Parameters**:
- `name` (string): Name of the symbol

**LLM Usage**:
```
"Show me where getUserById is called"
"Find all references to the PaymentService class"
```

**Returns**: List of call sites with:
- Source file and line number
- Caller name
- Target file and type

---

### 6. `get_file_summary`
**Purpose**: Get a summary of a file's contents.

**Parameters**:
- `file_path` (string): Path to the file

**LLM Usage**:
```
"Summarize what's in app/main.py"
"Give me an overview of UserController.java"
```

**Returns**: File summary with:
- Total lines
- Top-level structure (classes, functions)

---

### 7. `get_usage_context`
**Purpose**: Get the full context for a symbol (definition + usage).

**Parameters**:
- `symbol_name` (string): Name of the symbol

**LLM Usage**:
```
"Show me the full context for the login function"
"Get usage context for DatabaseConnection"
```

**Returns**: Combined information:
- Where it's defined (with code)
- Where it's used (with surrounding context)

---

### 8. `validate_edit`
**Purpose**: Validate code syntax before making changes (supports Python, JavaScript, TypeScript, Java, Go).

**Parameters**:
- `file_path` (string): Path to the file
- `new_content` (string): Proposed new content

**LLM Usage**:
```
"Check if this code change is syntactically valid"
"Validate this refactored function before applying"
```

**Returns**: Validation result:
- `valid`: true/false
- `errors`: List of syntax errors (if any)

---

## Example Workflows

### Initial Setup
```
1. LLM: "Index the project at /workspace/my-app"
2. System: Uses reindex_codebase
```

### Code Exploration
```
1. LLM: "Search for authentication logic"
2. System: Uses search_code (hybrid search)
3. LLM: "Show me the structure of auth/login.py"
4. System: Uses get_structure
```

### Refactoring
```
1. LLM: "Find all references to getUserById"
2. System: Uses find_references
3. LLM: "Show me the definition"
4. System: Uses find_definitions
5. LLM: "Validate this refactored version"
6. System: Uses validate_edit
```

## Performance

- **Indexing Speed**: ~500 files/s (initial), ~3500 files/s (incremental)
- **Search Speed**: <100ms for most queries
- **Languages**: Python, JavaScript, TypeScript, Java, Go
- **Smart Scope**: Auto-excludes ~3000+ dependency files

## License

MIT License - see [LICENSE](LICENSE) for details
