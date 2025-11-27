# Knowledge Graph MCP Server

A high-performance, lightweight Knowledge Graph MCP server for code analysis.

## Features

- **Fast Indexing**: Optimized for Apple Silicon (MLX) and multi-core CPUs.
- **Smart Indexing**: Incremental updates via content hashing.
- **Smart Scope**: Automatically ignores dependencies (node_modules, venv, etc.).
- **Semantic Search**: Find code by meaning, not just keywords.
- **Graph Analysis**: Understand relationships (definitions, references, calls).

## Installation

```bash
# Run directly with uv
uvx --from . kgraph

# Or install via pip
pip install .
kgraph
```

## Usage

Add to your MCP client configuration:

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
