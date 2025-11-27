#!/usr/bin/env bash
# Development wrapper for kgraph MCP server
# This bypasses uvx caching issues during local development

cd "$(dirname "$0")"
exec uv run python -m kgraph.server "$@"
