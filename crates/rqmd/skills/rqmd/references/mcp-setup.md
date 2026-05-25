# rqmd MCP Server Setup

## Install

```bash
cargo install rqmd
```

Then create an index:

```bash
rqmd collection add ~/path/to/markdown --name myknowledge
rqmd pull            # optional: pre-download models (otherwise fetched on first use)
rqmd embed
```

## Configure MCP Client

The rqmd MCP server starts with `rqmd mcp` (stdio transport) and identifies itself
as `rqmd`.

**Claude Code** (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "rqmd": { "command": "rqmd", "args": ["mcp"] }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "rqmd": { "command": "rqmd", "args": ["mcp"] }
  }
}
```

**OpenClaw** (`~/.openclaw/openclaw.json`):
```json
{
  "mcp": {
    "servers": {
      "rqmd": { "command": "rqmd", "args": ["mcp"] }
    }
  }
}
```

## HTTP Mode

```bash
rqmd mcp --http              # Streamable HTTP on port 8181
rqmd mcp --http --port 9000  # custom port
```

The HTTP server exposes the MCP endpoint at `/mcp` plus REST helpers (`/health`,
`/query`, `/search`).

## Tools

The server exposes four read-only tools.

### query

Search with pre-expanded, typed sub-queries.

```json
{
  "searches": [
    { "type": "lex", "query": "keyword phrases" },
    { "type": "vec", "query": "natural language question" },
    { "type": "hyde", "query": "hypothetical answer passage..." }
  ],
  "intent": "optional domain hint",
  "collections": ["optional"],
  "limit": 10,
  "minScore": 0.0
}
```

| Type | Method | Input |
|------|--------|-------|
| `lex` | BM25 | Keywords (2-5 terms) |
| `vec` | Vector | Question |
| `hyde` | Vector | Answer passage (50-100 words) |

Other optional params: `candidateLimit` (reranker candidate count) and `rerank`
(set `false` to skip LLM reranking and return RRF-blended ranks).

### get

Retrieve a document by file path, `qmd://` virtual path, or `#docid`.

| Param | Type | Description |
|-------|------|-------------|
| `file` | string | File path, `qmd://...` path, or `#docid` |
| `fromLine` | number? | Start at this line |
| `maxLines` | number? | Limit number of lines returned |
| `lineNumbers` | bool? | Prefix lines with line numbers |

### multi_get

Retrieve multiple documents.

| Param | Type | Description |
|-------|------|-------------|
| `pattern` | string | Glob (e.g. `journals/2025-05*.md`) or comma-separated list |
| `maxLines` | number? | Limit lines per file |
| `maxBytes` | number? | Skip files larger than N bytes (default 10240) |
| `lineNumbers` | bool? | Prefix lines with line numbers |

### status

Index health and collections. No params.

## Troubleshooting

- **Not starting**: confirm `rqmd` is on PATH (`rqmd --version`), then run
  `rqmd mcp` manually to see startup errors.
- **No results**: `rqmd collection list`, then `rqmd pull` and `rqmd embed`.
- **Slow first search**: Normal — local models download on first use (~3GB);
  pre-fetch with `rqmd pull` to avoid the wait.
