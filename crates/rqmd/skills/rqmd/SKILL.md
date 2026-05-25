---
name: rqmd
description: Search local markdown knowledge bases, notes, docs, and wikis with rqmd. Use when users ask to find notes, retrieve documents, inspect a wiki, answer from indexed markdown, or set up rqmd access.
license: MIT
compatibility: Requires the rqmd CLI or MCP server. Build from source with cargo (see Setup).
metadata:
  author: tobi (original qmd skill); ported to rqmd
  version: "2.1.0"
allowed-tools: Bash(rqmd:*), mcp__rqmd__*
---

# rqmd — Query Markdown Documents

## How search works

rqmd searches local markdown collections: notes, docs, wikis, transcripts, and
project knowledge bases. Use it before web search when the answer may already be
in indexed local files.

The workflow is always:

1. Search for candidate documents.
2. Retrieve the full source with `rqmd get` or `rqmd multi-get`.
3. Answer from retrieved text, citing paths or docids.

Do not answer from snippets alone when the user needs facts, decisions, quotes,
or nuance. Snippets are only leads.

Typical loop:

```bash
rqmd search "merchant reality support interviews" -n 5
# leads: #abc123 concepts/customer-proximity.md; #def432 sources/merchant-call.md
rqmd multi-get 'concepts/customer-proximity.md,sources/merchant-call.md' --md
```

For harder searches, use `rqmd query` structured queries with `intent:`, `lex:`,
`vec:`, and `hyde:` fields.

When reporting what you retrieved, a compact note is enough; do not paste whole
files unless needed:

```text
Retrieved:
- #abc123 concepts/customer-proximity.md
- #def432 sources/merchant-call.md
```

## Pick the right search mode

Use **BM25 lexical search** when you know exact words, titles, names, code
symbols, or rare phrases:

```bash
rqmd search "cockpit OKR Goodhart" -n 10
rqmd search '"AI Before Headcount"' -c concepts -n 5
```

Use **hybrid semantic search** when the user describes an idea indirectly, uses
different wording than the source, or needs conceptual recall:

```bash
rqmd query "decision quality depends on surfacing assumptions and context" -n 10
rqmd query --json --explain "metrics as cockpit instruments but not OKRs"
```

Use **structured queries** for hard searches. They combine exact anchors with
semantic recall:

```bash
rqmd query $'intent: Find the concept note about metrics as instruments without letting OKRs replace judgment.\nlex: cockpit instruments OKR Goodhart metrics judgment\nvec: data informed not metric driven product judgment\nhyde: A concept note says metrics are useful like cockpit instruments, but leaders should remain data-informed rather than metric-driven because OKRs and dashboards can Goodhart product judgment.'
```

Structured query fields:

- `intent:` states what you are trying to find and what to avoid.
- `lex:` uses exact terms, aliases, titles, and rare words.
- `vec:` paraphrases the idea in natural language.
- `hyde:` describes the document or answer that would satisfy the request.

If `rqmd query` is slow or model/GPU setup fails, fall back to `rqmd search` with
better lexical terms (or force CPU with the global `--no-gpu` flag).

## Retrieve sources

Search results include docids like `#abc123` and `qmd://...` paths. Fetch them:

```bash
rqmd get "#abc123"
rqmd get qmd://concepts/ai-before-headcount.md --full
rqmd multi-get 'concepts/{ai-before-headcount.md,data-informed-not-metric-driven.md}' --md
rqmd multi-get 'sources/podcast-2025-*.md' -l 80
```

`rqmd get` returns the full document by default (`--full` is accepted for
compatibility). Slice with `--from` / `-l`, and add `--line-numbers` to prefix
each line with its number. Use `multi-get` — a glob or comma-separated list of
paths — when comparing several hits or gathering context across pages.

Note: rqmd keeps qmd's `qmd://` virtual-path scheme for compatibility — search
JSON output, `get`, and `ls` all use `qmd://`. `multi-get` resolves paths and
globs (not docids); fetch a single docid with `rqmd get "#abc123"`.

## Discover what is indexed

```bash
rqmd collection list
rqmd ls
rqmd status
```

Add collection filters when broad searches drift into the wrong corpus:

```bash
rqmd search "headcount autonomous agents" -c concepts -n 10
rqmd query "merchant support product reality" -c concepts -c sources -n 10
```

Omit `-c` to search everything.

## MCP Tool: `query`

When using the MCP server, prefer structured searches:

```json
{
  "searches": [
    { "type": "lex", "query": "cockpit OKR Goodhart" },
    { "type": "vec", "query": "data informed not metric driven product judgment" },
    { "type": "hyde", "query": "A concept note explains that metrics are useful as instruments, but leaders should not let OKRs or dashboards replace judgment." }
  ],
  "intent": "Find the concept note about using metrics as instruments without becoming metric-driven.",
  "collections": ["concepts"],
  "limit": 10
}
```

Query types:

- `lex` — BM25 keyword search. Best for exact terms, names, titles, and code.
- `vec` — vector semantic search. Best for natural-language concepts.
- `hyde` — vector search using a hypothetical answer/document passage.

## Query craft

Good rqmd searches mix three things:

1. **Title/alias anchors:** exact page titles, named entities, phrases.
2. **Semantic paraphrase:** how a human would describe the idea.
3. **Negative space:** enough intent to avoid nearby-but-wrong concepts.

Examples:

```bash
# Exact-ish title lookup
rqmd search '"arm the rebels" merchants tools big companies' -c concepts

# Semantic concept lookup
rqmd query $'intent: Find the customer proximity concept, not generic customer delight.\nlex: support pseudonymous merchant customer interviews\nvec: founder stays close to merchant reality through support and product use'

# Source lookup
rqmd search "six-week cadence WhatsApp merchant relationships Shawn Ryan" -c sources -n 10
```

## Setup and maintenance

Only mutate indexes when the user asked for setup or maintenance. Searching and
retrieving are safe; collection/index mutation is not a casual first step.

Install from crates.io:

```bash
cargo install rqmd
```

Then create and maintain an index:

```bash
rqmd collection add ~/notes --name notes
rqmd update
rqmd embed
```

Health and diagnostics:

```bash
rqmd doctor
rqmd status
rqmd pull
```

`rqmd doctor` checks index config, model cache, device/GPU setup, vector
fingerprints, and common environment overrides. If a model-backed command fails,
run it before changing configuration. `rqmd pull` pre-downloads the configured
models (otherwise fetched on first use).

## MCP setup

See `references/mcp-setup.md` for Claude Code, Claude Desktop, OpenClaw, and HTTP
server configuration.

## Pitfalls

- **Do not stop at snippets.** Fetch documents before making claims.
- **Do not overuse semantic search.** If you know exact titles or terms, BM25 is
  faster and often better.
- **Do not mutate indexes casually.** `rqmd collection add`, `rqmd update`, and
  `rqmd embed` change local state and can be expensive.
- **Model-backed commands can be environment-sensitive.** If `rqmd query`,
  `rqmd vsearch`, or reranking fails because local models/GPU are unavailable,
  use `rqmd search` and stronger lexical/structured terms (or `--no-gpu`).
- **Ambiguous user wording needs intent.** Add `intent:` rather than hoping query
  expansion guesses the right domain.
- **Collection names matter.** Search `concepts` for synthesized wiki pages,
  `sources` for transcripts/raw source pages, and docs collections for code or
  project documentation.
