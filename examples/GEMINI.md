# Project Knowledge & AI Agent Memory Directives

This project uses **GraphLite** as its embedded GraphRAG knowledge base and long-term memory engine persisted in `knowledge.graph`.

## Mandatory Rules for AI Assistants:

1. **Knowledge & Context Retrieval:**
   - Whenever answering questions regarding system architecture, business rules, policies, APIs, or user preferences, you **MUST** call the MCP tool `graphlite_retrieve(query: "...")`.
2. **Prohibition of Direct Binary File Reading:**
   - The `.graph` files (e.g. `knowledge.graph`) are compiled single-file binary databases. **NEVER** attempt to inspect them using `cat`, `read_file`, `head`, `strings`, or SQLite CLI tools.
   - All knowledge inspection and retrieval must be performed exclusively via the `graphlite_retrieve` MCP tool.
3. **Continuous Agent Memory & Knowledge Persistence:**
   - When discovering new business rules, domain facts, or user preferences during conversations, persist them using `graphlite_remember(name: "...", type: "...", description: "...")` and connect dependencies via `graphlite_connect`.
