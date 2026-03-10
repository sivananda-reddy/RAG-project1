# Hybrid RAG: Vector Search + Knowledge Graph

This project uses a **hybrid retrieval** approach: **vector search** and **knowledge graph** together.

## How It Works

### 1. Vector search (existing)
- Documents are split into chunks and embedded.
- At query time, the question is embedded and the top‑k similar chunks are retrieved from ChromaDB.

### 2. Knowledge graph (new)
- When you index documents, the LLM extracts **knowledge triples** (subject, predicate, object) from a sample of chunks.
- Triples are stored in a graph (e.g. `Machine Learning --[is a subset of]--> Artificial Intelligence`).
- The graph is saved to `knowledge_graph.json` and reused across sessions.

### 3. Query time (hybrid)
- **Vector path**: Embed the question → retrieve top‑k chunks → use as “Relevant passages”.
- **Graph path**: Extract key entities from the question → get a 2‑hop subgraph around those entities → format as “Knowledge graph (related facts)”.
- Both are concatenated into one context and sent to the LLM for the final answer.

## Configuration

In `.env`:

- **`USE_KNOWLEDGE_GRAPH=true`**  
  Turn hybrid mode on (default: true). Set to `false` to use only vector search.

- **`KG_MAX_CHUNKS=40`**  
  Max number of chunks used to build the graph (limits LLM calls during indexing). Default: 40.

## When the Graph Is Built

- The graph is built when you run **“Update with New Documents”** (or the first **“Initialize Pipeline”** that indexes documents).
- It is **rebuilt from scratch** when you use **“Update with New Documents”** (force re-index).
- It is **cleared** when you use **“Clear embeddings”** in the chatbot (and on next index it will be rebuilt).

## Files

- **`src/knowledge_graph.py`**  
  - `KnowledgeGraph`: store/load triples, subgraph retrieval.  
  - `extract_triples_with_llm()`: extract triples from text.  
  - `extract_entities_from_query()`: extract entities from the user question.

- **`knowledge_graph.json`** (project root)  
  Persisted triples; created/updated when the graph is built.

## Disabling the Knowledge Graph

- In `.env`: set **`USE_KNOWLEDGE_GRAPH=false`**, or  
- In code:  
  `OptimizedRAGPipeline(..., use_knowledge_graph=False)`  
  so retrieval is vector-only.

## Dependencies

- **networkx**  
  Used for in-memory graph and subgraph traversal.  
  Install: `pip install networkx`
