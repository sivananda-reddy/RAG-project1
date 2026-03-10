"""
Knowledge Graph Module for Hybrid RAG

Extracts (subject, predicate, object) triples from text using LLM,
stores them in a graph, and retrieves relevant subgraphs for queries.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

try:
    import networkx as nx
except ImportError:
    nx = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default path for persisting the graph
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "knowledge_graph.json"


class KnowledgeGraph:
    """
    In-memory knowledge graph built from (head, relation, tail) triples.
    Persists to JSON for reuse across sessions.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = Path(persist_path) if persist_path else DEFAULT_GRAPH_PATH
        self.triples: List[Dict[str, str]] = []  # [{"head", "relation", "tail"}, ...]
        self._graph = None  # networkx DiGraph built from triples

    def add_triple(self, head: str, relation: str, tail: str) -> None:
        """Add a single triple. Head and tail are nodes, relation is edge label."""
        head = head.strip()
        tail = tail.strip()
        relation = relation.strip()
        if not head or not tail or not relation:
            return
        if len(head) > 200 or len(tail) > 200 or len(relation) > 200:
            return
        self.triples.append({"head": head, "relation": relation, "tail": tail})
        self._graph = None  # invalidate graph

    def add_triples(self, triples: List[Dict[str, str]]) -> None:
        """Add multiple triples."""
        for t in triples:
            h = t.get("head") or t.get("subject")
            r = t.get("relation") or t.get("predicate")
            tail = t.get("tail") or t.get("object")
            if h and r and tail:
                self.add_triple(str(h), str(r), str(tail))
        self._graph = None

    def _build_graph(self) -> "nx.DiGraph":
        """Build networkx graph from triples."""
        if nx is None:
            raise ImportError("networkx is required. Install with: pip install networkx")
        G = nx.DiGraph()
        for t in self.triples:
            h, r, tail = t["head"], t["relation"], t["tail"]
            G.add_edge(h, tail, relation=r)
        self._graph = G
        return G

    @property
    def graph(self) -> "nx.DiGraph":
        if self._graph is None and self.triples:
            self._build_graph()
        return self._graph or (nx.DiGraph() if nx else None)

    def get_entities(self) -> Set[str]:
        """Return all entity names (nodes)."""
        entities = set()
        for t in self.triples:
            entities.add(t["head"])
            entities.add(t["tail"])
        return entities

    def get_subgraph_triples(
        self,
        seed_entities: List[str],
        hops: int = 2,
        max_triples: int = 50
    ) -> List[Dict[str, str]]:
        """
        Get triples that are within `hops` steps from any seed entity.
        Returns list of triples for context.
        """
        if not self.triples or not seed_entities:
            return []

        G = self.graph
        if G is None or G.number_of_nodes() == 0:
            return []

        # Normalize seed names: try to match case-insensitive
        seed_set = set(s.strip().lower() for s in seed_entities if s and str(s).strip())
        node_to_lower = {n: n.lower() for n in G.nodes()}
        lower_to_node = {v: k for k, v in node_to_lower.items()}
        start_nodes = set()
        for s in seed_set:
            if s in lower_to_node:
                start_nodes.add(lower_to_node[s])
            else:
                for node in G.nodes():
                    if s in node.lower() or node.lower() in s:
                        start_nodes.add(node)
                        break

        if not start_nodes:
            # No match: return up to max_triples from all
            return self.triples[:max_triples]

        # BFS to get nodes within hops
        seen = set(start_nodes)
        frontier = list(start_nodes)
        for _ in range(hops):
            next_frontier = []
            for u in frontier:
                for v in G.successors(u):
                    if v not in seen:
                        seen.add(v)
                        next_frontier.append(v)
                for v in G.predecessors(u):
                    if v not in seen:
                        seen.add(v)
                        next_frontier.append(v)
            frontier = next_frontier
            if not frontier:
                break

        # Collect triples that have both head and tail in seen
        result = []
        for t in self.triples:
            if t["head"] in seen or t["tail"] in seen:
                result.append(t)
                if len(result) >= max_triples:
                    break
        return result

    def get_triples_for_query(
        self,
        seed_entities: List[str],
        query_text: str = "",
        hops: int = 2,
        max_triples: int = 30,
    ) -> List[Dict[str, str]]:
        """
        Get triples relevant to the query: first try entity-based subgraph,
        then fallback to triples containing query words, then sample.
        """
        if not self.triples:
            return []

        # 1) Entity-based subgraph
        if seed_entities:
            sub = self.get_subgraph_triples(seed_entities, hops=hops, max_triples=max_triples)
            if sub:
                return sub

        # 2) Fallback: triples containing any query word (case-insensitive)
        if query_text:
            words = set(re.findall(r"\w+", query_text.lower()))
            words.discard("")
            if words:
                result = []
                for t in self.triples:
                    h, r, tail = t["head"].lower(), t["relation"].lower(), t["tail"].lower()
                    if any(w in h or w in r or w in tail for w in words):
                        result.append(t)
                        if len(result) >= max_triples:
                            return result
                if result:
                    return result

        # 3) Sample of all triples
        return self.triples[:max_triples]

    def subgraph_to_text(self, triples: List[Dict[str, str]]) -> str:
        """Format triples as readable text for LLM context."""
        if not triples:
            return ""
        lines = []
        for t in triples:
            lines.append(f"- {t['head']} --[{t['relation']}]--> {t['tail']}")
        return "\n".join(lines)

    def save(self, path: Optional[str] = None) -> None:
        """Persist triples to JSON."""
        p = Path(path) if path else self.persist_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.triples, f, ensure_ascii=False, indent=0)
        logger.info(f"Knowledge graph saved: {len(self.triples)} triples to {p}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load triples from JSON. Returns True if loaded."""
        p = Path(path) if path else self.persist_path
        if not p.exists():
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                self.triples = json.load(f)
            self._graph = None
            logger.info(f"Knowledge graph loaded: {len(self.triples)} triples from {p}")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            return False

    def clear(self) -> None:
        """Clear all triples."""
        self.triples = []
        self._graph = None
        logger.info("Knowledge graph cleared")

    def __len__(self) -> int:
        return len(self.triples)


def extract_triples_with_llm(text: str, llm, max_triples: int = 15) -> List[Dict[str, str]]:
    """
    Use LLM to extract (subject, predicate, object) triples from text.
    Returns list of dicts with keys subject, predicate, object (or head, relation, tail).
    """
    if not text or not text.strip():
        return []

    # Truncate very long text to avoid token limits
    text_snippet = text.strip()[:3000]
    if len(text.strip()) > 3000:
        text_snippet += "..."

    prompt = f"""Extract knowledge triples from the following text. Each triple has:
- subject: an entity (person, concept, thing, process)
- predicate: the relationship between subject and object
- object: another entity or value

Output ONLY a JSON array of triples, no other text. Use keys "subject", "predicate", "object".
Example: [{{"subject": "Machine Learning", "predicate": "is a subset of", "object": "Artificial Intelligence"}}]

Text:
{text_snippet}

JSON array:"""

    try:
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content=prompt)]
        if hasattr(llm, "invoke"):
            result = llm.invoke(messages)
            out = result.content if hasattr(result, "content") else str(result)
        elif hasattr(llm, "predict"):
            out = llm.predict(prompt)
        else:
            return []

        # Parse JSON from response (handle markdown code block)
        out = out.strip()
        if "```" in out:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", out)
            if match:
                out = match.group(1).strip()
        # Find first [ and last ]
        start = out.find("[")
        end = out.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        out = out[start : end + 1]
        data = json.loads(out)
        if not isinstance(data, list):
            return []
        triples = []
        for item in data[:max_triples]:
            if isinstance(item, dict):
                s = item.get("subject") or item.get("head") or ""
                p = item.get("predicate") or item.get("relation") or ""
                o = item.get("object") or item.get("tail") or ""
                if s and p and o:
                    triples.append({"subject": str(s), "predicate": str(p), "object": str(o)})
        return triples
    except Exception as e:
        logger.debug(f"Triple extraction failed: {e}")
        return []


def extract_entities_from_query(query: str, llm) -> List[str]:
    """Extract key entities from the user query for graph lookup."""
    if not query or not query.strip():
        return []

    prompt = f"""List the key entities (concepts, topics, names, technical terms) in this question. Output as a JSON array of strings, nothing else.
Question: {query.strip()}

JSON array:"""

    try:
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content=prompt)]
        if hasattr(llm, "invoke"):
            result = llm.invoke(messages)
            out = result.content if hasattr(result, "content") else str(result)
        elif hasattr(llm, "predict"):
            out = llm.predict(prompt)
        else:
            return []

        out = out.strip()
        if "```" in out:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", out)
            if match:
                out = match.group(1).strip()
        start = out.find("[")
        end = out.rfind("]")
        if start == -1 or end == -1:
            return []
        data = json.loads(out[start : end + 1])
        if isinstance(data, list):
            return [str(x).strip() for x in data if x][:10]
        return []
    except Exception as e:
        logger.debug(f"Entity extraction failed: {e}")
        return []
