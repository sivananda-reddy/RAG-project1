"""
Optimized RAG Implementation with Persistent Embeddings

This module avoids re-embedding documents by checking if they already exist.
"""

import os
import logging
import hashlib
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.llms.fake import FakeListLLM
except ImportError:
    pass

from document_loader import DocumentLoader
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from knowledge_graph import (
    KnowledgeGraph,
    extract_triples_with_llm,
    extract_entities_from_query,
)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env from project root
env_path = PROJECT_ROOT / '.env'
load_dotenv(env_path)


class OptimizedRAGPipeline:
    """
    Optimized RAG pipeline that avoids re-embedding documents.
    Features:
    - Persistent embeddings (check before re-embedding)
    - Document fingerprinting
    - Incremental updates
    - Chat history management
    """
    
    def __init__(
        self,
        embeddings_type: str = "local",
        llm_type: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        use_knowledge_graph: bool = True,
    ):
        """Initialize optimized RAG pipeline (hybrid: vector + knowledge graph)."""
        self.embeddings_type = embeddings_type
        self.embeddings = EmbeddingGenerator(embedding_type=embeddings_type)
        self.vector_store = VectorStore()
        self.llm = self._init_llm(llm_type, model_name, temperature)
        self.chat_history = []
        # Env override: USE_KNOWLEDGE_GRAPH=false to disable
        kg_env = os.getenv("USE_KNOWLEDGE_GRAPH", "true").lower() in ("true", "1", "yes")
        self.use_knowledge_graph = use_knowledge_graph and kg_env
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.load()  # load existing graph if present
        
        mode = "hybrid (vector + knowledge graph)" if use_knowledge_graph else "vector only"
        logger.info(f"Optimized RAG pipeline initialized [{mode}]")
    
    def _init_llm(self, llm_type: str, model_name: str, temperature: float):
        """Initialize the LLM."""
        if llm_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set. Using demo mode.")
                return FakeListLLM(responses=["Demo mode: Set OPENAI_API_KEY for real responses."])
            
            try:
                api_base = os.getenv("OPENAI_API_BASE")
                llm_params = {
                    "openai_api_key": api_key,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_retries": 2
                }
                
                if api_base:
                    llm_params["openai_api_base"] = api_base
                    logger.info(f"Using custom API base: {api_base}")
                
                return ChatOpenAI(**llm_params)
            except Exception as e:
                logger.error(f"Error initializing LLM: {str(e)}")
                return FakeListLLM(responses=["Demo mode: LLM initialization failed."])
        
        return FakeListLLM(responses=["Demo mode."])
    
    def get_document_fingerprint(self, pdf_path: str) -> str:
        """
        Generate a unique fingerprint for a PDF document.
        Used to check if document has changed.
        """
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def check_embeddings_exist(self) -> bool:
        """Check if embeddings already exist in the vector store."""
        try:
            info = self.vector_store.get_collection_info()
            count = info.get('count', 0)
            return count > 0
        except Exception as e:
            logger.error(f"Error checking embeddings: {str(e)}")
            return False
    
    def load_documents_incremental(self, pdf_path: str = None, force_reload: bool = False) -> Dict:
        """
        Load documents with intelligent caching.
        
        Args:
            pdf_path: Path to PDF directory
            force_reload: Force re-embedding even if embeddings exist
            
        Returns:
            Status dictionary with operation details
        """
        if pdf_path is None:
            pdf_path = str(PROJECT_ROOT / "data")
        elif not os.path.isabs(pdf_path):
            pdf_path = str(PROJECT_ROOT / pdf_path)
        
        status = {
            'embeddings_existed': False,
            'documents_loaded': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'skipped': False,
            'message': ''
        }
        
        # Check if embeddings already exist
        if not force_reload and self.check_embeddings_exist():
            info = self.vector_store.get_collection_info()
            status['embeddings_existed'] = True
            status['skipped'] = True
            status['message'] = f"Using existing embeddings ({info.get('count', 0)} documents already indexed)"
            logger.info(status['message'])
            return status
        
        # When force_reload, clear existing embeddings so we replace with a fresh index
        if force_reload and self.check_embeddings_exist():
            logger.info("Clearing existing embeddings for full re-index...")
            self.vector_store.reset()
        
        logger.info("Loading and indexing documents (PDF, TXT, MD, PY)...")
        
        # Load all documents (PDF, TXT, MD) from data folder and subfolders
        loader = DocumentLoader(pdf_path)
        documents = loader.load_all_documents()
        
        if not documents:
            status['message'] = "No documents loaded (add PDF, TXT, MD, or PY files to data folder)"
            logger.warning(status['message'])
            return status
        
        status['documents_loaded'] = len(documents)
        file_names = [d.get('filename', '?') for d in documents]
        logger.info(f"Loaded {len(documents)} file(s): {', '.join(file_names)}")
        
        # Split documents into chunks
        chunks = []
        for doc in documents:
            text_chunks = loader.split_text(doc['text'], chunk_size=500, overlap=50)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'metadata': {
                        **doc.get('metadata', {}),
                        'chunk_id': i
                    }
                })
        
        status['chunks_created'] = len(chunks)
        logger.info(f"Created {len(chunks)} text chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embeddings.embed_texts(texts)
        
        status['embeddings_generated'] = len(embeddings)
        
        # Store in vector database
        logger.info("Storing in vector database...")
        self.vector_store.add_documents(chunks, embeddings)
        
        # Build or update knowledge graph (hybrid mode)
        if self.use_knowledge_graph:
            if force_reload:
                self.knowledge_graph.clear()
            if force_reload or len(self.knowledge_graph) == 0:
                self._build_knowledge_graph(chunks, status)
            self.knowledge_graph.save()
        
        status['message'] = f"Successfully indexed {len(documents)} documents ({len(chunks)} chunks)"
        logger.info(status['message'])
        
        return status
    
    def _build_knowledge_graph(self, chunks: List[Dict], status: Dict) -> None:
        """Extract triples from chunks and add to knowledge graph. Samples chunks to limit LLM calls."""
        max_chunks_for_kg = int(os.getenv("KG_MAX_CHUNKS", "40"))  # process up to 40 chunks
        step = max(1, len(chunks) // max_chunks_for_kg) if len(chunks) > max_chunks_for_kg else 1
        sampled = [chunks[i] for i in range(0, len(chunks), step)][:max_chunks_for_kg]
        logger.info(f"Building knowledge graph from {len(sampled)} chunks...")
        triples_count = 0
        for i, chunk in enumerate(sampled):
            text = chunk.get("text", "")
            if not text or len(text) < 50:
                continue
            try:
                triples = extract_triples_with_llm(text, self.llm, max_triples=12)
                if triples:
                    for t in triples:
                        self.knowledge_graph.add_triple(
                            t.get("subject", t.get("head", "")),
                            t.get("predicate", t.get("relation", "")),
                            t.get("object", t.get("tail", "")),
                        )
                    triples_count += len(triples)
            except Exception as e:
                logger.debug(f"Chunk {i} triple extraction failed: {e}")
        status['kg_triples'] = len(self.knowledge_graph)
        logger.info(f"Knowledge graph: {len(self.knowledge_graph)} triples")
    
    def _query_terms(self, question: str) -> set:
        """Extract significant words from the question for filename boosting (e.g. 'simple config' -> {'simple', 'config'})."""
        import re
        words = set(re.findall(r"[a-z0-9]{2,}", question.lower()))
        # Drop very common words
        stop = {"the", "and", "for", "what", "how", "does", "is", "are", "can", "you", "me", "this", "that", "from", "with", "explain", "describe", "tell", "about"}
        return words - stop

    def _retrieve_diverse(self, question_embedding: List[float], question: str = "", k: int = 8, max_candidates: int = 35, max_per_source: int = 3) -> List[Dict]:
        """
        Retrieve chunks from multiple documents. Boosts documents whose filename
        matches the query (e.g. 'simple config' -> simple-config-cli-guide.txt) so
        the right guide is used.
        """
        # Fetch more candidates so we have enough for diversity and query-term matches
        candidates = self.vector_store.search(question_embedding, k=max_candidates)
        if not candidates:
            return []
        # Group by source (filename)
        by_source = {}
        for doc in candidates:
            src = doc.get("metadata", {}).get("source", "unknown")
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(doc)
        # Boost sources whose filename contains query terms (e.g. "simple config" -> simple-config-cli-guide.txt first)
        query_terms = self._query_terms(question) if question else set()
        def source_score(src):
            low = src.lower()
            return sum(1 for w in query_terms if w in low)
        # Sort sources: those matching more query terms first, then by best similarity in that source
        ordered_sources = sorted(
            by_source.keys(),
            key=lambda s: (-source_score(s), -max((d["similarity"] for d in by_source[s]), default=0))
        )
        result = []
        for src in ordered_sources:
            docs = by_source[src]
            for doc in docs[:max_per_source]:
                result.append(doc)
                if len(result) >= k:
                    break
            if len(result) >= k:
                break
        # If we have fewer than k, add remaining by global similarity (no duplicate text)
        if len(result) < k:
            seen = {(d["text"][:150], d.get("metadata", {}).get("source", "")) for d in result}
            for doc in candidates:
                if len(result) >= k:
                    break
                key = (doc["text"][:150], doc.get("metadata", {}).get("source", ""))
                if key not in seen:
                    result.append(doc)
                    seen.add(key)
        return result[:k]

    def chat(self, question: str, k: int = 8) -> Dict:
        """
        Chat with the RAG system.
        
        Args:
            question: User question
            k: Number of chunks to use (diverse across documents)
            
        Returns:
            Response dictionary with answer and sources
        """
        logger.info(f"Processing query: {question}")
        
        # Generate embedding for question
        question_embedding = self.embeddings.embed_text(question)
        
        # Retrieve with diversity + query-term boost (e.g. "simple config" boosts simple-config-cli-guide.txt)
        relevant_docs = self._retrieve_diverse(
            question_embedding,
            question=question,
            k=k,
            max_candidates=35,
            max_per_source=3,
        )
        
        if not relevant_docs:
            return {
                'answer': 'No relevant documents found in the knowledge base.',
                'sources': [],
                'question': question
            }
        
        # ---- 1) VECTOR SEARCH CONTEXT ----
        vector_context = "Relevant passages from documents:\n" + "\n\n".join([doc['text'] for doc in relevant_docs])
        context_parts = [vector_context]
        graph_triples_used = 0
        
        # ---- 2) KNOWLEDGE GRAPH CONTEXT (hybrid) ----
        if self.use_knowledge_graph and len(self.knowledge_graph) > 0:
            entities = []
            try:
                entities = extract_entities_from_query(question, self.llm)
                if entities:
                    logger.info(f"Hybrid: query entities = {entities[:5]}")
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
            
            try:
                subgraph_triples = self.knowledge_graph.get_triples_for_query(
                    seed_entities=entities,
                    query_text=question,
                    hops=2,
                    max_triples=30,
                )
                if subgraph_triples:
                    graph_text = self.knowledge_graph.subgraph_to_text(subgraph_triples)
                    context_parts.append("Knowledge graph (related facts):\n" + graph_text)
                    graph_triples_used = len(subgraph_triples)
                    logger.info(f"Hybrid retrieval: vector={len(relevant_docs)} chunks, graph={graph_triples_used} triples")
                else:
                    logger.info("Hybrid: vector only (no graph triples matched)")
            except Exception as e:
                logger.warning(f"Knowledge graph retrieval failed: {e}")
        else:
            if self.use_knowledge_graph and len(self.knowledge_graph) == 0:
                logger.info("Hybrid: graph empty (run Update with New Documents to build graph); using vector only")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt with chat history context
        history_context = ""
        if self.chat_history:
            history_context = "\n\nPrevious conversation:\n"
            for exchange in self.chat_history[-3:]:  # Last 3 exchanges
                history_context += f"Q: {exchange['question']}\nA: {exchange['answer']}\n"
        
        prompt = f"""You are a helpful AI assistant. Use BOTH parts of the context below to answer the question:
1) Relevant passages from documents (vector search)
2) Knowledge graph facts (structured relations), if present

Context:
{context}
{history_context}

Question: {question}

Provide a clear, comprehensive answer using both the document passages and the knowledge graph facts. If the context doesn't contain enough information, say so.

Answer:"""
        
        # Generate answer
        try:
            from langchain.schema import HumanMessage
            messages = [HumanMessage(content=prompt)]
            
            if hasattr(self.llm, 'invoke'):
                result = self.llm.invoke(messages)
                answer = result.content if hasattr(result, 'content') else str(result)
            elif hasattr(self.llm, 'predict'):
                answer = self.llm.predict(prompt)
            elif hasattr(self.llm, 'responses'):
                answer = self.llm.responses[0]
            else:
                answer = "Unable to generate response"
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = f"Error generating response: {str(e)}"
        
        # Store in chat history
        self.chat_history.append({
            'question': question,
            'answer': answer,
            'sources': relevant_docs
        })
        
        return {
            'answer': answer,
            'sources': [
                {
                    'text': doc['text'][:200] + '...',
                    'similarity': doc['similarity'],
                    'metadata': doc['metadata']
                }
                for doc in relevant_docs
            ],
            'question': question,
            'retrieval_used': {
                'vector_chunks': len(relevant_docs),
                'graph_triples': graph_triples_used,
                'hybrid': graph_triples_used > 0,
            }
        }
    
    def clear_chat_history(self):
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics. Never returns stale indexed_sources when count is 0."""
        try:
            info = self.vector_store.get_collection_info()
        except Exception:
            info = {'count': 0, 'document_count': 0, 'name': '', 'db_path': ''}
        count = info.get('count') or info.get('document_count') or 0
        stats = {
            'embeddings_type': self.embeddings_type,
            'embedding_dimension': self.embeddings.get_embedding_dimension(),
            'vector_store': info,
            'chat_history_length': len(self.chat_history),
            'llm_type': type(self.llm).__name__,
        }
        if count == 0:
            stats['indexed_sources'] = []
        else:
            try:
                stats['indexed_sources'] = self.vector_store.get_indexed_sources()
            except Exception:
                stats['indexed_sources'] = []
        if self.use_knowledge_graph:
            stats['knowledge_graph_triples'] = len(self.knowledge_graph)
            stats['retrieval_mode'] = 'hybrid (vector + graph)'
        else:
            stats['retrieval_mode'] = 'vector only'
        return stats
    
    def clear_embeddings(self):
        """Clear all embeddings and knowledge graph (useful for re-indexing)."""
        self.vector_store.delete_collection()
        if self.use_knowledge_graph:
            self.knowledge_graph.clear()
            self.knowledge_graph.save()
        logger.info("Embeddings and knowledge graph cleared")


if __name__ == "__main__":
    # Example usage
    rag = OptimizedRAGPipeline(embeddings_type="local", llm_type="openai")
    
    # Load documents (will skip if already indexed)
    status = rag.load_documents_incremental()
    print(f"\nStatus: {status['message']}")
    
    # Get stats
    stats = rag.get_stats()
    print(f"\nPipeline Stats:")
    print(f"  Documents indexed: {stats['vector_store'].get('count', 0)}")
    print(f"  Chat history: {stats['chat_history_length']} messages")
    
    # Example query
    result = rag.chat("What is machine learning?")
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents retrieved")
