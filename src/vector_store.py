"""
Vector Store Management Module

This module manages the ChromaDB vector database for storing and retrieving embeddings.
"""

import os
import logging
from typing import List, Dict
from pathlib import Path
import json

# Get the project root directory (parent of src folder)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB vector database."""
    
    def __init__(self, db_path: str = None, collection_name: str = "rag_documents"):
        """
        Initialize vector store.
        
        Args:
            db_path: Path to ChromaDB database. If None, uses default ./chroma_db
            collection_name: Name of the collection
        """
        if not chromadb:
            raise ImportError(
                "chromadb not installed. "
                "Install with: pip install chromadb"
            )
        
        # Use project root chroma_db directory if no path specified
        if db_path is None:
            db_path = str(PROJECT_ROOT / "chroma_db")
        elif not os.path.isabs(db_path):
            # If relative path, make it relative to project root
            db_path = str(PROJECT_ROOT / db_path)
        
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]) -> None:
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            embeddings: List of embedding vectors
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        if not documents:
            logger.warning("No documents to add")
            return
        
        ids = []
        texts = []
        metadatas = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{i}_{hash(doc['text']) % 10000}"
            ids.append(doc_id)
            texts.append(doc['text'])
            metadatas.append(doc.get('metadata', {}))
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, embedding: List[float], k: int = 3, where: Dict = None) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            where: Optional Chroma metadata filter (e.g. {"source": "filename.txt"})
            
        Returns:
            List of similar documents with scores
        """
        try:
            kwargs = dict(
                query_embeddings=[embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            if where:
                kwargs["where"] = where
            results = self.collection.query(**kwargs)
            
            if not results or not results['documents'] or not results['documents'][0]:
                return []
            
            # Convert distances to similarity scores (cosine distance to similarity)
            documents = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Cosine distance to similarity: similarity = 1 - distance
                similarity = 1 - distance
                documents.append({
                    'text': doc,
                    'metadata': metadata,
                    'similarity': similarity
                })
            
            return documents
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection. Returns count 0 if collection was deleted."""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'document_count': count,
                'db_path': self.db_path
            }
        except Exception as e:
            logger.debug(f"Collection info unavailable (e.g. deleted): {str(e)}")
            return {'name': self.collection_name, 'count': 0, 'document_count': 0, 'db_path': self.db_path}

    def get_indexed_sources(self, limit: int = 5000) -> List[str]:
        """Return list of unique source filenames in the index. Returns [] if collection empty or deleted."""
        try:
            count = self.collection.count()
            if count == 0:
                return []
            data = self.collection.get(limit=limit, include=["metadatas"])
            if not data or not data.get("metadatas"):
                return []
            sources = set()
            for m in data["metadatas"]:
                if isinstance(m, dict) and m.get("source"):
                    sources.add(m["source"])
            return sorted(sources)
        except Exception as e:
            logger.debug(f"Indexed sources unavailable: {str(e)}")
            return []

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def reset(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Get all IDs and delete them
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error resetting store: {str(e)}")


if __name__ == "__main__":
    # Example usage
    store = VectorStore()
    
    # Check collection info
    info = store.get_collection_info()
    print(f"Collection info: {json.dumps(info, indent=2)}")
