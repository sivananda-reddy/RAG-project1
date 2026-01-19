"""
Vector Store Management Module

This module manages the ChromaDB vector database for storing and retrieving embeddings.
"""

import os
import logging
from typing import List, Dict
import json

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages ChromaDB vector database."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "rag_documents"):
        """
        Initialize vector store.
        
        Args:
            db_path: Path to ChromaDB database
            collection_name: Name of the collection
        """
        if not chromadb:
            raise ImportError(
                "chromadb not installed. "
                "Install with: pip install chromadb"
            )
        
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        settings = Settings(
            chroma_db_impl="duckdb",
            persist_directory=db_path,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(settings)
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
    
    def search(self, embedding: List[float], k: int = 3) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
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
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'document_count': count,
                'db_path': self.db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
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
