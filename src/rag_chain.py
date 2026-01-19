"""
RAG Chain Implementation

This module implements the complete Retrieval-Augmented Generation pipeline.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv

try:
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
    from langchain.llms.fake import FakeListLLM
except ImportError:
    RetrievalQA = None

from pdf_loader import PDFLoader
from embeddings import EmbeddingGenerator
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class RAGPipeline:
    """Implements the RAG pipeline."""
    
    def __init__(
        self,
        embeddings_type: str = "local",
        llm_type: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embeddings_type: Type of embeddings ('local' or 'openai')
            llm_type: Type of LLM ('openai' or 'fake')
            model_name: Name of the LLM model
            temperature: Temperature for LLM generation
        """
        self.embeddings_type = embeddings_type
        self.embeddings = EmbeddingGenerator(embedding_type=embeddings_type)
        self.vector_store = VectorStore()
        self.llm = self._init_llm(llm_type, model_name, temperature)
        
        logger.info("RAG pipeline initialized")
    
    def _init_llm(self, llm_type: str, model_name: str, temperature: float):
        """Initialize the LLM."""
        if llm_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not set. Using fake LLM for demonstration."
                )
                return FakeListLLM(responses=["This is a demo response."])
            
            try:
                return OpenAI(
                    openai_api_key=api_key,
                    model_name=model_name,
                    temperature=temperature
                )
            except Exception as e:
                logger.error(f"Error initializing OpenAI LLM: {str(e)}")
                return FakeListLLM(responses=["This is a demo response."])
        
        else:  # fake
            return FakeListLLM(responses=[
                "This is a demo response from the fake LLM. "
                "In production, this would be a real response from OpenAI or another LLM."
            ])
    
    def load_documents(self, pdf_path: str = "./data") -> None:
        """
        Load PDF documents and create embeddings.
        
        Args:
            pdf_path: Path to directory containing PDFs
        """
        logger.info("Loading documents...")
        
        # Load PDFs
        loader = PDFLoader(pdf_path)
        documents = loader.load_all_pdfs()
        
        if not documents:
            logger.warning("No documents loaded")
            return
        
        # Split documents into chunks
        chunks = []
        for doc in documents:
            text_chunks = loader.split_text(doc['text'], chunk_size=500, overlap=50)
            for chunk in text_chunks:
                chunks.append({
                    'text': chunk,
                    'metadata': {'source': doc['filename']}
                })
        
        logger.info(f"Created {len(chunks)} text chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embeddings.embed_texts(texts)
        
        # Store in vector database
        logger.info("Storing in vector database...")
        self.vector_store.add_documents(chunks, embeddings)
        
        logger.info("Document loading complete")
    
    def query(self, question: str, k: int = 3) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: The user's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing query: {question}")
        
        # Generate embedding for question
        question_embedding = self.embeddings.embed_text(question)
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(question_embedding, k=k)
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            return {
                'answer': 'No relevant documents found in the knowledge base.',
                'sources': []
            }
        
        # Prepare context
        context = "\n\n".join([doc['text'] for doc in relevant_docs])
        
        # Generate answer using LLM
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            answer = self.llm.predict(prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = "Error generating response"
        
        return {
            'answer': answer,
            'sources': [
                {
                    'text': doc['text'][:100] + '...',
                    'similarity': doc['similarity'],
                    'metadata': doc['metadata']
                }
                for doc in relevant_docs
            ]
        }
    
    def get_pipeline_info(self) -> dict:
        """Get information about the RAG pipeline."""
        collection_info = self.vector_store.get_collection_info()
        return {
            'embeddings_type': self.embeddings_type,
            'embedding_dimension': self.embeddings.get_embedding_dimension(),
            'vector_store': collection_info,
            'llm_type': type(self.llm).__name__
        }


def main():
    """Main example usage."""
    # Initialize RAG pipeline
    rag = RAGPipeline(embeddings_type="local", llm_type="openai")
    
    # Load documents
    rag.load_documents()
    
    # Query
    question = "What is machine learning?"
    result = rag.query(question)
    
    print("\n" + "="*50)
    print(f"Question: {question}")
    print("="*50)
    print(f"Answer: {result['answer']}")
    print("\nSources:")
    for source in result['sources']:
        print(f"  - {source['text']}")
        print(f"    Similarity: {source['similarity']:.4f}")


if __name__ == "__main__":
    main()
