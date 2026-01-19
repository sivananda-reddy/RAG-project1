# RAG (Retrieval-Augmented Generation) Learning Project

A complete project for learning and implementing Retrieval-Augmented Generation with PDF document processing.

## Project Overview

This project demonstrates how to:
- Extract and process PDF documents
- Create vector embeddings from document content
- Store embeddings in a vector database (Chroma)
- Build a RAG system that retrieves relevant documents and generates responses
- Use LangChain for orchestration

## Project Structure

```
RAG project1/
├── src/                          # Source code
│   ├── pdf_loader.py            # PDF loading and processing
│   ├── embeddings.py            # Embedding generation
│   ├── vector_store.py          # Vector database operations
│   ├── rag_chain.py             # RAG pipeline
│   └── utils.py                 # Helper functions
├── data/                         # PDF documents directory
│   └── README.md                # Instructions for placing PDFs
├── notebooks/                    # Jupyter notebooks for learning
│   ├── 01_pdf_loading.ipynb     # PDF loading tutorial
│   ├── 02_embeddings.ipynb      # Embeddings exploration
│   ├── 03_vector_store.ipynb    # Vector database setup
│   └── 04_rag_complete.ipynb    # Complete RAG pipeline
├── docs/                         # Documentation
│   ├── RAG_BASICS.md            # RAG concept explanation
│   ├── SETUP.md                 # Setup instructions
│   └── TROUBLESHOOTING.md       # Common issues
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
# Copy the example env file
cp .env.example .env

# Add your API keys to .env (if using OpenAI)
# OPENAI_API_KEY=your_key_here
```

### 3. Prepare PDF Documents
- Place your PDF files in the `data/` directory
- Or use the provided sample PDFs for learning

### 4. Run Examples
```bash
# Option 1: Use Jupyter notebooks for interactive learning
jupyter notebook notebooks/

# Option 2: Run the RAG pipeline directly
python src/main.py
```

## Key Components

### PDF Loader (`src/pdf_loader.py`)
- Loads PDF documents
- Extracts text content
- Handles multi-page documents
- Supports OCR for scanned PDFs

### Embeddings (`src/embeddings.py`)
- Generates vector embeddings
- Uses sentence-transformers for local embeddings
- Supports OpenAI embeddings
- Handles batch processing

### Vector Store (`src/vector_store.py`)
- Manages Chroma vector database
- Stores and retrieves embeddings
- Performs similarity searches
- Manages collections

### RAG Chain (`src/rag_chain.py`)
- Orchestrates the RAG pipeline
- Retrieves relevant documents
- Generates responses using LLM
- Combines retrieval and generation

## Learning Path

1. **RAG Basics** - Read [RAG_BASICS.md](docs/RAG_BASICS.md)
2. **Setup** - Follow [SETUP.md](docs/SETUP.md)
3. **Notebooks** - Work through Jupyter notebooks in order
4. **Implementation** - Build your own RAG system

## Technologies Used

- **LangChain**: Framework for building LLM applications
- **ChromaDB**: Vector database for embeddings
- **Sentence-Transformers**: Local embedding model
- **PyPDF2**: PDF processing
- **OpenAI**: Optional LLM backend

## Configuration

### Using Local Models (Recommended for Learning)
- No API keys needed
- Uses free, open-source models
- Slower but good for understanding concepts

### Using OpenAI API
- Set `OPENAI_API_KEY` in `.env`
- Faster and better quality responses
- Requires API credits

## Common Commands

```bash
# Load PDFs and create embeddings
python src/pdf_loader.py

# Query the RAG system
python src/rag_chain.py "Your question here"

# Run Jupyter notebook
jupyter notebook notebooks/04_rag_complete.ipynb

# Check vector store contents
python src/vector_store.py --status
```

## Troubleshooting

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

## Next Steps

- [ ] Install dependencies
- [ ] Read RAG_BASICS.md
- [ ] Complete setup
- [ ] Run first notebook
- [ ] Add your own PDFs
- [ ] Build custom RAG application

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [RAG Research Papers](https://arxiv.org/search/?query=retrieval+augmented+generation)
- [Sentence-Transformers](https://www.sbert.net/)

## License

MIT License - Free to use for learning and projects
