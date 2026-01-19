# RAG Project Setup Guide

## Prerequisites

- **Python 3.8+** installed
- **pip** package manager
- **Git** (optional, for version control)
- **4GB+ RAM** for running local models
- **Internet connection** for downloading models

## Installation Steps

### 1. Create Python Environment

Using venv (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Using conda:
```bash
conda create -n rag python=3.10
conda activate rag
```

### 2. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 3. Setup Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your settings
# Only set OPENAI_API_KEY if you have an OpenAI account
```

### 4. Prepare Data

Create a `data` folder and add your PDF files:

```
RAG project1/
├── data/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ... (your PDFs here)
└── ...
```

## Verification

### Check Installation

```bash
python -c "import langchain; print('LangChain OK')"
python -c "import chromadb; print('ChromaDB OK')"
python -c "import sentence_transformers; print('Sentence-Transformers OK')"
```

### Run Example

```bash
# Test pdf loader
python src/pdf_loader.py

# Test embeddings
python src/embeddings.py

# Test vector store
python src/vector_store.py
```

## Configuration Options

### Local Setup (No API Key Needed)

```env
# .env file
EMBEDDINGS_TYPE=local
LLM_TYPE=fake
DEBUG=False
```

**Pros:**
- Free to use
- No API calls
- Good for learning
- Private (stays on your computer)

**Cons:**
- Slower response time
- Lower quality answers
- Limited to demo responses

### OpenAI Setup (Recommended for Production)

1. **Get API Key**
   - Go to https://platform.openai.com/api-keys
   - Create new secret key
   - Copy the key

2. **Configure .env**
   ```env
   OPENAI_API_KEY=sk-your-key-here
   EMBEDDINGS_TYPE=openai
   LLM_TYPE=openai
   LLM_MODEL=gpt-3.5-turbo
   TEMPERATURE=0.7
   ```

3. **Add Credit**
   - Go to https://platform.openai.com/account/billing/overview
   - Add payment method
   - Set usage limits for safety

## Learning Path

### Day 1: Basics
- [ ] Read `docs/RAG_BASICS.md`
- [ ] Install dependencies
- [ ] Run `notebooks/01_pdf_loading.ipynb`

### Day 2: Embeddings
- [ ] Run `notebooks/02_embeddings.ipynb`
- [ ] Understand embedding concepts
- [ ] Experiment with different models

### Day 3: Vector Database
- [ ] Run `notebooks/03_vector_store.ipynb`
- [ ] Learn ChromaDB operations
- [ ] Practice similarity search

### Day 4: Complete RAG
- [ ] Run `notebooks/04_rag_complete.ipynb`
- [ ] Build end-to-end system
- [ ] Query your documents

### Day 5+: Experimentation
- [ ] Add your own documents
- [ ] Tune parameters
- [ ] Build custom applications

## Running Jupyter Notebooks

### Start Jupyter

```bash
# Make sure your virtual environment is activated
jupyter notebook
```

This opens a browser window with Jupyter. Navigate to the `notebooks` folder.

### Common Jupyter Commands

- **Run cell**: Shift + Enter
- **Run all cells**: Cell → Run All
- **Insert cell**: Insert → Cell Below
- **Delete cell**: Edit → Delete Cells
- **Clear output**: Kernel → Restart & Clear Output

## Troubleshooting

### Issue: "ModuleNotFoundError"

```bash
# Make sure virtual environment is activated
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "No module named 'chromadb'"

```bash
# Install specific version
pip install chromadb==0.4.21
```

### Issue: "OPENAI_API_KEY not found"

- Make sure `.env` file exists in project root
- Check spelling: `OPENAI_API_KEY`
- Make sure you added your actual key (not placeholder)

### Issue: "PDF not loading"

- Verify PDFs are in `data/` folder
- Check PDF file names don't have special characters
- Ensure PDFs are not corrupted
- Try opening in a PDF reader first

### Issue: "Connection error" with OpenAI

- Check your API key is correct
- Verify internet connection
- Check OpenAI API status at https://status.openai.com
- Make sure you have API credits

### Issue: "Out of memory"

- Reduce batch size in embeddings generation
- Use smaller PDF documents
- Close other applications
- Use smaller embedding models

## Performance Tips

### Faster Embeddings
- Use smaller model: `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2`
- Batch processing: Process multiple texts at once
- GPU acceleration: Install CUDA for faster inference

### Faster Retrieval
- Reduce number of documents
- Use smaller chunks
- Optimize vector database indexing
- Cache common queries

### Better Results
- Use larger LLM models (gpt-4 vs gpt-3.5-turbo)
- Increase context window
- Use better embeddings (OpenAI vs local)
- Fine-tune on your specific domain

## File Structure Explanation

```
RAG project1/
├── src/                         # Main source code
│   ├── pdf_loader.py           # Load and process PDFs
│   ├── embeddings.py           # Generate embeddings
│   ├── vector_store.py         # Vector database operations
│   ├── rag_chain.py            # Main RAG pipeline
│   └── utils.py                # Utility functions
│
├── data/                        # Store your PDFs here
│
├── notebooks/                   # Interactive learning
│   ├── 01_pdf_loading.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_vector_store.ipynb
│   └── 04_rag_complete.ipynb
│
├── docs/                        # Documentation
│   ├── RAG_BASICS.md
│   ├── SETUP.md (this file)
│   └── TROUBLESHOOTING.md
│
├── .env.example                 # Configuration template
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

## Next Steps

1. ✓ Install dependencies
2. ✓ Setup `.env` file
3. ✓ Add PDF documents to `data/`
4. → Run first notebook: `01_pdf_loading.ipynb`
5. → Complete learning path

## Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Read error messages carefully
3. Search GitHub issues for similar problems
4. Check LangChain docs: https://python.langchain.com/
5. Join RAG community: https://discord.gg/langchain
