# RAG Project - Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Error: "pip: command not found"
**Problem**: Python or pip not properly installed

**Solution**:
```bash
# Verify Python installation
python --version

# If not found, install Python from python.org
# Make sure "Add Python to PATH" is checked during installation
```

#### Error: "No module named 'langchain'"
**Problem**: Dependencies not installed or wrong virtual environment

**Solution**:
```bash
# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import langchain; print(langchain.__version__)"
```

#### Error: "venv: command not found"
**Problem**: Virtual environment module not available

**Solution**:
```bash
# Use python module directly
python -m venv venv

# Or use conda instead
conda create -n rag python=3.10
```

### PDF Loading Issues

#### Error: "No PDF files found in ./data"
**Problem**: PDF folder is empty or in wrong location

**Solution**:
```bash
# Make sure data folder exists
mkdir data

# Copy your PDFs there
# Windows
copy "C:\Users\YourName\Documents\file.pdf" data\

# macOS/Linux
cp ~/Documents/file.pdf data/
```

#### Error: "PyPDF2.utils.PdfReadError: startxref not found"
**Problem**: PDF is corrupted or not a valid PDF

**Solution**:
1. Try opening the PDF in Adobe Reader or your system PDF viewer
2. If it opens fine, try converting it to PDF format:
   - Export from original application as PDF
   - Use online converter (smallpdf.com, etc.)
3. Test with the sample PDF first:
   ```bash
   python -c "from src.pdf_loader import load_sample_documents; docs = load_sample_documents(); print(f'Loaded {len(docs)} samples')"
   ```

#### Error: "No text extracted from PDF"
**Problem**: PDF might be scanned image or encrypted

**Solution**:
1. For scanned PDFs, you need OCR:
   ```bash
   pip install pdf2image pytesseract
   ```

2. For encrypted PDFs, try unlocking them:
   - Use online tool to remove password
   - Or use specialized library

3. Try with a different PDF first

### Embedding Issues

#### Error: "Unable to download model from HuggingFace"
**Problem**: Network issue or HuggingFace timeout

**Solution**:
```bash
# Try again (often temporary)
python src/embeddings.py

# If persistent, download model manually
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Or use different model
# all-mpnet-base-v2
# distiluse-base-multilingual-cased-v2
```

#### Error: "CUDA out of memory"
**Problem**: GPU memory exceeded with large embeddings

**Solution**:
```python
# Use CPU instead of GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Or reduce batch size
embeddings = generator.embed_texts(texts, batch_size=8)  # smaller batch
```

#### Error: "No module named 'sentence_transformers'"
**Problem**: sentence-transformers not installed

**Solution**:
```bash
pip install sentence-transformers
```

### Vector Database Issues

#### Error: "chromadb.errors.InvalidArgumentError"
**Problem**: ChromaDB database corrupted or invalid

**Solution**:
```bash
# Delete corrupted database
rm -rf chroma_db/  # macOS/Linux
rmdir /s chroma_db  # Windows

# Re-create from scratch
python src/vector_store.py
```

#### Error: "Connection refused" or database locked
**Problem**: Database already in use or permission issue

**Solution**:
```bash
# Make sure no other process is using ChromaDB
# Close all Python instances

# Check file permissions
chmod 755 chroma_db  # macOS/Linux

# Restart Python kernel in Jupyter
# Kernel → Restart & Clear Output
```

#### Error: "Collection not found"
**Problem**: Collection was deleted or database reset

**Solution**:
```bash
# Re-add documents to vector store
from src.rag_chain import RAGPipeline
rag = RAGPipeline()
rag.load_documents()  # This recreates the collection
```

### OpenAI API Issues

#### Error: "Invalid API key"
**Problem**: OpenAI API key is missing, invalid, or expired

**Solution**:
1. Get your API key:
   - Go to https://platform.openai.com/api-keys
   - Create new secret key if needed
   - Copy the FULL key

2. Add to .env:
   ```env
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx
   ```

3. Make sure .env file is in project root:
   ```bash
   # Check file exists
   ls .env  # macOS/Linux
   dir .env  # Windows
   ```

4. Test the key:
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
   ```

#### Error: "Rate limit exceeded"
**Problem**: Too many API requests

**Solution**:
```bash
# Wait a few minutes and try again
# Add delay between requests
import time
time.sleep(2)  # Wait 2 seconds between calls

# Reduce concurrent requests
# Use batch processing instead of individual calls
```

#### Error: "Insufficient quota"
**Problem**: No API credits remaining

**Solution**:
1. Add payment method:
   - Go to https://platform.openai.com/account/billing/overview
   - Add credit card
   - Set usage limit

2. Use local models instead (no cost):
   ```env
   EMBEDDINGS_TYPE=local
   LLM_TYPE=fake
   ```

#### Error: "Connection timed out"
**Problem**: OpenAI API server not responding

**Solution**:
```bash
# Check OpenAI status
# https://status.openai.com

# Try again after a few minutes
# Add retry logic
import time
max_retries = 3
for i in range(max_retries):
    try:
        response = llm.predict(prompt)
        break
    except Exception as e:
        if i < max_retries - 1:
            time.sleep(2**i)  # Exponential backoff
        else:
            raise
```

### Jupyter Notebook Issues

#### Error: "kernel is not responding"
**Problem**: Jupyter kernel crashed or hung

**Solution**:
```bash
# Restart the kernel
# Kernel → Restart

# Or restart Jupyter completely
# Stop Jupyter: Ctrl+C
# Start again: jupyter notebook
```

#### Error: "No module named" in notebook
**Problem**: Package not installed in notebook environment

**Solution**:
```python
# Install directly in notebook
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "package_name"])

# Or activate correct virtual environment
# Select Python Interpreter → Select venv
```

#### Error: "Connection failed" in JupyterHub
**Problem**: Connection between notebook and kernel lost

**Solution**:
1. Restart kernel: Kernel → Restart & Clear Output
2. Close browser tab and reopen
3. Stop and restart Jupyter:
   ```bash
   Ctrl+C  # Stop Jupyter
   jupyter notebook  # Start again
   ```

### Performance Issues

#### Problem: Embeddings generation is slow
**Solution**:
```python
# Use smaller model
generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

# Increase batch size
embeddings = generator.embed_texts(texts, batch_size=64)

# Use GPU if available
# Install CUDA and cuDNN
```

#### Problem: Retrieval is slow
**Solution**:
```python
# Reduce document count
# Use smaller chunks
# Optimize vector database
store.collection.delete(old_ids)  # Remove old documents

# Use faster similarity search
results = store.search(embedding, k=5)  # Smaller k value
```

#### Problem: Out of memory
**Solution**:
```bash
# Check available memory
free -h  # macOS/Linux
wmic logicaldisk get name,freespace  # Windows

# Close unnecessary applications
# Reduce batch size
# Use smaller documents
```

### Configuration Issues

#### Error: "No such file or directory: '.env'"
**Problem**: .env file not in project root

**Solution**:
```bash
# Create .env from example
cp .env.example .env

# Edit with your settings
# Make sure it's in project root directory
```

#### Error: "Invalid configuration value"
**Problem**: Wrong value format in .env

**Solution**:
```env
# Correct format:
OPENAI_API_KEY=sk-xxxxx  # No quotes
DEBUG=False  # True or False
K_RETRIEVE=3  # Number
TEMPERATURE=0.7  # Float between 0-1
```

### Data Issues

#### Problem: Getting poor quality answers
**Solutions**:
1. Check document quality:
   ```bash
   # Make sure PDFs have good text
   # Avoid scanned images without OCR
   # Use clear, well-structured documents
   ```

2. Improve chunking:
   ```python
   # Adjust chunk size
   chunks = loader.split_text(text, chunk_size=1000)  # Larger
   
   # Add overlap
   chunks = loader.split_text(text, chunk_size=500, overlap=100)
   ```

3. Use better embeddings:
   ```python
   # Use OpenAI embeddings instead of local
   generator = EmbeddingGenerator(embedding_type="openai")
   ```

4. Increase retrieval count:
   ```python
   # Get more context
   results = store.search(embedding, k=5)  # More documents
   ```

#### Problem: No relevant documents retrieved
**Solutions**:
1. Check if documents were added:
   ```python
   info = store.get_collection_info()
   print(info)  # Should show document_count > 0
   ```

2. Verify embedding quality:
   ```python
   # Test with very similar text
   embedding1 = generator.embed_text("machine learning")
   embedding2 = generator.embed_text("machine learning")
   similarity = generator.similarity(embedding1, embedding2)
   print(similarity)  # Should be close to 1.0
   ```

3. Check chunk content:
   ```python
   # Print retrieved documents
   results = store.search(embedding, k=3)
   for result in results:
       print(result['text'][:100])
   ```

## Debug Mode

Enable debug logging for more information:

```bash
# In .env file
DEBUG=True

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Getting Help

1. **Check the logs** - Look for error messages with context
2. **Google the error** - Most errors have solutions online
3. **Check documentation**:
   - LangChain: https://python.langchain.com/
   - ChromaDB: https://docs.trychroma.com/
   - Sentence-Transformers: https://www.sbert.net/
4. **Stack Overflow** - Search error messages
5. **GitHub Issues** - Check related projects
6. **Community Discord** - LangChain community is helpful

## Prevention Tips

1. **Use version control**: `git init` and commit your changes
2. **Create backups**: Save important files
3. **Test incrementally**: Don't add everything at once
4. **Read error messages carefully**: They often contain the solution
5. **Keep logs**: Save outputs for debugging
6. **Document your setup**: Note what works for you
