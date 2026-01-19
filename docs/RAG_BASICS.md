# RAG (Retrieval-Augmented Generation) - Learning Basics

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It allows AI systems to:

1. **Search** a knowledge base for relevant information
2. **Retrieve** documents related to a query
3. **Generate** accurate responses based on the retrieved information

## Why Use RAG?

### Problems RAG Solves

1. **Outdated Information** - LLMs are trained on fixed data and don't know about recent events
2. **Hallucination** - LLMs sometimes generate false information
3. **Domain-Specific Knowledge** - You can add your own documents to the AI's knowledge
4. **Verifiability** - You know where the answer comes from

### Traditional LLM vs RAG

**Traditional LLM:**
```
User Question → LLM → Answer (based on training data)
```

**RAG System:**
```
User Question → Search Knowledge Base → Retrieve Relevant Docs → LLM + Context → Better Answer
```

## Core Components of RAG

### 1. **Document Loading**
- Read PDF files or documents
- Extract text from various formats
- Prepare documents for processing

### 2. **Text Chunking**
- Split large documents into smaller pieces
- Create overlapping chunks for context
- Each chunk becomes a searchable unit

### 3. **Embeddings**
- Convert text into vector representations
- Numbers that capture semantic meaning
- Similar texts have similar embeddings

Example:
```
"Machine learning" → [0.1, 0.5, 0.8, 0.3, ...]
"Deep learning"    → [0.1, 0.6, 0.7, 0.2, ...]
                       ↑ More similar!
```

### 4. **Vector Database**
- Store all embeddings
- Enable fast similarity search
- Examples: Chroma, Pinecone, Weaviate

### 5. **Retrieval**
- Convert user question to embedding
- Search vector database
- Find most similar documents
- Return top K results

### 6. **Generation**
- Take retrieved documents as context
- Provide context to LLM
- LLM generates answer based on context
- More accurate and grounded responses

## RAG Workflow

```
┌─────────────────┐
│  Load PDFs      │
└────────┬────────┘
         │
         ↓
┌──────────────────┐
│  Chunk Text      │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Create          │
│  Embeddings      │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Store in Vector │
│  Database        │
└────────┬─────────┘
         │
    ┌────┴─────────────┐
    │                  │
    ↓                  ↓
┌──────────┐    ┌─────────────┐
│ Question │    │ New PDFs?   │
└────┬─────┘    │ → Update DB │
     │          └─────────────┘
     ↓
┌──────────────────┐
│  Embed Question  │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Search Similar  │
│  Documents       │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Create Prompt   │
│  with Context    │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  LLM Generates   │
│  Answer          │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Return Answer   │
│  + Sources       │
└──────────────────┘
```

## Key Concepts Explained

### Embeddings
Embeddings are numerical representations of text that capture meaning:

- A sentence gets converted into a list of numbers (vector)
- Similar sentences have similar vectors
- Used to find "nearest neighbors" efficiently
- Example dimension: 384 or 768 numbers per text

### Vector Similarity
Methods to measure how similar two embeddings are:

1. **Cosine Similarity** - Most common, ranges from -1 to 1
2. **Euclidean Distance** - Straight-line distance
3. **Manhattan Distance** - City-block distance

### Chunking Strategy
Choosing the right chunk size affects performance:

- **Chunk Size** - Too small (loses context), too large (retrieves irrelevant info)
- **Overlap** - Helps preserve context at chunk boundaries
- **Typical range** - 256-1024 tokens per chunk

### K-Nearest Neighbors (K)
How many documents to retrieve:

- `K=1` - Most specific, might miss context
- `K=3-5` - Good balance (recommended)
- `K=10+` - More context, less specific

## RAG Implementation Steps

### Step 1: Prepare Your Documents
```python
from src.pdf_loader import PDFLoader

loader = PDFLoader("./data")
documents = loader.load_all_pdfs()
```

### Step 2: Create Embeddings
```python
from src.embeddings import Em
beddingGenerator

generator = EmbeddingGenerator(embedding_type="local")
embeddings = generator.embed_texts([doc['text'] for doc in documents])
```

### Step 3: Store in Vector Database
```python
from src.vector_store import VectorStore

store = VectorStore()
store.add_documents(documents, embeddings)
```

### Step 4: Query the System
```python
question_embedding = generator.embed_text("Your question?")
results = store.search(question_embedding, k=3)
```

### Step 5: Generate Answer
```python
context = "\n".join([doc['text'] for doc in results])
prompt = f"Context:\n{context}\n\nQuestion: Your question?"
answer = llm.predict(prompt)
```

## Common RAG Patterns

### Pattern 1: Question-Answering
```
Document DB → Find relevant docs → Generate answer
```

### Pattern 2: Summarization
```
Document DB → Retrieve all docs → Summarize key points
```

### Pattern 3: Fact Checking
```
Document DB → Verify claims → Return sources
```

### Pattern 4: Information Extraction
```
Document DB → Find specific info → Extract and structure
```

## Best Practices

1. **Document Quality**
   - Clean, well-formatted documents
   - Remove noise and irrelevant content
   - Proper pagination information

2. **Chunking**
   - Use semantic boundaries (paragraphs, sections)
   - Include overlaps for context
   - Keep chunks reasonably sized

3. **Embedding Model**
   - Local models (sentence-transformers) - fast, free, private
   - OpenAI models - more accurate, but costs money
   - Choose based on your needs

4. **Retrieval**
   - Tune K value based on your use case
   - Consider similarity threshold
   - Implement fallback strategies

5. **Generation**
   - Provide clear context to LLM
   - Include source attribution
   - Handle no-result cases gracefully

## Common Issues and Solutions

### Issue: No Relevant Documents Found
- **Solution**: Check document quality, adjust K, verify embeddings

### Issue: Poor Answer Quality
- **Solution**: Better documents, larger chunks, better LLM model

### Issue: Slow Retrieval
- **Solution**: Use smaller embeddings, optimize vector DB, add indexing

### Issue: Hallucinations Still Occurring
- **Solution**: Provide more context, use better LLM, enforce source citation

## Next Steps for Learning

1. Read [SETUP.md](SETUP.md) for installation
2. Run [01_pdf_loading.ipynb](../notebooks/01_pdf_loading.ipynb)
3. Explore [02_embeddings.ipynb](../notebooks/02_embeddings.ipynb)
4. Build your own RAG system with your documents

## Resources for Further Learning

- **LangChain Documentation**: https://python.langchain.com/
- **ChromaDB Guide**: https://docs.trychroma.com/
- **Embeddings Overview**: https://www.sbert.net/
- **Original RAG Paper**: https://arxiv.org/abs/2005.11401

## Key Takeaways

✓ RAG combines retrieval and generation for better AI responses
✓ Embeddings enable semantic similarity search
✓ Vector databases make retrieval efficient
✓ You can augment LLMs with your own knowledge
✓ RAG is more accurate and verifiable than raw LLMs
