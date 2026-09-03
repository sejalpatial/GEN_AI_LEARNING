# RAG System - Document Ingestion Pipeline

## Overview

This notebook creates a Retrieval-Augmented Generation (RAG) system that:

1. Loads PDFs from `./docs_dir`
2. Splits them into chunks
3. Converts chunks to embeddings
4. Stores in Chroma vector database

## Flow

```
PDFs → Load → Split into chunks → Embeddings → Vector DB → Ready for queries
```

## Key Concepts

- **Embedding**: Text → Numbers (captures semantic meaning)
- **Vector DB**: Fast similarity search
- **Chunking**: Break large docs into ~2000-token pieces
- **RAG**: Retrieve relevant docs, then use LLM to generate answers

## Files

- `6.1_document_ingestion.ipynb` - Load & process documents
- `docs_dir/` - Place your PDF files here
- `vector_db/` - Stores embeddings (auto-created)

## Next Steps

1. Add PDFs to `docs_dir/`
2. Run notebook cells in order
3. Query the vector DB with questions
