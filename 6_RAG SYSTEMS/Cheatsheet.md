## Quick Reference - RAG Pipeline

### What is each component?

| Component             | Purpose                | Input       | Output                      |
| --------------------- | ---------------------- | ----------- | --------------------------- |
| PyPDFLoader           | Extract text from PDFs | PDF file    | List of Document objects    |
| CharacterTextSplitter | Break into chunks      | Raw text    | Chunks (default 1000 chars) |
| HuggingFaceEmbeddings | Text → Numbers         | Text string | Vector (384 dimensions)     |
| Chroma                | Vector database        | Embeddings  | Fast similarity search      |

### Common Issues & Fixes

- **No PDFs found**: Check `docs_dir/` has `.pdf` files
- **Memory error**: Reduce chunk_size or use smaller PDFs
- **Slow embedding**: First run downloads model (~400MB)

### Key Parameters

- `chunk_size=1000` - Size of text chunks
- `chunk_overlap=200` - Overlap between chunks (for context)
- `collection_name` - Name of vector database collection
