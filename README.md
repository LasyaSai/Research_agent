This project is a multi-agent Retrieval-Augmented Generation (RAG) assistant designed for academic research. It leverages autonomous agents to search, embed, retrieve, and synthesize information from ArXiv papers using semantic vector search and state-of-the-art NLP models. The system is built with Python, Streamlit, ChromaDB, Sentence Transformers, and HuggingFace Transformers.

**Features**

**Document Retrieval:** Searches ArXiv for relevant academic papers.

**Semantic Embedding:** Chunks and embeds paper content for efficient vector search.

**Vector Database:** Stores and queries semantic chunks using ChromaDB.

**Semantic Search: **Finds the most relevant information for a research query.

**Research Synthesis:** Summarizes retrieved content and provides citations.

**Streamlit UI:** Interactive web interface for research queries and results visualization.

**Architecture**
The system is composed of four main agents:

**Document Retriever Agent:** Searches ArXiv and fetches paper metadata and summaries.

**Embedding Agent:** Chunks paper text and creates semantic embeddings, storing them in a vector database.

**Query Agent:** Performs semantic search in the vector database to retrieve relevant chunks.

**Synthesis Agent:** Summarizes retrieved chunks and generates a comprehensive research synthesis with citations.

All agents are orchestrated by a central controller for seamless workflow.

**Tech Stack**
Python
Streamlit
arxiv (API)

chromadb (vector database)

sentence-transformers

transformers (HuggingFace)

facebook/bart-large-cnn (summarization model)
