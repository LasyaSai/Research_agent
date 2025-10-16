"""
Multi-Agent RAG Research Assistant
Autonomous agentic system for academic research synthesis using vector search
"""

import streamlit as st
import arxiv
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import time
from datetime import datetime

# ===========================
# AGENT 1: DOCUMENT RETRIEVER
# ===========================
class DocumentRetrieverAgent:
    def __init__(self):
        self.name = "Document Retriever Agent"
        
    def search_papers(self, query, max_results=5):
        """Search ArXiv for academic papers"""
        st.info(f"🔍 {self.name}: Searching ArXiv for '{query}'...")
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            papers.append({
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'published': result.published,
                'pdf_url': result.pdf_url,
                'entry_id': result.entry_id
            })
        
        st.success(f"✅ Retrieved {len(papers)} papers from ArXiv")
        return papers

# ===========================
# AGENT 2: EMBEDDING AGENT
# ===========================
class EmbeddingAgent:
    def __init__(self):
        self.name = "Embedding Agent"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name="research_papers",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.client.delete_collection(name="research_papers")
            self.collection = self.client.create_collection(
                name="research_papers",
                metadata={"hnsw:space": "cosine"}
            )
    
    def chunk_text(self, text, chunk_size=500):
        """Split text into semantic chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
    def process_papers(self, papers):
        """Create embeddings and store in ChromaDB"""
        st.info(f"🧠 {self.name}: Creating embeddings and storing in vector database...")
        
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        chunk_ids = []
        
        chunk_counter = 0
        for paper in papers:
            # Combine title and summary for better context
            full_text = f"{paper['title']}. {paper['summary']}"
            chunks = self.chunk_text(full_text)
            
            for chunk in chunks:
                embedding = self.model.encode(chunk).tolist()
                
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
                all_metadata.append({
                    'title': paper['title'],
                    'authors': ', '.join(paper['authors'][:3]),
                    'published': str(paper['published'])
                })
                chunk_ids.append(f"chunk_{chunk_counter}")
                chunk_counter += 1
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadata,
            ids=chunk_ids
        )
        
        st.success(f"✅ Created {len(all_chunks)} semantic chunks and stored embeddings (dim: 384)")
        return len(all_chunks)

# ===========================
# AGENT 3: QUERY AGENT
# ===========================
class QueryAgent:
    def __init__(self, collection, model):
        self.name = "Query Agent"
        self.collection = collection
        self.model = model
    
    def semantic_search(self, query, top_k=8):
        """Perform semantic search in vector database"""
        st.info(f"🔎 {self.name}: Performing semantic search for relevant chunks...")
        
        query_embedding = self.model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_chunks = []
        for i, doc in enumerate(results['documents'][0]):
            retrieved_chunks.append({
                'text': doc,
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        avg_similarity = 1 - (sum([c['distance'] for c in retrieved_chunks if c['distance']]) / len(retrieved_chunks))
        st.success(f"✅ Retrieved top {len(retrieved_chunks)} chunks | Avg similarity: {avg_similarity:.3f}")
        
        return retrieved_chunks, avg_similarity

# ===========================
# AGENT 4: SYNTHESIS AGENT
# ===========================
class SynthesisAgent:
    def __init__(self):
        self.name = "Synthesis Agent"
        # Using a smaller model for faster inference
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def generate_summary(self, chunks, query):
        """Generate comprehensive research summary from retrieved chunks"""
        st.info(f"✨ {self.name}: Generating comprehensive research summary...")
        
        # Combine all chunks
        combined_text = "\n\n".join([chunk['text'] for chunk in chunks[:5]])  # Use top 5
        
        # Generate summary
        max_length = min(len(combined_text.split()) // 2, 400)
        min_length = min(100, max_length - 50)
        
        summary = self.summarizer(
            combined_text[:4000],  # Limit input size
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        summary_text = summary[0]['summary_text']
        
        # Extract citations
        citations = {}
        for chunk in chunks:
            title = chunk['metadata']['title']
            if title not in citations:
                citations[title] = {
                    'authors': chunk['metadata']['authors'],
                    'published': chunk['metadata']['published'],
                    'count': 0
                }
            citations[title]['count'] += 1
        
        st.success(f"✅ Generated summary with {len(citations)} source citations")
        
        return summary_text, citations

# ===========================
# MAIN ORCHESTRATOR
# ===========================
class MultiAgentOrchestrator:
    def __init__(self):
        self.doc_retriever = DocumentRetrieverAgent()
        self.embedding_agent = EmbeddingAgent()
        self.query_agent = QueryAgent(
            self.embedding_agent.collection,
            self.embedding_agent.model
        )
        self.synthesis_agent = SynthesisAgent()
    
    def process_research_query(self, query, max_papers=5):
        """Orchestrate all agents to process research query"""
        start_time = time.time()
        
        # Agent 1: Retrieve papers
        papers = self.doc_retriever.search_papers(query, max_papers)
        
        # Agent 2: Create embeddings
        num_chunks = self.embedding_agent.process_papers(papers)
        
        # Agent 3: Semantic search
        retrieved_chunks, avg_similarity = self.query_agent.semantic_search(query)
        
        # Agent 4: Generate summary
        summary, citations = self.synthesis_agent.generate_summary(retrieved_chunks, query)
        
        processing_time = time.time() - start_time
        
        return {
            'summary': summary,
            'citations': citations,
            'papers': papers,
            'num_chunks': num_chunks,
            'retrieved_chunks': len(retrieved_chunks),
            'avg_similarity': avg_similarity,
            'processing_time': processing_time
        }

# ===========================
# STREAMLIT UI
# ===========================
def main():
    st.set_page_config(page_title="Multi-Agent RAG Research Assistant", layout="wide")
    
    st.title("🤖 Multi-Agent RAG Research Assistant")
    st.markdown("*Autonomous agents for academic research synthesis using vector search*")
    
    # Agent Architecture Display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**🔍 Agent 1**\nDocument Retriever")
    with col2:
        st.info("**🧠 Agent 2**\nEmbedding Agent")
    with col3:
        st.info("**🔎 Agent 3**\nQuery Agent")
    with col4:
        st.info("**✨ Agent 4**\nSynthesis Agent")
    
    st.markdown("---")
    
    # Query Input
    query = st.text_input(
        "Enter your research query:",
        placeholder="e.g., Transformer architecture in natural language processing",
        help="Ask any research question - agents will search, retrieve, and synthesize information"
    )
    
    max_papers = st.slider("Number of papers to retrieve:", 3, 10, 5)
    
    if st.button("🚀 Start Multi-Agent Analysis", type="primary"):
        if not query:
            st.warning("Please enter a research query")
            return
        
        with st.spinner("Agents processing..."):
            # Initialize orchestrator
            orchestrator = MultiAgentOrchestrator()
            
            # Process query
            results = orchestrator.process_research_query(query, max_papers)
            
            st.success(f"✅ All agents completed in {results['processing_time']:.2f}s")
            
            # Display Results
            st.markdown("---")
            st.header("📄 Generated Research Summary")
            st.write(results['summary'])
            
            # Metrics
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Processing Metrics")
                st.metric("Papers Retrieved", results['papers'].__len__())
                st.metric("Semantic Chunks Created", results['num_chunks'])
                st.metric("Top-K Chunks Retrieved", results['retrieved_chunks'])
                st.metric("Average Similarity Score", f"{results['avg_similarity']:.3f}")
                st.metric("Processing Time", f"{results['processing_time']:.2f}s")
            
            with col2:
                st.subheader("📚 Citations")
                for title, info in results['citations'].items():
                    with st.expander(f"{title[:80]}..."):
                        st.write(f"**Authors:** {info['authors']}")
                        st.write(f"**Published:** {info['published']}")
                        st.write(f"**Chunks used:** {info['count']}")
            
            # Show retrieved papers
            st.markdown("---")
            st.subheader("📖 Retrieved Papers")
            for i, paper in enumerate(results['papers'], 1):
                with st.expander(f"{i}. {paper['title']}"):
                    st.write(f"**Authors:** {', '.join(paper['authors'][:5])}")
                    st.write(f"**Published:** {paper['published']}")
                    st.write(f"**Summary:** {paper['summary'][:300]}...")
                    st.write(f"[View PDF]({paper['pdf_url']})")
    
    # Tech Stack
    st.markdown("---")
    st.caption("**Tech Stack:** Python • PyTorch • Transformers • ChromaDB • Sentence-Transformers • ArXiv API • Streamlit")

if __name__ == "__main__":
    main()
