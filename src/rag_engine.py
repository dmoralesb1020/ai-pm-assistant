"""
RAG Engine - Retrieval-Augmented Generation using ChromaDB
"""
import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.utils import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine using ChromaDB for vector storage
    """
    
    def __init__(self, collection_name: str = "pm_knowledge"):
        """
        Initialize RAG engine
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.config = get_config()
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.config.VECTOR_STORE_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function (OpenAI embeddings)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.config.OPENAI_API_KEY,
            model_name=self.config.EMBEDDING_MODEL
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"RAG Engine initialized with collection: {collection_name}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Project Management knowledge base"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def load_documents(self, force_reload: bool = False) -> int:
        """
        Load documents from knowledge base directory
        
        Args:
            force_reload: If True, clear existing data and reload all documents
            
        Returns:
            Number of chunks loaded
        """
        # Check if collection already has documents
        count = self.collection.count()
        if count > 0 and not force_reload:
            logger.info(f"Collection already has {count} documents. Use force_reload=True to reload.")
            return count
        
        if force_reload:
            logger.info("Force reload: clearing existing collection...")
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
        
        # Load all .txt files from knowledge base
        knowledge_path = Path(self.config.KNOWLEDGE_BASE_PATH)
        txt_files = list(knowledge_path.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"No .txt files found in {knowledge_path}")
            return 0
        
        logger.info(f"Found {len(txt_files)} documents to process")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for file_path in txt_files:
            logger.info(f"Processing: {file_path.name}")
            
            # Read document
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunk document
            chunks = self._chunk_document(content, file_path.stem)
            
            # Prepare for ChromaDB
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path.stem}_chunk_{i}"
                all_chunks.append(chunk['text'])
                all_metadatas.append(chunk['metadata'])
                all_ids.append(chunk_id)
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i+batch_size]
            batch_metadata = all_metadatas[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_chunks,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            
            logger.info(f"Added batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
        
        total_chunks = len(all_chunks)
        logger.info(f"✅ Successfully loaded {total_chunks} chunks from {len(txt_files)} documents")
        
        return total_chunks
    
    def _chunk_document(
        self,
        content: str,
        source: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict]:
        """
        Split document into overlapping chunks
        
        Args:
            content: Document text
            source: Source document name
            chunk_size: Target size of each chunk (characters)
            overlap: Overlap between chunks (characters)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Split by sections (=== headers ===)
        sections = content.split('\n===')
        
        chunks = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Extract section title if present
            lines = section.split('\n')
            section_title = lines[0].strip('= ').strip() if lines else "Unknown"
            section_text = '\n'.join(lines[1:]) if len(lines) > 1 else section
            
            # If section is small enough, use as single chunk
            if len(section_text) <= chunk_size:
                chunks.append({
                    'text': section_text.strip(),
                    'metadata': {
                        'source': source,
                        'section': section_title,
                        'chunk_size': len(section_text)
                    }
                })
            else:
                # Split large sections into chunks with overlap
                for i in range(0, len(section_text), chunk_size - overlap):
                    chunk_text = section_text[i:i + chunk_size]
                    
                    # Don't create tiny chunks at the end
                    if len(chunk_text) < 100:
                        continue
                    
                    chunks.append({
                        'text': chunk_text.strip(),
                        'metadata': {
                            'source': source,
                            'section': section_title,
                            'chunk_index': i // (chunk_size - overlap),
                            'chunk_size': len(chunk_text)
                        }
                    })
        
        return chunks
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_source: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_source: Optional source filter (e.g., "pmbok_summary")
            
        Returns:
            List of dicts with keys: text, metadata, similarity
        """
        # Build filter if needed
        where_filter = None
        if filter_source:
            where_filter = {"source": filter_source}
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        # Format results
        retrieved = []
        for i in range(len(results['documents'][0])):
            retrieved.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None,
                'id': results['ids'][0][i]
            })
        
        logger.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
        
        return retrieved
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 3
    ) -> str:
        """
        Retrieve chunks and format as context for LLM
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            Formatted context string
        """
        chunks = self.retrieve(query, top_k=top_k)
        
        if not chunks:
            return "No relevant information found in the knowledge base."
        
        # Format as context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk['metadata'].get('source', 'unknown')
            section = chunk['metadata'].get('section', 'unknown')
            text = chunk['text']
            
            context_parts.append(
                f"[Source {i}: {source} - {section}]\n{text}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics
        
        Returns:
            Dict with collection stats
        """
        count = self.collection.count()
        
        # Get sample to show sources
        if count > 0:
            sample = self.collection.get(limit=min(count, 10))
            sources = set(m['source'] for m in sample['metadatas'])
        else:
            sources = set()
        
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'sources': list(sources),
            'embedding_model': self.config.EMBEDDING_MODEL
        }
    
    def reset(self):
        """Delete collection and recreate empty"""
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info("Collection reset complete")


def main():
    """Test/demo the RAG engine"""
    print("\n" + "="*60)
    print("RAG ENGINE DEMO")
    print("="*60)
    
    # Initialize
    rag = RAGEngine()
    
    # Load documents
    print("\n📚 Loading documents...")
    num_chunks = rag.load_documents(force_reload=False)
    print(f"✅ Loaded {num_chunks} chunks")
    
    # Show stats
    print("\n📊 Collection Stats:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    test_queries = [
        "What is a sprint retrospective?",
        "How to manage project risks?",
        "What are the Agile values?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = rag.retrieve(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Source: {result['metadata']['source']}")
            print(f"    Section: {result['metadata'].get('section', 'N/A')}")
            print(f"    Text preview: {result['text'][:100]}...")
    
    print("\n" + "="*60)
    print("✅ RAG ENGINE DEMO COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()