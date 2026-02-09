"""
Test suite for RAG Engine
"""
import sys
sys.path.append('.')

from src.rag_engine import RAGEngine


def test_initialization():
    """Test RAG engine initialization"""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)
    
    rag = RAGEngine(collection_name="test_collection")
    
    stats = rag.get_stats()
    print(f"✅ Collection created: {stats['collection_name']}")
    print(f"📊 Initial chunks: {stats['total_chunks']}")
    
    # Cleanup
    rag.reset()


def test_document_loading():
    """Test document loading"""
    print("\n" + "="*60)
    print("TEST 2: Document Loading")
    print("="*60)
    
    rag = RAGEngine(collection_name="test_load")
    
    # Load documents
    num_chunks = rag.load_documents(force_reload=True)
    
    print(f"✅ Loaded {num_chunks} chunks")
    assert num_chunks > 0, "Should load at least one chunk"
    
    # Verify stats
    stats = rag.get_stats()
    print(f"📊 Sources found: {stats['sources']}")
    assert len(stats['sources']) == 1, "Should have 3 source files"
    
    # Cleanup
    rag.reset()


def test_retrieval():
    """Test semantic retrieval"""
    print("\n" + "="*60)
    print("TEST 3: Semantic Retrieval")
    print("="*60)
    
    rag = RAGEngine(collection_name="test_retrieval")
    rag.load_documents(force_reload=True)
    
    # Test query
    query = "What is Scrum?"
    results = rag.retrieve(query, top_k=3)
    
    print(f"Query: {query}")
    print(f"✅ Retrieved {len(results)} results")
    
    assert len(results) == 3, "Should retrieve 3 results"
    
    # Check result structure
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Source: {result['metadata']['source']}")
        print(f"  Preview: {result['text'][:80]}...")
        
        assert 'text' in result
        assert 'metadata' in result
        assert 'source' in result['metadata']
    
    # Cleanup
    rag.reset()


def test_context_formatting():
    """Test context formatting for LLM"""
    print("\n" + "="*60)
    print("TEST 4: Context Formatting")
    print("="*60)
    
    rag = RAGEngine(collection_name="test_context")
    rag.load_documents(force_reload=True)
    
    query = "What are Agile values?"
    context = rag.retrieve_with_context(query, top_k=2)
    
    print(f"Query: {query}")
    print(f"✅ Generated context ({len(context)} characters)")
    print(f"\nContext preview:\n{context[:300]}...\n")
    
    assert len(context) > 0
    assert "[Source" in context
    
    # Cleanup
    rag.reset()


def test_source_filtering():
    """Test filtering by source"""
    print("\n" + "="*60)
    print("TEST 5: Source Filtering")
    print("="*60)
    
    rag = RAGEngine(collection_name="test_filter")
    rag.load_documents(force_reload=True)
    
    # Query with filter
    query = "project management"
    results = rag.retrieve(query, top_k=3, filter_source="pmbok_summary")
    
    print(f"Query: {query} (filtered to PMBOK)")
    print(f"✅ Retrieved {len(results)} results")
    
    # All results should be from PMBOK
    for result in results:
        source = result['metadata']['source']
        print(f"  Source: {source}")
        assert source == "pmbok_summary", "Should only return PMBOK results"
    
    # Cleanup
    rag.reset()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 RAG ENGINE TEST SUITE")
    print("="*60)
    
    try:
        test_initialization()
        test_document_loading()
        test_retrieval()
        test_context_formatting()
        test_source_filtering()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()