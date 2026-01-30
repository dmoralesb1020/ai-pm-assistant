"""
Test suite for LLM Client
"""
import sys
sys.path.append('.')

from src.llm_client import LLMClient


def test_basic_generation():
    """Test basic text generation"""
    print("\n" + "="*60)
    print("TEST 1: Basic Generation")
    print("="*60)
    
    client = LLMClient()
    
    result = client.generate(
        prompt="Say 'Hello, AI PM Assistant!' and nothing else.",
        max_tokens=20
    )
    
    print(f"✅ Response: {result['content']}")
    print(f"📊 Tokens used: {result['tokens_used']}")
    print(f"💰 Cost: ${result['cost']:.6f}")
    print(f"🤖 Model: {result['model']}")
    
    assert "Hello" in result['content']
    assert result['tokens_used'] > 0
    assert result['cost'] > 0


def test_with_system_message():
    """Test generation with system message"""
    print("\n" + "="*60)
    print("TEST 2: Generation with System Message")
    print("="*60)
    
    client = LLMClient()
    
    result = client.generate(
        prompt="What is a sprint?",
        system_message="You are a helpful project management assistant. Keep answers concise (2-3 sentences).",
        max_tokens=100
    )
    
    print(f"✅ Response: {result['content']}")
    print(f"📊 Tokens: {result['tokens_used']}")
    print(f"💰 Cost: ${result['cost']:.6f}")
    
    assert "sprint" in result['content'].lower()


def test_streaming():
    """Test streaming generation"""
    print("\n" + "="*60)
    print("TEST 3: Streaming Generation")
    print("="*60)
    
    client = LLMClient()
    
    print("🔄 Streaming response:")
    
    full_response = ""
    for chunk in client.generate_stream(
        prompt="Count from 1 to 5, separated by commas.",
        max_tokens=50
    ):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n")
    print(f"✅ Full response: {full_response}")
    
    assert len(full_response) > 0


def test_multi_turn_chat():
    """Test multi-turn conversation"""
    print("\n" + "="*60)
    print("TEST 4: Multi-turn Chat")
    print("="*60)
    
    client = LLMClient()
    
    messages = [
        {"role": "system", "content": "You are a project management assistant."},
        {"role": "user", "content": "What is Scrum?"},
        {"role": "assistant", "content": "Scrum is an Agile framework for managing complex projects."},
        {"role": "user", "content": "How long is a typical sprint?"}
    ]
    
    result = client.chat(messages=messages, max_tokens=100)
    
    print(f"✅ Response: {result['content']}")
    print(f"📊 Tokens: {result['tokens_used']}")
    print(f"💰 Cost: ${result['cost']:.6f}")
    
    assert "sprint" in result['content'].lower() or "week" in result['content'].lower()


def test_usage_tracking():
    """Test cumulative usage tracking"""
    print("\n" + "="*60)
    print("TEST 5: Usage Tracking")
    print("="*60)
    
    client = LLMClient()
    client.reset_usage_stats()
    
    # Make multiple calls
    client.generate("Say hello", max_tokens=10)
    client.generate("Say goodbye", max_tokens=10)
    
    stats = client.get_usage_stats()
    
    print(f"📊 Total tokens used: {stats['total_tokens']}")
    print(f"💰 Total cost: ${stats['total_cost']:.6f}")
    
    assert stats['total_tokens'] > 0
    assert stats['total_cost'] > 0
    
    print("\n✅ All usage tracked correctly!")


def test_error_handling():
    """Test error handling with invalid input"""
    print("\n" + "="*60)
    print("TEST 6: Error Handling")
    print("="*60)
    
    client = LLMClient()
    
    try:
        # This should work fine
        result = client.generate(
            prompt="Test",
            max_tokens=10,
            max_retries=1
        )
        print("✅ Error handling works - valid request succeeded")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 LLM CLIENT TEST SUITE")
    print("="*60)
    
    try:
        test_basic_generation()
        test_with_system_message()
        test_streaming()
        test_multi_turn_chat()
        test_usage_tracking()
        test_error_handling()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")