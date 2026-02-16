"""
Test suite for PM Assistant features
"""
import sys
sys.path.append('.')

from src.features import PMAssistant


def test_project_plan_generator():
    """Test project plan generation"""
    print("\n" + "="*60)
    print("TEST 1: Project Plan Generator")
    print("="*60)
    
    assistant = PMAssistant()
    
    description = "Create a mobile app for tracking daily water intake"
    result = assistant.generate_project_plan(description)
    
    # Verify structure
    assert 'plan' in result
    assert 'tokens_used' in result
    assert 'cost' in result
    
    plan = result['plan']
    
    # Check for key sections
    assert any(keyword in plan for keyword in ['Overview', 'WBS', 'Work Breakdown'])
    assert any(keyword in plan for keyword in ['Timeline', 'Milestone'])
    assert 'Risk' in plan
    
    print(f"✅ Project plan generated")
    print(f"📊 Length: {len(plan)} chars")
    print(f"💰 Cost: ${result['cost']:.4f}")


def test_meeting_notes_parser():
    """Test action item extraction"""
    print("\n" + "="*60)
    print("TEST 2: Meeting Notes Parser")
    print("="*60)
    
    assistant = PMAssistant()
    
    notes = """
    Team meeting yesterday. John will complete the API documentation.
    Sarah mentioned she needs to review the designs by end of week.
    We should schedule a follow-up meeting for next Monday.
    """
    
    result = assistant.extract_action_items(notes)
    
    # Verify structure
    assert 'action_items' in result
    assert 'tokens_used' in result
    
    action_items = result['action_items']
    
    # Check extraction quality
    assert 'John' in action_items or 'API' in action_items
    assert 'Sarah' in action_items or 'design' in action_items
    
    print(f"✅ Action items extracted")
    print(f"💰 Cost: ${result['cost']:.4f}")


def test_status_report_generator():
    """Test status report generation"""
    print("\n" + "="*60)
    print("TEST 3: Status Report Generator")
    print("="*60)
    
    assistant = PMAssistant()
    
    bullets = """
    - Completed user login feature
    - Started work on dashboard
    - Fixed 5 bugs
    - Sprint velocity: 45 points
    """
    
    result = assistant.generate_status_report(bullets, report_type="weekly")
    
    # Verify structure
    assert 'report' in result
    assert 'tokens_used' in result
    
    report = result['report']
    
    # Check for professional formatting
    assert any(keyword in report for keyword in ['Summary', 'Accomplishment', 'Progress'])
    assert len(report) > 200  # Should be substantive
    
    print(f"✅ Status report generated")
    print(f"📊 Length: {len(report)} chars")
    print(f"💰 Cost: ${result['cost']:.4f}")


def test_pm_qa_assistant():
    """Test PM Q&A with RAG"""
    print("\n" + "="*60)
    print("TEST 4: PM Q&A Assistant (RAG)")
    print("="*60)
    
    assistant = PMAssistant()
    
    question = "What is Scrum?"
    result = assistant.answer_pm_question(question, top_k=2)
    
    # Verify structure
    assert 'answer' in result
    assert 'sources' in result
    assert 'tokens_used' in result
    
    answer = result['answer']
    sources = result['sources']
    
    # Check quality
    assert len(answer) > 100  # Should be substantive
    assert sources is not None and len(sources) > 0  # Should have retrieved context
    
    # Verify answer relevance
    assert 'scrum' in answer.lower() or 'agile' in answer.lower()
    
    print(f"✅ Question answered with RAG")
    print(f"📚 Sources retrieved: {len(sources)}")
    print(f"💰 Cost: ${result['cost']:.4f}")


def test_input_validation():
    """Test input validation"""
    print("\n" + "="*60)
    print("TEST 5: Input Validation")
    print("="*60)
    
    assistant = PMAssistant()
    
    # Test too-short inputs
    try:
        assistant.generate_project_plan("test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Project plan validation: {str(e)[:50]}...")
    
    try:
        assistant.extract_action_items("hi")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Meeting notes validation: {str(e)[:50]}...")
    
    try:
        assistant.generate_status_report("done")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Status report validation: {str(e)[:50]}...")
    
    try:
        assistant.answer_pm_question("?")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✅ Q&A validation: {str(e)[:50]}...")


def test_usage_tracking():
    """Test usage statistics tracking"""
    print("\n" + "="*60)
    print("TEST 6: Usage Tracking")
    print("="*60)
    
    assistant = PMAssistant()
    assistant.llm_client.reset_usage_stats()
    
    # Make a call
    assistant.generate_project_plan(
        "Build a simple todo app for team collaboration"
    )
    
    # Check stats
    stats = assistant.get_usage_stats()
    
    assert 'total_tokens' in stats
    assert 'total_cost' in stats
    assert stats['total_tokens'] > 0
    assert stats['total_cost'] > 0
    
    print(f"✅ Usage tracking working")
    print(f"📊 Tokens: {stats['total_tokens']}")
    print(f"💰 Cost: ${stats['total_cost']:.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 PM ASSISTANT FEATURE TEST SUITE")
    print("="*60)
    print("\n⚠️  This test makes real API calls (cost ~$0.15-0.20)")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled")
        exit(0)
    
    try:
        test_project_plan_generator()
        test_meeting_notes_parser()
        test_status_report_generator()
        test_pm_qa_assistant()
        test_input_validation()
        test_usage_tracking()
        
        print("\n" + "="*60)
        print("🎉 ALL FEATURE TESTS PASSED!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()