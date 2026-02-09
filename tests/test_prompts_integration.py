"""
Integration tests for prompts with actual LLM calls
"""
import sys
sys.path.append('.')

from src.llm_client import LLMClient
from src.prompts import (
    get_project_plan_prompt,
    get_meeting_notes_prompt,
    get_status_report_prompt,
    get_pm_qa_prompt,
    PROJECT_PLAN_SYSTEM_PROMPT,
    MEETING_NOTES_SYSTEM_PROMPT,
    STATUS_REPORT_SYSTEM_PROMPT,
    PM_QA_SYSTEM_PROMPT
)


def test_project_plan_prompt():
    """Test project plan generation with LLM"""
    print("\n" + "="*60)
    print("TEST 1: Project Plan Generation")
    print("="*60)
    
    client = LLMClient()
    
    description = "Create a website redesign project for an e-commerce company"
    prompt = get_project_plan_prompt(description)
    
    result = client.generate(
        prompt=prompt,
        system_message=PROJECT_PLAN_SYSTEM_PROMPT,
        max_tokens=1500
    )
    
    response = result['content']
    
    print(f"\n📋 Generated Project Plan:\n")
    print(response[:800] + "...\n")
    
    # Verify structure
    assert "## Project Overview" in response or "Project Overview" in response
    assert "WBS" in response or "Work Breakdown" in response
    assert "Timeline" in response or "Milestone" in response
    assert "Risk" in response
    
    print(f"✅ Valid structure")
    print(f"📊 Tokens: {result['tokens_used']}, Cost: ${result['cost']:.4f}")


def test_meeting_notes_prompt():
    """Test meeting notes extraction with LLM"""
    print("\n" + "="*60)
    print("TEST 2: Meeting Notes Extraction")
    print("="*60)
    
    client = LLMClient()
    
    sample_notes = """
    Team sync - Jan 15, 2025
    
    Attendees: Sarah (PM), John (Dev), Lisa (Design)
    
    Discussion points:
    - Sprint 12 completed successfully, deployed to production
    - Sprint 13 planning: focus on payment integration
    - Sarah mentioned she'll send updated roadmap to stakeholders by EOW
    - John raised concern about database performance, will investigate this week
    - Lisa to finalize mobile mockups by Wednesday
    - Need to schedule review meeting with marketing team
    - Discussed hiring timeline - HR to provide update next week
    """
    
    prompt = get_meeting_notes_prompt(sample_notes)
    
    result = client.generate(
        prompt=prompt,
        system_message=MEETING_NOTES_SYSTEM_PROMPT,
        max_tokens=1000
    )
    
    response = result['content']
    
    print(f"\n📝 Extracted Action Items:\n")
    print(response[:600] + "...\n")
    
    # Verify extraction
    assert "Sarah" in response or "roadmap" in response
    assert "John" in response or "database" in response
    assert "Lisa" in response or "mockup" in response
    assert "Priority" in response or "priority" in response
    
    print(f"✅ Action items extracted")
    print(f"📊 Tokens: {result['tokens_used']}, Cost: ${result['cost']:.4f}")


def test_status_report_prompt():
    """Test status report generation with LLM"""
    print("\n" + "="*60)
    print("TEST 3: Status Report Generation")
    print("="*60)
    
    client = LLMClient()
    
    sample_bullets = """
    - Completed authentication module, deployed to staging
    - Payment gateway integration 80% done, blocked on vendor API documentation
    - New designer started this week, onboarding in progress
    - Sprint velocity: 45 story points (vs 50 target)
    - Fixed 12 bugs from QA testing
    - Database migration scheduled for next weekend
    - Upcoming: mobile app beta launch in 2 weeks
    """
    
    prompt = get_status_report_prompt(sample_bullets, report_type="weekly")
    
    result = client.generate(
        prompt=prompt,
        system_message=STATUS_REPORT_SYSTEM_PROMPT,
        max_tokens=1200
    )
    
    response = result['content']
    
    print(f"\n📊 Generated Status Report:\n")
    print(response[:700] + "...\n")
    
    # Verify structure
    assert "Executive Summary" in response or "Summary" in response
    assert "Accomplishment" in response or "Progress" in response
    assert "Risk" in response or "Issue" in response or "Blocker" in response
    
    print(f"✅ Professional format")
    print(f"📊 Tokens: {result['tokens_used']}, Cost: ${result['cost']:.4f}")


def test_pm_qa_prompt():
    """Test PM Q&A with context"""
    print("\n" + "="*60)
    print("TEST 4: PM Q&A with Context")
    print("="*60)
    
    client = LLMClient()
    
    context = """
    Sprint Retrospective:
    Purpose: Plan ways to increase quality and effectiveness
    Timebox: Maximum 3 hours for one-month Sprint
    Attendees: Scrum Team
    
    Topics:
    - What went well during Sprint?
    - What problems did we encounter?
    - How were problems solved?
    - What will we commit to improve in next Sprint?
    """
    
    question = "What is the purpose of a sprint retrospective and who should attend?"
    
    prompt = get_pm_qa_prompt(question, context)
    
    result = client.generate(
        prompt=prompt,
        system_message=PM_QA_SYSTEM_PROMPT,
        max_tokens=800
    )
    
    response = result['content']
    
    print(f"\n💬 Q&A Response:\n")
    print(response + "\n")
    
    # Verify quality
    assert "retrospective" in response.lower()
    assert "team" in response.lower() or "scrum team" in response.lower()
    
    print(f"✅ Relevant answer with context")
    print(f"📊 Tokens: {result['tokens_used']}, Cost: ${result['cost']:.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 PROMPT INTEGRATION TEST SUITE")
    print("="*60)
    print("\n⚠️  This test makes real API calls and will cost ~$0.02-0.05")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled")
        exit(0)
    
    try:
        test_project_plan_prompt()
        test_meeting_notes_prompt()
        test_status_report_prompt()
        test_pm_qa_prompt()
        
        print("\n" + "="*60)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()