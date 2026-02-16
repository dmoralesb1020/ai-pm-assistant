"""
PM Assistant Features

This module implements the 4 core features:
1. Project Plan Generator
2. Meeting Notes → Action Items
3. Status Report Generator
4. PM Q&A Assistant (RAG-powered)
"""
import logging
from typing import Dict, Optional

from src.llm_client import LLMClient
from src.rag_engine import RAGEngine
from src.prompts import (
    PROJECT_PLAN_SYSTEM_PROMPT,
    MEETING_NOTES_SYSTEM_PROMPT,
    STATUS_REPORT_SYSTEM_PROMPT,
    PM_QA_SYSTEM_PROMPT,
    get_project_plan_prompt,
    get_meeting_notes_prompt,
    get_status_report_prompt,
    get_pm_qa_prompt,
    get_pm_qa_prompt_no_context
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMAssistant:
    """
    AI Project Management Assistant with 4 core features
    """
    
    def __init__(self):
        """Initialize PM Assistant with LLM client and RAG engine"""
        logger.info("Initializing PM Assistant...")
        
        self.llm_client = LLMClient()
        self.rag_engine = None  # Lazy load for Q&A feature
        
        logger.info("✅ PM Assistant ready")
    
    def _ensure_rag_initialized(self):
        """Lazy initialization of RAG engine (only when needed)"""
        if self.rag_engine is None:
            logger.info("Initializing RAG engine for Q&A...")
            self.rag_engine = RAGEngine()
            
            # Load documents if not already loaded
            count = self.rag_engine.collection.count()
            if count == 0:
                logger.info("Loading knowledge base...")
                self.rag_engine.load_documents()
            else:
                logger.info(f"Knowledge base already loaded ({count} chunks)")
    
    # =========================================================================
    # FEATURE 1: PROJECT PLAN GENERATOR
    # =========================================================================
    
    def generate_project_plan(
        self,
        description: str,
        max_tokens: int = 2000
    ) -> Dict:
        """
        Generate comprehensive project plan from brief description
        
        Args:
            description: Brief project description (2-3 sentences minimum)
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict with keys:
                - plan: Generated project plan (markdown)
                - tokens_used: Token count
                - cost: Cost in USD
                - metadata: Additional info
        """
        logger.info(f"Generating project plan for: '{description[:50]}...'")
        
        # Validate input
        if len(description.strip()) < 20:
            raise ValueError(
                "Description too short. Please provide at least 2-3 sentences "
                "describing your project."
            )
        
        # Generate prompt
        prompt = get_project_plan_prompt(description)
        
        # Call LLM
        result = self.llm_client.generate(
            prompt=prompt,
            system_message=PROJECT_PLAN_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        logger.info(f"✅ Project plan generated ({result['tokens_used']} tokens)")
        
        return {
            'plan': result['content'],
            'tokens_used': result['tokens_used'],
            'cost': result['cost'],
            'metadata': {
                'feature': 'project_plan_generator',
                'input_length': len(description),
                'model': result['model']
            }
        }
    
    # =========================================================================
    # FEATURE 2: MEETING NOTES → ACTION ITEMS
    # =========================================================================
    
    def extract_action_items(
        self,
        meeting_notes: str,
        max_tokens: int = 1500
    ) -> Dict:
        """
        Extract action items from meeting notes or transcript
        
        Args:
            meeting_notes: Raw meeting notes or transcript
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict with keys:
                - action_items: Formatted action items (markdown table)
                - summary: Meeting summary
                - tokens_used: Token count
                - cost: Cost in USD
                - metadata: Additional info
        """
        logger.info(f"Extracting action items from notes ({len(meeting_notes)} chars)")
        
        # Validate input
        if len(meeting_notes.strip()) < 50:
            raise ValueError(
                "Meeting notes too short. Please provide more detailed notes "
                "(at least a few sentences)."
            )
        
        # Generate prompt
        prompt = get_meeting_notes_prompt(meeting_notes)
        
        # Call LLM
        result = self.llm_client.generate(
            prompt=prompt,
            system_message=MEETING_NOTES_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.5  # Lower temperature for more consistent extraction
        )
        
        logger.info(f"✅ Action items extracted ({result['tokens_used']} tokens)")
        
        return {
            'action_items': result['content'],
            'tokens_used': result['tokens_used'],
            'cost': result['cost'],
            'metadata': {
                'feature': 'meeting_notes_parser',
                'input_length': len(meeting_notes),
                'model': result['model']
            }
        }
    
    # =========================================================================
    # FEATURE 3: STATUS REPORT GENERATOR
    # =========================================================================
    
    def generate_status_report(
        self,
        bullet_points: str,
        report_type: str = "weekly",
        audience: str = "team and stakeholders",
        max_tokens: int = 1500
    ) -> Dict:
        """
        Transform bullet points into professional status report
        
        Args:
            bullet_points: Raw notes/bullets about project progress
            report_type: Type of report (weekly, monthly, executive summary)
            audience: Target audience
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict with keys:
                - report: Generated status report (markdown)
                - tokens_used: Token count
                - cost: Cost in USD
                - metadata: Additional info
        """
        logger.info(f"Generating {report_type} status report")
        
        # Validate input
        if len(bullet_points.strip()) < 30:
            raise ValueError(
                "Not enough content for a status report. Please provide more details "
                "about project progress."
            )
        
        # Generate prompt
        prompt = get_status_report_prompt(
            bullet_points=bullet_points,
            report_type=report_type,
            audience=audience
        )
        
        # Call LLM
        result = self.llm_client.generate(
            prompt=prompt,
            system_message=STATUS_REPORT_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        logger.info(f"✅ Status report generated ({result['tokens_used']} tokens)")
        
        return {
            'report': result['content'],
            'tokens_used': result['tokens_used'],
            'cost': result['cost'],
            'metadata': {
                'feature': 'status_report_generator',
                'report_type': report_type,
                'audience': audience,
                'input_length': len(bullet_points),
                'model': result['model']
            }
        }
    
    # =========================================================================
    # FEATURE 4: PM Q&A ASSISTANT (RAG-POWERED)
    # =========================================================================
    
    def answer_pm_question(
        self,
        question: str,
        top_k: int = 3,
        max_tokens: int = 1000
    ) -> Dict:
        """
        Answer project management question using RAG with knowledge base
        
        Args:
            question: PM question from user
            top_k: Number of context chunks to retrieve
            max_tokens: Maximum tokens for response
            
        Returns:
            Dict with keys:
                - answer: Generated answer
                - sources: Retrieved source chunks
                - tokens_used: Token count
                - cost: Cost in USD
                - metadata: Additional info
        """
        logger.info(f"Answering PM question: '{question[:50]}...'")
        
        # Validate input
        if len(question.strip()) < 10:
            raise ValueError("Question too short. Please ask a complete question.")
        
        # Ensure RAG engine is initialized
        self._ensure_rag_initialized()
        
        # Retrieve relevant context
        retrieved_chunks = self.rag_engine.retrieve(question, top_k=top_k)
        
        if not retrieved_chunks:
            # No relevant context found - use fallback prompt
            logger.warning("No relevant context found in knowledge base")
            prompt = get_pm_qa_prompt_no_context(question)
            context_used = None
        else:
            # Format context for LLM
            context = self.rag_engine.retrieve_with_context(question, top_k=top_k)
            prompt = get_pm_qa_prompt(question, context)
            context_used = retrieved_chunks
            
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Call LLM
        result = self.llm_client.generate(
            prompt=prompt,
            system_message=PM_QA_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.6
        )
        
        logger.info(f"✅ Answer generated ({result['tokens_used']} tokens)")
        
        return {
            'answer': result['content'],
            'sources': context_used,
            'tokens_used': result['tokens_used'],
            'cost': result['cost'],
            'metadata': {
                'feature': 'pm_qa_assistant',
                'question_length': len(question),
                'chunks_retrieved': len(retrieved_chunks) if retrieved_chunks else 0,
                'model': result['model']
            }
        }
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_usage_stats(self) -> Dict:
        """
        Get cumulative usage statistics
        
        Returns:
            Dict with usage stats
        """
        llm_stats = self.llm_client.get_usage_stats()
        
        stats = {
            'total_tokens': llm_stats['total_tokens'],
            'total_cost': llm_stats['total_cost'],
            'llm_model': self.llm_client.config.OPENAI_MODEL
        }
        
        # Add RAG stats if initialized
        if self.rag_engine:
            rag_stats = self.rag_engine.get_stats()
            stats['rag_collection'] = rag_stats['collection_name']
            stats['rag_chunks'] = rag_stats['total_chunks']
            stats['rag_sources'] = rag_stats['sources']
        
        return stats
    
    def reset_usage_stats(self):
        """Reset usage tracking counters"""
        self.llm_client.reset_usage_stats()
        logger.info("Usage stats reset")


# =============================================================================
# DEMO & TESTING
# =============================================================================

def demo_all_features():
    """Demonstrate all 4 features"""
    print("\n" + "="*70)
    print("AI PM ASSISTANT - FEATURE DEMO")
    print("="*70)
    
    assistant = PMAssistant()
    
    # -------------------------------------------------------------------------
    # FEATURE 1: PROJECT PLAN
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("FEATURE 1: PROJECT PLAN GENERATOR")
    print("="*70)
    
    project_desc = """
    Build a customer feedback portal where users can submit feature requests,
    vote on existing requests, and track implementation status. Target launch
    in 3 months with a team of 4 developers.
    """
    
    result = assistant.generate_project_plan(project_desc.strip())
    
    print(f"\n📋 Generated Project Plan:\n")
    print(result['plan'][:1000] + "...\n")
    print(f"💰 Cost: ${result['cost']:.4f} | Tokens: {result['tokens_used']}")
    
    # -------------------------------------------------------------------------
    # FEATURE 2: MEETING NOTES
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("FEATURE 2: MEETING NOTES → ACTION ITEMS")
    print("="*70)
    
    meeting_notes = """
    Sprint Planning - Sprint 24
    Date: Jan 20, 2025
    Attendees: Sarah (PM), Mike (Tech Lead), Lisa (Designer), Team
    
    - Reviewed sprint 23 velocity: 48 points (target was 50)
    - Sprint 24 goal: Complete payment integration and mobile UI
    - Mike raised concern about API rate limits, will investigate this week
    - Sarah to send updated roadmap to executives by Friday
    - Lisa finalizing mobile mockups, needs feedback by Wednesday
    - Team agreed to include refactoring tasks: 8 story points
    - Discussed hiring: need to interview 2 backend candidates next week
    - John mentioned production bug in reporting module, fix needed urgently
    """
    
    result = assistant.extract_action_items(meeting_notes.strip())
    
    print(f"\n📝 Extracted Action Items:\n")
    print(result['action_items'][:800] + "...\n")
    print(f"💰 Cost: ${result['cost']:.4f} | Tokens: {result['tokens_used']}")
    
    # -------------------------------------------------------------------------
    # FEATURE 3: STATUS REPORT
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("FEATURE 3: STATUS REPORT GENERATOR")
    print("="*70)
    
    bullet_points = """
    - Completed user authentication module, deployed to production
    - Payment gateway integration 85% complete, waiting on vendor API docs
    - Mobile app beta testing started with 50 users, feedback positive
    - Fixed 15 bugs from QA, 3 critical issues remain
    - Sprint velocity trending up: 45 → 48 → 52 points
    - New designer onboarded, ramping up well
    - Database migration scheduled for next weekend (high risk)
    - Upcoming: feature freeze in 2 weeks for Q1 release
    """
    
    result = assistant.generate_status_report(
        bullet_points.strip(),
        report_type="weekly"
    )
    
    print(f"\n📊 Generated Status Report:\n")
    print(result['report'][:800] + "...\n")
    print(f"💰 Cost: ${result['cost']:.4f} | Tokens: {result['tokens_used']}")
    
    # -------------------------------------------------------------------------
    # FEATURE 4: PM Q&A
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("FEATURE 4: PM Q&A ASSISTANT (RAG-POWERED)")
    print("="*70)
    
    questions = [
        "What is the purpose of a sprint retrospective?",
        "How do I create a good work breakdown structure?",
        "What are the key Agile values?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        
        result = assistant.answer_pm_question(question, top_k=2)
        
        print(f"\n💬 Answer:\n{result['answer']}\n")
        
        if result['sources']:
            print(f"📚 Sources used:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['metadata']['source']} - {source['metadata'].get('section', 'N/A')}")
        
        print(f"💰 Cost: ${result['cost']:.4f} | Tokens: {result['tokens_used']}")
        print("-" * 70)
    
    # -------------------------------------------------------------------------
    # FINAL STATS
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("CUMULATIVE USAGE STATISTICS")
    print("="*70)
    
    stats = assistant.get_usage_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ ALL FEATURES DEMONSTRATED SUCCESSFULLY")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_all_features()