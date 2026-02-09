"""
Prompt Templates for AI PM Assistant Features

This module contains all prompt templates used across the application.
Centralizing prompts makes them easy to version, test, and iterate.
"""

# =============================================================================
# FEATURE 1: PROJECT PLAN GENERATOR
# =============================================================================

PROJECT_PLAN_SYSTEM_PROMPT = """You are an expert project manager with 15+ years of experience. 
You specialize in creating comprehensive, realistic project plans that follow PMI and Agile best practices.

Your task is to generate detailed project plans that include:
- Work Breakdown Structure (WBS)
- Timeline estimates
- Key milestones
- Risk identification
- Resource allocation suggestions

Be realistic with estimates. Consider dependencies and potential bottlenecks."""


def get_project_plan_prompt(description: str) -> str:
    """
    Generate prompt for project plan creation
    
    Args:
        description: Brief project description from user
        
    Returns:
        Formatted prompt string
    """
    return f"""Based on the following project description, create a comprehensive project plan:

PROJECT DESCRIPTION:
{description}

Generate a structured project plan with the following sections:

## Project Overview
- Project name and summary
- Objectives (3-5 key objectives)
- Success criteria

## Work Breakdown Structure (WBS)
- Break down into 4-6 major phases/components
- Each phase should have 3-5 key deliverables
- Use hierarchical numbering (1.0, 1.1, 1.2, etc.)

## Timeline & Milestones
- Estimated project duration
- 5-8 key milestones with target dates (use relative timing like "Week 2", "Month 1")
- Dependencies between milestones

## Resource Requirements
- Team roles needed (3-5 roles)
- Estimated effort for each role
- Required skills

## Risk Assessment
- Identify 4-6 potential risks
- For each risk, provide:
  * Risk description
  * Probability (High/Medium/Low)
  * Impact (High/Medium/Low)
  * Mitigation strategy

## Key Assumptions
- List 3-5 assumptions made in this plan

Format the output in clear Markdown with proper headings and bullet points.
Be specific and actionable. Avoid generic advice."""


# =============================================================================
# FEATURE 2: MEETING NOTES → ACTION ITEMS EXTRACTOR
# =============================================================================

MEETING_NOTES_SYSTEM_PROMPT = """You are an expert executive assistant specialized in parsing meeting notes 
and extracting actionable tasks with precision.

Your task is to:
1. Extract ALL action items from meeting notes
2. Identify task owners (if mentioned)
3. Extract or infer deadlines
4. Assign priority based on context and urgency
5. Clarify vague tasks into specific, actionable items

You are thorough and catch every commitment made in the meeting."""


def get_meeting_notes_prompt(meeting_notes: str) -> str:
    """
    Generate prompt for meeting notes parsing
    
    Args:
        meeting_notes: Raw meeting notes or transcript
        
    Returns:
        Formatted prompt string
    """
    return f"""Extract action items from the following meeting notes:

MEETING NOTES:
{meeting_notes}

For each action item, extract or infer:
- **Task**: Clear, specific description of what needs to be done
- **Owner**: Person responsible (if mentioned, otherwise mark as "Unassigned")
- **Deadline**: Due date (if mentioned, otherwise mark as "TBD")
- **Priority**: High/Medium/Low (infer from context - urgent items, blockers, and items with tight deadlines are High priority)
- **Context**: Brief note about why this is important (1 sentence)

FORMAT YOUR RESPONSE AS A MARKDOWN TABLE:

| # | Task | Owner | Deadline | Priority | Context |
|---|------|-------|----------|----------|---------|
| 1 | ... | ... | ... | ... | ... |

After the table, add a summary section:

## Meeting Summary
- Key decisions made
- Open questions or blockers
- Next meeting date/agenda (if mentioned)

Guidelines:
- Be specific: Transform "John will look into this" into "John will investigate database performance issues"
- Don't miss implied tasks: If someone says "I'll get back to you", that's an action item
- Mark dependencies clearly in the context
- If multiple people share a task, list all owners separated by commas"""


# =============================================================================
# FEATURE 3: STATUS REPORT GENERATOR
# =============================================================================

STATUS_REPORT_SYSTEM_PROMPT = """You are an experienced program manager who writes clear, 
professional status reports that executives love to read.

Your reports are:
- Concise but complete
- Focused on outcomes and impact
- Highlight both progress and risks
- Action-oriented with clear next steps

You know how to present information that helps stakeholders make decisions."""


def get_status_report_prompt(
    bullet_points: str,
    report_type: str = "weekly",
    audience: str = "team and stakeholders"
) -> str:
    """
    Generate prompt for status report creation
    
    Args:
        bullet_points: Raw notes/bullets about project progress
        report_type: Type of report (weekly, monthly, executive summary)
        audience: Target audience
        
    Returns:
        Formatted prompt string
    """
    return f"""Transform the following notes into a professional {report_type} status report 
for {audience}:

PROJECT NOTES:
{bullet_points}

Generate a well-structured status report with these sections:

## Executive Summary
- 2-3 sentence overview of current status
- Overall health indicator (🟢 On Track / 🟡 At Risk / 🔴 Off Track)

## Key Accomplishments
- 3-5 major accomplishments this period
- Focus on completed deliverables and outcomes
- Use metrics where available

## Work In Progress
- 3-5 active workstreams
- Current status of each
- Expected completion dates

## Upcoming Milestones
- Next 2-3 major milestones
- Target dates
- What needs to happen to achieve them

## Risks & Issues
- Top 2-3 risks or blockers
- Impact and mitigation plans
- Items needing stakeholder support/decisions

## Metrics (if available in notes)
- Key performance indicators
- Progress vs. targets
- Trends

## Next Steps
- 3-5 concrete actions for next period
- Who's responsible for each

Keep the tone professional but conversational. Use clear, jargon-free language.
Make it scannable with good formatting. Total length: 300-500 words."""


# =============================================================================
# FEATURE 4: PM Q&A ASSISTANT (RAG-Powered)
# =============================================================================

PM_QA_SYSTEM_PROMPT = """You are a knowledgeable project management consultant with expertise in 
PMBOK, Agile methodologies, and Scrum framework.

Your responses should:
- Be based primarily on the provided context from authoritative PM sources
- Be practical and actionable
- Include specific examples when helpful
- Cite which framework/guide you're referencing (PMBOK, Agile, Scrum)
- Be concise but comprehensive (2-4 paragraphs typically)

If the context doesn't contain enough information to fully answer the question:
- Answer what you can from the context
- Clearly state what's not covered
- Suggest where they might find more information"""


def get_pm_qa_prompt(question: str, context: str) -> str:
    """
    Generate prompt for PM Q&A with RAG context
    
    Args:
        question: User's question
        context: Retrieved context from knowledge base
        
    Returns:
        Formatted prompt string
    """
    return f"""Answer the following project management question using the provided context.

CONTEXT FROM PM KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}

Instructions:
1. Base your answer primarily on the context provided
2. Structure your answer clearly with headings if appropriate
3. Include specific examples or scenarios when helpful
4. Mention which framework/guide your answer comes from (e.g., "According to the Scrum Guide...")
5. If the context doesn't fully answer the question, acknowledge this and answer what you can
6. Keep your answer practical and actionable
7. Use formatting (bold, bullets) to make it scannable

Provide a helpful, accurate response (2-5 paragraphs)."""


def get_pm_qa_prompt_no_context(question: str) -> str:
    """
    Generate prompt when no relevant context is found
    
    Args:
        question: User's question
        
    Returns:
        Formatted prompt string
    """
    return f"""The user asked this project management question:

QUESTION:
{question}

Unfortunately, no relevant information was found in the knowledge base for this specific question.

Provide a helpful response that:
1. Acknowledges that this specific information isn't in your knowledge base
2. Offers general guidance if you have relevant PM knowledge on the topic
3. Suggests where they might find authoritative information (e.g., "You might want to consult the PMBOK Guide chapter on...")
4. Asks clarifying questions if the question is too broad

Keep it brief (1-2 paragraphs) and helpful."""


# =============================================================================
# VALIDATION & HELPER PROMPTS
# =============================================================================

def get_input_validation_prompt(user_input: str, feature: str) -> str:
    """
    Validate and clean user input before processing
    
    Args:
        user_input: Raw input from user
        feature: Which feature is requesting validation
        
    Returns:
        Validation prompt
    """
    return f"""Analyze this user input for the {feature} feature:

INPUT:
{user_input}

Determine if this input is:
1. Valid and sufficient for the task
2. Too vague or lacking detail
3. Contains any concerning or inappropriate content

Respond in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0-100,
    "issues": ["list of issues if any"],
    "suggestion": "suggestion for user if input needs improvement"
}}"""


# =============================================================================
# PROMPT TEMPLATES REGISTRY
# =============================================================================

PROMPT_TEMPLATES = {
    "project_plan": {
        "system": PROJECT_PLAN_SYSTEM_PROMPT,
        "generator": get_project_plan_prompt
    },
    "meeting_notes": {
        "system": MEETING_NOTES_SYSTEM_PROMPT,
        "generator": get_meeting_notes_prompt
    },
    "status_report": {
        "system": STATUS_REPORT_SYSTEM_PROMPT,
        "generator": get_status_report_prompt
    },
    "pm_qa": {
        "system": PM_QA_SYSTEM_PROMPT,
        "generator": get_pm_qa_prompt,
        "no_context_generator": get_pm_qa_prompt_no_context
    }
}


def get_prompt_template(feature: str):
    """
    Get prompt template for a feature
    
    Args:
        feature: Feature name (project_plan, meeting_notes, status_report, pm_qa)
        
    Returns:
        Dict with system prompt and generator function
    """
    if feature not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown feature: {feature}. Available: {list(PROMPT_TEMPLATES.keys())}")
    
    return PROMPT_TEMPLATES[feature]


# =============================================================================
# TESTING & EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """Test prompt generation"""
    
    print("\n" + "="*60)
    print("PROMPT TEMPLATES DEMO")
    print("="*60)
    
    # Test 1: Project Plan
    print("\n📋 PROJECT PLAN PROMPT:")
    print("-" * 60)
    prompt = get_project_plan_prompt(
        "Build a mobile app for tracking personal fitness goals"
    )
    print(prompt[:500] + "...\n")
    
    # Test 2: Meeting Notes
    print("📝 MEETING NOTES PROMPT:")
    print("-" * 60)
    sample_notes = """
    Discussed Q1 roadmap. Sarah will finalize the PRD by Friday.
    John mentioned database performance issues - needs investigation.
    Team agreed to increase sprint length to 3 weeks starting next month.
    """
    prompt = get_meeting_notes_prompt(sample_notes)
    print(prompt[:500] + "...\n")
    
    # Test 3: Status Report
    print("📊 STATUS REPORT PROMPT:")
    print("-" * 60)
    sample_bullets = """
    - Completed user authentication module
    - Payment integration delayed due to API issues
    - 3 new team members onboarded
    - Sprint velocity: 42 points (target was 50)
    """
    prompt = get_status_report_prompt(sample_bullets)
    print(prompt[:500] + "...\n")
    
    # Test 4: PM Q&A
    print("💬 PM Q&A PROMPT:")
    print("-" * 60)
    sample_context = "A sprint is a time-boxed iteration in Scrum, typically 2-4 weeks..."
    prompt = get_pm_qa_prompt(
        question="How long should a sprint be?",
        context=sample_context
    )
    print(prompt[:500] + "...\n")
    
    print("="*60)
    print("✅ All prompt templates generated successfully!")
    print("="*60 + "\n")