"""
AI PM Assistant - Streamlit Frontend

A professional AI-powered project management assistant with 4 core features:
1. Project Plan Generator
2. Meeting Notes → Action Items Extractor
3. Status Report Generator
4. PM Q&A Assistant (RAG-powered)
"""
import streamlit as st
from datetime import datetime
import traceback

from src.features import PMAssistant

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI PM Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Feature cards */
    .feature-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #155724;
    }
    
    /* Warning message */
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #856404;
    }
    
    /* Stats box */
    .stats-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'assistant' not in st.session_state:
        with st.spinner("🚀 Initializing AI PM Assistant..."):
            st.session_state.assistant = PMAssistant()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0

initialize_session_state()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🤖 AI PM Assistant")
    st.markdown("---")
    
    st.markdown("### 📊 Features")
    st.markdown("""
    - **Project Plan Generator**: Create comprehensive project plans
    - **Meeting Notes Parser**: Extract action items automatically
    - **Status Report Generator**: Transform notes into reports
    - **PM Q&A Assistant**: Answer PM questions with RAG
    """)
    
    st.markdown("---")
    
    # Usage Statistics
    st.markdown("### 📈 Usage Statistics")
    
    if st.button("🔄 Refresh Stats"):
        stats = st.session_state.assistant.get_usage_stats()
        
        st.metric("Total Tokens", f"{stats['total_tokens']:,}")
        st.metric("Total Cost", f"${stats['total_cost']:.4f}")
        st.metric("Generations", st.session_state.generation_count)
        
        if 'rag_chunks' in stats:
            st.markdown("**RAG Knowledge Base:**")
            st.write(f"📚 {stats['rag_chunks']} chunks")
            st.write(f"📖 {len(stats['rag_sources'])} sources")
    
    st.markdown("---")
    
    # Reset option
    if st.button("🗑️ Reset Usage Stats"):
        st.session_state.assistant.reset_usage_stats()
        st.session_state.generation_count = 0
        st.success("Stats reset!")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    Built with:
    - OpenAI GPT-4o-mini
    - ChromaDB (RAG)
    - Streamlit
    
    **Knowledge Base:**
    - PMBOK Guide
    - Agile Practice Guide
    - Scrum Guide
    """)

# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🤖 AI Project Manager Assistant</h1>', 
            unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
    Your intelligent assistant for project management tasks - powered by AI
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FEATURE TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Project Plan Generator",
    "📝 Meeting Notes Parser",
    "📊 Status Report Generator",
    "💬 PM Q&A Assistant"
])

# =============================================================================
# TAB 1: PROJECT PLAN GENERATOR
# =============================================================================

with tab1:
    st.markdown("## 📋 Project Plan Generator")
    st.markdown("Generate comprehensive project plans from brief descriptions.")
    
    with st.form("project_plan_form"):
        st.markdown("### Project Description")
        
        project_description = st.text_area(
            "Describe your project (2-3 sentences minimum):",
            height=150,
            placeholder="Example: Build a mobile app for tracking personal fitness goals. "
                       "The app should allow users to log workouts, track nutrition, and "
                       "set fitness goals. Target launch in 4 months with a team of 5 developers."
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_plan = st.form_submit_button(
                "🚀 Generate Project Plan",
                use_container_width=True
            )
        with col2:
            max_tokens_plan = st.selectbox("Length", [1500, 2000, 2500], index=1)
    
    if generate_plan:
        if len(project_description.strip()) < 20:
            st.error("⚠️ Please provide a more detailed project description (at least 2-3 sentences).")
        else:
            try:
                with st.spinner("🎯 Generating comprehensive project plan..."):
                    result = st.session_state.assistant.generate_project_plan(
                        project_description,
                        max_tokens=max_tokens_plan
                    )
                    st.session_state.generation_count += 1
                
                st.success("✅ Project plan generated successfully!")
                
                # Display result
                st.markdown("### 📋 Generated Project Plan")
                st.markdown(result['plan'])
                
                # Download button
                st.download_button(
                    label="📥 Download as Markdown",
                    data=result['plan'],
                    file_name=f"project_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Used", result['tokens_used'])
                with col2:
                    st.metric("Cost", f"${result['cost']:.4f}")
                with col3:
                    st.metric("Words", len(result['plan'].split()))
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Debug Info"):
                    st.code(traceback.format_exc())

# =============================================================================
# TAB 2: MEETING NOTES PARSER
# =============================================================================

with tab2:
    st.markdown("## 📝 Meeting Notes → Action Items")
    st.markdown("Extract action items automatically from meeting notes or transcripts.")
    
    with st.form("meeting_notes_form"):
        st.markdown("### Meeting Notes")
        
        meeting_notes = st.text_area(
            "Paste your meeting notes or transcript:",
            height=200,
            placeholder="Example:\n"
                       "Team sync - Jan 15\n"
                       "- Sarah will finalize the PRD by Friday\n"
                       "- John to investigate database performance issues\n"
                       "- Need to schedule review meeting with marketing\n"
                       "- Lisa finalizing mobile mockups by Wednesday"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            extract_items = st.form_submit_button(
                "🔍 Extract Action Items",
                use_container_width=True
            )
        with col2:
            max_tokens_notes = st.selectbox("Detail Level", [1000, 1500, 2000], index=1)
    
    if extract_items:
        if len(meeting_notes.strip()) < 50:
            st.error("⚠️ Please provide more detailed meeting notes.")
        else:
            try:
                with st.spinner("🔍 Extracting action items..."):
                    result = st.session_state.assistant.extract_action_items(
                        meeting_notes,
                        max_tokens=max_tokens_notes
                    )
                    st.session_state.generation_count += 1
                
                st.success("✅ Action items extracted successfully!")
                
                # Display result
                st.markdown("### ✅ Extracted Action Items")
                st.markdown(result['action_items'])
                
                # Download button
                st.download_button(
                    label="📥 Download as Markdown",
                    data=result['action_items'],
                    file_name=f"action_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
                # Show stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tokens Used", result['tokens_used'])
                with col2:
                    st.metric("Cost", f"${result['cost']:.4f}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Debug Info"):
                    st.code(traceback.format_exc())

# =============================================================================
# TAB 3: STATUS REPORT GENERATOR
# =============================================================================

with tab3:
    st.markdown("## 📊 Status Report Generator")
    st.markdown("Transform bullet points into professional status reports.")
    
    with st.form("status_report_form"):
        st.markdown("### Project Updates")
        
        col1, col2 = st.columns(2)
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["weekly", "monthly", "executive summary"],
                index=0
            )
        with col2:
            audience = st.selectbox(
                "Audience",
                ["team and stakeholders", "executives", "team only", "stakeholders only"],
                index=0
            )
        
        bullet_points = st.text_area(
            "Enter your project updates (bullet points or notes):",
            height=200,
            placeholder="Example:\n"
                       "- Completed user authentication module\n"
                       "- Payment integration 80% done\n"
                       "- Sprint velocity: 45 points\n"
                       "- Database migration scheduled for next weekend\n"
                       "- Fixed 12 bugs from QA"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_report = st.form_submit_button(
                "📊 Generate Status Report",
                use_container_width=True
            )
        with col2:
            max_tokens_report = st.selectbox("Length", [1000, 1500, 2000], index=1)
    
    if generate_report:
        if len(bullet_points.strip()) < 30:
            st.error("⚠️ Please provide more project updates.")
        else:
            try:
                with st.spinner("📊 Generating professional status report..."):
                    result = st.session_state.assistant.generate_status_report(
                        bullet_points,
                        report_type=report_type,
                        audience=audience,
                        max_tokens=max_tokens_report
                    )
                    st.session_state.generation_count += 1
                
                st.success("✅ Status report generated successfully!")
                
                # Display result
                st.markdown("### 📊 Generated Status Report")
                st.markdown(result['report'])
                
                # Download button
                st.download_button(
                    label="📥 Download as Markdown",
                    data=result['report'],
                    file_name=f"status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Used", result['tokens_used'])
                with col2:
                    st.metric("Cost", f"${result['cost']:.4f}")
                with col3:
                    st.metric("Words", len(result['report'].split()))
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Debug Info"):
                    st.code(traceback.format_exc())

# =============================================================================
# TAB 4: PM Q&A ASSISTANT
# =============================================================================

with tab4:
    st.markdown("## 💬 PM Q&A Assistant (RAG-Powered)")
    st.markdown("Ask questions about project management - answers based on PMBOK, Agile, and Scrum guides.")
    
    # Question input
    with st.form("qa_form"):
        question = st.text_input(
            "Ask a project management question:",
            placeholder="Example: What is the purpose of a sprint retrospective?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            ask_question = st.form_submit_button(
                "💬 Get Answer",
                use_container_width=True
            )
        with col2:
            top_k = st.selectbox("Context Chunks", [2, 3, 4, 5], index=1)
    
    if ask_question:
        if len(question.strip()) < 10:
            st.error("⚠️ Please ask a complete question.")
        else:
            try:
                with st.spinner("🔍 Searching knowledge base and generating answer..."):
                    result = st.session_state.assistant.answer_pm_question(
                        question,
                        top_k=top_k
                    )
                    st.session_state.generation_count += 1
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'sources': result['sources'],
                        'timestamp': datetime.now()
                    })
                
                st.success("✅ Answer generated!")
                
                # Display answer
                st.markdown("### 💬 Answer")
                st.markdown(result['answer'])
                
                # Display sources
                if result['sources']:
                    with st.expander("📚 Sources Used"):
                        for i, source in enumerate(result['sources'], 1):
                            st.markdown(f"**Source {i}:** {source['metadata']['source']} - "
                                      f"{source['metadata'].get('section', 'N/A')}")
                            st.markdown(f"> {source['text'][:200]}...")
                            st.markdown("---")
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tokens Used", result['tokens_used'])
                with col2:
                    st.metric("Cost", f"${result['cost']:.4f}")
                with col3:
                    st.metric("Sources Retrieved", 
                             len(result['sources']) if result['sources'] else 0)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Debug Info"):
                    st.code(traceback.format_exc())
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {chat['question'][:60]}..."):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.caption(f"Asked at: {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 AI PM Assistant | Built with Streamlit, OpenAI GPT-4o-mini, and ChromaDB</p>
    <p>💡 <strong>Tip:</strong> Use the sidebar to track your usage and costs</p>
</div>
""", unsafe_allow_html=True)