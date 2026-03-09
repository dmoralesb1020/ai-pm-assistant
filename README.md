# AI PM Assistant 🤖

An AI-powered project management assistant that automates administrative tasks using GPT-4 and RAG (Retrieval-Augmented Generation).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL_HERE)

## 🎯 Features

### 1. 📋 Project Plan Generator
Generate comprehensive project plans from brief descriptions including:
- Work Breakdown Structure (WBS)
- Timeline estimates and milestones
- Risk identification and mitigation strategies
- Resource allocation suggestions

### 2. 📝 Meeting Notes → Action Items
Automatically extract action items from meeting notes with:
- Task descriptions and owners
- Deadlines and priorities
- Context and dependencies

### 3. 📊 Status Report Generator
Transform bullet points into professional status reports:
- Executive summaries
- Accomplishments and progress
- Risks and blockers
- Next steps

### 4. 💬 PM Q&A Assistant (RAG-Powered)
Answer project management questions using authoritative sources:
- PMBOK Guide (7th Edition)
- Agile Practice Guide
- Scrum Guide
- Citations and sources provided

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-4o-mini
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: OpenAI text-embedding-3-small
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud

## 🚀 Live Demo

Try it here: [AI PM Assistant](YOUR_APP_URL_HERE)

## 📁 Project Structure
```
ai-pm-assistant/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── src/
│   ├── llm_client.py          # OpenAI API wrapper
│   ├── rag_engine.py          # Vector DB + retrieval logic
│   ├── prompts.py             # Prompt templates
│   ├── features.py            # 4 PM assistant features
│   ├── utils.py               # Configuration management
│   └── examples.py            # Example inputs
├── data/
│   ├── pm_knowledge/          # Knowledge base (PMBOK, Agile, Scrum)
│   └── vectorstore/           # Chroma DB storage (gitignored)
├── tests/                     # Unit tests
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## 💻 Local Development

### Prerequisites
- Python 3.10+
- OpenAI API key

### Setup

1. **Clone the repository**
```bash
   git clone https://github.com/dmoralesb1020/ai-pm-assistant.git
   cd ai-pm-assistant
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
```

5. **Run the app**
```bash
   streamlit run app.py
```

6. **Open your browser**
   - Navigate to `http://localhost:8501`

## ☁️ Deployment to Streamlit Cloud

### Quick Deploy

1. **Fork this repository** to your GitHub account

2. **Get OpenAI API Key**
   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Create an API key

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `ai-pm-assistant`
   - Main file path: `app.py`
   - Click "Advanced settings"
   - Add secrets:
```toml
     OPENAI_API_KEY = "sk-proj-your-key-here"
     OPENAI_MODEL = "gpt-4o-mini"
     EMBEDDING_MODEL = "text-embedding-3-small"
     MAX_TOKENS = "2000"
     TEMPERATURE = "0.7"
```
   - Click "Deploy"

4. **Wait for deployment** (2-3 minutes)

5. **Share your app URL!**

## 📊 Usage & Costs

### Token Usage
- Project Plan: ~1,500-2,000 tokens (~$0.02-0.03)
- Meeting Notes: ~1,000-1,500 tokens (~$0.01-0.02)
- Status Report: ~1,000-1,500 tokens (~$0.01-0.02)
- PM Q&A: ~500-1,000 tokens (~$0.008-0.015)

### Cost Estimates
- **Light usage** (10 generations/day): ~$0.20/day
- **Moderate usage** (50 generations/day): ~$1.00/day
- **Heavy usage** (200 generations/day): ~$4.00/day

💡 **Tip**: Monitor usage in the app sidebar

## 🧪 Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_llm_client.py
python tests/test_rag_engine.py
python tests/test_features.py
```

## 📝 Knowledge Base

The assistant's knowledge base includes:
- **PMBOK Guide**: Project management fundamentals, knowledge areas, process groups
- **Agile Practice Guide**: Agile values, principles, frameworks, and practices
- **Scrum Guide**: Scrum roles, events, artifacts, and rules

Total: ~1,200 lines of curated PM content, chunked into ~187 semantic segments.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Add more PM knowledge sources

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Diego Morales**
- GitHub: [@yourusername](https://github.com/dmoralesb1020)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/diego-morales-barrera-695515118/)






