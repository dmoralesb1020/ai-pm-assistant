# AI PM Assistant 🤖

An AI-powered project management assistant that automates administrative tasks using GPT-4 and RAG (Retrieval-Augmented Generation).

## 🎯 Features

- **Project Plan Generator**: Create structured project plans from brief descriptions
- **Meeting Notes Parser**: Extract action items from meeting transcripts
- **Status Report Generator**: Transform bullet points into professional status reports
- **PM Q&A Assistant**: Answer project management questions using RAG with PMBOK, Agile, and Scrum guides

## 🛠️ Tech Stack

- **LLM**: OpenAI GPT-4o-mini
- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **Frontend**: Streamlit
- **Deployment**: Streamlit Cloud

## 📁 Project Structure
```
ai-pm-assistant/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── src/
│   ├── llm_client.py          # OpenAI API wrapper
│   ├── rag_engine.py          # Vector DB + retrieval logic
│   ├── prompts.py             # Prompt templates
│   └── utils.py               # Helper functions
├── data/
│   ├── pm_knowledge/          # Text files for RAG
│   └── vectorstore/           # Chroma DB storage (gitignored)
└── tests/
    └── test_features.py       # Unit tests
```

## 🚀 Getting Started

_Coming soon - setup instructions will be added as the project develops_

## 📊 Development Status

- [x] Project setup
- [ ] RAG engine implementation
- [ ] Feature development
- [ ] Frontend development
- [ ] Deployment


