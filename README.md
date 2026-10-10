# 🎓 Socratic Tutor for History GCSE 

An adaptive Socratic tutoring system using LangGraph, LangChain, and Streamlit. This application dynamically ingests specification documents and guides students through computer science concepts using pedagogical scaffolding and safety-critical guardrails.

## 📂 Directory Structure

```text
msc-socratic-chatbot/
├── .streamlit/
│   └── secrets.toml
├── chroma_db/
├── syllabus/
│   └── GCSE_computer_science_Syllabus.pdf
├── app.py
├── ingest_corpus.py
├── requirements.txt
└── socratic_fsm.py
```

## 🚀 Quick Start Guide
1. Install Dependencies
Install all core libraries using the requirements file:

pip install -r requirements.txt

2. Configure the Gemini API Key
Instead of using terminal environment variables, this project utilizes native Streamlit secrets management.

Create a folder named .streamlit in the project root directory.

Inside that folder, create a file named secrets.toml.

Add your Gemini API key inside it exactly like this:

GOOGLE_API_KEY = "your-actual-gemini-api-key-here"

3. Ingest the Syllabus Document (Optional)
The vector database is already populated under chroma_db/. However, if you wish to re-initialize or update the corpus from the PDF folder, run:

python ingest_corpus.py

4. Launch the Tutor Workspace
Run the Streamlit local application server to begin the learning session:

python -m streamlit run app.py

4. Quantitative RAG Quality Matrix (evaluate_system_rag.py)
Benchmarks factual faithfulness (Groundedness) and Context Relevance under adversarial conditions using an LLM-as-a-Judge framework:

python evaluate_system_rag.py

