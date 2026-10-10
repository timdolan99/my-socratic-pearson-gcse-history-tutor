import json
import os
import re
import tomllib
from typing import Dict, List, Union
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# --- Key Retrieval ---
secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path) and "GOOGLE_API_KEY" not in os.environ:
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        os.environ["GOOGLE_API_KEY"] = secrets.get(
            "GOOGLE_API_KEY"
        ) or secrets.get("GEMINI_API_KEY", "")

# --- Dynamic Schema Definition ---
class SubjectSpec(BaseModel):
    course_title: str = Field(
        description="Overall course title, e.g. Edexcel GCSE History or OCR A-Level Computer Science"
    )
    subject: str = Field(description="Subject name, e.g. History or Computer Science")
    level: str = Field(description="GCSE or A-Level")
    target_turns: int = Field(
        description="Minimum target exchange turns: 5 for GCSE, 7 for A-Level"
    )
    topics: Dict[str, Union[List[str], Dict[str, List[str]]]] = Field(
        description="Hierarchical map of Components/Units to their respective subtopics"
    )


def process_corpus():
    syllabus_dir = "./syllabus"
    if not os.path.exists(syllabus_dir):
        print(f"⚠️ Directory {syllabus_dir} does not exist.")
        return

    pdf_files = sorted(
        [f for f in os.listdir(syllabus_dir) if f.lower().endswith(".pdf")]
    )
    if not pdf_files:
        print("⚠️ No PDFs found in ./syllabus/")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        embedding_function=embeddings, persist_directory="./chroma_db"
    )

    master_spec = {}

    for pdf in pdf_files:
        print(f"🔍 Processing {pdf}...")
        docs = PyPDFLoader(os.path.join(syllabus_dir, pdf)).load()

        # Read preliminary pages (up to 40) sequentially to capture the syllabus outline
        overview_docs = docs[: min(40, len(docs))]
        full_overview_text = "\n--- PAGE ---\n".join(
            [d.page_content for d in overview_docs]
        )

        prompt = f"""You are an expert curriculum data architect.
Analyze the following specification PDF text and extract the exact course hierarchy and metadata.

CRITICAL DIRECTIVES:
1. Extract the exact 'course_title' (e.g. "Edexcel GCSE History") and 'level' (e.g. "GCSE") from the text. DO NOT hardcode or default to any other subject.
2. Build a hierarchy under 'topics':
   - Tier 1: Component / Paper / Section / Unit name
   - Tier 2: Subtopic or key content point list

SPECIFICATION TEXT EXCERPT:
{full_overview_text[:12000]}
"""

        try:
            spec = llm.with_structured_output(SubjectSpec).invoke(prompt)
            
            # Dynamically construct course_id from extracted title
            clean_title = re.sub(r"[^a-zA-Z0-9_]", "", spec.course_title.lower().replace(" ", "_"))
            
            # Derive target turns dynamically if needed
            target_turns = 7 if "A-Level" in spec.level or "ALevel" in spec.level else 5

            master_spec.update({
                "course_id": clean_title,
                "course_title": spec.course_title,
                "subject": spec.subject,
                "level": spec.level,
                "target_turns": spec.target_turns or target_turns,
                "topics": spec.topics
            })
            print(f"   ✅ Successfully extracted {len(spec.topics)} main units for {spec.course_title}")
        except Exception as e:
            print(f"   ⚠️ Spec extraction error on {pdf}: {e}")

        # Index document chunks into ChromaDB
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["source_file"] = pdf
        
        for i in range(0, len(chunks), 100):
            vectorstore.add_documents(chunks[i:i + 100])
        print(f"   ✅ Indexed {len(chunks)} chunks into ChromaDB.")

    # Save dynamic course_spec.json
    with open("course_spec.json", "w", encoding="utf-8") as f:
        json.dump(master_spec, f, indent=2, ensure_ascii=False)
    print("\n🚀 Saved course_spec.json successfully!")


if __name__ == "__main__":
    process_corpus()