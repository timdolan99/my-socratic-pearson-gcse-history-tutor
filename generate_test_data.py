import json
import os
import random
from datetime import datetime, timedelta
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import streamlit as st

# --- 1. Database Connection ---
try:
    DB_URL = st.secrets["postgres"]["url"]
except Exception:
    DB_URL = os.getenv(
        "POSTGRES_URL", "postgresql://postgres:password@localhost:5432/postgres"
    )

engine = create_engine(DB_URL, echo=False)
Base = declarative_base()


# --- 2. ORM Schema ---
class ActivityLog(Base):
    __tablename__ = "activity_logs_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    subject = Column(Text, nullable=False)
    level = Column(Text, nullable=False)
    unit = Column(Text, nullable=False)
    subtopic = Column(Text, nullable=False)
    app_mode = Column(Text, nullable=False)
    score_pct = Column(Float, nullable=False)
    keywords_used = Column(Text, nullable=True)
    keywords_missed = Column(Text, nullable=True)
    misconception_flag = Column(Integer, default=0)


APP_MODES = ["socratic", "quiz", "extended", "rewrite"]


def extract_keywords_from_subtopic(subtopic: str) -> list:
    """Extracts candidate keywords dynamically from subtopic text without hardcoding."""
    # Split by punctuation and spaces, filtering out short stop words
    words = [
        w.strip(",.():–—")
        for w in subtopic.split()
        if len(w.strip(",.():–—")) > 3 and w.lower() not in ["and", "with", "from", "that", "this"]
    ]
    return words if words else ["Key Term 1", "Key Term 2", "Key Term 3"]


# --- 3. Domain-Agnostic Generation Logic ---
def generate_synthetic_data(num_records=200):
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    spec_path = "course_spec.json"
    if os.path.exists(spec_path):
        with open(spec_path, "r", encoding="utf-8") as f:
            course_spec = json.load(f)
    else:
        print("Error: course_spec.json not found.")
        return

    topics = course_spec.get("topics", {})
    if not topics:
        print("Warning: course_spec.json has no topics.")
        return

    subject_title = course_spec.get("course_title", "General Course")
    course_level = course_spec.get("level", "GCSE")

    # Flatten hierarchy to easily sample valid (unit, subtopic) pairs
    topic_pairs = []
    for top_level, content in topics.items():
        if isinstance(content, dict):
            # 3-tier structure: Component -> Unit -> Subtopics
            for unit_name, subtopics in content.items():
                if isinstance(subtopics, list):
                    for st_item in subtopics:
                        topic_pairs.append((unit_name, st_item))
        elif isinstance(content, list):
            # 2-tier structure: Unit -> Subtopics
            for st_item in content:
                topic_pairs.append((top_level, st_item))

    if not topic_pairs:
        print("Error: No valid units and subtopics found in course_spec.json.")
        return

    print(f"Seeding {num_records} records for '{subject_title}'...")

    start_date = datetime.now() - timedelta(days=30)

    for _ in range(num_records):
        unit_name, subtopic_name = random.choice(topic_pairs)
        app_mode = random.choice(APP_MODES)

        base_score = random.choices(
            [40.0, 55.0, 70.0, 85.0, 100.0], weights=[0.1, 0.25, 0.35, 0.2, 0.1]
        )[0]
        score_pct = max(
            10.0, min(100.0, base_score + random.uniform(-10.0, 10.0))
        )

        all_kw = extract_keywords_from_subtopic(subtopic_name)
        num_used = int((score_pct / 100.0) * len(all_kw))
        random.shuffle(all_kw)
        used_kw = all_kw[:num_used]
        missed_kw = all_kw[num_used:]

        misconception_flag = 1 if score_pct < 45.0 else 0

        timestamp = start_date + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(8, 17),
            minutes=random.randint(0, 59),
        )

        log = ActivityLog(
            timestamp=timestamp,
            subject=subject_title,
            level=course_level,
            unit=unit_name,
            subtopic=subtopic_name,
            app_mode=app_mode,
            score_pct=round(score_pct, 1),
            keywords_used=", ".join(used_kw) if used_kw else None,
            keywords_missed=", ".join(missed_kw) if missed_kw else None,
            misconception_flag=misconception_flag,
        )

        session.add(log)

    session.commit()
    session.close()
    print(f"Successfully populated {num_records} activity logs into activity_logs_history!")


if __name__ == "__main__":
    generate_synthetic_data(350)