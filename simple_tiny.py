import os
import sys
from typing import Optional
import logging
import re

# Disable LangChain debug logs
logging.getLogger("langchain").setLevel(logging.ERROR)
import langchain
langchain.debug = False

# Import LangChain Community utilities and HuggingFacePipeline wrapper
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline

from sqlalchemy import create_engine, inspect
from sqlalchemy import text

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_tinyllama_langchain_llm():
    print("⏳ Loading TinyLlama model (HuggingFace style)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("✅ Model loaded!")
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    return llm

def pick_tables(question: str, all_tables: list) -> list:
    """Naive approach: pick tables whose names appear in the user's question.
       Fallback: pick the first 3 tables."""
    question_lower = question.lower()
    relevant = [t for t in all_tables if t.lower() in question_lower]
    return relevant or all_tables[:3]

def get_schema_text(db: SQLDatabase, db_uri: str) -> str:
    """
    Build a textual representation of the schema from the database.
    For each table, list its columns and types.
    Uses SQLDatabase.get_table_info(); if that fails, falls back on SQLAlchemy inspector.
    """
    engine = create_engine(db_uri)
    inspector = inspect(engine)
    tables = db.get_usable_table_names()
    lines = []
    for tbl in tables:
        lines.append(f"Table: {tbl}")
        try:
            columns = db.get_table_info(tbl)
            if not columns:
                raise Exception("No columns found")
            for col in columns:
                lines.append(f"  - {col['name']} ({col['type']})")
        except Exception as e:
            # Fallback using SQLAlchemy inspector:
            try:
                cols = inspector.get_columns(tbl)
                for col in cols:
                    lines.append(f"  - {col['name']} ({col['type']})")
            except Exception as e2:
                lines.append("  - [Error retrieving columns]")
        lines.append("")  # Blank line between tables
    return "\n".join(lines).strip()

# ===== This function extracts only the SQL query from the model's response =====
def extract_sql_query(text: str) -> str:
    """
    Extracts the SQL query from the model's response.
    Assumes the SQL query starts with a SQL keyword (e.g., SELECT, INSERT, UPDATE, DELETE)
    and ends with a semicolon.
    Adjust the regex as needed.
    """
    pattern = re.compile(r"(?i)(SELECT|INSERT|UPDATE|DELETE).*?;", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(0).strip()
    else:
        # Fallback: return the original text if no match is found.
        return text.strip()
# ===== End extraction function =====

# This function is your manual SQL-generation logic.
def generate_sql_custom(question: str, schema_text: str, llm) -> str:
    """
    Manually constructs a prompt (similar to your base version) and uses the LLM
    to generate a SQL query.
    """
    prompt = (
        "Generate an SQL query strictly based on the schema provided.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question:\n{question}\n\n"
        "Only output SQL code. Do not output any explanation or additional text.\n"
        "SQL:"
    )
    result = llm(prompt)
    return result.strip()

def main():
    # 1) Load the TinyLlama model
    llm = load_tinyllama_langchain_llm()

    # 2) MySQL credentials
    user = "root"
    password = "admin"
    host = "localhost"
    port = 3306
    database = "chatbot"
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

    print("\n🔹TinyLlama Chat w/ MySQL using custom prompt (no chain-of-thought).\n")
    print("Type 'exit' or 'quit' to stop.\n")

    # 3) Build a wide DB for table discovery
    wide_db = SQLDatabase.from_uri(db_uri)
    all_table_names = wide_db.get_usable_table_names()

    while True:
        question = input("User Question: ")
        if question.strip().lower() in ["exit", "quit"]:
            break

        # 4) Choose relevant tables using partial schema selection
        relevant_tables = pick_tables(question, all_table_names)

        # 5) Reflect columns only for relevant tables
        filtered_db = SQLDatabase.from_uri(db_uri, include_tables=relevant_tables)

        # 6) Build a schema text from the filtered DB
        schema_text = get_schema_text(filtered_db, db_uri)

        # 7) Generate SQL using the custom prompt (manual logic)
        sql_query_raw = generate_sql_custom(question, schema_text, llm)
        # Extract only the SQL query from the LLM's response
        final_sql = extract_sql_query(sql_query_raw)
        print(f"\nFinal SQL Query:\n{final_sql}\n")

        # 8) Execute the SQL query using SQLAlchemy
        try:
            engine = create_engine(db_uri)
            with engine.connect() as connection:
                result = connection.execute(text(final_sql))
                rows = result.fetchall()
            print("DB Results:")
            for row in rows:
                print(row)
            print("")
        except Exception as e:
            print(f"\n❌ Error executing query: {e}\n")

    print("👋 Exiting. Goodbye!")

if __name__ == "__main__":
    main()
