"""
app.py
Streamlit UI: user asks a question -> embed question -> vector search in Supabase
-> retrieve relevant chunks -> send to Groq LLM -> display answer with citations.
"""

import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv

# ---------- Setup ----------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

TOP_K = 5  # number of chunks to retrieve per question

st.set_page_config(page_title="AI/ML Research Assistant", page_icon="🔎", layout="wide")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()


# ---------- Core RAG functions ----------
def retrieve_chunks(question, top_k=TOP_K):
    """Embed the question and run similarity search via Supabase RPC."""
    query_embedding = embedder.encode(question).tolist()

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k
        }
    ).execute()

    return response.data


def build_prompt(question, chunks):
    """Construct a grounded prompt with numbered sources."""
    context_blocks = []
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page", "?")
        context_blocks.append(f"[Source {i+1}: {source}, page {page}]\n{chunk['content']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a research assistant answering questions about AI/ML papers.
Use ONLY the context below to answer. If the context doesn't contain the answer, say so clearly.
Cite sources using [Source N] notation matching the numbers below.

Context:
{context}

Question: {question}

Answer (with [Source N] citations inline):"""

    return prompt


def generate_answer(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------- UI ----------
st.title("🔎 AI/ML Research Paper Assistant")
st.caption("Ask questions across your uploaded research papers. Answers are grounded in retrieved chunks with citations.")

question = st.text_input("Ask a question about your papers:", placeholder="e.g. How does RAG reduce hallucination?")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving relevant chunks..."):
        chunks = retrieve_chunks(question)

    if not chunks:
        st.warning("No relevant chunks found. Did you run ingest.py first?")
    else:
        with st.spinner("Generating answer..."):
            prompt = build_prompt(question, chunks)
            answer = generate_answer(prompt)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources retrieved")
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "unknown")
            page = chunk["metadata"].get("page", "?")
            with st.expander(f"Source {i+1}: {source} (page {page})"):
                st.write(chunk["content"])
