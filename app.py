import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai

# ======================
# CONFIG
# ======================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ======================
# SESSION STATE
# ======================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ======================
# UI
# ======================

st.title("📚 RAG-Based Document Q&A Chatbot")

st.markdown(
    "Upload a PDF document and ask questions about its content."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# ======================
# PDF PROCESSING
# ======================

if uploaded_file and st.session_state.vector_db is None:

    with st.spinner("Processing PDF..."):

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_texts(
            chunks,
            embedding_model
        )

        st.session_state.vector_db = vector_db

    st.success(
        f"PDF Processed Successfully! ({len(chunks)} chunks created)"
    )

# ======================
# QUESTION SECTION
# ======================

if st.session_state.vector_db:

    question = st.text_input(
        "Ask a Question"
    )

    if st.button("Submit Question"):

        if question:

            with st.spinner("Generating Answer..."):

                docs = st.session_state.vector_db.similarity_search(
                    question,
                    k=3
                )

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say:
'I could not find the answer in the document.'
"""

                try:

                    response = model.generate_content(
                        prompt
                    )

                    answer = response.text

                    st.session_state.messages.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Error: {str(e)}"
                    )

# ======================
# CHAT HISTORY
# ======================

if st.session_state.messages:

    st.subheader("💬 Chat History")

    for msg in reversed(
        st.session_state.messages
    ):

        st.markdown(
            f"**🧑 Question:** {msg['question']}"
        )

        st.markdown(
            f"**🤖 Answer:** {msg['answer']}"
        )

        st.markdown("---")

# ======================
# CLEAR CHAT
# ======================

if st.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()