# 📚 RAG-Based Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions in natural language.

## Features

* PDF Upload
* Text Extraction
* Text Chunking
* Semantic Search
* FAISS Vector Database
* HuggingFace Embeddings
* Gemini 2.5 Flash Integration
* Streamlit User Interface

## Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Sentence Transformers
* Google Gemini 2.5 Flash
* PyPDF

## How It Works

1. Upload PDF
2. Extract Text
3. Split Text into Chunks
4. Generate Embeddings
5. Store Embeddings in FAISS
6. Retrieve Relevant Chunks
7. Generate Context-Aware Answer using Gemini

## Installation

pip install -r requirements.txt

streamlit run app.py
