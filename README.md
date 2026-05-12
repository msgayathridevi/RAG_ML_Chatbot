# RAG ML Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** built with **LangChain**, **Flask**, **Google Gemini**, and **FAISS**, designed to answer questions from machine learning documents using semantic search and LLM-powered responses.

This project demonstrates how to build an end-to-end AI chatbot that retrieves relevant context from PDF documents and generates grounded answers using a modern RAG pipeline.

**Author:** Gayathri Devi MS

---

## Project Overview

This application allows users to:

- Upload and process machine learning PDF documents
- Convert documents into semantic embeddings
- Store embeddings in a FAISS vector database
- Retrieve relevant document chunks based on user questions
- Generate context-aware answers using **Google Gemini**
- Interact through a simple web-based chatbot UI

The chatbot uses **Retrieval-Augmented Generation (RAG)**, which helps reduce hallucinations by grounding responses in actual document content.

---

## Features

✅ PDF document ingestion  
✅ Automatic text chunking  
✅ Semantic embeddings using Sentence Transformers  
✅ FAISS vector database for fast retrieval  
✅ Similarity-based document search  
✅ Google Gemini (`gemini-2.5-flash`) for response generation  
✅ Flask web application  
✅ Simple chatbot frontend  
✅ Modular project structure

---

## Tech Stack

### Backend
- **Python**
- **Flask**

### LLM Framework
- **LangChain**

### Embeddings
- **Sentence Transformers**
- `all-MiniLM-L6-v2`

### Vector Store
- **FAISS**

### Language Model
- **Google Gemini**
- `gemini-2.5-flash`

### Frontend
- HTML
- CSS
- JavaScript (AJAX)

---

## Project Structure

```text
RAG_ML_Chatbot/
│
├── app.py                  # Flask application entry point
├── vector_store.py         # Creates FAISS vector database
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables
├── LICENSE
├── README.md
│
├── Data/
│   └── Machine Learning.pdf
│
├── src/
│   ├── helper.py           # Document loading, chunking, embeddings
│   ├── prompts.py          # System prompt for Gemini
│   └── __init__.py
│
├── templates/
│   └── chat.html           # Chat UI
│
└── static/
    └── style.css           # Frontend styling
```

---

## How It Works

### Step 1: Load PDF Documents

The chatbot loads machine learning PDFs from the `Data/` folder.

```python
loader = DirectoryLoader(
    path,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
```

---

### Step 2: Split Text into Chunks

Documents are split into smaller overlapping chunks for better retrieval.

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

---

### Step 3: Generate Embeddings

Each text chunk is converted into vector embeddings using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

### Step 4: Store in FAISS

Embeddings are stored in a FAISS vector database for fast semantic similarity search.

```python
docs = FAISS.from_documents(...)
```

---

### Step 5: Retrieve Relevant Context

When the user asks a question, the chatbot retrieves the top 3 most relevant chunks.

```python
retriever = docs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

---

### Step 6: Generate Answer with Gemini

The retrieved context is sent to **Google Gemini** for grounded answer generation.

```python
ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/RAG_ML_Chatbot.git
cd RAG_ML_Chatbot
```

---

### 2. Create Virtual Environment

### Using Conda

```bash
conda create -n rag_ml_chatbot python=3.11
conda activate rag_ml_chatbot
```

### Using venv

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from:

https://ai.google.dev

---

## Run the Application

Start the Flask app:

```bash
python app.py
```

Open your browser:

```text
http://localhost:8080
```

---

## Example Questions

You can ask:

- What is machine learning?
- Explain supervised learning.
- What is reinforcement learning?
- What is gradient descent?
- Explain neural networks.

---

## Prompt Engineering

The chatbot uses a custom system prompt to:

- Answer only from retrieved context
- Avoid hallucinations
- Keep responses concise
- Return "I don't know" when context is missing

---

## Learning Outcomes

This project demonstrates:

- Building a complete RAG pipeline
- Using LangChain with Gemini
- Creating embeddings
- Working with FAISS
- Flask-based chatbot deployment
- Prompt engineering best practices

---
