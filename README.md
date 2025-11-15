# 📚 Chat with Multiple PDFs (Gemini 1.5)

Upload multiple PDFs and ask questions about their content. This is a minimal Retrieval-Augmented Generation (RAG) app built using Google Gemini 1.5 and FAISS.

---

## 🚀 Features

- Upload multiple PDFs
- Process and build a FAISS vector index locally
- Ask questions and get context-aware answers from Gemini 1.5
- Uses Google Generative AI embeddings for vectorization
- Exports conversation history as CSV

---

## ⚙️ Tech Stack

- Streamlit – UI framework
- LangChain – orchestration of retrieval + LLM
- Google Generative AI – Gemini 1.5 + Embeddings
- FAISS – local vector store
- PyPDF2 – text extraction from PDFs
- Pandas – CSV export

---

## 🗂️ File Structure

main.py
faiss_index/          # created after processing PDFs
requirements.txt
.env                  # your API key (not committed)
README.md

---

## 🔧 Setup

1. Create a virtual environment

python -m venv .venv

Activate it:
mac / linux:
source .venv/bin/activate

windows:
.venv\Scripts\activate

2. Install dependencies

pip install -r requirements.txt

3. Add your Google API Key

Create a .env file in the project folder:

GOOGLE_API_KEY=your_google_api_key_here

(or paste the key in the Streamlit sidebar)

---

## ▶️ Running the App

streamlit run main.py

Then:

- Add your API key in the sidebar
- Upload PDFs
- Click “Process PDFs”
- Ask questions about the documents

---

## 🧠 How It Works

1. PDFs → extracted text (PyPDF2)
2. Text → chunked (RecursiveCharacterTextSplitter)
3. Chunks → embedded using Google Generative AI embeddings
4. Stored in FAISS
5. User asks a question
6. Top relevant chunks are retrieved
7. Gemini 1.5 answers based only on retrieved context

---

## 🧾 .gitignore

.env
faiss_index/
.venv/

---

## 👍 Good to Mention in Interviews

- Adjustable chunk size & overlap
- Retrieval first → then LLM
- Gemini is set with low temperature to avoid hallucinations
- FAISS makes the app fully local and offline after embedding
- Can be extended with source previews, streaming responses, or Pinecone/Chroma
