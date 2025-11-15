import os
from datetime import datetime
import base64

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()

def split_text_to_chunks(text, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def build_vectorstore(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    store = FAISS.from_texts(chunks, embeddings)
    store.save_local("faiss_index")
    return store

def load_vectorstore(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_top_docs(store, query, k=6):
    return store.similarity_search(query, k=k)

def call_llm_with_fallback(llm, prompt_text):
    try:
        return llm(prompt_text)
    except Exception:
        try:
            return llm.generate([prompt_text])
        except Exception:
            try:
                return llm.predict(prompt_text)
            except Exception:
                return None

def extract_text_from_docs(docs):
    parts = []
    for d in docs:
        content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        parts.append(content)
    return "\n\n---\n\n".join(parts)

def answer_question(question, store, api_key):
    docs = get_top_docs(store, question, k=6)
    context = extract_text_from_docs(docs)
    prompt_template = """Answer the question using only the provided context. If the information is not present, say so clearly.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    prompt_text = prompt.format(context=context, question=question)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=api_key)
    resp = call_llm_with_fallback(llm, prompt_text)
    if resp is None:
        raise RuntimeError("LLM call failed (no usable interface).")
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict) and "output_text" in resp:
        return resp["output_text"]
    try:
        if hasattr(resp, "generations"):
            gens = resp.generations
            if isinstance(gens, list) and len(gens) > 0 and len(gens[0]) > 0:
                return gens[0][0].text
        if hasattr(resp, "text"):
            return resp.text
    except Exception:
        pass
    return str(resp)

def main():
    st.set_page_config(page_title="PDF RAG Chat", page_icon="📚")
    st.title("Chat with PDFs")

    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.sidebar.text_input("Google API Key", value=env_api_key, type="password")
    api_key = api_key_input.strip() or env_api_key

    if st.sidebar.button("Clear conversation"):
        st.session_state.conversation_history = []

    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.sidebar.button("Process PDFs"):
        if uploaded_files and api_key:
            text = extract_text_from_pdfs(uploaded_files)
            if text:
                chunks = split_text_to_chunks(text)
                try:
                    build_vectorstore(chunks, api_key)
                    st.success("Index built.")
                except Exception as e:
                    st.error(f"Error building index: {e}")
            else:
                st.error("No extractable text found.")
        else:
            st.sidebar.warning("Upload PDFs and enter API key.")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    question = st.text_input("Ask a question")

    if question:
        if not api_key:
            st.warning("Enter API key.")
        else:
            try:
                store = load_vectorstore(api_key)
            except Exception as e:
                st.error(f"Vector store not found. Process PDFs first. {e}")
                store = None

            if store is not None:
                try:
                    answer = answer_question(question, store, api_key)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.conversation_history.append((question, answer, "Google AI", timestamp))
                    st.success("Done.")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    for q, a, model, ts in reversed(st.session_state.conversation_history):
        st.markdown(f"**Q — {q}**")
        st.markdown(f"> {a}")
        st.caption(f"{model} • {ts}")
        st.markdown("---")

    if st.session_state.conversation_history:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">Download conversation CSV</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
