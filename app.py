# main.py
import os
from datetime import datetime

import base64
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain imports (modern patterns)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Google GenAI integration (embeddings + chat model wrapper)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()  # load .env if present

# ---------- Helpers ----------
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
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
    store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return store

def make_qa_chain(api_key, temperature=0.2):
    # prompt tailored for Indian annual report / financial analysis scenario (keeps original spirit)
    prompt_template = """Answer the question using only the provided context. If the information is not present, say so clearly.
Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature, google_api_key=api_key)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=None, return_source_documents=False)
    # we'll inject the retriever at runtime
    return chain, prompt

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="PDF RAG Chat", page_icon="📚")
    st.title("Chat with PDFs — simple RAG")

    # API Key: environment variable prioritized, then sidebar input
    env_api_key = os.getenv("GOOGLE_API_KEY")
    api_key_input = st.sidebar.text_input("Google API Key", value=env_api_key or "", type="password")
    api_key = api_key_input.strip() or env_api_key

    if not api_key:
        st.sidebar.warning("Set GOOGLE_API_KEY in environment or paste it here to use the app.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Menu**")
    if st.sidebar.button("Clear conversation"):
        st.session_state.conversation_history = []

    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.sidebar.button("Process PDFs"):
        if not uploaded_files:
            st.sidebar.warning("Upload at least one PDF before processing.")
        elif not api_key:
            st.sidebar.warning("Provide the Google API key first.")
        else:
            with st.spinner("Extracting text and building index..."):
                text = extract_text_from_pdfs(uploaded_files)
                if not text:
                    st.error("No extractable text found in uploaded PDFs.")
                else:
                    chunks = split_text_to_chunks(text)
                    build_vectorstore(chunks, api_key)
                    st.success("Index built and saved locally (faiss_index).")

    # initialize conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    question = st.text_input("Ask a question about the uploaded PDFs")
    if question:
        if not api_key:
            st.warning("Provide API key to ask questions.")
        else:
            # load vectorstore and run retrieval + QA
            try:
                store = load_vectorstore(api_key)
            except Exception as e:
                st.error("Vector store not found. Process PDFs first.")
                return

            retriever = store.as_retriever(search_kwargs={"k": 6})
            chain, prompt_template = make_qa_chain(api_key)
            # attach retriever to chain
            chain.retriever = retriever

            # run chain
            with st.spinner("Generating answer..."):
                answer = chain.run(question)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.conversation_history.append((question, answer, "Google AI", timestamp))
            st.success("Answer generated")

    # show conversation (latest at top)
    for q, a, model, ts in reversed(st.session_state.conversation_history):
        st.markdown(f"**Q — {q}**")
        st.markdown(f"> {a}")
        st.caption(f"{model} • {ts}")
        st.markdown("---")

    # download conversation history as CSV
    if st.session_state.conversation_history:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">Download conversation CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
