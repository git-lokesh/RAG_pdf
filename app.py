import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()

def split_text_to_chunks(text, chunk_size=1500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

@st.cache_data(show_spinner="Indexing documents...")
def get_vector_store(_chunks):
    if not _chunks:
        return None
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    store = FAISS.from_texts(_chunks, embeddings)
    return store

def create_rag_chain():
    llm = ChatOllama(model="llama3.1:8b", temperature=0.2)
    prompt_template = """Answer the question using only the provided context. If the information is not present, say so clearly.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain

def answer_question(question, store, chain):
    if not store:
        return "Error: The document vector store is not initialized."
    docs = store.similarity_search(question, k=4)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    try:
        response = chain.invoke({"context": context, "question": question})
        return response
    except Exception as e:
        st.error(f"Error invoking LLM: {e}")
        return "There was an error processing your question."

def main():
    st.set_page_config(page_title="Local PDF Chat 📚", layout="centered")
    st.title("Chat with Your PDFs (Locally)")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = create_rag_chain()

    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload your PDFs here and click 'Process'",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Extracting text..."):
                    text = extract_text_from_pdfs(uploaded_files)
                if text:
                    with st.spinner("Splitting text into chunks..."):
                        chunks = split_text_to_chunks(text)
                    st.session_state.vector_store = get_vector_store(chunks)
                    st.success("PDFs processed! You can now ask questions.")
                else:
                    st.error("No extractable text found in the PDFs.")
            else:
                st.warning("Please upload at least one PDF.")

        st.divider()

        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.success("Conversation cleared.")

    for q, a in st.session_state.conversation_history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    question = st.chat_input("Ask a question about your documents...")

    if question:
        if st.session_state.vector_store is None:
            st.warning("Please upload and process your PDFs first.")
        else:
            st.chat_message("user").write(question)
            with st.spinner("Llama is thinking..."):
                answer = answer_question(
                    question,
                    st.session_state.vector_store,
                    st.session_state.rag_chain
                )
            st.chat_message("assistant").write(answer)
            st.session_state.conversation_history.append((question, answer))

if __name__ == "__main__":
    main()
