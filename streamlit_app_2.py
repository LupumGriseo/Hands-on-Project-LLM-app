# --- SQLite patch: must run BEFORE importing chroma/langchain_chroma ---
import sys
import pysqlite3 as sqlite3  # ensures required SQLite features for chromadb on Streamlit Cloud
sys.modules["sqlite3"] = sqlite3
sys.modules["sqlite3.dbapi2"] = sqlite3.dbapi2
# ----------------------------------------------------------------------

import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="🦜🔗 Ask the Doc App")
st.title("🦜🔗 Ask the Doc App")

def generate_response(uploaded_file, api_key, query_text):
    # Provide the key for langchain_openai
    os.environ["OPENAI_API_KEY"] = api_key

    # Read upload
    uploaded_file.seek(0)
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    docs = [Document(page_content=raw_text, metadata={"filename": uploaded_file.name})]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Build vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)  # in-memory
    retriever = vectordb.as_retriever()

    # LLM and prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer ONLY with information present in the provided context. "
                   "If the answer is not in the context, reply that you don't know."),
        ("human", "Question: {input}\n\nContext:\n{context}")
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    result = rag_chain.invoke({"input": query_text})
    return result["answer"]

# Sidebar: API key
with st.sidebar:
    st.markdown("#### OpenAI API Key")
    default_key = st.secrets.get("OPENAI_API_KEY", "")
    openai_api_key = st.text_input("sk-...", type="password", value=default_key)

# Main inputs
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
query_text = st.text_input(
    "Enter your question:",
    placeholder="Please provide a short summary.",
    disabled=not uploaded_file
)

# Submit button
if st.button("Submit", disabled=not (uploaded_file and query_text)):
    if not openai_api_key:
        st.error("Please provide your OpenAI API key (or set it in Secrets).")
    else:
        with st.spinner("Thinking..."):
            try:
                answer = generate_response(uploaded_file, openai_api_key, query_text)
                st.info(answer)
            except Exception as e:
                # Show the full exception to help with debugging if something else pops up
                st.exception(e)
