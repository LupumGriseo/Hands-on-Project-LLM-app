import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def generate_response(uploaded_file, openai_api_key, query_text):
    if uploaded_file is None:
        return "Please upload a file."
    # Read & split
    raw_text = uploaded_file.getvalue().decode(errors="ignore")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.create_documents([raw_text])

    # Embeddings & vector store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    db = Chroma.from_documents(docs, embedding=embeddings)  # requires chromadb installed

    retriever = db.as_retriever()

    # LLM (chat model is recommended now)
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.run(query_text)

# Streamlit UI
st.set_page_config(page_title="🦜🔗 Ask the Doc App")
st.title("🦜🔗 Ask the Doc App")

uploaded_file = st.file_uploader("Upload an article", type="txt")
query_text = st.text_input("Enter your question:", placeholder="Please provide a short summary.", disabled=not uploaded_file)

result = []
with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input("OpenAI API Key", type="password", disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button("Submit", disabled=not (uploaded_file and query_text))
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner("Calculating..."):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)

if result:
    st.info(result[-1])