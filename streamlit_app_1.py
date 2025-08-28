import streamlit as st
from langchain_openai import ChatOpenAI

st.title("🦜🔗 Quickstart App")

# Option A1: read from Streamlit Secrets (secure on Streamlit Cloud)
# openai_api_key = st.secrets["OPENAI_API_KEY"]

# Option A2: allow manual entry (good for local runs / debugging)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def generate_response(input_text: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)
    resp = llm.invoke(input_text)      # returns an AIMessage
    st.info(resp.content)

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key or not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)