import streamlit as st
import requests
import json

API_URL = "https://tmg93ng4w3.execute-api.us-east-1.amazonaws.com/dev/generate"

st.set_page_config("Chat PDF")
st.header("Chat with PDF using Amazon Bedrock and Knowledge Base and AWS Lambda")

user_question = st.text_input("Ask a question about Indian tourism:")

if st.button("Generate Output"):
    with st.spinner("Processing..."):
        response = requests.post(
            API_URL,
            json={"question": user_question}
        )
        if response.status_code == 200:
            st.write(response.json()["response"])
        else:
            st.error(f"Error: {response.status_code}")