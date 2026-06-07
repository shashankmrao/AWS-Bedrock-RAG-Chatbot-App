import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import retrieval_qa


bedrock=boto3.client(service_name="bedrock_runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)

    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )

    vectorstore_faiss.save_local("faiss_index")

def get_llama_llm():
    llm= BedrockLLM(
        model_id="meta.llama2-70b-chat-v1",
        client=bedrock)
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
Context:
{context}

Question: {question}
Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = retrieval_qa(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}

    )
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Amazon Bedrock")

    user_question = st.text_input("Ask a question from the PDF files:")

    with st.sidebar:
        st.title("Update or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Generate Output"):
        with st.spinner("Processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings)
            llm=get_llama_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__=="__main__":
    main()














