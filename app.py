import json
import os
import sys
import boto3
import logging
import streamlit as st

from langchain_aws import BedrockEmbeddings
from langchain_aws import BedrockLLM

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from botocore.exceptions import ClientError

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("Bedrock-RAG")
logging.basicConfig(level=logging.INFO)


bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)

    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    try:
        vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
        )
    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " +
              format(message))

    vectorstore_faiss.save_local("faiss_index")

def get_llm():
    llm= BedrockLLM(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock)
    return llm


prompt_template = """
You are a helpful tourist guide assistant who will use the following context
to give a concise 100 worded answer or less to the question asked at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context:
{context}


Question: {input}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "input"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)
    retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":2})
    chain = create_retrieval_chain(retriever,
            question_answer_chain)
    try:
        response=chain.invoke({"input": query})
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke model. Reason: {e}")
        exit(1)
    return response['answer']

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
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__=="__main__":
    main()













