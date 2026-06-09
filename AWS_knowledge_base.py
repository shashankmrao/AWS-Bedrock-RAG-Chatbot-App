import json
import os
import sys
import boto3
import logging
import streamlit as st

from langchain_aws import BedrockLLM

import numpy as np


from langchain_core.prompts import PromptTemplate
from botocore.exceptions import ClientError

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever

logger = logging.getLogger("Bedrock-RAG")
logging.basicConfig(level=logging.INFO)


bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")


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

def get_response_llm(llm,query):
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)
    retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="TSMZEBHADH",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 2}},
    )
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
    st.header("Chat with PDF using Amazon Bedrock and Knowledge Base")

    user_question = st.text_input("Ask a question about Indian tourism:")


    if st.button("Generate Output"):
        with st.spinner("Processing..."):
            llm=get_llm()
            st.write(get_response_llm(llm,user_question))
            st.success("Done")

if __name__=="__main__":
    main()













