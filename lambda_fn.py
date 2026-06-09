import json
import boto3
import botocore.config
from langchain_aws import BedrockLLM
from langchain_core.prompts import PromptTemplate
from botocore.exceptions import ClientError
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever


def get_llm():
    bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1", config=botocore.config.Config(read_timeout=300,retries={'max_attempts':1}))
    llm= BedrockLLM(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock)
    return llm


def get_response_llm(llm,query):
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
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)
    retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="XXXXXXXX",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 2}},
    )
    chain = create_retrieval_chain(retriever,
            question_answer_chain)
    try:
        response=chain.invoke({"input": query})
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke model. Reason: {e}")
        exit(1)
    return response["answer"]

def lambda_handler(event, context):
    event=json.loads(event['body'])
    user_question=event['question']

    llm=get_llm()
    response=get_response_llm(llm,user_question)
    return {
        'statusCode': 200,
        'body': json.dumps({
        "response": response
    })
    }
