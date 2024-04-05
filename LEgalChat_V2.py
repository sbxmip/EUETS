
import streamlit as st
import random
import time
import openai
import os
import csv
import singlestoredb as s2
import json
import openai
from openai import OpenAI
import time
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# generic variables
sasserver = "https://viya4-s2.zeus.sashq-d.openstack.sas.com"
open_ai_key=''
clientId='sas.ec'
clientSecret=''
# openai client
client = OpenAI(api_key=open_ai_key)
# S2 conn details
s2_host = '10.104.82.118'
s2_user = 'admin'
s2_password = 'Orion123'
s2_db = 'MyDB'

# Custom class that interfaces with s2
class SingleStoreRetriever:
    def __init__(self, host, user, password, db):
        self.host = s2_host
        self.user = s2_user
        self.password= s2_password
        self.db = s2_db
    
    def connect(self):
        return s2.connect(host=self.host, user=self.user, password=self.password, database=self.db, ssl_disabled=True, ssl_verify_cert=False,ssl_verify_identity=False)

    def execute_query(self, stmt):
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(stmt)
                return cursor.fetchall()  # Fetch all rows  
                 
    def create_embeddings_and_insert(self, documents, model="text-embedding-3-small", batch_size=2):
        # Split into batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            texts = [text for _, text in batch]
            response = client.embeddings.create(input=texts, model=model)
            embeddings = [item.embedding for item in response.data]
            
            with self.connect() as connection:
                with connection.cursor() as cursor:
                    for j, (filename, content) in enumerate(batch):
                        embedding = embeddings[j]
                        json_embedding = json.dumps(embedding)
                        stmt = """
                        INSERT INTO embeddings_openai2 (id, content, vector)
                        VALUES (%s, %s, JSON_ARRAY_PACK(%s))
                        """
                        cursor.execute(stmt, (filename, content, json_embedding))
                    connection.commit()

    def cosine_similarity(self, query_embedding, top_x=3):
        ## This returns the top_x documents as a list of strings based on the query_embedding.
        # Convert query_embedding to JSON for the query
        json_embedding = json.dumps(query_embedding)
        # sql
        stmt = f'''
        select id, content, dot_product(vector,JSON_ARRAY_PACK(%s)) as score
        from embeddings_openai2
        order by score desc
        limit %s;
        '''
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(stmt, (json_embedding, top_x))
                results = cursor.fetchall()
        return results

# Streamed response emulator
def response_generator(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

client = openai.OpenAI()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

# Embed user query using OpenAI
def get_query_embedding(query):
    query = query.replace("\n", " ")
    response = client.embeddings.create(input=query, model="text-embedding-3-small")
    return response.data[0].embedding 


PROMPT_TEMPLATE ="""Please answer (in DUTCH language) the following question clearly and concisely. 
After providing your answer, list the document name(s) and cite key sentences or passages 
from those documents that support your answer.

Question: {query}

Context:
{context}

----
Instructions for the API:
1. Write the answer in Dutch
2. Identify the document name(s) that include specific and 
relevant information in responding the question. (in Dutch)
4. For each document mentioned, cite the key sentences or passages (in Dutch)
that could be relevant in answering the user's question. (in Dutch)
5. Provide a direct answer to the question. (in Dutch)
6. Provide the answer in the following format : 
    Relevante Documenten: '\n'
    Document Naam 1 : BElangrijke passages '\n'
   
    Document Naam 2 : Belangrijke passages '\n'

    Anwtoord :  
----
"""

def process_request(query):
    # Embed the user's query
    query_embedding = get_query_embedding(query)
    
    # Get top relevant documents based on cosine similarity score
    results=retriever.cosine_similarity(query_embedding)
    
    # Create prompt template
    context_text = "\n\n---\n\n".join([f"Document Name: {name}\nContent: {doc_content}"for name, doc_content, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query)

    # Init model
    model=ChatOpenAI(api_key=open_ai_key)
    
    # Generate a response based on the query and retrieved documents
    return model.predict(prompt)   
    
retriever = SingleStoreRetriever(s2_host, s2_user, s2_password, s2_db)

st.title("Chat with your documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if question := st.chat_input("Ask any question on Belgian labour legislation?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content":question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        #prompt = f""" You are a legal expert tasked with answering questions 
                #on the interpretation of Belgian laws concerning working conditions. More specifically
                #you have received following question : '''{question}'''. Please answer the question by taking into account and analysing
                #the contents of the following law: ```{wet}```. If you are not sure, do not provide an answer and request for additional information. Please provide the answer in Dutch
                #"""
        #response = get_completion(prompt)
        #print(response)

        response=process_request(question)

        text = st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": text})