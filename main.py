from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import streamlit as st
import os

openai_api_key = os.getenv("API_KEY")


def update_embeddings():
    """
    Updates the embeddings for the documents loaded from a PDF file.
    This function performs the following steps:
    1. Loads a PDF file specified by the `pdf_path`.
    2. Splits the loaded PDF into smaller chunks using a text splitter.
    3. Generates embeddings for the document chunks using the OpenAI embeddings model.
    4. Stores the embeddings in a Chroma vector store.
    Returns:
        Chroma: A Chroma vector store containing the embeddings of the document chunks.
    """

    pdf_path = "MG3.pdf"
    pdf_loader = PyPDFLoader(pdf_path)
    docs = pdf_loader.load()

    print("Loaded PDF")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(docs)

    embeddings_engine = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    vector_store = Chroma.from_documents(chunked_docs, embeddings_engine, persist_directory="./chroma_db")

    return vector_store

st.title("RAG PDF Reader")
st.write("This is a RAG PDF Reader") 

if st.button("Update Embeddings"):
    vector_store = update_embeddings()
    st.write("Embeddings Updated")

# Define the prompt template
prompt_template = """
You are a smart helping agent specialized in the MG3 user manual.
Answer the user's questions {input} relationated to the MG3 user manual based in {context}.
Don't make up information, if you don't know the answer, just say that you don't know.
Only answer questions that are related to the {context}
"""

# Create an instance of the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o", max_tokens=1024, api_key=openai_api_key)

# Create an instance of the LLMChain model
qa_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

question = st.text_area("Enter your question here:")

# Create a button to send the question
if st.button("Send"):
    if question:
        embeddings_engine = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_engine)

        similar_results = vector_store.similarity_search(question, k=5)

        context = ""

        for result in similar_results:
            context += result.page_content

        response = qa_chain.invoke(question=question, context=context)
        response_text = response["text"]

        st.write(response_text)
    else:
        st.write("Please enter a question")
