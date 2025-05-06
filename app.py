#!/usr/bin/env python
# coding: utf-8

# # LearnSphere 

# ### import libraries

# In[14]:


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


# ### coding strats here 

# In[ ]:


# GOOGLE_API_KEY="AIzaSyBWlwRXCKq4YxK2CPPoxNvUoZKIGzu5ANo"
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# get PDF and converted  to text
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

# divide the text in to smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# chunks converted to embeddings to vector and store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

#
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# get user inputs
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    # new_db = FAISS.load_local("faiss_index", embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}, return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


# frontend main function
def main():
    st.set_page_config(page_title="LearnSphere App", layout="wide")

    # Header Section
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📚 LearnSphere</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #666;'>Empowering Knowledge Through Conversational PDFs 🚀</h4>", unsafe_allow_html=True)
    st.write("---")

    # Two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        user_question = st.text_input("🔍 Ask something from your uploaded PDFs:")
        if user_question:
            user_input(user_question)

    with col2:
        with st.sidebar:
            st.markdown("### 📥 Upload & Process")
            pdf_docs = st.file_uploader(
                "Upload one or more PDF files", accept_multiple_files=True, type=["pdf"]
            )
            if st.button("📤 Submit & Process"):
                with st.spinner("🔄 Processing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! Ready to chat.")

if __name__ == "__main__":
    main()

