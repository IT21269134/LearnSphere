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
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

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
def get_conversational_chain(detail_level):
    if detail_level == "Brief Summary":
        prompt_template = """
        Provide a brief and concise answer to the question from the given context.
        If the answer is not in the context, respond with: 'Answer is not available in the context.'

        Context:\n {context}\n
        Question:\n{question}\n
        Answer:
        """
    else:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the context, respond with: 'Answer is not available in the context.'

        Context:\n {context}\n
        Question:\n{question}\n
        Answer:
        """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# get user inputs
def user_input(user_question, detail_level):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    # new_db = FAISS.load_local("faiss_index", embeddings)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(detail_level)

    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    st.write("### 💬Reply:",response["output_text"])
    # st.markdown("#### 📄 Relevant Content")
    # for doc in docs:
    #     st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:10px'>{doc.page_content}</div>", unsafe_allow_html=True)


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
         with st.expander("🆘 How to Use"):
            st.markdown("""
            - Upload PDF lecture notes using the sidebar.
            - Ask clear, specific questions about the content.
            - Choose how detailed you want the answers to be.
            """)

            user_question = st.text_input("🔍 Ask something from your uploaded PDFs:")
            detail_level = st.selectbox("Response Detail Level", ["Brief Summary", "Detailed Explanation"])

            if user_question:
                user_input(user_question, detail_level)
    with col2:
        with st.sidebar:
            st.markdown("### 📥 Upload & Process")
            pdf_docs = st.file_uploader(
                "Upload one or more PDF files", accept_multiple_files=True, type=["pdf"])
            if st.button("📤 Submit & Process"):
                if not pdf_docs:
                    st.warning("⚠️ Please upload at least one PDF file.")
                    return

                for pdf in pdf_docs:
                    if not pdf.name.lower().endswith(".pdf"):
                        st.error(f"❌ {pdf.name} is not a valid PDF file.")
                        return

                with st.spinner("🔄 Processing your documents..."):
                    progress = st.progress(0)
                    raw_text = get_pdf_text(pdf_docs)
                    progress.progress(30)
                    text_chunks = get_text_chunks(raw_text)
                    progress.progress(60)
                    get_vector_store(text_chunks)
                    progress.progress(100)
                    st.success("✅ Documents processed! Ready to chat.")


if __name__ == "__main__":
    main()

