import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))


def extract_pdf_content(pdf):
    return extract_text(pdf)

def create_vector_database(pdf_content):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
        
    chunks = text_splitter.split_text(pdf_content)
    embeddings = FastEmbedEmbeddings(model="ms-marco-MiniLM-L-12-v2")
    vector_database = FAISS.from_texts(chunks, embeddings)
    return vector_database

def response(question, vector_database):
    similar_docs = vector_database.similarity_search(question)

    llm = genai.GenerativeModel(model_name = "gemini-pro")
    prompt_template = f"""
        Context: {similar_docs}

        Task: Based on the provided context, answer the following question with a detailed and clear explanation. Avoid one-word or overly brief responses.

        Question: {question}
        """
    result = llm.generate_content(prompt_template)
    return result


def user_interface():
    st.set_page_config(
        page_title="Talk to PDF",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Arial Black', sans-serif;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
    }
    .prompt-box {
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 10px;
        background-color: #ecf5fd;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown("<h1 class='main-header'>💬 Talk to Your PDF</h1>", unsafe_allow_html=True)

    st.subheader("Upload Your PDF File")
    pdf_uploaded = st.file_uploader(
        "Drag and drop your PDF file here or click to browse.",
        type="pdf",
        label_visibility="collapsed"
    )

    if pdf_uploaded is not None:
        st.success("PDF uploaded successfully! Processing... 📄")
        pdf_content = extract_pdf_content(pdf_uploaded)
        with open("text.txt", "w", encoding="utf-8") as f:
            f.write(pdf_content)
        vector_database = create_vector_database(pdf_content)

        st.subheader("Ask Your Question")
        question = st.text_input("Your Question:", placeholder="Ask me anything about the PDF content...")

        if question:
            result = response(question, vector_database)
            st.subheader("Response")
            st.markdown(
                f"""
                <div style="padding: 15px; border: 2px solid #2ecc71; border-radius: 10px; background-color: #f4fdf4; color: #2c3e50; font-family: Arial, sans-serif;">
                    <b>Answer:</b> {result.text}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Please upload a PDF file to get started. 📂")



if __name__ == "__main__":
    user_interface()