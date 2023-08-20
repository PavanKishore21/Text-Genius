import streamlit as st
import docx
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from transformers import pipeline

def main():
    st.title("TextGenius: Summarization & QA")
    load_dotenv()
    
    text = ""

    with st.sidebar:
        st.title("File Upload")
        filetype = st.selectbox("Choose File Type",['PDF','Word'])
        # File Upload

        if filetype == "PDF":
            pdf = st.file_uploader("",type=["pdf"])
            if pdf is not None: 
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        if filetype == "Word":
            uploaded_file = st.file_uploader("", type=["docx"])
            if uploaded_file is not None: 
                doc = docx.Document(uploaded_file)
                content = []
                for paragraph in doc.paragraphs:
                    content.append(paragraph.text)
                text = content

        st.write("###  Enter text manually:")
        if text == "":
            textInput = st.text_area("Input",height=300)
        else:
            textInput = st.text_area("Input",value=text,height=300)

    st.write("### Choose an action:")

    option = st.selectbox("", ("None","Summarize", "Chat"))
    
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
    chunks = text_splitter.split_text(text=text)

    
    # embeddings = OpenAIEmbeddings()
    # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        
    model_name = "facebook/bart-large-cnn"
    if option == "Summarize":
        length_mapping = {
            "Short": (100, 150),    
            "Medium": (150, 200),
            "Long": (200, 300)
        }
        length = st.selectbox("Choose length:", ("Short", "Medium","Long"))
        min_length, max_length = length_mapping[length]
        if st.button("Generate Summary"):
            summarizer = pipeline("summarization", model = model_name)
            summary = summarizer(textInput, max_length=max_length, min_length=min_length)
            st.subheader("Generated Summary:")
            st.write(summary[0]['summary_text'])
    elif option == "Chat":
        query = st.text_input("Ask questions about your file:")
        if query:
            qa_pipeline = pipeline("question-answering",model=model_name)
            context = textInput
            answer = qa_pipeline(question=query, context=context)
            st.write("Question:", query)
            st.write("Answer:", answer['answer'])
    



if __name__ == "__main__":
    main()