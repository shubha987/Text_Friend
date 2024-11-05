import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_openai import ChatOpenAI
from htmlTemplate import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    raw_text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page_num in range(len(pdf_reader.pages)):
            raw_text+=pdf_reader.pages[page_num].extract_text()
    return raw_text


def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len)
    text_chunks=text_splitter.split_text(raw_text)
    return text_chunks

def get_vectorstore(text_chunks):
    embeddings=OpenAIEmbeddings()
    #embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    for chunk in text_chunks:
        vectorstore.add_text(chunk)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain=create_history_aware_retriever(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain


def handle_user_input(user_question):
    response=st.session_state.conversation({"question":user_question})
    st.session_state.chat_history=response["chat_history"]

    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    st.header("Chat with PDFs")
    st.write("Upload a PDF file and start chatting with it!")

    user_question=st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)
    st.write(user_template.replace("{{MSG}}", "What is the document about?"), unsafe_allow_html=True)   
    st.write(bot_template.replace("{{MSG}}", "The document is about..."), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload your PDFs here and click on 'Process'", type="pdf",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get pdf text
                raw_text=get_pdf_text(pdf_docs)
                #get the text chunks
                text_chunks=get_text_chunks(raw_text)
                # create vector storage
                vectorstore=get_vectorstore(text_chunks)
                # create conversationchain
                st.session_state.conversation_chain=get_conversation_chain(vectorstore)

        
if __name__ == "__main__":
    main()