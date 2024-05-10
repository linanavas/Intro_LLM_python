import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = "sk-proj-ZeYMSofKo5e8n8qpcbon9vT3BlbkFJZ3V5uyzkVBOMM4uI2d"

# Upload PDF files
st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # st.write(text)

    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    # Creating vector store
    vector_store = FAISS.from_texts(chunks,embeddings)

    # Get user question
    user_question = st.text_input("Type your question here")

    # Do similarity search
    if user_question is not None:
        match = vector_store.similarity_search(user_question)

        # Define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0, # Randomness
            max_tokens = 500, # Length of response
            model_name = "gpt-3.5-turbo"
        )

        # Output results
        # Chain: Take the question, get relevant text (match),
        # pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(
            question = user_question,
            input_documents = match
        )

        st.write(response)





