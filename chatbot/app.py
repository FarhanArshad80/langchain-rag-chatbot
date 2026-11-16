from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# LangSmith tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

st.title("RAG Chatbot with Local LLM (Ollama & LangChain)")

# PDF upload
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # PyPDFLoader needs a file path, so save the upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    @st.cache_resource  # Avoid re-processing the same PDF
    def load_and_vectorize_pdf(file_path):
        # Load PDF pages
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Split into overlapping chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Embed chunks and store in Chroma (requires: ollama pull nomic-embed-text)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        # Retriever returns the top 2 matching chunks
        return vectorstore.as_retriever(search_kwargs={"k": 2})

    with st.spinner("Processing PDF... Please wait."):
        retriever = load_and_vectorize_pdf(tmp_file_path)

    st.success("PDF successfully uploaded and processed!")

    # User question
    input_text = st.text_input("Ask a question from your PDF:")

    if input_text:
        # Local LLM
        llm = ChatOllama(model="qwen:0.5b")

        # Prompt: answer only from the retrieved context
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Join retrieved chunks into one string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # RAG chain: retrieve -> prompt -> LLM -> string
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Generating answer..."):
            response = rag_chain.invoke(input_text)
            st.write("### Answer:")
            st.write(response)