from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

# LangGraph imports for Agentic RAG
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# LangSmith tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

st.title("Agentic RAG Chatbot with Local LLM (Ollama & LangGraph)")

# PDF upload
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # PyPDFLoader needs a file path, so save the upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    @st.cache_resource  # Avoid re-processing the same PDF
    def load_and_vectorize_pdf(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        return vectorstore.as_retriever(search_kwargs={"k": 2})

    with st.spinner("Processing PDF... Please wait."):
        retriever = load_and_vectorize_pdf(tmp_file_path)

    st.success("PDF successfully uploaded and processed!")

    # User question
    input_text = st.text_input("Ask a question from your PDF:")

    if input_text:
        # Local LLM (Aap chahay toh larma3.2 use karein)
        llm = ChatOllama(model="llama3.2", temperature=0)

        # ---------------------------------------------------------
        # AGENTIC RAG SETUP USING LANGGRAPH
        # ---------------------------------------------------------

        # 1. Define State for the Agent
        class AgentState(TypedDict):
            question: str
            context: str
            generation: str
            decision: str

        # 2. Node: Retrieve Documents
        def retrieve_node(state: AgentState):
            print("---RETRIEVING CONTEXT---")
            question = state["question"]
            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            return {"context": context}

        # 3. Node: Grade / Decide if retrieval was useful (Agentic Logic)
        def evaluate_and_generate_node(state: AgentState):
            print("---GENERATING ANSWER WITH AGENTIC CHECK---")
            question = state["question"]
            context = state.get("context", "")

            prompt_template = """You are an advanced Agentic RAG assistant. 
            Analyze the context below and answer the question. If the context does not contain the answer, say "I cannot find the answer in the provided document." cleanly.

            Context:
            {context}

            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm | StrOutputParser()
            generation = chain.invoke({"context": context, "question": question})
            
            return {"generation": generation}

        # 4. Build LangGraph Workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("generate", evaluate_and_generate_node)

        # Define Edges (Linear Agentic Flow)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile graph
        app = workflow.compile()

        # Run the Agentic App
        with st.spinner("Agent is thinking and analyzing PDF..."):
            inputs = {"question": input_text}
            result = app.invoke(inputs)
            
            st.write("### Agent Answer:")
            st.write(result["generation"])