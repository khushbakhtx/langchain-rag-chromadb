import os
import pandas as pd
import streamlit as st
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error(
        "OPENAI_API_KEY is not set. Please add it to your .env file or Streamlit Cloud secrets. "
        "For Streamlit Cloud, go to 'Manage app' > 'Secrets' and add: OPENAI_API_KEY='your-key'"
    )
    st.stop()

def load_csv_documents(csv_paths: List[str]) -> List[Document]:
    documents = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                row_content = ", ".join([f"{col}: {val}" for col, val in row.items()])
                doc = Document(
                    page_content=row_content,
                    metadata={"source": csv_path, "row_index": idx}
                )
                documents.append(doc)
        except Exception as e:
            st.error(f"Error loading {csv_path}: {e}")
            continue
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    vector_store = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
    return vector_store

def load_existing_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def table_query_tool(query: str) -> str:
    prompt = PromptTemplate(
        template="Analyze tabular CSV data to answer: {query}\n\nIf the query involves ARPU (Average Revenue Per User), assume historical data is retrieved separately and provide a forecast based on trends (e.g., average of past values). If data is insufficient, state so and suggest what’s needed (e.g., revenue and user counts). Provide a concise answer.",
        input_variables=["query"]
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_api_key, temperature=0)
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"query": query})
    except Exception as e:
        return f"Error processing query: {e}"

def create_table_query_tool() -> Tool:
    return Tool(
        name="table_query",
        func=lambda query: table_query_tool(query, context=""),  
        description="Analyze tabular CSV data to answer queries, including ARPU calculations or forecasts."
    )

def create_retriever_tool_instance(vector_store: Chroma) -> Tool:
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    return create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search tabular CSV data for relevant rows, especially for revenue and user data."
    )

def create_agent(vector_store: Chroma) -> AgentExecutor:
    llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=openai_api_key, temperature=0)
    tools = [create_retriever_tool_instance(vector_store), create_table_query_tool()]
    
    prompt = PromptTemplate(
        template="Ты ИИ агент для извлечения информации под подзовательский запрос. Используй инструменты чтобы извлечь релевантную информацию через строки и столбцы чтобы дать точный ответ пользователю. For ARPU forecasts, use 'retrieve_documents' to fetch revenue and user data, then pass it to 'table_query' for calculation. Provide a single, concise answer. If data is insufficient, state so and make a reasonable assumption or request clarification. Do not loop excessively; limit to one tool call per tool if possible.\n\nChat History: {chat_history}\n\nQuestion: {question}\n\n{agent_scratchpad}",
        input_variables=["question", "chat_history"],
        partial_variables={"agent_scratchpad": MessagesPlaceholder(variable_name="agent_scratchpad")}
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="output"
    )
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=3, 
        return_intermediate_steps=True
    )

def main():
    st.title("LangChain RAG Agent")
    st.write("Epsilon RAG agent uses OpenAI embedding model and Chroma vector store to process CSV data and answer queries.")

    csv_paths = [
            "csv_data/forecast_yearly.csv",
            "csv_data/full_data.csv",
            "csv_data/operation_expenses.csv"
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "file_names" not in st.session_state:
        st.session_state.file_names = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_initialized" not in st.session_state:
        st.session_state.is_initialized = False

    if not st.session_state.is_initialized:
        with st.spinner("Processing your data and creating vector store..."):
            current_file_names = sorted(csv_paths)
            if current_file_names != st.session_state.file_names or not st.session_state.vector_store:
                st.session_state.file_names = current_file_names
                documents = load_csv_documents(csv_paths)
                if not documents:
                    st.error("No valid CSV data loaded. Check file paths and formats.")
                    return
                split_docs = split_documents(documents)
                st.session_state.vector_store = create_vector_store(split_docs)
                st.session_state.agent_executor = create_agent(st.session_state.vector_store)
                st.session_state.is_initialized = True
                st.success("CSV files processed and vector store created.")
            else:
                st.session_state.vector_store = load_existing_vector_store()
                if not st.session_state.agent_executor:
                    st.session_state.agent_executor = create_agent(st.session_state.vector_store)
                st.session_state.is_initialized = True
                st.info("Using existing vector store.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    placeholder = "Enter your question about the CSV data..." if st.session_state.is_initialized else "Please wait, processing CSV files..."
    query = st.chat_input(placeholder, disabled=not st.session_state.is_initialized)

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        try:
            with st.spinner("Processing query..."):
                response = st.session_state.agent_executor.invoke({"question": query})
                answer = response["output"]
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
        except Exception as e:
            st.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()