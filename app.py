import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Page setup
st.set_page_config(page_title="Contract AI Assistant", page_icon="📄")
st.title("📄 Smart Contract AI Assistant (Data-Optimized)")
st.write("Ask anything about specifications, weights, or values in your contract.")

# Securely grab the API key from Streamlit's settings dashboard
if "GOOGLE_API_KEY" not in os.environ:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets.")
        st.stop()

# Initialize chatbot logic once so it stays fast
@st.cache_resource
def load_rag_system():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    
    # Loads the updated vector store from the folder we uploaded
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1) # Lower temperature for numerical precision
    
    context_ualize_q_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("ai", "Generate a precise keyword search query to look up exact contract data or metrics."),
    ])
    
    # Dig deeper (k=8) to ensure we fetch hidden specs and technical table segments
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(search_kwargs={"k": 8}), context_ualize_q_prompt
    )
    
    # Prompt updated to prioritize exact numbers, metrics, and data layouts
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert contract auditor. Answer the user's questions with absolute factual precision using ONLY the provided context below. "
                   "Pay close attention to numerical values, weights, metrics, and specific equipment labels. "
                   "If the context contains technical details or rows from a data table, extract the numbers exactly as written.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

rag_chain = load_rag_system()

# Handle conversational chat history interface
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User types a question
if user_query := st.chat_input("Ask for specifications, weights, or contract details..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Run through the RAG chain
    with st.chat_message("assistant"):
        response = rag_chain.invoke({
            "input": user_query,
            "chat_history": st.session_state.chat_history
        })
        answer = response["answer"]
        st.markdown(answer)
        
    # Keep track of history for multi-turn conversations
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.extend([
        ("human", user_query),
        ("ai", answer)
    ])