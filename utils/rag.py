import os
import tempfile
import streamlit as st
from typing import List, Dict, Any, Optional
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)

def upload_documents(uploaded_docs):
    """
    Process and save uploaded documents for RAG.
    
    Args:
        uploaded_docs: List of uploaded documents from Streamlit
    """
    # Create a temporary directory to store uploaded documents if it doesn't exist
    if "rag_temp_dir" not in st.session_state:
        st.session_state["rag_temp_dir"] = tempfile.mkdtemp()
    
    # Save uploaded documents to the temporary directory
    for uploaded_doc in uploaded_docs:
        file_path = os.path.join(st.session_state["rag_temp_dir"], uploaded_doc.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_doc.getbuffer())
    
    # Store document paths in session state
    st.session_state["uploaded_doc_paths"] = [
        os.path.join(st.session_state["rag_temp_dir"], uploaded_doc.name)
        for uploaded_doc in uploaded_docs
    ]
    
    return st.session_state["uploaded_doc_paths"]

def load_document(file_path: str):
    """
    Load a document based on its file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of document chunks
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == ".txt":
            loader = TextLoader(file_path)
        elif file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {file_ext}")
            return []
        
        return loader.load()
    
    except Exception as e:
        st.error(f"Error loading document {os.path.basename(file_path)}: {str(e)}")
        return []

def create_collection(collection_name: str, embedding_model: str):
    """
    Create a RAG collection from uploaded documents.
    
    Args:
        collection_name: Name of the collection to create
        embedding_model: Name of the embedding model to use
    """
    if "uploaded_doc_paths" not in st.session_state:
        st.warning("Please upload documents first.")
        return
    
    # Create a progress placeholder
    progress_placeholder = st.empty()
    progress_placeholder.info("Creating RAG collection...")
    
    # Load documents
    documents = []
    for doc_path in st.session_state["uploaded_doc_paths"]:
        doc_chunks = load_document(doc_path)
        documents.extend(doc_chunks)
    
    if not documents:
        progress_placeholder.error("No valid documents found.")
        return
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Create vector store
    persist_directory = os.path.join(os.getcwd(), "rag_collections", collection_name)
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the vector store
    vectorstore.persist()
    
    # Store collection info in session state
    if "rag_collections" not in st.session_state:
        st.session_state["rag_collections"] = {}
    
    st.session_state["rag_collections"][collection_name] = {
        "persist_directory": persist_directory,
        "embedding_model": embedding_model,
        "document_count": len(chunks)
    }
    
    progress_placeholder.success(f"Collection '{collection_name}' created with {len(chunks)} chunks.")
    return persist_directory

def query_collection(collection_name: str, query: str, k: int = 5):
    """
    Query a RAG collection.
    
    Args:
        collection_name: Name of the collection to query
        query: Query string
        k: Number of results to return
        
    Returns:
        List of results
    """
    if "rag_collections" not in st.session_state or collection_name not in st.session_state["rag_collections"]:
        st.warning(f"Collection '{collection_name}' not found.")
        return []
    
    collection_info = st.session_state["rag_collections"][collection_name]
    persist_directory = collection_info["persist_directory"]
    embedding_model = collection_info["embedding_model"]
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Load vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Query the vector store
    results = vectorstore.similarity_search(query, k=k)
    
    # Format results
    formatted_results = []
    for i, doc in enumerate(results):
        formatted_results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "similarity": doc.metadata.get("similarity", "N/A")
        })
    
    return formatted_results

def list_collections():
    """
    List all available RAG collections.
    
    Returns:
        List of collection names
    """
    if "rag_collections" not in st.session_state:
        return []
    
    return list(st.session_state["rag_collections"].keys())

def delete_collection(collection_name: str):
    """
    Delete a RAG collection.
    
    Args:
        collection_name: Name of the collection to delete
    """
    if "rag_collections" not in st.session_state or collection_name not in st.session_state["rag_collections"]:
        st.warning(f"Collection '{collection_name}' not found.")
        return False
    
    collection_info = st.session_state["rag_collections"][collection_name]
    persist_directory = collection_info["persist_directory"]
    
    # Delete the collection directory
    import shutil
    try:
        shutil.rmtree(persist_directory)
        del st.session_state["rag_collections"][collection_name]
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        return False
