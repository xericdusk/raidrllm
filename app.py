import streamlit as st
import os
from dotenv import load_dotenv

# Import utility modules
from utils.fine_tuning import (
    upload_training_data,
    format_training_data,
    start_fine_tuning
)
from utils.rag import (
    upload_documents,
    create_collection,
    query_collection
)

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAIDRLLM - Mistral 8b Fine-tuning & RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
st.sidebar.title("RAIDRLLM")
st.sidebar.subheader("Mistral 8b Fine-tuning & RAG")

# Navigation
page = st.sidebar.radio(
    "Navigate to",
    ["Fine-tuning", "RAG Collection Builder"]
)

# Fine-tuning page
if page == "Fine-tuning":
    st.title("Mistral 8b Fine-tuning")
    
    # Create tabs for different fine-tuning steps
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Format Data", "Start Fine-tuning"])
    
    with tab1:
        st.header("Upload Training Data")
        st.write("Upload files containing training data for fine-tuning.")
        
        uploaded_files = st.file_uploader(
            "Upload training data files",
            accept_multiple_files=True,
            type=["txt", "csv", "json", "jsonl"]
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                upload_training_data(uploaded_files)
                st.success(f"Successfully uploaded {len(uploaded_files)} files.")
    
    with tab2:
        st.header("Format Training Data")
        st.write("Format your training data for optimal fine-tuning results.")
        
        format_options = st.selectbox(
            "Select data format",
            ["Instruction-Response", "Chat", "Completion"]
        )
        
        if st.button("Format Data"):
            formatted_data = format_training_data(format_options)
            if formatted_data:
                st.success("Data formatted successfully!")
                st.download_button(
                    "Download Formatted Data",
                    formatted_data,
                    file_name="formatted_training_data.jsonl",
                    mime="application/json"
                )
    
    with tab3:
        st.header("Start Fine-tuning")
        st.write("Configure and start the fine-tuning process.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input("Base Model", value="mistralai/Mistral-7B-v0.1")
            num_epochs = st.slider("Number of Epochs", min_value=1, max_value=10, value=3)
            learning_rate = st.text_input("Learning Rate", value="2e-5")
            
        with col2:
            batch_size = st.slider("Batch Size", min_value=1, max_value=32, value=8)
            gradient_accumulation_steps = st.slider("Gradient Accumulation Steps", min_value=1, max_value=16, value=4)
            lora_r = st.slider("LoRA Rank (r)", min_value=4, max_value=256, value=16)
            lora_alpha = st.slider("LoRA Alpha", min_value=1, max_value=128, value=32)
        
        if st.button("Start Fine-tuning"):
            fine_tuning_config = {
                "model_name": model_name,
                "num_epochs": num_epochs,
                "learning_rate": float(learning_rate),
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha
            }
            
            start_fine_tuning(fine_tuning_config)
            st.success("Fine-tuning process started!")

# RAG Collection Builder page
elif page == "RAG Collection Builder":
    st.title("RAG Collection Builder")
    
    # Create tabs for different RAG steps
    tab1, tab2, tab3 = st.tabs(["Upload Documents", "Create Collection", "Query Collection"])
    
    with tab1:
        st.header("Upload Documents")
        st.write("Upload documents to be included in your RAG collection.")
        
        uploaded_docs = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=["txt", "pdf", "docx", "md"]
        )
        
        if uploaded_docs:
            if st.button("Process Documents"):
                upload_documents(uploaded_docs)
                st.success(f"Successfully uploaded {len(uploaded_docs)} documents.")
    
    with tab2:
        st.header("Create Collection")
        st.write("Create a new RAG collection from your uploaded documents.")
        
        collection_name = st.text_input("Collection Name", value="my_rag_collection")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
        )
        
        if st.button("Create Collection"):
            create_collection(collection_name, embedding_model)
            st.success(f"Collection '{collection_name}' created successfully!")
    
    with tab3:
        st.header("Query Collection")
        st.write("Test your RAG collection with queries.")
        
        available_collections = ["my_rag_collection"]  # This would be dynamically populated
        selected_collection = st.selectbox("Select Collection", available_collections)
        
        query = st.text_input("Enter your query")
        
        if query and st.button("Submit Query"):
            results = query_collection(selected_collection, query)
            st.subheader("Results")
            for i, result in enumerate(results, 1):
                st.markdown(f"**Result {i}**")
                st.write(result)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 RAIDRLLM")
