import os
import json
import tempfile
import streamlit as st
from typing import List, Dict, Any, Optional
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from trl import SFTTrainer

def upload_training_data(uploaded_files):
    """
    Process and save uploaded training data files.
    
    Args:
        uploaded_files: List of uploaded files from Streamlit
    """
    # Create a temporary directory to store uploaded files
    temp_dir = tempfile.mkdtemp()
    st.session_state["temp_dir"] = temp_dir
    
    # Save uploaded files to the temporary directory
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Store file paths in session state
    st.session_state["uploaded_file_paths"] = [
        os.path.join(temp_dir, uploaded_file.name)
        for uploaded_file in uploaded_files
    ]
    
    return st.session_state["uploaded_file_paths"]

def format_training_data(format_type: str) -> Optional[str]:
    """
    Format the uploaded training data based on the selected format type.
    
    Args:
        format_type: Type of formatting to apply (Instruction-Response, Chat, Completion)
        
    Returns:
        Formatted data as a JSON string or None if no data is available
    """
    if "uploaded_file_paths" not in st.session_state:
        st.warning("Please upload training data files first.")
        return None
    
    formatted_data = []
    
    for file_path in st.session_state["uploaded_file_paths"]:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == ".json" or file_ext == ".jsonl":
                with open(file_path, "r") as f:
                    if file_ext == ".json":
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                formatted_data.append(format_item(item, format_type))
                        else:
                            formatted_data.append(format_item(data, format_type))
                    else:  # .jsonl
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                formatted_data.append(format_item(item, format_type))
            
            elif file_ext == ".csv":
                import csv
                with open(file_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        formatted_data.append(format_item(row, format_type))
            
            elif file_ext == ".txt":
                with open(file_path, "r") as f:
                    content = f.read()
                    # Simple parsing for text files - assumes alternating instruction/response
                    # This is a basic implementation and might need customization
                    if format_type == "Instruction-Response":
                        parts = content.split("\n\n")
                        for i in range(0, len(parts) - 1, 2):
                            if i + 1 < len(parts):
                                formatted_data.append({
                                    "instruction": parts[i].strip(),
                                    "response": parts[i + 1].strip()
                                })
                    elif format_type == "Chat":
                        # Assume format like "User: ... Assistant: ..."
                        conversations = []
                        current_conversation = []
                        
                        for line in content.split("\n"):
                            line = line.strip()
                            if not line:
                                if current_conversation:
                                    conversations.append(current_conversation)
                                    current_conversation = []
                                continue
                                
                            if line.startswith("User:"):
                                current_conversation.append({
                                    "role": "user",
                                    "content": line[5:].strip()
                                })
                            elif line.startswith("Assistant:"):
                                current_conversation.append({
                                    "role": "assistant",
                                    "content": line[10:].strip()
                                })
                        
                        if current_conversation:
                            conversations.append(current_conversation)
                        
                        for conversation in conversations:
                            formatted_data.append({"messages": conversation})
                    else:  # Completion
                        formatted_data.append({"text": content.strip()})
        
        except Exception as e:
            st.error(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
    
    # Save formatted data to a temporary file
    if formatted_data:
        formatted_file_path = os.path.join(st.session_state["temp_dir"], "formatted_data.jsonl")
        with open(formatted_file_path, "w") as f:
            for item in formatted_data:
                f.write(json.dumps(item) + "\n")
        
        st.session_state["formatted_data_path"] = formatted_file_path
        
        # Return the formatted data as a string
        return "\n".join([json.dumps(item) for item in formatted_data])
    
    return None

def format_item(item: Dict[str, Any], format_type: str) -> Dict[str, Any]:
    """
    Format a single data item based on the selected format type.
    
    Args:
        item: The data item to format
        format_type: Type of formatting to apply
        
    Returns:
        Formatted data item
    """
    if format_type == "Instruction-Response":
        # Try to map common field names to instruction/response format
        instruction = None
        response = None
        
        for field in ["instruction", "input", "question", "prompt"]:
            if field in item:
                instruction = item[field]
                break
        
        for field in ["response", "output", "answer", "completion"]:
            if field in item:
                response = item[field]
                break
        
        if instruction is not None and response is not None:
            return {
                "instruction": instruction,
                "response": response
            }
    
    elif format_type == "Chat":
        # Try to map to chat format
        if "messages" in item:
            return {"messages": item["messages"]}
        elif "conversation" in item:
            return {"messages": item["conversation"]}
        else:
            # Try to create a simple conversation
            messages = []
            
            if any(field in item for field in ["instruction", "input", "question", "prompt"]):
                for field in ["instruction", "input", "question", "prompt"]:
                    if field in item:
                        messages.append({
                            "role": "user",
                            "content": item[field]
                        })
                        break
            
            if any(field in item for field in ["response", "output", "answer", "completion"]):
                for field in ["response", "output", "answer", "completion"]:
                    if field in item:
                        messages.append({
                            "role": "assistant",
                            "content": item[field]
                        })
                        break
            
            if messages:
                return {"messages": messages}
    
    elif format_type == "Completion":
        # Try to map to completion format
        for field in ["text", "content", "completion"]:
            if field in item:
                return {"text": item[field]}
        
        # If instruction and response are available, concatenate them
        if "instruction" in item and "response" in item:
            return {"text": f"{item['instruction']}\n{item['response']}"}
    
    # If we couldn't format properly, return the item as is
    return item

def start_fine_tuning(config: Dict[str, Any]):
    """
    Start the fine-tuning process with the given configuration.
    
    Args:
        config: Dictionary containing fine-tuning configuration
    """
    if "formatted_data_path" not in st.session_state:
        st.warning("Please format your training data first.")
        return
    
    # Load the formatted data
    with open(st.session_state["formatted_data_path"], "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    # Create a progress placeholder
    progress_placeholder = st.empty()
    progress_placeholder.info("Preparing for fine-tuning...")
    
    # Define a function to update progress
    def update_progress(message):
        progress_placeholder.info(message)
    
    # Start fine-tuning in a separate thread to not block the UI
    import threading
    
    def run_fine_tuning():
        try:
            update_progress("Loading model and tokenizer...")
            
            # Load model and tokenizer
            model_name = config["model_name"]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with quantization for efficiency
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Prepare model for training
            update_progress("Preparing model for training...")
            model = prepare_model_for_kbit_training(model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Get PEFT model
            model = get_peft_model(model, lora_config)
            
            # Set up training arguments
            output_dir = "./fine_tuned_model"
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=config["num_epochs"],
                per_device_train_batch_size=config["batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                learning_rate=config["learning_rate"],
                weight_decay=0.01,
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="no",
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                fp16=True,
                report_to="none"
            )
            
            # Determine formatting type based on data
            if "messages" in data[0]:
                formatting_func = "chatml"
            elif "instruction" in data[0] and "response" in data[0]:
                formatting_func = "alpaca"
            else:
                formatting_func = None
            
            # Set up SFT trainer
            update_progress("Setting up trainer...")
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                args=training_args,
                tokenizer=tokenizer,
                peft_config=lora_config,
                formatting_func=formatting_func,
                max_seq_length=2048
            )
            
            # Start training
            update_progress("Starting fine-tuning process...")
            trainer.train()
            
            # Save the model
            update_progress("Saving fine-tuned model...")
            trainer.save_model(output_dir)
            
            update_progress("Fine-tuning completed successfully!")
        
        except Exception as e:
            update_progress(f"Error during fine-tuning: {str(e)}")
    
    # Start the fine-tuning process in a separate thread
    thread = threading.Thread(target=run_fine_tuning)
    thread.start()
    
    return "Fine-tuning process started in the background."
