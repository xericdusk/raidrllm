import os
import json
import tempfile
import streamlit as st
import requests
import io
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

def process_url_for_training(urls: List[str]) -> List[str]:
    """
    Download and process content from URLs for training data.
    
    Args:
        urls: List of URLs to download content from
        
    Returns:
        List of file paths where the downloaded content is saved
    """
    if "temp_dir" not in st.session_state:
        st.session_state["temp_dir"] = tempfile.mkdtemp()
    
    temp_dir = st.session_state["temp_dir"]
    file_paths = []
    
    for i, url in enumerate(urls):
        try:
            # Download content from URL
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            content = response.text
            
            # Determine file type based on content or headers
            content_type = response.headers.get('Content-Type', '')
            
            if 'json' in content_type:
                file_ext = '.json'
                # Try to parse as JSON to validate
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    # If not valid JSON, try JSONL format
                    try:
                        for line in content.splitlines():
                            if line.strip():
                                json.loads(line)
                        file_ext = '.jsonl'
                    except json.JSONDecodeError:
                        file_ext = '.txt'  # Default to text if parsing fails
            elif 'csv' in content_type:
                file_ext = '.csv'
            elif 'pdf' in content_type or url.lower().endswith('.pdf'):
                file_ext = '.pdf'
                # For PDFs, we need to save the binary content, not the text
                content = response.content
                # Use binary mode for writing
                binary_mode = True
            else:
                file_ext = '.txt'  # Default to text
                binary_mode = False
            
            # Generate a filename based on the URL
            filename = f"url_content_{i}{file_ext}"
            file_path = os.path.join(temp_dir, filename)
            
            # Save content to file - use binary mode for PDFs
            if 'binary_mode' in locals() and binary_mode:
                with open(file_path, 'wb') as f:
                    f.write(content)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            file_paths.append(file_path)
            
        except Exception as e:
            st.error(f"Error processing URL {url}: {str(e)}")
    
    # Add the new file paths to the existing ones
    if "uploaded_file_paths" not in st.session_state:
        st.session_state["uploaded_file_paths"] = []
    
    st.session_state["uploaded_file_paths"].extend(file_paths)
    
    return file_paths

def process_pdf_for_training(file_path: str) -> str:
    """
    Extract text from a PDF file and format it for training.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    try:
        from PyPDF2 import PdfReader
        
        # Read the PDF file
        with open(file_path, "rb") as f:
            pdf = PdfReader(f)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            return text.strip()
    
    except Exception as e:
        st.error(f"Error processing PDF {os.path.basename(file_path)}: {str(e)}")
        return ""

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
            
            elif file_ext == ".yaml" or file_ext == ".yml":
                import yaml
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, list):
                        for item in data:
                            formatted_data.append(format_item(item, format_type))
                    else:
                        formatted_data.append(format_item(data, format_type))
            
            elif file_ext == ".pdf":
                # Process PDF file
                text = process_pdf_for_training(file_path)
                if text:
                    # Attempt to structure the PDF content based on the format type
                    if format_type == "Instruction-Response":
                        # Try to split the text into sections that might represent instruction/response pairs
                        sections = [s.strip() for s in text.split('\n\n') if s.strip()]
                        
                        # If we have at least two sections, treat them as instruction/response pairs
                        if len(sections) >= 2:
                            for i in range(0, len(sections) - 1, 2):
                                if i + 1 < len(sections):
                                    formatted_data.append({
                                        "instruction": sections[i],
                                        "response": sections[i + 1]
                                    })
                        else:
                            # If we can't split into pairs, treat the whole text as a single item
                            formatted_data.append({"text": text})
                    
                    elif format_type == "Chat":
                        # Try to identify conversation patterns in the text
                        lines = text.split('\n')
                        conversation = []
                        current_role = None
                        current_content = []
                        
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                                
                            # Check if line starts with a role identifier
                            if line.lower().startswith(('user:', 'human:', 'question:')):
                                # If we were building a previous message, add it
                                if current_role and current_content:
                                    conversation.append({
                                        "role": current_role,
                                        "content": '\n'.join(current_content).strip()
                                    })
                                    current_content = []
                                
                                current_role = "user"
                                current_content.append(line.split(':', 1)[1].strip())
                            
                            elif line.lower().startswith(('assistant:', 'ai:', 'answer:', 'response:')):
                                # If we were building a previous message, add it
                                if current_role and current_content:
                                    conversation.append({
                                        "role": current_role,
                                        "content": '\n'.join(current_content).strip()
                                    })
                                    current_content = []
                                
                                current_role = "assistant"
                                current_content.append(line.split(':', 1)[1].strip())
                            
                            else:
                                # Continue with the current role
                                if current_role:
                                    current_content.append(line)
                                else:
                                    # If no role has been identified yet, assume it's user
                                    current_role = "user"
                                    current_content.append(line)
                        
                        # Add the last message if there is one
                        if current_role and current_content:
                            conversation.append({
                                "role": current_role,
                                "content": '\n'.join(current_content).strip()
                            })
                        
                        # If we have a conversation with at least one exchange, add it
                        if len(conversation) >= 2:
                            formatted_data.append({"messages": conversation})
                        else:
                            # If we couldn't identify a conversation, treat as completion
                            formatted_data.append({"text": text})
                    
                    else:  # Completion
                        formatted_data.append({"text": text})
            
            elif file_ext == ".md":
                with open(file_path, "r") as f:
                    content = f.read()
                    # Process markdown - extract code blocks or sections
                    import re
                    # Find code blocks (```...```)
                    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
                    if code_blocks:
                        for block in code_blocks:
                            if format_type == "Completion":
                                formatted_data.append({"text": block.strip()})
                            else:
                                # Try to determine if it's a conversation or instruction
                                if ":" in block and ("user" in block.lower() or "assistant" in block.lower()):
                                    # Likely a conversation
                                    messages = []
                                    for line in block.split('\n'):
                                        if line.strip():
                                            if line.lower().startswith("user:"):
                                                messages.append({
                                                    "role": "user",
                                                    "content": line[5:].strip()
                                                })
                                            elif line.lower().startswith("assistant:"):
                                                messages.append({
                                                    "role": "assistant",
                                                    "content": line[10:].strip()
                                                })
                                    if messages:
                                        formatted_data.append({"messages": messages})
                                else:
                                    # Treat as instruction-response
                                    parts = block.split('\n\n', 1)
                                    if len(parts) > 1:
                                        formatted_data.append({
                                            "instruction": parts[0].strip(),
                                            "response": parts[1].strip()
                                        })
                                    else:
                                        formatted_data.append({"text": block.strip()})
                    else:
                        # Process as regular text
                        formatted_data.append({"text": content.strip()})
            
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

def process_url(url: str) -> str:
    """
    Process a URL and return the content.
    
    Args:
        url: URL to process
        
    Returns:
        Content of the URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error processing URL: {str(e)}")
        return None

def test_fine_tuned_model(prompt: str, model_path: str) -> str:
    """
    Test a fine-tuned model with a given prompt.
    
    Args:
        prompt: The input prompt to test with
        model_path: Path to the fine-tuned model
        
    Returns:
        Generated text from the model
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from peft import PeftModel, PeftConfig
        
        # Create a progress placeholder
        progress_placeholder = st.empty()
        progress_placeholder.info("Loading fine-tuned model...")
        
        # Load the base model first
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load the fine-tuned model
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        
        # Set up the generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        progress_placeholder.info("Generating response...")
        
        # Generate text based on the prompt
        result = pipe(prompt)[0]["generated_text"]
        
        # Clear the progress placeholder
        progress_placeholder.empty()
        
        return result
    
    except Exception as e:
        st.error(f"Error testing fine-tuned model: {str(e)}")
        return ""

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
            output_dir = f"./fine_tuned_model_{int(time.time())}"
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save the model path in session state for easy access in testing
            st.session_state["fine_tuned_model_path"] = output_dir
            
            update_progress(f"Fine-tuning completed! Model saved to {output_dir}")
        
        except Exception as e:
            update_progress(f"Error during fine-tuning: {str(e)}")
    
    # Start the fine-tuning process in a separate thread
    thread = threading.Thread(target=run_fine_tuning)
    thread.start()
    
    return "Fine-tuning process started in the background."
