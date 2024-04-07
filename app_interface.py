"""
import gradio as gr
from model_handler import load_llama_model, generate_text_with_llama

# Load the Llama 2 model
model_path = "llama-2-7b-chat.ggmlv3.q8_0.bin"  # Update this path to your actual model location
llama_model = load_llama_model(model_path)

def gradio_interface(prompt):
    # Generate text using the Llama 2 model
    return generate_text_with_llama(prompt, llama_model)

interface = gr.Interface(fn=gradio_interface,
                         inputs=gr.inputs.Textbox(lines=5, placeholder="Enter Prompt Here"),
                         outputs="text",
                         title="Llama 2 Text Generation Interface",
                         description="This interface generates text responses using the Llama 2 model. Enter a prompt to get started.")

if __name__ == "__main__":
    interface.launch()
"""

import os
import gradio as gr
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from sentence_transformers import SentenceTransformer

# Define paths and parameters
USE_CASE_DATASET_PATH = "path/to/dataset"
MODEL_NAME_RAG = "facebook/rag-token-base"
MODEL_NAME_ST = "sentence-transformers/stsb-roberta-large"
CHUNK_SIZE = 1000
OVERLAPPING = 100

# Initialize tokenizers and models
tokenizer_rag = RagTokenizer.from_pretrained(MODEL_NAME_RAG)
retriever = RagRetriever.from_pretrained(MODEL_NAME_RAG, index_name="compressed", use_dummy_dataset=True)
generator = RagSequenceForGeneration.from_pretrained(MODEL_NAME_RAG)
model_st = SentenceTransformer(MODEL_NAME_ST)


# Function to chunk documents
def chunk_documents(dataset_path, chunk_size, overlapping):
    chunks = []
    for file_name in os.listdir(dataset_path):
        with open(os.path.join(dataset_path, file_name), "r") as file:
            content = file.read()
            for i in range(0, len(content), chunk_size - overlapping):
                chunk = content[i:i + chunk_size]
                chunks.append(chunk)
    return chunks


# Function to interact with LLM
def interact_with_llm(prompt, context_chunks, model_name):
    if model_name == "RAG":
        tokenizer = tokenizer_rag
        generator = generator_rag
    elif model_name == "SentenceTransformers":
        tokenizer = None
        generator = model_st
    else:
        raise ValueError("Invalid model name. Please choose either 'RAG' or 'SentenceTransformers'.")

    inputs = tokenizer(prompt, context_chunks, return_tensors="pt", padding="max_length", max_length=512,
                       truncation=True)
    outputs = generator.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Define Gradio interface
def interact_with_documents(prompt, model_name):
    # Chunk documents
    context_chunks = chunk_documents(USE_CASE_DATASET_PATH, CHUNK_SIZE, OVERLAPPING)
    # Interact with LLM
    response = interact_with_llm(prompt, context_chunks, model_name)
    return response



# Define Gradio interface
gr.Interface(fn=interact_with_documents, inputs=["text", gr.inputs.Dropdown(["RAG", "SentenceTransformers"])],
             outputs="text").launch()

import gradio as gr

# Define Gradio interface
def interact_with_documents(prompt, model_name):
    # Chunk documents
    context_chunks = chunk_documents(USE_CASE_DATASET_PATH, CHUNK_SIZE, OVERLAPPING)
    # Interact with LLM
    response = interact_with_llm(prompt, context_chunks, model_name)
    return response

# Define Gradio interface
gr.Interface(fn=interact_with_documents,
             inputs=["text", gr.inputs.Dropdown(["RAG", "SentenceTransformers"], label="Model")],
             outputs="text",
             title="Document Interaction",
             description="Interact with documents using RAG or Sentence Transformers models."
            ).launch()
