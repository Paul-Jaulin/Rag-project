import gradio as gr
import pandas as pd
import numpy as np
from transformers import pipeline, set_seed
from sentence_transformers import SentenceTransformer, util
import os
import torch


# Define global variables for the model and tokenizer
MODEL_NAME = "mosaicml/mpt-7b"

# Initialize the pipeline
qna_pipeline = pipeline("text-generation", model=MODEL_NAME)

# Define function to load the PDF and convert it to text
def load_pdf(file_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"  # Extract text from each page
    return text

# Define function to chunk the text
def chunk_text(text, chunk_size=2048, overlap=0):
    text_length = len(text)
    chunks = []
    for i in range(0, text_length, chunk_size-overlap):
        end = min(i+chunk_size, text_length)
        chunks.append(text[i:end])
    return chunks

# Define function to embed chunks using SentenceTransformer
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L12-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Define RAG function
def rag_retrieval(question, embeddings, chunks):
    query_embedding = embed_chunks([question])[0]  # Embed the question
    scores = util.cos_sim(query_embedding, embeddings)[0]  # Compute similarity scores
    top_result_idx = torch.argmax(scores).item()  # Find the index of the chunk with the highest score
    top_chunk = chunks[top_result_idx]  # Retrieve the corresponding chunk
    return top_chunk, top_result_idx

# Define function to generate response from RAG
def generate_response(question, context):
    # Specify the maximum number of tokens to generate after the input
    max_new_tokens = 150  # For example, generate up to 100 new tokens beyond the input
    answer = qna_pipeline( question=question, context=context, max_size=150)
    print('\n\n')
    print(answer,'\n\n')
    return answer['answer']

# Adjust the return statement of the process_document function to match the expected output format:
def process_document(file, question):
    text = load_pdf(file)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    context, context_idx = rag_retrieval(question, embeddings, chunks)
    response = generate_response(question, context)
    return context, response  # Return each output separately

demo = gr.Interface(
    fn=process_document,
    inputs=[
        gr.File(label="Upload PDF Document"),  # Updated to gr.File
        gr.Textbox(label="Enter your question")  # Updated to gr.Textbox
    ],
    outputs=[
        gr.Textbox(label="Context"),  # Correct as is
        gr.Textbox(label="Response")  # Correct as is
    ],
    title="Document RAG System",
    description="Upload a PDF document and ask a question. The system will use RAG to find the relevant context and generate a response."
)

if __name__ == "__main__":
    demo.launch()
