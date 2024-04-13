# Code in app_interface.py:

import os
import time
import gradio as gr
from pdf_processor import load_pdf, chunk_text
from embedding_processor import embed_chunks, rag_retrieval
from response_generator import generate_response

dataset_path = "dataset/"  # Path to the dataset folder

def process_document(filename, question):
    """
    Processes the selected PDF document and generates a response based on a question,
    with detailed timing and logging for each processing step.
    """
    print("Starting the document processing pipeline...")
    start_time = time.time()
    
    file_path = os.path.join(dataset_path, filename)
    
    # Load PDF and extract text
    print("Loading PDF...")
    load_start = time.time()
    text = load_pdf(file_path)
    load_duration = time.time() - load_start
    print(f"PDF Loaded. Duration: {load_duration:.2f} seconds")
    
    # Chunk the extracted text
    print("Chunking text...")
    chunk_start = time.time()
    chunks = chunk_text(text)
    chunk_duration = time.time() - chunk_start
    print(f"Text chunked. Duration: {chunk_duration:.2f} seconds")
    
    # Embed the chunks for context retrieval
    print("Embedding text chunks...")
    embed_start = time.time()
    embeddings = embed_chunks(chunks)
    embed_duration = time.time() - embed_start
    print(f"Chunks embedded. Duration: {embed_duration:.2f} seconds")
    
    # Retrieve the most relevant context
    print("Retrieving context...")
    retrieval_start = time.time()
    context, context_idx = rag_retrieval(question, embeddings, chunks)
    retrieval_duration = time.time() - retrieval_start
    print(f"Context retrieved. Duration: {retrieval_duration:.2f} seconds")
    
    # Generate a response based on the context
    print("Generating response...")
    response_start = time.time()
    response = generate_response(question, context)
    response_duration = time.time() - response_start
    print(f"Response generated. Duration: {response_duration:.2f} seconds")
    
    total_duration = time.time() - start_time
    print(f"Total processing time: {total_duration:.2f} seconds")
    
    return context, response

demo = gr.Interface(
    fn=process_document,
    inputs=[
        gr.Dropdown(label="Select PDF Document", choices=os.listdir(dataset_path)),
        gr.Textbox(label="Enter your question")
    ],
    outputs=[
        gr.Textbox(label="Context"),
        gr.Textbox(label="Response")
    ],
    title="Document RAG System",
    description="Select a PDF document from the dataset and ask a question. The system will use RAG to find the relevant context and generate a response."
)

if __name__ == "__main__":
    demo.launch()
