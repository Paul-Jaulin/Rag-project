import os
import gradio as gr
from pdf_processor import load_pdf, chunk_text
from embedding_processor import embed_chunks, rag_retrieval
from response_generator import generate_response

dataset_path = "dataset/"
model_options = ["gpt2", "distilgpt2", "sshleifer/tiny-gpt2", "openai-gpt", "microsoft/DialoGPT-small"]

def process_document(model_name, filename, question, chunk_size, overlap_size):
    file_path = os.path.join(dataset_path, filename)
    text = load_pdf(file_path)
    chunks = chunk_text(text, chunk_size, overlap_size)  # Assuming chunk_text can handle these params
    embeddings = embed_chunks(chunks)
    context, context_idx = rag_retrieval(question, embeddings, chunks)
    response = generate_response(model_name, question, context)
    return context, response

demo = gr.Interface(
    fn=process_document,
    inputs=[
        gr.Dropdown(label="Choose Model", choices=model_options),
        gr.Dropdown(label="Select PDF Document", choices=os.listdir(dataset_path)),
        gr.Textbox(label="Enter your question"),
        gr.Slider(label="Chunk Size", minimum=128, maximum=1024, step=128, value=512),
        gr.Slider(label="Overlap Size", minimum=0, maximum=500, step=10, value=100)
    ],
    outputs=[
        gr.Textbox(label="Context"),
        gr.Textbox(label="Response")
    ]
)

if __name__ == "__main__":
    demo.launch()