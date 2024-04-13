import os
import gradio as gr
from pdf_processor import load_pdf, chunk_text
from embedding_processor import embed_chunks, rag_retrieval
from response_generator import generate_response

dataset_path = "dataset/"
model_options = ["gpt2", "distilgpt2", "sshleifer/tiny-gpt2", "openai-gpt", "microsoft/DialoGPT-small"]

def process_document(model_name, filename, question):
    file_path = os.path.join(dataset_path, filename)
    text = load_pdf(file_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    context, context_idx = rag_retrieval(question, embeddings, chunks)
    response = generate_response(model_name, question, context)
    return context, response

demo = gr.Interface(
    fn=process_document,
    inputs=[
        gr.Dropdown(label="Choose Model", choices=model_options),
        gr.Dropdown(label="Select PDF Document", choices=os.listdir(dataset_path)),
        gr.Textbox(label="Enter your question")
    ],
    outputs=[
        gr.Textbox(label="Context"),
        gr.Textbox(label="Response")
    ],
    title="Document RAG System",
    description="Choose a model, select a PDF document from the dataset, and ask a question. The system will find the relevant context and generate a response."
)

if __name__ == "__main__":
    demo.launch()
