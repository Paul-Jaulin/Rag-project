import os
import gradio as gr
from pdf_reader import read_pdf
from text_preprocessor import chunk_text
from llm_interactor import generate_response

dataset_path = "dataset/"  # Ensure this path is correct and contains your PDFs

def process_and_respond(prompt, filename):
    pdf_path = os.path.join(dataset_path, filename)
    text = read_pdf(pdf_path)
    context_chunks = chunk_text(text)
    context = " ".join(context_chunks)  # Using the whole text as context; adjust based on your LLM's limitations
    response = generate_response(prompt, context)
    return response

def get_pdf_filenames():
    """
    Returns a list of PDF filenames in the dataset directory.
    """
    return [f for f in os.listdir(dataset_path) if f.endswith('.pdf')]

interface = gr.Interface(fn=process_and_respond,
                         inputs=[gr.Textbox(placeholder="Enter your prompt here"), gr.Dropdown(choices=get_pdf_filenames(), label="Select Document")],
                         outputs="text",
                         title="LLM Text Generation from PDF Context",
                         description="Select a PDF document and enter a prompt to generate text based on the document's content.")

if __name__ == "__main__":
    interface.launch()
