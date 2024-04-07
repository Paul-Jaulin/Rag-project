import os
import gradio as gr
from pdf_reader import read_pdf
from text_preprocessor import chunk_text
from model_selector import encode_text, model_choices  # Assuming model_choices is defined here
from llm_interactor import generate_response

dataset_path = "dataset/"  # Ensure this path is correct and contains your PDFs

def process_and_respond(prompt, filename, model_name):
    pdf_path = os.path.join(dataset_path, filename)
    text = read_pdf(pdf_path)
    context_chunks = chunk_text(text)
    # Now using model_name to select and encode with the chosen model
    encoded_chunks = encode_text(model_name, context_chunks)  # Adjusted to encode directly
    # Since generate_response expects textual context, either adjust it to use encoded_chunks
    # or select the textual context based on the encoding process
    response = generate_response(prompt, text)  # Using full text for simplification
    return response

interface = gr.Interface(fn=process_and_respond,
                         inputs=[
                             gr.Textbox(placeholder="Enter your prompt here"),
                             gr.Dropdown(choices=os.listdir(dataset_path), label="Select Document"),
                             gr.Dropdown(choices=model_choices, label="Select Model")
                         ],
                         outputs="text",
                         title="Text QnA from PDF Context",
                         description="Select a PDF document, choose a model, and enter a prompt to generate an answer based on the document's content.")

if __name__ == "__main__":
    interface.launch(share=True)
