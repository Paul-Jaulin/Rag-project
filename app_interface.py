# Code in app_interface.py:

import gradio as gr
from model_handler import load_llama_model, generate_text_with_llama

# Load the Llama 2 model
model_path = "path/to/your/llama/model"  # Update this path to your actual model location
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
