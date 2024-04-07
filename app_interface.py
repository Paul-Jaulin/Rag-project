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

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('How big is London')
print(query_embedding)