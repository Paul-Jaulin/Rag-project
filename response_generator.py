from transformers import pipeline
from model_loader import load_model

def generate_response(model_name, prompt, context):
    """
    Generates a response using a text generation model based on the prompt and context.
    """
    tokenizer, model = load_model(model_name)
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    input_text = f"{prompt} {context}"
    responses = text_gen_pipeline(input_text, max_new_tokens=150)
    return responses[0]['generated_text']
