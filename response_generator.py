# Code in response_generator.py:

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize the model and tokenizer
MODEL_NAME = "mosaicml/mpt-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Ensure that pad_token_id is set to eos_token_id for proper sequence ending
model.config.pad_token_id = tokenizer.eos_token_id

# Initialize the text generation pipeline with the configured model
text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt, context):
    """
    Generates a response using the model pipeline based on the prompt and context.
    Combines prompt and context into a single input text and handles generation.
    """
    input_text = prompt + " " + context  # Combine prompt and context into a single text input
    responses = text_gen_pipeline(input_text, max_new_tokens=150)  # Generate up to 150 new tokens
    return responses[0]['generated_text']  # Assuming we want the first generated response
