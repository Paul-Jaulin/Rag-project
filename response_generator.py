from transformers import pipeline
from old_model_loader import load_model

def generate_response(model_name, prompt, context):
    tokenizer, model = load_model(model_name)
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    input_text = f"{prompt} {context}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)


    # Set dynamic max_length based on desired output length
    max_length = input_ids.shape[1] + 50  # Allows 30 new tokens beyond the input
    responses = text_gen_pipeline(input_text, max_length=max_length, num_return_sequences=1)

    final_text = ' '.join(responses[0]['generated_text'].split())

    return final_text
