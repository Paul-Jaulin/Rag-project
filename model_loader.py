from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name):
    """
    Load the tokenizer and model for text generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model
