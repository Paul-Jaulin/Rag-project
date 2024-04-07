# Code in model_selector.py:

from sentence_transformers import SentenceTransformer

# Define available model choices
model_choices = ["mistralai/Mistral-7B-Instruct-v0.2", "multi-qa-distilbert-dot-v1", "multi-qa-mpnet-base-dot-v1", "msmarco-distilbert-dot-v5", "msmarco-MiniLM-L6-cos-v5"]

# Dictionary mapping model names to their respective SentenceTransformer instances
models = {name: SentenceTransformer(name) for name in model_choices}

def encode_text(model_name, text_chunks):
    """
    Encodes the given text chunks using the specified model.
    """
    model = models.get(model_name, None)
    if model:
        return model.encode(text_chunks)
    else:
        return None
