from sentence_transformers import SentenceTransformer

model_st = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def chunk_text(text, chunk_size=1024):
    """
    Splits the text into manageable chunks.
    """
    tokens = text.split()
    chunks = [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

def encode_chunks(chunks):
    """
    Encodes each chunk into a high-dimensional vector using SentenceTransformer.
    """
    return model_st.encode(chunks)
