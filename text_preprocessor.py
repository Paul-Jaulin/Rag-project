def chunk_text(text, chunk_size=1024):
    """
    Splits the text into manageable chunks.
    """
    tokens = text.split()
    chunks = [" ".join(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks
