from sentence_transformers import SentenceTransformer, util
import torch

def embed_chunks(chunks):
    """
    Embeds text chunks using Sentence Transformers.
    """
    model = SentenceTransformer('all-MiniLM-L12-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def rag_retrieval(question, embeddings, chunks):
    """
    Retrieves the most relevant chunk based on the question.
    """
    query_embedding = embed_chunks([question])[0]  # Embed the question
    scores = util.cos_sim(query_embedding, embeddings)[0]  # Compute similarity scores
    top_result_idx = torch.argmax(scores).item()  # Find the index of the chunk with the highest score
    top_chunk = chunks[top_result_idx]  # Retrieve the corresponding chunk
    return top_chunk, top_result_idx
