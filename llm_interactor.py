from transformers import pipeline

# Initialize the QnA pipeline with a specific model
qna_pipeline = pipeline("question-answering", model="google/t5-efficient-mini")

def generate_response(question, context_chunks):
    """
    Generates a response using a Q&A model based on the question and the provided context chunks.
    """
    # Combine the context_chunks into a single context string.
    # Note: Depending on your model's max input size, you may need to truncate or select relevant chunks.
    context = " ".join(context_chunks)
    
    # Use the first 512 tokens from the context to avoid model input size limit.
    # Adjust this slicing as needed based on the model's limitations.
    context = context[:512*2]

    # Generate an answer using the Q&A pipeline
    answer = qna_pipeline(question=question, context=context)
    print(answer['answer'])
    return answer['answer']