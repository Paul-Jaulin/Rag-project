from transformers import pipeline

# Initialize a QnA pipeline with a pre-trained model
qna_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def generate_response(question, context_text):
    """
    Generates an answer using a QnA model based on the question and the provided textual context.
    """
    # Generate an answer using the QnA pipeline
    answer = qna_pipeline(question=question, context=context_text)
    return answer['answer']

# Note: `context_text` should be the textual content you've deemed most relevant for the question. This could be
# the content of the most relevant chunk determined through some heuristic based on the encodings or simply
# using the text of a selected document chunk.
