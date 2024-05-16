import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import gradio as gr
import torch.nn.functional as F

# Mapping of model sizes and names to their corresponding Hugging Face model identifiers
model_names = {
    "Small - Google Flan-T5 (80M parameters)": "google/flan-t5-small",  # Flan-T5 Small with 80 million parameters
    "Medium - Google Flan-T5 (250M parameters)": "google/flan-t5-base",  # Flan-T5 Base with 250 million parameters
    "Large - Google Flan-T5 (783M parameters)": "google/flan-t5-large",  # Flan-T5 Large with 783 million parameters
    "XLarge - Google Flan-T5 (2.85B parameters)": "google/flan-t5-xl",  # Flan-T5 XLarge with 2.85B parameters
}

# Template for generating responses based on a given context and question
rag_template = """Use the context below ("context") to answer the question ("question"):
context: {context} 
question: {question}
Please respond concisely with just the answer to the question.[/INST]
"""
RAG_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=rag_template)

# Directory containing dataset files
dataset_path = "dataset/"

# Replace with the specific revision to ensure reproducibility if the model is updated.
revision = None

# Set generation parameters
generate_kwargs = {
    "max_new_tokens": 300,  # Specify the maximum number of tokens to generate
    "num_return_sequences": 1
}

def process_document(model_size, filename, question, chunk_size, overlap_size):
    # Initialize the model and tokenizer based on the selected model size
    model_name = model_names[model_size]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    # Load the sentence embedding model
    embeddings = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-small-Embedding-v0")

    # Process each file or all files if 'All Documents' is selected
    if filename == "All Documents":
        files = os.listdir(dataset_path)
    else:
        files = [filename]

    contexts = []
    for file in files:
        file_path = os.path.join(dataset_path, file)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # Split document text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
        splits = text_splitter.split_documents(docs)
        # Create vector store from document chunks
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        # Retrieve chunks most relevant to the question
        relevant_chunks = vectorstore.max_marginal_relevance_search(question, k=3)
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        contexts.append(context)

    # Combine contexts from all processed documents
    combined_context = "\n\n".join(contexts)

    # Ensure the input sequence length does not exceed the model's maximum length
    input_tokens = tokenizer(combined_context + question, truncation=True, max_length=512, return_tensors="pt")

    final_prompt = RAG_CHAIN_PROMPT.format(question=question, context=combined_context)
    response = llm(final_prompt, **generate_kwargs)[0]["generated_text"]
    return combined_context, response

# List files in the dataset directory and add an option to process all documents
file_options = os.listdir(dataset_path)
file_options.append("All Documents")

# Set up the Gradio interface
demo = gr.Interface(
    fn=process_document,
    inputs=[
        gr.Dropdown(label="Choose Model Size", choices=list(model_names.keys())),
        gr.Dropdown(label="Select PDF Document", choices=file_options),
        gr.Textbox(label="Enter your question", value="What is GenAI?"),
        gr.Slider(label="Chunk Size", minimum=128, maximum=1024, step=128, value=512),
        gr.Slider(label="Overlap Size", minimum=0, maximum=500, step=10, value=100)
    ],
    outputs=[
        gr.Textbox(label="Context"),
        gr.Textbox(label="Response")
    ],
    title="Document-Based Question Answering System",
    description="Select a document or all documents from the list and pose a question. The system will analyze the content and generate an answer."
)

if __name__ == "__main__":
    demo.launch()
