import os
import gradio as gr

from pathlib import Path
from langchain_community.embeddings import VertexAIEmbeddings
from langchain_community.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import  Chroma
import google.auth
from google.cloud import aiplatform


PROJECT_ID = "XXXX"
print(PROJECT_ID)


creds, _ = google.auth.default(quota_project_id=PROJECT_ID)
aiplatform.init(
        project=PROJECT_ID,
        location="europe-west1",
        credentials=creds,
    )
# textembedding-gecko-multilingual
embeddings = VertexAIEmbeddings(model_name='textembedding-gecko@001')
parameters = {
    "temperature": 0.0,
    "max_output_tokens": 2048,
    "top_p": 0.1,
    "top_k": 3,
}
llm = VertexAI(model_name="gemini-1.5-pro-preview-0409", **parameters)

dataset_path = "dataset/"
model_options = ["gpt2", "distilgpt2", "sshleifer/tiny-gpt2", "openai-gpt", "microsoft/DialoGPT-small"]


def process_document(model_name, filename, question, chunk_size, overlap_size):
    rag_template = """Use the context below ("context") to answer the question ("question")
    context: {context} 
    question: {question}
    Respond just with  the answer to the question, please be consise.[/INST]
    """
    RAG_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=rag_template,)
    file_path = os.path.join(dataset_path, filename)
    loader = PyPDFLoader(file_path)
    docs= loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    relevant_chunks = vectorstore.max_marginal_relevance_search(question, k=5)
    context = "\n\n".join([d.page_content for d in relevant_chunks])
    final_prompt = RAG_CHAIN_PROMPT.format(question=question, context=context)
    response = llm.predict(final_prompt)
    return context, response


demo = gr.Interface(
    fn=process_document,
    inputs=[
        gr.Dropdown(label="Choose Model", choices=model_options),
        gr.Dropdown(label="Select PDF Document", choices=os.listdir(dataset_path)),
        gr.Textbox(label="Enter your question"),
        gr.Slider(label="Chunk Size", minimum=128, maximum=1024, step=128, value=512),
        gr.Slider(label="Overlap Size", minimum=0, maximum=500, step=10, value=100)
    ],
    outputs=[
        gr.Textbox(label="Context"),
        gr.Textbox(label="Response")
    ]
)

if __name__ == "__main__":
    demo.launch()