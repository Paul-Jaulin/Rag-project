# Document RAG System with Dynamic Model Selection

This project showcases the integration of various small NLP models from Hugging Face for generating responses from text extracted from PDF documents using a Gradio interface for easy interaction. Users can select different models and documents to generate context-specific responses.

## Project Structure

This project now uses a single Python script `RAG_project.py` that integrates all functionalities including PDF processing, embedding, response generation, and the Gradio interface.

## Pre-requisites

- Python 3.7 or newer.
- Gradio
- Transformers library by Hugging Face.
- Sentence Transformers for embeddings.
- langchain_community
- PyPDF2 for PDF processing.

## Setup Instructions

1. Ensure Python 3.7 or newer is installed on your system.
2. Create a virtual environment and install required Python libraries by running:

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

3. Download `Huggingface_llms.py` into your working directory.

## Model Selection

This project uses multiple small models from Hugging Face's Model Hub, suitable for real-time applications. Here are the models already integrated into the project:

- `google/flan-t5-small`
- `google/flan-t5-base`
- `google/flan-t5-large`
- `google/flan-t5-xl`

You can easily add more models by expanding the `model_names` dictionary in the script with appropriate model identifiers.

## Running the Project

1. Ensure `RAG_project.py` is in your working directory and the virtual environment is activated.
2. Run the script:

```bash
python RAG_project.py
```

3. The Gradio interface will launch in your default web browser. You can select a model, choose a PDF document from the dataset, and enter a question. The system processes the document, retrieves the context, and generates a response based on the chosen model.

## Example Usage

- Choose a model size (small or medium) from the dropdown menu.
- Select a PDF file from the dataset or choose "All Documents" to process all files.
- Enter a question related to the PDF content.
- Adjust the Chunk Size and Overlap Size sliders as needed (these control how the document is split for processing).
- Press submit to see the generated response and the context used for generation.

For any questions regarding the setup or operation of this project, refer to the detailed comments in the code.