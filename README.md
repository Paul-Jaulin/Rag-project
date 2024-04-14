# Document RAG System with Dynamic Model Selection

This project showcases the integration of various small NLP models for generating responses from text extracted from PDF documents using a Gradio interface for easy interaction. Users can select different models and documents to generate context-specific responses.

## Project Structure

- `pdf_processor.py`: Handles the loading and text extraction of PDF files.
- `embedding_processor.py`: Manages the embedding of text chunks and retrieval of relevant contexts.
- `response_generator.py`: Facilitates the generation of responses based on the selected model and context.
- `app_interface.py`: Defines and manages the Gradio interface, integrating all components for user interaction.

## Pre-requisites

- Python 3.7 or newer.
- Gradio
- Transformers library by Hugging Face.
- Sentence Transformers for embeddings.
- PyPDF2 for PDF processing.

## Setup Instructions

1. Ensure Python 3.7 or newer is installed on your system.
2. Install required Python libraries by running:

```bash
pip install gradio transformers sentence_transformers PyPDF2
```

3. Clone the repository or download the provided Python scripts into your working directory.

## Model Selection

This project uses multiple small models from Hugging Face's Model Hub, suitable for real-time applications. Here are the models integrated into the project:

- `distilbert-base-uncased`
- `google/electra-small-discriminator`
- `sshleifer/tiny-gpt2`
- `prajjwal1/bert-small`
- `microsoft/MiniLM-L12-H384-uncased`

Ensure to replace the model selection in `app_interface.py` with any of the models listed as needed.

## Running the Project

1. Ensure all Python scripts (`pdf_processor.py`, `embedding_processor.py`, `response_generator.py`, `app_interface.py`) are in the same directory.
2. Run the Gradio interface script:

```bash
python app_interface.py
```

3. The Gradio interface will launch in your default web browser. You can select a model, choose a PDF document from the dataset, and enter a question. The system processes the document, retrieves the context, and generates a response based on the chosen model.

## Example Usage

- Choose a model from the dropdown menu.
- Select a PDF file from the dataset.
- Enter a question related to the PDF content.
- Press submit to see the generated response and the context used for generation.

For any questions regarding the setup or operation of this project, refer to the detailed comments in the code or contact the repository maintainer.