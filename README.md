
# Llama 2 Model Integration and Gradio Interface

This project demonstrates how to integrate a Llama 2 model for text generation and create a Gradio interface for easy interaction.

## Project Structure

- `model_handler.py`: Handles the loading of the Llama 2 model and text generation.
- `app_interface.py`: Defines the Gradio interface and integrates the model handling for user interaction.

## Pre-requisites

- Python 3.7 or newer.
- Gradio
- Transformers library by Hugging Face.

## Setup Instructions

1. Ensure Python 3.7 or newer is installed on your system.
2. Install required Python libraries by running:

```bash
pip install gradio transformers
```

3. Clone the repository or download the provided Python scripts (`model_handler.py` and `app_interface.py`) into your working directory.

## Downloading the Llama 2 Model

To download the Llama 2 model used in this project, follow this link: [Download Llama 2 Model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin)

Ensure to replace `"path/to/your/llama/model"` in the Python scripts with the actual path where your downloaded model is stored.

## Running the Project

1. Update `model_handler.py` and `app_interface.py` with the correct path to your Llama 2 model.
2. Ensure both Python scripts are in the same directory.
3. Run the Gradio interface script:

```bash
python app_interface.py
```

4. A Gradio interface will launch in your default web browser. Enter a prompt into the text box and press submit to generate text using the Llama 2 model.
