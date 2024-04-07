# Rag-project

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

## Running the Project

1. Update `model_handler.py` and `app_interface.py` with the correct path to your Llama 2 model. Replace `"path/to/your/llama/model"` with the actual path where your model is stored.

2. Ensure both Python scripts are in the same directory.