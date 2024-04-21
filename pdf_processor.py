from PyPDF2 import PdfReader

def load_pdf(file_path):
    """
    Loads a PDF and extracts text from each page.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"  # Extract text from each page
    return text

def chunk_text(text, chunk_size, overlap):
    """
    Chunks the text into segments.
    """
    text_length = len(text)
    chunks = []
    for i in range(0, text_length, chunk_size - overlap):
        end = min(i + chunk_size, text_length)
        chunks.append(text[i:end])
    return chunks
