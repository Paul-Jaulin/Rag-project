import fitz  # PyMuPDF

def read_pdf(pdf_path):
    """
    Reads a PDF and returns its text content.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    print(text)
    return text
    
