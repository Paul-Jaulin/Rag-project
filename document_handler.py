import os

# Function to chunk documents
def chunk_documents(dataset_path, chunk_size, overlapping):
    chunks = []
    for file_name in os.listdir(dataset_path):
        with open(os.path.join(dataset_path, file_name), "r", encoding="utf-8") as file:
            content = file.read()
            for i in range(0, len(content), chunk_size - overlapping):
                chunk = content[i:i + chunk_size]
                chunks.append(chunk)
    return chunks
