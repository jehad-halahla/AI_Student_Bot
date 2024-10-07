
# Arabic Text Correction Pipeline

This pipeline describes the process of extracting Arabic text from a PDF using Tesseract OCR, correcting it with a Language Model (LLM), and generating embeddings for the corrected text using ChromaDB. The pipeline involves several steps, each described in detail below.

## Pipeline Overview

1. **Step 1**: Extract Arabic text from a PDF using the Tesseract OCR library.
2. **Step 2**: Chunk the extracted text and send it for correction using the Gemini LLM API, facilitated by the `ArabicTextCorrector` class.
3. **Step 3**: Manually review and correct the chunked data.
4. **Step 4**: Use ChromaDB to generate and store embeddings for the corrected text.

---

## Step 1: Extracting Arabic Text from PDF Using Tesseract OCR

To extract Arabic text from a PDF, we use Tesseract OCR. Tesseract is an optical character recognition tool that can recognize text from scanned images and PDFs.

### 1.1 Install Tesseract

#### On Ubuntu

```bash
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev


#### On macOS (using Homebrew)

```bash
brew install tesseract
```

#### On Windows

1. Download the Windows installer from the [Tesseract GitHub page](https://github.com/tesseract-ocr/tesseract/wiki).
2. Run the installer and follow the instructions.

#### Install Python Wrapper for Tesseract

```bash
pip install pytesseract
```

### 1.2 Set up Tesseract for Arabic Text

Ensure that Tesseract supports Arabic. You can download the Arabic language pack if it's not already installed. 

To install the Arabic language pack:

```bash
sudo apt-get install tesseract-ocr-ara
```

### 1.3 Extract Text from PDF

Here's how to use `pytesseract` to extract text from a PDF:

```python
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# Convert PDF to images
pages = convert_from_path('input_file.pdf')

# Extract text from each page
for page_number, page_data in enumerate(pages):
    page_data.save(f'page_{page_number}.jpg', 'JPEG')
    text = pytesseract.image_to_string(page_data, lang='ara')  # Use Arabic language pack
    with open(f'output_text_{page_number}.txt', 'w', encoding='utf-8') as file:
        file.write(text)
```

This code converts a PDF to images and extracts the text for each page, saving it as a text file.

---

## Step 2: Correcting Arabic Text Using the Gemini LLM API

After extracting the text, the next step is to process it using the `ArabicTextCorrector` class, which sends the text chunks to the Gemini LLM for correction.

### 2.1 Chunking the Text

The `ArabicTextCorrector` class can split large text into smaller, manageable chunks before sending them to the Gemini LLM. Here's an overview of the text correction process:

1. **Chunking**: The text is split into chunks of a specified size (default is 1000 characters).
2. **LLM Correction**: Each chunk is sent to the Gemini LLM API for correction.
3. **Retry Logic**: If an error occurs during the API call, it retries up to a defined limit.

Here’s an example of how to use the `ArabicTextCorrector` class for this process:

```python
from ArabicTextCorrector import ArabicTextCorrector
from llm import GeminiLLM

# Initialize the Gemini LLM model
model = GeminiLLM()

# Initialize the ArabicTextCorrector with the model
corrector = ArabicTextCorrector(model)

# Process the extracted text file, chunk it, and correct it using the LLM
corrector.process_file(
    input_file="extracted_text.txt",
    output_file="corrected_text.txt",
    output_folder="corrected_chunks",
    chunk_size=1000  # Split text into 1000 character chunks
)
```

---

## Step 3: Manual Review of the Corrected Chunks

After the chunks have been processed through the Gemini LLM, it's recommended to manually review the corrected text to ensure the quality of the corrections. The chunks are saved in a specified output folder for easy access and revision.

You can review each chunk in the `corrected_chunks` folder and make any necessary adjustments to the text.

---

## Step 4: Embedding and Storing Text with ChromaDB

Once the corrected text has been finalized, you can store embeddings for the text using **ChromaDB**. ChromaDB is a vector database that stores embeddings, allowing you to search, index, and retrieve text efficiently.

### 4.1 Install ChromaDB

```bash
pip install chromadb
```

### 4.2 Generate and Store Embeddings

You can generate embeddings for the text and store them in ChromaDB. Here's an example of how to generate embeddings for each chunk and store them:

```python
import chromadb

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection to store the embeddings
collection = client.create_collection("arabic_text_embeddings")

# Loop through the corrected chunks and generate embeddings (assuming you have an embedding model)
for i, chunk in enumerate(corrected_chunks):
    embedding = generate_embedding(chunk)  # This would be your embedding function or model
    collection.add(
        embeddings=[embedding],  # The embedding vector for the chunk
        metadatas=[{"chunk_number": i}],  # Metadata, in this case, the chunk number
        documents=[chunk]  # The actual text of the chunk
    )
```

This process adds the embeddings and the corresponding text into a ChromaDB collection, allowing you to query and retrieve the text later using its embedding.

---

## Requirements

To run this pipeline, ensure you have the following dependencies installed:

- **Tesseract** (for OCR)
- **pytesseract** (Python wrapper for Tesseract)
- **pdf2image** (to convert PDFs into images for Tesseract)
- **Gemini LLM** (for text correction)
- **python-dotenv** (for managing environment variables)
- **ChromaDB** (for storing embeddings)

### Installing All Requirements

You can install all the dependencies at once by running:

```bash
pip install -r requirements.txt
```

### Install Individual Dependencies

To install dependencies individually:

1. **Tesseract**: [Installation instructions](https://github.com/tesseract-ocr/tesseract)
2. **pytesseract**:

```bash
pip install pytesseract
```

3. **pdf2image**:

```bash
pip install pdf2image
```

4. **llm**:

```bash
pip install llm
```

5. **python-dotenv**:

```bash
pip install python-dotenv
```

6. **ChromaDB**:

```bash
pip install chromadb
```

---

## Conclusion

This pipeline guides you through extracting Arabic text from a PDF, correcting it with an LLM, manually reviewing the corrections, and storing embeddings for the text using ChromaDB. By following these steps, you can efficiently process and work with large amounts of Arabic text in a structured way.
