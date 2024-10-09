import os
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter

# Define a base TextSplitter class
class TextSplitter:
    """
    Base class for splitting text into chunks.
    Subclasses should implement the split_text method.
    """
    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError("Subclasses should implement this method.")

# Implement specific text splitters
class RecursiveCharacterTextSplitterAdapter(TextSplitter):
    """
    Adapter for splitting text using the RecursiveCharacterTextSplitter from Langchain.
    """
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        """
        Initializes the RecursiveCharacterTextSplitterAdapter with the specified chunk size and overlap.
        
        Args:
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Number of characters to overlap between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=True
        )

    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into chunks using the RecursiveCharacterTextSplitter.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of text chunks.
        """
        return self.splitter.split_text(text)

class NLTKTextSplitterAdapter(TextSplitter):
    """
    Adapter for splitting text using the NLTKTextSplitter from Langchain.
    """
    def __init__(self):
        """
        Initializes the NLTKTextSplitterAdapter.
        """
        self.splitter = NLTKTextSplitter()

    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into chunks using the NLTKTextSplitter.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of text chunks.
        """
        return self.splitter.split_text(text)

# Define a custom embedding function
class CustomSentenceTransformerEmbedding(embedding_functions.EmbeddingFunction):
    """
    Custom embedding function that generates embeddings using the SentenceTransformer model.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initializes the CustomSentenceTransformerEmbedding with the specified model.

        Args:
            model_name (str): Name of the SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[List[float]]: List of embeddings for each input text.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

# Define the ChromaInterface class
class ChromaInterface:
    """
    Interface for interacting with ChromaDB to store and query document embeddings.
    """
    def __init__(self, collection_name: str, persist_directory: str, text_splitter: TextSplitter):
        """
        Initializes the ChromaInterface with a persistent ChromaDB collection and a text splitter.

        Args:
            collection_name (str): Name of the collection in ChromaDB.
            persist_directory (str): Directory where the ChromaDB data will be stored.
            text_splitter (TextSplitter): Text splitter used to divide documents into chunks.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = CustomSentenceTransformerEmbedding()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        self.text_splitter = text_splitter

    def add_documents_from_files(self, file_paths: List[str], metadatas: Optional[List[Dict[str, str]]] = None):
        """
        Adds documents from the specified files into the ChromaDB collection after splitting them into chunks.
        
        Args:
            file_paths (List[str]): List of file paths containing the documents to add.
            metadatas (Optional[List[Dict[str, str]]]): List of metadata dictionaries corresponding to each file.
        """
        documents = []
        ids = []
        id_counter = 0  # Initialize a counter for unique IDs

        # Process each file
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Split the content into chunks
                split_texts = self.text_splitter.split_text(content)
                documents.extend(split_texts)
                
                # Generate unique IDs for each chunk
                for _ in split_texts:
                    ids.append(f"{os.path.basename(file_path)}_{id_counter}")
                    id_counter += 1  # Increment the counter for each chunk

        if metadatas is None:
            # If no metadata is provided, use the file path as the source metadata
            metadatas = [{"source": file_path} for file_path in file_paths]

        # Adjust metadatas to match the number of document chunks
        extended_metadatas = []
        for i, file_path in enumerate(file_paths):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {"source": file_path}
            # Duplicate the metadata for all chunks of the same document
            extended_metadatas.extend([metadata] * len(self.text_splitter.split_text(open(file_path, 'r', encoding='utf-8').read())))

        # Add documents and their metadata into the collection
        self.collection.add(
            documents=documents,
            metadatas=extended_metadatas,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 30) -> List[str]:
        """
        Queries the ChromaDB collection for the most relevant documents based on the query text.
        
        Args:
            query_text (str): The text query for searching relevant documents.
            n_results (int): The number of results to return (default is 30).

        Returns:
            List[str]: List of relevant document chunks.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )['documents']
        return results
