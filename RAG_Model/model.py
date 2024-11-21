from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA


# Load and split the data
def preprocess_data(file_paths, chunk_size=500, chunk_overlap=50):
    """
    Processes multiple text files into chunks for retrieval.
    Args:
        file_paths (list of str): List of file paths to process.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        list: List of text chunks from all files.
    """
    all_chunks = []

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Loop through each file
    for file_path in file_paths:
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split the text into smaller chunks
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)  # Combine chunks from all files

    return all_chunks

# Example usage
file_path = "bible_and_commentaries.txt"
chunks = preprocess_data(file_path)
print(f"Generated {len(chunks)} chunks.")

# Initialize FAISS with embeddings
def create_faiss_index(chunks, embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    """
    Creates a FAISS retriever from text chunks.
    Args:
        chunks (list): Preprocessed text chunks.
        embedding_model (str): Model for embeddings.
    Returns:
        FAISS: FAISS retriever.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Example usage
retriever = create_faiss_index(chunks)
retriever.save_local("faiss_index")

llm = OpenAI(model="text-davinci-003", temperature=0.7, max_tokens=300)


# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_retriever(),
    return_source_documents=True
)

# Example query
query = "What does the Bible say about forgiveness?"
result = qa_chain.run(query)

# Print the response and sources
print("Answer:", result['result'])
print("Sources:", result['source_documents'])
