from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from PyPDF2 import PdfReader
from langchain.schema import Document
import os
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

def load_or_create_faiss_index(pdf_paths, retriever_path, embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    """
    Load FAISS index if it exists; otherwise, create and save it.
    Args:
        pdf_paths (list of str): Paths to PDF files to process.
        retriever_path (str): Path to save/retrieve the FAISS index.
    Returns:
        FAISS: The loaded or newly created FAISS index.
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    if os.path.exists(retriever_path):
        retriever = FAISS.load_local(retriever_path, embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = preprocess_data(pdf_paths)
        retriever = create_faiss_index(chunks, embedding_model)
        retriever.save_local(retriever_path)
    return retriever

def extract_text_from_pdfs(pdf_paths):
    """
    Extract text from multiple PDF files.
    Args:
        pdf_paths (list of str): List of PDF file paths.
    Returns:
        list: List of raw text strings from all PDFs.
    """
    pdf_texts = []

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        texts = [p.extract_text().strip() for p in reader.pages]
        pdf_texts.extend(text for text in texts if text)  # Filter empty strings

    return pdf_texts

def preprocess_data(file_paths, chunk_size=500, chunk_overlap=50, tokens_per_chunk=256):
    """
    Processes a list of PDF files into chunks for retrieval.
    Args:
        pdf_texts (list of str): List of text strings.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    Returns:
        list: List of Document objects.
    """
    pdf_texts = extract_text_from_pdfs(file_paths)
    # Stage 1: Character-level splitting
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

    # Stage 2: Token-level splitting
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, tokens_per_chunk=tokens_per_chunk
    )
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    # Convert chunks to Document objects
    documents = [Document(page_content=text) for text in token_split_texts]
    return documents

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

# Either use the augmented query or the multiple queries

def generate_multiple_queries(query, num_queries=5, model=None):
    """
    Generates multiple related questions based on a Bible-related query.
    Args:
        query (str): The original question.
        num_queries (int): Number of related questions to generate (default: 5).
        model (str): The language model to use (default: "gpt-3.5-turbo").
        client: The OpenAI client instance for API calls.
    Returns:
        list: List of related questions generated from the query.
    """
    prompt = f"""
    You are a knowledgeable and compassionate biblical counselor. 
    For the given query "{query}", propose up to {num_queries} related questions to help them explore the topic further. 
    List each question on a separate line without numbering.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content
    content = [q.strip() for q in content.split("\n") if q.strip()]
    return content
    

def generate_final_answer( original_query, retriever, model=None):
    """
    Generates a final, consolidated answer based on the original query, augmented queries, and retrieved documents.
    Args:
        original_query (str): The main user question.
        retriever: FAISS retriever for document retrieval.
        client: OpenAI client for API interactions.
        model (str): The language model to use (default: "gpt-3.5-turbo").
        n_results (int): Number of results to retrieve for each query.
    Returns:
        str: The final consolidated answer.
    """
    if model is None:
        client = OpenAI()

    augmented_queries = generate_multiple_queries(original_query)
    joined_queries = original_query + " " + " ".join(augmented_queries)
    # get context from retriever
    context = retriever.similarity_search(joined_queries, k=5)
    # Prepare context for the model
    context_text = "\n\n".join([doc.page_content for doc in context])

    prompt = f"""
        Based on the query "{original_query}" and the following context:

        {context_text}

        Write a concise answer in 200 words or less as if you are providing compassionate advice over the phone.
        """

    # Generate answer using OpenAI's ChatCompletion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful and compassionate biblical counselor."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content