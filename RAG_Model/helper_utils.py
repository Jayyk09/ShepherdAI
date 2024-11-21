from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        list: List of text chunks.
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

    return token_split_texts


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

def augment_query_generated(query, model="gpt-3.5-turbo", client=client):
    """
    Generates an example answer to the given query in the context of the Bible.
    Args:
        query (str): The user's question or query.
        model (str): The language model to use (default: "gpt-3.5-turbo").
    Returns:
        str: A generated example answer to the query.
    """
    prompt = """
    You are a knowledgeable and compassionate biblical counselor. 
    Your role is to provide example answers to questions based on the Bible, incorporating scripture references and commentary insights. 
    Use language that is clear, respectful, and faithful to biblical teachings.
    Provide an example response to the following query as if it might be found in a discussion of scripture or a theological commentary.
    """
    
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

def generate_multiple_queries(query, num_queries=5, model="gpt-3.5-turbo", client=client):
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
    Your users are asking questions about the Bible and Christian living. 
    For the given query, propose up to {num_queries} related questions to help them explore the topic further. 
    Use clear, concise language and ensure each question focuses on a single aspect of the topic. 
    Questions should be directly related to scripture, theology, or practical Christian living. 
    List each question on a separate line without numbering.
    """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = [q.strip() for q in content.split("\n") if q.strip()]
    return content