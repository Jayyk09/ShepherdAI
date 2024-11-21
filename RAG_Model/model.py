import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from helper_utils import preprocess_data, create_faiss_index, augment_query_generated, generate_multiple_queries, generate_final_answer

load_dotenv()


pdf_paths = ["data/Matthew_Henrys_Concise_Commentary_On_The_Bible.pdf", "data/The-Bible,-New-Revised-Standard-Version.pdf"]
chunks = preprocess_data(pdf_paths)

retriever = create_faiss_index(chunks)

# Example Usage
original_query = "What does the Bible say about forgiveness?"
augmented_queries = generate_multiple_queries(original_query)

# Generate final answer
final_answer = generate_final_answer(
    original_query=original_query,
    augmented_queries=augmented_queries,
    retriever=retriever,
    model="gpt-3.5-turbo",
    n_results=5,
)

# Output the final answer
print(f"Final Answer:\n{final_answer}")