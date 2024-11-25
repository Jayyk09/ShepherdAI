from helper_utils import generate_final_answer, load_or_create_faiss_index
import google.generativeai as genai
import os

google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

pdf_paths = ["data/Matthew_Henrys_Concise_Commentary_On_The_Bible.pdf", "data/The-Bible,-New-Revised-Standard-Version.pdf"]

retriever_path = "retriever.json"

retriever = load_or_create_faiss_index(pdf_paths, retriever_path)

# save to local
retriever.save_local("retriever.json")
# Example Usage
original_query = "I feel like I am failing in life and not doing well in school. What should I do?"
# Generate final answer
final_answer = generate_final_answer(
    original_query=original_query,
    retriever=retriever,
    model=model,
)

print(f"Final Answer:\n{final_answer}")