from RAG_Model.helper_utils import generate_final_answer, load_or_create_faiss_index
import os
from openai import OpenAI, AsyncOpenAI
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


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
    model=client,
)

print(f"Final Answer:\n{final_answer}")

begin_sentence = "Hey there, I'm your personal Shepherd, how can I help you?"
agent_prompt = "You are a biblical counselor. Based on the query and the context provided, write a concise answer in 200 words or less as if you are providing compassionate advice over the phone."

class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=openai_api_key,
        )
    
    def draft_begin_message(self):
        response = ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response


