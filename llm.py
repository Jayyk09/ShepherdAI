from RAG_Model.helper_utils import generate_final_answer, load_or_create_faiss_index
import os
from openai import OpenAI, AsyncOpenAI
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List

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
    def __init__(self, retriever, embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        """
        Initialize the LLMClient with a retriever and embedding model.
        Args:
            retriever: Preloaded FAISS retriever instance.
            embedding_model: HuggingFace embedding model name for compatibility.
        """
        self.retriever = retriever
        self.embedding_model = embedding_model
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
    
    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                messages.append({"role": "user", "content": utterance.content})
        return messages
    
    def generate_agent_prompt(self, request: ResponseRequiredRequest):
        # TODO: Add context from RAG model      
        context = self.retriever.similarity_search(request.query, k=5)
        context_text = "\n\n".join([doc.page_content for doc in context])

        prompt = {
            "role": "system",
            "content": f"You are a compassionate biblical counselor. Use the following context to assist the user:\n\n{context_text}",
        }

        transcript_messages = self.convert_transcript_to_openai_messages(
            request.transcript
        )
        
        for message in transcript_messages:
            prompt.append(message)

        if request.interaction_type == "reminder_required":
            prompt.append(
                {
                    "role": "user",
                    "content": "(Now the user has not responded in a while, you would say:)",
                }
            )
        return prompt

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        stream = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Or use a 3.5 model for speed
            messages=prompt,
            stream=True,
        )
        for chunk in stream:
            if "content" in chunk and chunk['content']:
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk['content'],
                    content_complete=False,
                    end_call=False,
                )
                yield response

        # Send final response with "content_complete" set to True to signal completion
        response = ResponseResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )
        yield response