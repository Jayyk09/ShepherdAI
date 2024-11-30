from RAG_Model.helper_utils import generate_final_answer, load_or_create_faiss_index
import os
from openai import AsyncOpenAI
from custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)
from typing import List

openai_api_key = os.getenv("OPENAI_API_KEY")

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
        query = request.transcript[-1].content if request.transcript else ""
        context = self.retriever.similarity_search(query, k=5)
        context_text = "\n\n".join([doc.page_content for doc in context])

        prompt = [
            {
                "role": "system",
                "content": f"You are a compassionate biblical counselor. Use the following context to assist the user:\n\n{context_text}",
            }
        ]

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
        prompt = self.generate_agent_prompt(request)
        stream = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Or use a 3.5 model for speed
            messages=prompt,
            stream=True,
        )
        print(f"Prompt being sent to OpenAI for request ID {request.response_id}: {prompt}")

        accumulated_content = ""

        async for chunk in stream:
        # Ensure "choices" and "delta" keys are present in the chunk
            if chunk.choices and chunk.choices[0].delta.content:
                new_content = chunk.choices[0].delta.content
                accumulated_content += new_content
                # Build and yield the response
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=new_content,
                    content_complete=False,
                    end_call=False,
                )
                print(f"Response chunk sent: {response}")
                yield response


        # Send final response with "content_complete" set to True to signal completion
        response = ResponseResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )
        yield response