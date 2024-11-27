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
    
    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                messages.append({"role": "user", "content": utterance.content})
        return messages
    
    def generate_agent_prompt(self, request: ResponseRequiredRequest):
        prompt = [
            {"role": "system",
                "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn\'t catch that", "some noise", "pardon", "you\'re coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n'
            },
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
        prompt = self.prepare_prompt(request)
        stream = self.swarm.run(prompt, stream=True)

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



