import json
import os
import asyncio
from dotenv import load_dotenv
from RAG_Model.helper_utils import load_or_create_faiss_index
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
from retell import Retell
from custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
from llm import LLMClient

load_dotenv(override=True)
retell = Retell(api_key=os.environ["RETELL_API_KEY"])

retriever = None  # Global retriever instance

# use lifespan to initialize retriever
async def app_lifespan(app):
    global retriever
    pdf_paths = ["RAG_Model/data/Matthew_Henrys_Concise_Commentary_On_The_Bible.pdf", "RAG_Model/data/The-Bible,-New-Revised-Standard-Version.pdf"]
    retriever_path = "RAG_Model/retriever.json"

    # Startup logic
    print("Initializing retriever during lifespan startup...")
    if not os.path.exists(retriever_path):
        print("Retriever file not found. Initializing a new FAISS index...")
        retriever = load_or_create_faiss_index(pdf_paths, retriever_path)
        retriever.save_local(retriever_path)
        print("Retriever initialized and saved locally.")
    else:
        print("Retriever file found. Loading existing FAISS index...")
        retriever = load_or_create_faiss_index([], retriever_path)
        print("Retriever loaded successfully.")

    yield  # Lifespan enters the main application runtime here

app = FastAPI(lifespan=app_lifespan)

# Handle webhook from Retell server. This is used to receive events from Retell server.
# Including call_started, call_ended, call_analyzed
# Handle webhook from Retell server. This is used to receive events from Retell server.
# Including call_started, call_ended, call_analyzed
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        post_data = await request.json()
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":"), ensure_ascii=False),
            api_key=str(os.environ["RETELL_API_KEY"]),
            signature=str(request.headers.get("X-Retell-Signature")),
        )
        if not valid_signature:
            print(
                "Received Unauthorized",
                post_data["event"],
                post_data["data"]["call_id"],
            )
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})
        if post_data["event"] == "call_started":
            print("Call started event", post_data["data"]["call_id"])
        elif post_data["event"] == "call_ended":
            print("Call ended event", post_data["data"]["call_id"])
        elif post_data["event"] == "call_analyzed":
            print("Call analyzed event", post_data["data"]["call_id"])
        else:
            print("Unknown event", post_data["event"])
        return JSONResponse(status_code=200, content={"received": True})
    except Exception as err:
        print(f"Error in webhook: {err}")
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )
    
# Start a websocket server to exchange text input and output with Retell server. Retell server
# will send over transcriptions and other information. This server here will be responsible for
# generating responses with LLM and send back to Retell server.
@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    try:
        await websocket.accept()
        llm_client = None

        if retriever is None:
            raise Exception("Retriever not initialized")

        llm_client = LLMClient(retriever)

        # Send optional config to Retell server
        config = ConfigResponse(
            response_type="config",
            config={
                "auto_reconnect": True,
                "call_details": True,
            },
            response_id=1,
        )
        await websocket.send_json(config.__dict__)
        response_id = 0


        async def handle_message(request_json):
            nonlocal response_id
            nonlocal llm_client
            # There are 5 types of interaction_type: call_details, pingpong, update_only, response_required, and reminder_required.
            # Not all of them need to be handled, only response_required and reminder_required.
            if request_json["interaction_type"] == "call_details":
                print(json.dumps(request_json, indent=2))
                # Send first message to signal ready of server
                first_event = llm_client.draft_begin_message()
                await websocket.send_json(first_event.__dict__)

                return
            if request_json["interaction_type"] == "ping_pong":
                await websocket.send_json(
                    {
                        "response_type": "ping_pong",
                        "timestamp": request_json["timestamp"],
                    }
                )
                return
            if request_json["interaction_type"] == "update_only":
                return
            if (
                request_json["interaction_type"] == "response_required"
                or request_json["interaction_type"] == "reminder_required"
            ):
                response_id = request_json["response_id"]
                request = ResponseRequiredRequest(
                    interaction_type=request_json["interaction_type"],
                    response_id=response_id,
                    transcript=request_json["transcript"],
                )
                print(
                    f"""Received interaction_type={request_json['interaction_type']}, response_id={response_id}, last_transcript={request_json['transcript'][-1]['content']}"""
                )

                async for event in llm_client.draft_response(request):
                    await websocket.send_json(event.__dict__)
                    if request.response_id < response_id:
                        break  # new response needed, abandon this one

        async for data in websocket.iter_json():
            asyncio.create_task(handle_message(data))

    except WebSocketDisconnect:
        print(f"LLM WebSocket disconnected for {call_id}")
    except ConnectionTimeoutError as e:
        print("Connection timeout error for {call_id}")
    except Exception as e:
        print(f"Error in LLM WebSocket: {e} for {call_id}")
        await websocket.close(1011, "Server error")
    finally:
        print(f"LLM WebSocket connection closed for {call_id}")