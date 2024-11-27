import json
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
from retell import Retell
from .custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
from .llm import LLMClient

load_dotenv(override=True)
app = FastAPI()
retell = Retell(api_key=os.environ["RETELL_API_KEY"])

