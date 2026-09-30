import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm.llm_client import LLM_MODEL, OLLAMA_BASE_URL
from routes.chat import router as chat_router
from routes.cocktails import router as cocktails_router


load_dotenv()
USE_LLM_ANSWER = os.getenv("USE_LLM_ANSWER", "true").lower() == "true"

app = FastAPI(
    title="CocktailGPT API",
    description="Backend für CocktailGPT mit KI-gestützter Intent-Erkennung und Cocktail-Suche",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cocktails_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    ollama_connected = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/v1/models")
            ollama_connected = response.is_success
    except httpx.HTTPError:
        pass

    return {
        "message": "CocktailGPT API läuft",
        "ollama_url": OLLAMA_BASE_URL,
        "model": LLM_MODEL,
        "use_llm_answer": USE_LLM_ANSWER,
        "intent_routing": "llm",
        "ollama_connected": ollama_connected,
    }
