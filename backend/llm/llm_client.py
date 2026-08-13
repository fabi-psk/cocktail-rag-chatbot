import json
import os
from typing import Any

import httpx


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b")


class LLMError(RuntimeError):
    pass


class InvalidLLMOutputError(LLMError):
    pass


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            parsed, end = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
            index = start + end
        except json.JSONDecodeError:
            index = start + 1

    raise InvalidLLMOutputError("LLM response did not contain a JSON object.")


async def query_ollama(messages: list[dict[str, str]], response_format: str = "json_object") -> str:
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.15,
        "top_p": 0.7,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = {"type": response_format}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
        raise LLMError("LLM ist nicht erreichbar oder lieferte keine gueltige Antwort.") from exc
