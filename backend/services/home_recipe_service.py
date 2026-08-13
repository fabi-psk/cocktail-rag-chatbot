import re
import unicodedata
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

import httpx

from llm.llm_client import InvalidLLMOutputError, LLMError, extract_json_object, query_ollama
from models.cocktail import ChatMessage


IBA_BASE_URL = "https://iba-world.com/iba-cocktail"


def recipe_slug(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name.strip().lower())
    ascii_name = "".join(char for char in normalized if not unicodedata.combining(char))
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    return quote(slug, safe="-")


class IBARecipeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.ingredients: list[str] = []
        self.method: list[str] = []
        self.garnish: list[str] = []
        self._capture_title = False
        self._capture_heading = False
        self._capture_item = False
        self._capture_paragraph = False
        self._heading = ""
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._capture_title = True
            self._buffer = []
        elif tag in {"h3", "h4"}:
            self._capture_heading = True
            self._buffer = []
        elif tag == "li" and self._heading == "ingredients":
            self._capture_item = True
            self._buffer = []
        elif tag == "p" and self._heading in {"method", "garnish"}:
            self._capture_paragraph = True
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        text = " ".join("".join(self._buffer).split())
        if tag == "title" and self._capture_title:
            self.title = re.split(r"\s+[–-]\s+IBA\s*$", text)[0].strip()
            self._capture_title = False
        elif tag in {"h3", "h4"} and self._capture_heading:
            self._heading = text.lower()
            self._capture_heading = False
        elif tag == "li" and self._capture_item:
            if text:
                self.ingredients.append(text)
            self._capture_item = False
        elif tag == "p" and self._capture_paragraph:
            if text:
                target = self.method if self._heading == "method" else self.garnish
                target.append(text)
                if self._heading == "garnish":
                    self._heading = ""
            self._capture_paragraph = False
        self._buffer = []

    def handle_data(self, data: str) -> None:
        if any([
            self._capture_title,
            self._capture_heading,
            self._capture_item,
            self._capture_paragraph,
        ]):
            self._buffer.append(data)


def parse_iba_recipe(html: str, source_url: str) -> dict[str, Any]:
    parser = IBARecipeParser()
    parser.feed(html)
    if not parser.title or not parser.ingredients or not parser.method:
        raise ValueError("Die IBA-Seite enthielt kein vollstaendiges Rezept.")
    return {
        "name": parser.title,
        "ingredients": parser.ingredients,
        "instructions": parser.method,
        "garnish": parser.garnish,
        "source_name": "International Bartenders Association",
        "source_url": source_url,
    }


async def extract_requested_cocktail(
    user_message: str,
    history: list[ChatMessage] | None = None,
) -> str:
    messages: list[dict[str, str]] = [{
        "role": "system",
        "content": (
            "Extrahiere den Namen des Cocktails, dessen Rezept der Nutzer fuer zuhause sucht. "
            "Nutze bei Pronomen den kurzen Chatverlauf. Antworte ausschliesslich als JSON: "
            '{"cocktail_name":"Mojito","confidence":0.98}. '
            "Erfinde keinen Namen. Wenn kein konkreter Cocktail erkennbar ist, setze cocktail_name auf null."
        ),
    }]
    if history:
        for message in history[-4:]:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": user_message})

    parsed = extract_json_object(await query_ollama(messages))
    name = parsed.get("cocktail_name")
    confidence = parsed.get("confidence", 0)
    if not isinstance(name, str) or not name.strip() or not isinstance(confidence, (int, float)):
        raise InvalidLLMOutputError("Kein konkreter Cocktailname erkannt.")
    if confidence < 0.6:
        raise InvalidLLMOutputError("Cocktailname wurde nicht sicher erkannt.")
    return name.strip()


async def fetch_iba_recipe(cocktail_name: str) -> dict[str, Any] | None:
    slug = recipe_slug(cocktail_name)
    if not slug:
        return None
    source_url = f"{IBA_BASE_URL}/{slug}/"
    headers = {"User-Agent": "CocktailGPT educational recipe prototype/1.0"}
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(source_url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMError("Die Rezeptquelle ist gerade nicht erreichbar.") from exc

    try:
        return parse_iba_recipe(response.text, source_url)
    except ValueError:
        return None


def format_home_recipe_answer(recipe: dict[str, Any]) -> str:
    ingredients = "\n".join(f"- {item}" for item in recipe["ingredients"])
    instructions = "\n".join(
        f"{index}. {step}" for index, step in enumerate(recipe["instructions"], start=1)
    )
    garnish = ""
    if recipe.get("garnish"):
        garnish = "\n\n**Garnitur**\n" + " ".join(recipe["garnish"])
    return (
        f"**{recipe['name']} für zuhause**\n\n"
        f"**Zutaten**\n{ingredients}\n\n"
        f"**Zubereitung**\n{instructions}{garnish}\n\n"
        f"Quelle: {recipe['source_name']} – {recipe['source_url']}"
    )


async def build_home_recipe_response(
    user_message: str,
    history: list[ChatMessage] | None = None,
) -> dict[str, Any]:
    try:
        cocktail_name = await extract_requested_cocktail(user_message, history)
    except (LLMError, InvalidLLMOutputError):
        answer = "Welchen konkreten Cocktail möchtest du zuhause zubereiten?"
        return {
            "type": "follow_up",
            "intent": "unknown",
            "message": answer,
            "answer": answer,
            "cocktails": [],
            "web_recipes": [],
            "criteria": None,
        }

    try:
        recipe = await fetch_iba_recipe(cocktail_name)
    except LLMError as exc:
        answer = str(exc)
        return {
            "type": "follow_up",
            "intent": "catalog_query",
            "message": answer,
            "answer": answer,
            "cocktails": [],
            "web_recipes": [],
            "criteria": None,
        }

    if not recipe:
        answer = (
            f"Für {cocktail_name} habe ich in der getesteten IBA-Quelle kein Rezept gefunden. "
            "Ich erfinde deshalb keine Zutaten."
        )
        return {
            "type": "message",
            "intent": "catalog_query",
            "message": answer,
            "answer": answer,
            "cocktails": [],
            "web_recipes": [],
            "criteria": None,
        }

    answer = format_home_recipe_answer(recipe)
    return {
        "type": "message",
        "intent": "catalog_query",
        "message": answer,
        "answer": answer,
        "cocktails": [],
        "web_recipes": [recipe],
        "criteria": None,
    }
