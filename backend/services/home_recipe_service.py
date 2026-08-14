import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

import httpx

from llm.llm_client import InvalidLLMOutputError, LLMError, extract_json_object, query_ollama
from models.cocktail import ChatMessage


COCKTAIL_DB_BASE_URL = "https://www.cocktaildatenbank.de"
COCKTAIL_DB_RECIPE_PATH = "/cocktail-rezepte/"
logger = logging.getLogger("cocktail-rag")


def normalize_recipe_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name.strip().lower())
    ascii_name = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_name).strip()


def recipe_index_key(name: str) -> str | None:
    normalized = normalize_recipe_name(name)
    if not normalized:
        return None
    return "0-9" if normalized[0].isdigit() else normalized[0]


class CocktailLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        href = attributes.get("href")
        if (
            tag == "a"
            and isinstance(href, str)
            and re.fullmatch(r"/cocktail-rezepte/\d+-[^#?]+(?:#autoplay)?", href)
        ):
            self._href = href.split("#", 1)[0]
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            name = " ".join("".join(self._buffer).split())
            if name:
                self.links.append((name, self._href))
            self._href = None
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._buffer.append(data)


def find_recipe_path(index_html: str, cocktail_name: str) -> str | None:
    parser = CocktailLinkParser()
    parser.feed(index_html)
    requested = normalize_recipe_name(cocktail_name)
    unique_links = list(dict.fromkeys(parser.links))
    exact = [path for name, path in unique_links if normalize_recipe_name(name) == requested]
    if exact:
        return exact[0]

    ranked = sorted(
        (
            SequenceMatcher(None, requested, normalize_recipe_name(name)).ratio(),
            path,
        )
        for name, path in unique_links
    )
    if ranked and ranked[-1][0] >= 0.88:
        return ranked[-1][1]
    return None


class CocktailDatabaseRecipeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.ingredients: list[str] = []
        self.method: list[str] = []
        self._capture_title = False
        self._capture_heading = False
        self._capture_item = False
        self._capture_paragraph = False
        self._heading = ""
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "span" and attributes.get("itemprop") == "name":
            self._capture_title = True
            self._buffer = []
        elif tag == "h2":
            self._capture_heading = True
            self._buffer = []
        elif tag == "li" and attributes.get("itemprop") == "ingredients":
            self._capture_item = True
            self._buffer = []
        elif tag == "p" and self._heading == "zubereitung":
            self._capture_paragraph = True
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        text = " ".join("".join(self._buffer).split())
        if tag == "span" and self._capture_title:
            self.title = text
            self._capture_title = False
            self._buffer = []
        elif tag == "h2" and self._capture_heading:
            self._heading = text.lower()
            self._capture_heading = False
            self._buffer = []
        elif tag == "li" and self._capture_item:
            if text:
                self.ingredients.append(text)
            self._capture_item = False
            self._buffer = []
        elif tag == "p" and self._capture_paragraph:
            if text:
                self.method.extend(split_recipe_steps(text))
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


def split_recipe_steps(text: str) -> list[str]:
    steps = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", text.strip())
    return [step.strip() for step in steps if step.strip()]


def parse_cocktail_database_recipe(html: str, source_url: str) -> dict[str, Any]:
    parser = CocktailDatabaseRecipeParser()
    parser.feed(html)
    if not parser.title or not parser.ingredients or not parser.method:
        raise ValueError("Cocktaildatenbank enthielt kein vollstaendiges Rezept.")
    return {
        "name": parser.title,
        "ingredients": parser.ingredients,
        "instructions": parser.method,
        "garnish": [],
        "source_name": "Cocktaildatenbank.de",
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


async def fetch_cocktail_database_recipe(cocktail_name: str) -> dict[str, Any] | None:
    index_key = recipe_index_key(cocktail_name)
    if not index_key:
        return None
    index_url = urljoin(
        COCKTAIL_DB_BASE_URL,
        f"{COCKTAIL_DB_RECIPE_PATH}{index_key}",
    )
    headers = {
        "User-Agent": "CocktailGPT educational recipe prototype/1.0",
        "Accept-Language": "de-DE,de;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
            index_response = await client.get(index_url)
            index_response.raise_for_status()
            recipe_path = find_recipe_path(index_response.text, cocktail_name)
            if not recipe_path:
                return None
            source_url = urljoin(COCKTAIL_DB_BASE_URL, recipe_path)
            recipe_response = await client.get(source_url)
            if recipe_response.status_code == 404:
                return None
            recipe_response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LLMError("Die Rezeptquelle ist gerade nicht erreichbar.") from exc

    try:
        return parse_cocktail_database_recipe(recipe_response.text, source_url)
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


def validate_rewritten_recipe(
    payload: dict[str, Any],
    source_recipe: dict[str, Any],
) -> None:
    for field in ("ingredients", "instructions", "garnish"):
        values = payload.get(field)
        source_values = source_recipe.get(field, [])
        if (
            not isinstance(values, list)
            or len(values) != len(source_values)
            or not all(isinstance(item, str) and item.strip() for item in values)
        ):
            raise InvalidLLMOutputError(
                f"Die deutsche Rezeptfassung hat ein ungueltiges Feld: {field}."
            )


RECIPE_STOP_WORDS = {
    "als", "auf", "aus", "das", "dem", "den", "der", "die", "ein", "eine",
    "einem", "einen", "einer", "geben", "hinzufugen", "in", "ins", "mit", "oder",
    "und", "zum",
}


def rewritten_step_is_grounded(source: str, rewritten: str) -> bool:
    source_tokens = {
        token
        for token in normalize_recipe_name(source).split()
        if len(token) > 2 and token not in RECIPE_STOP_WORDS
    }
    rewritten_tokens = {
        token
        for token in normalize_recipe_name(rewritten).split()
        if len(token) > 2 and token not in RECIPE_STOP_WORDS
    }
    if not source_tokens:
        return True
    return bool(source_tokens & rewritten_tokens)


async def rewrite_recipe_with_llm(recipe: dict[str, Any]) -> dict[str, Any]:
    source_payload = {
        "name": recipe["name"],
        "ingredients": recipe["ingredients"],
        "instructions": recipe["instructions"],
        "garnish": recipe.get("garnish", []),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein deutschsprachiger Rezeptredakteur. Das bereitgestellte Cocktailrezept ist bereits "
                "auf Deutsch. Formuliere Zutaten und Zubereitung uebersichtlich, grammatikalisch korrekt und "
                "natuerlich in eigenen Worten. Formuliere Arbeitsschritte im ueblichen deutschen Rezeptstil mit "
                "Infinitiv, zum Beispiel 'Minzzweige mit Zucker und Limettensaft vermengen', 'Rum eingiessen' "
                "oder 'Mit Minze und Limette garnieren'. Formuliere Zutaten als korrekte deutsche Nominalgruppen, "
                "zum Beispiel '20 ml frischer Limettensaft' statt '20 ml frische Limettensaft'. "
                "Bleibe strikt bei den Quelldaten: Veraendere keine Mengen, Einheiten oder Zutaten und fuege "
                "keine Arbeitsschritte, Zutaten oder Tipps hinzu. Behalte fuer ingredients, instructions und "
                "garnish jeweils exakt dieselbe Anzahl an Eintraegen wie in der Quelle. Cocktailnamen bleiben "
                "unveraendert. Ersetze keine Zutat durch eine andere und interpretiere keine fehlenden Angaben. "
                "Bearbeite jeden Listeneintrag strikt einzeln und positionsgleich: ingredients[0] darf nur "
                "ingredients[0] umformulieren, instructions[0] nur instructions[0] und so weiter. Teile Eintraege "
                "nicht auf, fasse sie nicht zusammen und verschiebe keinen Inhalt in einen anderen Eintrag. "
                "Wenn ein deutscher Eintrag bereits klar formuliert ist, uebernimm ihn unveraendert. "
                "Gib ausschliesslich gueltiges JSON mit genau den Feldern ingredients, "
                "instructions und garnish zurueck. Alle Werte sind Listen deutscher Texte."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Erwartet werden exakt {len(recipe['ingredients'])} Zutaten, "
                f"{len(recipe['instructions'])} Arbeitsschritte und "
                f"{len(recipe.get('garnish', []))} Garnitur-Eintraege.\n"
                "QUELLREZEPT: " + json.dumps(source_payload, ensure_ascii=False)
            ),
        },
    ]
    payload: dict[str, Any] | None = None
    for attempt in range(2):
        raw_response = await query_ollama(messages)
        payload = extract_json_object(raw_response)
        try:
            validate_rewritten_recipe(payload, recipe)
            break
        except InvalidLLMOutputError:
            if attempt == 1:
                raise
            messages.extend([
                {"role": "assistant", "content": raw_response},
                {
                    "role": "user",
                    "content": (
                        "Korrigiere nur die JSON-Struktur. Gib exakt "
                        f"{len(recipe['ingredients'])} Zutaten, "
                        f"{len(recipe['instructions'])} Arbeitsschritte und "
                        f"{len(recipe.get('garnish', []))} Garnitur-Eintraege aus. "
                        "Lasse keinen Quelleninhalt weg und fuege nichts hinzu."
                    ),
                },
            ])

    if payload is None:
        raise InvalidLLMOutputError("Die deutsche Rezeptfassung war leer.")
    rewritten_instructions = [
        rewritten.strip()
        if rewritten_step_is_grounded(source, rewritten)
        else source
        for source, rewritten in zip(recipe["instructions"], payload["instructions"])
    ]
    rewritten_garnish = [
        rewritten.strip()
        if rewritten_step_is_grounded(source, rewritten)
        else source
        for source, rewritten in zip(recipe.get("garnish", []), payload["garnish"])
    ]
    return {
        **recipe,
        "ingredients": recipe["ingredients"],
        "instructions": rewritten_instructions,
        "garnish": rewritten_garnish,
    }


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
        recipe = await fetch_cocktail_database_recipe(cocktail_name)
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
            f"Für {cocktail_name} habe ich bei Cocktaildatenbank.de kein Rezept gefunden. "
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

    try:
        localized_recipe = await rewrite_recipe_with_llm(recipe)
    except (LLMError, InvalidLLMOutputError) as exc:
        logger.warning("Recipe localization failed, using extracted source text: %s", exc)
        localized_recipe = recipe

    answer = format_home_recipe_answer(localized_recipe)
    return {
        "type": "message",
        "intent": "catalog_query",
        "message": answer,
        "answer": answer,
        "cocktails": [],
        "web_recipes": [localized_recipe],
        "criteria": None,
    }
