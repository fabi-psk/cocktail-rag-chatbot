import json
import logging
import os
import random
import re
from difflib import SequenceMatcher
from typing import Any

from llm.llm_client import InvalidLLMOutputError, LLMError, extract_json_object, query_ollama
from models.cocktail import (
    CatalogContext,
    ChatIntent,
    ChatMessage,
    CocktailSearchCriteria,
    IntentAnalysis,
)
from repositories.cocktail_repository import CocktailRepository, CocktailRepositoryError
from services.cocktail_service import cocktail_contains, matches_strength, normalize_text, search_cocktails, term_matches
from services.conversation_service import ConversationService, conversation_service


logger = logging.getLogger("cocktail-rag")


def intent_routing_enabled() -> bool:
    return os.getenv("INTENT_ROUTING_ENABLED", "true").lower() == "true"


def detect_cocktail_name(user_message: str, cocktails: list[dict[str, Any]]) -> str | None:
    matches = find_cocktails_by_name(user_message, cocktails)
    return matches[0].get("name") if matches else None


def singular_token(token: str) -> str:
    return token[:-1] if len(token) > 4 and token.endswith("s") else token


NAME_MATCH_STOP_WORDS = {
    "aber", "auch", "das", "dem", "den", "der", "die", "ein", "eine", "einen",
    "einer", "er", "es", "euer", "eure", "eurer", "in", "im", "ist", "karte",
    "mit", "nicht", "of", "oder", "sind", "the", "und", "was",
}


def compact_name_similarity(compact_name: str, message_tokens: list[str], name_length: int) -> float:
    candidates = list(message_tokens)
    for size in range(2, min(name_length + 1, len(message_tokens)) + 1):
        candidates.extend(
            "".join(message_tokens[index:index + size])
            for index in range(len(message_tokens) - size + 1)
        )
    return max(
        (SequenceMatcher(None, compact_name, candidate).ratio() for candidate in candidates),
        default=0.0,
    )


def find_cocktails_by_name(
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_message = normalize_text(user_message)
    compact_message = normalized_message.replace(" ", "")
    ordered_message_tokens = [
        singular_token(token)
        for token in normalized_message.split()
        if token not in NAME_MATCH_STOP_WORDS
    ]
    message_tokens = set(ordered_message_tokens)
    ranked: list[tuple[int, int, int, dict[str, Any]]] = []
    for cocktail in cocktails:
        name = cocktail.get("name")
        if not isinstance(name, str):
            continue
        normalized_name = normalize_text(name)
        name_tokens = [
            singular_token(token)
            for token in normalized_name.split()
            if token not in NAME_MATCH_STOP_WORDS
        ]
        if not name_tokens:
            continue
        overlap = sum(token in message_tokens for token in name_tokens)
        token_name_match = overlap == len(name_tokens)
        compact_name = normalized_name.replace(" ", "")
        compact_name_match = compact_name in compact_message
        fuzzy_name_match = (
            not compact_name_match
            and len(compact_name) >= 5
            and compact_name_similarity(
                compact_name, ordered_message_tokens, len(normalized_name.split())
            ) >= 0.86
        )
        if not token_name_match and not compact_name_match and not fuzzy_name_match:
            continue
        missing_name_tokens = 0 if compact_name_match or fuzzy_name_match else len(name_tokens) - overlap
        match_rank = 0 if compact_name_match else 1 if fuzzy_name_match else 2
        ranked.append((match_rank, missing_name_tokens, -overlap, cocktail))

    ranked.sort(key=lambda item: (item[0], item[1], item[2], len(item[3].get("name", ""))))
    return [item[3] for item in ranked]


def detect_queried_flavor(user_message: str, cocktails: list[dict[str, Any]]) -> str | None:
    flavors = unique_text_values(cocktails, "geschmack")
    return next((flavor for flavor in flavors if term_matches(flavor, user_message)), None)


def is_availability_question(user_message: str) -> bool:
    normalized = normalize_text(user_message)
    return any(phrase in normalized for phrase in {
        "habt ihr", "habe ihr", "hast du", "gibt es", "gibts", "auf der karte", "im angebot",
    })


def is_catalog_query(user_message: str) -> bool:
    normalized = normalize_text(user_message)
    recommendation_explanation = (
        any(word in normalized.split() for word in {"warum", "wieso", "weshalb"})
        and any(term in normalized for term in {
            "vorschlag", "vorgeschlagen", "empfehl", "schlaegst", "schlagst",
        })
    )
    return recommendation_explanation or is_availability_question(user_message) or is_more_catalog_request(user_message) or any(phrase in normalized for phrase in {
        "welche cocktails enthalten", "welche drinks enthalten", "welche cocktails mit",
        "welche drinks mit", "welche cocktails sind", "welche drinks sind",
    })


def is_recommendation_explanation_query(user_message: str) -> bool:
    normalized = normalize_text(user_message)
    return (
        any(word in normalized.split() for word in {"warum", "wieso", "weshalb"})
        and any(term in normalized for term in {
            "vorschlag", "vorgeschlagen", "empfehl", "schlaegst", "schlagst",
        })
    )


def is_previous_cocktail_detail_query(
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> bool:
    normalized = normalize_text(user_message)
    words = set(normalized.split())
    pronouns = {"der", "den", "dieser", "er", "ihn"}
    detail_markers = {
        "preis", "kostet", "zutat", "zutaten", "drin", "enthalten", "rezept",
        "staerke", "stärke", "stark", "beschreibung", "geschmack", "schmeckt",
    }
    return bool(
        words.intersection(pronouns)
        and (
            detect_queried_flavor(user_message, cocktails)
            or any(marker in normalized for marker in detail_markers)
        )
    )


def resolve_previous_cocktail_detail_message(
    user_message: str,
    cocktail_name: str,
    cocktails: list[dict[str, Any]],
) -> str:
    normalized = normalize_text(user_message)
    queried_flavor = detect_queried_flavor(user_message, cocktails)
    if queried_flavor:
        return f"Ist {cocktail_name} {queried_flavor}?"
    if any(marker in normalized for marker in {"preis", "kostet", "kosten"}):
        return f"Was kostet {cocktail_name}?"
    if any(marker in normalized for marker in {"zutat", "zutaten", "drin", "enthalten", "rezept"}):
        return f"Welche Zutaten sind in {cocktail_name}?"
    if any(marker in normalized for marker in {"staerke", "stärke", "stark"}):
        return f"Wie stark ist {cocktail_name}?"
    return f"Wie schmeckt {cocktail_name}?"


def is_more_catalog_request(user_message: str) -> bool:
    normalized = normalize_text(user_message)
    return any(phrase in normalized for phrase in {
        "noch mehr", "weitere", "mehr davon", "sonst noch",
    })


def is_explicit_catalog_list_query(
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> bool:
    if not is_catalog_query(user_message):
        return False
    normalized = normalize_text(user_message)
    list_markers = {"cocktail", "cocktails", "drink", "drinks", "etwas", "welche"}
    if not any(marker in normalized.split() for marker in list_markers):
        return False
    return has_search_criteria(fallback_search_criteria(user_message, cocktails))


def is_greeting_message(user_message: str) -> bool:
    words = set(normalize_text(user_message).split())
    greeting_words = {
        "hallo", "hi", "hey", "moin", "servus", "danke", "tschuss", "tschuess",
        "guten", "morgen", "abend", "tag",
    }
    return bool(words) and words <= greeting_words


def is_cocktail_detail_query(user_message: str, cocktails: list[dict[str, Any]]) -> bool:
    normalized = normalize_text(user_message)
    words = set(normalized.split())
    cocktail_name = detect_cocktail_name(user_message, cocktails)
    queried_flavor = detect_queried_flavor(user_message, cocktails)
    if cocktail_name and queried_flavor and (
        "?" in user_message or any(word in words for word in {"ist", "sind", "auch", "wirklich"})
    ):
        return True

    detail_markers = {
        "preis", "kostet", "kosten", "zutat", "zutaten", "drin", "enthalten",
        "rezept", "staerke", "stärke", "stark", "beschreibung", "spirituose",
    }
    detail_questions = {
        "was kostet", "wie teuer", "was ist in", "welche zutaten", "wie stark ist",
        "informationen zu", "erzaehl mir etwas ueber", "erzähl mir etwas über",
    }
    return bool(
        (cocktail_name and any(marker in normalized for marker in detail_markers))
        or any(question in normalized for question in detail_questions)
    )


def is_random_request(user_message: str) -> bool:
    normalized = normalize_text(user_message).strip()
    if not normalized:
        return False

    random_phrases = {
        "ueberrasch mich",
        "uberrasch mich",
        "zufaelliger cocktail",
        "zufalliger cocktail",
        "zufaelligen cocktail",
        "zufalligen cocktail",
        "zufaellig",
        "zufallig",
        "such mir irgendwas aus",
        "such mir etwas aus",
        "such was aus",
        "ich kann mich nicht entscheiden",
        "mach mir einen zufaelligen cocktail",
        "mach mir einen zufalligen cocktail",
        "schlage mir einen zufaelligen cocktail vor",
        "schlage mir einen zufalligen cocktail vor",
        "roulette",
        "nochmal",
        "nochmal bitte",
    }
    return any(phrase in normalized for phrase in random_phrases)


def detect_intent(
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> ChatIntent:
    normalized = normalize_text(user_message).strip()
    words = set(normalized.split())

    if is_greeting_message(user_message):
        return "conversation"

    short_conversation_replies = {"ne", "nee", "nein", "noe", "ja", "jo", "okay", "ok"}
    if normalized in short_conversation_replies:
        return "conversation"

    if normalized in {"ich mag cocktail", "ich mag cocktails", "cocktails mag ich"}:
        return "conversation"

    if is_random_request(user_message):
        return "random"

    if is_catalog_query(user_message):
        return "catalog_query"

    if is_cocktail_detail_query(user_message, cocktails):
        return "catalog_query"
    out_of_scope_markers = {
        "wetter", "fussball", "fußball", "aktien", "programmieren", "python", "politik",
        "pizza", "hotel", "flug", "nachrichten", "hausaufgabe", "beziehung",
        "liebeskummer", "trennung", "partnerin", "partner", "gesundheit", "arzt",
        "anwalt", "rechtsberatung", "finanzberatung",
    }
    if any(marker in normalized for marker in out_of_scope_markers):
        return "out_of_scope"

    recommendation_markers = {
        "empfiehl", "empfehl", "such", "find", "zeig", "welcher cocktail", "welchen cocktail",
        "vorschlag", "ueberrasch", "überrasch", "zufall", "lust auf", "etwas anderes",
        "ich moechte", "ich möchte", "ich haette gern", "ich hätte gern", "ich mag",
    }
    if any(marker in normalized for marker in recommendation_markers):
        return "recommendation"

    if any(word in words for word in {"cocktail", "cocktails", "drink", "drinks"}):
        return "conversation"

    if any(phrase in normalized for phrase in {"gut und dir", "mir geht es", "mir gehts", "wie geht es dir"}):
        return "conversation"

    return "unknown"


async def analyze_intent_with_llm(
    user_message: str,
    history: list[ChatMessage] | None,
    cocktails: list[dict[str, Any]],
    catalog_context: CatalogContext | None = None,
) -> IntentAnalysis:
    cocktail_names = [cocktail.get("name") for cocktail in cocktails if cocktail.get("name")]
    values = catalog_values(cocktails)
    messages: list[dict[str, str]] = [{
        "role": "system",
        "content": (
            "Du bist der semantische Router eines Cocktail-Assistenten. Interpretiere die aktuelle Nachricht frei "
            "im Kontext des Verlaufs und des strukturierten Sitzungszustands. Der Anwendungscode wird deine "
            "Entscheidung nicht anhand einzelner Woerter umdeuten. Du beantwortest Kartenfragen noch nicht selbst, "
            "sondern beschreibst, welche Daten benoetigt werden. Erfinde keine Fakten. "
            "Gib ausschliesslich JSON mit genau diesen Feldern zurueck: intent, action, cocktail_name, attribute, "
            "value, context_mode, confidence, answer. Erlaubte Intents: conversation, recommendation, "
            "random, catalog_query, out_of_scope, unknown. "
            "Erlaubte actions: respond, recommend, random, check_availability, check_attribute, "
            "list_catalog, explain_recommendation, reject, clarify. "
            "Nutze respond nur fuer conversation. Bei catalog_query muss action check_availability, check_attribute, "
            "list_catalog oder explain_recommendation sein. Wenn sich die aktuelle Nachricht auf den im Zustand "
            "gespeicherten Cocktail bezieht und etwas ueber ihn wissen will, ist sie catalog_query und nicht conversation. "
            "attribute ist null oder availability, price, ingredients, flavor, strength, description, recipe. "
            "context_mode ist new_query, previous_cocktail, previous_search oder unclear. "
            "new_query bedeutet eine neue Suche. previous_cocktail gilt bei einem Bezug auf den zuletzt besprochenen "
            "Cocktail. Entscheide anhand der Bedeutung der gesamten aktuellen Nachricht, nicht anhand der letzten Antwort. "
            "previous_search bedeutet, dass nur die unmittelbar vorherige Kartensuche fortgesetzt wird. "
            "Wenn der Bezug nicht sicher erkennbar ist, nutze unclear und action clarify. "
            "confidence ist eine Zahl von 0 bis 1. Nutze den Verlauf, um Pronomen wie 'ihn' aufzuloesen. "
            "Normalisiere einen erkannten Cocktailnamen auf einen Namen aus COCKTAILNAMEN_DER_KARTE. "
            "recommendation gilt nur, wenn der Nutzer ausdruecklich eine Empfehlung, Suche oder Auswahl verlangt. "
            "random gilt fuer Wuensche wie 'Ueberrasch mich', 'zufaelliger Cocktail', "
            "'such mir irgendwas aus' oder 'ich kann mich nicht entscheiden'. "
            "catalog_query prueft nur Fakten, Eigenschaften und Verfuegbarkeit in der Cocktailkarte, ohne etwas zu empfehlen. "
            "Fragen nach Existenz, Eigenschaften, Zutaten, Preis oder Beschreibung der Karte sind catalog_query. "
            "Waehle das Attribut nach der Bedeutung: ingredients fuer Fragen zur Zusammensetzung und description "
            "fuer offene oder subjektive Fragen zum Charakter eines Cocktails. Die spaetere Antwort-LLM wertet "
            "das gewaehlte Datenfeld passend zur konkreten Frage semantisch aus. "
            "Eine allgemeine Aussage wie 'Ich mag Cocktails' ist conversation, keine recommendation. "
            "Begruessungen, rein soziale Hoeflichkeit und kurze Antworten wie 'gut und dir?' sind conversation. "
            "Substantielle Fragen oder Bitten um Rat ausserhalb von Cocktails sind niemals conversation, sondern "
            "out_of_scope. Dazu gehoeren insbesondere persoenliche Probleme, Beziehungen, Liebe, Familie, psychische "
            "oder koerperliche Gesundheit, Recht, Finanzen, Politik, Schule, Beruf und allgemeine Wissensfragen. "
            "Eine konkrete Vorliebe wie 'Ich mag Gin' oder 'stark und cremig' ist eine einmalige recommendation; "
            "sie wird nicht gespeichert. Bei conversation, out_of_scope und unknown schreibst du in answer eine "
            "kurze, grammatikalisch natuerliche deutsche Antwort. Bei einer Begruessung begruesst du kurz und fragst "
            "direkt nach Cocktailwuenschen oder Geschmack, niemals nach dem persoenlichen Befinden. Sonst reagierst "
            "du auf den bisherigen Verlauf und leitest mit einer passenden Rueckfrage zum Cocktail-Thema zurueck. "
            "answer muss dabei ein Fragezeichen enthalten. Wiederhole keine vorherige Assistentenantwort. Bei "
            "out_of_scope beantwortest du die fachfremde Frage inhaltlich nicht und gibst keinerlei Tipps, Analyse, "
            "Bewertung oder Handlungsempfehlung dazu. Antworte stattdessen in hoechstens zwei kurzen Saetzen: Grenze "
            "freundlich deine Rolle als Cocktail-Assistent ab und stelle genau eine konkrete Uebergangsfrage zu "
            "Cocktailwuenschen, Geschmack, Zutaten oder einem Rezept. Duze den Nutzer immer und vermeide unpassende "
            "Floskeln oder Wuensche wie 'Viel Spass'. "
            "Bei recommendation, random und catalog_query bleibt answer leer. "
            "value ist immer null oder Text, niemals ein Boolean, Objekt oder Array. Pruefe vor der Ausgabe intern, "
            "dass alle acht Felder vorhanden sind und zum gewaehlten Intent passen. Bei Mehrdeutigkeit nutze action "
            "clarify, intent unknown und erklaere die Rueckfrage in answer.\n\n"
            f"COCKTAILNAMEN_DER_KARTE: {json.dumps(cocktail_names, ensure_ascii=False)}\n"
            f"GESCHMACKSWERTE: {json.dumps(values['flavors'], ensure_ascii=False)}\n"
            f"SPIRITUOSEN: {json.dumps(values['spirits'], ensure_ascii=False)}\n"
            f"ZUTATEN: {json.dumps(values['ingredients'], ensure_ascii=False)}"
        ),
    }]
    if catalog_context:
        messages.append({
            "role": "system",
            "content": "STRUKTURIERTER_SITZUNGSZUSTAND: " + catalog_context.model_dump_json(),
        })
    if history:
        for message in history[-4:]:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": user_message})

    parsed = sanitize_interpretation_payload(
        extract_json_object(await query_ollama(messages)), cocktails
    )
    try:
        analysis = IntentAnalysis.model_validate(parsed)
    except Exception as exc:
        raise InvalidLLMOutputError("LLM-Intent konnte nicht validiert werden.") from exc
    if not intent_answer_is_usable(analysis, user_message, history):
        raise InvalidLLMOutputError("LLM-Intent-Antwort war unvollstaendig.")
    return analysis


def sanitize_interpretation_payload(
    parsed: dict[str, Any],
    cocktails: list[dict[str, Any]],
) -> dict[str, Any]:
    attribute = parsed.get("attribute")
    value = parsed.get("value")
    if attribute == "availability" or not isinstance(value, str):
        value = None

    context_aliases = {
        "new": "new_query",
        "new_search": "new_query",
        "previous": "previous_cocktail",
        "previous_query": "previous_cocktail",
        "search": "previous_search",
    }
    context_mode = context_aliases.get(parsed.get("context_mode"), parsed.get("context_mode"))
    if context_mode not in {
        "new_query", "previous_cocktail", "previous_search", "unclear",
    }:
        context_mode = "unclear"

    return {
        "intent": parsed.get("intent"),
        "answer": parsed.get("answer", ""),
        "action": parsed.get("action", "respond"),
        "cocktail_name": parsed.get("cocktail_name"),
        "attribute": attribute,
        "value": value,
        "context_mode": context_mode,
        "confidence": parsed.get("confidence", 1.0),
    }


def canonical_cocktail_name(value: str | None, cocktails: list[dict[str, Any]]) -> str | None:
    if not value:
        return None
    matches = find_cocktails_by_name(value, cocktails)
    return matches[0].get("name") if matches else None


def normalize_interpretation(
    analysis: IntentAnalysis,
    user_message: str,
    cocktails: list[dict[str, Any]],
    catalog_context: CatalogContext | None = None,
) -> IntentAnalysis:
    cocktail_name = (
        canonical_cocktail_name(analysis.cocktail_name, cocktails)
        or detect_cocktail_name(user_message, cocktails)
    )
    should_resolve_reference = (
        analysis.action == "explain_recommendation"
        or analysis.context_mode == "previous_cocktail"
    )
    if not cocktail_name and should_resolve_reference and catalog_context:
        cocktail_name = canonical_cocktail_name(
            catalog_context.referenced_cocktail, cocktails
        )

    value = analysis.value
    if analysis.attribute == "flavor":
        value = canonical_value(value, unique_text_values(cocktails, "geschmack"))
    elif analysis.attribute == "ingredients":
        value = canonical_value(value, unique_text_values(cocktails, "zutaten"))
    elif analysis.attribute == "strength":
        value = canonical_value(value, unique_text_values(cocktails, "staerke"))

    action = analysis.action
    if action == "check_attribute" and not cocktail_name:
        action = "clarify"

    return analysis.model_copy(update={
        "action": action,
        "cocktail_name": cocktail_name,
        "value": value,
    })


def interpreted_catalog_message(analysis: IntentAnalysis, original_message: str) -> str:
    name = analysis.cocktail_name
    if analysis.action == "check_availability" and name:
        return f"Habt ihr {name}?"
    if analysis.action == "check_attribute" and name:
        if analysis.attribute == "price":
            return f"Was kostet {name}?"
        if analysis.attribute in {"ingredients", "recipe"}:
            return f"Welche Zutaten sind in {name}?"
        if analysis.attribute == "strength":
            return f"Wie stark ist {name}?"
        if analysis.attribute == "flavor" and analysis.value:
            return f"Ist {name} {analysis.value}?"
        if analysis.attribute in {"flavor", "description"}:
            return f"Wie schmeckt {name}?"
    return original_message


def interpretation_clarification_response(
    analysis: IntentAnalysis,
) -> dict[str, Any]:
    answer = analysis.answer.strip() or (
        "Ich bin mir nicht sicher, was ich prüfen soll. Nenne mir bitte den Cocktail und die gewünschte Information."
    )
    return {
        "type": "follow_up",
        "intent": "unknown",
        "message": answer,
        "answer": answer,
        "cocktails": [],
        "criteria": None,
    }


def execute_catalog_action(
    analysis: IntentAnalysis,
    cocktails: list[dict[str, Any]],
    catalog_context: CatalogContext,
) -> tuple[list[dict[str, Any]], CocktailSearchCriteria | None]:
    """Execute the LLM-selected catalog operation without reclassifying the message."""
    if analysis.action in {
        "check_availability", "check_attribute", "explain_recommendation",
    }:
        matches = find_cocktails_by_name(analysis.cocktail_name or "", cocktails)
        return matches[:1], None

    if analysis.action != "list_catalog":
        return [], None

    if analysis.cocktail_name:
        matches = find_cocktails_by_name(analysis.cocktail_name, cocktails)
        return matches[:10], None

    if analysis.context_mode == "previous_search" and catalog_context.criteria:
        criteria = catalog_context.criteria
        shown_names = set(catalog_context.shown_names)
        matches = [
            cocktail
            for cocktail in search_cocktails(criteria, cocktails, limit=1000)
            if cocktail.get("name") not in shown_names
        ]
        return matches[:10], criteria

    value = analysis.value
    if not value:
        return cocktails[:10], None

    criteria: CocktailSearchCriteria | None = None
    if analysis.attribute == "flavor":
        criteria = CocktailSearchCriteria(geschmack=value)
        matches = [
            cocktail for cocktail in cocktails
            if any(term_matches(value, item) for item in cocktail.get("geschmack", []))
        ]
    elif analysis.attribute == "strength":
        criteria = CocktailSearchCriteria(staerke=value)
        matches = [cocktail for cocktail in cocktails if matches_strength(value, cocktail)]
    elif analysis.attribute == "ingredients":
        matches = [
            cocktail for cocktail in cocktails
            if any(term_matches(value, item) for item in cocktail.get("zutaten", []))
            or any(term_matches(value, item) for item in cocktail.get("spirituose", []))
        ]
    else:
        matches = [cocktail for cocktail in cocktails if cocktail_contains(cocktail, value)]

    return matches[:10], criteria


async def generate_grounded_catalog_answer(
    user_message: str,
    analysis: IntentAnalysis,
    matches: list[dict[str, Any]],
    history: list[ChatMessage] | None = None,
) -> str:
    catalog_json = json.dumps(
        [public_cocktail(cocktail) for cocktail in matches], ensure_ascii=False
    )
    messages: list[dict[str, str]] = [{
        "role": "system",
        "content": (
            "Du bist CocktailGPT. Ein vorgeschalteter semantischer Router hat die Nutzerfrage interpretiert und "
            "das Backend hat die passende Kartenabfrage bereits ausgefuehrt. Antworte jetzt natuerlich und direkt "
            "auf die aktuelle Frage. Nutze ausschliesslich Fakten aus DATENBANKERGEBNIS; erfinde keine Cocktails, "
            "Zutaten, Eigenschaften oder Preise. Werte die vorhandenen Felder semantisch aus: Wenn nach Fruechten "
            "gefragt wird, nenne nur passende fruchtbezogene Zutaten. Bei subjektiven Fragen erklaere, dass dies "
            "Geschmackssache ist, und beschreibe den Cocktail anhand der Kartendaten. Wenn das Ergebnis leer ist, "
            "sage klar, dass dazu kein Eintrag gefunden wurde. Gib ausschliesslich JSON mit dem Feld answer zurueck.\n\n"
            f"ROUTER_ENTSCHEIDUNG: {analysis.model_dump_json()}\n"
            f"DATENBANKERGEBNIS: {catalog_json}"
        ),
    }]
    if history:
        for message in history[-6:]:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": user_message})

    raw_answer = (await query_ollama(messages)).strip()
    try:
        parsed = extract_json_object(raw_answer)
    except InvalidLLMOutputError:
        return raw_answer

    answer = parsed.get("answer")
    if isinstance(answer, list) and all(isinstance(item, str) for item in answer):
        values = [item.strip() for item in answer if item.strip()]
        if len(values) > 1:
            answer = ", ".join(values[:-1]) + " und " + values[-1] + "."
        elif values:
            answer = values[0] + "."
    if not isinstance(answer, str):
        text_values = [value for value in parsed.values() if isinstance(value, str)]
        answer = text_values[0] if len(text_values) == 1 else None
    if not isinstance(answer, str) or not answer.strip():
        raise InvalidLLMOutputError("LLM-Katalogantwort war unvollstaendig.")
    return answer.strip()


def build_catalog_fallback_answer(
    analysis: IntentAnalysis,
    matches: list[dict[str, Any]],
) -> str:
    if not matches:
        return "Dazu habe ich auf unserer Cocktailkarte keinen passenden Eintrag gefunden."
    if analysis.action == "check_availability":
        return "Auf unserer Karte gibt es: " + ", ".join(item["name"] for item in matches) + "."
    cocktail = matches[0]
    if analysis.attribute == "flavor" and analysis.value:
        has_flavor = any(
            term_matches(analysis.value, flavor)
            for flavor in cocktail.get("geschmack", [])
        )
        if has_flavor:
            return f"Ja, {cocktail['name']} ist laut unserer Karte {analysis.value}."
        return f"Nein, {cocktail['name']} ist laut unserer Karte nicht als {analysis.value} eingeordnet."
    return (
        f"{cocktail['name']}: {cocktail.get('beschreibung', '')} "
        f"Zutaten: {', '.join(cocktail.get('zutaten', []))}."
    ).strip()


def intent_answer_is_usable(
    analysis: IntentAnalysis,
    user_message: str = "",
    history: list[ChatMessage] | None = None,
) -> bool:
    answer_intents = {"conversation", "out_of_scope", "unknown"}
    if analysis.intent not in answer_intents:
        return True
    if not analysis.answer.strip():
        return False

    if analysis.intent == "conversation" and is_greeting_message(user_message):
        normalized_answer = normalize_text(analysis.answer)
        if any(phrase in normalized_answer for phrase in {"wie geht es dir", "wie gehts dir", "wie geht es ihnen"}):
            return False

    if analysis.intent == "conversation" and "?" not in analysis.answer:
        return False

    if analysis.intent == "conversation" and contains_recommendation_language(
        analysis.answer
    ):
        return False

    if analysis.intent == "out_of_scope":
        normalized_answer = normalize_text(analysis.answer)
        has_cocktail_redirect = "?" in analysis.answer and any(
            term in normalized_answer
            for term in {"cocktail", "geschmack", "zutat", "rezept", "alkoholfrei"}
        )
        if not has_cocktail_redirect or len(analysis.answer) > 320:
            return False

    if history:
        previous_answers = [message.content for message in history if message.role == "assistant"]
        if previous_answers:
            current = normalize_text(analysis.answer)
            previous = normalize_text(previous_answers[-1])
            if current and previous and SequenceMatcher(None, current, previous).ratio() >= 0.78:
                return False

    return True


def contains_recommendation_language(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(phrase in normalized for phrase in {
        "ich empfehle", "ich schlage", "wie waere es mit", "wie wäre es mit",
        "probier", "versuch den", "versuch die", "cocktail vorschlag",
    })


async def generate_intent_answer(
    intent: ChatIntent,
    user_message: str,
    history: list[ChatMessage] | None,
) -> str:
    messages: list[dict[str, str]] = [{
        "role": "system",
        "content": (
            "Du bist CocktailGPT und antwortest kurz, natuerlich und auf Deutsch. Duze den Nutzer. "
            f"Die bereits gepruefte Absicht ist: {intent}. "
            "Reagiere auf die aktuelle Nachricht und den Verlauf, ohne eine vorherige Antwort zu wiederholen. "
            "Bei conversation leitest du freundlich zu Cocktailwuenschen, Geschmacksrichtungen oder "
            "Zutaten ueber und stellst genau eine passende Frage. Frage nicht nach dem persoenlichen Befinden. "
            "Nenne bei conversation keinen konkreten Cocktail. Konkrete Empfehlungen werden an anderer "
            "Stelle aus der geprueften Cocktailkarte erzeugt. "
            "Wenn bereits ein Verlauf existiert, begruesse nicht erneut. Du hilfst dem Nutzer bei der Auswahl und "
            "forderst den Nutzer niemals auf, dir bei der Auswahl zu helfen. "
            "Bei out_of_scope beantwortest du die fachfremde Frage nicht, auch nicht teilweise. Gib keine Tipps, "
            "Analyse, Bewertung oder Handlungsempfehlung zum genannten Thema. Schreibe hoechstens zwei kurze Saetze: "
            "Grenze zuerst freundlich deine Rolle ab und stelle danach genau eine Frage zu Cocktailwuenschen, "
            "Geschmack, Zutaten oder einem Rezept. Bei unknown fragst du "
            "nach, ob eine Empfehlung oder Information zu einem Cocktail gesucht wird. Erfinde keine Cocktaildaten."
        ),
    }]
    if history:
        for message in history[-4:]:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": user_message})

    answer = (await query_ollama(messages, response_format="")).strip()
    if not answer:
        raise InvalidLLMOutputError("LLM lieferte keine Intent-Antwort.")
    if intent == "conversation" and contains_recommendation_language(answer):
        raise InvalidLLMOutputError("LLM-Gespraechsantwort enthielt eine ungepruefte Empfehlung.")
    if intent in {"conversation", "out_of_scope", "unknown"} and not intent_answer_is_usable(
        IntentAnalysis(intent=intent, answer=answer), user_message, history
    ):
        raise InvalidLLMOutputError("LLM-Intent-Antwort blieb nicht sicher im Cocktail-Kontext.")
    return answer


def basic_intent_response(
    intent: ChatIntent,
    generated_answer: str = "",
    user_message: str = "",
) -> dict[str, Any]:
    conversation_fallback = (
        "Hallo! Ich bin CocktailGPT. Suchst du einen bestimmten Geschmack oder eine Cocktail-Empfehlung?"
        if is_greeting_message(user_message)
        else "Cocktails sind wirklich vielseitig. Magst du sie eher fruchtig, cremig, sauer oder stark?"
    )
    messages = {
        "conversation": conversation_fallback,
        "out_of_scope": (
            "Dazu kann ich dir keine inhaltlichen Ratschläge geben. Ich bin dein Cocktail-Assistent. "
            "Suchst du eine Empfehlung, ein Rezept oder Informationen zu einem Cocktail?"
        ),
        "unknown": (
            "Ich habe deine Frage leider nicht verstanden. Suchst du eine Cocktail-Empfehlung oder "
            "Informationen zu einem bestimmten Cocktail?"
        ),
    }
    answer = generated_answer.strip() or messages[intent]
    return {
        "type": "message",
        "intent": intent,
        "message": answer,
        "answer": answer,
        "cocktails": [],
        "criteria": None,
    }


def build_cocktail_detail_response(
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> dict[str, Any]:
    name_matches = find_cocktails_by_name(user_message, cocktails)
    cocktail = name_matches[0] if name_matches else None
    if not cocktail:
        answer = (
            "Diesen Cocktail finde ich nicht auf unserer Karte. Nenne mir bitte einen Cocktail aus dem Angebot, "
            "dann kann ich dir Preis, Zutaten, Geschmack und Stärke nennen."
        )
        return {
            "type": "follow_up",
            "intent": "catalog_query",
            "message": answer,
            "answer": answer,
            "cocktails": [],
            "criteria": None,
        }

    normalized = normalize_text(user_message)
    queried_flavor = detect_queried_flavor(user_message, cocktails)
    if queried_flavor:
        has_flavor = any(
            term_matches(queried_flavor, flavor) for flavor in cocktail.get("geschmack", [])
        )
        if has_flavor:
            answer = f"Ja, {cocktail['name']} ist laut unserer Karte {queried_flavor}."
        else:
            answer = (
                f"Nein, {cocktail['name']} ist auf unserer Karte nicht als {queried_flavor} eingeordnet."
            )
        return {
            "type": "message",
            "intent": "catalog_query",
            "message": answer,
            "answer": answer,
            "cocktails": [cocktail],
            "criteria": None,
        }

    details: list[str] = []
    if any(term in normalized for term in {"preis", "kostet", "kosten"}):
        details.append(f"Er kostet {cocktail.get('preis', 0):.2f} Euro.")
    if any(term in normalized for term in {"zutat", "zutaten", "drin", "enthalten", "rezept"}):
        details.append("Enthalten sind: " + ", ".join(cocktail.get("zutaten", [])) + ".")
    if any(term in normalized for term in {"staerke", "stärke", "stark"}):
        details.append(f"Seine Stärke ist {cocktail.get('staerke', 'nicht angegeben')}.")
    if any(term in normalized for term in {"beschreibung", "geschmack", "schmeckt"}):
        details.append(cocktail.get("beschreibung", ""))
    if not details:
        details.append(cocktail.get("beschreibung", ""))

    answer = f"{cocktail['name']}: " + " ".join(detail for detail in details if detail)
    return {
        "type": "message",
        "intent": "catalog_query",
        "message": answer,
        "answer": answer,
        "cocktails": [cocktail],
        "criteria": None,
    }


def build_catalog_query_response(
    user_message: str,
    cocktails: list[dict[str, Any]],
    catalog_context: CatalogContext | None = None,
) -> dict[str, Any]:
    catalog_context = catalog_context or CatalogContext()
    if is_recommendation_explanation_query(user_message):
        referenced = find_cocktails_by_name(user_message, cocktails)
        if not referenced and catalog_context.referenced_cocktail:
            referenced = find_cocktails_by_name(
                catalog_context.referenced_cocktail, cocktails
            )

        cocktail = referenced[0] if referenced else None
        queried_flavor = detect_queried_flavor(user_message, cocktails)
        if cocktail and queried_flavor:
            has_flavor = any(
                term_matches(queried_flavor, flavor)
                for flavor in cocktail.get("geschmack", [])
            )
            if has_flavor:
                answer = (
                    f"{cocktail['name']} passt ebenfalls zu {queried_flavor}. "
                    "Ich zeige pro Empfehlung höchstens drei passende Treffer; dass dieser Cocktail nicht dabei war, "
                    "bedeutet keinen Ausschluss."
                )
            else:
                answer = (
                    f"{cocktail['name']} ist auf unserer Karte nicht als {queried_flavor} eingeordnet und wurde "
                    "deshalb dafür nicht vorgeschlagen."
                )
            return {
                "type": "message",
                "intent": "catalog_query",
                "message": answer,
                "answer": answer,
                "cocktails": [cocktail],
                "criteria": None,
            }

    if (
        catalog_context.referenced_cocktail
        and is_previous_cocktail_detail_query(user_message, cocktails)
    ):
        cocktail_name = canonical_cocktail_name(
            catalog_context.referenced_cocktail, cocktails
        )
        if cocktail_name:
            resolved_message = resolve_previous_cocktail_detail_message(
                user_message, cocktail_name, cocktails
            )
            return build_cocktail_detail_response(
                resolved_message, cocktails
            )

    if not is_catalog_query(user_message) and is_cocktail_detail_query(user_message, cocktails):
        return build_cocktail_detail_response(user_message, cocktails)

    more_request = is_more_catalog_request(user_message)
    criteria: CocktailSearchCriteria | None = None
    name_matches = [] if more_request else find_cocktails_by_name(user_message, cocktails)
    if name_matches:
        base_tokens = {
            singular_token(token)
            for token in normalize_text(name_matches[0].get("name", "")).split()
            if token not in NAME_MATCH_STOP_WORDS
        }
        matches = [
            item for item in cocktails
            if base_tokens.issubset({
                singular_token(token) for token in normalize_text(item.get("name", "")).split()
                if token not in NAME_MATCH_STOP_WORDS
            })
        ]
    else:
        criteria = fallback_search_criteria(user_message, cocktails)
        normalized_words = set(normalize_text(user_message).split())
        concrete_unknown_cocktail = (
            is_availability_question(user_message)
            and not is_explicit_catalog_list_query(user_message, cocktails)
            and bool(normalized_words.intersection({"ein", "eine", "einen"}))
        )
        if concrete_unknown_cocktail:
            criteria = CocktailSearchCriteria()
        if more_request and catalog_context.criteria:
            criteria = catalog_context.criteria
        if message_mentions_any(user_message, {"alkoholfrei", "ohne alkohol"}):
            criteria = criteria.model_copy(update={"staerke": "alkoholfrei"})
        matches = search_cocktails(criteria, cocktails, limit=1000) if has_search_criteria(criteria) else []
        if more_request:
            shown_names = set(catalog_context.shown_names)
            matches = [cocktail for cocktail in matches if cocktail["name"] not in shown_names]
        matches = matches[:10]

    if matches:
        names = [item["name"] for item in matches]
        prefix = "Auf unserer Karte gibt es außerdem: " if more_request else "Auf unserer Karte gibt es: "
        answer = prefix + ", ".join(names) + "."
    elif more_request:
        answer = "Weitere passende Cocktails habe ich zu dieser Suche nicht auf unserer Karte."
    else:
        answer = "Dazu habe ich auf unserer Cocktailkarte keinen passenden Eintrag gefunden."

    return {
        "type": "message",
        "intent": "catalog_query",
        "message": answer,
        "answer": answer,
        "cocktails": matches,
        "criteria": criteria.model_dump() if criteria and has_search_criteria(criteria) else None,
    }


def public_cocktail(cocktail: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": cocktail.get("name"),
        "preis": cocktail.get("preis"),
        "spirituose": cocktail.get("spirituose", []),
        "geschmack": cocktail.get("geschmack", []),
        "staerke": cocktail.get("staerke"),
        "zutaten": cocktail.get("zutaten", []),
        "beschreibung": cocktail.get("beschreibung"),
    }


def unique_text_values(cocktails: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for cocktail in cocktails:
        raw_value = cocktail.get(field, [])
        field_values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in field_values:
            if isinstance(value, str) and value and value.lower() not in seen:
                seen.add(value.lower())
                values.append(value)
    return values


def canonical_value(value: str | None, allowed_values: list[str]) -> str | None:
    if not value:
        return None
    normalized_value = normalize_text(value)
    for allowed in allowed_values:
        if normalize_text(allowed) == normalized_value:
            return allowed
    matches = [
        allowed
        for allowed in allowed_values
        if term_matches(value, allowed) or term_matches(allowed, value)
    ]
    if matches:
        return min(matches, key=len)
    return None


def inferred_value_from_message(user_message: str, allowed_values: list[str]) -> str | None:
    matches = [value for value in allowed_values if term_matches(value, user_message)]
    if not matches:
        return None
    return max(matches, key=len)


def normalize_search_criteria(
    criteria: CocktailSearchCriteria,
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> CocktailSearchCriteria:
    spirits = unique_text_values(cocktails, "spirituose")
    tastes = unique_text_values(cocktails, "geschmack")

    spirituose = canonical_value(criteria.spirituose, spirits) or inferred_value_from_message(user_message, spirits)
    geschmack = canonical_value(criteria.geschmack, tastes)

    if criteria.geschmack and not geschmack:
        geschmack = canonical_value(criteria.geschmack, tastes)

    geschmack = geschmack or inferred_value_from_message(user_message, tastes)

    staerke = criteria.staerke
    if staerke and not any(matches_strength(staerke, cocktail) for cocktail in cocktails):
        staerke = None

    return CocktailSearchCriteria(
        spirituose=spirituose,
        geschmack=geschmack,
        staerke=staerke,
        ausschluesse=criteria.ausschluesse,
    )


def infer_exclusions_from_message(user_message: str, cocktails: list[dict[str, Any]]) -> list[str]:
    normalized_message = normalize_text(user_message)
    words = normalized_message.split()
    exclusion_markers = {"ohne", "kein", "keine", "keinen"}
    if not any(marker in words for marker in exclusion_markers):
        return []

    exclusion_terms: list[str] = []
    for index, word in enumerate(words):
        if word in exclusion_markers:
            for term in words[index + 1:]:
                if term in {"und", "oder", "mit", "aber"}:
                    break
                exclusion_terms.append(term)

    values = (
        unique_text_values(cocktails, "zutaten")
        + unique_text_values(cocktails, "spirituose")
        + unique_text_values(cocktails, "geschmack")
    )
    exclusions: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized_value = normalize_text(value)
        if not normalized_value or normalized_value in seen:
            continue
        if any(term_matches(term, value) or term in normalized_value for term in exclusion_terms):
            seen.add(normalized_value)
            exclusions.append(value)
    return exclusions


def fallback_search_criteria(user_message: str, cocktails: list[dict[str, Any]]) -> CocktailSearchCriteria:
    criteria = normalize_search_criteria(CocktailSearchCriteria(), user_message, cocktails)
    exclusions = infer_exclusions_from_message(user_message, cocktails)
    return criteria.model_copy(update={"ausschluesse": exclusions})


def has_search_criteria(criteria: CocktailSearchCriteria) -> bool:
    return any([
        criteria.spirituose,
        criteria.geschmack,
        criteria.staerke,
        criteria.ausschluesse,
    ])


def catalog_values(cocktails: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "ingredients": unique_text_values(cocktails, "zutaten"),
        "spirits": unique_text_values(cocktails, "spirituose"),
        "flavors": unique_text_values(cocktails, "geschmack"),
    }


def message_mentions_any(user_message: str, needles: set[str]) -> bool:
    normalized = normalize_text(user_message)
    return any(needle in normalized for needle in needles)


def names_from_answer(answer: str, allowed_names: set[str]) -> set[str]:
    answer_lower = answer.lower()
    return {
        name
        for name in allowed_names
        if re.search(rf"(?<!\w){re.escape(name.lower())}(?!\w)", answer_lower)
    }


async def extract_search_criteria(user_message: str) -> CocktailSearchCriteria:
    messages = [
        {
            "role": "system",
            "content": (
                "Du extrahierst Suchkriterien fuer eine Cocktailkarte aus freier deutscher Nutzereingabe. "
                "Du bekommst keinen Cocktailkatalog und suchst keine Cocktails. "
                "Gib ausschliesslich ein JSON-Objekt mit genau diesen Feldern zurueck: "
                "spirituose, geschmack, staerke, ausschluesse. "
                "Extrahiere nur Angaben, die der Nutzer in dieser Nachricht ausdruecklich nennt. "
                "Leite keine typischen Zutaten, Spirituosen, Rezepte oder Ausschluesse aus einem Geschmack ab. "
                "Setze jedes nicht ausdruecklich genannte positive Feld auf null. ausschluesse enthaelt nur "
                "ausdruecklich ausgeschlossene Angaben und ist sonst eine leere Liste. "
                "staerke darf nur null, leicht, mittel, stark, hoch oder alkoholfrei sein. "
                "Beispiele: 'starker Cocktail' -> staerke 'stark'; 'ohne Kokos' -> ausschluesse ['Kokos']; "
                "'mit Rum' -> spirituose 'Rum'; 'saurer Cocktail' -> geschmack 'sauer', alle anderen Felder leer."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    raw_response = await query_ollama(messages)
    parsed = extract_json_object(raw_response)
    try:
        return CocktailSearchCriteria.model_validate(parsed)
    except Exception as exc:
        raise InvalidLLMOutputError("LLM-Kriterien konnten nicht validiert werden.") from exc


def build_answer_prompt(
    user_message: str,
    matching_cocktails: list[dict[str, Any]],
    history: list[ChatMessage] | None = None,
) -> list[dict[str, str]]:
    allowed_names = [cocktail["name"] for cocktail in matching_cocktails]
    cocktails_json = json.dumps([public_cocktail(c) for c in matching_cocktails], ensure_ascii=False)

    system_prompt = (
        "Du bist CocktailGPT, ein Barkeeper-Assistent. "
        "Das Backend hat die Cocktailkarte bereits deterministisch durchsucht. "
        "Du bekommst ausschliesslich die gefundenen Cocktails als Kontext in GEFUNDENE_COCKTAILS_JSON. "
        "Du darfst keine anderen Cocktails, Zutaten, Preise oder Staerken erfinden. "
        "Wenn keine Cocktails uebergeben wurden, sage freundlich, dass keine passenden Treffer gefunden wurden, "
        "und schlage vor, einzelne Kriterien zu lockern. "
        "Wenn Cocktails uebergeben wurden, empfehle hoechstens drei davon und nenne kurz, warum sie passen. "
        "cocktail_names muss genau die Cocktails enthalten, die du in answer nennst. "
        "Erwaehne in answer keinen Cocktailnamen, der nicht in ERLAUBTE_COCKTAILNAMEN steht. "
        "Gib ausschliesslich ein gueltiges JSON-Objekt ohne Markdown-Codeblock zurueck: "
        '{"answer":"deine deutsche Antwort","cocktail_names":["Name 1"]}\n\n'
        f"ERLAUBTE_COCKTAILNAMEN: {json.dumps(allowed_names, ensure_ascii=False)}\n\n"
        f"GEFUNDENE_COCKTAILS_JSON:\n{cocktails_json}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for message in history[-6:]:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": user_message})
    return messages


def validate_answer_payload(payload: dict[str, Any], matching_cocktails: list[dict[str, Any]]) -> None:
    if not isinstance(payload.get("answer"), str) or not isinstance(payload.get("cocktail_names"), list):
        raise InvalidLLMOutputError("LLM-Antwort muss answer und cocktail_names enthalten.")

    allowed_names = {cocktail["name"] for cocktail in matching_cocktails}
    requested_names = {
        name
        for name in payload["cocktail_names"]
        if isinstance(name, str) and name in allowed_names
    }
    if len(requested_names) != len(payload["cocktail_names"]):
        raise InvalidLLMOutputError("LLM-Antwort enthaelt unbekannte Cocktailnamen.")


def build_grounded_selection_answer(selected: list[dict[str, Any]]) -> str:
    if not selected:
        return (
            "Ich habe auf unserer Cocktailkarte keine passende Empfehlung gefunden. "
            "Magst du ein Kriterium ändern oder mir eine Geschmacksrichtung nennen?"
        )

    lines = ["Diese Cocktails aus unserer Karte passen zu deinen Wünschen:"]
    for cocktail in selected[:3]:
        tastes = ", ".join(cocktail.get("geschmack", [])[:3])
        spirits = ", ".join(cocktail.get("spirituose", [])) or "alkoholfrei"
        lines.append(
            f"- **{cocktail['name']}**: {tastes}; Basis {spirits}; Stärke {cocktail.get('staerke', 'unbekannt')}."
        )
    return "\n".join(lines)


async def generate_answer(
    user_message: str,
    matching_cocktails: list[dict[str, Any]],
    history: list[ChatMessage] | None = None,
) -> dict[str, Any]:
    raw_response = await query_ollama(build_answer_prompt(user_message, matching_cocktails, history))
    payload = extract_json_object(raw_response)
    validate_answer_payload(payload, matching_cocktails)

    by_name = {cocktail["name"]: cocktail for cocktail in matching_cocktails}
    selected = [by_name[name] for name in payload["cocktail_names"] if name in by_name]
    answer = build_grounded_selection_answer(selected)
    return {
        "message": answer,
        "answer": answer,
        "cocktails": selected,
    }


def build_local_answer(criteria: CocktailSearchCriteria, matching_cocktails: list[dict[str, Any]]) -> str:
    if not matching_cocktails:
        return (
            "Ich habe keine Cocktails gefunden, die alle Kriterien erfuellen. "
            "Lockere am besten eine Vorgabe, zum Beispiel Spirituose, Staerke oder ausgeschlossene Zutaten."
        )

    exclusion_text = ""
    if criteria.ausschluesse:
        exclusion_text = " ohne " + ", ".join(criteria.ausschluesse)

    lines = [f"Ich habe passende Cocktails{exclusion_text} gefunden:"]
    for cocktail in matching_cocktails[:3]:
        spirits = ", ".join(cocktail.get("spirituose", [])) or "alkoholfrei"
        tastes = ", ".join(cocktail.get("geschmack", [])[:3])
        lines.append(
            f"- **{cocktail['name']}**: {tastes}, Staerke {cocktail['staerke']}, Basis {spirits}."
        )
    return "\n".join(lines)


def select_recommendation_candidates(
    criteria: CocktailSearchCriteria,
    cocktails: list[dict[str, Any]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    matches = search_cocktails(criteria, cocktails, limit=len(cocktails))
    if len(matches) <= limit:
        return matches
    return random.sample(matches, limit)


def build_random_response(
    cocktails: list[dict[str, Any]],
    session_id: str | None,
    conversations: ConversationService,
) -> dict[str, Any]:
    candidates = [cocktail.copy() for cocktail in cocktails]

    if not candidates:
        answer = "Ich habe gerade keine Cocktails zum Auslosen gefunden."
        return {
            "type": "follow_up",
            "intent": "random",
            "message": answer,
            "answer": answer,
            "cocktails": [],
            "criteria": None,
            "roulette_cocktails": [],
            "selected_cocktail": None,
        }

    previous_name = conversations.get_last_random_cocktail(session_id)
    selection_pool = [
        cocktail for cocktail in candidates
        if cocktail.get("name") != previous_name
    ]
    if not selection_pool:
        selection_pool = candidates

    selected = random.choice(selection_pool)
    conversations.update_last_random_cocktail(session_id, selected.get("name"))

    roulette_pool = candidates[:]
    random.shuffle(roulette_pool)
    if selected not in roulette_pool:
        roulette_pool.append(selected)
    roulette_cocktails = roulette_pool[:8]
    if selected not in roulette_cocktails:
        roulette_cocktails[-1:] = [selected]

    selected_public = public_cocktail(selected)
    roulette_public = [public_cocktail(cocktail) for cocktail in roulette_cocktails]
    answer = f"Das Cocktail-Roulette hat entschieden: {selected_public['name']}!"

    return {
        "type": "random",
        "intent": "random",
        "message": "Ich lose dir etwas aus!",
        "answer": answer,
        "cocktails": [selected_public],
        "criteria": None,
        "roulette_cocktails": roulette_public,
        "selected_cocktail": selected_public,
    }


async def build_chat_response(
    user_message: str,
    history: list[ChatMessage] | None = None,
    repository: CocktailRepository | None = None,
    session_id: str | None = None,
    conversations: ConversationService | None = None,
) -> dict[str, Any]:
    repository = repository or CocktailRepository()
    conversations = conversations or conversation_service

    try:
        cocktails = repository.list_all()
    except CocktailRepositoryError as exc:
        logger.error("Cocktail repository failed: %s", exc)
        message = "Die Cocktail-Daten konnten gerade nicht geladen werden."
        return {
            "type": "follow_up",
            "message": message,
            "answer": message,
            "cocktails": [],
            "criteria": None,
        }

    catalog_context = conversations.get_catalog_context(session_id)
    detected_intent: ChatIntent | None = None
    generated_intent_answer = ""
    interpretation: IntentAnalysis | None = None
    if intent_routing_enabled():
        local_intent = detect_intent(user_message, cocktails)
        if (
            catalog_context.referenced_cocktail
            and is_previous_cocktail_detail_query(user_message, cocktails)
        ):
            local_intent = "catalog_query"
        try:
            interpretation = normalize_interpretation(
                await analyze_intent_with_llm(
                    user_message, history, cocktails, catalog_context
                ),
                user_message,
                cocktails,
                catalog_context,
            )
            detected_intent = interpretation.intent
            generated_intent_answer = interpretation.answer
        except (LLMError, InvalidLLMOutputError) as exc:
            logger.warning("LLM intent detection failed, using local fallback: %s", exc)
            detected_intent = local_intent
            if detected_intent in {"conversation", "out_of_scope", "unknown"}:
                try:
                    generated_intent_answer = await generate_intent_answer(
                        detected_intent, user_message, history
                    )
                except (LLMError, InvalidLLMOutputError) as answer_exc:
                    logger.warning("LLM intent answer failed, using static fallback: %s", answer_exc)

        if interpretation and (
            interpretation.action == "clarify" or interpretation.confidence < 0.55
        ):
            return interpretation_clarification_response(interpretation)
        if (
            detected_intent in {"conversation", "unknown"}
            and interpretation
            and interpretation.context_mode == "previous_cocktail"
            and catalog_context.referenced_cocktail
        ):
            referenced = find_cocktails_by_name(
                catalog_context.referenced_cocktail, cocktails
            )[:1]
            try:
                answer = await generate_grounded_catalog_answer(
                    user_message, interpretation, referenced, history
                )
            except (LLMError, InvalidLLMOutputError) as exc:
                logger.warning(
                    "Grounded contextual answer failed, using intent answer: %s", exc
                )
                answer = generated_intent_answer
            if answer:
                return {
                    "type": "message",
                    "intent": detected_intent,
                    "message": answer,
                    "answer": answer,
                    "cocktails": referenced,
                    "criteria": None,
                }
        if detected_intent in {"conversation", "out_of_scope", "unknown"}:
            return basic_intent_response(
                detected_intent, generated_intent_answer, user_message
            )
        if detected_intent == "catalog_query":
            if interpretation:
                matches, criteria = execute_catalog_action(
                    interpretation, cocktails, catalog_context
                )
                try:
                    answer = await generate_grounded_catalog_answer(
                        user_message, interpretation, matches, history
                    )
                except (LLMError, InvalidLLMOutputError) as exc:
                    logger.warning(
                        "Grounded catalog answer failed, using safe fallback: %s", exc
                    )
                    answer = build_catalog_fallback_answer(interpretation, matches)

                response = {
                    "type": "message",
                    "intent": "catalog_query",
                    "message": answer,
                    "answer": answer,
                    "cocktails": matches,
                    "criteria": criteria.model_dump() if criteria else None,
                }
            else:
                response = build_catalog_query_response(
                    user_message, cocktails, catalog_context
                )
                criteria_payload = response.get("criteria")
                criteria = (
                    CocktailSearchCriteria.model_validate(criteria_payload)
                    if criteria_payload else None
                )

            if criteria:
                conversations.record_catalog_search(
                    session_id,
                    criteria,
                    [cocktail["name"] for cocktail in response.get("cocktails", [])],
                    append=bool(
                        interpretation.context_mode == "previous_search"
                        if interpretation else is_more_catalog_request(user_message)
                    ),
                )
            else:
                referenced_name = (
                    interpretation.cocktail_name
                    if interpretation and interpretation.cocktail_name
                    else detect_cocktail_name(user_message, cocktails)
                )
                if not referenced_name and len(response.get("cocktails", [])) == 1:
                    referenced_name = response["cocktails"][0]["name"]
                conversations.set_referenced_cocktail(
                    session_id, referenced_name
                )
            return response

    intent = detected_intent if intent_routing_enabled() else detect_intent(user_message, cocktails)
    if intent == "random":
        return build_random_response(cocktails, session_id, conversations)

    try:
        criteria = await extract_search_criteria(user_message)
        criteria = normalize_search_criteria(criteria, user_message, cocktails)
    except (InvalidLLMOutputError, LLMError) as exc:
        logger.warning("LLM criteria extraction failed, using local fallback: %s", exc)
        criteria = fallback_search_criteria(user_message, cocktails)

    matching_cocktails = select_recommendation_candidates(criteria, cocktails)

    try:
        response = await generate_answer(user_message, matching_cocktails, history)
    except (LLMError, InvalidLLMOutputError) as exc:
        logger.warning("LLM answer generation failed, using local answer: %s", exc)
        answer = build_local_answer(criteria, matching_cocktails)
        response = {"message": answer, "answer": answer, "cocktails": matching_cocktails}

    response["type"] = "recommendation"
    response["intent"] = intent
    response["criteria"] = criteria.model_dump()
    return response
