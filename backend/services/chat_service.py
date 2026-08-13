import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any

from llm.llm_client import InvalidLLMOutputError, LLMError, extract_json_object, query_ollama
from models.cocktail import ChatIntent, ChatMessage, CocktailPreferences, CocktailSearchCriteria, IntentAnalysis
from repositories.cocktail_repository import CocktailRepository, CocktailRepositoryError
from services.cocktail_service import matches_strength, normalize_text, search_cocktails, term_matches
from services.conversation_service import ConversationService, conversation_service


logger = logging.getLogger("cocktail-rag")


def intent_routing_enabled() -> bool:
    return os.getenv("INTENT_ROUTING_ENABLED", "true").lower() == "true"


def detect_cocktail_name(user_message: str, cocktails: list[dict[str, Any]]) -> str | None:
    matches = find_cocktails_by_name(user_message, cocktails)
    return matches[0].get("name") if matches else None


def singular_token(token: str) -> str:
    return token[:-1] if len(token) > 4 and token.endswith("s") else token


def find_cocktails_by_name(
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    message_tokens = {singular_token(token) for token in normalize_text(user_message).split()}
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for cocktail in cocktails:
        name = cocktail.get("name")
        if not isinstance(name, str):
            continue
        name_tokens = [singular_token(token) for token in normalize_text(name).split()]
        overlap = sum(token in message_tokens for token in name_tokens)
        if not overlap:
            continue
        missing_name_tokens = len(name_tokens) - overlap
        ranked.append((missing_name_tokens, -overlap, cocktail))

    ranked.sort(key=lambda item: (item[0], item[1], len(item[2].get("name", ""))))
    return [item[2] for item in ranked]


def is_availability_question(user_message: str) -> bool:
    normalized = normalize_text(user_message)
    return any(phrase in normalized for phrase in {
        "habt ihr", "hast du", "gibt es", "gibts", "auf der karte", "im angebot",
    })


def is_catalog_query(user_message: str) -> bool:
    normalized = normalize_text(user_message)
    return is_availability_question(user_message) or any(phrase in normalized for phrase in {
        "welche cocktails enthalten", "welche drinks enthalten", "welche cocktails mit",
        "welche drinks mit", "welche cocktails sind", "welche drinks sind",
    })


def detect_intent(
    user_message: str,
    current_preferences: CocktailPreferences,
    updated_preferences: CocktailPreferences,
    cocktails: list[dict[str, Any]],
) -> ChatIntent:
    normalized = normalize_text(user_message).strip()
    words = set(normalized.split())

    if any(phrase in normalized for phrase in {
        "von vorne", "alles vergessen", "praeferenzen loeschen", "präferenzen löschen",
        "auswahl loeschen", "auswahl löschen", "zuruecksetzen", "zurücksetzen", "neu anfangen",
    }):
        return "reset_preferences"

    greeting_words = {"hallo", "hi", "hey", "moin", "servus", "danke", "tschuss", "tschuess"}
    if words and words <= greeting_words:
        return "greeting"

    short_conversation_replies = {"ne", "nee", "nein", "noe", "ja", "jo", "okay", "ok"}
    if normalized in short_conversation_replies:
        return "conversation"

    if is_catalog_query(user_message):
        return "catalog_query"

    cocktail_name = detect_cocktail_name(user_message, cocktails)
    detail_markers = {
        "preis", "kostet", "kosten", "zutat", "zutaten", "drin", "enthalten",
        "rezept", "staerke", "stärke", "stark", "beschreibung", "spirituose",
    }
    detail_questions = {
        "was kostet", "wie teuer", "was ist in", "welche zutaten", "wie stark ist",
        "informationen zu", "erzaehl mir etwas ueber", "erzähl mir etwas über",
    }
    if (
        cocktail_name and any(marker in normalized for marker in detail_markers)
    ) or any(question in normalized for question in detail_questions):
        return "cocktail_details"

    out_of_scope_markers = {
        "wetter", "fussball", "fußball", "aktien", "programmieren", "python", "politik",
        "pizza", "hotel", "flug", "nachrichten", "hausaufgabe",
    }
    if any(marker in normalized for marker in out_of_scope_markers):
        return "out_of_scope"

    if updated_preferences != current_preferences:
        return "preference_update"

    recommendation_markers = {
        "empfiehl", "empfehl", "such", "find", "zeig", "welcher cocktail", "welchen cocktail",
        "vorschlag", "ueberrasch", "überrasch", "zufall", "lust auf", "etwas anderes",
        "ich moechte", "ich möchte", "ich haette gern", "ich hätte gern",
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
) -> IntentAnalysis:
    cocktail_names = [cocktail.get("name") for cocktail in cocktails if cocktail.get("name")]
    messages: list[dict[str, str]] = [{
        "role": "system",
        "content": (
            "Du erkennst die kommunikative Absicht einer Nachricht an einen Cocktail-Assistenten. "
            "Gib ausschliesslich JSON mit intent und answer zurueck. Erlaubte Intents: greeting, conversation, "
            "recommendation, preference_update, catalog_query, cocktail_details, reset_preferences, out_of_scope, unknown. "
            "recommendation gilt nur, wenn der Nutzer ausdruecklich eine Empfehlung, Suche oder Auswahl verlangt. "
            "catalog_query prueft nur Fakten und Verfuegbarkeit in der Cocktailkarte, ohne etwas zu empfehlen. "
            "Beispiele fuer catalog_query: 'Habt ihr Mojitos?', 'Gibt es alkoholfreie Cocktails?' und "
            "'Welche Cocktails enthalten Rum?'. "
            "Eine allgemeine Aussage wie 'Ich mag Cocktails' ist conversation, keine recommendation. "
            "Eine Antwort auf eine vorherige Smalltalk-Frage wie 'gut und dir?' ist conversation, nicht greeting. "
            "Eine konkrete Vorliebe wie 'Ich mag Gin' oder 'stark und cremig' ist preference_update. "
            "Fragen zu Preis, Zutaten, Geschmack oder Staerke eines konkreten Cocktails sind cocktail_details. "
            "Bei greeting, conversation, reset_preferences, out_of_scope und unknown schreibst du in answer eine "
            "kurze, grammatikalisch natuerliche deutsche Antwort. greeting begruesst kurz und fragt direkt nach "
            "Cocktailwuenschen oder Geschmack, niemals nach dem persoenlichen Befinden. Bei conversation reagierst "
            "du auf den bisherigen Verlauf und leitest mit einer passenden Rueckfrage zum Cocktail-Thema zurueck. "
            "answer muss dabei ein Fragezeichen enthalten. Wiederhole keine vorherige Assistentenantwort. Bei "
            "out_of_scope erklaerst du freundlich deine Rolle. Duze den Nutzer immer, reagiere direkt auf den Inhalt "
            "und vermeide unpassende Floskeln oder Wuensche wie 'Viel Spass'. "
            "Bei recommendation, preference_update, catalog_query und cocktail_details bleibt answer leer. "
            "Erfinde keine Cocktaildaten.\n\n"
            f"COCKTAILNAMEN_DER_KARTE: {json.dumps(cocktail_names, ensure_ascii=False)}"
        ),
    }]
    if history:
        for message in history[-4:]:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": user_message})

    parsed = extract_json_object(await query_ollama(messages))
    try:
        analysis = IntentAnalysis.model_validate(parsed)
    except Exception as exc:
        raise InvalidLLMOutputError("LLM-Intent konnte nicht validiert werden.") from exc
    if not intent_answer_is_usable(analysis, user_message, history):
        raise InvalidLLMOutputError("LLM-Intent-Antwort war unvollstaendig.")
    return analysis


def intent_answer_is_usable(
    analysis: IntentAnalysis,
    user_message: str = "",
    history: list[ChatMessage] | None = None,
) -> bool:
    answer_intents = {"greeting", "conversation", "reset_preferences", "out_of_scope", "unknown"}
    if analysis.intent not in answer_intents:
        return True
    if not analysis.answer.strip():
        return False

    normalized_user = normalize_text(user_message)
    if analysis.intent == "greeting":
        greeting_words = {"hallo", "hi", "hey", "moin", "servus", "guten", "morgen", "abend", "tag"}
        if normalized_user and not set(normalized_user.split()) <= greeting_words:
            return False
        normalized_answer = normalize_text(analysis.answer)
        if any(phrase in normalized_answer for phrase in {"wie geht es dir", "wie gehts dir", "wie geht es ihnen"}):
            return False

    if analysis.intent == "conversation" and "?" not in analysis.answer:
        return False

    if analysis.intent in {"greeting", "conversation"} and contains_recommendation_language(
        analysis.answer
    ):
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
            "Bei greeting oder conversation leitest du freundlich zu Cocktailwuenschen, Geschmacksrichtungen oder "
            "Zutaten ueber und stellst genau eine passende Frage. Frage nicht nach dem persoenlichen Befinden. "
            "Nenne bei greeting oder conversation keinen konkreten Cocktail. Konkrete Empfehlungen werden an anderer "
            "Stelle aus der geprueften Cocktailkarte erzeugt. "
            "Wenn bereits ein Verlauf existiert, begruesse nicht erneut. Du hilfst dem Nutzer bei der Auswahl und "
            "forderst den Nutzer niemals auf, dir bei der Auswahl zu helfen. "
            "Bei out_of_scope erklaerst du knapp, wobei ein Cocktail-Assistent helfen kann. Bei unknown fragst du "
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
    if intent in {"greeting", "conversation"} and contains_recommendation_language(answer):
        raise InvalidLLMOutputError("LLM-Gespraechsantwort enthielt eine ungepruefte Empfehlung.")
    return answer


def basic_intent_response(
    intent: ChatIntent,
    preferences: CocktailPreferences,
    generated_answer: str = "",
) -> dict[str, Any]:
    messages = {
        "greeting": (
            "Hallo! Ich bin CocktailGPT. Ich kann dir einen Cocktail empfehlen oder Fragen zu "
            "Zutaten, Geschmack, Stärke und Preisen beantworten."
        ),
        "conversation": (
            "Cocktails sind wirklich vielseitig. Magst du sie eher fruchtig, cremig, sauer oder stark?"
        ),
        "reset_preferences": (
            "Ich habe deine bisherigen Vorlieben zurückgesetzt. Wir können mit einer neuen Auswahl starten: "
            "Magst du es eher fruchtig, sauer, cremig oder stark?"
        ),
        "out_of_scope": (
            "Dafür bin ich nicht zuständig. Ich bin dein Cocktail-Assistent und helfe dir gern bei "
            "Empfehlungen, Zutaten, Geschmack, Stärke oder Preisen."
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
        "preferences": preferences.model_dump(),
    }


def build_cocktail_detail_response(
    user_message: str,
    cocktails: list[dict[str, Any]],
    preferences: CocktailPreferences,
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
            "intent": "cocktail_details",
            "message": answer,
            "answer": answer,
            "cocktails": [],
            "criteria": None,
            "preferences": preferences.model_dump(),
        }

    normalized = normalize_text(user_message)
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
        "intent": "cocktail_details",
        "message": answer,
        "answer": answer,
        "cocktails": [cocktail],
        "criteria": None,
        "preferences": preferences.model_dump(),
    }


def build_catalog_query_response(
    user_message: str,
    cocktails: list[dict[str, Any]],
    preferences: CocktailPreferences,
) -> dict[str, Any]:
    name_matches = find_cocktails_by_name(user_message, cocktails)
    if name_matches:
        base_tokens = {
            singular_token(token) for token in normalize_text(name_matches[0].get("name", "")).split()
        }
        matches = [
            item for item in name_matches
            if base_tokens.intersection(
                singular_token(token) for token in normalize_text(item.get("name", "")).split()
            )
        ]
    else:
        criteria = fallback_search_criteria(user_message, cocktails)
        if message_mentions_any(user_message, {"alkoholfrei", "ohne alkohol"}):
            criteria = criteria.model_copy(update={"staerke": "alkoholfrei"})
        matches = search_cocktails(criteria, cocktails, limit=10) if has_search_criteria(criteria) else []

    if matches:
        names = [item["name"] for item in matches]
        answer = "Auf unserer Karte gibt es: " + ", ".join(names) + "."
    else:
        answer = "Dazu habe ich auf unserer Cocktailkarte keinen passenden Eintrag gefunden."

    return {
        "type": "message",
        "intent": "catalog_query",
        "message": answer,
        "answer": answer,
        "cocktails": matches,
        "criteria": None,
        "preferences": preferences.model_dump(),
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


def preference_values(cocktails: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "ingredients": unique_text_values(cocktails, "zutaten"),
        "spirits": unique_text_values(cocktails, "spirituose"),
        "flavors": unique_text_values(cocktails, "geschmack"),
    }


def canonical_values(values: list[str], allowed_values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        canonical = canonical_value(value, allowed_values)
        if canonical and canonical.lower() not in seen:
            seen.add(canonical.lower())
            result.append(canonical)
    return result


def remove_overlaps(positive: list[str], negative: list[str]) -> list[str]:
    negative_normalized = {normalize_text(value) for value in negative}
    return [value for value in positive if normalize_text(value) not in negative_normalized]


def constrain_preferences_to_local_signal(
    current_preferences: CocktailPreferences,
    llm_preferences: CocktailPreferences,
    local_preferences: CocktailPreferences,
) -> CocktailPreferences:
    updates: dict[str, Any] = {}
    for field in [
        "liked_ingredients",
        "disliked_ingredients",
        "spirits",
        "liked_flavors",
        "disliked_flavors",
        "strength",
        "alcoholic",
    ]:
        if getattr(local_preferences, field) != getattr(current_preferences, field):
            updates[field] = getattr(local_preferences, field)
        else:
            updates[field] = getattr(current_preferences, field)

    # Keep future optional fields, such as price preference, if the model defines them.
    if hasattr(current_preferences, "price_preference"):
        field = "price_preference"
        if getattr(local_preferences, field) != getattr(current_preferences, field):
            updates[field] = getattr(local_preferences, field)
        else:
            updates[field] = getattr(current_preferences, field)

    return llm_preferences.model_copy(update=updates)


def canonicalize_preferences(
    preferences: CocktailPreferences,
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> CocktailPreferences:
    values = preference_values(cocktails)
    liked_ingredients = canonical_values(preferences.liked_ingredients, values["ingredients"])
    disliked_ingredients = canonical_values(preferences.disliked_ingredients, values["ingredients"] + values["spirits"])
    spirits = canonical_values(preferences.spirits, values["spirits"])
    liked_flavors = canonical_values(preferences.liked_flavors, values["flavors"])
    disliked_flavors = canonical_values(preferences.disliked_flavors, values["flavors"])

    disliked_ingredients.extend(
        item for item in infer_exclusions_from_message(user_message, cocktails)
        if normalize_text(item) not in {normalize_text(value) for value in disliked_ingredients}
    )

    spirits = remove_overlaps(spirits, disliked_ingredients)
    liked_ingredients = remove_overlaps(liked_ingredients, disliked_ingredients)
    liked_flavors = remove_overlaps(liked_flavors, disliked_flavors)

    return CocktailPreferences(
        liked_ingredients=liked_ingredients,
        disliked_ingredients=disliked_ingredients,
        spirits=spirits,
        liked_flavors=liked_flavors,
        disliked_flavors=disliked_flavors,
        strength=preferences.strength,
        alcoholic=preferences.alcoholic,
    )


def find_terms_after_exclusion_marker(user_message: str) -> set[str]:
    words = normalize_text(user_message).split()
    markers = {"ohne", "kein", "keine", "keinen", "nichts", "nicht"}
    stop_words = {"und", "oder", "mit", "aber", "doch", "lieber"}
    terms: set[str] = set()
    for index, word in enumerate(words):
        if word in markers:
            for term in words[index + 1:]:
                if term in stop_words:
                    break
                terms.add(term)
    return terms


def message_mentions_any(user_message: str, needles: set[str]) -> bool:
    normalized = normalize_text(user_message)
    return any(needle in normalized for needle in needles)


def has_positive_preference_marker(user_message: str) -> bool:
    return message_mentions_any(user_message, {
        "ich mag", "ich liebe", "ich haette gerne", "ich hätte gerne", "gerne", "mit",
        "basis", "auf basis", "bevorzuge", "lust auf",
    })


def explicitly_mentions_value(user_message: str, value: str) -> bool:
    normalized_message = normalize_text(user_message)
    normalized_value = normalize_text(value)
    if not normalized_value:
        return False
    return bool(re.search(rf"\b{re.escape(normalized_value)}\b", normalized_message))


def local_update_preferences(
    current_preferences: CocktailPreferences,
    user_message: str,
    cocktails: list[dict[str, Any]],
) -> CocktailPreferences:
    preferences = current_preferences.model_copy(deep=True)
    values = preference_values(cocktails)
    exclusion_terms = find_terms_after_exclusion_marker(user_message)
    normalized_message = normalize_text(user_message)
    strength_context = any(
        word.startswith(("stark", "leicht", "mild", "kraeftig", "kraftig"))
        for word in normalized_message.split()
    )

    def is_excluded(value: str) -> bool:
        normalized_value = normalize_text(value)
        return any(term_matches(term, value) or term in normalized_value for term in exclusion_terms)

    for spirit in values["spirits"]:
        if explicitly_mentions_value(user_message, spirit):
            if is_excluded(spirit):
                preferences.spirits = [value for value in preferences.spirits if normalize_text(value) != normalize_text(spirit)]
                preferences.disliked_ingredients.append(spirit)
            elif has_positive_preference_marker(user_message):
                preferences.spirits.append(spirit)

    for flavor in values["flavors"]:
        normalized_flavor = normalize_text(flavor)
        if strength_context and (
            normalized_flavor == "stark"
            or normalized_flavor.startswith("leicht ")
            or normalized_flavor == "leicht"
        ):
            continue
        if explicitly_mentions_value(user_message, flavor) or term_matches(flavor, user_message):
            if is_excluded(flavor):
                preferences.liked_flavors = [value for value in preferences.liked_flavors if normalize_text(value) != normalize_text(flavor)]
                preferences.disliked_flavors.append(flavor)
            else:
                preferences.liked_flavors.append(flavor)

    for ingredient in values["ingredients"]:
        if explicitly_mentions_value(user_message, ingredient):
            if is_excluded(ingredient):
                preferences.liked_ingredients = [
                    value for value in preferences.liked_ingredients if normalize_text(value) != normalize_text(ingredient)
                ]
                preferences.disliked_ingredients.append(ingredient)
            elif has_positive_preference_marker(user_message):
                preferences.liked_ingredients.append(ingredient)

    if message_mentions_any(user_message, {"nicht so stark", "leicht", "leichtes", "mild", "milder"}):
        preferences.strength = "mild"
    elif message_mentions_any(user_message, {"stark", "starkes", "kraeftig", "kraftig"}):
        preferences.strength = "stark"
    elif "mittel" in normalize_text(user_message):
        preferences.strength = "mittel"

    if message_mentions_any(user_message, {"alkoholfrei", "ohne alkohol"}):
        preferences.alcoholic = False

    return canonicalize_preferences(preferences, user_message, cocktails)


async def update_preferences_with_llm(
    user_message: str,
    current_preferences: CocktailPreferences,
    cocktails: list[dict[str, Any]],
) -> CocktailPreferences:
    values = preference_values(cocktails)
    messages = [
        {
            "role": "system",
            "content": (
                "Du aktualisierst Cocktail-Praeferenzen fuer genau eine Chat-Session. "
                "Nutze die bisherigen Praeferenzen und die neue Nutzernachricht und gib den neuen vollstaendigen "
                "Zustand als JSON-Objekt zurueck. Erlaubte Felder: liked_ingredients, disliked_ingredients, spirits, "
                "liked_flavors, disliked_flavors, strength, alcoholic. "
                "Uebernimm nur Praeferenzen, die der Nutzer in der neuen Nachricht ausdruecklich nennt. "
                "Leite keine Zutaten, Spirituosen, Staerke oder Alkoholstatus aus passenden Cocktails ab. "
                "Beispiel: 'Ich mag fruchtige Cocktails' setzt nur liked_flavors ['fruchtig'] und sonst nichts Neues. "
                "Listen duerfen keine Duplikate enthalten. Wenn der Nutzer seine Meinung aendert, entferne widerspruechliche "
                "positive Werte. Beispiel: 'doch keinen Gin' entfernt Gin aus spirits und setzt Gin in disliked_ingredients. "
                "strength darf nur null, mild, mittel, stark, hoch oder alkoholfrei sein. "
                "Nutze nur Werte, die zur Cocktailkarte passen.\n\n"
                f"ERLAUBTE_SPIRITUOSEN: {json.dumps(values['spirits'], ensure_ascii=False)}\n"
                f"ERLAUBTE_GESCHMAECKER: {json.dumps(values['flavors'], ensure_ascii=False)}\n"
                f"ERLAUBTE_ZUTATEN: {json.dumps(values['ingredients'], ensure_ascii=False)}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"BISHERIGE_PRAEFERENZEN:\n{current_preferences.model_dump_json()}\n\n"
                f"NEUE_NACHRICHT:\n{user_message}"
            ),
        },
    ]
    raw_response = await query_ollama(messages)
    parsed = extract_json_object(raw_response)
    try:
        preferences = CocktailPreferences.model_validate(parsed)
    except Exception as exc:
        raise InvalidLLMOutputError("LLM-Praeferenzen konnten nicht validiert werden.") from exc
    return canonicalize_preferences(preferences, user_message, cocktails)


def preferences_to_search_criteria(preferences: CocktailPreferences) -> CocktailSearchCriteria:
    strength = preferences.strength
    criteria_strength = "mittel" if strength == "mild" else strength
    if preferences.alcoholic is False:
        criteria_strength = "alkoholfrei"

    return CocktailSearchCriteria(
        spirituose=preferences.spirits[0] if preferences.spirits else None,
        geschmack=preferences.liked_flavors[0] if preferences.liked_flavors else None,
        staerke=criteria_strength,
        ausschluesse=preferences.disliked_ingredients + preferences.disliked_flavors,
    )


def should_ask_follow_up(preferences: CocktailPreferences) -> bool:
    return not any([
        preferences.liked_ingredients,
        preferences.disliked_ingredients,
        preferences.spirits,
        preferences.liked_flavors,
        preferences.disliked_flavors,
        preferences.strength,
        preferences.alcoholic is not None,
    ])


def build_local_follow_up() -> str:
    return (
        "Gerne. Magst du es eher fruchtig, sauer, cremig oder stark? "
        "Oder gibt es eine Spirituose, die du besonders magst oder vermeiden moechtest?"
    )


async def generate_follow_up(
    user_message: str,
    preferences: CocktailPreferences,
    history: list[ChatMessage] | None = None,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist CocktailGPT. Es liegen noch zu wenige Cocktail-Praeferenzen fuer eine gute Empfehlung vor. "
                "Stelle auf Deutsch eine oder zwei kurze Rueckfragen zu Spirituose, Geschmack, Staerke "
                "oder Zutaten. Frage nicht nach irrelevanten Dingen."
            ),
        }
    ]
    if history:
        for message in history[-4:]:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
    messages.append({
        "role": "user",
        "content": (
            f"Praeferenzen: {preferences.model_dump_json()}\n"
            f"Aktuelle Nachricht: {user_message}"
        ),
    })
    try:
        return (await query_ollama(messages, response_format="")).strip()
    except LLMError as exc:
        logger.warning("LLM follow-up generation failed, using local follow-up: %s", exc)
        return build_local_follow_up()


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
                "Setze unbekannte positive Felder auf null. ausschluesse ist immer eine Liste. "
                "staerke darf nur null, leicht, mittel, stark, hoch oder alkoholfrei sein. "
                "Beispiele: 'starker Cocktail' -> staerke 'stark'; 'ohne Kokos' -> ausschluesse ['Kokos']; "
                "'mit Rum' -> spirituose 'Rum'."
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
            "preferences": CocktailPreferences().model_dump(),
        }

    current_preferences = conversations.get_preferences(session_id)
    local_preferences = local_update_preferences(current_preferences, user_message, cocktails)
    detected_intent: ChatIntent | None = None
    generated_intent_answer = ""
    if intent_routing_enabled():
        local_intent = detect_intent(user_message, current_preferences, local_preferences, cocktails)
        normalized_message = normalize_text(user_message)
        is_short_conversation_reply = normalized_message in {
            "ne", "nee", "nein", "noe", "ja", "jo", "okay", "ok",
        }
        is_deterministic_catalog_query = local_intent == "catalog_query"
        try:
            if is_deterministic_catalog_query:
                detected_intent = local_intent
            elif is_short_conversation_reply:
                detected_intent = local_intent
                generated_intent_answer = await generate_intent_answer(
                    detected_intent, user_message, history
                )
            else:
                analysis = await analyze_intent_with_llm(user_message, history, cocktails)
                detected_intent = analysis.intent
                generated_intent_answer = analysis.answer
        except (LLMError, InvalidLLMOutputError) as exc:
            logger.warning("LLM intent detection failed, using local fallback: %s", exc)
            detected_intent = local_intent
            if detected_intent in {"greeting", "conversation", "out_of_scope", "unknown"}:
                try:
                    generated_intent_answer = await generate_intent_answer(
                        detected_intent, user_message, history
                    )
                except (LLMError, InvalidLLMOutputError) as answer_exc:
                    logger.warning("LLM intent answer failed, using static fallback: %s", answer_exc)

        if detected_intent == "reset_preferences":
            conversations.reset_preferences(session_id)
            return basic_intent_response(
                detected_intent, CocktailPreferences(), generated_intent_answer
            )
        if detected_intent in {"greeting", "conversation", "out_of_scope", "unknown"}:
            return basic_intent_response(detected_intent, current_preferences, generated_intent_answer)
        if detected_intent == "catalog_query":
            return build_catalog_query_response(user_message, cocktails, current_preferences)
        if detected_intent == "cocktail_details":
            return build_cocktail_detail_response(user_message, cocktails, current_preferences)

    if intent_routing_enabled():
        preferences = local_preferences
    else:
        try:
            preferences = await update_preferences_with_llm(user_message, current_preferences, cocktails)
        except InvalidLLMOutputError as exc:
            logger.warning("LLM criteria validation failed, using local fallback: %s", exc)
            preferences = local_preferences
        except LLMError as exc:
            logger.warning("LLM criteria extraction failed, using local fallback: %s", exc)
            preferences = local_preferences

    intent: ChatIntent | None = None
    if intent_routing_enabled():
        intent = detected_intent

    conversations.update_preferences(session_id, preferences)

    if should_ask_follow_up(preferences):
        answer = await generate_follow_up(user_message, preferences, history)
        return {
            "type": "follow_up",
            "intent": intent,
            "message": answer,
            "answer": answer,
            "cocktails": [],
            "criteria": None,
            "preferences": preferences.model_dump(),
        }

    criteria = preferences_to_search_criteria(preferences)
    criteria = normalize_search_criteria(criteria, "", cocktails)
    matching_cocktails = search_cocktails(criteria, cocktails, limit=3)

    try:
        response = await generate_answer(user_message, matching_cocktails, history)
    except (LLMError, InvalidLLMOutputError) as exc:
        logger.warning("LLM answer generation failed, using local answer: %s", exc)
        answer = build_local_answer(criteria, matching_cocktails)
        response = {"message": answer, "answer": answer, "cocktails": matching_cocktails}

    response["type"] = "recommendation"
    response["intent"] = intent
    response["preferences"] = preferences.model_dump()
    response["criteria"] = criteria.model_dump()
    return response


async def build_session_recommendation_response(
    session_id: str | None,
    repository: CocktailRepository | None = None,
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
            "preferences": CocktailPreferences().model_dump(),
        }

    preferences = conversations.get_preferences(session_id)
    if should_ask_follow_up(preferences):
        return {
            "type": "follow_up",
            "message": "",
            "answer": "",
            "cocktails": [],
            "criteria": None,
            "preferences": preferences.model_dump(),
        }

    criteria = normalize_search_criteria(preferences_to_search_criteria(preferences), "", cocktails)
    matching_cocktails = search_cocktails(criteria, cocktails, limit=3)
    return {
        "type": "recommendation",
        "message": "",
        "answer": "",
        "cocktails": matching_cocktails,
        "criteria": criteria.model_dump(),
        "preferences": preferences.model_dump(),
    }
