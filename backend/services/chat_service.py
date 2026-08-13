import json
import logging
import os
import re
from typing import Any

from llm.llm_client import InvalidLLMOutputError, LLMError, extract_json_object, query_ollama
from models.cocktail import ChatIntent, ChatMessage, CocktailPreferences, CocktailSearchCriteria
from repositories.cocktail_repository import CocktailRepository, CocktailRepositoryError
from services.cocktail_service import matches_strength, normalize_text, search_cocktails, term_matches
from services.conversation_service import ConversationService, conversation_service


logger = logging.getLogger("cocktail-rag")


def intent_routing_enabled() -> bool:
    return os.getenv("INTENT_ROUTING_ENABLED", "true").lower() == "true"


def detect_cocktail_name(user_message: str, cocktails: list[dict[str, Any]]) -> str | None:
    matches = [
        cocktail.get("name")
        for cocktail in cocktails
        if isinstance(cocktail.get("name"), str) and term_matches(cocktail["name"], user_message)
    ]
    return max(matches, key=len) if matches else None


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
        "cocktail", "drink", "empfiehl", "empfehl", "such", "find", "vorschlag",
        "ueberrasch", "überrasch", "zufall", "trinken", "lust auf", "etwas anderes",
    }
    if any(marker in normalized for marker in recommendation_markers):
        return "recommendation"

    return "unknown"


def basic_intent_response(
    intent: ChatIntent,
    preferences: CocktailPreferences,
) -> dict[str, Any]:
    messages = {
        "greeting": (
            "Hallo! Ich bin CocktailGPT. Ich kann dir einen Cocktail empfehlen oder Fragen zu "
            "Zutaten, Geschmack, Stärke und Preisen beantworten."
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
    answer = messages[intent]
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
    cocktail_name = detect_cocktail_name(user_message, cocktails)
    cocktail = next((item for item in cocktails if item.get("name") == cocktail_name), None)
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

    answer = f"{cocktail_name}: " + " ".join(detail for detail in details if detail)
    return {
        "type": "message",
        "intent": "cocktail_details",
        "message": answer,
        "answer": answer,
        "cocktails": [cocktail],
        "criteria": None,
        "preferences": preferences.model_dump(),
    }


def public_cocktail(cocktail: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": cocktail.get("name"),
        "kategorie": cocktail.get("kategorie"),
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
    categories = unique_text_values(cocktails, "kategorie")

    spirituose = canonical_value(criteria.spirituose, spirits) or inferred_value_from_message(user_message, spirits)
    geschmack = canonical_value(criteria.geschmack, tastes)
    kategorie = canonical_value(criteria.kategorie, categories)

    if criteria.kategorie and not kategorie:
        geschmack = geschmack or canonical_value(criteria.kategorie, tastes)
    if criteria.geschmack and not geschmack:
        kategorie = kategorie or canonical_value(criteria.geschmack, categories)

    geschmack = geschmack or inferred_value_from_message(user_message, tastes)
    kategorie = kategorie or inferred_value_from_message(user_message, categories)

    staerke = criteria.staerke
    if staerke and not any(matches_strength(staerke, cocktail) for cocktail in cocktails):
        staerke = None

    return CocktailSearchCriteria(
        spirituose=spirituose,
        geschmack=geschmack,
        kategorie=kategorie,
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
        + unique_text_values(cocktails, "kategorie")
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
        criteria.kategorie,
        criteria.staerke,
        criteria.ausschluesse,
    ])


def preference_values(cocktails: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "ingredients": unique_text_values(cocktails, "zutaten"),
        "spirits": unique_text_values(cocktails, "spirituose"),
        "flavors": unique_text_values(cocktails, "geschmack"),
        "categories": unique_text_values(cocktails, "kategorie"),
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
    categories = canonical_values(preferences.categories, values["categories"])

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
        categories=categories,
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
        phrase in normalized_message
        for phrase in ["nicht so stark", "etwas stark", "starkes", "leichtes", "lieber etwas leicht"]
    )

    def is_excluded(value: str) -> bool:
        normalized_value = normalize_text(value)
        return any(term_matches(term, value) or term in normalized_value for term in exclusion_terms)

    for spirit in values["spirits"]:
        if term_matches(spirit, user_message):
            if is_excluded(spirit):
                preferences.spirits = [value for value in preferences.spirits if normalize_text(value) != normalize_text(spirit)]
                preferences.disliked_ingredients.append(spirit)
            else:
                preferences.spirits.append(spirit)

    for flavor in values["flavors"]:
        normalized_flavor = normalize_text(flavor)
        if strength_context and (
            normalized_flavor == "stark"
            or normalized_flavor.startswith("leicht ")
            or normalized_flavor == "leicht"
        ):
            continue
        if term_matches(flavor, user_message):
            if is_excluded(flavor):
                preferences.liked_flavors = [value for value in preferences.liked_flavors if normalize_text(value) != normalize_text(flavor)]
                preferences.disliked_flavors.append(flavor)
            else:
                preferences.liked_flavors.append(flavor)

    for category in values["categories"]:
        if term_matches(category, user_message) and not is_excluded(category):
            preferences.categories.append(category)

    for ingredient in values["ingredients"]:
        if term_matches(ingredient, user_message):
            if is_excluded(ingredient):
                preferences.liked_ingredients = [
                    value for value in preferences.liked_ingredients if normalize_text(value) != normalize_text(ingredient)
                ]
                preferences.disliked_ingredients.append(ingredient)
            elif message_mentions_any(user_message, {"mag", "liebe", "mit", "gerne"}):
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
                "liked_flavors, disliked_flavors, categories, strength, alcoholic. "
                "Listen duerfen keine Duplikate enthalten. Wenn der Nutzer seine Meinung aendert, entferne widerspruechliche "
                "positive Werte. Beispiel: 'doch keinen Gin' entfernt Gin aus spirits und setzt Gin in disliked_ingredients. "
                "strength darf nur null, mild, mittel, stark, hoch oder alkoholfrei sein. "
                "Nutze nur Werte, die zur Cocktailkarte passen.\n\n"
                f"ERLAUBTE_SPIRITUOSEN: {json.dumps(values['spirits'], ensure_ascii=False)}\n"
                f"ERLAUBTE_GESCHMAECKER: {json.dumps(values['flavors'], ensure_ascii=False)}\n"
                f"ERLAUBTE_KATEGORIEN: {json.dumps(values['categories'], ensure_ascii=False)}\n"
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
        kategorie=preferences.categories[0] if preferences.categories else None,
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
        preferences.categories,
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
                "Stelle auf Deutsch eine oder zwei kurze Rueckfragen zu Spirituose, Geschmack, Staerke, Kategorie "
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
                "spirituose, geschmack, kategorie, staerke, ausschluesse. "
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
        "Du darfst keine anderen Cocktails, Zutaten, Preise, Kategorien oder Staerken erfinden. "
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

    mentioned_allowed_names = names_from_answer(payload["answer"], allowed_names)
    if mentioned_allowed_names != requested_names:
        raise InvalidLLMOutputError("LLM-Antwort und cocktail_names stimmen nicht ueberein.")


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
    return {
        "message": payload["answer"],
        "answer": payload["answer"],
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
    detected_intent: ChatIntent | None = None
    if intent_routing_enabled():
        local_preferences = local_update_preferences(current_preferences, user_message, cocktails)
        detected_intent = detect_intent(user_message, current_preferences, local_preferences, cocktails)
        if detected_intent == "reset_preferences":
            conversations.reset_preferences(session_id)
            return basic_intent_response(detected_intent, CocktailPreferences())
        if detected_intent in {"greeting", "out_of_scope", "unknown"}:
            return basic_intent_response(detected_intent, current_preferences)
        if detected_intent == "cocktail_details":
            return build_cocktail_detail_response(user_message, cocktails, current_preferences)

    try:
        preferences = await update_preferences_with_llm(user_message, current_preferences, cocktails)
    except InvalidLLMOutputError as exc:
        logger.warning("LLM criteria validation failed, using local fallback: %s", exc)
        preferences = local_update_preferences(current_preferences, user_message, cocktails)
    except LLMError as exc:
        logger.warning("LLM criteria extraction failed, using local fallback: %s", exc)
        preferences = local_update_preferences(current_preferences, user_message, cocktails)

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
