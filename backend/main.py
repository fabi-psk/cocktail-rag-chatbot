import os
import json
import httpx
import re
import unicodedata
import logging
import random
from difflib import SequenceMatcher
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b")
USE_LLM_ANSWER = os.getenv("USE_LLM_ANSWER", "true").lower() == "true"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "cocktails.json"
logger = logging.getLogger("cocktail-rag")

app = FastAPI(
    title="Cocktail RAG API",
    description="Backend fÃ¼r den Cocktail-RAG-Chatbot mit Ollama LLM-Integration",
    version="2.0"
)

# CORS-Konfiguration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # FÃ¼r einfachere lokale Entwicklung
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cocktaildaten laden
try:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        cocktails = json.load(file)
except Exception as e:
    logger.exception("Fehler beim Laden von cocktails.json: %s", e)
    cocktails = []

# Schemas fÃ¼r POST Chat Request
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)

UMLAUT_REPLACEMENTS = str.maketrans({
    "\u00e4": "ae",
    "\u00f6": "oe",
    "\u00fc": "ue",
    "\u00df": "ss",
})

TYPO_ALIASES = {
    "alkholfrei": "alkoholfrei",
    "alkolfrei": "alkoholfrei",
    "alkoholfre": "alkoholfrei",
    "fruchtg": "fruchtig",
    "fruchtich": "fruchtig",
    "cremig": "cremig",
    "kremig": "cremig",
    "suer": "sauer",
    "sauerr": "sauer",
    "suess": "suess",
    "suss": "suess",
    "wodka": "vodka",
    "vodcka": "vodka",
    "votka": "vodka",
    "tequilla": "tequila",
    "teqila": "tequila",
    "kokuss": "kokos",
    "cocos": "kokos",
    "rum": "rum",
    "rhum": "rum",
}

NEGATION_WORDS = {"kein", "keine", "keinen", "keinem", "ohne", "nicht"}
ALCOHOL_TERMS = {"rum", "gin", "vodka", "tequila", "peachtree", "triple sec", "alkohol"}
CATEGORY_KEYWORDS = {
    "klassisch": "Klassisch", "klassische": "Klassisch", "klassischen": "Klassisch",
    "cremig": "Cremig", "cremige": "Cremig", "cremigen": "Cremig",
    "fruchtig": "Fruchtig", "fruchtige": "Fruchtig", "fruchtigen": "Fruchtig",
    "stark": "Stark", "starke": "Stark", "starken": "Stark",
    "caipis": "Caipis", "caipi": "Caipis",
    "alkoholfrei": "Alkoholfrei", "alkoholfreie": "Alkoholfrei",
    "alkoholfreien": "Alkoholfrei", "alkoholfreier": "Alkoholfrei",
}
STOP_WORDS = {
    "ich", "mag", "liebe", "suche", "moechte", "mochte", "mÃ¶chte", "will", "haette", "hÃ¤tte",
    "kein", "ohne", "keine", "keinen", "keinem", "nicht",
    "und", "oder", "ein", "einen", "eine", "eines", "einem", "fuer", "fÃ¼r", "mit",
    "was", "gibt", "es", "cocktail", "cocktails", "drink", "drinks",
    "rezept", "rezepte", "zeige", "zeig", "schlage", "schlag", "hast", "habt",
    "du", "ihr", "etwas", "jemand", "kannst", "mir", "bitte", "empfehlen",
    "empfiehl", "machen", "zubereiten", "finden",
    "der", "die", "das", "den", "dem", "des", "in", "im", "an", "am",
    "um", "zu", "zum", "zur", "von", "vom", "aus", "auf", "bei", "als",
    "wie", "nach", "gern", "gerne", "haben", "waere", "wÃ¤re", "sein",
    "ist", "sind", "nur", "noch", "mehr", "sehr", "gut", "mal", "aber", "auch",
}

def normalize_text(value: str) -> str:
    normalized = value.lower().translate(UMLAUT_REPLACEMENTS)
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()

def tokenize_query(value: str) -> list:
    return [
        TYPO_ALIASES.get(token, token)
        for token in normalize_text(value).split()
        if token
    ]

def fuzzy_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()

def fuzzy_token_match(term: str, text: str) -> bool:
    if len(term) < 4:
        return False

    tokens = text.split()
    for token in tokens:
        if len(token) < 4:
            continue
        if fuzzy_similarity(term, token) >= 0.82:
            return True

    # Handle two-word ingredients such as "triple sec" or "blue curacao".
    for idx in range(len(tokens) - 1):
        combined = f"{tokens[idx]} {tokens[idx + 1]}"
        if fuzzy_similarity(term, combined) >= 0.86:
            return True

    return False

def parse_query_exclusions(query: str, words: Optional[list] = None) -> list:
    query_lower = normalize_text(query)
    words = words or tokenize_query(query)
    exclusions = []

    for i, word in enumerate(words):
        if word in NEGATION_WORDS and i + 1 < len(words):
            exclusions.append(words[i + 1])

    if "alkoholfrei" in query_lower or "ohne alkohol" in query_lower:
        exclusions.extend(ALCOHOL_TERMS)
    if "keine sahne" in query_lower or "ohne sahne" in query_lower:
        exclusions.append("sahne")
    if "kein kokos" in query_lower or "ohne kokos" in query_lower:
        exclusions.append("kokos")

    return list(dict.fromkeys(exclusions))

def term_matches(term: str, text: str) -> bool:
    """
    ÃœberprÃ¼ft, ob ein Suchbegriff in einem Text vorkommt.
    Bei kurzen Begriffen (<= 3 Zeichen wie 'gin', 'rum', 'eis') wird Regex mit Wortgrenzen (\b) verwendet,
    um Fehlalarm-Matches zu vermeiden (z. B. 'gin' in 'ginger ale' oder 'in' in 'gin').
    Bei lÃ¤ngeren Begriffen (> 3 Zeichen) erlauben wir bidirektionales Matching (z. B. 'fruchtigen' matchet 'fruchtig').
    ZusÃ¤tzlich wird ein einfaches deutsches Stemming angewendet, um Einzahl/Mehrzahl auszugleichen (z. B. 'erdbeeren' -> 'erdbeere').
    """
    term_lower = TYPO_ALIASES.get(normalize_text(term), normalize_text(term))
    text_lower = normalize_text(text)
    if not term_lower or not text_lower:
        return False
    
    # Standard-Vergleich
    if len(term_lower) <= 3:
        pattern = rf"\b{re.escape(term_lower)}\b"
        if bool(re.search(pattern, text_lower)):
            return True
    else:
        if (term_lower in text_lower) or (text_lower in term_lower):
            return True
        if fuzzy_token_match(term_lower, text_lower):
            return True
            
    # Stammform-Reduktion fÃ¼r deutsche Pluralendungen (-en, -n, -s)
    if len(term_lower) > 4:
        for suffix in ["en", "n", "s"]:
            if term_lower.endswith(suffix):
                stem = term_lower[:-len(suffix)]
                # Der Stamm muss lang genug sein, um Fehlmatches zu verhindern (z. B. 'rum' -> 'ru' vermeiden)
                if len(stem) > 3 and ((stem in text_lower) or (text_lower in stem)):
                    return True
                    
    return False

def smart_retrieve_cocktails(query: str, cocktails_list: list, limit: int = 3) -> list:
    """
    Sucht und bewertet Cocktails basierend auf einer Freitext-Anfrage.
    UnterstÃ¼tzt auch explizite AusschlÃ¼sse wie 'kein kokos', 'ohne alkohol'.
    """
    query_lower = normalize_text(query)
    
    # 0. Spezialfall: ZufÃ¤llige Auswahl
    if "zuf" in query_lower or "random" in query_lower:
        # WÃ¤hle 1 zufÃ¤lligen Cocktail aus
        random_cocktails = random.sample(cocktails_list, min(1, len(cocktails_list)))
        result = []
        for c in random_cocktails:
            c_copy = c.copy()
            c_copy["match_score"] = 99
            c_copy["match_details"] = ["Zufallsauswahl (Ãœberraschung! ðŸŽ²)"]
            result.append(c_copy)
        return result
    
    words = tokenize_query(query)
    exclusions = parse_query_exclusions(query, words)
        
    # 2. Kategorie-Filterung
    # Falls eine bestimmte Kategorie explizit gesucht wird, filtern wir die Liste vorab.
    # Dies verhindert, dass z.B. "Mojito" (Kategorie Caipis) im Rezeptkatalog erscheint,
    # nur weil in seiner Beschreibung das Wort "klassischer" vorkommt.
    target_category = None
    for idx, word in enumerate(words):
        if idx > 0 and words[idx - 1] in NEGATION_WORDS:
            continue
        for keyword, category_name in CATEGORY_KEYWORDS.items():
            if term_matches(word, keyword):
                target_category = category_name
                break
        if target_category:
            break

    filtered_cocktails = cocktails_list
    if target_category:
        filtered_cocktails = [c for c in cocktails_list if c["kategorie"].lower() == target_category.lower()]
        
    scored_cocktails = []
    
    for cocktail in filtered_cocktails:
        name = normalize_text(cocktail["name"])
        description = normalize_text(cocktail["beschreibung"])
        ingredients = [normalize_text(z) for z in cocktail["zutaten"]]
        spirits = [normalize_text(s) for s in cocktail["spirituose"]]
        tastes = [normalize_text(g) for g in cocktail["geschmack"]]
        category = normalize_text(cocktail["kategorie"])
        strength = normalize_text(cocktail["staerke"])
        
        # Falls ein Ausschlusskriterium zutrifft, diesen Cocktail komplett Ã¼berspringen
        is_excluded = False
        for exc in exclusions:
            # Wenn der Cocktail alkoholfrei ist und das Ausschlusskriterium ein Alkohol-Ausschluss ist, ignorieren wir es
            if category == "alkoholfrei" and exc in ALCOHOL_TERMS:
                continue
            if (
                term_matches(exc, name)
                or term_matches(exc, description)
                or any(term_matches(exc, ing) for ing in ingredients)
                or any(term_matches(exc, sp) for sp in spirits)
                or any(term_matches(exc, taste) for taste in tastes)
                or term_matches(exc, category)
                or term_matches(exc, strength)
            ):
                is_excluded = True
                break
        if is_excluded:
            continue
            
        score = 0
        match_details = []
        
        # Falls eine Kategorie Ã¼ber suggestion-pill/Keyword explizit gefiltert wurde, kriegt das Produkt einen Basis-Score
        if target_category and category == normalize_text(target_category):
            score += 10
            match_details.append(f"Kategorie '{target_category}' Ãœbereinstimmung (+10)")
        
        query_terms = [w for w in words if w not in STOP_WORDS and len(w) > 1 and w not in exclusions]
        if exclusions and not query_terms and not target_category:
            score += 1
            match_details.append("Passt zu den Ausschlusskriterien (+1)")
        
        # Hoher Score bei direktem Match des Namens
        if query_lower in name:
            score += 20
            match_details.append("Name enthÃ¤lt Suchanfrage (+20)")
            
        for term in query_terms:
            # 1. Match im Namen
            if term_matches(term, name) or (len(term) > 3 and name in term):
                score += 15
                match_details.append(f"Name enthÃ¤lt '{term}' (+15)")
            # 2. Match in den Spirituosen
            matched_spirits = [sp for sp in spirits if term_matches(term, sp)]
            if matched_spirits:
                score += 10
                match_details.append(f"Spirituose enthÃ¤lt '{', '.join(matched_spirits)}' (+10)")
            # 3. Match in der Kategorie
            if term_matches(term, category):
                # Verhindert doppelte Kategorie-Punkte, falls oben bereits vergeben/gefiltert
                if not target_category:
                    score += 8
                    match_details.append(f"Kategorie enthÃ¤lt '{category}' (+8)")
            # 4. Match in der StÃ¤rke
            if term_matches(term, strength):
                score += 6
                match_details.append(f"StÃ¤rke enthÃ¤lt '{strength}' (+6)")
            # 5. Match in den Geschmacksrichtungen
            matched_tastes = [t for t in tastes if term_matches(term, t)]
            if matched_tastes:
                score += 5
                match_details.append(f"Geschmack enthÃ¤lt '{', '.join(matched_tastes)}' (+5)")
            # 6. Match in den Zutaten
            matched_ingredients = [ing for ing in ingredients if term_matches(term, ing)]
            if matched_ingredients:
                score += 4
                match_details.append(f"Zutat enthÃ¤lt '{', '.join(matched_ingredients)}' (+4)")
            # 7. Match in der Beschreibung
            if term_matches(term, description):
                score += 2
                match_details.append(f"Beschreibung enthÃ¤lt '{term}' (+2)")
                
        if score > 0:
            c_copy = cocktail.copy()
            c_copy["match_score"] = score
            c_copy["match_details"] = match_details
            scored_cocktails.append((score, c_copy))
            
    # Zufall vor dem Sortieren sorgt bei gleich guten Treffern fuer Variation,
    # waehrend hoehere Scores weiterhin bevorzugt bleiben.
    random.shuffle(scored_cocktails)

    # Sortieren nach Score absteigend
    scored_cocktails.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_cocktails[:limit]]

async def query_ollama(prompt_messages: list) -> str:
    """
    Sendet die Anfrage an den Ollama Server Ã¼ber die OpenAI-kompatible Schnittstelle.
    """
    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": prompt_messages,
        "temperature": 0.25,
        "top_p": 0.7,
        "response_format": {"type": "json_object"},
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

def public_cocktail_catalog(cocktails_list: list) -> list:
    return [
        {
            "name": c["name"],
            "kategorie": c["kategorie"],
            "preis": c["preis"],
            "spirituose": c["spirituose"],
            "geschmack": c["geschmack"],
            "staerke": c["staerke"],
            "zutaten": c["zutaten"],
            "beschreibung": c["beschreibung"],
        }
        for c in cocktails_list
    ]

def find_cocktails_by_names(names: list, cocktails_list: list) -> list:
    by_name = {c["name"].lower(): c for c in cocktails_list}
    selected = []
    seen = set()
    for name in names:
        if not isinstance(name, str):
            continue
        key = name.lower().strip()
        if key in by_name and key not in seen:
            selected.append(by_name[key].copy())
            seen.add(key)
    return selected

def infer_cocktail_names_from_answer(answer: str, cocktails_list: list) -> list:
    answer_lower = answer.lower()
    matches = []
    for cocktail in cocktails_list:
        name = cocktail["name"]
        pattern = rf"(?<!\w){re.escape(name.lower())}(?!\w)"
        match = re.search(pattern, answer_lower)
        if match:
            matches.append((match.start(), name))
    matches.sort(key=lambda item: item[0])
    return [name for _, name in matches[:3]]

def infer_cocktail_names_from_text(text: str, cocktails_list: list) -> list:
    normalized_text = normalize_text(text)
    matches = []
    for cocktail in cocktails_list:
        normalized_name = normalize_text(cocktail["name"])
        if term_matches(normalized_name, normalized_text):
            match_index = normalized_text.find(normalized_name)
            matches.append((match_index if match_index >= 0 else 9999, cocktail["name"]))
    matches.sort(key=lambda item: item[0])
    return [name for _, name in matches[:3]]

def extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    objects = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                objects.append(obj)
            index = start + end
        except json.JSONDecodeError:
            index = start + 1

    if not objects:
        raise ValueError("LLM response did not contain a JSON object.")

    merged = {}
    for obj in objects:
        merged.update(obj)
    return merged

def build_no_match_answer() -> str:
    return (
        "Bier, Wein oder andere Getränke außerhalb der Cocktailkarte habe ich hier leider nicht im Katalog. "
        "Ich kann dir aber gern etwas Passendes aus unserer Cocktailkarte empfehlen, zum Beispiel frisch, "
        "fruchtig, alkoholfrei oder mit deiner Lieblingsspirituose."
    )

def make_empty_result_answer_helpful(answer: str) -> str:
    stripped = answer.strip()
    stripped = stripped.replace("keine Bier", "kein Bier")
    stripped = stripped.replace("kein Bier-Optionen", "kein Bier")
    if not stripped:
        return build_no_match_answer()

    if "cocktail" in normalize_text(stripped) or "drink" in normalize_text(stripped):
        return stripped

    return (
        f"{stripped} Wenn du magst, schaue ich dir stattdessen gern in der Cocktailkarte nach "
        "etwas Frischem, Fruchtigem oder Alkoholfreiem."
    )

def find_mentioned_component(message: str, cocktail: dict) -> Optional[str]:
    components = cocktail.get("spirituose", []) + cocktail.get("zutaten", [])
    for component in components:
        if term_matches(component, message):
            return component
    return None

def expand_short_single_cocktail_answer(message: str, answer: str, cocktail: dict) -> str:
    normalized_answer = normalize_text(answer)
    component = find_mentioned_component(message, cocktail)
    if normalized_answer in {"ja", "yes"} and component:
        return f"Ja, der {cocktail['name']} enthält {component}."
    if normalized_answer in {"nein", "no"} and component:
        return f"Nein, der {cocktail['name']} enthält kein {component}."
    return answer

def build_database_answer(retrieved: list, connection_warning: bool = False, message: str = "") -> str:
    if not retrieved:
        answer = build_no_match_answer()
    else:
        exclusions = parse_query_exclusions(message) if message else []
        if exclusions:
            exclusion_text = ", ".join(exc.capitalize() for exc in exclusions)
            lines = [
                f"Klar, ich habe ein paar passende Cocktails ohne {exclusion_text} gefunden. "
                "Hier sind drei Vorschläge aus der Karte:",
                ""
            ]
        else:
            lines = [
                "Gerne. Ich habe in der Karte ein paar passende Kandidaten gefunden. "
                "Diese drei würden gut zu deinem Wunsch passen:",
                ""
            ]
        for c in retrieved[:3]:
            spirits = ", ".join(c["spirituose"]) if c["spirituose"] else "alkoholfrei"
            tastes = ", ".join(c["geschmack"][:3])
            lines.append(
                f"- **{c['name']}** passt gut, wenn du etwas {tastes} möchtest. "
                f"Er ist {c['staerke']} und liegt bei {c['preis']} EUR. "
                f"Die Basis ist {spirits}; die Details siehst du rechts im Rezeptkatalog."
            )
        lines.append("")
        lines.append(f"Mein erster Griff wäre **{retrieved[0]['name']}**.")
        answer = "\n".join(lines)

    if connection_warning:
        return answer
    return answer

def build_hybrid_rag_prompt(user_message: str, cocktails_list: list, history: Optional[List[ChatMessage]] = None) -> list:
    catalog_json = json.dumps(public_cocktail_catalog(cocktails_list), ensure_ascii=False)
    cocktail_names = [c["name"] for c in cocktails_list]

    system_prompt = (
        "Du bist CocktailGPT, ein charmanter Barkeeper-Assistent fuer eine feste Cocktailkarte.\n"
        "Das Backend hat die Nutzereingabe bereits analysiert und passende Kandidaten aus der JSON-Datei herausgesucht.\n"
        "Du arbeitest strikt RAG-basiert mit dem Abschnitt KANDIDATEN_JSON.\n"
        "Du darfst ausschliesslich Cocktails nennen, deren Name exakt in ERLAUBTE_COCKTAILNAMEN steht.\n"
        "Du darfst keine Cocktailnamen, Zutaten, Preise, Staerken, Kategorien oder Rezepte erfinden.\n"
        "Du kannst zwei Arten von Anfragen beantworten:\n"
        "1. Bei Fragen zu konkreten Cocktails beantwortest du die Frage direkt anhand der Kandidaten, ohne ungefragt weitere Cocktails zu empfehlen.\n"
        "2. Bei Empfehlungswuenschen empfiehl die besten bis zu drei Kandidaten. Nutze bevorzugt die ersten Kandidaten, weil sie am besten gematcht wurden.\n"
        "Wenn keine Kandidaten uebergeben wurden oder die Frage nicht aus den Kandidaten beantwortbar ist, gib eine leere cocktail_names-Liste zurueck und erfinde keinen Ersatz.\n"
        "Bei Fragen zu Bier, Wein, Essen oder anderen Dingen ausserhalb der Cocktailkarte antworte freundlich, dass diese nicht im Katalog stehen, und biete eine passende Cocktail-Alternative an.\n"
        "cocktail_names enthaelt die Cocktails, die fuer deine Antwort relevant sind: bei Fragen die betroffenen Cocktails, bei Empfehlungen die empfohlenen Cocktails.\n"
        "Erwaehne in answer keine weiteren Cocktailnamen ausser denen in cocktail_names, auch nicht als Vergleich oder Variante.\n"
        "answer MUSS jeden Namen aus cocktail_names exakt nennen.\n"
        "Die Antwort soll charmant und beratend klingen, kurz auf den Nutzerwunsch eingehen und keine vollstaendigen Zutatenlisten herunterrattern.\n"
        "Zutaten stehen im Rezeptkatalog; erwaehne sie nur sparsam, wenn sie fuer die Antwort wichtig sind.\n"
        "Bei Ja/Nein-Fragen zu Zutaten oder Spirituosen pruefe die Felder zutaten und spirituose exakt.\n"
        "Bei Empfehlungen nenne zu jedem empfohlenen Cocktail kurz Geschmack, Staerke und Preis.\n"
        "Das JSON-Objekt MUSS genau die Felder answer und cocktail_names enthalten.\n"
        "cocktail_names MUSS die exakt geschriebenen Namen der relevanten Cocktails enthalten.\n"
        "Gib ausschliesslich ein gueltiges JSON-Objekt ohne Markdown-Codeblock zurueck.\n"
        'Format: {"answer":"deine deutsche Antwort","cocktail_names":["Name 1","Name 2","Name 3"]}\n\n'
        f"ERLAUBTE_COCKTAILNAMEN: {json.dumps(cocktail_names, ensure_ascii=False)}\n\n"
        f"KANDIDATEN_JSON:\n{catalog_json}"
    )

    prompt_messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[-6:]:
            if msg.role in {"user", "assistant"}:
                prompt_messages.append({"role": msg.role, "content": msg.content})

    prompt_messages.append({"role": "user", "content": user_message})
    return prompt_messages

def validate_catalog_llm_result(result: dict, selected: list, all_cocktails: list) -> bool:
    if not isinstance(result.get("answer"), str):
        return False
    if not isinstance(result.get("cocktail_names"), list):
        return False

    requested_names = [
        name.lower().strip()
        for name in result["cocktail_names"]
        if isinstance(name, str) and name.strip()
    ]
    selected_names = {c["name"].lower() for c in selected}

    if len(requested_names) != len(set(requested_names)):
        return False
    if len(requested_names) > 3:
        return False
    if set(requested_names) != selected_names:
        return False

    answer_lower = result["answer"].lower()
    known_names = {c["name"].lower() for c in all_cocktails}
    mentioned_known_names = {
        name
        for name in known_names
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", answer_lower)
    }

    if not selected_names:
        return not mentioned_known_names

    return selected_names.issubset(mentioned_known_names) and mentioned_known_names.issubset(selected_names)

async def build_llm_hybrid_response(message: str, candidates: list, history: Optional[List[ChatMessage]] = None) -> dict:
    prompt_messages = build_hybrid_rag_prompt(message, candidates, history)
    raw_answer = await query_ollama(prompt_messages)
    parsed = extract_json_object(raw_answer)
    if not parsed.get("cocktail_names") and isinstance(parsed.get("answer"), str):
        parsed["cocktail_names"] = (
            infer_cocktail_names_from_answer(parsed["answer"], candidates)
            or infer_cocktail_names_from_text(message, candidates)
        )
    selected = find_cocktails_by_names(parsed.get("cocktail_names", []), candidates)

    if not selected:
        if parse_query_exclusions(message):
            raise ValueError("LLM returned no cocktails for an exclusion-based recommendation.")
        if (
            isinstance(parsed.get("answer"), str)
            and validate_catalog_llm_result({"answer": parsed["answer"], "cocktail_names": []}, [], candidates)
        ):
            return {
                "answer": make_empty_result_answer_helpful(parsed["answer"]),
                "cocktails": []
            }
        return {
            "answer": build_no_match_answer(),
            "cocktails": []
        }

    if len(selected) == 1 and isinstance(parsed.get("answer"), str):
        selected_name = selected[0]["name"]
        parsed["answer"] = expand_short_single_cocktail_answer(message, parsed["answer"], selected[0])
        if not infer_cocktail_names_from_answer(parsed["answer"], selected):
            parsed["answer"] = f"Zum {selected_name}: {parsed['answer']}"

    if not validate_catalog_llm_result(parsed, selected, candidates):
        raise ValueError("LLM response failed catalog validation.")

    return {
        "answer": parsed["answer"],
        "cocktails": selected
    }

async def build_chat_response(message: str, history: Optional[List[ChatMessage]] = None) -> dict:
    ollama_connection_failed = False
    mentioned_names = infer_cocktail_names_from_text(message, cocktails)
    matched_candidates = find_cocktails_by_names(mentioned_names, cocktails)
    if not matched_candidates:
        matched_candidates = smart_retrieve_cocktails(message, cocktails, limit=12)

    if not matched_candidates:
        return {
            "answer": build_no_match_answer(),
            "cocktails": []
        }

    query_terms = [
        word for word in tokenize_query(message)
        if word not in STOP_WORDS and word not in parse_query_exclusions(message) and len(word) > 1
    ]
    positive_terms = [
        word for word in query_terms
        if any(
            term_matches(word, value)
            for cocktail in cocktails
            for value in (
                cocktail["spirituose"]
                + cocktail["geschmack"]
                + cocktail["zutaten"]
                + [cocktail["kategorie"], cocktail["staerke"]]
            )
        )
    ]
    exclusion_only_request = bool(parse_query_exclusions(message)) and not positive_terms and not mentioned_names

    if exclusion_only_request:
        top_3_retrieved = matched_candidates[:3]
        return {
            "answer": build_database_answer(top_3_retrieved, message=message),
            "cocktails": top_3_retrieved
        }

    if USE_LLM_ANSWER:
        try:
            return await build_llm_hybrid_response(message, matched_candidates, history)
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("Ollama request failed, falling back to local retrieval: %s", exc)
            ollama_connection_failed = True
        except Exception as exc:
            logger.warning("LLM catalog response failed validation, falling back to local retrieval: %s", exc)
            ollama_connection_failed = False

    fallback_retrieved = matched_candidates
    top_3_retrieved = fallback_retrieved[:3]

    if not top_3_retrieved:
        return {
            "answer": build_no_match_answer(),
            "cocktails": fallback_retrieved
        }

    return {
        "answer": build_database_answer(top_3_retrieved, connection_warning=ollama_connection_failed, message=message),
        "cocktails": top_3_retrieved
    }

@app.get("/")
def root():
    return {
        "message": "Cocktail RAG API lÃ¤uft",
        "ollama_url": OLLAMA_BASE_URL,
        "model": LLM_MODEL,
        "use_llm_answer": USE_LLM_ANSWER
    }

@app.get("/hello")
def hello():
    return {
        "message": "Hallo Fabian"
    }

@app.get("/cocktails")
def get_cocktails():
    return cocktails

@app.get("/cocktails/{name}")
def get_cocktail_by_name(name: str):
    for cocktail in cocktails:
        if cocktail["name"].lower() == name.lower():
            return cocktail
    return {"error": "Cocktail nicht gefunden"}

@app.get("/search")
def search_cocktails(
        spirituose: Optional[str] = Query(default=None),
        geschmack: Optional[str] = Query(default=None),
        kategorie: Optional[str] = Query(default=None),
        staerke: Optional[str] = Query(default=None)
):
    result = []
    for cocktail in cocktails:
        if spirituose and not any(spirituose.lower() in s.lower() for s in cocktail["spirituose"]):
            continue
        if geschmack and not any(geschmack.lower() in g.lower() for g in cocktail["geschmack"]):
            continue
        if kategorie and kategorie.lower() != cocktail["kategorie"].lower():
            continue
        if staerke and staerke.lower() != cocktail["staerke"].lower():
            continue
        result.append(cocktail)

    return {
        "anzahl": len(result),
        "cocktails": result
    }

@app.get("/chat")
async def chat_search_get(frage: str):
    """
    GET Schnittstelle fÃ¼r RAG. FÃ¼hrt Suche durch und generiert Antwort mit dem LLM.
    """
    response = await build_chat_response(frage)
    response["frage"] = frage
    return response

@app.post("/chat")
async def chat_search_post(request: ChatRequest):
    """
    POST Schnittstelle fÃ¼r RAG mit Konversationsverlauf (Memory).
    """
    return await build_chat_response(request.message, request.history)
