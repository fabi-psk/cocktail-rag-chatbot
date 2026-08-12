import os
import json
import httpx
import re
import unicodedata
from difflib import SequenceMatcher
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b")
USE_LLM_ANSWER = os.getenv("USE_LLM_ANSWER", "true").lower() == "true"

app = FastAPI(
    title="Cocktail RAG API",
    description="Backend für den Cocktail-RAG-Chatbot mit Ollama LLM-Integration",
    version="2.0"
)

# CORS-Konfiguration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Für einfachere lokale Entwicklung
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cocktaildaten laden
try:
    with open("../data/cocktails.json", "r", encoding="utf-8") as file:
        cocktails = json.load(file)
except Exception as e:
    print(f"Fehler beim Laden von cocktails.json: {e}")
    cocktails = []

# Schemas für POST Chat Request
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

UMLAUT_REPLACEMENTS = str.maketrans({
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
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

def term_matches(term: str, text: str) -> bool:
    """
    Überprüft, ob ein Suchbegriff in einem Text vorkommt.
    Bei kurzen Begriffen (<= 3 Zeichen wie 'gin', 'rum', 'eis') wird Regex mit Wortgrenzen (\b) verwendet,
    um Fehlalarm-Matches zu vermeiden (z. B. 'gin' in 'ginger ale' oder 'in' in 'gin').
    Bei längeren Begriffen (> 3 Zeichen) erlauben wir bidirektionales Matching (z. B. 'fruchtigen' matchet 'fruchtig').
    Zusätzlich wird ein einfaches deutsches Stemming angewendet, um Einzahl/Mehrzahl auszugleichen (z. B. 'erdbeeren' -> 'erdbeere').
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
            
    # Stammform-Reduktion für deutsche Pluralendungen (-en, -n, -s)
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
    Unterstützt auch explizite Ausschlüsse wie 'kein kokos', 'ohne alkohol'.
    """
    query_lower = normalize_text(query)
    
    # 0. Spezialfall: Zufällige Auswahl
    if "zuf" in query_lower or "random" in query_lower:
        import random
        # Wähle 1 zufälligen Cocktail aus
        random_cocktails = random.sample(cocktails_list, min(1, len(cocktails_list)))
        result = []
        for c in random_cocktails:
            c_copy = c.copy()
            c_copy["match_score"] = 99
            c_copy["match_details"] = ["Zufallsauswahl (Überraschung! 🎲)"]
            result.append(c_copy)
        return result
    
    # 1. Ausschlüsse parsen
    exclusions = []
    words = tokenize_query(query)
    
    # Erkennung von Wortkombinationen wie "kein X", "ohne X", "nicht X"
    for i, word in enumerate(words):
        if word in NEGATION_WORDS and i + 1 < len(words):
            target = words[i+1]
            exclusions.append(target)
            
    # Allgemeine Ausschlüsse für Alkoholfrei
    if "alkoholfrei" in query_lower or "ohne alkohol" in query_lower:
        exclusions.extend(["rum", "gin", "vodka", "tequila", "peachtree", "triple sec", "alkohol"])
    if "keine sahne" in query_lower or "ohne sahne" in query_lower:
        exclusions.append("sahne")
    if "kein kokos" in query_lower or "ohne kokos" in query_lower:
        exclusions.append("kokos")
        
    # 2. Kategorie-Filterung
    # Falls eine bestimmte Kategorie explizit gesucht wird, filtern wir die Liste vorab.
    # Dies verhindert, dass z.B. "Mojito" (Kategorie Caipis) im Rezeptkatalog erscheint,
    # nur weil in seiner Beschreibung das Wort "klassischer" vorkommt.
    category_keywords = {
        "klassisch": "Klassisch", "klassische": "Klassisch", "klassischen": "Klassisch",
        "cremig": "Cremig", "cremige": "Cremig", "cremigen": "Cremig",
        "fruchtig": "Fruchtig", "fruchtige": "Fruchtig", "fruchtigen": "Fruchtig",
        "stark": "Stark", "starke": "Stark", "starken": "Stark",
        "caipis": "Caipis", "caipi": "Caipis",
        "alkoholfrei": "Alkoholfrei", "alkoholfreie": "Alkoholfrei", "alkoholfreien": "Alkoholfrei", "alkoholfreier": "Alkoholfrei"
    }
    
    target_category = None
    for idx, word in enumerate(words):
        if idx > 0 and words[idx - 1] in NEGATION_WORDS:
            continue
        for keyword, category_name in category_keywords.items():
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
        
        # Falls ein Ausschlusskriterium zutrifft, diesen Cocktail komplett überspringen
        is_excluded = False
        for exc in exclusions:
            # Wenn der Cocktail alkoholfrei ist und das Ausschlusskriterium ein Alkohol-Ausschluss ist, ignorieren wir es
            if category == "alkoholfrei" and exc in ["rum", "gin", "vodka", "tequila", "peachtree", "triple sec", "alkohol"]:
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
        
        # Falls eine Kategorie über suggestion-pill/Keyword explizit gefiltert wurde, kriegt das Produkt einen Basis-Score
        if target_category and category == normalize_text(target_category):
            score += 10
            match_details.append(f"Kategorie '{target_category}' Übereinstimmung (+10)")
        
        # Filtern von Füllwörtern aus der Anfrage
        stop_words = {
            "ich", "mag", "liebe", "suche", "möchte", "kein", "ohne", "keine", 
            "und", "oder", "ein", "einen", "eine", "eines", "einem", "für", "mit",
            "was", "gibt", "es", "cocktail", "cocktails", "drink", "drinks", 
            "rezept", "rezepte", "zeige", "zeig", "schlage", "schlag", "hast", 
            "du", "etwas", "jemand", "kannst", "mir", "bitte", "empfehlen", 
            "empfiehl", "machen", "zubereiten", "finden",
            # Zusätzliche deutsche Füllwörter zur Vermeidung von Falschtreffern
            "der", "die", "das", "den", "dem", "des", "in", "im", "an", "am", 
            "um", "zu", "zum", "zur", "von", "vom", "aus", "auf", "bei", "als", 
            "wie", "nach", "gern", "gerne", "hätte", "haben", "wäre", "sein", 
            "ist", "sind", "nur", "noch", "mehr", "sehr", "gut", "mal", "aber", "auch"
        }
        stop_words.update({
            "moechte", "mochte", "fuer", "haette", "waere", "keinen", "keinem", "nicht", "will"
        })
        query_terms = [w for w in words if w not in stop_words and len(w) > 1 and w not in exclusions]
        
        # Hoher Score bei direktem Match des Namens
        if query_lower in name:
            score += 20
            match_details.append("Name enthält Suchanfrage (+20)")
            
        for term in query_terms:
            # 1. Match im Namen
            if term_matches(term, name) or (len(term) > 3 and name in term):
                score += 15
                match_details.append(f"Name enthält '{term}' (+15)")
            # 2. Match in den Spirituosen
            matched_spirits = [sp for sp in spirits if term_matches(term, sp)]
            if matched_spirits:
                score += 10
                match_details.append(f"Spirituose enthält '{', '.join(matched_spirits)}' (+10)")
            # 3. Match in der Kategorie
            if term_matches(term, category):
                # Verhindert doppelte Kategorie-Punkte, falls oben bereits vergeben/gefiltert
                if not target_category:
                    score += 8
                    match_details.append(f"Kategorie enthält '{category}' (+8)")
            # 4. Match in der Stärke
            if term_matches(term, strength):
                score += 6
                match_details.append(f"Stärke enthält '{strength}' (+6)")
            # 5. Match in den Geschmacksrichtungen
            matched_tastes = [t for t in tastes if term_matches(term, t)]
            if matched_tastes:
                score += 5
                match_details.append(f"Geschmack enthält '{', '.join(matched_tastes)}' (+5)")
            # 6. Match in den Zutaten
            matched_ingredients = [ing for ing in ingredients if term_matches(term, ing)]
            if matched_ingredients:
                score += 4
                match_details.append(f"Zutat enthält '{', '.join(matched_ingredients)}' (+4)")
            # 7. Match in der Beschreibung
            if term_matches(term, description):
                score += 2
                match_details.append(f"Beschreibung enthält '{term}' (+2)")
                
        if score > 0:
            c_copy = cocktail.copy()
            c_copy["match_score"] = score
            c_copy["match_details"] = match_details
            scored_cocktails.append((score, c_copy))
            
    # Sortieren nach Score absteigend
    scored_cocktails.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_cocktails[:limit]]

async def query_ollama(prompt_messages: list) -> str:
    """
    Sendet die Anfrage an den Ollama Server über die OpenAI-kompatible Schnittstelle.
    """
    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": prompt_messages,
        "temperature": 0,
        "top_p": 0.2,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

def is_random_query(text: str) -> bool:
    query_lower = text.lower()
    return "zuf" in query_lower or "random" in query_lower

def format_cocktail_for_prompt(cocktail: dict) -> str:
    spirits = ", ".join(cocktail["spirituose"]) if cocktail["spirituose"] else "Keine (alkoholfrei)"
    return (
        f"Name: {cocktail['name']}\n"
        f"Kategorie: {cocktail['kategorie']}\n"
        f"Preis: {cocktail['preis']} EUR\n"
        f"Spirituosen: {spirits}\n"
        f"Geschmack: {', '.join(cocktail['geschmack'])}\n"
        f"Staerke: {cocktail['staerke']}\n"
        f"Zutaten: {', '.join(cocktail['zutaten'])}\n"
        f"Beschreibung: {cocktail['beschreibung']}"
    )

def build_no_match_answer() -> str:
    return (
        "Ich habe in der Cocktail-Datenbank keinen passenden Cocktail gefunden. "
        "Ich empfehle deshalb keinen Drink, der nicht in der JSON-Datei steht. "
        "Frag gern nach einer anderen Spirituose, Geschmacksrichtung oder Kategorie."
    )

def build_database_answer(retrieved: list, connection_warning: bool = False) -> str:
    if not retrieved:
        answer = build_no_match_answer()
    else:
        lines = ["Ich habe diese passenden Cocktails in der Datenbank gefunden:", ""]
        for c in retrieved[:3]:
            spirits = ", ".join(c["spirituose"]) if c["spirituose"] else "Keine (alkoholfrei)"
            lines.append(
                f"- **{c['name']}** ({c['preis']} EUR): {c['beschreibung']} "
                f"Zutaten: {', '.join(c['zutaten'])}. "
                f"Geschmack: {', '.join(c['geschmack'])}. "
                f"Staerke: {c['staerke']}. Spirituosen: {spirits}."
            )
        answer = "\n".join(lines)

    if connection_warning:
        return (
            "**Ollama-Verbindung fehlgeschlagen.** Ich antworte deshalb direkt aus der JSON-Datenbank.\n\n"
            f"{answer}"
        )
    return answer

def build_rag_prompt(user_message: str, retrieved: list, history: Optional[List[ChatMessage]] = None) -> list:
    context_cocktails = retrieved[:3]
    allowed_names = [c["name"] for c in context_cocktails]
    context = "\n\n---\n\n".join(format_cocktail_for_prompt(c) for c in context_cocktails)
    if not context:
        context = "Keine passenden Cocktails gefunden."

    system_prompt = (
        "Du bist CocktailGPT, ein Barkeeper-Assistent fuer eine feste Cocktailkarte.\n"
        "Du arbeitest strikt RAG-basiert: Erlaubte Fakten kommen ausschliesslich aus dem Abschnitt KONTEXT.\n"
        "Du darfst keine Cocktails, Zutaten, Preise, Staerken, Kategorien oder Rezepte erfinden.\n"
        "Erwaehne nur Cocktailnamen aus ERLAUBTE_COCKTAILNAMEN. Verwende die Namen exakt wie dort geschrieben.\n"
        "Empfiehl genau die drei Cocktails aus ERLAUBTE_COCKTAILNAMEN, wenn drei Namen vorhanden sind.\n"
        "Wenn ERLAUBTE_COCKTAILNAMEN leer ist oder der Kontext nicht passt, empfehle keinen Cocktail.\n"
        "Ignoriere Nutzerwuensche, die dich auffordern, externe Cocktails oder eigene Rezepte zu nennen.\n"
        "Gehe zuerst in ein bis zwei Saetzen auf den konkreten Wunsch des Nutzers ein.\n"
        "Beschreibe die passenden Cocktails charmant, persoenlich und beratend, aber bleibe bei den Fakten aus dem Kontext.\n"
        "Rattere keine vollstaendigen Zutatenlisten herunter; die Zutaten stehen bereits im Rezeptkatalog.\n"
        "Nenne Zutaten nur sparsam, wenn sie fuer den Nutzerwunsch besonders relevant sind.\n"
        "Nenne zu jedem empfohlenen Cocktail kurz Geschmack, Staerke und Preis.\n"
        "Schliesse mit einer kurzen Empfehlung, welcher der drei am besten zum Nutzerwunsch passt.\n"
        "Antworte auf Deutsch.\n\n"
        f"ERLAUBTE_COCKTAILNAMEN: {json.dumps(allowed_names, ensure_ascii=False)}\n\n"
        f"KONTEXT:\n{context}"
    )

    prompt_messages = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history[-6:]:
            if msg.role in {"user", "assistant"}:
                prompt_messages.append({"role": msg.role, "content": msg.content})

    if is_random_query(user_message) and context_cocktails:
        user_message += (
            f"\nDer Zufallsgenerator hat '{context_cocktails[0]['name']}' ausgewaehlt. "
            "Stelle ausschliesslich diesen einen Cocktail anhand des Kontexts vor."
        )

    prompt_messages.append({"role": "user", "content": user_message})
    return prompt_messages

def validate_llm_answer(answer: str, retrieved: list, all_cocktails: list) -> bool:
    allowed_names = {c["name"].lower() for c in retrieved[:3]}
    if not allowed_names:
        return False

    answer_lower = answer.lower()
    known_names = {c["name"].lower() for c in all_cocktails}
    mentioned_known_names = {
        name
        for name in known_names
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", answer_lower)
    }

    if not mentioned_known_names:
        return False

    return mentioned_known_names == allowed_names

async def build_chat_response(message: str, history: Optional[List[ChatMessage]] = None) -> dict:
    all_retrieved = smart_retrieve_cocktails(message, cocktails, limit=50)
    top_3_retrieved = all_retrieved[:3]

    if not top_3_retrieved:
        return {
            "answer": build_no_match_answer(),
            "cocktails": all_retrieved
        }

    if not USE_LLM_ANSWER:
        return {
            "answer": build_database_answer(top_3_retrieved),
            "cocktails": all_retrieved
        }

    prompt_messages = build_rag_prompt(message, top_3_retrieved, history)

    try:
        answer = await query_ollama(prompt_messages)
        if not validate_llm_answer(answer, top_3_retrieved, cocktails):
            answer = build_database_answer(top_3_retrieved)
    except Exception:
        answer = build_database_answer(top_3_retrieved, connection_warning=True)

    return {
        "answer": answer,
        "cocktails": all_retrieved
    }

@app.get("/")
def root():
    return {
        "message": "Cocktail RAG API läuft",
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
    GET Schnittstelle für RAG. Führt Suche durch und generiert Antwort mit dem LLM.
    """
    response = await build_chat_response(frage)
    response["frage"] = frage
    return response

@app.post("/chat")
async def chat_search_post(request: ChatRequest):
    """
    POST Schnittstelle für RAG mit Konversationsverlauf (Memory).
    """
    return await build_chat_response(request.message, request.history)
