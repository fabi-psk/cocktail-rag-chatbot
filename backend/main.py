import os
import json
import httpx
import re
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b")

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

def term_matches(term: str, text: str) -> bool:
    """
    Überprüft, ob ein Suchbegriff in einem Text vorkommt.
    Bei kurzen Begriffen (<= 3 Zeichen wie 'gin', 'rum', 'eis') wird Regex mit Wortgrenzen (\b) verwendet,
    um Fehlalarm-Matches zu vermeiden (z. B. 'gin' in 'ginger ale' oder 'in' in 'gin').
    Bei längeren Begriffen (> 3 Zeichen) erlauben wir bidirektionales Matching (z. B. 'fruchtigen' matchet 'fruchtig').
    Zusätzlich wird ein einfaches deutsches Stemming angewendet, um Einzahl/Mehrzahl auszugleichen (z. B. 'erdbeeren' -> 'erdbeere').
    """
    term_lower = term.lower()
    text_lower = text.lower()
    
    # Standard-Vergleich
    if len(term_lower) <= 3:
        pattern = rf"\b{re.escape(term_lower)}\b"
        if bool(re.search(pattern, text_lower, re.IGNORECASE)):
            return True
    else:
        if (term_lower in text_lower) or (text_lower in term_lower):
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
    query_lower = query.lower()
    
    # 0. Spezialfall: Zufällige Auswahl
    if "zufällig" in query_lower or "zufälligen" in query_lower or "zufällige" in query_lower:
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
    words = query_lower.split()
    
    # Erkennung von Wortkombinationen wie "kein X", "ohne X"
    for i, word in enumerate(words):
        if word in ["kein", "ohne", "keine"] and i + 1 < len(words):
            target = words[i+1].strip(".,?!;:")
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
    query_clean_words = [w.strip(".,?!;:") for w in words]
    for word in query_clean_words:
        if word in category_keywords:
            target_category = category_keywords[word]
            break

    filtered_cocktails = cocktails_list
    if target_category:
        filtered_cocktails = [c for c in cocktails_list if c["kategorie"].lower() == target_category.lower()]
        
    scored_cocktails = []
    
    for cocktail in filtered_cocktails:
        name = cocktail["name"].lower()
        description = cocktail["beschreibung"].lower()
        ingredients = [z.lower() for z in cocktail["zutaten"]]
        spirits = [s.lower() for s in cocktail["spirituose"]]
        tastes = [g.lower() for g in cocktail["geschmack"]]
        category = cocktail["kategorie"].lower()
        strength = cocktail["staerke"].lower()
        
        # Falls ein Ausschlusskriterium zutrifft, diesen Cocktail komplett überspringen
        is_excluded = False
        for exc in exclusions:
            # Wenn der Cocktail alkoholfrei ist und das Ausschlusskriterium ein Alkohol-Ausschluss ist, ignorieren wir es
            if category == "alkoholfrei" and exc in ["rum", "gin", "vodka", "tequila", "peachtree", "triple sec", "alkohol"]:
                continue
            if exc in name or exc in description or any(exc in ing for ing in ingredients) or any(exc in sp for sp in spirits):
                is_excluded = True
                break
        if is_excluded:
            continue
            
        score = 0
        match_details = []
        
        # Falls eine Kategorie über suggestion-pill/Keyword explizit gefiltert wurde, kriegt das Produkt einen Basis-Score
        if target_category and category == target_category.lower():
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
        query_terms = [w.strip(".,?!;:") for w in words if w not in stop_words and len(w.strip(".,?!;:")) > 1]
        
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
        "temperature": 0.7,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

@app.get("/")
def root():
    return {
        "message": "Cocktail RAG API läuft",
        "ollama_url": OLLAMA_BASE_URL,
        "model": LLM_MODEL
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
    all_retrieved = smart_retrieve_cocktails(frage, cocktails, limit=50)
    top_3_retrieved = all_retrieved[:3]
    
    # Prompt vorbereiten
    system_prompt = (
        "Du bist CocktailGPT, ein freundlicher und professioneller Barkeeper-Assistent.\n"
        "Deine Aufgabe ist es, Fragen zu beantworten und passende Cocktails zu empfehlen.\n"
        "Beantworte Fragen auf Deutsch. Verwende AUSSCHLIESSLICH die unten bereitgestellten Cocktail-Daten.\n"
        "Erwähne Details wie Zutaten, Geschmack, Stärke und Preis. Erfinde UNTER KEINEN UMSTÄNDEN Rezepte "
        "oder Cocktails, die nicht in den unten bereitgestellten Daten aufgelistet sind!\n"
        "WICHTIG: Empfiehl dem Benutzer in deiner Text-Antwort maximal 3 Cocktails, selbst wenn mehr bereitgestellt sind. "
        "Die vollständige Liste der passenden Cocktails wird dem Benutzer separat im Rezeptkatalog angezeigt.\n\n"
        "Relevante Cocktail-Daten:\n"
    )
    
    if top_3_retrieved:
        for c in top_3_retrieved:
            system_prompt += (
                f"- **{c['name']}**:\n"
                f"  Kategorie: {c['kategorie']}\n"
                f"  Preis: {c['preis']} €\n"
                f"  Spirituosen: {', '.join(c['spirituose']) if c['spirituose'] else 'Keine (alkoholfrei)'}\n"
                f"  Geschmack: {', '.join(c['geschmack'])}\n"
                f"  Stärke: {c['staerke']}\n"
                f"  Zutaten: {', '.join(c['zutaten'])}\n"
                f"  Beschreibung: {c['beschreibung']}\n\n"
            )
    else:
        system_prompt += (
            "ES WURDEN KEINE PASSENDEN COCKTAILS IN DER DATENBANK GEFUNDEN.\n"
            "WICHTIGSTE ANWEISUNG: Da die Liste der relevanten Cocktails leer ist, darfst du KEINEN Cocktail "
            "empfehlen oder erfinden. Antworte dem Benutzer höflich, dass kein passender Cocktail in der Datenbank "
            "gefunden wurde, der seinen Wünschen entspricht. Erkläre ihm (falls er nach cremig gesucht hat), "
            "dass alle cremigen Cocktails in der Karte Kokos enthalten, oder schlage allgemein vor, nach anderen "
            "Geschmäckern oder Spirituosen zu fragen."
        )
        
    user_frage = frage
    if "zufällig" in frage.lower() or "zufälligen" in frage.lower() or "zufällige" in frage.lower():
        if top_3_retrieved:
            user_frage += f"\n(Hinweis: Der Zufallsgenerator hat '{top_3_retrieved[0]['name']}' ausgewählt. Bitte stelle dem Benutzer ausschließlich diesen einen Cocktail vor und beschreibe ihn anhand der bereitgestellten Daten!)"
            
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_frage}
    ]
    
    try:
        answer = await query_ollama(prompt_messages)
    except Exception as e:
        # Fallback, falls Ollama offline ist (z.B. SSH-Tunnel nicht aktiv)
        answer = (
            "⚠️ **Ollama-Verbindung fehlgeschlagen!** (Ist der VPN- & SSH-Tunnel aktiv?)\n\n"
            "Ich habe passende Vorschläge direkt aus der Datenbank geladen. Die vollständige Liste (insgesamt "
            f"{len(all_retrieved)} Cocktails) findest du rechts im Rezeptkatalog.\n\n"
            "Hier sind die Top 3 Vorschläge:\n\n"
        )
        if top_3_retrieved:
            for c in top_3_retrieved:
                answer += f"- **{c['name']}** ({c['preis']} €): {c['beschreibung']}. Zutaten: {', '.join(c['zutaten'])}.\n"
        else:
            answer += "Leider wurden keine passenden Cocktails in der Datenbank gefunden."
            
    return {
        "frage": frage,
        "answer": answer,
        "cocktails": all_retrieved
    }

@app.post("/chat")
async def chat_search_post(request: ChatRequest):
    """
    POST Schnittstelle für RAG mit Konversationsverlauf (Memory).
    """
    # 1. Cocktails basierend auf der aktuellen Nachricht suchen
    all_retrieved = smart_retrieve_cocktails(request.message, cocktails, limit=50)
    top_3_retrieved = all_retrieved[:3]
    
    # 2. System-Prompt mit Kontext-Cocktails zusammenbauen
    system_prompt = (
        "Du bist CocktailGPT, ein freundlicher und professioneller Barkeeper-Assistent.\n"
        "Deine Aufgabe ist es, Fragen zu beantworten und passende Cocktails zu empfehlen.\n"
        "Beantworte Fragen auf Deutsch. Verwende AUSSCHLIESSLICH die unten bereitgestellten Cocktail-Daten.\n"
        "Erwähne Details wie Zutaten, Geschmack, Stärke und Preis. Erfinde UNTER KEINEN UMSTÄNDEN Rezepte "
        "oder Cocktails, die nicht in den unten bereitgestellten Daten aufgelistet sind!\n"
        "WICHTIG: Empfiehl dem Benutzer in deiner Text-Antwort maximal 3 Cocktails, selbst wenn mehr bereitgestellt sind. "
        "Die vollständige Liste der passenden Cocktails wird dem Benutzer separat im Rezeptkatalog angezeigt.\n\n"
        "Relevante Cocktail-Daten:\n"
    )
    
    if top_3_retrieved:
        for c in top_3_retrieved:
            system_prompt += (
                f"- **{c['name']}**:\n"
                f"  Kategorie: {c['kategorie']}\n"
                f"  Preis: {c['preis']} €\n"
                f"  Spirituosen: {', '.join(c['spirituose']) if c['spirituose'] else 'Keine (alkoholfrei)'}\n"
                f"  Geschmack: {', '.join(c['geschmack'])}\n"
                f"  Stärke: {c['staerke']}\n"
                f"  Zutaten: {', '.join(c['zutaten'])}\n"
                f"  Beschreibung: {c['beschreibung']}\n\n"
            )
    else:
        system_prompt += (
            "ES WURDEN KEINE PASSENDEN COCKTAILS IN DER DATENBANK GEFUNDEN.\n"
            "WICHTIGSTE ANWEISUNG: Da die Liste der relevanten Cocktails leer ist, darfst du KEINEN Cocktail "
            "empfehlen oder erfinden. Antworte dem Benutzer höflich, dass kein passender Cocktail in der Datenbank "
            "gefunden wurde, der seinen Wünschen entspricht. Erkläre ihm (falls er nach cremig gesucht hat), "
            "dass alle cremigen Cocktails in der Karte Kokos enthalten, oder schlage allgemein vor, nach anderen "
            "Geschmäckern oder Spirituosen zu fragen."
        )
        
    prompt_messages = [{"role": "system", "content": system_prompt}]
    
    # 3. Verlauf hinzufügen
    for msg in request.history:
        prompt_messages.append({"role": msg.role, "content": msg.content})
        
    # 4. Aktuelle Frage anhängen
    user_message = request.message
    if "zufällig" in request.message.lower() or "zufälligen" in request.message.lower() or "zufällige" in request.message.lower():
        if top_3_retrieved:
            user_message += f"\n(Hinweis: Der Zufallsgenerator hat '{top_3_retrieved[0]['name']}' ausgewählt. Bitte stelle dem Benutzer ausschließlich diesen einen Cocktail vor und beschreibe ihn anhand der bereitgestellten Daten!)"
            
    prompt_messages.append({"role": "user", "content": user_message})
    
    try:
        answer = await query_ollama(prompt_messages)
    except Exception as e:
        # Fallback, falls Ollama offline ist
        answer = (
            "⚠️ **Ollama-Verbindung fehlgeschlagen!** (Ist der VPN- & SSH-Tunnel aktiv?)\n\n"
            "Ich habe passende Vorschläge direkt aus der Datenbank geladen. Die vollständige Liste (insgesamt "
            f"{len(all_retrieved)} Cocktails) findest du rechts im Rezeptkatalog.\n\n"
            "Hier sind die Top 3 Vorschläge:\n\n"
        )
        if top_3_retrieved:
            for c in top_3_retrieved:
                answer += f"- **{c['name']}** ({c['preis']} €): {c['beschreibung']}. Zutaten: {', '.join(c['zutaten'])}.\n"
        else:
            answer += "Leider wurden keine passenden Cocktails in der Datenbank gefunden."
            
    return {
        "answer": answer,
        "cocktails": all_retrieved
    }
