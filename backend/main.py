from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(
    title="Cocktail RAG API",
    description="Backend für den Cocktail-RAG-Chatbot",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cocktaildaten laden
with open("../data/cocktails.json", "r", encoding="utf-8") as file:
    cocktails = json.load(file)


@app.get("/")
def root():
    return {
        "message": "Cocktail RAG API läuft"
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

    return {
        "error": "Cocktail nicht gefunden"
    }


@app.get("/search")
def search_cocktails(
        spirituose: str | None = Query(default=None),
        geschmack: str | None = Query(default=None),
        kategorie: str | None = Query(default=None),
        staerke: str | None = Query(default=None)
):

    ausschluesse = []
    result = []

    for cocktail in cocktails:

        # Spirituose prüfen
        if spirituose:
            if not any(
                    spirituose.lower() in s.lower()
                    for s in cocktail["spirituose"]
            ):
                continue

        # Geschmack prüfen
        if geschmack:
            if not any(
                    geschmack.lower() in g.lower()
                    for g in cocktail["geschmack"]
            ):
                continue

        # Kategorie prüfen
        if kategorie:
            if kategorie.lower() != cocktail["kategorie"].lower():
                continue

        # Stärke prüfen
        if staerke:
            if staerke.lower() != cocktail["staerke"].lower():
                continue

        result.append(cocktail)

    return {
        "anzahl": len(result),
        "cocktails": result
    }

@app.get("/chat")
def chat_search(frage: str):

    text = frage.lower()

    spirituose = None
    kategorie = None
    staerke = None
    geschmack = None
    zutat = None

    ausschluesse = []

    # Spirituosen erkennen
    if "rum" in text:
        spirituose = "Rum"
    elif "gin" in text:
        spirituose = "Gin"
    elif "vodka" in text:
        spirituose = "Vodka"
    elif "tequila" in text:
        spirituose = "Tequila"

    # Kategorien erkennen
    if "fruchtig" in text:
        kategorie = "Fruchtig"
    elif "cremig" in text:
        kategorie = "Cremig"
    elif "klassisch" in text:
        kategorie = "Klassisch"

    # Geschmäcker erkennen

    if "sahnig" in text:
        geschmack = "cremig"

    elif "cremig" in text:
        geschmack = "cremig"

    elif "süß" in text:
        geschmack = "süß"

    elif "frisch" in text:
        geschmack = "frisch"

    elif "tropisch" in text:
        geschmack = "tropisch"

    elif "sauer" in text:
        geschmack = "sauer"

    elif "fruchtig" in text:
        geschmack = "fruchtig"

    # Zutaten erkennen

    if ("kokos" in text
            and "kein kokos" not in text
            and "ohne kokos" not in text):
        zutat = "kokos"

    elif "ananas" in text:
        zutat = "ananas"

    elif "erdbeer" in text:
        zutat = "erdbeer"

    elif "mango" in text:
         zutat = "mango"

    # Stärke erkennen
    if "stark" in text:
        staerke = "hoch"

    # Ausschlüsse erkennen
    if "kein kokos" in text or "ohne kokos" in text:
        ausschluesse.append("kokos")

    if "keine sahne" in text or "ohne sahne" in text:
        ausschluesse.append("sahne")

    if "kein tequila" in text or "ohne tequila" in text:
        ausschluesse.append("tequila")

    if "kein rum" in text or "ohne rum" in text:
        ausschluesse.append("rum")

    if "kein gin" in text or "ohne gin" in text:
        ausschluesse.append("gin")

    if "kein vodka" in text or "ohne vodka" in text:
        ausschluesse.append("vodka")

    result = []

    for cocktail in cocktails:

        if spirituose:
            if not any(
                    spirituose.lower() in s.lower()
                    for s in cocktail["spirituose"]
            ):
                continue

        if geschmack:
            if not any(
                    geschmack.lower() in g.lower()
                    for g in cocktail["geschmack"]
            ):
                continue

        if zutat:

            zutaten_text = " ".join(cocktail["zutaten"]).lower()

            if zutat not in zutaten_text:
                continue

        if kategorie:
            if cocktail["kategorie"].lower() != kategorie.lower():
                continue

        if staerke:
            if cocktail["staerke"].lower() != staerke.lower():
                continue

        # Ausschlüsse prüfen
        if ausschluesse:

            zutaten_text = " ".join(cocktail["zutaten"]).lower()
            spirituosen_text = " ".join(cocktail["spirituose"]).lower()

            ausgeschlossen = False

            for ausschluss in ausschluesse:

                if ausschluss in zutaten_text:
                    ausgeschlossen = True

                if ausschluss in spirituosen_text:
                    ausgeschlossen = True

            if ausgeschlossen:
                continue

        result.append(cocktail)

    return {
        "frage": frage,
        "anzahl": len(result),
        "cocktails": result
    }
