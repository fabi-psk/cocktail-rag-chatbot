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
