# CocktailGPT

CocktailGPT ist ein KI-gestützter Cocktailberater mit Weboberfläche. Die Anwendung
verbindet eine strukturierte Cocktailkarte mit einem lokal betriebenen Sprachmodell
über Ollama. Sie bietet einen Bar-Modus für Empfehlungen aus der hinterlegten Karte
und einen Home-Modus für Rezeptanfragen.

## Funktionen

- semantische Erkennung von Benutzerabsichten durch ein lokales LLM
- Cocktail-Empfehlungen anhand von Geschmack, Spirituose, Stärke und Ausschlüssen
- Fragen zu Verfügbarkeit, Zutaten, Preis, Geschmack und Rezepten
- zufällige Cocktailauswahl
- kontextbezogene Folgefragen innerhalb einer Sitzung
- Bar-Modus mit Warenkorb und optionaler Bestellfreigabe
- Home-Modus für die Suche und Aufbereitung externer Cocktailrezepte
- automatisierte Backend-Tests

## Voraussetzungen

- Python 3.11 oder neuer
- Node.js 20.19 oder neuer
- npm
- [Ollama](https://ollama.com/) mit einem verfügbaren Modell

Standardmäßig verwendet das Backend das Modell `llama3:8b`. Es kann beispielsweise
mit folgenden Befehlen vorbereitet werden:

```bash
ollama pull llama3:8b
ollama serve
```

## Installation

### Backend

Im Projektverzeichnis:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd backend
cp .env.example .env
uvicorn main:app --reload
```

Das Backend ist anschließend unter `http://localhost:8000` erreichbar. Die
automatisch erzeugte API-Dokumentation befindet sich unter
`http://localhost:8000/docs`.

Unter Windows wird die virtuelle Umgebung mit folgendem Befehl aktiviert:

```powershell
.venv\Scripts\activate
```

### Frontend

In einem zweiten Terminal:

```bash
cd frontend
cp .env.example .env
npm ci
npm run dev
```

Die Weboberfläche ist standardmäßig unter `http://localhost:5173` verfügbar.

## Konfiguration

Die Backend-Konfiguration liegt in `backend/.env`:

| Variable | Standardwert | Bedeutung |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Adresse des Ollama-Servers |
| `LLM_MODEL` | `llama3:8b` | Name des verwendeten Sprachmodells |
| `USE_LLM_ANSWER` | `true` | Aktiviert die LLM-gestützte Antworterzeugung |
| `BAR_UNLOCK_PASSWORD` | `1111` | Passwort für die Bestellfreigabe im Bar-Modus |

Die optionale Frontend-Variable `VITE_API_BASE_URL` legt die Adresse des Backends
fest. Ohne diese Variable verwendet das Frontend Port `8000` auf demselben Host.

Lokale `.env`-Dateien werden nicht versioniert. Für die Abgabe sind ausschließlich
die Beispielkonfigurationen enthalten.

## Tests und Qualitätsprüfung

Backend-Tests ausführen:

```bash
cd backend
python -m unittest discover -s tests
```

Frontend prüfen und Produktions-Build erzeugen:

```bash
cd frontend
npm run lint
npm run build
```

## Projektstruktur

```text
cocktail-rag-chatbot/
├── backend/
│   ├── llm/              # Kommunikation mit Ollama
│   ├── models/           # Daten- und API-Modelle
│   ├── repositories/     # Zugriff auf die Cocktaildaten
│   ├── routes/           # FastAPI-Endpunkte
│   ├── services/         # Intent-, Such- und Rezeptlogik
│   ├── tests/            # automatisierte Tests
│   └── main.py           # Einstiegspunkt des Backends
├── data/
│   └── cocktails.json    # strukturierte Cocktailkarte
├── frontend/
│   ├── src/              # React-Anwendung
│   ├── package.json
│   └── package-lock.json
├── requirements.txt      # Python-Abhängigkeiten
└── README.md
```

## Hinweise

Für den Bar-Modus werden ausschließlich die Einträge aus `data/cocktails.json`
verwendet. Der Home-Modus benötigt zusätzlich eine Internetverbindung, um externe
Rezeptinformationen abzurufen. Das Sprachmodell wird lokal über Ollama angesprochen;
es werden keine Zugangsdaten für einen externen KI-Dienst benötigt.
