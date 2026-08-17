# cocktail-rag-chatbot
KI-gestützter Cocktailberater mit RAG, Ollama und Web-Interface.

## KI-basierte Intent-Erkennung

Das LLM interpretiert jede Nachricht zusammen mit dem kurzen Chatverlauf und dem
strukturierten Sitzungszustand. Seine strukturierte Ausgabe bestimmt Intent,
Aktion und Kontextbezug. Schlägt die LLM-Interpretation fehl, wird kein Intent
anhand lokaler Schlüsselwortregeln geraten.
