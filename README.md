# cocktail-rag-chatbot
KI-gestützter Cocktailberater mit RAG, Ollama und Web-Interface.

## Testweise Intent-Erkennung

Die Intent-Erkennung ist standardmäßig aktiv. Zum Abschalten das Backend mit
`INTENT_ROUTING_ENABLED=false` starten. Ohne die Variable oder mit dem Wert `true`
werden Begrüßungen, Empfehlungen, Präferenzänderungen, Cocktail-Details, Reset,
Gesprächsaussagen, fachfremde Anfragen und unbekannte Eingaben unterschieden. Die
normalen Gesprächsantworten formuliert das LLM; feste Texte dienen nur als Fallback.
