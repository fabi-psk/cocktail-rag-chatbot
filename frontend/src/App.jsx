import { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hallo! Ich bin **CocktailGPT**, dein persönlicher Barkeeper-Assistent. 🍹\n\nWelchen Cocktail suchst du heute oder nach welchen Geschmäckern steht dir der Sinn? Beschreibe einfach deinen Wunsch (z.B. *'Ich möchte einen fruchtigen Cocktail mit Rum'* oder *'Ich mag Gin, aber keinen Wodka und kein Kokos'*). Ich schlage dir passende Rezepte vor!",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [retrievedCocktails, setRetrievedCocktails] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [allCocktails, setAllCocktails] = useState([]);
  const [isShuffling, setIsShuffling] = useState(false);
  const [shuffleColor, setShuffleColor] = useState("");

  const messagesEndRef = useRef(null);

  // Alle Cocktails laden für den Slot-Machine-Effekt
  useEffect(() => {
    fetch("http://127.0.0.1:8000/cocktails")
      .then((res) => res.json())
      .then((data) => setAllCocktails(data))
      .catch((err) => console.error("Fehler beim Laden aller Cocktails:", err));
  }, []);

  // Automatisches Scrollen zum Ende des Chats bei neuen Nachrichten
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Funktion zum Zurücksetzen des Chats
  const handleReset = () => {
    setMessages([
      {
        role: "assistant",
        content:
          "Hallo! Ich bin **CocktailGPT**, dein persönlicher Barkeeper-Assistent. 🍹\n\nWelchen Cocktail suchst du heute oder nach welchen Geschmäckern steht dir der Sinn? Beschreibe einfach deinen Wunsch (z.B. *'Ich möchte einen fruchtigen Cocktail mit Rum'* oder *'Ich mag Gin, aber keinen Wodka und kein Kokos'*). Ich schlage dir passende Rezepte vor!",
      },
    ]);
    setRetrievedCocktails([]);
    setInputValue("");
  };

  // Funktion zum Senden einer Nachricht
  const handleSend = async (textToSend) => {
    const text = textToSend || inputValue;
    if (!text.trim()) return;

    // Benutzer-Nachricht hinzufügen
    const newMessages = [...messages, { role: "user", content: text }];
    setMessages(newMessages);
    setInputValue("");
    setIsLoading(true);

    // Den Chatverlauf für das Backend vorbereiten (ohne die Systemprompts)
    const history = messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: text,
          history: history,
        }),
      });

      if (!response.ok) {
        throw new Error("API-Fehler beim Abrufen der Antwort");
      }

      const data = await response.json();

      // Bot-Antwort hinzufügen
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer },
      ]);

      // Gefundene Cocktails aktualisieren
      if (data.cocktails && data.cocktails.length > 0) {
        setRetrievedCocktails(data.cocktails);
      }
    } catch (error) {
      console.error("Fehler bei der Kommunikation mit dem Backend:", error);
      
      // Fehler-Fallback-Antwort hinzufügen
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "⚠️ **Verbindungsproblem:** Die Antwort konnte nicht geladen werden. Bitte vergewissere dich, dass der Backend-Server auf Port 8000 läuft und die Verbindung zum KI-Server aktiv ist.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Slot-Machine-Effekt für die zufällige Auswahl von Cocktails
  const triggerRandomShuffle = async (randomText) => {
    if (allCocktails.length === 0) {
      handleSend(randomText);
      return;
    }

    setIsShuffling(true);
    setIsLoading(true);

    const history = messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));

    // Anfrage parallel starten
    const backendPromise = fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: randomText,
        history: history,
      }),
    }).then((res) => {
      if (!res.ok) throw new Error("API-Fehler");
      return res.json();
    });

    const colors = ["#ff0055", "#ff9900", "#10b981", "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899"];
    let colorIdx = 0;

    // Mische Cocktails alle 150ms (etwas langsamer für bessere Lesbarkeit)
    const shuffleInterval = setInterval(() => {
      const tempCocktails = [];
      const pool = [...allCocktails];
      if (pool.length > 0) {
        const randIdx = Math.floor(Math.random() * pool.length);
        const selected = pool[randIdx];
        tempCocktails.push({
          ...selected,
          match_score: 99,
          match_details: ["Zufallsauswahl... 🎲"],
        });
      }
      setRetrievedCocktails(tempCocktails);

      // Wechsle die Farbe synchron zum Cocktail-Wechsel
      setShuffleColor(colors[colorIdx % colors.length]);
      colorIdx++;
    }, 150);

    // Mindestlaufzeit der Animation: 2.4 Sekunden (doppelt so lang wie vorher)
    const animationTimer = new Promise((resolve) => setTimeout(resolve, 2400));

    try {
      const [data] = await Promise.all([backendPromise, animationTimer]);

      clearInterval(shuffleInterval);
      setIsShuffling(false);
      setShuffleColor("");

      setMessages((prev) => [
        ...prev,
        { role: "user", content: randomText },
        { role: "assistant", content: data.answer },
      ]);

      if (data.cocktails && data.cocktails.length > 0) {
        setRetrievedCocktails(data.cocktails);
      }
    } catch (error) {
      clearInterval(shuffleInterval);
      setIsShuffling(false);
      setShuffleColor("");
      console.error("Fehler beim Mischen:", error);
      handleSend(randomText);
    } finally {
      setIsLoading(false);
    }
  };

  // Hilfsfunktion zum Rendern von einfachem Markdown im Chat
  const renderMessageContent = (text) => {
    const lines = text.split("\n");
    return lines.map((line, idx) => {
      // Erkennung von Listenpunkten
      if (line.trim().startsWith("- ") || line.trim().startsWith("* ")) {
        const content = line.trim().substring(2);
        return (
          <li key={idx} style={{ marginLeft: "20px", marginBottom: "4px" }}>
            {renderInlineFormatting(content)}
          </li>
        );
      }
      // Erkennung von nummerierten Listen
      const numMatch = line.trim().match(/^\d+\.\s(.*)/);
      if (numMatch) {
        return (
          <li key={idx} style={{ marginLeft: "20px", marginBottom: "4px", listStyleType: "decimal" }}>
            {renderInlineFormatting(numMatch[1])}
          </li>
        );
      }
      // Normale Textzeile
      return (
        <p key={idx} style={{ margin: "4px 0", minHeight: "1em" }}>
          {renderInlineFormatting(line)}
        </p>
      );
    });
  };

  // Hilfsfunktion zum Rendern von fetten (**bold**) und kursiven (*italic*) Wörtern
  const renderInlineFormatting = (text) => {
    // 1. Bold und Italic parsen
    let parts = [text];

    // Regex für **bold**
    const boldRegex = /\*\*(.*?)\*\*/g;
    let boldParts = [];
    
    // Einfache Ersetzung für Bold
    for (let part of parts) {
      if (typeof part === "string") {
        let lastIndex = 0;
        let match;
        while ((match = boldRegex.exec(part)) !== null) {
          const before = part.substring(lastIndex, match.index);
          if (before) boldParts.push(before);
          boldParts.push(
            <strong key={`b-${match.index}`} style={{ color: "#d8b4fe", fontWeight: "700" }}>
              {match[1]}
            </strong>
          );
          lastIndex = boldRegex.lastIndex;
        }
        const after = part.substring(lastIndex);
        if (after) boldParts.push(after);
      } else {
        boldParts.push(part);
      }
    }
    
    // Regex für *italic*
    let finalParts = [];
    const italicRegex = /\*(.*?)\*/g;
    
    for (let part of boldParts) {
      if (typeof part === "string") {
        let lastIndex = 0;
        let match;
        while ((match = italicRegex.exec(part)) !== null) {
          const before = part.substring(lastIndex, match.index);
          if (before) finalParts.push(before);
          finalParts.push(
            <em key={`i-${match.index}`} style={{ color: "#a5b4fc", fontStyle: "italic" }}>
              {match[1]}
            </em>
          );
          lastIndex = italicRegex.lastIndex;
        }
        const after = part.substring(lastIndex);
        if (after) finalParts.push(after);
      } else {
        finalParts.push(part);
      }
    }

    return finalParts.length > 0 ? finalParts : text;
  };

  const suggestions = [
    { label: "🍸 Klassisch", text: "Zeige mir klassische Cocktails", color: "#3b82f6", glow: "rgba(59, 130, 246, 0.3)" },
    { label: "🍓 Fruchtig", text: "Ich möchte einen fruchtigen Cocktail", color: "#ec4899", glow: "rgba(236, 72, 153, 0.3)" },
    { label: "☁️ Cremig", text: "Ich suche einen cremigen Cocktail", color: "#eab308", glow: "rgba(234, 179, 8, 0.3)" },
    { label: "💪 Stark", text: "Zeige mir starke Cocktails", color: "#a855f7", glow: "rgba(168, 85, 247, 0.3)" },
    { label: "🍋 Caipis", text: "Zeige mir Cocktails aus der Kategorie Caipis", color: "#10b981", glow: "rgba(16, 185, 129, 0.3)" },
    { label: "🚫 Alkoholfrei", text: "Ich suche einen alkoholfreien Cocktail", color: "#06b6d4", glow: "rgba(6, 182, 212, 0.3)" },
    { label: "🍋 Sauer", text: "Ich suche einen sauren Cocktail (Sour)", color: "#eab308", glow: "rgba(234, 179, 8, 0.3)" },
    { label: "🥥 Ohne Kokos", text: "Ich suche einen Cocktail ohne Kokos", color: "#f97316", glow: "rgba(249, 115, 22, 0.3)" },
    { label: "🎲 Zufällig", text: "Schlage mir einen zufälligen Cocktail vor!", color: "#10b981", glow: "rgba(16, 185, 129, 0.3)" },
    { label: "🧹 Reset", text: "RESET", color: "#ef4444", glow: "rgba(239, 68, 68, 0.3)" }
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header Bereich */}
      <div className="chat-header">
        <div className="chat-header-title">
          <span style={{ fontSize: "28px" }}>🍹</span>
          <div>
            <h1>CocktailGPT</h1>
            <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
              Wirtschaftsinformatik Projekt • RAG Chatbot
            </div>
          </div>
        </div>
        <div className="status-badge">
          <span className="status-dot"></span>
          KI-Server (Ollama) verbunden
        </div>
      </div>

      {/* Haupt-Chat und Panel-Bereich */}
      <div className="chat-container">
        {/* Linke Seite: Chat-Verlauf */}
        <div className="chat-window">
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message-wrapper ${msg.role}`}>
                <div className="message-bubble">
                  {renderMessageContent(msg.content)}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message-wrapper assistant">
                <div className="message-bubble" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>CocktailGPT mixt</span>
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Eingabebereich */}
          <div className="chat-input-area">
            {/* Quick Suggestions Pills */}
            <div className="suggestions-bar">
              {suggestions.map((s, index) => (
                <button
                  key={index}
                  className="suggestion-pill"
                  onClick={() => {
                    if (s.text === "RESET") {
                      handleReset();
                    } else if (s.text.includes("zufällig") || s.label.includes("Zufällig")) {
                      triggerRandomShuffle(s.text);
                    } else {
                      handleSend(s.text);
                    }
                  }}
                  disabled={isLoading && s.text !== "RESET"}
                  style={{
                    color: s.color,
                    borderColor: `${s.color}35`,
                    backgroundColor: `${s.color}12`,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = s.color;
                    e.currentTarget.style.boxShadow = `0 0 12px ${s.glow}`;
                    e.currentTarget.style.transform = 'translateY(-1px) scale(1.03)';
                    e.currentTarget.style.backgroundColor = `${s.color}22`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = `${s.color}35`;
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.backgroundColor = `${s.color}12`;
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>

            {/* Formular zum Absenden */}
            <form
              className="chat-input-form"
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
            >
              <input
                type="text"
                className="chat-input"
                placeholder="Frag CocktailGPT nach einem Rezept, Geschmack oder Zutaten..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                disabled={isLoading}
              />
              <button
                type="submit"
                className="send-button"
                disabled={isLoading || !inputValue.trim()}
              >
                Senden ➔
              </button>
            </form>
          </div>
        </div>

        {/* Rechte Seite: Cocktail-Details Panel */}
        <div className="cocktail-panel">
          <h2 className="panel-title">
            <span>📋</span> Rezeptekatalog ({retrievedCocktails.length})
          </h2>
          
          {retrievedCocktails.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "12px" }}>
                Folgende Cocktails wurden aus der Datenbank geladen (die Top 3 wurden als Kontext an die KI übergeben):
              </div>
              {retrievedCocktails.map((cocktail) => (
                <div 
                  key={cocktail.name} 
                  className={`cocktail-card ${isShuffling ? "shuffling-card" : ""}`}
                  style={isShuffling && shuffleColor ? {
                    borderColor: shuffleColor,
                    boxShadow: `0 0 20px ${shuffleColor}50`,
                    background: `${shuffleColor}10`,
                    borderWidth: "2px",
                    borderStyle: "solid"
                  } : {}}
                >
                  <h3>
                    <span>🍹 {cocktail.name}</span>
                    <span className="cocktail-price">{cocktail.preis.toFixed(2)} €</span>
                  </h3>
                  <div className="cocktail-meta">
                    <span>{cocktail.kategorie}</span>
                    <span>Stärke: {cocktail.staerke}</span>
                  </div>
                  {cocktail.spirituose && cocktail.spirituose.length > 0 ? (
                    <div style={{ fontSize: "13px", margin: "6px 0", color: "#e2e8f0" }}>
                      <strong>Spirituosen:</strong> {cocktail.spirituose.join(", ")}
                    </div>
                  ) : (
                    <div style={{ fontSize: "13px", margin: "6px 0", color: "#a7f3d0" }}>
                      <strong>Spirituosen:</strong> Keine (Alkoholfrei)
                    </div>
                  )}
                  <div style={{ fontSize: "13px", margin: "6px 0", color: "#e2e8f0" }}>
                    <strong>Zutaten:</strong> {cocktail.zutaten.join(", ")}
                  </div>
                  <div className="cocktail-desc">{cocktail.beschreibung}</div>
                  
                </div>
              ))}
            </div>
          ) : (
            <div className="no-cocktails-placeholder">
              <span style={{ fontSize: "36px", marginBottom: "12px" }}>🔍</span>
              <p style={{ fontWeight: "600", marginBottom: "4px", color: "#fff" }}>Keine Cocktails geladen</p>
              <p style={{ fontSize: "12px" }}>
                Wenn du eine Frage stellst, erscheinen hier die relevanten Cocktail-Rezepte aus der Datenbank.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
