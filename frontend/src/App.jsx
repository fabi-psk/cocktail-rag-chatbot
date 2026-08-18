import { useState, useEffect, useMemo, useRef } from "react";
import "./App.css";
import {
  fetchBackendStatus,
  fetchOrderingStatus,
  resetChatSession,
  sendChatMessage,
  unlockOrdering,
} from "./services/api";

const createSessionId = () => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
};

const getOrCreateSessionId = () => {
  try {
    const storedSessionId = window.localStorage.getItem("cocktailgpt-session-id");
    if (storedSessionId) return storedSessionId;
    const sessionId = createSessionId();
    window.localStorage.setItem("cocktailgpt-session-id", sessionId);
    return sessionId;
  } catch {
    return createSessionId();
  }
};

const isAlcoholicCocktail = (cocktail) =>
  normalize(cocktail?.staerke) !== "alkoholfrei" && (cocktail?.spirituose?.length || 0) > 0;

const welcomeMessage = (mode) => mode === "home"
  ? "Willkommen im **Zuhause-Modus**. Nenne mir einen konkreten Cocktail, dann suche ich das Rezept auf einer vertrauenswürdigen Webseite und zeige dir Zutaten, Zubereitung und Quelle."
  : "Hallo! Ich bin **CocktailGPT**, dein persönlicher Barkeeper-Assistent. 🍹\n\nWelchen Cocktail suchst du heute oder nach welchen Geschmäckern steht dir der Sinn? Beschreibe einfach deinen Wunsch (z.B. *'Ich möchte einen fruchtigen Cocktail mit Rum'* oder *'Ich mag Gin, aber keinen Wodka und kein Kokos'*). Ich schlage dir passende Rezepte vor!";

const normalize = (value) => String(value || "").trim().toLowerCase();

const getMatchLabel = (index) => {
  if (index === 0) return "Passt sehr gut";
  if (index === 1) return "Passt gut";
  return "Interessante Alternative";
};

const isRandomText = (text) => {
  const normalized = normalize(text)
    .replace(/[ä]/g, "ae")
    .replace(/[ü]/g, "ue")
    .replace(/[ö]/g, "oe");
  return (
    normalized.includes("zufaellig") ||
    normalized.includes("zufallig") ||
    normalized.includes("ueberrasch") ||
    normalized.includes("uberrasch") ||
    normalized.includes("irgendwas aus") ||
    normalized.includes("nicht entscheiden") ||
    normalized.includes("nochmal")
  );
};

function CocktailRoulette({ cocktails, selectedCocktail, onComplete }) {
  const sequence = useMemo(
    () => (cocktails?.length ? cocktails : selectedCocktail ? [selectedCocktail] : []),
    [cocktails, selectedCocktail]
  );
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const colors = ["#10b981", "#06b6d4", "#3b82f6", "#a855f7", "#ec4899", "#f97316", "#eab308"];

  useEffect(() => {
    if (!sequence.length || !selectedCocktail) return undefined;

    const delays = sequence.length === 1
      ? [350]
      : [100, 100, 120, 150, 190, 250, 330, 430];
    const timers = [];
    let elapsed = 0;

    delays.forEach((delay, index) => {
      elapsed += delay;
      timers.push(setTimeout(() => {
        if (index === delays.length - 1) {
          const selectedIndex = sequence.findIndex((item) => item.name === selectedCocktail.name);
          setCurrentIndex(selectedIndex >= 0 ? selectedIndex : 0);
          setIsFinished(true);
          onComplete?.();
          return;
        }
        setCurrentIndex((prev) => (prev + 1) % sequence.length);
      }, elapsed));
    });

    return () => timers.forEach((timer) => clearTimeout(timer));
  }, [sequence, selectedCocktail, onComplete]);

  if (!sequence.length) return null;

  const cocktail = isFinished ? selectedCocktail : sequence[currentIndex % sequence.length];
  const color = colors[currentIndex % colors.length];
  const highlights = [
    ...(cocktail?.geschmack || []),
    cocktail?.staerke,
  ].filter(Boolean).slice(0, 4);

  return (
    <div className={`roulette-shell ${isFinished ? "finished" : ""}`}>
      <div className="roulette-kicker">
        {isFinished ? "Dein Zufalls-Cocktail" : "CocktailGPT mixt dein Schicksal..."}
      </div>
      <div
        key={`${cocktail?.name}-${currentIndex}-${isFinished}`}
        className={`cocktail-card roulette-card ${isFinished ? "winner" : ""}`}
        style={{
          borderColor: color,
          boxShadow: `0 0 ${isFinished ? 28 : 18}px ${color}55`,
          background: `linear-gradient(135deg, ${color}1f, rgba(30, 41, 59, 0.72))`,
        }}
      >
        <h3>
          <span>{isFinished ? "🎉" : "🎲"} {cocktail?.name}</span>
          {cocktail?.preis != null && <span className="cocktail-price">{cocktail.preis.toFixed(2)} €</span>}
        </h3>
        <div className="match-label">{isFinished ? "Gewinner" : "Roulette laeuft"}</div>
        <div className="cocktail-meta">
          {highlights.map((value) => (
            <span key={`${cocktail?.name}-${value}`}>{value}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState("menu");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: welcomeMessage("menu"),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [retrievedCocktails, setRetrievedCocktails] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [rouletteResult, setRouletteResult] = useState(null);
  const [expandedCocktails, setExpandedCocktails] = useState({});
  const [webRecipes, setWebRecipes] = useState([]);
  const [cartItems, setCartItems] = useState([]);
  const [orderingBlocked, setOrderingBlocked] = useState(false);
  const [unlockOpen, setUnlockOpen] = useState(false);
  const [unlockPassword, setUnlockPassword] = useState("");
  const [unlockError, setUnlockError] = useState("");
  const [isUnlocking, setIsUnlocking] = useState(false);
  const [ollamaConnected, setOllamaConnected] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [viewMode, setViewMode] = useState(() => {
    try {
      return window.localStorage.getItem("cocktailgpt-view") === "mobile" ? "mobile" : "desktop";
    } catch {
      return "desktop";
    }
  });

  const messagesEndRef = useRef(null);
  const sessionIdRef = useRef(getOrCreateSessionId());
  const rouletteRunRef = useRef(0);

  // Automatisches Scrollen zum Ende des Chats bei neuen Nachrichten
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    let active = true;
    const refreshStatus = async () => {
      try {
        const status = await fetchBackendStatus();
        if (active) setOllamaConnected(Boolean(status.ollama_connected));
      } catch {
        if (active) setOllamaConnected(false);
      }
    };
    refreshStatus();
    const interval = window.setInterval(refreshStatus, 10000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    fetchOrderingStatus(sessionIdRef.current)
      .then((data) => setOrderingBlocked(Boolean(data.ordering_blocked)))
      .catch(() => setOrderingBlocked(false));
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("cocktailgpt-view", viewMode);
    } catch {
      // Die Ansicht funktioniert auch, wenn der Browser keinen lokalen Speicher erlaubt.
    }
  }, [viewMode]);

  const applyChatData = (data) => {
    if (typeof data.ordering_blocked === "boolean") {
      setOrderingBlocked(data.ordering_blocked);
    }
    setWebRecipes(data.web_recipes || []);
    if (data.type === "random" && data.selected_cocktail) {
      rouletteRunRef.current += 1;
      setRouletteResult({
        id: `${data.selected_cocktail.name}-${rouletteRunRef.current}`,
        rouletteCocktails: data.roulette_cocktails || [data.selected_cocktail],
        selectedCocktail: data.selected_cocktail,
        completed: false,
      });
      setRetrievedCocktails([]);
      setExpandedCocktails({});
      return;
    }
    setRouletteResult(null);
    setRetrievedCocktails(data.cocktails || []);
    if (!data.cocktails || data.cocktails.length === 0) {
      setExpandedCocktails({});
    }
  };

  // Funktion zum Zurücksetzen des Chats
  const handleReset = async () => {
    const oldSessionId = sessionIdRef.current;
    setMessages([
      {
        role: "assistant",
        content: welcomeMessage(mode),
      },
    ]);
    setRetrievedCocktails([]);
    setRouletteResult(null);
    setExpandedCocktails({});
    setWebRecipes([]);
    setInputValue("");
    try {
      await resetChatSession(oldSessionId);
    } catch (error) {
      console.error("Fehler beim ZurÃ¼cksetzen der Session:", error);
    }
  };

  const handleModeChange = async (nextMode) => {
    if (nextMode === mode || isLoading) return;
    const oldSessionId = sessionIdRef.current;
    setMode(nextMode);
    setMessages([{ role: "assistant", content: welcomeMessage(nextMode) }]);
    setRetrievedCocktails([]);
    setWebRecipes([]);
    setRouletteResult(null);
    setExpandedCocktails({});
    setInputValue("");
    try {
      await resetChatSession(oldSessionId);
    } catch (error) {
      console.error("Fehler beim Wechseln des Modus:", error);
    }
  };

  const addToCart = (cocktail) => {
    if (orderingBlocked && isAlcoholicCocktail(cocktail)) return;
    setCartItems((currentItems) => {
      const existingItem = currentItems.find((item) => item.name === cocktail.name);
      if (existingItem) {
        return currentItems.map((item) =>
          item.name === cocktail.name
            ? { ...item, quantity: item.quantity + 1 }
            : item
        );
      }
      return [...currentItems, { ...cocktail, quantity: 1 }];
    });
  };

  const changeCartQuantity = (cocktailName, amount) => {
    setCartItems((currentItems) => {
      const selectedItem = currentItems.find((item) => item.name === cocktailName);
      if (orderingBlocked && amount > 0 && isAlcoholicCocktail(selectedItem)) {
        return currentItems;
      }
      return currentItems
        .map((item) =>
          item.name === cocktailName
            ? { ...item, quantity: item.quantity + amount }
            : item
        )
        .filter((item) => item.quantity > 0);
    });
  };

  const removeFromCart = (cocktailName) => {
    setCartItems((currentItems) =>
      currentItems.filter((item) => item.name !== cocktailName)
    );
  };

  const handleUnlockOrdering = async (event) => {
    event.preventDefault();
    setUnlockError("");
    setIsUnlocking(true);
    try {
      await unlockOrdering(sessionIdRef.current, unlockPassword);
      setOrderingBlocked(false);
      setUnlockOpen(false);
      setUnlockPassword("");
    } catch {
      setUnlockError("Passwort nicht korrekt.");
    } finally {
      setIsUnlocking(false);
    }
  };

  // Funktion zum Senden einer Nachricht
  const handleSend = async (textToSend) => {
    const text = textToSend || inputValue;
    if (!text.trim() || isLoading || (rouletteResult && !rouletteResult.completed)) return;

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
      const data = await sendChatMessage({
        sessionId: sessionIdRef.current,
        message: text,
        history: history,
        mode,
      });

      // Bot-Antwort hinzufügen
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer || data.message },
      ]);

      // Gefundene Cocktails aktualisieren
      applyChatData(data);
    } catch (error) {
      console.error("Fehler bei der Kommunikation mit dem Backend:", error);
      
      // Fehler-Fallback-Antwort hinzufügen
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "CocktailGPT konnte gerade keine Antwort erzeugen. Versuch es bitte noch einmal.",
        },
      ]);
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
    { label: "🍓 Fruchtig", text: "Empfiehl mir einen fruchtigen Cocktail", color: "#ec4899", glow: "rgba(236, 72, 153, 0.3)" },
    { label: "☁️ Cremig", text: "Empfiehl mir einen cremigen Cocktail", color: "#eab308", glow: "rgba(234, 179, 8, 0.3)" },
    { label: "💪 Stark", text: "Empfiehl mir einen starken Cocktail", color: "#a855f7", glow: "rgba(168, 85, 247, 0.3)" },
    { label: "🚫 Alkoholfrei", text: "Empfiehl mir einen alkoholfreien Cocktail", color: "#06b6d4", glow: "rgba(6, 182, 212, 0.3)" },
    { label: "🍋 Sauer", text: "Empfiehl mir einen sauren Cocktail", color: "#eab308", glow: "rgba(234, 179, 8, 0.3)" },
    { label: "🥥 Ohne Kokos", text: "Empfiehl mir einen Cocktail ohne Kokos", color: "#f97316", glow: "rgba(249, 115, 22, 0.3)" }
  ];
  const homeSuggestions = [
    { label: "Mojito", text: "Wie mache ich einen Mojito?", color: "#10b981", glow: "rgba(16, 185, 129, 0.3)" },
    { label: "Margarita", text: "Zeig mir das Rezept für eine Margarita", color: "#eab308", glow: "rgba(234, 179, 8, 0.3)" },
    { label: "Moscow Mule", text: "Welche Zutaten brauche ich für einen Moscow Mule?", color: "#06b6d4", glow: "rgba(6, 182, 212, 0.3)" },
    { label: "Sex on the Beach", text: "Zeig mir das Rezept für Sex on the Beach", color: "#ec4899", glow: "rgba(236, 72, 153, 0.3)" },
    { label: "Pina Colada", text: "Wie mache ich eine Pina Colada?", color: "#f59e0b", glow: "rgba(245, 158, 11, 0.3)" },
    { label: "Mai Tai", text: "Zeig mir das Rezept für einen Mai Tai", color: "#f97316", glow: "rgba(249, 115, 22, 0.3)" },
    { label: "Caipirinha", text: "Wie mache ich eine Caipirinha?", color: "#84cc16", glow: "rgba(132, 204, 22, 0.3)" },
    { label: "Long Island Ice Tea", text: "Zeig mir das Rezept für einen Long Island Ice Tea", color: "#38bdf8", glow: "rgba(56, 189, 248, 0.3)" },
  ];
  const randomSuggestion = mode === "home"
    ? { label: "🎲 Zufälliges Rezept entdecken", text: "Starte das Rezept-Roulette", color: "#10b981" }
    : { label: "🎲 Zufälligen Cocktail auslosen", text: "Schlage mir einen zufälligen Cocktail vor!", color: "#10b981" };
  const isRouletteRunning = rouletteResult && !rouletteResult.completed;
  const cartCount = cartItems.reduce((total, item) => total + item.quantity, 0);
  const cartTotal = cartItems.reduce(
    (total, item) => total + (item.preis || 0) * item.quantity,
    0
  );

  return (
    <div className={`app-shell ${viewMode === "mobile" ? "mobile-presentation" : ""}`}>
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
        <div className="header-actions">
          <div className="mode-switch" aria-label="CocktailGPT Modus">
            <button
              type="button"
              className={mode === "menu" ? "active" : ""}
              onClick={() => handleModeChange("menu")}
            >
              Bar-Karte
            </button>
            <button
              type="button"
              className={mode === "home" ? "active" : ""}
              onClick={() => handleModeChange("home")}
            >
              Für Zuhause
            </button>
          </div>
          <div className={`status-badge ${mode === "menu" && ollamaConnected === false ? "offline" : ""}`}>
            <span className="status-dot"></span>
            {mode === "home"
              ? "Web-Rezepttest aktiv"
              : ollamaConnected === null
                ? "KI-Server wird geprüft"
                : ollamaConnected
                  ? "KI-Server (Ollama) verbunden"
                  : "KI-Server offline · Fallback aktiv"}
          </div>
          <div className="settings-control">
            <button
              type="button"
              className={`settings-button ${settingsOpen ? "active" : ""}`}
              onClick={() => setSettingsOpen((open) => !open)}
              aria-expanded={settingsOpen}
              aria-haspopup="menu"
            >
              <span aria-hidden="true">⚙</span>
              <span>Einstellungen</span>
            </button>
            {settingsOpen && (
              <div className="settings-menu" role="menu" aria-label="Ansicht auswählen">
                <div className="settings-menu-title">Ansicht</div>
                <button
                  type="button"
                  className={viewMode === "desktop" ? "selected" : ""}
                  onClick={() => {
                    setViewMode("desktop");
                    setSettingsOpen(false);
                  }}
                  role="menuitemradio"
                  aria-checked={viewMode === "desktop"}
                >
                  <span className="settings-option-icon">▰</span>
                  <span>
                    <strong>Desktop</strong>
                    <small>Normale Präsentationsansicht</small>
                  </span>
                  <span className="settings-check">{viewMode === "desktop" ? "✓" : ""}</span>
                </button>
                <button
                  type="button"
                  className={viewMode === "mobile" ? "selected" : ""}
                  onClick={() => {
                    setViewMode("mobile");
                    setSettingsOpen(false);
                  }}
                  role="menuitemradio"
                  aria-checked={viewMode === "mobile"}
                >
                  <span className="settings-option-icon">▯</span>
                  <span>
                    <strong>Mobile Präsentation</strong>
                    <small>Smartphone-Vorschau im Browser</small>
                  </span>
                  <span className="settings-check">{viewMode === "mobile" ? "✓" : ""}</span>
                </button>
              </div>
            )}
          </div>
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
              {(mode === "home" ? homeSuggestions : suggestions).map((s, index) => {
                return (
                <button
                  key={index}
                  className="suggestion-pill"
                  onClick={() => {
                    if (s.text === "RESET") {
                      handleReset();
                    } else if (s.random || isRandomText(s.text)) {
                      handleSend(s.text);
                    } else {
                      handleSend(s.text);
                    }
                  }}
                  disabled={(isLoading || isRouletteRunning) && s.text !== "RESET"}
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
                );
              })}
            </div>
            <div className="primary-action-row">
              <button
                type="button"
                className="random-action-button"
                onClick={() => handleSend(randomSuggestion.text)}
                disabled={isLoading || isRouletteRunning}
              >
                <span className="random-action-icon">🎲</span>
                <span>{randomSuggestion.label.replace("🎲 ", "")}</span>
                <span className="random-action-arrow">→</span>
              </button>
              <button
                type="button"
                className="reset-action-button"
                onClick={handleReset}
                disabled={isLoading}
              >
                <span className="reset-action-icon">🧹</span>
                <span>Reset</span>
              </button>
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
                placeholder={mode === "home"
                  ? "Welches Cocktailrezept möchtest du zuhause mixen?"
                  : "Frag CocktailGPT nach einem Rezept, Geschmack oder Zutaten..."}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                disabled={isLoading || isRouletteRunning}
              />
              <button
                type="submit"
                className="send-button"
                disabled={isLoading || isRouletteRunning || !inputValue.trim()}
              >
                Senden ➔
              </button>
            </form>
          </div>
        </div>

        {/* Rechte Seite: Cocktail-Details Panel */}
        <div className="cocktail-panel">
          {mode === "home" && (
            <h2 className="panel-title">
              <span>🏠</span>
              {` Web-Rezept (${webRecipes.length})`}
            </h2>
          )}

          {mode === "menu" && (
            <section className="shopping-cart" aria-label="Warenkorb">
              <div className="cart-header">
                <h3>🛒 Warenkorb</h3>
                <span className="cart-count">{cartCount}</span>
              </div>
              {orderingBlocked && (
                <div className="ordering-lock" role="alert">
                  <strong>Alkoholische Bestellungen gesperrt</strong>
                  <p>Die Sperre kann nur durch das Barpersonal aufgehoben werden.</p>
                  {!unlockOpen ? (
                    <button type="button" onClick={() => setUnlockOpen(true)}>
                      Durch Barpersonal entsperren
                    </button>
                  ) : (
                    <form onSubmit={handleUnlockOrdering} className="unlock-form">
                      <label htmlFor="bar-password">Personal-Passwort</label>
                      <div>
                        <input
                          id="bar-password"
                          type="password"
                          inputMode="numeric"
                          value={unlockPassword}
                          onChange={(event) => setUnlockPassword(event.target.value)}
                          autoFocus
                        />
                        <button type="submit" disabled={isUnlocking || !unlockPassword}>
                          {isUnlocking ? "Prüfe..." : "Entsperren"}
                        </button>
                      </div>
                      {unlockError && <span>{unlockError}</span>}
                    </form>
                  )}
                </div>
              )}
              {cartItems.length === 0 ? (
                <p className="cart-empty">Noch keine Cocktails hinzugefügt.</p>
              ) : (
                <>
                  <div className="cart-list">
                    {cartItems.map((item) => (
                      <div className="cart-item" key={item.name}>
                        <div className="cart-item-info">
                          <strong>{item.name}</strong>
                          <span>{((item.preis || 0) * item.quantity).toFixed(2)} €</span>
                        </div>
                        <div className="cart-item-actions">
                          <div className="cart-quantity" aria-label={`Menge für ${item.name}`}>
                            <button
                              type="button"
                              onClick={() => changeCartQuantity(item.name, -1)}
                              aria-label={`${item.name} einmal entfernen`}
                            >
                              −
                            </button>
                            <span>{item.quantity}</span>
                            <button
                              type="button"
                              onClick={() => changeCartQuantity(item.name, 1)}
                              aria-label={`${item.name} einmal hinzufügen`}
                              disabled={orderingBlocked && isAlcoholicCocktail(item)}
                            >
                              +
                            </button>
                          </div>
                          <button
                            type="button"
                            className="cart-remove-button"
                            onClick={() => removeFromCart(item.name)}
                            aria-label={`${item.name} aus dem Warenkorb entfernen`}
                          >
                            Entfernen
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="cart-footer">
                    <div>
                      <span>Gesamt</span>
                      <strong>{cartTotal.toFixed(2)} €</strong>
                    </div>
                    <button type="button" onClick={() => setCartItems([])}>
                      Warenkorb leeren
                    </button>
                  </div>
                </>
              )}
            </section>
          )}
          
          {rouletteResult && !rouletteResult.completed ? (
            <CocktailRoulette
              key={rouletteResult.id}
              cocktails={rouletteResult.rouletteCocktails}
              selectedCocktail={rouletteResult.selectedCocktail}
              onComplete={() => {
                setRetrievedCocktails([rouletteResult.selectedCocktail]);
                setRouletteResult((prev) => prev ? { ...prev, completed: true } : prev);
              }}
            />
          ) : mode === "home" ? (
            webRecipes.length > 0 ? (
              <div className="web-recipe-list">
                {webRecipes.map((recipe) => (
                  <article className="web-recipe-card" key={recipe.source_url}>
                    <div className="web-recipe-kicker">Aus der Quelle extrahiert und auf Deutsch umformuliert</div>
                    <h3>{recipe.name}</h3>
                    <section>
                      <strong>Zutaten</strong>
                      <ul>
                        {recipe.ingredients.map((ingredient) => <li key={ingredient}>{ingredient}</li>)}
                      </ul>
                    </section>
                    <section>
                      <strong>Zubereitung</strong>
                      <ol>
                        {recipe.instructions.map((step) => <li key={step}>{step}</li>)}
                      </ol>
                    </section>
                    {recipe.garnish?.length > 0 && (
                      <section>
                        <strong>Garnitur</strong>
                        <p>{recipe.garnish.join(" ")}</p>
                      </section>
                    )}
                    <a href={recipe.source_url} target="_blank" rel="noreferrer">
                      Quelle öffnen: {recipe.source_name}
                    </a>
                  </article>
                ))}
              </div>
            ) : (
              <div className="no-cocktails-placeholder">
                <span style={{ fontSize: "36px", marginBottom: "12px" }}>🌐</span>
                <p style={{ fontWeight: "600", marginBottom: "4px", color: "#fff" }}>Nenne einen konkreten Cocktail.</p>
                <p style={{ fontSize: "12px" }}>Das Rezept stammt von Cocktaildatenbank.de und wird von der LLM übersichtlich in eigenen Worten wiedergegeben.</p>
              </div>
            )
          ) : retrievedCocktails.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              {rouletteResult?.completed && (
                <div className="roulette-winner-banner">
                  <span>🎉 Dein Cocktail ist:</span>
                  <strong>{rouletteResult.selectedCocktail.name}</strong>
                </div>
              )}
              <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "12px" }}>
                Ergebnisse aus der Cocktailkarte für deine aktuelle Anfrage:
              </div>
              {retrievedCocktails.map((cocktail, index) => {
                const isExpanded = expandedCocktails[cocktail.name] || false;
                const quantityInCart = cartItems.find((item) => item.name === cocktail.name)?.quantity || 0;
                const highlights = [
                  ...(cocktail.geschmack || []),
                  cocktail.staerke,
                ].filter(Boolean).slice(0, 4);

                return (
                  <div
                    key={cocktail.name}
                    className="cocktail-card"
                  >
                    <h3>
                      <span>🍹 {cocktail.name}</span>
                      <span className="cocktail-price">{cocktail.preis.toFixed(2)} €</span>
                    </h3>
                    <div className="match-label">{getMatchLabel(index)}</div>
                    <div className="cocktail-meta">
                      {highlights.map((value) => (
                        <span key={`${cocktail.name}-${value}`}>{value}</span>
                      ))}
                    </div>
                    <div className="cocktail-card-actions">
                      <button
                        type="button"
                        className="recipe-toggle"
                        onClick={() =>
                          setExpandedCocktails((prev) => ({
                            ...prev,
                            [cocktail.name]: !prev[cocktail.name],
                          }))
                        }
                      >
                        {isExpanded ? "Rezept ausblenden" : "Rezept ansehen"}
                      </button>
                      <button
                        type="button"
                        className="cart-add-button"
                        onClick={() => addToCart(cocktail)}
                        disabled={orderingBlocked && isAlcoholicCocktail(cocktail)}
                        title={orderingBlocked && isAlcoholicCocktail(cocktail)
                          ? "Alkoholische Bestellungen sind durch das Barpersonal gesperrt."
                          : undefined}
                      >
                        {orderingBlocked && isAlcoholicCocktail(cocktail)
                          ? "Bestellung gesperrt"
                          : "Zum Warenkorb hinzufügen"}
                        {quantityInCart > 0 && <span>{quantityInCart}</span>}
                      </button>
                    </div>
                    {isExpanded && (
                      <div className="recipe-details">
                        <div>
                          <strong>Spirituosen</strong>
                          <p>{cocktail.spirituose?.length ? cocktail.spirituose.join(", ") : "Keine (alkoholfrei)"}</p>
                        </div>
                        <div>
                          <strong>Zutaten</strong>
                          <p>{cocktail.zutaten.join(", ")}</p>
                        </div>
                        <div>
                          <strong>Beschreibung</strong>
                          <p>{cocktail.beschreibung}</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="no-cocktails-placeholder">
              <span style={{ fontSize: "36px", marginBottom: "12px" }}>🔍</span>
              <p style={{ fontWeight: "600", marginBottom: "4px", color: "#fff" }}>Erzähl CocktailGPT, worauf du Lust hast.</p>
              <p style={{ fontSize: "12px" }}>
                Deine Empfehlungen erscheinen dann hier.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
