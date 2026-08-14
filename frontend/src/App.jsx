import { useState, useEffect, useMemo, useRef } from "react";
import "./App.css";
import { fetchBackendStatus, removePreference, resetChatSession, sendChatMessage } from "./services/api";

const createSessionId = () => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
};

const emptyPreferences = {
  liked_ingredients: [],
  disliked_ingredients: [],
  spirits: [],
  liked_flavors: [],
  disliked_flavors: [],
  strength: null,
  alcoholic: null,
};

const welcomeMessage = (mode) => mode === "home"
  ? "Willkommen im **Zuhause-Modus**. Nenne mir einen konkreten Cocktail, dann suche ich das Rezept auf einer vertrauenswürdigen Webseite und zeige dir Zutaten, Zubereitung und Quelle."
  : "Hallo! Ich bin **CocktailGPT**, dein persönlicher Barkeeper-Assistent. 🍹\n\nWelchen Cocktail suchst du heute oder nach welchen Geschmäckern steht dir der Sinn? Beschreibe einfach deinen Wunsch (z.B. *'Ich möchte einen fruchtigen Cocktail mit Rum'* oder *'Ich mag Gin, aber keinen Wodka und kein Kokos'*). Ich schlage dir passende Rezepte vor!";

const normalize = (value) => String(value || "").trim().toLowerCase();

const includesValue = (values, value) =>
  Array.isArray(values) && values.some((item) => normalize(item) === normalize(value));

const formatStrength = (value) => {
  if (value === "mild") return "mild";
  if (value === "hoch") return "stark";
  return value;
};

const buildPreferenceTags = (preferences) => {
  const prefs = preferences || emptyPreferences;
  const tags = [];
  const hasCoconutExclusion = [
    ...(prefs.disliked_ingredients || []),
    ...(prefs.disliked_flavors || []),
  ].some((value) => normalize(value).includes("kokos"));
  prefs.spirits?.forEach((value) => tags.push({ field: "spirits", value, label: value, type: "positive" }));
  prefs.liked_ingredients?.forEach((value) => tags.push({ field: "liked_ingredients", value, label: value, type: "positive" }));
  prefs.liked_flavors?.forEach((value) => tags.push({ field: "liked_flavors", value, label: value, type: "positive" }));
  if (hasCoconutExclusion) {
    tags.push({ field: "disliked_ingredients", value: "Kokos", label: "Kein Kokos", type: "negative" });
  }
  prefs.disliked_ingredients
    ?.filter((value) => !normalize(value).includes("kokos"))
    .forEach((value) => tags.push({ field: "disliked_ingredients", value, label: `Kein ${value}`, type: "negative" }));
  prefs.disliked_flavors
    ?.filter((value) => !normalize(value).includes("kokos"))
    .forEach((value) => tags.push({ field: "disliked_flavors", value, label: `Nicht ${value}`, type: "negative" }));
  if (prefs.strength) tags.push({ field: "strength", value: prefs.strength, label: `Staerke: ${formatStrength(prefs.strength)}`, type: "neutral" });
  if (prefs.alcoholic === false) tags.push({ field: "alcoholic", value: "false", label: "Alkoholfrei", type: "neutral" });
  if (prefs.alcoholic === true) tags.push({ field: "alcoholic", value: "true", label: "Mit Alkohol", type: "neutral" });
  return tags;
};

const cocktailTextValues = (cocktail) => [
  cocktail.name,
  cocktail.staerke,
  cocktail.beschreibung,
  ...(cocktail.spirituose || []),
  ...(cocktail.geschmack || []),
  ...(cocktail.zutaten || []),
].map(normalize);

const cocktailHasValue = (cocktail, value) =>
  cocktailTextValues(cocktail).some((item) => item.includes(normalize(value)) || normalize(value).includes(item));

const buildMatchReasons = (cocktail, preferences) => {
  const prefs = preferences || emptyPreferences;
  const reasons = [];
  prefs.spirits?.forEach((value) => {
    if (cocktailHasValue(cocktail, value)) reasons.push(`${value} entspricht deiner Vorliebe.`);
  });
  prefs.liked_flavors?.forEach((value) => {
    if (cocktailHasValue(cocktail, value)) reasons.push(`${value} passt zu deinem Geschmack.`);
  });
  prefs.disliked_ingredients?.forEach((value) => {
    if (!cocktailHasValue(cocktail, value)) reasons.push(`Kein ${value} enthalten.`);
  });
  prefs.disliked_flavors?.forEach((value) => {
    if (!cocktailHasValue(cocktail, value)) reasons.push(`Nicht ${value}.`);
  });
  if (prefs.strength && cocktail.staerke) reasons.push(`Staerke: ${cocktail.staerke}.`);
  return reasons.slice(0, 3);
};

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
  const [preferences, setPreferences] = useState(emptyPreferences);
  const [rouletteResult, setRouletteResult] = useState(null);
  const [expandedCocktails, setExpandedCocktails] = useState({});
  const [webRecipes, setWebRecipes] = useState([]);
  const [ollamaConnected, setOllamaConnected] = useState(null);

  const messagesEndRef = useRef(null);
  const sessionIdRef = useRef(createSessionId());
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

  const applyChatData = (data) => {
    setWebRecipes(data.web_recipes || []);
    setPreferences(data.preferences || emptyPreferences);
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

  const handleRemovePreference = async (tag) => {
    if (isLoading) return;
    setIsLoading(true);
    try {
      const data = await removePreference({
        sessionId: sessionIdRef.current,
        field: tag.field,
        value: tag.value,
      });
      applyChatData(data);
    } catch (error) {
      console.error("Fehler beim Entfernen der Praeferenz:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "CocktailGPT konnte den Wunsch gerade nicht entfernen. Versuch es bitte noch einmal.",
        },
      ]);
    } finally {
      setIsLoading(false);
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
    setPreferences(emptyPreferences);
    setRouletteResult(null);
    setExpandedCocktails({});
    setWebRecipes([]);
    setInputValue("");
    sessionIdRef.current = createSessionId();

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
    setPreferences(emptyPreferences);
    setRouletteResult(null);
    setExpandedCocktails({});
    setInputValue("");
    sessionIdRef.current = createSessionId();
    try {
      await resetChatSession(oldSessionId);
    } catch (error) {
      console.error("Fehler beim Wechseln des Modus:", error);
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
    { label: "🍓 Fruchtig", text: "Ich mag fruchtige Cocktails", color: "#ec4899", glow: "rgba(236, 72, 153, 0.3)", active: () => includesValue(preferences.liked_flavors, "fruchtig") },
    { label: "☁️ Cremig", text: "Ich mag cremige Cocktails", color: "#eab308", glow: "rgba(234, 179, 8, 0.3)", active: () => includesValue(preferences.liked_flavors, "cremig") },
    { label: "💪 Stark", text: "Ich möchte etwas Starkes", color: "#a855f7", glow: "rgba(168, 85, 247, 0.3)", active: () => ["stark", "hoch"].includes(preferences.strength) },
    { label: "🚫 Alkoholfrei", text: "Ich suche einen alkoholfreien Cocktail", color: "#06b6d4", glow: "rgba(6, 182, 212, 0.3)", active: () => preferences.alcoholic === false || preferences.strength === "alkoholfrei" },
    { label: "🍋 Sauer", text: "Ich mag saure Cocktails", color: "#eab308", glow: "rgba(234, 179, 8, 0.3)", active: () => includesValue(preferences.liked_flavors, "sauer") },
    { label: "🥥 Ohne Kokos", text: "Ich möchte keinen Kokos", color: "#f97316", glow: "rgba(249, 115, 22, 0.3)", active: () => (preferences.disliked_ingredients || []).some((value) => normalize(value).includes("kokos")) },
    { label: "🎲 Zufällig", text: "Schlage mir einen zufälligen Cocktail vor!", color: "#10b981", glow: "rgba(16, 185, 129, 0.3)", random: true },
    { label: "🧹 Reset", text: "RESET", color: "#ef4444", glow: "rgba(239, 68, 68, 0.3)" }
  ];
  const preferenceTags = buildPreferenceTags(preferences);
  const isRouletteRunning = rouletteResult && !rouletteResult.completed;

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
            {mode === "menu" && <div className="preferences-strip">
              <div className="preferences-title">Gemerkte Wünsche</div>
              {preferenceTags.length > 0 ? (
                <div className="preference-tags">
                  {preferenceTags.map((tag) => (
                    <button
                      key={`${tag.field}-${tag.value}`}
                      type="button"
                      className={`preference-tag ${tag.type}`}
                      onClick={() => handleRemovePreference(tag)}
                      disabled={isLoading || isRouletteRunning}
                      title={`${tag.label} entfernen`}
                    >
                      <span>{tag.label}</span>
                      <span aria-hidden="true">×</span>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="preferences-empty">keine</div>
              )}
            </div>}

            {/* Quick Suggestions Pills */}
            <div className="suggestions-bar">
              {(mode === "home" ? [
                { label: "Mojito", text: "Wie mache ich einen Mojito?", color: "#10b981", glow: "rgba(16, 185, 129, 0.3)" },
                { label: "Margarita", text: "Zeig mir das Rezept für eine Margarita", color: "#eab308", glow: "rgba(234, 179, 8, 0.3)" },
                { label: "Moscow Mule", text: "Welche Zutaten brauche ich für einen Moscow Mule?", color: "#06b6d4", glow: "rgba(6, 182, 212, 0.3)" },
                { label: "Zurücksetzen", text: "RESET", color: "#ef4444", glow: "rgba(239, 68, 68, 0.3)" },
              ] : suggestions).map((s, index) => {
                const isActive = s.active?.() || false;
                return (
                <button
                  key={index}
                  className={`suggestion-pill ${isActive ? "active" : ""}`}
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
                    borderColor: isActive ? s.color : `${s.color}35`,
                    backgroundColor: isActive ? `${s.color}28` : `${s.color}12`,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = s.color;
                    e.currentTarget.style.boxShadow = `0 0 12px ${s.glow}`;
                    e.currentTarget.style.transform = 'translateY(-1px) scale(1.03)';
                    e.currentTarget.style.backgroundColor = `${s.color}22`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = isActive ? s.color : `${s.color}35`;
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.backgroundColor = isActive ? `${s.color}28` : `${s.color}12`;
                  }}
                >
                  {s.label}
                </button>
                );
              })}
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
          <h2 className="panel-title">
            <span>{mode === "home" ? "🏠" : "📋"}</span>
            {mode === "home" ? ` Web-Rezept (${webRecipes.length})` : ` Rezeptekatalog (${retrievedCocktails.length})`}
          </h2>
          
          {mode === "home" ? (
            webRecipes.length > 0 ? (
              <div className="web-recipe-list">
                {webRecipes.map((recipe) => (
                  <article className="web-recipe-card" key={recipe.source_url}>
                    <div className="web-recipe-kicker">Direkt von der Quelle extrahiert</div>
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
                <p style={{ fontSize: "12px" }}>Das Testsystem extrahiert das Rezept direkt von der IBA-Webseite.</p>
              </div>
            )
          ) : rouletteResult && !rouletteResult.completed ? (
            <CocktailRoulette
              key={rouletteResult.id}
              cocktails={rouletteResult.rouletteCocktails}
              selectedCocktail={rouletteResult.selectedCocktail}
              onComplete={() => {
                setRetrievedCocktails([rouletteResult.selectedCocktail]);
                setRouletteResult((prev) => prev ? { ...prev, completed: true } : prev);
              }}
            />
          ) : retrievedCocktails.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              {rouletteResult?.completed && (
                <div className="roulette-winner-banner">
                  <span>🎉 Dein Cocktail ist:</span>
                  <strong>{rouletteResult.selectedCocktail.name}</strong>
                </div>
              )}
              <div style={{ fontSize: "12px", color: "var(--text-muted)", marginBottom: "12px" }}>
                Empfehlungen aus der Cocktailkarte passend zu den gemerkten Wünschen:
              </div>
              {retrievedCocktails.map((cocktail, index) => {
                const isExpanded = expandedCocktails[cocktail.name] || false;
                const reasons = buildMatchReasons(cocktail, preferences);
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
                    {reasons.length > 0 && (
                      <div className="match-reasons">
                        <strong>Warum passt er?</strong>
                        {reasons.map((reason) => (
                          <div key={`${cocktail.name}-${reason}`}>{reason}</div>
                        ))}
                      </div>
                    )}
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
