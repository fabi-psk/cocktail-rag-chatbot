import { useState } from "react";

function App() {
  const [spirituose, setSpirituose] = useState("");
  const [cocktails, setCocktails] = useState([]);

  const sucheCocktails = async () => {
    try {
      const response = await fetch(
          `http://127.0.0.1:8000/search?spirituose=${spirituose}`
      );

      const data = await response.json();

      console.log("Komplette Antwort:");
      console.log(data);

      setCocktails(data.cocktails);
    } catch (error) {
      console.error("Fehler:", error);
    }
  };

  return (
      <div style={{ padding: "20px" }}>
        <h1
            style={{
              marginBottom: "50px",
              marginLeft: "-60px"
            }}
        >
          🍹 CocktailGPT
        </h1>

          <h2
              style={{
                  marginBottom: "10px"
              }}
          >
              Wie ist dein Geschmack?
          </h2>

          <p
              style={{
                  color: "#b0b0b0",
                  marginBottom: "25px"
              }}
          >
              Beschreibe deinen Wunschcocktail und CocktailGPT
              findet passende Vorschläge.
          </p>

        <input
            type="text"
            placeholder="Wonach ist dir heute?"
            value={spirituose}
            onChange={(e) => setSpirituose(e.target.value)}
            style={{
                width: "500px",
                padding: "16px",
                fontSize: "18px",
                borderRadius: "14px",
                border: "2px solid #8b5cf6",
                outline: "none",
                marginRight: "10px",
                backgroundColor: "#1e1e2f",
                color: "white",
                boxShadow: "0 0 15px rgba(139, 92, 246, 0.4)"
            }}
        />

          <button
              onClick={sucheCocktails}
              style={{
                  padding: "16px 24px",
                  fontSize: "16px",
                  borderRadius: "14px",
                  border: "none",
                  backgroundColor: "#8b5cf6",
                  color: "white",
                  cursor: "pointer",
                  fontWeight: "bold"
              }}
          >
              Suchen
          </button>



          <div
              style={{
                  marginTop: "30px",
                  color: "#a0a0a0"
              }}
          >
              <div
                  style={{
                      display: "flex",
                      justifyContent: "center",
                      gap: "12px",
                      flexWrap: "wrap",
                      marginTop: "25px",
                  }}
              >
                  <div
                      onClick={() => setSpirituose("Klassisch")}
                      style={{
                          padding: "10px 16px",
                          border: "2px solid #3b82f6",
                          borderRadius: "20px",
                          cursor: "pointer",
                          color: "#3b82f6",
                      }}
                  >
                      🍸 Klassisch
                  </div>

                  <div
                      onClick={() => setSpirituose("Fruchtig")}
                      style={{
                          padding: "10px 16px",
                          border: "2px solid #ef4444",
                          borderRadius: "20px",
                          cursor: "pointer",
                          color: "#ef4444",
                      }}
                  >
                      🍓 Fruchtig
                  </div>

                  <div
                      onClick={() => setSpirituose("Cremig")}
                      style={{
                          padding: "10px 16px",
                          border: "2px solid #22c55e",
                          borderRadius: "20px",
                          cursor: "pointer",
                          color: "#22c55e",
                      }}
                  >
                      ☁️ Cremig
                  </div>

                  <div
                      onClick={() => setSpirituose("Stark")}
                      style={{
                          padding: "10px 16px",
                          border: "2px solid #f59e0b",
                          borderRadius: "20px",
                          cursor: "pointer",
                          color: "#f59e0b",
                      }}
                  >
                      💪 Stark
                  </div>
              </div>
          </div>

          <div
              style={{
                  marginTop: "35px",
                  textAlign: "center"
              }}
          >
              <h3
                  style={{
                      color: "#cfcfcf",
                      marginBottom: "15px"
                  }}
              >
                  Beispieleingaben
              </h3>

              <p>💬 Ich mag kein Kokos.</p>

              <p>💬 Ich liebe Vodka und Ananas.</p>

              <p>💬 Ich suche einen fruchtigen Cocktail für den Sommer.</p>

              <p>💬 Ich möchte etwas Starkes mit Rum.</p>
          </div>

          {cocktails.length > 0 && (
              <p
                  style={{
                      fontSize: "18px",
                      color: "#cfcfcf",
                      marginTop: "30px",
                      marginBottom: "20px"
                  }}
              >
                  🍹 {cocktails.length} passende Cocktails gefunden
              </p>
          )}

          <div
              style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "20px",
                  marginTop: "20px",
                  alignItems: "center"
              }}
          >
              {cocktails.map((cocktail) => (
                  <div
                      key={cocktail.name}
                      style={{
                          width: "500px",
                          backgroundColor: "#1e1e2f",
                          border: "1px solid #8b5cf6",
                          borderRadius: "16px",
                          padding: "20px",
                          boxShadow: "0 0 15px rgba(139, 92, 246, 0.25)",
                          textAlign: "left"
                      }}
                  >
                      <h3
                          style={{
                              marginTop: 0,
                              color: "#ffffff"
                          }}
                      >
                          🍹 {cocktail.name}
                      </h3>

                      <p>
                          <strong>Kategorie:</strong> {cocktail.kategorie}
                      </p>

                      <p>
                          <strong>Stärke:</strong> {cocktail.staerke}
                      </p>

                      <p>
                          <strong>Preis:</strong> {cocktail.preis} €
                      </p>

                      <p>
                          <strong>Spirituose:</strong>{" "}
                          {cocktail.spirituose.join(", ")}
                      </p>

                      <p>
                          <strong>Geschmack:</strong>{" "}
                          {cocktail.geschmack.join(", ")}
                      </p>

                      <p
                          style={{
                              color: "#cfcfcf"
                          }}
                      >
                          {cocktail.beschreibung}
                      </p>
                  </div>
              ))}
          </div>
      </div>
  );
}

export default App;
