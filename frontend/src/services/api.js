const API_BASE_URL = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error("API request failed");
  }

  return response.json();
}

export function fetchCocktails() {
  return request("/cocktails");
}

export function fetchBackendStatus() {
  return request("/");
}

export function sendChatMessage({ sessionId, message, history, mode = "menu" }) {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      message,
      history,
      mode,
    }),
  });
}

export function resetChatSession(sessionId) {
  return request(`/chat/session/${sessionId}`, {
    method: "DELETE",
  });
}

export function removePreference({ sessionId, field, value }) {
  return request(`/chat/session/${sessionId}/preferences/remove`, {
    method: "POST",
    body: JSON.stringify({ field, value }),
  });
}
