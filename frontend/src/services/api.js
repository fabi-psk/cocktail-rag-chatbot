const API_BASE_URL = import.meta.env.VITE_API_BASE_URL
  || `${window.location.protocol}//${window.location.hostname}:8000`;

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

export function fetchOrderingStatus(sessionId) {
  return request(`/chat/session/${sessionId}/ordering-status`);
}

export function unlockOrdering(sessionId, password) {
  return request(`/chat/session/${sessionId}/unlock-ordering`, {
    method: "POST",
    body: JSON.stringify({ password }),
  });
}
