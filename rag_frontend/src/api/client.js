const BASE_URL = "http://127.0.0.1:8000";

async function handleResponse(response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const message = body.detail
      ? typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail)
      : `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return response.json();
}

export async function listDocuments() {
  const res = await fetch(`${BASE_URL}/documents`);
  return handleResponse(res);
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/documents`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(res);
}

export async function updateDocument(docId, file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/documents/${docId}`, {
    method: "PUT",
    body: formData,
  });
  return handleResponse(res);
}

export async function deleteDocument(docId) {
  const res = await fetch(`${BASE_URL}/documents/${docId}`, {
    method: "DELETE",
  });
  return handleResponse(res);
}

export async function askQuestion(question) {
  const res = await fetch(`${BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return handleResponse(res);
}

export async function askQuestionStream(
  question,
  history,
  { onStatus, onToken, onSources, onDone, onError }
) {
  try {
    const res = await fetch(`${BASE_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    if (!res.ok || !res.body) {
      throw new Error(`Request failed with status ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;
        const event = JSON.parse(jsonStr);
        if (event.type === "status") onStatus?.(event.message);
        else if (event.type === "token") onToken?.(event.text);
        else if (event.type === "sources") onSources?.(event.sources, event.answerable);
        else if (event.type === "done") onDone?.();
      }
    }
  } catch (err) {
    onError?.(err.message);
  }
}

export async function textToSpeech(text) {
  const res = await fetch(`${BASE_URL}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Text-to-speech failed");
  }
  return res.blob();
}