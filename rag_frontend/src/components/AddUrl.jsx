import { useState } from "react";
import { ingestUrl } from "../api/client";

export default function AddUrl({ onAdded }) {
  const [url, setUrl] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [errorMessage, setErrorMessage] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = url.trim();
    if (!trimmed || status === "loading") return;

    setStatus("loading");
    setErrorMessage("");
    try {
      const document = await ingestUrl(trimmed);
      setUrl("");
      setStatus("idle");
      onAdded?.(document);
    } catch (err) {
      setStatus("error");
      setErrorMessage(err.message);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="add-url-form">
      <input
        type="url"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        placeholder="Paste a URL to add..."
        disabled={status === "loading"}
      />
      <button type="submit" disabled={status === "loading" || !url.trim()}>
        {status === "loading" ? "..." : "Add"}
      </button>
      {status === "error" && <p className="upload-error">{errorMessage}</p>}
    </form>
  );
}