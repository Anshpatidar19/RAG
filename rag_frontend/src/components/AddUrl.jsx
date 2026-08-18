import { useState } from "react";
import { ingestUrl } from "../api/client";

export default function AddUrl({ onAdded }) {
  const [url, setUrl] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [errorMessage, setErrorMessage] = useState("");
  const [pendingUrl, setPendingUrl] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = url.trim();
    if (!trimmed || status === "loading") return;

    // Clear the input immediately so a later paste can't silently
    // concatenate onto stale text — but keep the URL around separately
    // so we can still show "Adding <url>..." while it processes.
    setUrl("");
    setPendingUrl(trimmed);
    setStatus("loading");
    setErrorMessage("");
    try {
      const document = await ingestUrl(trimmed);
      setStatus("idle");
      setPendingUrl("");
      onAdded?.(document);
    } catch (err) {
      setStatus("error");
      setErrorMessage(err.message);
      setPendingUrl("");
    }
  }

  return (
    <div className="add-url-wrap">
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
      </form>
      {status === "loading" && pendingUrl && (
        <p className="add-url-status">Adding {pendingUrl}…</p>
      )}
      {status === "error" && <p className="upload-error">{errorMessage}</p>}
    </div>
  );
}