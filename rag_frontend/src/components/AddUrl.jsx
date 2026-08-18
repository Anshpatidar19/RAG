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

    // Clear immediately, whether this succeeds or fails — leaving stale
    // text in the box let a later paste silently concatenate onto it
    // (e.g. "...wiki/Marvel" + "https://other-site.com/..." glued together
    // with no separator), which is what caused malformed URLs before.
    setUrl("");
    setStatus("loading");
    setErrorMessage("");
    try {
      const document = await ingestUrl(trimmed);
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