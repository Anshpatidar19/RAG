import { useRef, useState } from "react";
import { deleteDocument, updateDocument } from "../api/client";

const API_BASE = "http://127.0.0.1:8000";

export default function DocumentList({ documents, onChanged }) {
  const [pendingId, setPendingId] = useState(null);
  const [error, setError] = useState("");
  const updateInputRef = useRef(null);
  const [updateTargetId, setUpdateTargetId] = useState(null);

  async function handleDelete(docId) {
    setPendingId(docId);
    setError("");
    try {
      await deleteDocument(docId);
      onChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setPendingId(null);
    }
  }

  function triggerUpdate(docId) {
    setUpdateTargetId(docId);
    updateInputRef.current?.click();
  }

  async function handleUpdateFileChosen(event) {
    const file = event.target.files?.[0];
    if (!file || !updateTargetId) return;

    setPendingId(updateTargetId);
    setError("");
    try {
      await updateDocument(updateTargetId, file);
      onChanged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setPendingId(null);
      setUpdateTargetId(null);
      if (updateInputRef.current) updateInputRef.current.value = "";
    }
  }

  if (documents.length === 0) {
    return <p className="document-list-empty">No documents in the knowledge base yet.</p>;
  }

  return (
    <div className="document-list">
      <input
        ref={updateInputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md"
        onChange={handleUpdateFileChosen}
        hidden
      />
      {error && <p className="document-list-error">{error}</p>}
      <ul>
        {documents.map((doc) => (
          <li key={doc.doc_id} className="document-list-item">
            <div className="document-info">
              <a
                href={`${API_BASE}${doc.file_url}`}
                target="_blank"
                rel="noopener noreferrer"
                className="document-name document-name-link"
                title="Open this file"
              >
                {doc.filename}
              </a>
              <span className="document-meta">
                {doc.status} · {doc.num_chunks} chunks
              </span>
            </div>
            <div className="document-actions">
              <button
                onClick={() => triggerUpdate(doc.doc_id)}
                disabled={pendingId === doc.doc_id}
              >
                Update
              </button>
              <button
                onClick={() => handleDelete(doc.doc_id)}
                disabled={pendingId === doc.doc_id}
              >
                {pendingId === doc.doc_id ? "..." : "Delete"}
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}