import { useRef, useState } from "react";
import { uploadDocument } from "../api/client";

export default function DocumentUpload({ onUploaded }) {
  const [status, setStatus] = useState("idle"); // idle | uploading | error
  const [errorMessage, setErrorMessage] = useState("");
  const inputRef = useRef(null);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setStatus("uploading");
    setErrorMessage("");

    try {
      const document = await uploadDocument(file);
      setStatus("idle");
      onUploaded?.(document);
    } catch (err) {
      setStatus("error");
      setErrorMessage(err.message);
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="upload-panel">
      <label className="upload-button">
        {status === "uploading" ? "Uploading..." : "Upload document"}
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md,.png,.jpg,.jpeg,.webp"
          onChange={handleFileChange}
          disabled={status === "uploading"}
          hidden
        />
      </label>
      {status === "error" && (
        <p className="upload-error">Upload failed: {errorMessage}</p>
      )}
    </div>
  );
}