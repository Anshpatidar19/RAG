import { useCallback, useEffect, useState } from "react";
import DocumentUpload from "./components/DocumentUpload";
import DocumentList from "./components/DocumentList";
import Chat from "./components/Chat";
import { listDocuments } from "./api/client";
import "./App.css";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [loadError, setLoadError] = useState("");

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setLoadError("");
    } catch (err) {
      setLoadError(err.message);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  return (
    <div className="app">
      <h1>Universal RAG</h1>

      <section>
        <h2>Knowledge base</h2>
        <DocumentUpload onUploaded={refreshDocuments} />
        {loadError && <p className="document-list-error">{loadError}</p>}
        <DocumentList documents={documents} onChanged={refreshDocuments} />
      </section>

      <section>
        <h2>Ask a question</h2>
        <Chat />
      </section>
    </div>
  );
}