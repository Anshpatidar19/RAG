import { useState } from "react";
import DocumentUpload from "./DocumentUpload";
import DocumentList from "./DocumentList";

export default function Sidebar({
  conversations,
  activeId,
  onNewChat,
  onSelectChat,
  documents,
  documentsError,
  onDocsChanged,
}) {
  const [kbOpen, setKbOpen] = useState(true);

  return (
    <aside className="sidebar">
      <button className="new-chat-button" onClick={onNewChat}>
        + New chat
      </button>

      <div className="conversation-list">
        {conversations.length === 0 && (
          <p className="conversation-list-empty">No chats yet.</p>
        )}
        {conversations.map((c) => (
          <button
            key={c.id}
            className={`conversation-item ${c.id === activeId ? "active" : ""}`}
            onClick={() => onSelectChat(c.id)}
          >
            {c.title}
          </button>
        ))}
      </div>

      <div className="sidebar-kb">
        <button className="sidebar-kb-toggle" onClick={() => setKbOpen((v) => !v)}>
          <span>Knowledge base</span>
          <span>{kbOpen ? "−" : "+"}</span>
        </button>
        {kbOpen && (
          <div className="sidebar-kb-body">
            <DocumentUpload onUploaded={onDocsChanged} />
            {documentsError && <p className="document-list-error">{documentsError}</p>}
            <DocumentList documents={documents} onChanged={onDocsChanged} />
          </div>
        )}
      </div>
    </aside>
  );
}