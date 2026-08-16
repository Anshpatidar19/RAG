import { useState } from "react";
import DocumentUpload from "./DocumentUpload";
import DocumentList from "./DocumentList";
import AddUrl from "./AddUrl";

export default function Sidebar({
  conversations,
  activeId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
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
          <div
            key={c.id}
            className={`conversation-row ${c.id === activeId ? "active" : ""}`}
          >
            <button className="conversation-item" onClick={() => onSelectChat(c.id)}>
              {c.title}
            </button>
            <button
              className="conversation-delete"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteChat(c.id);
              }}
              title="Delete chat"
            >
              ✕
            </button>
          </div>
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
            <AddUrl onAdded={onDocsChanged} />
            {documentsError && <p className="document-list-error">{documentsError}</p>}
            <DocumentList documents={documents} onChanged={onDocsChanged} />
          </div>
        )}
      </div>
    </aside>
  );
}