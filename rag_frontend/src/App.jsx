import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import { listDocuments } from "./api/client";
import "./App.css";

function makeConversation() {
  return { id: crypto.randomUUID(), title: "New chat", messages: [] };
}

function titleFromQuestion(question) {
  const trimmed = question.trim();
  return trimmed.length > 40 ? trimmed.slice(0, 40) + "…" : trimmed;
}

export default function App() {
  const [conversations, setConversations] = useState([makeConversation()]);
  const [activeId, setActiveId] = useState(conversations[0].id);

  const [documents, setDocuments] = useState([]);
  const [documentsError, setDocumentsError] = useState("");

  const refreshDocuments = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setDocumentsError("");
    } catch (err) {
      setDocumentsError(err.message);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  function handleNewChat() {
    const fresh = makeConversation();
    setConversations((prev) => [fresh, ...prev]);
    setActiveId(fresh.id);
  }

  function handleSelectChat(id) {
    setActiveId(id);
  }

  function handleNewMessage(turn) {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== activeId) return c;
        const isFirstMessage = c.messages.length === 0;
        return {
          ...c,
          title: isFirstMessage ? titleFromQuestion(turn.question) : c.title,
          messages: [...c.messages, turn],
        };
      })
    );
  }

  const activeConversation = conversations.find((c) => c.id === activeId);

  return (
    <div className="app-shell">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        documents={documents}
        documentsError={documentsError}
        onDocsChanged={refreshDocuments}
      />
      <main className="main-column">
        <header className="main-header">
          <span className="brand"> 🤖 AGENTIC RAG</span>
        </header>
        <Chat messages={activeConversation.messages} onNewMessage={handleNewMessage} />
      </main>
    </div>
  );
}