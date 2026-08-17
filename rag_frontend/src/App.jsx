import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import {
  listDocuments,
  listConversations,
  createConversation,
  getConversationMessages,
  deleteConversation,
} from "./api/client";
import "./App.css";

function sourcesFromServer(sources) {
  return (sources || []).map((s) => ({
    type: s.type,
    label: s.label,
    url: s.url,
    page_number: s.page_number,
    score: s.score,
    text_snippet: s.text_snippet,
  }));
}

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [activeMessages, setActiveMessages] = useState([]);

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

  const refreshConversations = useCallback(async () => {
    try {
      const convos = await listConversations();
      setConversations(convos);
      return convos;
    } catch {
      return [];
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
    refreshConversations();
  }, [refreshDocuments, refreshConversations]);

  async function loadConversation(id) {
    setActiveId(id);
    try {
      const messages = await getConversationMessages(id);
      const paired = [];
      for (let i = 0; i < messages.length; i++) {
        if (messages[i].role === "user" && messages[i + 1]?.role === "assistant") {
          const assistantMsg = messages[i + 1];
          paired.push({
            question: messages[i].content,
            answer: assistantMsg.content,
            sources: sourcesFromServer(assistantMsg.sources),
            answerable: (assistantMsg.sources || []).length > 0 || true,
          });
          i++;
        }
      }
      setActiveMessages(paired);
    } catch {
      setActiveMessages([]);
    }
  }

  async function handleNewChat() {
    try {
      const convo = await createConversation("New chat");
      setConversations((prev) => [{ id: convo.id, title: convo.title }, ...prev]);
      setActiveId(convo.id);
      setActiveMessages([]);
    } catch {
      // If conversation creation fails, the first /ask call will create
      // one server-side anyway (conversation_id starts null).
      setActiveId(null);
      setActiveMessages([]);
    }
  }

  function handleSelectChat(id) {
    loadConversation(id);
  }

  async function handleDeleteChat(id) {
    try {
      await deleteConversation(id);
    } catch {
      // Even if the delete request fails, still remove it locally so the
      // UI doesn't get stuck — worst case it reappears on next refresh.
    }
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (id === activeId) {
      setActiveId(null);
      setActiveMessages([]);
    }
  }

  function handleNewMessage(turn) {
    setActiveMessages((prev) => [...prev, turn]);
  }

  function handleConversationCreated(id) {
    if (!activeId) {
      setActiveId(id);
      refreshConversations();
    }
  }

  return (
    <div className="app-shell">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
        documents={documents}
        documentsError={documentsError}
        onDocsChanged={refreshDocuments}
      />
      <main className="main-column">
        <header className="main-header">
          <span className="brand">▲ Agentic RAG</span>
        </header>
        <Chat
          conversationId={activeId}
          messages={activeMessages}
          onNewMessage={handleNewMessage}
          onConversationCreated={handleConversationCreated}
        />
      </main>
    </div>
  );
}