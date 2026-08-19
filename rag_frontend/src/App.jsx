import { useCallback, useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import {
  listDocuments,
  listConversations,
  getConversationMessages,
  deleteConversation,
} from "./api/client";
import "./App.css";

const PLACEHOLDER_PREFIX = "local-";

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

function stripPlaceholders(list) {
  return list.filter((c) => !c.id.startsWith(PLACEHOLDER_PREFIX));
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
      setConversations((prev) => {
        // Keep any pending placeholder (it doesn't exist on the backend
        // yet, so a plain refresh would otherwise wipe it out).
        const placeholder = prev.find((c) => c.id.startsWith(PLACEHOLDER_PREFIX));
        return placeholder ? [placeholder, ...convos] : convos;
      });
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

  function handleNewChat() {
    // Show an instant local placeholder so the sidebar doesn't look
    // empty/broken — it isn't saved to the backend yet. The moment the
    // first message actually creates a real conversation server-side,
    // handleConversationCreated swaps this placeholder out for the real,
    // properly-titled one.
    const placeholderId = `${PLACEHOLDER_PREFIX}${Date.now()}`;
    setConversations((prev) => [
      { id: placeholderId, title: "New chat" },
      ...stripPlaceholders(prev),
    ]);
    setActiveId(placeholderId);
    setActiveMessages([]);
  }

  function handleSelectChat(id) {
    // Switching to a different real chat abandons any unsent placeholder.
    setConversations((prev) => stripPlaceholders(prev));
    loadConversation(id);
  }

  async function handleDeleteChat(id) {
    if (id.startsWith(PLACEHOLDER_PREFIX)) {
      // Never sent a message in it — nothing exists on the backend to delete.
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (id === activeId) {
        setActiveId(null);
        setActiveMessages([]);
      }
      return;
    }
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
    setActiveId((currentActiveId) => {
      const wasPlaceholder =
        typeof currentActiveId === "string" &&
        currentActiveId.startsWith(PLACEHOLDER_PREFIX);
      if (wasPlaceholder || !currentActiveId) {
        setConversations((prev) => stripPlaceholders(prev));
        refreshConversations();
        return id;
      }
      return currentActiveId;
    });
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
          conversationId={activeId?.startsWith(PLACEHOLDER_PREFIX) ? null : activeId}
          messages={activeMessages}
          onNewMessage={handleNewMessage}
          onConversationCreated={handleConversationCreated}
        />
      </main>
    </div>
  );
}