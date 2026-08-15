import { useEffect, useRef, useState } from "react";
import { askQuestionStream } from "../api/client";
import VoiceOverlay from "./VoiceOverlay";

const SpeechRecognitionAPI =
  typeof window !== "undefined"
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;

const speechSynthesisAPI = typeof window !== "undefined" ? window.speechSynthesis : null;

function cleanForSpeech(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`(.*?)`/g, "$1")
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/^[-*+]\s+/gm, "")
    .replace(/[_~]/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

export default function Chat({ conversationId, messages, onNewMessage, onConversationCreated }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [listening, setListening] = useState(false);
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [streamingText, setStreamingText] = useState("");

  const [voiceOverlayOpen, setVoiceOverlayOpen] = useState(false);
  const [autoSpeaking, setAutoSpeaking] = useState(false);
  const [lastSpokenText, setLastSpokenText] = useState("");

  const bottomRef = useRef(null);
  const recognitionRef = useRef(null);
  const messagesRef = useRef(messages);
  const voiceOverlayOpenRef = useRef(false);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    voiceOverlayOpenRef.current = voiceOverlayOpen;
  }, [voiceOverlayOpen]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, streamingText]);

  useEffect(() => {
    return () => speechSynthesisAPI?.cancel();
  }, []);

  function speak(text, index) {
    if (!speechSynthesisAPI) return;
    speechSynthesisAPI.cancel();
    if (speakingIndex === index) {
      setSpeakingIndex(null);
      return;
    }
    const utterance = new SpeechSynthesisUtterance(cleanForSpeech(text));
    utterance.rate = 1.1;
    utterance.onend = () => setSpeakingIndex(null);
    utterance.onerror = () => setSpeakingIndex(null);
    setSpeakingIndex(index);
    speechSynthesisAPI.speak(utterance);
  }

  function startListening() {
    if (!SpeechRecognitionAPI) {
      setError("Speech recognition isn't supported in this browser (try Chrome or Edge).");
      return;
    }
    if (listening) return;

    const recognition = new SpeechRecognitionAPI();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      if (!finalTranscript) return;

      if (voiceOverlayOpenRef.current) {
        sendQuestion(finalTranscript.trim());
      } else {
        setQuestion((prev) => (prev ? `${prev} ${finalTranscript}`.trim() : finalTranscript.trim()));
      }
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  }

  function stopListening() {
    recognitionRef.current?.stop();
  }

  function toggleListening() {
    if (listening) stopListening();
    else startListening();
  }

  async function sendQuestion(text) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");
    setPendingQuestion(trimmed);
    setStatusMessage("Starting...");
    setStreamingText("");
    setQuestion("");

    let finalText = "";
    let finalSources = [];
    let finalAnswerable = true;

    const history = messagesRef.current.slice(-6).map((turn) => ({
      question: turn.question,
      answer: turn.answer,
    }));

    await askQuestionStream(trimmed, history, conversationId, {
      onStatus: (msg) => setStatusMessage(msg),
      onToken: (chunk) => {
        finalText += chunk;
        setStreamingText(finalText);
      },
      onSources: (sources, answerable) => {
        finalSources = sources;
        finalAnswerable = answerable;
      },
      onConversationId: (id) => {
        onConversationCreated?.(id);
      },
      onDone: () => {
        onNewMessage({
          question: trimmed,
          answer: finalText,
          sources: finalSources,
          answerable: finalAnswerable,
        });
        setStreamingText("");
        setStatusMessage("");
        setPendingQuestion("");
        setLoading(false);

        if (voiceOverlayOpenRef.current && finalText && speechSynthesisAPI) {
          const cleaned = cleanForSpeech(finalText);
          const utterance = new SpeechSynthesisUtterance(cleaned);
          utterance.rate = 1.1;
          setLastSpokenText(cleaned);
          setAutoSpeaking(true);
          utterance.onend = () => {
            setAutoSpeaking(false);
            // Continuous conversation: listen again automatically,
            // unless the overlay was closed while the answer was speaking.
            if (voiceOverlayOpenRef.current) startListening();
          };
          utterance.onerror = () => setAutoSpeaking(false);
          speechSynthesisAPI.cancel();
          speechSynthesisAPI.speak(utterance);
        }
      },
      onError: (msg) => {
        setError(msg);
        setStreamingText("");
        setStatusMessage("");
        setPendingQuestion("");
        setLoading(false);
      },
    });
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendQuestion(question);
  }

  function openVoiceOverlay() {
    setVoiceOverlayOpen(true);
    voiceOverlayOpenRef.current = true;
    startListening();
  }

  function closeVoiceOverlay() {
    setVoiceOverlayOpen(false);
    voiceOverlayOpenRef.current = false;
    stopListening();
    speechSynthesisAPI?.cancel();
    setAutoSpeaking(false);
  }

  function handleOrbTap() {
    if (loading || autoSpeaking) return;
    toggleListening();
  }

  // Determine the overlay's current visual phase
  let phase = "idle";
  if (listening) phase = "listening";
  else if (loading) phase = "thinking";
  else if (autoSpeaking) phase = "speaking";

  const captionText = autoSpeaking ? lastSpokenText : streamingText || pendingQuestion || "";

  return (
    <div className="chat-panel">
      {voiceOverlayOpen && (
        <VoiceOverlay
          phase={phase}
          captionText={captionText}
          onClose={closeVoiceOverlay}
          onTapOrb={handleOrbTap}
        />
      )}

      <div className="chat-history">
        {messages.length === 0 && !loading && (
          <p className="chat-empty">Ask a question about your uploaded documents.</p>
        )}
        {messages.map((turn, i) => (
          <div key={i} className="chat-turn">
            <p className="chat-question">{turn.question}</p>
            <div className="chat-answer-row">
              <p className={`chat-answer ${!turn.answerable ? "chat-answer-empty" : ""}`}>
                {turn.answer}
              </p>
              <button
                className={`speak-button ${speakingIndex === i ? "speaking" : ""}`}
                onClick={() => speak(turn.answer, i)}
                title="Listen to this answer"
                type="button"
              >
                {speakingIndex === i ? "■" : "🔊"}
              </button>
            </div>
            {turn.answerable && turn.sources.length > 0 && (
              <details className="chat-sources">
                <summary>
                  {turn.sources.length} source{turn.sources.length > 1 ? "s" : ""}
                </summary>
                <ul>
                  {turn.sources.map((source, j) => (
                    <li key={j} className="source-item">
                      <div className="source-header">
                        <span className={`source-type-badge source-type-${source.type}`}>
                          {source.type === "web" ? "web" : "doc"}
                        </span>
                        {source.url ? (
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-filename source-link"
                          >
                            {source.label}
                          </a>
                        ) : (
                          <span className="source-filename">{source.label}</span>
                        )}
                        {source.page_number !== null && source.page_number !== undefined && (
                          <span className="source-page">p.{source.page_number}</span>
                        )}
                        {source.type !== "web" && (
                          <>
                            <span className="source-score-bar" aria-hidden="true">
                              <span
                                className="source-score-fill"
                                style={{ width: `${Math.round(source.score * 100)}%` }}
                              />
                            </span>
                            <span className="source-score-value">{source.score.toFixed(2)}</span>
                          </>
                        )}
                      </div>
                      <p className="source-snippet">{source.text_snippet}</p>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}
        {loading && !voiceOverlayOpen && (
          <div className="chat-turn">
            <p className="chat-question">{pendingQuestion}</p>
            {streamingText ? (
              <p className="chat-answer">
                {streamingText}
                <span className="stream-cursor">▌</span>
              </p>
            ) : (
              <p className="chat-status">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
                <span className="chat-status-text">{statusMessage}</span>
              </p>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <p className="chat-error">{error}</p>}

      <form onSubmit={handleSubmit} className="chat-input-row">
        <button
          type="button"
          className="voice-mode-toggle"
          onClick={openVoiceOverlay}
          title="Open voice assistant"
        >
          🗣️
        </button>
        <button
          type="button"
          className={`mic-button ${listening ? "listening" : ""}`}
          onClick={toggleListening}
          title={listening ? "Stop listening" : "Ask by voice"}
        >
          {listening ? "●" : "🎤"}
        </button>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={listening ? "Listening..." : "Ask a question about your documents..."}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>
    </div>
  );
}