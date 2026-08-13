import { useEffect, useRef, useState } from "react";
import { askQuestion, textToSpeech } from "../api/client";

const SpeechRecognitionAPI =
  typeof window !== "undefined"
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;

export default function Chat({ messages, onNewMessage }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [listening, setListening] = useState(false);
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const bottomRef = useRef(null);
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");
    try {
      const result = await askQuestion(trimmed);
      onNewMessage({
        question: trimmed,
        answer: result.answer,
        sources: result.sources,
        answerable: result.answerable,
      });
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function toggleListening() {
    if (!SpeechRecognitionAPI) {
      setError("Speech recognition isn't supported in this browser (try Chrome or Edge).");
      return;
    }

    if (listening) {
      recognitionRef.current?.stop();
      return;
    }

    const recognition = new SpeechRecognitionAPI();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setQuestion((prev) => (prev ? `${prev} ${transcript}` : transcript));
    };
    recognition.onerror = () => {
      setListening(false);
    };
    recognition.onend = () => {
      setListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  }

  async function handleSpeak(text, index) {
    if (speakingIndex === index) {
      audioRef.current?.pause();
      setSpeakingIndex(null);
      return;
    }
    try {
      setSpeakingIndex(index);
      const blob = await textToSpeech(text);
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => setSpeakingIndex(null);
      audio.onerror = () => setSpeakingIndex(null);
      await audio.play();
    } catch (err) {
      setError(err.message);
      setSpeakingIndex(null);
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-history">
        {messages.length === 0 && (
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
                onClick={() => handleSpeak(turn.answer, i)}
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
                        <span className="source-score-bar" aria-hidden="true">
                          <span
                            className="source-score-fill"
                            style={{ width: `${Math.round(source.score * 100)}%` }}
                          />
                        </span>
                        <span className="source-score-value">{source.score.toFixed(2)}</span>
                      </div>
                      <p className="source-snippet">{source.text_snippet}</p>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-turn">
            <p className="chat-question">{question}</p>
            <p className="chat-thinking">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <p className="chat-error">{error}</p>}

      <form onSubmit={handleSubmit} className="chat-input-row">
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