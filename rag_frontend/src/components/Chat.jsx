import { useState, useRef, useEffect } from "react";
import { askQuestion } from "../api/client";

export default function Chat() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState([]); // { question, answer, sources, answerable }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, loading]);

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");
    try {
      const result = await askQuestion(trimmed);
      setHistory((prev) => [
        ...prev,
        {
          question: trimmed,
          answer: result.answer,
          sources: result.sources,
          answerable: result.answerable,
        },
      ]);
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-history">
        {history.length === 0 && (
          <p className="chat-empty">Ask a question about your uploaded documents.</p>
        )}
        {history.map((turn, i) => (
          <div key={i} className="chat-turn">
            <p className="chat-question">{turn.question}</p>
            <p className={`chat-answer ${!turn.answerable ? "chat-answer-empty" : ""}`}>
              {turn.answer}
            </p>
            {turn.answerable && turn.sources.length > 0 && (
              <details className="chat-sources">
                <summary>{turn.sources.length} source{turn.sources.length > 1 ? "s" : ""}</summary>
                <ul>
                  {turn.sources.map((source, j) => (
                    <li key={j} className="source-item">
                      <div className="source-header">
                        <span className="source-filename">{source.filename}</span>
                        {source.page_number !== null && (
                          <span className="source-page">p.{source.page_number}</span>
                        )}
                        <span className="source-score-bar" aria-hidden="true">
                          <span
                            className="source-score-fill"
                            style={{ width: `${Math.round(source.score * 100)}%` }}
                          />
                        </span>
                        <span className="source-score-value">
                          {source.score.toFixed(2)}
                        </span>
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
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documents..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </form>
    </div>
  );
}