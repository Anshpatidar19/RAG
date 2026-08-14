import { useEffect, useRef, useState } from "react";
import { askQuestionStream, textToSpeech } from "../api/client";

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
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [streamingText, setStreamingText] = useState("");

  const bottomRef = useRef(null);
  const recognitionRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, streamingText]);

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = question.trim();
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

    const history = messages.slice(-6).map((turn) => ({
      question: turn.question,
      answer: turn.answer,
    }));

    await askQuestionStream(trimmed, history, {
      onStatus: (msg) => setStatusMessage(msg),
      onToken: (text) => {
        finalText += text;
        setStreamingText(finalText);
      },
      onSources: (sources, answerable) => {
        finalSources = sources;
        finalAnswerable = answerable;
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
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    // Only read NEW results starting at event.resultIndex — reading
    // results[0] unconditionally re-appends the first-ever recognized
    // phrase every time onresult fires again, which is what caused
    // the repeated/duplicated text.
    recognition.onresult = (event) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        }
      }
      if (finalTranscript) {
        setQuestion((prev) => (prev ? `${prev} ${finalTranscript}`.trim() : finalTranscript.trim()));
      }
    };
    recognition.onerror = () => setListening(false);
    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  }

  async function handleSpeak(text, index) {
    // Always stop whatever is currently playing first — clicking a
    // different answer's speaker while one is already playing used to
    // leave the old audio running in the background, which made it
    // look like playback "couldn't be stopped."
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }

    if (speakingIndex === index) {
      setSpeakingIndex(null);
      return;
    }

    try {
      setSpeakingIndex(index);
      const blob = await textToSpeech(text);
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.playbackRate = 1.25;
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
        {loading && (
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