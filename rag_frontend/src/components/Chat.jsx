import { useEffect, useRef, useState } from "react";
import { askQuestionStream } from "../api/client";
import VoiceOverlay from "./VoiceOverlay";

const SpeechRecognitionAPI =
  typeof window !== "undefined"
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null;

const speechSynthesisAPI =
  typeof window !== "undefined"
    ? window.speechSynthesis
    : null;


/* ---------------------------------------------------------------------------
   Speech cleanup
--------------------------------------------------------------------------- */

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


/* ---------------------------------------------------------------------------
   Email draft detection
--------------------------------------------------------------------------- */

/*
 * EmailWriterAgent currently returns something like:
 *
 * Subject: Absence Notification - [Your Name]
 *
 * Dear [Manager Name],
 *
 * Please accept this email...
 *
 * We detect that structure and extract:
 *
 * {
 *   subject: "...",
 *   body: "..."
 * }
 */
function parseEmailDraft(text) {
  if (!text) {
    return null;
  }

  const subjectMatch = text.match(
    /^Subject:\s*(.+?)(?:\r?\n\r?\n|\r?\n)/i
  );

  if (!subjectMatch) {
    return null;
  }

  const subject = subjectMatch[1].trim();

  const body = text
    .slice(subjectMatch[0].length)
    .trim();

  if (!subject || !body) {
    return null;
  }

  return {
    subject,
    body,
  };
}


/* ---------------------------------------------------------------------------
   Extract recipient email from user's request
--------------------------------------------------------------------------- */

function extractEmailAddress(text) {
  if (!text) {
    return "";
  }

  const match = text.match(
    /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i
  );

  return match ? match[0] : "";
}


/* ---------------------------------------------------------------------------
   Chat component
--------------------------------------------------------------------------- */

export default function Chat({
  conversationId,
  messages,
  onNewMessage,
  onConversationCreated,
}) {
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

  /*
   * -------------------------------------------------------------------------
   * Email state
   * -------------------------------------------------------------------------
   *
   * emailDraft:
   *
   * {
   *   subject: "...",
   *   body: "...",
   *   messageIndex: 0
   * }
   *
   * messageIndex tells us which answer should display
   * the "Open in Gmail" button.
   */

  const [emailDraft, setEmailDraft] = useState(null);
  const [emailRecipient, setEmailRecipient] = useState("");

  const bottomRef = useRef(null);
  const recognitionRef = useRef(null);
  const messagesRef = useRef(messages);
  const voiceOverlayOpenRef = useRef(false);


  /* -------------------------------------------------------------------------
     Keep latest messages available inside callbacks
  ------------------------------------------------------------------------- */

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);


  /* -------------------------------------------------------------------------
     Voice overlay ref
  ------------------------------------------------------------------------- */

  useEffect(() => {
    voiceOverlayOpenRef.current = voiceOverlayOpen;
  }, [voiceOverlayOpen]);


  /* -------------------------------------------------------------------------
     Auto-scroll chat
  ------------------------------------------------------------------------- */

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading, streamingText]);


  /* -------------------------------------------------------------------------
     Cleanup speech when component unmounts
  ------------------------------------------------------------------------- */

  useEffect(() => {
    return () => {
      speechSynthesisAPI?.cancel();
    };
  }, []);


  /* =========================================================================
     TEXT TO SPEECH
  ========================================================================= */

  function speak(text, index) {
    if (!speechSynthesisAPI) {
      return;
    }

    speechSynthesisAPI.cancel();

    if (speakingIndex === index) {
      setSpeakingIndex(null);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(
      cleanForSpeech(text)
    );

    utterance.rate = 1.1;

    utterance.onend = () => {
      setSpeakingIndex(null);
    };

    utterance.onerror = () => {
      setSpeakingIndex(null);
    };

    setSpeakingIndex(index);

    speechSynthesisAPI.speak(utterance);
  }


  /* =========================================================================
     SPEECH RECOGNITION
  ========================================================================= */

  function startListening() {
    if (!SpeechRecognitionAPI) {
      setError(
        "Speech recognition isn't supported in this browser (try Chrome or Edge)."
      );

      return;
    }

    if (listening) {
      return;
    }

    const recognition = new SpeechRecognitionAPI();

    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      let finalTranscript = "";

      for (
        let i = event.resultIndex;
        i < event.results.length;
        i++
      ) {
        if (event.results[i].isFinal) {
          finalTranscript +=
            event.results[i][0].transcript;
        }
      }

      if (!finalTranscript) {
        return;
      }

      if (voiceOverlayOpenRef.current) {
        sendQuestion(finalTranscript.trim());
      } else {
        setQuestion((prev) =>
          prev
            ? `${prev} ${finalTranscript}`.trim()
            : finalTranscript.trim()
        );
      }
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


  function stopListening() {
    recognitionRef.current?.stop();
  }


  function toggleListening() {
    if (listening) {
      stopListening();
    } else {
      startListening();
    }
  }


  /* =========================================================================
     OPEN EMAIL IN GMAIL
  ========================================================================= */

  function openInGmail() {
    if (!emailDraft) {
      return;
    }

    /*
     * Recipient can be empty.
     *
     * Gmail will still open the compose window with:
     * - Subject
     * - Body
     *
     * already filled.
     */

    const recipient = emailRecipient.trim();

    const params = new URLSearchParams({
      view: "cm",
      fs: "1",
      su: emailDraft.subject,
      body: emailDraft.body,
    });

    if (recipient) {
      params.set("to", recipient);
    }

    const gmailComposeUrl =
      `https://mail.google.com/mail/?${params.toString()}`;

    window.open(
      gmailComposeUrl,
      "_blank",
      "noopener,noreferrer"
    );
  }


  /* =========================================================================
     SEND QUESTION
  ========================================================================= */

  async function sendQuestion(text) {
    const trimmed = text.trim();

    if (!trimmed || loading) {
      return;
    }

    /*
     * Remove previous email button when a new question starts.
     */
    setEmailDraft(null);
    setEmailRecipient("");

    setLoading(true);
    setError("");
    setPendingQuestion(trimmed);
    setStatusMessage("Starting...");
    setStreamingText("");
    setQuestion("");

    let finalText = "";
    let finalSources = [];
    let finalAnswerable = true;

    /*
     * Use the last 6 conversation turns as history.
     */
    const history = messagesRef.current
      .slice(-6)
      .map((turn) => ({
        question: turn.question,
        answer: turn.answer,
      }));

    await askQuestionStream(
      trimmed,
      history,
      conversationId,
      {
        /* -------------------------------------------------------------------
           Status event
        ------------------------------------------------------------------- */

        onStatus: (msg) => {
          setStatusMessage(msg);
        },


        /* -------------------------------------------------------------------
           Streaming token
        ------------------------------------------------------------------- */

        onToken: (chunk) => {
          finalText += chunk;

          setStreamingText(finalText);
        },


        /* -------------------------------------------------------------------
           Sources
        ------------------------------------------------------------------- */

        onSources: (sources, answerable) => {
          finalSources = sources;

          finalAnswerable = answerable;
        },


        /* -------------------------------------------------------------------
           Conversation ID
        ------------------------------------------------------------------- */

        onConversationId: (id) => {
          onConversationCreated?.(id);
        },


        /* -------------------------------------------------------------------
           Stream finished
        ------------------------------------------------------------------- */

        onDone: () => {
          /*
           * IMPORTANT:
           *
           * messagesRef.current.length is the index that this
           * new answer will receive after onNewMessage().
           *
           * Example:
           *
           * Existing messages = 3
           * New message index = 3
           */

          const newMessageIndex =
            messagesRef.current.length;


          /* -----------------------------------------------------------------
             Save normal chat message
          ----------------------------------------------------------------- */

          onNewMessage({
            question: trimmed,
            answer: finalText,
            sources: finalSources,
            answerable: finalAnswerable,
          });


          /* -----------------------------------------------------------------
             Detect Email Agent output
          ----------------------------------------------------------------- */

          const parsedEmail =
            parseEmailDraft(finalText);

          if (parsedEmail) {
            /*
             * Try to get recipient from the user's question.
             *
             * Example:
             *
             * "write an email to abc@gmail.com"
             *
             * → abc@gmail.com
             */

            const detectedRecipient =
              extractEmailAddress(trimmed);

            setEmailRecipient(
              detectedRecipient
            );

            /*
             * Store the answer index so the button
             * appears directly below this answer.
             */

            setEmailDraft({
              subject: parsedEmail.subject,
              body: parsedEmail.body,
              messageIndex: newMessageIndex,
            });
          }


          /* -----------------------------------------------------------------
             Reset loading state
          ----------------------------------------------------------------- */

          setStreamingText("");
          setStatusMessage("");
          setPendingQuestion("");
          setLoading(false);


          /* -----------------------------------------------------------------
             Voice output
          ----------------------------------------------------------------- */

          if (
            voiceOverlayOpenRef.current &&
            finalText &&
            speechSynthesisAPI
          ) {
            const cleaned =
              cleanForSpeech(finalText);

            const utterance =
              new SpeechSynthesisUtterance(
                cleaned
              );

            utterance.rate = 1.1;

            setLastSpokenText(cleaned);
            setAutoSpeaking(true);

            utterance.onend = () => {
              setAutoSpeaking(false);

              /*
               * Continue listening automatically
               * if voice overlay is still open.
               */

              if (
                voiceOverlayOpenRef.current
              ) {
                startListening();
              }
            };

            utterance.onerror = () => {
              setAutoSpeaking(false);
            };

            speechSynthesisAPI.cancel();

            speechSynthesisAPI.speak(
              utterance
            );
          }
        },


        /* -------------------------------------------------------------------
           Error
        ------------------------------------------------------------------- */

        onError: (msg) => {
          setError(msg);

          setStreamingText("");
          setStatusMessage("");
          setPendingQuestion("");
          setLoading(false);
        },
      }
    );
  }


  /* =========================================================================
     FORM SUBMIT
  ========================================================================= */

  function handleSubmit(event) {
    event.preventDefault();

    sendQuestion(question);
  }


  /* =========================================================================
     VOICE OVERLAY
  ========================================================================= */

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
    if (loading || autoSpeaking) {
      return;
    }

    toggleListening();
  }


  /* =========================================================================
     VOICE OVERLAY PHASE
  ========================================================================= */

  let phase = "idle";

  if (listening) {
    phase = "listening";
  } else if (loading) {
    phase = "thinking";
  } else if (autoSpeaking) {
    phase = "speaking";
  }


  const captionText =
    autoSpeaking
      ? lastSpokenText
      : streamingText ||
        pendingQuestion ||
        "";


  /* =========================================================================
     RENDER
  ========================================================================= */

  return (
    <div className="chat-panel">

      {/* ---------------------------------------------------------------------
          Voice Overlay
      --------------------------------------------------------------------- */}

      {voiceOverlayOpen && (
        <VoiceOverlay
          phase={phase}
          captionText={captionText}
          onClose={closeVoiceOverlay}
          onTapOrb={handleOrbTap}
        />
      )}


      {/* ---------------------------------------------------------------------
          Chat History
      --------------------------------------------------------------------- */}

      <div className="chat-history">

        {messages.length === 0 &&
          !loading && (
            <p className="chat-empty">
              Ask a question about your uploaded
              documents.
            </p>
          )}


        {messages.map((turn, i) => (
          <div
            key={i}
            className="chat-turn"
          >

            {/* ---------------------------------------------------------------
                User question
            --------------------------------------------------------------- */}

            <p className="chat-question">
              {turn.question}
            </p>


            {/* ---------------------------------------------------------------
                AI answer
            --------------------------------------------------------------- */}

            <div className="chat-answer-row">

              <p
                className={`chat-answer ${
                  !turn.answerable
                    ? "chat-answer-empty"
                    : ""
                }`}
              >
                {turn.answer}
              </p>


              {/* -------------------------------------------------------------
                  Speak answer
              ------------------------------------------------------------- */}

              <button
                className={`speak-button ${
                  speakingIndex === i
                    ? "speaking"
                    : ""
                }`}
                onClick={() =>
                  speak(
                    turn.answer,
                    i
                  )
                }
                title="Listen to this answer"
                type="button"
              >
                {speakingIndex === i
                  ? "■"
                  : "🔊"}
              </button>

            </div>


            {/* ===============================================================
                OPEN IN GMAIL BUTTON
                ===============================================================

                This appears ONLY below the Email Agent's answer.

                No email card.
                No cancel button.
                No SMTP.
                =============================================================== */}

            {emailDraft &&
              emailDraft.messageIndex === i && (

                <div className="email-answer-actions">

                  <button
                    type="button"
                    onClick={openInGmail}
                  >
                    Open in Gmail
                  </button>

                </div>

              )}


            {/* ---------------------------------------------------------------
                Sources
            --------------------------------------------------------------- */}

            {turn.answerable &&
              turn.sources.length > 0 && (

                <details className="chat-sources">

                  <summary>
                    {turn.sources.length}{" "}
                    source
                    {turn.sources.length > 1
                      ? "s"
                      : ""}
                  </summary>


                  <ul>

                    {turn.sources.map(
                      (source, j) => (

                        <li
                          key={j}
                          className="source-item"
                        >

                          <div className="source-header">

                            <span
                              className={`source-type-badge source-type-${source.type}`}
                            >
                              {source.type ===
                              "web"
                                ? "web"
                                : "doc"}
                            </span>


                            {source.url ? (

                              <a
                                href={
                                  source.url
                                }
                                target="_blank"
                                rel="noopener noreferrer"
                                className="source-filename source-link"
                              >
                                {source.label}
                              </a>

                            ) : (

                              <span className="source-filename">
                                {
                                  source.label
                                }
                              </span>

                            )}


                            {source.page_number !==
                              null &&
                              source.page_number !==
                                undefined && (

                                <span className="source-page">
                                  p.
                                  {
                                    source.page_number
                                  }
                                </span>

                              )}


                            {source.type !==
                              "web" && (

                              <>

                                <span
                                  className="source-score-bar"
                                  aria-hidden="true"
                                >

                                  <span
                                    className="source-score-fill"
                                    style={{
                                      width: `${Math.round(
                                        source.score *
                                          100
                                      )}%`,
                                    }}
                                  />

                                </span>


                                <span className="source-score-value">
                                  {source.score.toFixed(
                                    2
                                  )}
                                </span>

                              </>

                            )}

                          </div>


                          <p className="source-snippet">
                            {
                              source.text_snippet
                            }
                          </p>

                        </li>

                      )
                    )}

                  </ul>

                </details>

              )}

          </div>
        ))}


        {/* -------------------------------------------------------------------
            Streaming response
        ------------------------------------------------------------------- */}

        {loading &&
          !voiceOverlayOpen && (

            <div className="chat-turn">

              <p className="chat-question">
                {pendingQuestion}
              </p>


              {streamingText ? (

                <p className="chat-answer">

                  {streamingText}

                  <span className="stream-cursor">
                    ▌
                  </span>

                </p>

              ) : (

                <p className="chat-status">

                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />

                  <span className="chat-status-text">
                    {statusMessage}
                  </span>

                </p>

              )}

            </div>

          )}


        <div ref={bottomRef} />

      </div>


      {/* ---------------------------------------------------------------------
          Error
      --------------------------------------------------------------------- */}

      {error && (
        <p className="chat-error">
          {error}
        </p>
      )}


      {/* ---------------------------------------------------------------------
          Chat input
      --------------------------------------------------------------------- */}

      <form
        onSubmit={handleSubmit}
        className="chat-input-row"
      >

        {/* Voice assistant */}

        <button
          type="button"
          className="voice-mode-toggle"
          onClick={openVoiceOverlay}
          title="Open voice assistant"
        >
          🗣️
        </button>


        {/* Microphone */}

        <button
          type="button"
          className={`mic-button ${
            listening
              ? "listening"
              : ""
          }`}
          onClick={toggleListening}
          title={
            listening
              ? "Stop listening"
              : "Ask by voice"
          }
        >
          {listening
            ? "●"
            : "🎤"}
        </button>


        {/* Question input */}

        <input
          type="text"
          value={question}
          onChange={(event) =>
            setQuestion(
              event.target.value
            )
          }
          placeholder={
            listening
              ? "Listening..."
              : "Ask a question about your documents..."
          }
          disabled={loading}
        />


        {/* Ask */}

        <button
          type="submit"
          disabled={
            loading ||
            !question.trim()
          }
        >
          {loading
            ? "Thinking…"
            : "Ask"}
        </button>

      </form>

    </div>
  );
}