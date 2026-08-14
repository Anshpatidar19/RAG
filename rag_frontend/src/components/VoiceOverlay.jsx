export default function VoiceOverlay({ phase, captionText, onClose, onTapOrb }) {
  return (
    <div className="voice-overlay">
      <button className="voice-close" onClick={onClose} title="Exit voice mode">
        ✕
      </button>

      <div className={`voice-orb-wrap phase-${phase}`} onClick={onTapOrb}>
        <span className="voice-ring ring-1" />
        <span className="voice-ring ring-2" />
        <span className="voice-ring ring-3" />
        <span className="voice-orb" />
      </div>

      <p className="voice-phase-label">
        {phase === "listening" && "Listening..."}
        {phase === "thinking" && "Thinking..."}
        {phase === "speaking" && "Speaking..."}
        {phase === "idle" && "Tap to talk"}
      </p>

      {captionText && <p className="voice-caption">{captionText}</p>}
    </div>
  );
}