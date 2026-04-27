import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
} from 'react'

const INTRO_TOPICS = ['climate', 'immigration', 'minimum wage'] as const
const INTRO_CLAIMS_BY_TOPIC: Record<typeof INTRO_TOPICS[number], readonly string[]> = {
  climate: ['cut emissions', 'expand clean energy', 'hold polluters accountable'],
  immigration: ['protect asylum rights', 'expand legal pathways', 'support new arrivals'],
  'minimum wage': ['wages should rise', 'pay should track inflation', 'work should pay enough'],
}

type IntroPhase = 'topic' | 'claim' | 'done' | 'voice'
type VoiceMode = 'stance' | 'essay'
type EssaySource = 'paste' | 'envelope'

type RetrievalModelLabel = 'Lexical (TF·IDF)' | 'Semantic (SVD)' | 'Enhanced (MiniLM)'
type StanceLabel = 'NLI · red pencil' | 'LLM · lamplight'

export type LandingFlowProps = {
  topic: string
  opinion: string
  essayText: string
  importedPdfName: string | null
  isImportingPdf: boolean
  loading: boolean
  onTopicChange: (value: string) => void
  onOpinionChange: (value: string) => void
  onEssayTextChange: (value: string) => void
  onImportPdf: (event: ChangeEvent<HTMLInputElement>) => void
  onSubmitStance: () => void
  onSubmitEssayDraft: () => void
  onOpenSettings: () => void
  onOpenAbout: () => void
  retrievalModelLabel: RetrievalModelLabel
  stanceLabel: StanceLabel
  rerankModeLabel: string
  chunksLabel: string
}

function useTypewriterCycle(items: readonly string[], active: boolean): { value: string; done: boolean } {
  const [value, setValue] = useState('')
  const [done, setDone] = useState(false)

  useEffect(() => {
    if (!active) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const wait = (ms: number) => new Promise<void>((resolve) => {
      timer = setTimeout(resolve, ms)
    })

    const run = async () => {
      setDone(false)
      setValue('')
      for (let i = 0; i < items.length; i += 1) {
        const item = items[i]
        const last = i === items.length - 1
        for (let c = 1; c <= item.length; c += 1) {
          if (cancelled) return
          setValue(item.slice(0, c))
          await wait(55)
        }
        if (last) {
          await wait(480)
          if (cancelled) return
          setDone(true)
          return
        }
        await wait(280)
        for (let c = item.length - 1; c >= 0; c -= 1) {
          if (cancelled) return
          setValue(item.slice(0, c))
          await wait(32)
        }
        await wait(140)
      }
    }
    void run()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [items, active])

  return { value, done }
}

function RisingTypewriter(): JSX.Element {
  return (
    <svg width="800" height="300" viewBox="0 0 800 300">
      <g stroke="#1a1a1a" strokeWidth="1.4" fill="none" strokeLinecap="square">
        <rect x="40" y="0" width="720" height="22" rx="11" fill="#fafaf7" />
        <circle cx="40" cy="11" r="14" fill="#fafaf7" />
        <circle cx="760" cy="11" r="14" fill="#fafaf7" />
        <line x1="32" y1="11" x2="48" y2="11" />
        <line x1="40" y1="3" x2="40" y2="19" />
        <line x1="752" y1="11" x2="768" y2="11" />
        <line x1="760" y1="3" x2="760" y2="19" />
        <path d="M 80 50 L 720 50 Q 740 50 740 70 L 740 240 Q 740 260 720 260 L 80 260 Q 60 260 60 240 L 60 70 Q 60 50 80 50 Z" fill="#fafaf7" />
        <line x1="80" y1="92" x2="720" y2="92" />
        <rect x="360" y="62" width="80" height="22" />
        <text x="400" y="77" fontSize="10" fontFamily="'IM Fell English', serif" textAnchor="middle" fill="#1a1a1a" stroke="none" letterSpacing="3">HEAR!HEAR!</text>
        <g transform="translate(400 145)">
          {[-45, -30, -15, 0, 15, 30, 45].map((a, i) => (
            <line key={i} x1="0" y1="0" x2="0" y2="-30" transform={`rotate(${a})`} opacity="0.6" />
          ))}
        </g>
        {[0, 1, 2].map(r => (
          <g key={r}>
            {Array.from({ length: 11 - r }).map((_, i) => (
              <circle key={i} cx={140 + r * 10 + i * 52} cy={185 + r * 22} r="9" />
            ))}
          </g>
        ))}
        <rect x="280" y="240" width="240" height="10" rx="5" />
      </g>
    </svg>
  )
}

function IntroLine({
  label,
  value,
  showCaret,
  dimmed,
  hideUntilActive,
  editable,
  active,
  onChange,
  onFocus,
  placeholder,
  inputRef,
}: {
  label: string
  value: string
  showCaret?: boolean
  dimmed?: boolean
  hideUntilActive?: boolean
  editable?: boolean
  active?: boolean
  onChange?: (value: string) => void
  onFocus?: () => void
  placeholder?: string
  inputRef?: (el: HTMLInputElement | null) => void
}): JSX.Element {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'baseline',
      gap: 24,
      fontFamily: "'IM Fell English', serif",
      opacity: hideUntilActive ? 0.18 : (dimmed ? 0.55 : 1),
      transition: 'opacity 0.4s ease',
    }}>
      <span style={{ fontSize: 22, fontStyle: 'italic', color: '#3a3a36', minWidth: 140, textAlign: 'right' }}>
        {label}
      </span>
      {editable ? (
        <input
          ref={inputRef}
          value={value}
          onChange={(event) => onChange && onChange(event.target.value)}
          onFocus={onFocus}
          placeholder={placeholder}
          style={{
            fontFamily: "'Special Elite', monospace",
            fontSize: 32,
            letterSpacing: '-0.005em',
            background: 'transparent',
            border: 0,
            borderBottom: active ? '1.5px solid #1a1a1a' : '1px solid #cfcfc7',
            padding: '4px 0 6px',
            minWidth: 520,
            color: '#1a1a1a',
            outline: 'none',
          }}
        />
      ) : (
        <span style={{
          fontFamily: "'Special Elite', monospace",
          fontSize: 32,
          letterSpacing: '-0.005em',
          borderBottom: '1px solid #1a1a1a',
          paddingBottom: 6,
          minWidth: 520,
          display: 'inline-block',
        }}>
          {value || ' '}
          {showCaret && <span className="tw-caret" />}
        </span>
      )}
    </div>
  )
}

function WaxEnvelope({
  onChooseFile,
  isImporting,
  importedPdfName,
}: {
  onChooseFile: () => void
  isImporting: boolean
  importedPdfName: string | null
}): JSX.Element {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, padding: '8px 0 0' }}>
      <button
        type="button"
        onClick={onChooseFile}
        disabled={isImporting}
        style={{
          width: 380,
          height: 200,
          position: 'relative',
          background: 'linear-gradient(180deg, #ece4d0 0%, #d9cfb6 100%)',
          boxShadow: '0 14px 30px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.6)',
          border: '1px solid rgba(0,0,0,0.10)',
          padding: 0,
          cursor: isImporting ? 'wait' : 'pointer',
        }}
      >
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 116, background: 'linear-gradient(180deg, #d9cfb6, #c2b89e)', clipPath: 'polygon(0 0, 100% 0, 50% 100%)' }} />
        <div style={{ position: 'absolute', bottom: 14, left: 38, fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#1a1a1a' }}>
          To: <span style={{ borderBottom: '1px solid #1a1a1a' }}>The Editor, hear! hear!</span>
        </div>
        <div style={{
          position: 'absolute',
          top: 90,
          left: '50%',
          transform: 'translateX(-50%) rotate(-6deg)',
          width: 46,
          height: 42,
          background: 'radial-gradient(circle at 30% 30%, #a23036 0%, #7a1d1d 50%, #4a0c0c 100%)',
          borderRadius: '50% 30% 60% 40%',
          boxShadow: 'inset 0 -3px 6px rgba(0,0,0,0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fafaf7',
          fontFamily: "'IM Fell English', serif",
          fontStyle: 'italic',
          fontSize: 18,
        }}>h</div>
      </button>
      <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>
        {isImporting
          ? 'opening the envelope…'
          : importedPdfName
            ? `loaded: ${importedPdfName}`
            : 'click to break the seal & choose a PDF'}
      </span>
    </div>
  )
}

export function LandingFlow(props: LandingFlowProps): JSX.Element {
  const {
    topic,
    opinion,
    essayText,
    importedPdfName,
    isImportingPdf,
    loading,
    onTopicChange,
    onOpinionChange,
    onEssayTextChange,
    onImportPdf,
    onSubmitStance,
    onSubmitEssayDraft,
    onOpenSettings,
    onOpenAbout,
    retrievalModelLabel,
    stanceLabel,
    rerankModeLabel,
    chunksLabel,
  } = props

  const [phase, setPhase] = useState<IntroPhase>('topic')
  const [voiceMode, setVoiceMode] = useState<VoiceMode>('stance')
  const [activeField, setActiveField] = useState<'topic' | 'claim'>('topic')
  const [essaySource, setEssaySource] = useState<EssaySource>('paste')
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const topicInputRef = useRef<HTMLInputElement | null>(null)
  const opinionInputRef = useRef<HTMLInputElement | null>(null)

  const topicCycle = useTypewriterCycle(INTRO_TOPICS, phase === 'topic')
  const finalTopic = INTRO_TOPICS[INTRO_TOPICS.length - 1]
  const finalClaims = INTRO_CLAIMS_BY_TOPIC[finalTopic]
  const claimCycle = useTypewriterCycle(finalClaims, phase === 'claim')

  useEffect(() => {
    if (phase === 'topic' && topicCycle.done) {
      const t = setTimeout(() => setPhase('claim'), 220)
      return () => clearTimeout(t)
    }
    if (phase === 'claim' && claimCycle.done) {
      const t = setTimeout(() => setPhase('done'), 200)
      return () => clearTimeout(t)
    }
    return undefined
  }, [phase, topicCycle.done, claimCycle.done])

  useEffect(() => {
    if (phase !== 'voice') return
    if (activeField === 'topic') {
      requestAnimationFrame(() => topicInputRef.current?.focus())
    } else {
      requestAnimationFrame(() => opinionInputRef.current?.focus())
    }
  }, [phase, activeField])

  const isVoice = phase === 'voice'

  const findVoice = (): void => {
    setActiveField('topic')
    setPhase('voice')
  }

  const replay = (): void => {
    setPhase('topic')
    setActiveField('topic')
  }

  const topicShown = phase === 'topic'
    ? topicCycle.value
    : (isVoice ? topic : finalTopic)
  const claimShown = phase === 'claim'
    ? claimCycle.value
    : (phase === 'done' ? finalClaims[finalClaims.length - 1] : (isVoice ? opinion : ''))

  const stanceCanSubmit = topic.trim() !== '' && opinion.trim() !== '' && !loading
  const essayCanSubmit = essayText.trim().length > 0 && !loading && !isImportingPdf
  const essayPasteWordCount = essayText.trim().split(/\s+/).filter(Boolean).length

  return (
    <div className="stage-shell" style={{ minHeight: '100dvh', position: 'relative', overflow: 'hidden' }}>
      {/* Top rail */}
      <div className="top-rail">
        <button type="button" className="top-rail-brand">hear! hear!</button>
        <div className="top-rail-links">
          <button type="button" className="active">search</button>
          <button type="button" onClick={onOpenAbout}>about</button>
          <button type="button" onClick={onOpenAbout}>method</button>
        </div>
      </div>
      <div className="top-rule" />

      {/* INTRO MODE */}
      {!isVoice && (
        <div style={{
          position: 'absolute',
          inset: 0,
          paddingTop: 64,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 24,
        }}>
          <div className="tracker" style={{ color: 'var(--accent)', letterSpacing: '0.32em' }}>
            a reading companion · since 2026
          </div>
          <IntroLine
            label="Regarding"
            value={topicShown}
            showCaret={phase === 'topic'}
            dimmed={phase === 'claim' || phase === 'done'}
          />
          <IntroLine
            label="I believe"
            value={claimShown}
            showCaret={phase === 'claim'}
            hideUntilActive={phase === 'topic'}
          />
          <div style={{
            marginTop: 28,
            opacity: phase === 'done' ? 1 : 0,
            transform: phase === 'done' ? 'translateY(0)' : 'translateY(8px)',
            transition: 'opacity 0.6s ease, transform 0.6s ease',
            pointerEvents: phase === 'done' ? 'auto' : 'none',
          }}>
            <button type="button" onClick={findVoice} style={{
              background: 'transparent',
              border: 0,
              padding: '6px 4px',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 6,
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 11,
              letterSpacing: '0.32em',
              textTransform: 'uppercase',
              color: '#1a1a1a',
            }}>
              <span>find your voice</span>
              <span style={{ fontSize: 18, animation: 'tw-bob 1.6s ease-in-out infinite' }}>↓</span>
            </button>
          </div>
        </div>
      )}

      {/* VOICE MODE — toggle */}
      <div style={{
        position: 'absolute',
        left: 0,
        right: 0,
        top: 96,
        display: 'flex',
        justifyContent: 'center',
        gap: 0,
        fontFamily: "'IM Fell DW Pica SC', serif",
        fontSize: 11,
        letterSpacing: '0.28em',
        textTransform: 'uppercase',
        opacity: isVoice ? 1 : 0,
        transform: isVoice ? 'translateY(0)' : 'translateY(-8px)',
        transition: 'opacity 0.5s ease 0.3s, transform 0.5s ease 0.3s',
        pointerEvents: isVoice ? 'auto' : 'none',
        zIndex: 5,
      }}>
        <button type="button" onClick={() => setVoiceMode('stance')} style={{
          padding: '8px 22px',
          border: '1px solid #1a1a1a',
          background: voiceMode === 'stance' ? '#1a1a1a' : 'transparent',
          color: voiceMode === 'stance' ? '#fafaf7' : '#1a1a1a',
          fontFamily: 'inherit',
          fontSize: 'inherit',
          letterSpacing: 'inherit',
          textTransform: 'inherit',
          cursor: 'pointer',
        }}>topic & stance</button>
        <button type="button" onClick={() => setVoiceMode('essay')} style={{
          padding: '8px 22px',
          border: '1px solid #1a1a1a',
          borderLeft: 0,
          background: voiceMode === 'essay' ? '#1a1a1a' : 'transparent',
          color: voiceMode === 'essay' ? '#fafaf7' : '#1a1a1a',
          fontFamily: 'inherit',
          fontSize: 'inherit',
          letterSpacing: 'inherit',
          textTransform: 'inherit',
          cursor: 'pointer',
        }}>essay</button>
      </div>

      {/* Paper sheet */}
      <div style={{
        position: 'absolute',
        left: '50%',
        top: isVoice ? 168 : 0,
        transform: `translateX(-50%) ${isVoice ? '' : 'translateY(-100%)'}`,
        width: 760,
        height: voiceMode === 'essay' ? 380 : 220,
        background: '#fafaf7',
        border: isVoice ? '1px solid rgba(26,26,26,0.6)' : 0,
        borderBottom: 'none',
        boxShadow: isVoice ? '0 -2px 0 rgba(26,26,26,0.04), 0 -8px 24px rgba(26,26,26,0.04)' : 'none',
        opacity: isVoice ? 1 : 0,
        transition: 'top 0.9s cubic-bezier(.2,.7,.2,1), opacity 0.6s ease, height 0.5s ease',
        padding: '36px 36px 0',
        display: 'flex',
        flexDirection: 'column',
        gap: 22,
        pointerEvents: isVoice ? 'auto' : 'none',
        overflow: 'hidden',
        zIndex: 10,
      }}>
        <div style={{
          position: 'absolute',
          top: 8,
          left: 14,
          right: 14,
          fontFamily: "'IM Fell DW Pica SC', serif",
          fontSize: 9,
          letterSpacing: '0.32em',
          textTransform: 'uppercase',
          color: '#9a9a92',
          display: 'flex',
          justifyContent: 'space-between',
        }}>
          <span>hear! hear! / draft</span>
          <span>№ 001</span>
        </div>

        {voiceMode === 'stance' && (
          <>
            <IntroLine
              label="Regarding"
              value={topic}
              showCaret={isVoice && activeField === 'topic'}
              editable
              active={activeField === 'topic'}
              onChange={onTopicChange}
              onFocus={() => setActiveField('topic')}
              placeholder="a subject…"
              inputRef={(el) => { topicInputRef.current = el }}
            />
            <IntroLine
              label="I believe"
              value={opinion}
              showCaret={isVoice && activeField === 'claim'}
              editable
              active={activeField === 'claim'}
              onChange={onOpinionChange}
              onFocus={() => setActiveField('claim')}
              placeholder="an opinion…"
              inputRef={(el) => { opinionInputRef.current = el }}
            />
          </>
        )}

        {voiceMode === 'essay' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 22, fontStyle: 'italic', color: '#3a3a36' }}>An essay,</span>
              <div style={{ display: 'flex', fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase' }}>
                <button type="button" onClick={() => setEssaySource('paste')} style={{
                  padding: '4px 12px',
                  border: '1px solid #1a1a1a',
                  background: essaySource === 'paste' ? '#1a1a1a' : 'transparent',
                  color: essaySource === 'paste' ? '#fafaf7' : '#1a1a1a',
                  fontFamily: 'inherit',
                  fontSize: 'inherit',
                  letterSpacing: 'inherit',
                  textTransform: 'inherit',
                  cursor: 'pointer',
                }}>type / paste</button>
                <button type="button" onClick={() => setEssaySource('envelope')} style={{
                  padding: '4px 12px',
                  border: '1px solid #1a1a1a',
                  borderLeft: 0,
                  background: essaySource === 'envelope' ? '#1a1a1a' : 'transparent',
                  color: essaySource === 'envelope' ? '#fafaf7' : '#1a1a1a',
                  fontFamily: 'inherit',
                  fontSize: 'inherit',
                  letterSpacing: 'inherit',
                  textTransform: 'inherit',
                  cursor: 'pointer',
                }}>pdf envelope</button>
              </div>
            </div>

            {essaySource === 'paste' && (
              <div style={{ position: 'relative', border: '1px solid #cfcfc7', background: '#fafaf7' }}>
                <div aria-hidden style={{
                  position: 'absolute',
                  inset: 0,
                  backgroundImage: 'repeating-linear-gradient(0deg, transparent 0, transparent 25px, rgba(26,26,26,0.08) 26px)',
                  pointerEvents: 'none',
                }} />
                <textarea
                  value={essayText}
                  onChange={(event) => onEssayTextChange(event.target.value)}
                  placeholder="The case for sea protection has been overstated as sentiment and underargued as policy…"
                  style={{
                    position: 'relative',
                    width: '100%',
                    height: 240,
                    padding: '6px 14px',
                    background: 'transparent',
                    border: 0,
                    outline: 'none',
                    resize: 'none',
                    fontFamily: "'Special Elite', monospace",
                    fontSize: 14,
                    lineHeight: '26px',
                    color: '#1a1a1a',
                  }}
                />
              </div>
            )}

            {essaySource === 'envelope' && (
              <WaxEnvelope
                onChooseFile={() => fileInputRef.current?.click()}
                isImporting={isImportingPdf}
                importedPdfName={importedPdfName}
              />
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              style={{ display: 'none' }}
              onChange={onImportPdf}
            />

            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'Special Elite', monospace", fontSize: 10, color: '#6a6a62' }}>
              <span>{essaySource === 'paste' ? `${essayPasteWordCount} words` : ' '}</span>
              <span>auto-extract claims · on</span>
            </div>
          </div>
        )}
      </div>

      {/* Typewriter — slides in only in stance mode */}
      <div style={{
        position: 'absolute',
        left: '50%',
        bottom: 255,
        transform: `translateX(-50%) translateY(${isVoice && voiceMode === 'stance' ? 0 : 360}px)`,
        transition: 'transform 0.9s cubic-bezier(.2,.7,.2,1), opacity 0.4s ease',
        opacity: isVoice && voiceMode === 'stance' ? 1 : 0,
        pointerEvents: 'none',
        visibility: isVoice && voiceMode === 'stance' ? 'visible' : 'hidden',
        zIndex: 1,
      }}>
        <RisingTypewriter />
      </div>

      {/* Send to press button */}
      <div style={{
        position: 'absolute',
        left: 0,
        right: 0,
        bottom: 200,
        display: 'flex',
        justifyContent: 'center',
        opacity: isVoice ? 1 : 0,
        transition: 'opacity 0.5s ease 0.55s',
        pointerEvents: isVoice ? 'auto' : 'none',
      }}>
        <button
          type="button"
          onClick={voiceMode === 'stance' ? onSubmitStance : onSubmitEssayDraft}
          disabled={voiceMode === 'stance' ? !stanceCanSubmit : !essayCanSubmit}
          style={{
            background: '#1a1a1a',
            color: '#fafaf7',
            border: '1px solid #1a1a1a',
            padding: '12px 28px',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 12,
            letterSpacing: '0.32em',
            textTransform: 'uppercase',
            cursor: (voiceMode === 'stance' ? stanceCanSubmit : essayCanSubmit) ? 'pointer' : 'not-allowed',
            opacity: (voiceMode === 'stance' ? stanceCanSubmit : essayCanSubmit) ? 1 : 0.5,
          }}
        >
          {voiceMode === 'essay' ? 'extract claims & send to press →' : 'send to press →'}
        </button>
      </div>

      {/* Instrument settings — bottom right */}
      <div style={{
        position: 'absolute',
        right: 48,
        bottom: 90,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-end',
        gap: 8,
        opacity: isVoice ? 1 : 0,
        transition: 'opacity 0.5s ease 0.6s',
        pointerEvents: isVoice ? 'auto' : 'none',
      }}>
        <div className="tracker" style={{ color: 'var(--accent)', letterSpacing: '0.32em', fontSize: 9 }}>The Instrument</div>
        <div style={{ fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#1a1a1a', textAlign: 'right', lineHeight: 1.6 }}>
          reads · <strong>{retrievalModelLabel}</strong> &nbsp;·&nbsp; judges · <strong>{stanceLabel}</strong><br />
          rerank · <strong>{rerankModeLabel}</strong> &nbsp;·&nbsp; chunks · <strong>{chunksLabel}</strong>
        </div>
        <button
          type="button"
          onClick={onOpenSettings}
          style={{
            background: 'transparent',
            border: '1px solid #1a1a1a',
            padding: '6px 12px',
            cursor: 'pointer',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 9,
            letterSpacing: '0.28em',
            textTransform: 'uppercase',
            color: '#1a1a1a',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          ⚙ instrument settings
        </button>
        <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 11, color: '#6a6a62', maxWidth: 200, textAlign: 'right', lineHeight: 1.4 }}>
          most readers leave these alone.
        </div>
      </div>

      {/* Back to intro — bottom left */}
      <div style={{
        position: 'absolute',
        left: 48,
        bottom: 90,
        opacity: isVoice ? 1 : 0,
        transition: 'opacity 0.5s ease 0.4s',
        pointerEvents: isVoice ? 'auto' : 'none',
      }}>
        <button type="button" onClick={() => setPhase('done')} style={{
          background: 'transparent',
          border: '1px solid #1a1a1a',
          padding: '8px 18px',
          fontFamily: "'IM Fell DW Pica SC', serif",
          fontSize: 10,
          letterSpacing: '0.28em',
          textTransform: 'uppercase',
          color: '#1a1a1a',
          cursor: 'pointer',
        }}>
          ← back
        </button>
      </div>

      {/* Footer */}
      <div className="footer-rail" style={{ position: 'absolute', bottom: 0, left: 0, right: 0, margin: 0, padding: '12px 48px', borderTop: '1px solid #1a1a1a' }}>
        <span>landing · the front page</span>
        <button type="button" onClick={replay} style={{
          background: 'transparent',
          border: '1px solid #1a1a1a',
          padding: '6px 14px',
          fontFamily: "'IM Fell DW Pica SC', serif",
          fontSize: 9,
          letterSpacing: '0.28em',
          textTransform: 'uppercase',
          color: '#1a1a1a',
          cursor: 'pointer',
        }}>↻ replay</button>
        <span>guardian opinion · indexed</span>
      </div>
    </div>
  )
}

export default LandingFlow
