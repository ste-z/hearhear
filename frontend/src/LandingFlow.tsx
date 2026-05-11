import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from 'react'
import { PersonaName } from './PersonaName'
import SpotlightTour, { type SpotlightTourStep } from './SpotlightTour'
import ThemeToggle, { type Theme } from './ThemeToggle'
import { playCarriageReturn, playKeyStrike, playSend } from './typewriterAudio'
import typewriterFrameUrl from './assets/typewriter-frame.svg'

const INTRO_TOPICS = ['climate', 'immigration', 'minimum wage'] as const
const INTRO_CLAIMS_BY_TOPIC: Record<typeof INTRO_TOPICS[number], readonly string[]> = {
  climate: ['cut emissions', 'expand clean energy', 'hold polluters accountable'],
  immigration: ['protect asylum rights', 'expand legal pathways', 'support new arrivals'],
  'minimum wage': ['wages should rise', 'pay should track inflation', 'work should pay enough'],
}

const LANDING_SEEN_KEY = 'hearhear.hasSeenLanding'
const COMPOSE_TOUR_SEEN_KEY = 'hearhear.tour.composeSeen'

type IntroPhase = 'topic' | 'claim' | 'done' | 'voice'
type VoiceMode = 'stance' | 'essay'
type EssaySource = 'paste' | 'envelope'

type LengthFilterUnit = 'characters' | 'words' | 'reading_time'

const COMPOSE_TOUR_STEPS: SpotlightTourStep[] = [
  {
    target: 'compose-mode-toggle',
    title: 'Choose your search mode',
    body: 'Start from a short topic and stance, or switch to essay mode when you already have a draft.',
    placement: 'bottom',
  },
  {
    target: 'compose-stance-slip',
    title: 'Topic & stance mode',
    body: 'Use Regarding for the subject, then I believe for the position you want articles searched against.',
    placement: 'right',
  },
  {
    target: 'compose-essay-slip',
    title: 'Essay mode',
    body: 'Essay mode lets the app work from your own draft instead of a two-line claim.',
    placement: 'left',
  },
  {
    target: 'compose-essay-source',
    title: 'Paste text or open a PDF',
    body: 'Paste your essay into the lined page, or use the PDF envelope to extract text from a file.',
    placement: 'left',
  },
  {
    target: 'compose-settings',
    title: 'Choose the instrument',
    body: 'Settings control how hear! hear! reads the archive, judges stance, chunks long articles, and balances topic, stance, and recency.',
    placement: 'top',
  },
  {
    target: 'compose-filters',
    title: 'Narrow the archive',
    body: 'Use year, length, and avoid-word filters to shape the candidate set before ranking begins.',
    placement: 'top',
  },
  {
    target: 'compose-actions',
    title: 'Send to press',
    body: 'Start the search pipeline.',
    placement: 'top',
  },
]

export type LandingFlowProps = {
  topic: string
  opinion: string
  essayText: string
  importedPdfName: string | null
  isImportingPdf: boolean
  loading: boolean
  // Accept React.Dispatch so the typewriter can append via functional updates
  // (avoids stale-closure bugs when several keys are pressed in the same tick).
  onTopicChange: React.Dispatch<React.SetStateAction<string>>
  onOpinionChange: React.Dispatch<React.SetStateAction<string>>
  onEssayTextChange: (value: string) => void
  onImportPdf: (event: ChangeEvent<HTMLInputElement>) => void
  onSubmitStance: () => void
  onSubmitEssayDraft: () => void
  onOpenSettings: () => void
  onOpenAbout: () => void
  onOpenMethod: () => void
  onOpenExplore: () => void
  tutorialRequestId: number
  theme: Theme
  onToggleTheme: () => void
  chunksLabel: string
  // Persona ids drive the wavy-underline name component on the search page.
  effectiveRetrievalModel: 'tfidf' | 'svd' | 'minilm'
  // Determines whether the essay flow needs a thesis-extraction step (NLI) or not (LLM).
  effectiveStanceMethod: 'nli' | 'llm'

  // filters (folded back from settings)
  yearStart: number | null
  yearEnd: number | null
  minYear: number | null
  maxYear: number | null
  onYearStartChange: (value: number | null) => void
  onYearEndChange: (value: number | null) => void
  lengthFilterUnit: LengthFilterUnit
  onLengthFilterUnitChange: (next: LengthFilterUnit) => void
  lengthRangeStart: number | null
  lengthRangeEnd: number | null
  lengthRangeMin: number | null
  lengthRangeMax: number | null
  onLengthRangeStartChange: (value: number | null) => void
  onLengthRangeEndChange: (value: number | null) => void
  // words-to-avoid (only meaningful for lexical / TF·IDF search)
  wordsToAvoid: string[]
  onWordsToAvoidChange: (next: string[]) => void
  isLexicalSearchMode: boolean
}

function readLandingSeen(): boolean {
  if (typeof window === 'undefined') return false
  try {
    return window.localStorage.getItem(LANDING_SEEN_KEY) === 'true'
  } catch {
    return false
  }
}

function markLandingSeen(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(LANDING_SEEN_KEY, 'true')
  } catch {
    /* ignore */
  }
}

function clearLandingSeen(): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.removeItem(LANDING_SEEN_KEY)
  } catch {
    /* ignore */
  }
}

function readTourSeen(key: string): boolean {
  if (typeof window === 'undefined') return false
  try {
    return window.localStorage.getItem(key) === 'true'
  } catch {
    return false
  }
}

function markTourSeen(key: string): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(key, 'true')
  } catch {
    /* ignore */
  }
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

type TypewriterProps = {
  onType: (char: string) => void
  onBackspace: () => void
  onEnter: () => void
  disabled?: boolean
  // When set, the matching key briefly shows a depressed state. Use to mirror real-keyboard input.
  flashedKey?: string | null
}

const COMPOSE_SURFACE_WIDTH = 760

function TypewriterKey({
  children,
  onPress,
  width = 'var(--tw-key-size)',
  height = 'var(--tw-key-size)',
  shape = 'circle',
  fontSize = 'var(--tw-key-font-size)',
  disabled,
  flashed = false,
}: {
  children: React.ReactNode
  onPress: () => void
  width?: number | string
  height?: number | string
  shape?: 'circle' | 'pill'
  fontSize?: number | string
  disabled?: boolean
  flashed?: boolean
}): JSX.Element {
  const [pressed, setPressed] = useState(false)
  const release = (): void => setPressed(false)
  const isDown = pressed || flashed
  return (
    <button
      type="button"
      // preventDefault on mousedown keeps the focused input from blurring
      // when the user clicks a key — input stays editable through the keyboard.
      onMouseDown={(event) => {
        event.preventDefault()
        if (disabled) return
        setPressed(true)
      }}
      onMouseUp={release}
      onMouseLeave={release}
      onTouchStart={(event) => {
        event.preventDefault()
        if (disabled) return
        setPressed(true)
      }}
      onTouchEnd={release}
      onClick={(event) => {
        event.preventDefault()
        if (disabled) return
        onPress()
      }}
      disabled={disabled}
      style={{
        width,
        height,
        border: '1.5px solid var(--ink)',
        borderRadius: shape === 'pill' ? 18 : '50%',
        fontFamily: "'IM Fell English', serif",
        fontSize,
        lineHeight: 1,
        color: 'var(--ink)',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.55 : 1,
        padding: 0,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        transform: isDown ? 'translateY(2px)' : 'translateY(0)',
        boxShadow: isDown ? 'inset 0 1px 3px rgba(var(--ink-rgb),0.24)' : '0 2px 0 rgba(var(--ink-rgb),0.22)',
        background: isDown ? 'var(--paper-warm)' : 'rgba(var(--paper-rgb),0.96)',
        transition: 'transform 80ms ease, box-shadow 80ms ease, background 120ms ease',
        userSelect: 'none',
      }}
    >
      {children}
    </button>
  )
}

function RisingTypewriter({ onType, onBackspace, onEnter, disabled, flashedKey }: TypewriterProps): JSX.Element {
  const row1 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
  const row2 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
  const row3 = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
  const punctuation = [',', '.', '?']
  const isFlashed = (id: string): boolean => flashedKey !== null && flashedKey !== undefined && flashedKey === id

  return (
    <div className="landing-typewriter">
      <img
        src={typewriterFrameUrl}
        alt=""
        aria-hidden="true"
        draggable={false}
        className="landing-typewriter-frame"
      />

      <div className="landing-typewriter-keys">
        <div className="landing-typewriter-brand">
          HEAR! HEAR!
        </div>

        {/* Row 1: QWERTYUIOP + Enter */}
        <div className="landing-key-row">
          {row1.map(k => (
            <TypewriterKey
              key={k}
              onPress={() => onType(k.toLowerCase())}
              disabled={disabled}
              flashed={isFlashed(k.toLowerCase())}
            >
              {k}
            </TypewriterKey>
          ))}
          <TypewriterKey onPress={onEnter} width="var(--tw-pill-width)" shape="pill" disabled={disabled} fontSize="var(--tw-action-font-size)" flashed={isFlashed('enter')}>
            ↵
          </TypewriterKey>
        </div>

        {/* Row 2: ASDFGHJKL (slightly indented) */}
        <div className="landing-key-row landing-key-row-mid">
          {row2.map(k => (
            <TypewriterKey
              key={k}
              onPress={() => onType(k.toLowerCase())}
              disabled={disabled}
              flashed={isFlashed(k.toLowerCase())}
            >
              {k}
            </TypewriterKey>
          ))}
        </div>

        {/* Row 3: Backspace + ZXCVBNM,.? */}
        <div className="landing-key-row">
          <TypewriterKey onPress={onBackspace} width="var(--tw-pill-width)" shape="pill" disabled={disabled} fontSize="var(--tw-action-font-size)" flashed={isFlashed('backspace')}>
            ⌫
          </TypewriterKey>
          {row3.map(k => (
            <TypewriterKey
              key={k}
              onPress={() => onType(k.toLowerCase())}
              disabled={disabled}
              flashed={isFlashed(k.toLowerCase())}
            >
              {k}
            </TypewriterKey>
          ))}
          {punctuation.map(p => (
            <TypewriterKey
              key={p}
              onPress={() => onType(p)}
              disabled={disabled}
              fontSize={13}
              flashed={isFlashed(p)}
            >
              {p}
            </TypewriterKey>
          ))}
        </div>

        {/* Spacebar */}
        <div className="landing-space-row">
          <button
            type="button"
            onMouseDown={(event) => { event.preventDefault() }}
            onTouchStart={(event) => { event.preventDefault() }}
            onClick={() => { if (!disabled) onType(' ') }}
            disabled={disabled}
            style={{
              width: 'var(--tw-space-width)',
              height: 'var(--tw-space-height)',
              border: '1.5px solid var(--ink)',
              background: isFlashed('space') ? 'var(--paper-warm)' : 'rgba(var(--paper-rgb),0.96)',
              borderRadius: 14,
              cursor: disabled ? 'not-allowed' : 'pointer',
              padding: 0,
              opacity: disabled ? 0.55 : 1,
              userSelect: 'none',
              transform: isFlashed('space') ? 'translateY(2px)' : 'translateY(0)',
              boxShadow: isFlashed('space') ? 'inset 0 1px 3px rgba(var(--ink-rgb),0.24)' : '0 2px 0 rgba(var(--ink-rgb),0.22)',
              transition: 'transform 80ms ease, box-shadow 80ms ease, background 120ms ease',
            }}
            aria-label="space"
          />
        </div>
      </div>
    </div>
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
  onKeyDown,
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
  onKeyDown?: (event: ReactKeyboardEvent<HTMLInputElement>) => void
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
      width: '100%',
    }}>
      <span style={{ fontSize: 22, fontStyle: 'italic', color: 'var(--ink-soft)', flex: '0 0 140px', minWidth: 0, textAlign: 'right' }}>
        {label}
      </span>
      {editable ? (
        <input
          ref={inputRef}
          value={value}
          onChange={(event) => onChange && onChange(event.target.value)}
          onFocus={onFocus}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          style={{
            fontFamily: "'Special Elite', monospace",
            fontSize: 32,
            letterSpacing: '-0.005em',
            background: 'transparent',
            border: 0,
            borderBottom: active ? '1.5px solid var(--ink)' : '1px solid var(--ink-faint)',
            padding: '4px 0 6px',
            flex: 1,
            minWidth: 0,
            width: '100%',
            color: 'var(--ink)',
            outline: 'none',
          }}
        />
      ) : (
        <span style={{
          fontFamily: "'Special Elite', monospace",
          fontSize: 32,
          letterSpacing: '-0.005em',
          borderBottom: '1px solid var(--ink)',
          paddingBottom: 6,
          flex: 1,
          minWidth: 0,
          display: 'inline-block',
        }}>
          {value || ' '}
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
          background: 'linear-gradient(180deg, var(--paper-warm) 0%, var(--paper-edge) 100%)',
          boxShadow: '0 14px 30px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.6)',
          border: '1px solid rgba(0,0,0,0.10)',
          padding: 0,
          cursor: isImporting ? 'wait' : 'pointer',
        }}
      >
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 116, background: 'linear-gradient(180deg, var(--paper-edge), var(--paper-deep))', clipPath: 'polygon(0 0, 100% 0, 50% 100%)' }} />
        <div style={{ position: 'absolute', bottom: 14, left: 38, fontFamily: "'Special Elite', monospace", fontSize: 11, color: 'var(--ink)' }}>
          To: <span style={{ borderBottom: '1px solid var(--ink)' }}>The Editor, hear! hear!</span>
        </div>
        <div style={{
          position: 'absolute',
          top: 90,
          left: '50%',
          transform: 'translateX(-50%) rotate(-6deg)',
          width: 46,
          height: 42,
          background: 'radial-gradient(circle at 30% 30%, var(--accent-light) 0%, var(--accent) 50%, var(--accent-deep) 100%)',
          borderRadius: '50% 30% 60% 40%',
          boxShadow: 'inset 0 -3px 6px rgba(0,0,0,0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--paper)',
          fontFamily: "'IM Fell English', serif",
          fontStyle: 'italic',
          fontSize: 18,
        }}>h</div>
      </button>
      <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: 'var(--ink-mute)' }}>
        {isImporting
          ? 'opening the envelope…'
          : importedPdfName
            ? `loaded: ${importedPdfName}`
            : 'click to break the seal & choose a PDF'}
      </span>
    </div>
  )
}

function FilterPopover({
  open,
  anchorRect,
  onClose,
  children,
}: {
  open: boolean
  anchorRect: DOMRect | null
  onClose: () => void
  children: React.ReactNode
}): JSX.Element | null {
  if (!open || !anchorRect) return null
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 50,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          position: 'fixed',
          top: anchorRect.bottom + 6,
          left: Math.max(12, anchorRect.left - 10),
          width: 320,
          background: 'var(--paper)',
          border: '1px solid var(--ink)',
          boxShadow: '0 14px 30px var(--shadow-strong)',
          padding: '14px 16px',
          fontFamily: "'Old Standard TT', serif",
        }}
      >
        {children}
      </div>
    </div>
  )
}

function FilterRow({
  yearStart,
  yearEnd,
  minYear,
  maxYear,
  onYearStartChange,
  onYearEndChange,
  lengthFilterUnit,
  onLengthFilterUnitChange,
  lengthRangeStart,
  lengthRangeEnd,
  lengthRangeMin,
  lengthRangeMax,
  onLengthRangeStartChange,
  onLengthRangeEndChange,
  wordsToAvoid,
  onWordsToAvoidChange,
  isLexicalSearchMode,
}: {
  yearStart: number | null
  yearEnd: number | null
  minYear: number | null
  maxYear: number | null
  onYearStartChange: (value: number | null) => void
  onYearEndChange: (value: number | null) => void
  lengthFilterUnit: LengthFilterUnit
  onLengthFilterUnitChange: (next: LengthFilterUnit) => void
  lengthRangeStart: number | null
  lengthRangeEnd: number | null
  lengthRangeMin: number | null
  lengthRangeMax: number | null
  onLengthRangeStartChange: (value: number | null) => void
  onLengthRangeEndChange: (value: number | null) => void
  wordsToAvoid: string[]
  onWordsToAvoidChange: (next: string[]) => void
  isLexicalSearchMode: boolean
}): JSX.Element {
  const yearButtonRef = useRef<HTMLButtonElement | null>(null)
  const lengthButtonRef = useRef<HTMLButtonElement | null>(null)
  const avoidButtonRef = useRef<HTMLButtonElement | null>(null)
  const [openPopover, setOpenPopover] = useState<'year' | 'length' | 'avoid' | null>(null)
  const [avoidDraft, setAvoidDraft] = useState('')

  const addAvoidWord = (): void => {
    const word = avoidDraft.trim()
    if (!word) return
    const lower = word.toLocaleLowerCase()
    if (wordsToAvoid.some(w => w.toLocaleLowerCase() === lower)) {
      setAvoidDraft('')
      return
    }
    onWordsToAvoidChange([...wordsToAvoid, word])
    setAvoidDraft('')
  }
  const removeAvoidWord = (word: string): void => {
    onWordsToAvoidChange(wordsToAvoid.filter(w => w !== word))
  }
  const avoidActive = isLexicalSearchMode && wordsToAvoid.length > 0

  const yearActive = minYear !== null && maxYear !== null
    && (yearStart !== minYear || yearEnd !== maxYear)
  const lengthActive = lengthRangeMin !== null && lengthRangeMax !== null
    && (lengthRangeStart !== lengthRangeMin || lengthRangeEnd !== lengthRangeMax)

  const yearLabel = yearStart !== null && yearEnd !== null
    ? (yearActive ? `${yearStart} — ${yearEnd}` : `${yearStart} — ${yearEnd}`)
    : 'any'

  const lengthUnitLabel = lengthFilterUnit === 'reading_time' ? 'reading time' : lengthFilterUnit
  const lengthSummary = !lengthActive
    ? 'any'
    : lengthRangeStart !== null && lengthRangeEnd !== null
      ? `${lengthRangeStart.toLocaleString()} — ${lengthRangeEnd.toLocaleString()}`
      : 'any'

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      gap: 28,
      fontFamily: "'Special Elite', monospace",
      fontSize: 12,
      color: 'var(--ink-soft)',
    }}>
      <button
        ref={yearButtonRef}
        type="button"
        onClick={() => setOpenPopover(openPopover === 'year' ? null : 'year')}
        style={{
          background: 'transparent',
          border: 0,
          padding: '2px 6px',
          fontFamily: 'inherit',
          fontSize: 'inherit',
          color: yearActive ? 'var(--ink)' : 'var(--ink-soft)',
          cursor: 'pointer',
          borderBottom: yearActive ? '1px solid var(--ink)' : '1px dotted var(--rule-dotted)',
        }}
      >
        year · {yearLabel}
      </button>
      <span style={{ color: 'var(--ink-faint)' }}>·</span>
      <button
        ref={lengthButtonRef}
        type="button"
        onClick={() => setOpenPopover(openPopover === 'length' ? null : 'length')}
        style={{
          background: 'transparent',
          border: 0,
          padding: '2px 6px',
          fontFamily: 'inherit',
          fontSize: 'inherit',
          color: lengthActive ? 'var(--ink)' : 'var(--ink-soft)',
          cursor: 'pointer',
          borderBottom: lengthActive ? '1px solid var(--ink)' : '1px dotted var(--rule-dotted)',
        }}
      >
        length · {lengthUnitLabel} · {lengthSummary}
      </button>
      <span style={{ color: 'var(--ink-faint)' }}>·</span>
      <button
        ref={avoidButtonRef}
        type="button"
        onClick={() => setOpenPopover(openPopover === 'avoid' ? null : 'avoid')}
        title={isLexicalSearchMode ? 'words to avoid' : 'available only in TF·IDF (lexical) compositor mode'}
        style={{
          background: 'transparent',
          border: 0,
          padding: '2px 6px',
          fontFamily: 'inherit',
          fontSize: 'inherit',
          color: !isLexicalSearchMode ? 'var(--ink-faint)' : (avoidActive ? 'var(--ink)' : 'var(--ink-soft)'),
          cursor: 'pointer',
          borderBottom: avoidActive ? '1px solid var(--ink)' : '1px dotted var(--rule-dotted)',
        }}
      >
        avoid · {avoidActive ? `${wordsToAvoid.length} ${wordsToAvoid.length === 1 ? 'word' : 'words'}` : 'none'}
      </button>

      <FilterPopover
        open={openPopover === 'year'}
        anchorRect={yearButtonRef.current?.getBoundingClientRect() ?? null}
        onClose={() => setOpenPopover(null)}
      >
        <div className="tracker" style={{ marginBottom: 8 }}>year published</div>
        {minYear === null || maxYear === null ? (
          <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', color: 'var(--ink-mute)', fontSize: 13 }}>
            year bounds unavailable
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'Special Elite', monospace", fontSize: 12, color: 'var(--ink)' }}>
              <span>{yearStart ?? minYear}</span>
              <span>{yearEnd ?? maxYear}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 6 }}>
              <input
                type="range"
                className="tw-range"
                min={minYear}
                max={maxYear}
                step={1}
                value={yearStart ?? minYear}
                onChange={(event) => {
                  const next = parseInt(event.target.value, 10)
                  onYearStartChange(Math.min(next, yearEnd ?? maxYear))
                }}
              />
              <input
                type="range"
                className="tw-range"
                min={minYear}
                max={maxYear}
                step={1}
                value={yearEnd ?? maxYear}
                onChange={(event) => {
                  const next = parseInt(event.target.value, 10)
                  onYearEndChange(Math.max(next, yearStart ?? minYear))
                }}
              />
            </div>
            <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'var(--ink-mute)' }}>
              <span>{minYear}</span>
              <button
                type="button"
                onClick={() => { onYearStartChange(minYear); onYearEndChange(maxYear) }}
                style={{ background: 'transparent', border: 0, color: 'var(--ink-mute)', cursor: 'pointer', fontFamily: 'inherit', letterSpacing: 'inherit', textTransform: 'inherit', fontSize: 'inherit' }}
              >
                ↻ reset
              </button>
              <span>{maxYear}</span>
            </div>
          </>
        )}
      </FilterPopover>

      <FilterPopover
        open={openPopover === 'length'}
        anchorRect={lengthButtonRef.current?.getBoundingClientRect() ?? null}
        onClose={() => setOpenPopover(null)}
      >
        <div className="tracker" style={{ marginBottom: 8 }}>article length</div>
        <div style={{ display: 'flex', borderTop: '1px solid var(--ink)', borderBottom: '1px solid var(--ink)', marginBottom: 10 }}>
          {(['characters', 'words', 'reading_time'] as LengthFilterUnit[]).map((unit, i) => (
            <button
              key={unit}
              type="button"
              onClick={() => onLengthFilterUnitChange(unit)}
              style={{
                flex: 1,
                background: lengthFilterUnit === unit ? 'var(--ink)' : 'transparent',
                color: lengthFilterUnit === unit ? 'var(--paper)' : 'var(--ink)',
                border: 0,
                borderLeft: i === 0 ? 0 : '1px solid var(--ink)',
                padding: '6px 4px',
                cursor: 'pointer',
                fontFamily: "'IM Fell English', serif",
                fontSize: 12,
              }}
            >
              {unit === 'reading_time' ? 'reading' : unit}
            </button>
          ))}
        </div>
        {lengthRangeMin === null || lengthRangeMax === null ? (
          <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', color: 'var(--ink-mute)', fontSize: 13 }}>
            length bounds unavailable
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'Special Elite', monospace", fontSize: 12, color: 'var(--ink)' }}>
              <span>{(lengthRangeStart ?? lengthRangeMin).toLocaleString()}</span>
              <span>{(lengthRangeEnd ?? lengthRangeMax).toLocaleString()}</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 6 }}>
              <input
                type="range"
                className="tw-range"
                min={lengthRangeMin}
                max={lengthRangeMax}
                step={1}
                value={lengthRangeStart ?? lengthRangeMin}
                onChange={(event) => {
                  const next = parseInt(event.target.value, 10)
                  onLengthRangeStartChange(Math.min(next, lengthRangeEnd ?? lengthRangeMax))
                }}
              />
              <input
                type="range"
                className="tw-range"
                min={lengthRangeMin}
                max={lengthRangeMax}
                step={1}
                value={lengthRangeEnd ?? lengthRangeMax}
                onChange={(event) => {
                  const next = parseInt(event.target.value, 10)
                  onLengthRangeEndChange(Math.max(next, lengthRangeStart ?? lengthRangeMin))
                }}
              />
            </div>
            <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'var(--ink-mute)' }}>
              <span>{lengthRangeMin.toLocaleString()}</span>
              <button
                type="button"
                onClick={() => { onLengthRangeStartChange(lengthRangeMin); onLengthRangeEndChange(lengthRangeMax) }}
                style={{ background: 'transparent', border: 0, color: 'var(--ink-mute)', cursor: 'pointer', fontFamily: 'inherit', letterSpacing: 'inherit', textTransform: 'inherit', fontSize: 'inherit' }}
              >
                ↻ reset
              </button>
              <span>{lengthRangeMax.toLocaleString()}</span>
            </div>
          </>
        )}
      </FilterPopover>

      <FilterPopover
        open={openPopover === 'avoid'}
        anchorRect={avoidButtonRef.current?.getBoundingClientRect() ?? null}
        onClose={() => setOpenPopover(null)}
      >
        <div className="tracker" style={{ marginBottom: 8 }}>words to avoid (TF·IDF only)</div>
        {!isLexicalSearchMode && (
          <p style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: 'var(--ink-mute)', margin: '0 0 10px' }}>
            This stoplist is honoured only when the compositor is set to <strong>TF·IDF (Old Hewitt)</strong>. The semantic compositors ignore it.
          </p>
        )}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <input
            value={avoidDraft}
            onChange={(event) => setAvoidDraft(event.target.value)}
            placeholder="word to suppress…"
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault()
                addAvoidWord()
              }
            }}
            disabled={!isLexicalSearchMode}
            style={{
              flex: 1,
              border: '1px solid var(--ink)',
              background: 'var(--paper)',
              padding: '6px 10px',
              fontFamily: "'Special Elite', monospace",
              fontSize: 12,
              color: 'var(--ink)',
              outline: 'none',
            }}
          />
          <button
            type="button"
            onClick={addAvoidWord}
            disabled={!isLexicalSearchMode}
            style={{
              background: 'transparent',
              border: '1px solid var(--ink)',
              padding: '6px 12px',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 9,
              letterSpacing: '0.24em',
              textTransform: 'uppercase',
              cursor: isLexicalSearchMode ? 'pointer' : 'not-allowed',
              color: 'var(--ink)',
            }}
          >
            add
          </button>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
          {wordsToAvoid.map(word => (
            <span key={word} style={{
              border: '1px solid var(--ink)',
              padding: '3px 9px',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 9,
              letterSpacing: '0.16em',
              textTransform: 'uppercase',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
            }}>
              {word}
              <button
                type="button"
                onClick={() => removeAvoidWord(word)}
                style={{ background: 'transparent', border: 0, color: 'var(--accent)', cursor: 'pointer', fontFamily: 'inherit', padding: 0 }}
              >
                ×
              </button>
            </span>
          ))}
          {wordsToAvoid.length === 0 && (
            <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: 'var(--ink-mute)' }}>
              no words yet
            </span>
          )}
        </div>
      </FilterPopover>
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
    onOpenExplore,
    tutorialRequestId,
    theme,
    onToggleTheme,
    chunksLabel,
    effectiveRetrievalModel,
    effectiveStanceMethod,
    yearStart,
    yearEnd,
    minYear,
    maxYear,
    onYearStartChange,
    onYearEndChange,
    lengthFilterUnit,
    onLengthFilterUnitChange,
    lengthRangeStart,
    lengthRangeEnd,
    lengthRangeMin,
    lengthRangeMax,
    onLengthRangeStartChange,
    onLengthRangeEndChange,
    wordsToAvoid,
    onWordsToAvoidChange,
    isLexicalSearchMode,
  } = props

  // Persist landing-seen across visits — animation only plays on first visit.
  const [phase, setPhase] = useState<IntroPhase>(() => readLandingSeen() ? 'voice' : 'topic')
  const [voiceMode, setVoiceMode] = useState<VoiceMode>('stance')
  const [activeField, setActiveField] = useState<'topic' | 'claim'>('topic')
  const [essaySource, setEssaySource] = useState<EssaySource>('paste')
  const [composeTourOpen, setComposeTourOpen] = useState(false)
  const composeTourSeenRef = useRef(readTourSeen(COMPOSE_TOUR_SEEN_KEY))
  const composeTourAutoOpenedRef = useRef(false)
  const composeTourReturnModeRef = useRef<VoiceMode>('stance')
  const lastTutorialRequestIdRef = useRef(tutorialRequestId)
  const pendingManualTourRef = useRef(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const topicInputRef = useRef<HTMLInputElement | null>(null)
  const opinionInputRef = useRef<HTMLInputElement | null>(null)

  // Flash the matching on-screen key when the user types via real keyboard.
  const [flashedKey, setFlashedKey] = useState<string | null>(null)
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const flashKey = useCallback((id: string): void => {
    setFlashedKey(id)
    if (flashTimerRef.current) clearTimeout(flashTimerRef.current)
    flashTimerRef.current = setTimeout(() => setFlashedKey(null), 130)
  }, [])
  useEffect(() => () => {
    if (flashTimerRef.current) clearTimeout(flashTimerRef.current)
  }, [])
  const handleStanceEnter = useCallback((): void => {
    const trimmedTopic = topic.trim()
    const trimmedOpinion = opinion.trim()
    if (!trimmedTopic) return
    if (trimmedOpinion && !loading) {
      // Route through the same exit-animation path used by send-to-press
      // so the slip lifts off whether the reader clicks or hits Enter.
      setSlipExiting(prevExiting => {
        if (prevExiting) return prevExiting
        // The carriage-return ding has already been played by the caller
        // (handleInputKeyDown for Enter, the on-screen Enter key handler,
        // or handleSendToPress); follow it with the "sent" cue a beat
        // later so the two read as a sequence.
        window.setTimeout(() => playSend(), 180)
        window.setTimeout(() => onSubmitStance(), 360)
        return true
      })
      return
    }
    if (activeField === 'topic') {
      setActiveField('claim')
    }
  }, [activeField, loading, onSubmitStance, opinion, topic])
  const handleInputKeyDown = useCallback((event: ReactKeyboardEvent<HTMLInputElement>): void => {
    const key = event.key
    if (key === 'Enter') {
      event.preventDefault()
      flashKey('enter')
      playCarriageReturn()
      handleStanceEnter()
    }
    else if (key === 'Backspace') {
      flashKey('backspace')
      playKeyStrike()
    }
    else if (key === ' ') {
      flashKey('space')
      playKeyStrike()
    }
    else if (key.length === 1) {
      flashKey(key.toLowerCase())
      playKeyStrike()
    }
  }, [flashKey, handleStanceEnter])

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

  // Mark landing seen once user reaches 'done' or 'voice' for the first time.
  useEffect(() => {
    if (phase === 'done' || phase === 'voice') {
      markLandingSeen()
    }
  }, [phase])

  useEffect(() => {
    if (phase !== 'voice') return
    if (activeField === 'topic') {
      requestAnimationFrame(() => topicInputRef.current?.focus())
    } else {
      requestAnimationFrame(() => opinionInputRef.current?.focus())
    }
  }, [phase, activeField])

  const isVoice = phase === 'voice'

  // True for the brief window after `findVoice()` so we can run the
  // entrance animation (typewriter rises, controls fade up). The flag is
  // cleared after the longest animation has had time to complete.
  const [voiceEntering, setVoiceEntering] = useState(false)
  // True after the reader clicks "send to press"; the slip lifts off the
  // page toward the top while the search is dispatched, giving a sense
  // of motion into the pinned-slip position on the results page.
  const [slipExiting, setSlipExiting] = useState(false)

  const findVoice = (): void => {
    setActiveField('topic')
    setVoiceEntering(true)
    setPhase('voice')
  }

  useEffect(() => {
    if (!voiceEntering) return
    // Longest stagger above is ~970ms (0.42s delay + 0.55s duration).
    // Clear once we're safely past that so re-renders don't re-fire it.
    const t = setTimeout(() => setVoiceEntering(false), 1000)
    return () => clearTimeout(t)
  }, [voiceEntering])

  const startComposeTour = useCallback((manual = false): void => {
    if (phase !== 'voice') {
      pendingManualTourRef.current = manual
      return
    }
    pendingManualTourRef.current = false
    composeTourReturnModeRef.current = voiceMode
    setEssaySource('paste')
    setVoiceMode('stance')
    setComposeTourOpen(true)
  }, [phase, voiceMode])

  const closeComposeTour = useCallback((): void => {
    composeTourSeenRef.current = true
    composeTourAutoOpenedRef.current = true
    markTourSeen(COMPOSE_TOUR_SEEN_KEY)
    setComposeTourOpen(false)
    setVoiceMode(composeTourReturnModeRef.current)
  }, [])

  useEffect(() => {
    if (tutorialRequestId === lastTutorialRequestIdRef.current) return
    lastTutorialRequestIdRef.current = tutorialRequestId
    pendingManualTourRef.current = true
    startComposeTour(true)
  }, [startComposeTour, tutorialRequestId])

  useEffect(() => {
    if (phase !== 'voice') return
    if (pendingManualTourRef.current) {
      startComposeTour(true)
      return
    }
    if (!composeTourSeenRef.current && !composeTourAutoOpenedRef.current) {
      composeTourAutoOpenedRef.current = true
      startComposeTour(false)
    }
  }, [phase, startComposeTour])

  const handleComposeTourStep = useCallback((_step: SpotlightTourStep, index: number): void => {
    if (index >= 2) {
      setVoiceMode('essay')
      setEssaySource('paste')
    } else {
      setVoiceMode('stance')
    }
  }, [])

  const replay = (): void => {
    clearLandingSeen()
    setPhase('topic')
    setActiveField('topic')
  }

  const skipIntro = (): void => {
    markLandingSeen()
    setActiveField('topic')
    setVoiceEntering(true)
    setPhase('voice')
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

  const handleSendToPress = useCallback((): void => {
    if (slipExiting) return
    const canSubmit = voiceMode === 'stance' ? stanceCanSubmit : essayCanSubmit
    if (!canSubmit) return
    const submit = voiceMode === 'stance' ? onSubmitStance : onSubmitEssayDraft
    // Same audio sequence as pressing Enter on the on-screen typewriter:
    // the carriage-return bell rings as the slip is committed, followed by
    // the "sent" cue a beat later as it lifts off toward the press.
    playCarriageReturn()
    window.setTimeout(() => playSend(), 180)
    setSlipExiting(true)
    // Hold the slip's exit transition long enough for it to reach the
    // pinned-slip area before swapping pages — PinnedSlip's own
    // entrance picks up where this leaves off.
    window.setTimeout(() => {
      submit()
    }, 360)
  }, [essayCanSubmit, onSubmitEssayDraft, onSubmitStance, slipExiting, stanceCanSubmit, voiceMode])

  return (
    <div className="stage-shell" style={{ position: 'relative' }}>
      <div className="top-rail" style={{ flexShrink: 0 }}>
        <button type="button" className="top-rail-brand">hear! hear!</button>
        <div className="top-rail-links">
          <button type="button" className="active">search</button>
          <button type="button" onClick={onOpenExplore}>explore</button>
          <button type="button" onClick={onOpenAbout}>about</button>
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
          <button
            type="button"
            className="help-toggle"
            onClick={() => startComposeTour(true)}
            aria-label="Open tutorial"
            title="Open tutorial"
          >
            ?
          </button>
        </div>
      </div>
      <div className="top-rule" style={{ flexShrink: 0 }} />

      {/* INTRO MODE — fills the available stage area, slip in the middle */}
      {!isVoice && (
        <div style={{
          flex: 1,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 24,
          padding: '0 48px',
        }}>
          <div className="tracker" style={{ color: 'var(--accent)', letterSpacing: '0.32em' }}>
            a reading companion · since 2026
          </div>
          {/* Small draft slip with the cycling lines inside */}
          <div style={{
            background: 'var(--paper)',
            border: '1px solid var(--ink)',
            boxShadow: '0 8px 20px var(--shadow-mid)',
            padding: '24px 30px 26px',
            width: COMPOSE_SURFACE_WIDTH,
            maxWidth: '100%',
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
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
              color: 'var(--ink-faint)',
              display: 'flex',
              justifyContent: 'space-between',
            }}>
              <span>hear! hear! / draft</span>
              <span>№ 001</span>
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
          </div>
          <div style={{
            marginTop: 18,
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
              color: 'var(--ink)',
            }}>
              <span>find your voice</span>
              <span style={{ fontSize: 18, animation: 'tw-bob 1.6s ease-in-out infinite' }}>↓</span>
            </button>
          </div>
        </div>
      )}

      {/* VOICE MODE — flex column that scales to viewport */}
      {isVoice && (
        <div
          className={`${voiceEntering ? 'voice-entering' : ''} ${slipExiting ? 'slip-exiting' : ''}`.trim()}
          style={{
            flex: 1,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            padding: '14px 48px 12px',
            gap: 6,
            overflowX: 'hidden',
            overflowY: 'auto',
          }}
        >
          {/* Mode toggle */}
          <div className="voice-mode-toggle" data-tour="compose-mode-toggle" style={{
            display: 'flex',
            justifyContent: 'center',
            gap: 0,
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 11,
            letterSpacing: '0.28em',
            textTransform: 'uppercase',
            transform: 'translateY(0)',
            transition: 'opacity 0.5s ease 0.3s',
            zIndex: 5,
            flexShrink: 0,
          }}>
            <button type="button" onClick={() => setVoiceMode('stance')} style={{
              padding: '8px 22px',
              border: '1px solid var(--ink)',
              background: voiceMode === 'stance' ? 'var(--ink)' : 'transparent',
              color: voiceMode === 'stance' ? 'var(--paper)' : 'var(--ink)',
              fontFamily: 'inherit',
              fontSize: 'inherit',
              letterSpacing: 'inherit',
              textTransform: 'inherit',
              cursor: 'pointer',
            }}>topic & stance</button>
            <button type="button" onClick={() => setVoiceMode('essay')} style={{
              padding: '8px 22px',
              border: '1px solid var(--ink)',
              borderLeft: 0,
              background: voiceMode === 'essay' ? 'var(--ink)' : 'transparent',
              color: voiceMode === 'essay' ? 'var(--paper)' : 'var(--ink)',
              fontFamily: 'inherit',
              fontSize: 'inherit',
              letterSpacing: 'inherit',
              textTransform: 'inherit',
              cursor: 'pointer',
            }}>essay</button>
          </div>

          <div style={{
            flex: 1,
            minHeight: 0,
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            padding: '8px 0',
          }}>
            {/* Paper sheet — the stance slip matches the intro animation slip.
                On send-to-press the slip lifts toward where the pinned slip
                will sit on the results page, paired with PinnedSlip's
                entrance from above so the motion reads as one continuous
                flight. The transform/opacity are set inline (not via CSS
                class) so they layer cleanly over the existing min-height
                transition the slip needs for essay-mode resize. */}
            <div className="voice-slip" data-tour={voiceMode === 'essay' ? 'compose-essay-slip' : 'compose-stance-slip'} style={{
              marginTop: 0,
              width: COMPOSE_SURFACE_WIDTH,
              maxWidth: '100%',
              minHeight: voiceMode === 'essay' ? 380 : undefined,
              height: 'auto',
              background: 'var(--paper)',
              border: voiceMode === 'essay' ? '1px solid rgba(var(--ink-rgb),0.6)' : '1px solid var(--ink)',
              borderBottom: '1px solid var(--ink)',
              boxShadow: voiceMode === 'essay'
                ? '0 8px 20px var(--shadow-mid)'
                : '0 8px 20px var(--shadow-mid)',
              transition: 'min-height 0.5s ease, transform 460ms cubic-bezier(.55,.02,.2,1), opacity 460ms cubic-bezier(.55,.02,.2,1)',
              transform: slipExiting ? 'translateY(-32vh) scale(0.945)' : 'translateY(0) scale(1)',
              opacity: slipExiting ? 0 : 1,
              pointerEvents: slipExiting ? 'none' : undefined,
              padding: voiceMode === 'essay' ? '36px 36px 24px' : '24px 30px 26px',
              display: 'flex',
              flexDirection: 'column',
              gap: voiceMode === 'essay' ? 22 : 18,
              overflow: 'visible',
              position: 'relative',
              zIndex: 1,
              flexShrink: 0,
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
                color: 'var(--ink-faint)',
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
                    onKeyDown={handleInputKeyDown}
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
                    onKeyDown={handleInputKeyDown}
                    placeholder="an opinion…"
                    inputRef={(el) => { opinionInputRef.current = el }}
                  />
                </>
              )}

              {voiceMode === 'essay' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 22, fontStyle: 'italic', color: 'var(--ink-soft)' }}>An essay,</span>
                    <div data-tour="compose-essay-source" style={{ display: 'flex', fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase' }}>
                      <button type="button" onClick={() => setEssaySource('paste')} style={{
                        padding: '4px 12px',
                        border: '1px solid var(--ink)',
                        background: essaySource === 'paste' ? 'var(--ink)' : 'transparent',
                        color: essaySource === 'paste' ? 'var(--paper)' : 'var(--ink)',
                        fontFamily: 'inherit',
                        fontSize: 'inherit',
                        letterSpacing: 'inherit',
                        textTransform: 'inherit',
                        cursor: 'pointer',
                      }}>type / paste</button>
                      <button type="button" onClick={() => setEssaySource('envelope')} style={{
                        padding: '4px 12px',
                        border: '1px solid var(--ink)',
                        borderLeft: 0,
                        background: essaySource === 'envelope' ? 'var(--ink)' : 'transparent',
                        color: essaySource === 'envelope' ? 'var(--paper)' : 'var(--ink)',
                        fontFamily: 'inherit',
                        fontSize: 'inherit',
                        letterSpacing: 'inherit',
                        textTransform: 'inherit',
                        cursor: 'pointer',
                      }}>pdf envelope</button>
                    </div>
                  </div>

                  {essaySource === 'paste' && (
                    <div style={{ position: 'relative', border: '1px solid var(--ink-faint)', background: 'var(--paper)' }}>
                      <div aria-hidden style={{
                        position: 'absolute',
                        inset: 0,
                        backgroundImage: 'repeating-linear-gradient(0deg, transparent 0, transparent 25px, var(--shadow-mid) 26px)',
                        pointerEvents: 'none',
                      }} />
                      <textarea
                        value={essayText}
                        onChange={(event) => onEssayTextChange(event.target.value)}
                        onKeyDown={(event) => {
                          // Match the typewriter feel from the stance flow:
                          // each key strike clicks, Enter rings the carriage bell.
                          if (event.key === 'Enter') playCarriageReturn()
                          else if (event.key === 'Backspace' || event.key === ' ' || event.key.length === 1) {
                            playKeyStrike()
                          }
                        }}
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
                          color: 'var(--ink)',
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

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'Special Elite', monospace", fontSize: 10, color: 'var(--ink-mute)' }}>
                    <span>{essaySource === 'paste' ? `${essayPasteWordCount} words` : ' '}</span>
                    <span>{effectiveStanceMethod === 'nli' ? 'thesis pick · stage 1' : 'sub-editor reads the whole essay'}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Typewriter — held close to the slip without covering its contents. */}
            {voiceMode === 'stance' && (
              <div className="landing-typewriter-slot" data-tour="compose-typewriter">
                <RisingTypewriter
                  disabled={loading}
                  flashedKey={flashedKey}
                  onType={(char) => {
                    flashKey(char === ' ' ? 'space' : char)
                    playKeyStrike()
                    if (activeField === 'topic') {
                      onTopicChange((prev) => prev + char)
                    } else {
                      onOpinionChange((prev) => prev + char)
                    }
                  }}
                  onBackspace={() => {
                    playKeyStrike()
                    if (activeField === 'topic') {
                      onTopicChange((prev) => prev.slice(0, -1))
                    } else {
                      onOpinionChange((prev) => prev.slice(0, -1))
                    }
                  }}
                  onEnter={() => {
                    playCarriageReturn()
                    handleStanceEnter()
                  }}
                />
              </div>
            )}

            {/* Filters row */}
            <div className="voice-filters" data-tour="compose-filters" style={{ display: 'flex', justifyContent: 'center', flexShrink: 0 }}>
              <FilterRow
                yearStart={yearStart}
                yearEnd={yearEnd}
                minYear={minYear}
                maxYear={maxYear}
                onYearStartChange={onYearStartChange}
                onYearEndChange={onYearEndChange}
                lengthFilterUnit={lengthFilterUnit}
                onLengthFilterUnitChange={onLengthFilterUnitChange}
                lengthRangeStart={lengthRangeStart}
                lengthRangeEnd={lengthRangeEnd}
                lengthRangeMin={lengthRangeMin}
                lengthRangeMax={lengthRangeMax}
                onLengthRangeStartChange={onLengthRangeStartChange}
                onLengthRangeEndChange={onLengthRangeEndChange}
                wordsToAvoid={wordsToAvoid}
                onWordsToAvoidChange={onWordsToAvoidChange}
                isLexicalSearchMode={isLexicalSearchMode}
              />
            </div>

            {/* Send to press button */}
            <div className="voice-send" data-tour="compose-actions" style={{ display: 'flex', justifyContent: 'center', flexShrink: 0 }}>
              <button
                type="button"
                onClick={handleSendToPress}
                disabled={voiceMode === 'stance' ? !stanceCanSubmit : !essayCanSubmit}
                style={{
                  background: 'var(--ink)',
                  color: 'var(--paper)',
                  border: '1px solid var(--ink)',
                  padding: '12px 28px',
                  fontFamily: "'IM Fell DW Pica SC', serif",
                  fontSize: 12,
                  letterSpacing: '0.32em',
                  textTransform: 'uppercase',
                  cursor: (voiceMode === 'stance' ? stanceCanSubmit : essayCanSubmit) ? 'pointer' : 'not-allowed',
                  opacity: (voiceMode === 'stance' ? stanceCanSubmit : essayCanSubmit) ? 1 : 0.5,
                }}
              >
                {voiceMode === 'essay' && effectiveStanceMethod === 'nli'
                  ? 'extract a thesis & send to press →'
                  : 'send to press →'}
              </button>
            </div>
          </div>

          {/* Bottom row: replay intro on the left, instrument settings on the right */}
          <div className="voice-bottom-row" style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            width: '100%',
            flexShrink: 0,
            gap: 24,
            minHeight: 44,
          }}>
            <button type="button" onClick={replay} style={{
              background: 'transparent',
              border: '1px solid var(--ink)',
              padding: '6px 14px',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 9,
              letterSpacing: '0.28em',
              textTransform: 'uppercase',
              color: 'var(--ink)',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}>↻ replay intro</button>
            <div data-tour="compose-settings" style={{
              display: 'flex',
              alignItems: 'center',
              gap: 14,
              flexShrink: 0,
            }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
                <div className="tracker" style={{ color: 'var(--accent)', letterSpacing: '0.32em', fontSize: 9 }}>The Instrument</div>
                <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 13, color: 'var(--ink)', textAlign: 'right', lineHeight: 1.5 }}>
                  reads with <PersonaName persona={effectiveRetrievalModel} /> &nbsp;·&nbsp; judges with <PersonaName persona={effectiveStanceMethod} />
                  <br />
                  <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 10, color: 'var(--ink-mute)' }}>
                    chunks · <strong style={{ color: 'var(--ink)' }}>{chunksLabel}</strong>
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={onOpenSettings}
                style={{
                  background: 'transparent',
                  border: '1px solid var(--ink)',
                  padding: '6px 12px',
                  cursor: 'pointer',
                  fontFamily: "'IM Fell DW Pica SC', serif",
                  fontSize: 9,
                  letterSpacing: '0.28em',
                  textTransform: 'uppercase',
                  color: 'var(--ink)',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  whiteSpace: 'nowrap',
                }}
              >
                ⚙ settings
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer — intro can be skipped before the compose screen appears. */}
      {!isVoice && (
      <div style={{ flexShrink: 0, padding: '8px 48px 12px', display: 'flex', justifyContent: 'center' }}>
        <button type="button" onClick={skipIntro} style={{
          background: 'transparent',
          border: '1px solid var(--ink)',
          padding: '6px 14px',
          fontFamily: "'IM Fell DW Pica SC', serif",
          fontSize: 9,
          letterSpacing: '0.28em',
          textTransform: 'uppercase',
          color: 'var(--ink)',
          cursor: 'pointer',
        }}>skip intro →</button>
      </div>
      )}

      <SpotlightTour
        open={composeTourOpen}
        steps={COMPOSE_TOUR_STEPS}
        onClose={closeComposeTour}
        onStepChange={handleComposeTourStep}
      />
    </div>
  )
}

export default LandingFlow
