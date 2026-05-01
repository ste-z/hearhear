/* Renders a persona name (e.g. "Mrs. Calder") with a wavy underline and a
   hover/focus tooltip revealing:
   - the actual method name (TF·IDF / SVD / MiniLM / NLI / LLM)
   - the persona's working-style description
   - their portrait
   The tooltip is rendered in a `document.body` portal so it can't be clipped
   by `overflow: hidden` ancestors, and is positioned via the trigger's
   bounding rect — auto-flipping above/below and clamping horizontally to
   stay inside the viewport. */

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import oldHewittPortrait from './assets/personas/old-hewitt.svg'
import mrsCalderPortrait from './assets/personas/mrs-calder.svg'
import youngParkPortrait from './assets/personas/young-park.svg'
import juniorReevePortrait from './assets/personas/junior-reeve.svg'
import editorHollisPortrait from './assets/personas/editor-hollis.svg'

type PersonaId = 'tfidf' | 'svd' | 'minilm' | 'nli' | 'llm'

type PersonaInfo = {
  name: string
  method: string
  style: string
  portrait: string
}

const PERSONA_INFO: Record<PersonaId, PersonaInfo> = {
  tfidf: {
    name: 'Old Hewitt',
    method: 'TF·IDF',
    style: 'A literal lexical compositor — sets type by exact word matches and inverse document frequency.',
    portrait: oldHewittPortrait,
  },
  svd: {
    name: 'Mrs. Calder',
    method: 'SVD',
    style: 'A thematic semantic compositor — uses Singular Value Decomposition over the term–document matrix to catch kinship between concepts.',
    portrait: mrsCalderPortrait,
  },
  minilm: {
    name: 'Young Park',
    method: 'MiniLM',
    style: 'A contextual semantic compositor — uses a MiniLM transformer to embed full sentences for context-aware retrieval.',
    portrait: youngParkPortrait,
  },
  nli: {
    name: 'Junior Reeve',
    method: 'NLI (DeBERTa)',
    style: 'A fast natural-language-inference sub-editor — marks each sentence ✓ entail / ○ neutral / ✗ contradict against your thesis.',
    portrait: juniorReevePortrait,
  },
  llm: {
    name: 'Editor Hollis',
    method: 'LLM (gpt-oss-20b)',
    style: 'A considered LLM sub-editor — reads the full essay and weighs each article holistically.',
    portrait: editorHollisPortrait,
  },
}

const TOOLTIP_WIDTH = 280
const TOOLTIP_GAP = 10
const VIEWPORT_PAD = 12

type Position = {
  left: number
  top: number
  // Whether the tooltip's caret is on its bottom edge (i.e. the tooltip is
  // ABOVE the name) or top edge (tooltip is BELOW the name).
  caret: 'bottom' | 'top'
}

function PersonaTooltip({ info, anchor }: { info: PersonaInfo; anchor: DOMRect }): JSX.Element {
  const tipRef = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState<Position | null>(null)

  useLayoutEffect(() => {
    const measureAndPlace = (): void => {
      const el = tipRef.current
      if (!el) return
      const tipHeight = el.offsetHeight
      const viewportH = window.innerHeight
      const viewportW = window.innerWidth

      // Decide whether to render above or below the anchor based on free space.
      const spaceAbove = anchor.top - VIEWPORT_PAD
      const spaceBelow = viewportH - anchor.bottom - VIEWPORT_PAD
      const showAbove = spaceAbove >= tipHeight + TOOLTIP_GAP || spaceAbove >= spaceBelow

      const top = showAbove
        ? Math.max(VIEWPORT_PAD, anchor.top - tipHeight - TOOLTIP_GAP)
        : Math.min(viewportH - tipHeight - VIEWPORT_PAD, anchor.bottom + TOOLTIP_GAP)

      // Centre horizontally on the trigger, then clamp to the viewport.
      const anchorCentre = anchor.left + anchor.width / 2
      const rawLeft = anchorCentre - TOOLTIP_WIDTH / 2
      const clampedLeft = Math.max(
        VIEWPORT_PAD,
        Math.min(viewportW - TOOLTIP_WIDTH - VIEWPORT_PAD, rawLeft),
      )

      setPos({ left: clampedLeft, top, caret: showAbove ? 'bottom' : 'top' })
    }

    measureAndPlace()
    // Re-measure on scroll/resize while the tooltip is mounted so it tracks
    // the anchor instead of getting stranded.
    const onChange = (): void => measureAndPlace()
    window.addEventListener('scroll', onChange, true)
    window.addEventListener('resize', onChange)
    return () => {
      window.removeEventListener('scroll', onChange, true)
      window.removeEventListener('resize', onChange)
    }
  }, [anchor])

  // Position the caret horizontally over the trigger if the tooltip got
  // clamped sideways — otherwise it would float off-centre with no anchor.
  const anchorCentre = anchor.left + anchor.width / 2
  const caretX = pos
    ? Math.max(12, Math.min(TOOLTIP_WIDTH - 12, anchorCentre - pos.left))
    : TOOLTIP_WIDTH / 2

  // Build the caret style. We always set the same set of properties on
  // every render and use the multi-value `borderColor` shorthand so React
  // never sees a shorthand-vs-longhand transition (the browser will
  // otherwise collapse longhand entries and trip React's reconciler warning).
  const caretIsBottom = pos?.caret === 'bottom'
  const caretStyle: React.CSSProperties = {
    position: 'absolute',
    left: caretX,
    top: caretIsBottom ? '100%' : 'auto',
    bottom: caretIsBottom ? 'auto' : '100%',
    width: 0,
    height: 0,
    transform: 'translateX(-50%)',
    borderWidth: 6,
    borderStyle: 'solid',
    borderColor: caretIsBottom
      ? 'var(--ink) transparent transparent transparent'
      : 'transparent transparent var(--ink) transparent',
  }

  return createPortal(
    <div
      ref={tipRef}
      role="tooltip"
      style={{
        position: 'fixed',
        left: pos?.left ?? -9999,
        top: pos?.top ?? -9999,
        width: TOOLTIP_WIDTH,
        background: 'var(--ink)',
        color: 'var(--paper)',
        padding: '12px 14px',
        boxShadow: '0 12px 28px rgba(var(--ink-rgb), 0.34)',
        fontFamily: "'IM Fell DW Pica SC', serif",
        fontSize: 10,
        letterSpacing: '0.2em',
        textTransform: 'uppercase',
        textDecoration: 'none',
        lineHeight: 1.5,
        zIndex: 9999,
        pointerEvents: 'none',
        opacity: pos ? 1 : 0,
        transition: 'opacity 140ms ease',
      }}
    >
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <img
          src={info.portrait}
          alt={info.name}
          style={{
            width: 72,
            height: 72,
            objectFit: 'contain',
            background: 'var(--paper)',
            borderWidth: 1,
            borderStyle: 'solid',
            borderColor: 'rgba(var(--paper-rgb),0.2)',
            flexShrink: 0,
            padding: 2,
            boxSizing: 'border-box',
          }}
        />
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontWeight: 700, letterSpacing: '0.24em', fontSize: 10 }}>{info.name}</div>
          <div style={{ marginTop: 3, color: 'rgba(var(--paper-rgb),0.74)', fontSize: 9 }}>{info.method}</div>
        </div>
      </div>
      <div
        style={{
          marginTop: 10,
          paddingTop: 10,
          borderTop: '1px solid rgba(var(--paper-rgb),0.18)',
          fontFamily: "'IM Fell English', serif",
          fontStyle: 'italic',
          fontSize: 13,
          letterSpacing: 'normal',
          textTransform: 'none',
          color: 'rgba(var(--paper-rgb),0.92)',
          lineHeight: 1.45,
        }}
      >
        {info.style}
      </div>
      {/* caret pointing back at the anchor */}
      <span aria-hidden style={caretStyle} />
    </div>,
    document.body,
  )
}

export function PersonaName({
  persona,
  override,
  className = '',
}: {
  persona: PersonaId
  // Optional override for display name (rarely needed — defaults to PERSONA_INFO[persona].name)
  override?: string
  className?: string
}): JSX.Element {
  const info = PERSONA_INFO[persona]
  const display = override ?? info.name
  const triggerRef = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const [anchor, setAnchor] = useState<DOMRect | null>(null)

  const updateAnchor = (): void => {
    const el = triggerRef.current
    if (!el) return
    setAnchor(el.getBoundingClientRect())
  }

  // Refresh the anchor rect each time the tooltip is shown — the trigger
  // may have moved since last hover (scroll, layout shift, etc.).
  useEffect(() => {
    if (!open) return
    updateAnchor()
  }, [open])

  return (
    <>
      <span
        ref={triggerRef}
        className={`persona-name ${className}`}
        tabIndex={0}
        onMouseEnter={() => { updateAnchor(); setOpen(true) }}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => { updateAnchor(); setOpen(true) }}
        onBlur={() => setOpen(false)}
      >
        {display}
      </span>
      {open && anchor && <PersonaTooltip info={info} anchor={anchor} />}
    </>
  )
}

export default PersonaName
