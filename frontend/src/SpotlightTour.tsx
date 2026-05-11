import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'

export type SpotlightTourStep = {
  target: string
  title: string
  body: string
  placement?: 'auto' | 'left' | 'right' | 'top' | 'bottom'
  durationMs?: number
}

type SpotlightRect = {
  top: number
  left: number
  width: number
  height: number
}

const DEFAULT_STEP_DURATION_MS = 4700
const SPOTLIGHT_PAD = 10
const PANEL_WIDTH = 320
const PANEL_GAP = 18

function getTargetSelector(target: string): string {
  return `[data-tour="${target}"]`
}

function measureElement(target: string): SpotlightRect | null {
  if (typeof document === 'undefined') return null
  const element = document.querySelector(getTargetSelector(target))
  if (!(element instanceof HTMLElement)) return null
  const rect = element.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return null
  return {
    top: Math.max(8, rect.top - SPOTLIGHT_PAD),
    left: Math.max(8, rect.left - SPOTLIGHT_PAD),
    width: Math.min(window.innerWidth - 16, rect.width + SPOTLIGHT_PAD * 2),
    height: Math.min(window.innerHeight - 16, rect.height + SPOTLIGHT_PAD * 2),
  }
}

function getPanelPosition(rect: SpotlightRect | null, placement: SpotlightTourStep['placement']): CSSProperties {
  const viewportWidth = typeof window === 'undefined' ? 1200 : window.innerWidth
  const viewportHeight = typeof window === 'undefined' ? 800 : window.innerHeight
  const panelHeight = Math.min(260, viewportHeight - 32)
  if (!rect) {
    return {
      top: Math.max(16, (viewportHeight - panelHeight) / 2),
      left: Math.max(16, (viewportWidth - PANEL_WIDTH) / 2),
      width: Math.min(PANEL_WIDTH, viewportWidth - 32),
    }
  }

  const preferred = placement === 'auto'
    ? (rect.left + rect.width / 2 < viewportWidth / 2 ? 'right' : 'left')
    : (placement ?? 'auto')
  const clampedTop = Math.max(16, Math.min(viewportHeight - panelHeight - 16, rect.top + rect.height / 2 - panelHeight / 2))
  const rightLeft = rect.left + rect.width + PANEL_GAP
  const leftLeft = rect.left - PANEL_WIDTH - PANEL_GAP
  const bottomTop = rect.top + rect.height + PANEL_GAP
  const topTop = rect.top - panelHeight - PANEL_GAP

  if (preferred === 'right' && rightLeft + PANEL_WIDTH <= viewportWidth - 16) {
    return { top: clampedTop, left: rightLeft, width: PANEL_WIDTH }
  }
  if (preferred === 'left' && leftLeft >= 16) {
    return { top: clampedTop, left: leftLeft, width: PANEL_WIDTH }
  }
  if (preferred === 'bottom' && bottomTop + panelHeight <= viewportHeight - 16) {
    return { top: bottomTop, left: Math.max(16, Math.min(viewportWidth - PANEL_WIDTH - 16, rect.left)), width: PANEL_WIDTH }
  }
  if (preferred === 'top' && topTop >= 16) {
    return { top: topTop, left: Math.max(16, Math.min(viewportWidth - PANEL_WIDTH - 16, rect.left)), width: PANEL_WIDTH }
  }

  if (rightLeft + PANEL_WIDTH <= viewportWidth - 16) return { top: clampedTop, left: rightLeft, width: PANEL_WIDTH }
  if (leftLeft >= 16) return { top: clampedTop, left: leftLeft, width: PANEL_WIDTH }
  if (bottomTop + panelHeight <= viewportHeight - 16) return { top: bottomTop, left: Math.max(16, Math.min(viewportWidth - PANEL_WIDTH - 16, rect.left)), width: PANEL_WIDTH }
  return {
    top: Math.max(16, Math.min(viewportHeight - panelHeight - 16, rect.top + rect.height + PANEL_GAP)),
    left: Math.max(16, Math.min(viewportWidth - PANEL_WIDTH - 16, (viewportWidth - PANEL_WIDTH) / 2)),
    width: Math.min(PANEL_WIDTH, viewportWidth - 32),
  }
}

export function SpotlightTour({
  open,
  steps,
  onClose,
  onStepChange,
  autoAdvance = false,
}: {
  open: boolean
  steps: SpotlightTourStep[]
  onClose: () => void
  onStepChange?: (step: SpotlightTourStep, index: number) => void
  autoAdvance?: boolean
}): JSX.Element | null {
  const [stepIndex, setStepIndex] = useState(0)
  const [rect, setRect] = useState<SpotlightRect | null>(null)
  const rafRef = useRef<number | null>(null)
  const step = steps[stepIndex] ?? null

  const updateRect = useCallback((scrollTarget = false) => {
    if (!step) {
      setRect(null)
      return
    }
    const element = document.querySelector(getTargetSelector(step.target))
    if (scrollTarget && element instanceof HTMLElement) {
      element.scrollIntoView({ block: 'center', inline: 'center', behavior: 'smooth' })
    }
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = requestAnimationFrame(() => {
        setRect(measureElement(step.target))
      })
    })
  }, [step])

  useEffect(() => {
    if (!open) {
      setStepIndex(0)
      setRect(null)
      return undefined
    }
    setStepIndex(0)
    return undefined
  }, [open, steps])

  useEffect(() => {
    if (!open || !step) return undefined
    onStepChange?.(step, stepIndex)
    const settle = window.setTimeout(() => updateRect(true), 90)
    return () => window.clearTimeout(settle)
  }, [onStepChange, open, step, stepIndex, updateRect])

  useEffect(() => {
    if (!open || !step) return undefined
    const handleRefresh = (): void => updateRect()
    window.addEventListener('resize', handleRefresh)
    window.addEventListener('scroll', handleRefresh, true)
    return () => {
      window.removeEventListener('resize', handleRefresh)
      window.removeEventListener('scroll', handleRefresh, true)
    }
  }, [open, step, updateRect])

  useEffect(() => {
    if (!open || !step || !autoAdvance) return undefined
    const timer = window.setTimeout(() => {
      setStepIndex((current) => {
        if (current >= steps.length - 1) {
          onClose()
          return current
        }
        return current + 1
      })
    }, step.durationMs ?? DEFAULT_STEP_DURATION_MS)
    return () => window.clearTimeout(timer)
  }, [autoAdvance, onClose, open, step, steps.length])

  useEffect(() => {
    if (!open) return undefined
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [onClose, open, steps.length])

  useEffect(() => () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
  }, [])

  const panelPosition = useMemo(() => getPanelPosition(rect, step?.placement ?? 'auto'), [rect, step?.placement])

  if (!open || !step) return null

  const progress = steps.length <= 1 ? 100 : ((stepIndex + 1) / steps.length) * 100

  return (
    <section className="spotlight-tour" role="dialog" aria-live="polite" aria-label="Guided tutorial">
      {rect && (
        <div
          className="spotlight-tour-hole"
          style={{
            top: rect.top,
            left: rect.left,
            width: rect.width,
            height: rect.height,
          }}
        />
      )}
      <article className="spotlight-tour-card" style={panelPosition}>
        <div className="spotlight-tour-kicker">guided tour · {stepIndex + 1} / {steps.length}</div>
        <h2>{step.title}</h2>
        <p>{step.body}</p>
        <div className="spotlight-tour-progress" aria-hidden="true">
          <span style={{ width: `${progress}%` }} />
        </div>
        <div className="spotlight-tour-actions">
          <button type="button" onClick={onClose}>skip</button>
          <div>
            <button
              type="button"
              onClick={() => setStepIndex((current) => Math.max(0, current - 1))}
              disabled={stepIndex === 0}
            >
              prev
            </button>
            <button
              type="button"
              onClick={() => {
                if (stepIndex >= steps.length - 1) onClose()
                else setStepIndex((current) => Math.min(steps.length - 1, current + 1))
              }}
            >
              next
            </button>
          </div>
        </div>
      </article>
    </section>
  )
}

export default SpotlightTour
