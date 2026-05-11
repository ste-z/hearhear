import { useEffect, useRef, useState } from 'react'
import type { RetrievalModel } from './types'
import PersonaName from './PersonaName'
import oldHewittPortrait from './assets/personas/old-hewitt.svg'
import mrsCalderPortrait from './assets/personas/mrs-calder.svg'
import youngParkPortrait from './assets/personas/young-park.svg'
import juniorReevePortrait from './assets/personas/junior-reeve.svg'
import editorHollisPortrait from './assets/personas/editor-hollis.svg'

export type StanceMethod = 'nli' | 'llm'
export type FrontendChunkingMode = 'none' | 'semantic'
export type RerankSelectionMode = 'manual' | 'automatic'
export type LengthFilterUnit = 'characters' | 'words' | 'reading_time'

export type SettingsTrayProps = {
  open: boolean
  onClose: () => void

  // chunking
  chunkingMode: FrontendChunkingMode
  supportedChunkingModes: FrontendChunkingMode[]
  onChunkingModeChange: (next: FrontendChunkingMode) => void

  // retrieval (compositor)
  retrievalModel: RetrievalModel
  effectiveRetrievalModel: RetrievalModel
  supportedRetrievalModels: RetrievalModel[]
  onRetrievalModelChange: (next: RetrievalModel) => void

  // bench / rerank handoff
  rerankSelectionMode: RerankSelectionMode
  onRerankSelectionModeChange: (next: RerankSelectionMode) => void
  autoRerankThreshold: number
  onAutoRerankThresholdChange: (value: number) => void
  rerankTopK: number
  onRerankTopKChange: (value: number) => void
  maxAutoRerankCandidates: number

  // chunk count config (visible only when chunking)
  chunkCandidateTopK: number
  onChunkCandidateTopKChange: (value: number) => void
  maxChunkCandidateTopK: number
  chunkArticleTopK: number
  onChunkArticleTopKChange: (value: number) => void

  // sub-editor (stance)
  stanceMethod: StanceMethod
  effectiveStanceMethod: StanceMethod
  supportedStanceMethods: StanceMethod[]
  llmAgreementAvailable: boolean
  onStanceMethodChange: (next: StanceMethod) => void
  llmLabelIrrelevant: boolean
  onLlmLabelIrrelevantChange: (next: boolean) => void

  // weights (re-ranking)
  topicWeight: number
  stanceWeight: number
  recencyWeight: number
  onWeightsChange: (next: { topicWeight: number; stanceWeight: number; recencyWeight: number }) => void

  onResetDefaults?: () => void
  onApply: () => void
  // When the user has changed any setting since the last Apply, the tray
  // shows an active "Apply" button — clicking it triggers the backend
  // preload / unload pass for the new mode. Otherwise the bottom button
  // is just a passive "Close" affordance.
  settingsDirty?: boolean
  isApplying?: boolean
}

const STEP_LABELS = [
  'I · Chunking',
  'II · The Compositor',
  'III · Across the bench',
  'IV · The Sub-Editor',
  'V · Re-ranking',
] as const

const COMPOSITOR_PERSONAS: Record<RetrievalModel, { name: string; years: string; desc: string; mode: string; portrait: string }> = {
  tfidf: { name: 'Old Hewitt', years: '40 yrs', desc: 'Sets type by the literal letter. Inflexible and strict. Knows every word in the archive.', mode: 'TF·IDF', portrait: oldHewittPortrait },
  svd: { name: 'Mrs. Calder', years: '22 yrs', desc: 'Catches kinship between words — themes more than letters. Knows the archive well.', mode: 'SVD', portrait: mrsCalderPortrait },
  minilm: { name: 'Young Park', years: '4 yrs', desc: 'Trained abroad. Reads context, not just keys.', mode: 'MiniLM', portrait: youngParkPortrait },
}

const SUB_EDITOR_PERSONAS: Record<StanceMethod, { name: string; tool: string; desc: string; mode: string; portrait: string }> = {
  nli: { name: 'Junior Reeve', tool: 'a red pencil & checklist', desc: 'Marks each line ✓ agrees, ○ neutral, ✗ disagrees. Quick. Strict.', mode: 'NLI (DeBERTa)', portrait: juniorReevePortrait },
  llm: { name: 'Editor Hollis', tool: 'a green-shade lamp', desc: 'Reads the whole brief by lamplight. Slower. Considered.', mode: 'LLM (gpt-oss-20b)', portrait: editorHollisPortrait },
}

function StepPanel({
  idx,
  title,
  subtitle,
  stepRef,
  children,
}: {
  idx: number
  title: string
  subtitle: string
  stepRef: React.RefObject<HTMLDivElement>
  children: React.ReactNode
}): JSX.Element {
  const numerals = ['I', 'II', 'III', 'IV', 'V', 'VI']
  return (
    <section
      ref={stepRef}
      style={{ minHeight: 540, padding: '36px 40px 28px', borderBottom: '1px solid rgba(var(--ink-rgb), 0.2)' }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, marginBottom: 18 }}>
        <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 26, fontStyle: 'italic', color: 'var(--accent)' }}>{numerals[idx]}</span>
        <div>
          <h3 style={{ fontFamily: "'IM Fell English', serif", fontSize: 28, fontWeight: 400, margin: 0 }}>{title}</h3>
          <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 10, letterSpacing: '0.28em', textTransform: 'uppercase', color: 'var(--ink-mute)', marginTop: 2 }}>{subtitle}</div>
        </div>
      </div>
      {children}
    </section>
  )
}

function ModalRoleSelector<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: T
  onChange: (next: T) => void
  options: Array<{ id: T; label: string; sub: string; disabled?: boolean }>
}): JSX.Element {
  return (
    <div style={{ marginTop: 12, width: '100%', maxWidth: 520 }}>
      <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'var(--ink-mute)', marginBottom: 6 }}>{label}</div>
      <div style={{ display: 'flex', borderTop: '1px solid var(--ink)', borderBottom: '1px solid var(--ink)' }}>
        {options.map((o, i) => {
          const active = value === o.id
          return (
            <button
              key={o.id}
              type="button"
              onClick={() => !o.disabled && onChange(o.id)}
              disabled={o.disabled}
              style={{
                flex: 1,
                background: active ? 'var(--ink)' : 'transparent',
                color: active ? 'var(--paper)' : (o.disabled ? 'var(--ink-faint)' : 'var(--ink)'),
                border: 0,
                borderLeft: i === 0 ? 0 : '1px solid var(--ink)',
                padding: '10px 8px',
                cursor: o.disabled ? 'not-allowed' : 'pointer',
                fontFamily: "'IM Fell English', serif",
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
                opacity: o.disabled ? 0.5 : 1,
              }}
            >
              <span style={{ fontSize: 14 }}>{o.label}</span>
              <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 9, opacity: 0.8 }}>{o.sub}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function SmallSwitch({
  label,
  hint,
  checked,
  disabled = false,
  onChange,
}: {
  label: string
  hint?: string
  checked: boolean
  disabled?: boolean
  onChange: (next: boolean) => void
}): JSX.Element {
  return (
    <div style={{ marginTop: 16, maxWidth: 520, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 18 }}>
      <div>
        <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: disabled ? 'var(--ink-faint)' : 'var(--ink-mute)' }}>{label}</div>
        {hint && (
          <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: 'var(--ink-mute)', marginTop: 3, lineHeight: 1.35 }}>{hint}</div>
        )}
      </div>
      <button
        type="button"
        role="switch"
        aria-label={label}
        aria-checked={checked}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        style={{
          width: 64,
          height: 28,
          flexShrink: 0,
          border: '1px solid var(--ink)',
          background: checked ? 'var(--ink)' : 'transparent',
          color: checked ? 'var(--paper)' : 'var(--ink)',
          padding: 3,
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.45 : 1,
          display: 'flex',
          justifyContent: checked ? 'flex-end' : 'flex-start',
          alignItems: 'center',
        }}
      >
        <span
          aria-hidden
          style={{
            width: 20,
            height: 20,
            background: checked ? 'var(--paper)' : 'var(--ink)',
            display: 'block',
          }}
        />
      </button>
    </div>
  )
}

function SliderField({
  label,
  unit,
  min,
  max,
  step,
  value,
  onChange,
  fmt,
  hint,
}: {
  label: string
  unit?: string
  min: number
  max: number
  step: number
  value: number
  onChange: (next: number) => void
  fmt?: (value: number) => string
  hint?: string
}): JSX.Element {
  const display = fmt ? fmt(value) : `${value}${unit ? ' ' + unit : ''}`
  return (
    <div style={{ marginTop: 18, maxWidth: 520 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'var(--ink-mute)' }}>{label}</span>
        <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 13, color: 'var(--ink)' }}>{display}</span>
      </div>
      <input
        type="range"
        className="tw-range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(parseFloat(event.target.value))}
      />
      {hint && (
        <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: 'var(--ink-mute)' }}>{hint}</div>
      )}
    </div>
  )
}

function ChunkingScene({ mode }: { mode: FrontendChunkingMode | 'fixed' }): JSX.Element {
  return (
    <svg width="380" height="260" viewBox="0 0 380 260">
      <g stroke="var(--ink)" strokeWidth="1.2" fill="var(--paper)" strokeLinecap="square">
        <rect x="40" y="30" width="120" height="200" />
        {Array.from({ length: 16 }).map((_, i) => (
          <line key={i} x1="48" y1={48 + i * 11} x2={48 + ((i * 37) % 90) + 14} y2={48 + i * 11} opacity="0.5" stroke="var(--ink)" />
        ))}
        <line x1="170" y1="130" x2="200" y2="130" stroke="var(--ink)" />
        <polyline points="194,124 200,130 194,136" fill="none" />
        {mode === 'none' && (
          <g>
            <rect x="220" y="30" width="120" height="200" />
            {Array.from({ length: 16 }).map((_, i) => (
              <line key={i} x1="228" y1={48 + i * 11} x2={228 + ((i * 37) % 90) + 14} y2={48 + i * 11} opacity="0.5" stroke="var(--ink)" />
            ))}
            <text x="280" y="252" fontSize="10" fontFamily="'IM Fell English', serif" fontStyle="italic" textAnchor="middle" stroke="none" fill="var(--ink-mute)">filed whole</text>
          </g>
        )}
        {mode === 'semantic' && (
          <g>
            {[0, 1, 2, 3].map(i => (
              <g key={i} transform={`translate(${220 + (i % 2) * 64} ${30 + Math.floor(i / 2) * 102})`}>
                <rect width="56" height="86" />
                {Array.from({ length: 6 }).map((_, j) => (
                  <line key={j} x1="6" y1={12 + j * 12} x2={6 + ((j * 23) % 36) + 8} y2={12 + j * 12} opacity="0.5" stroke="var(--ink)" />
                ))}
                <text x="28" y="80" fontSize="8" fontFamily="'IM Fell DW Pica SC', serif" textAnchor="middle" stroke="none" fill="var(--accent)" letterSpacing="2">§{i + 1}</text>
              </g>
            ))}
            <text x="280" y="252" fontSize="10" fontFamily="'IM Fell English', serif" fontStyle="italic" textAnchor="middle" stroke="none" fill="var(--ink-mute)">cut on section breaks</text>
          </g>
        )}
      </g>
    </svg>
  )
}

function BenchScene({
  mode,
  threshold,
  count,
}: {
  mode: RerankSelectionMode
  threshold: number
  count: number
}): JSX.Element {
  const proofs = [0.91, 0.78, 0.66, 0.54, 0.41, 0.22]
  const clampedThreshold = Math.max(0, Math.min(1, threshold))
  const gateX = mode === 'automatic' ? 82 + ((1 - clampedThreshold) * 222) : 210
  const visibleN = Math.max(0, Math.min(6, Math.round((Math.min(100, count) / 100) * 6)))
  const ranks = proofs
    .map((c, i) => ({ c, i }))
    .sort((a, b) => b.c - a.c)
    .slice(0, visibleN)
    .map(p => p.i)
  return (
    <svg width="420" height="260" viewBox="0 0 420 260">
      <g stroke="var(--ink)" strokeWidth="1.2" fill="var(--paper)" strokeLinecap="square">
        <line x1="20" y1="200" x2="400" y2="200" />
        <rect x="20" y="160" width="56" height="40" />
        <text x="48" y="186" fontSize="9" fontFamily="'IM Fell DW Pica SC', serif" textAnchor="middle" stroke="none" fill="var(--ink)" letterSpacing="2">CASE</text>
        <rect x="344" y="160" width="56" height="40" />
        <text x="372" y="186" fontSize="9" fontFamily="'IM Fell DW Pica SC', serif" textAnchor="middle" stroke="none" fill="var(--ink)" letterSpacing="2">DESK</text>
        <g>
          <line x1={gateX} y1="40" x2={gateX} y2="180" stroke="var(--accent)" strokeDasharray="4 4" />
          <rect x={gateX - 30} y="20" width="60" height="22" fill="var(--paper)" stroke="var(--accent)" />
          <text x={gateX} y="35" fontSize="10" fontFamily="'Special Elite', monospace" textAnchor="middle" stroke="none" fill="var(--accent)">
            {mode === 'automatic' ? `≥ ${threshold.toFixed(2)}` : `top ${count}`}
          </text>
          {mode === 'automatic' && (
            <text x={gateX} y="55" fontSize="8" fontFamily="'IM Fell DW Pica SC', serif" textAnchor="middle" stroke="none" fill="var(--accent)" letterSpacing="1.4">
              MAX {count}
            </text>
          )}
          <text x={gateX} y="196" fontSize="9" fontFamily="'IM Fell DW Pica SC', serif" textAnchor="middle" stroke="none" fill="var(--accent)" letterSpacing="2">GATE</text>
        </g>
        {proofs.map((conf, i) => {
          const cross = mode === 'automatic' ? (conf >= threshold && ranks.includes(i)) : ranks.includes(i)
          const x = 88 + i * 38
          return (
            <g key={i}>
              <rect x={x} y={cross ? 110 : 220} width="22" height="28" fill="var(--paper)" />
              <line x1={x + 4} y1={cross ? 118 : 228} x2={x + 18} y2={cross ? 118 : 228} opacity="0.5" stroke="var(--ink)" />
              <line x1={x + 4} y1={cross ? 124 : 234} x2={x + 16} y2={cross ? 124 : 234} opacity="0.5" stroke="var(--ink)" />
              <text x={x + 11} y={cross ? 156 : 252} fontSize="8" fontFamily="'Special Elite', monospace" textAnchor="middle" stroke="none" fill={cross ? 'var(--ink)' : 'var(--ink-faint)'}>{conf.toFixed(2)}</text>
              {cross && i < proofs.length - 1 && (
                <line x1={x + 22} y1="124" x2={x + 38} y2="124" stroke="var(--ink)" opacity="0.6" />
              )}
            </g>
          )
        })}
      </g>
    </svg>
  )
}

/**
 * Persona portrait — a framed line-drawing of the persona, with a typeset
 * "mode" badge above (e.g. TF·IDF, SVD, MiniLM, NLI, LLM) so the reader
 * sees both the face and the underlying method at a glance.
 */
function PersonaPortrait({ src, mode, alt }: { src: string; mode: string; alt: string }): JSX.Element {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 8,
      width: 240,
    }}>
      <div style={{
        fontFamily: "'IM Fell DW Pica SC', serif",
        fontSize: 10,
        letterSpacing: '0.32em',
        textTransform: 'uppercase',
        color: 'var(--accent)',
        borderTop: '1px solid var(--ink)',
        borderBottom: '1px solid var(--ink)',
        padding: '4px 12px',
        background: 'var(--paper)',
      }}>
        {mode}
      </div>
      <div style={{
        width: 240,
        height: 200,
        border: '1px solid var(--ink)',
        background: 'var(--paper)',
        boxShadow: '2px 4px 0 var(--shadow-soft)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        padding: 6,
        boxSizing: 'border-box',
      }}>
        <img
          src={src}
          alt={alt}
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            objectFit: 'contain',
            display: 'block',
          }}
        />
      </div>
    </div>
  )
}

function CompositorPortrait({ who }: { who: RetrievalModel }): JSX.Element {
  const persona = COMPOSITOR_PERSONAS[who]
  return <PersonaPortrait src={persona.portrait} mode={persona.mode} alt={persona.name} />
}

function SubEditorPortrait({ who }: { who: StanceMethod }): JSX.Element {
  const persona = SUB_EDITOR_PERSONAS[who]
  return <PersonaPortrait src={persona.portrait} mode={persona.mode} alt={persona.name} />
}

function ModalNameCard({
  persona,
  detail,
  desc,
}: {
  persona: 'tfidf' | 'svd' | 'minilm' | 'nli' | 'llm'
  detail: string
  desc: string
}): JSX.Element {
  return (
    <div style={{ borderTop: '1px solid var(--ink)', borderBottom: '1px solid var(--ink)', padding: '8px 14px', maxWidth: 360, marginTop: 8, width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 16, fontStyle: 'italic' }}>
          <PersonaName persona={persona} />
        </span>
        <span style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'var(--ink-mute)' }}>{detail}</span>
      </div>
      <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 12, color: 'var(--ink-soft)', marginTop: 2, lineHeight: 1.45 }}>{desc}</div>
    </div>
  )
}

function WeightSliders({
  weights,
}: {
  weights: Array<{ id: string; label: string; desc: string; value: number; onChange: (next: number) => void }>
}): JSX.Element {
  const total = weights.reduce((acc, w) => acc + w.value, 0) || 1
  return (
    <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
      {weights.map(w => {
        const pct = Math.round((w.value / total) * 100)
        return (
          <div key={w.id}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 16, fontStyle: 'italic' }}>{w.label}</span>
              <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 13 }}>{pct}%</span>
            </div>
            <input
              type="range"
              className="tw-range"
              min={0}
              max={100}
              step={1}
              value={Math.round(w.value * 100)}
              onChange={(event) => w.onChange(parseInt(event.target.value, 10) / 100)}
            />
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: 'var(--ink-mute)', marginTop: -2 }}>{w.desc}</div>
          </div>
        )
      })}
    </div>
  )
}

function RerankPreview({
  weights,
}: {
  weights: { relevance: number; agreement: number; recency: number }
}): JSX.Element {
  const entries = [
    { id: 'relevance', label: 'relevance', value: weights.relevance, color: 'var(--ink)' },
    { id: 'agreement', label: 'agreement', value: weights.agreement, color: 'var(--accent)' },
    { id: 'recency', label: 'recency', value: weights.recency, color: 'var(--ink-soft)' },
  ]
  const total = entries.reduce((a, e) => a + e.value, 0) || 1
  const sorted = [...entries].sort((a, b) => b.value - a.value)
  const top = sorted[0]
  const leaning: Record<string, string> = {
    relevance: 'a tightly on-topic stack — articles that drift even slightly off the subject sink to the bottom.',
    agreement: 'a partisan running order — articles that side with your claim float; dissenters are buried.',
    recency: 'a fresh-ink running order — yesterday’s columns lead, last decade’s sit at the back.',
  }
  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'var(--ink-mute)', marginBottom: 8 }}>The editor's recipe</div>
      <div style={{ display: 'flex', height: 38, border: '1px solid var(--ink)' }}>
        {entries.map(e => (
          <div key={e.id} style={{ flexBasis: `${(e.value / total) * 100}%`, background: e.color, borderRight: '1px solid var(--paper)' }} />
        ))}
      </div>
      <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {entries.map(e => (
          <div key={e.id} style={{ display: 'flex', alignItems: 'center', gap: 10, fontFamily: "'IM Fell English', serif", fontSize: 13 }}>
            <span style={{ display: 'inline-block', width: 14, height: 14, background: e.color }} />
            <span style={{ flex: 1, fontStyle: 'italic' }}>{e.label}</span>
            <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 12 }}>{Math.round((e.value / total) * 100)}%</span>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 22, borderTop: '1px solid var(--ink)', paddingTop: 12 }}>
        <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'var(--ink-mute)', marginBottom: 6 }}>The editor's leaning</div>
        <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 15, color: 'var(--ink-soft)', lineHeight: 1.45 }}>
          With <strong style={{ color: 'var(--accent)' }}>{top.label}</strong> at {Math.round((top.value / total) * 100)}%, expect {leaning[top.id]}
        </div>
      </div>
    </div>
  )
}

export function SettingsTray(props: SettingsTrayProps): JSX.Element | null {
  const {
    open,
    onClose,
    chunkingMode,
    supportedChunkingModes,
    onChunkingModeChange,
    retrievalModel,
    effectiveRetrievalModel,
    supportedRetrievalModels,
    onRetrievalModelChange,
    rerankSelectionMode,
    onRerankSelectionModeChange,
    autoRerankThreshold,
    onAutoRerankThresholdChange,
    rerankTopK,
    onRerankTopKChange,
    maxAutoRerankCandidates,
    chunkCandidateTopK,
    onChunkCandidateTopKChange,
    maxChunkCandidateTopK,
    chunkArticleTopK,
    onChunkArticleTopKChange,
    stanceMethod,
    effectiveStanceMethod,
    supportedStanceMethods,
    llmAgreementAvailable,
    onStanceMethodChange,
    llmLabelIrrelevant,
    onLlmLabelIrrelevantChange,
    topicWeight,
    stanceWeight,
    recencyWeight,
    onWeightsChange,
    onResetDefaults,
    onApply,
    settingsDirty = false,
    isApplying = false,
  } = props

  const [step, setStep] = useState(0)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const stepRefs = [
    useRef<HTMLDivElement | null>(null),
    useRef<HTMLDivElement | null>(null),
    useRef<HTMLDivElement | null>(null),
    useRef<HTMLDivElement | null>(null),
    useRef<HTMLDivElement | null>(null),
  ]

  useEffect(() => {
    if (!open) {
      setStep(0)
    }
  }, [open])

  if (!open) return null

  const goToStep = (index: number): void => {
    const node = stepRefs[index].current
    if (node && scrollRef.current) {
      scrollRef.current.scrollTo({ top: node.offsetTop - 8, behavior: 'smooth' })
    }
    setStep(index)
  }

  const handleScroll = (): void => {
    if (!scrollRef.current) return
    const y = scrollRef.current.scrollTop + 80
    let active = 0
    for (let i = 0; i < stepRefs.length; i += 1) {
      const n = stepRefs[i].current
      if (n && n.offsetTop <= y) active = i
    }
    setStep(active)
  }

  const compositor = COMPOSITOR_PERSONAS[effectiveRetrievalModel]
  const subEditor = SUB_EDITOR_PERSONAS[effectiveStanceMethod]
  const useChunking = chunkingMode !== 'none'
  const canUseChunking = supportedChunkingModes.includes('semantic') && llmAgreementAvailable
  const benchArticleLimitMax = Math.max(1, Math.min(100, maxAutoRerankCandidates))

  return (
    <div className="modal-shell" onClick={onClose}>
      <div className="modal-tray" onClick={(event) => event.stopPropagation()} style={{ position: 'absolute', inset: 32 }}>
        {/* Sticky banner + step rail */}
        <div style={{ padding: '20px 40px 14px', borderBottom: '1px solid var(--ink)', background: 'var(--paper)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 26 }}>
              hear! hear! <span style={{ fontStyle: 'italic', fontSize: 18, color: 'var(--ink-mute)' }}>· the press room</span>
            </div>
            <button type="button" onClick={onClose} style={{
              background: 'transparent',
              border: '1px solid var(--ink)',
              padding: '6px 14px',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 10,
              letterSpacing: '0.28em',
              textTransform: 'uppercase',
              cursor: 'pointer',
              color: 'var(--ink)',
            }}>
              close ✕
            </button>
          </div>
          <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: `repeat(${STEP_LABELS.length}, 1fr)`, gap: 0, borderTop: '1px solid var(--ink)' }}>
            {STEP_LABELS.map((label, i) => {
              const active = step === i
              const passed = step > i
              return (
                <button
                  key={label}
                  type="button"
                  onClick={() => goToStep(i)}
                  style={{
                    background: active ? 'var(--ink)' : 'transparent',
                    color: active ? 'var(--paper)' : (passed ? 'var(--ink)' : 'var(--ink-mute)'),
                    border: 0,
                    borderLeft: i === 0 ? 0 : '1px solid var(--ink)',
                    borderBottom: '1px solid var(--ink)',
                    padding: '10px 8px',
                    cursor: 'pointer',
                    fontFamily: "'IM Fell DW Pica SC', serif",
                    fontSize: 10,
                    letterSpacing: '0.24em',
                    textTransform: 'uppercase',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 8,
                  }}
                >
                  {passed && <span style={{ fontSize: 12 }}>✓</span>}
                  <span>{label}</span>
                </button>
              )
            })}
          </div>
        </div>

        {/* Scrollable journey */}
        <div ref={scrollRef} onScroll={handleScroll} className="tray-scroll" style={{ flex: 1, overflowX: 'hidden', position: 'relative' }}>
          <StepPanel idx={0} title="Chunking" subtitle="How articles are split before search" stepRef={stepRefs[0]}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: 32, alignItems: 'center' }}>
              <div>
                <p style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.55, color: 'var(--ink-soft)', maxWidth: 520 }}>
                  Choose whether search uses each full article or splits articles into sections first. Full articles keep the complete context. Section chunks make it easier to find and cite a specific passage. Chunking requires the LLM sub-editor.
                </p>
                <ModalRoleSelector<FrontendChunkingMode>
                  label="Search unit"
                  value={chunkingMode}
                  onChange={onChunkingModeChange}
                  options={[
                    { id: 'none', label: 'OFF', sub: 'use whole articles · default' },
                    { id: 'semantic', label: 'SEMANTIC', sub: 'split by section', disabled: !canUseChunking },
                  ]}
                />
                {useChunking && (
                  <>
                    <SliderField
                      label="Chunks to consider"
                      unit="chunks"
                      min={25}
                      max={maxChunkCandidateTopK}
                      step={5}
                      value={chunkCandidateTopK}
                      onChange={onChunkCandidateTopKChange}
                      hint="top matching chunks to collect before combining them by article"
                    />
                    <SliderField
                      label="Article limit after chunking"
                      unit="articles"
                      min={1}
                      max={20}
                      step={1}
                      value={chunkArticleTopK}
                      onChange={onChunkArticleTopKChange}
                      hint="max articles sent to the sub-editor after chunk matches are merged"
                    />
                  </>
                )}
              </div>
              <ChunkingScene mode={chunkingMode} />
            </div>
          </StepPanel>

          <StepPanel idx={1} title="The Compositor" subtitle="Who pulls candidate articles from the case" stepRef={stepRefs[1]}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, alignItems: 'center' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <CompositorPortrait who={effectiveRetrievalModel} />
                <ModalNameCard persona={effectiveRetrievalModel} detail={compositor.years} desc={compositor.desc} />
              </div>
              <div>
                <p style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.55, color: 'var(--ink-soft)' }}>
                  Choose how the archive finds candidate articles. Each method ranks matches a little differently: some look for exact wording, while others look for broader meaning.
                </p>
                <ModalRoleSelector<RetrievalModel>
                  label="At the type case"
                  value={retrievalModel}
                  onChange={onRetrievalModelChange}
                  options={[
                    { id: 'tfidf', label: 'TF·IDF', sub: 'literal', disabled: !supportedRetrievalModels.includes('tfidf') || useChunking },
                    { id: 'svd', label: 'SVD', sub: 'thematic · default', disabled: !supportedRetrievalModels.includes('svd') },
                    { id: 'minilm', label: 'MiniLM', sub: 'contextual', disabled: !supportedRetrievalModels.includes('minilm') },
                  ]}
                />
              </div>
            </div>
          </StepPanel>

          <StepPanel idx={2} title="Across the bench" subtitle="The hand-off — a relevance cut, or a fixed count?" stepRef={stepRefs[2]}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: 32, alignItems: 'center' }}>
              <div>
                <p style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.55, color: 'var(--ink-soft)', maxWidth: 520 }}>
                  Decide which retrieved articles go to the sub-editor. <em>Auto</em> keeps articles that meet a minimum relevance score, then applies an article limit. <em>Manual</em> sends a fixed number of the strongest articles (1–{benchArticleLimitMax}), regardless of score.
                </p>
                <ModalRoleSelector<RerankSelectionMode>
                  label="Hand-off"
                  value={rerankSelectionMode}
                  onChange={onRerankSelectionModeChange}
                  options={[
                    { id: 'automatic', label: 'AUTO', sub: 'relevance threshold · default' },
                    { id: 'manual', label: 'MANUAL', sub: 'fixed count of articles' },
                  ]}
                />
                {rerankSelectionMode === 'automatic' && (
                  <>
                    <SliderField
                      label="Relevance threshold"
                      min={0}
                      max={1}
                      step={0.01}
                      value={autoRerankThreshold}
                      onChange={onAutoRerankThresholdChange}
                      fmt={(v) => v.toFixed(2)}
                      hint="articles below this score are filtered out"
                    />
                    <SliderField
                      label="Maximum articles"
                      unit="articles"
                      min={1}
                      max={benchArticleLimitMax}
                      step={1}
                      value={rerankTopK}
                      onChange={onRerankTopKChange}
                      hint="after filtering by score, send no more than this many articles"
                    />
                  </>
                )}
                {rerankSelectionMode === 'manual' && (
                  <SliderField
                    label="Number of articles"
                    unit="articles"
                    min={1}
                    max={benchArticleLimitMax}
                    step={1}
                    value={rerankTopK}
                    onChange={onRerankTopKChange}
                    hint="the top-N strongest articles cross to the lectern"
                  />
                )}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <BenchScene mode={rerankSelectionMode} threshold={autoRerankThreshold} count={rerankTopK} />
                <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: 'var(--ink-mute)' }}>
                  {rerankSelectionMode === 'automatic'
                    ? `articles above ${autoRerankThreshold.toFixed(2)} cross, up to ${rerankTopK} — others stay below the bench`
                    : `the strongest ${rerankTopK} articles cross — the rest stay below the bench`}
                </div>
              </div>
            </div>
          </StepPanel>

          <StepPanel idx={3} title="The Sub-Editor" subtitle="Who marks each article for stance" stepRef={stepRefs[3]}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, alignItems: 'center' }}>
              <div>
                <p style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.55, color: 'var(--ink-soft)' }}>
                  Choose how each article is classified against your claim: supports it, qualifies it, or argues against it.
                </p>
                <ModalRoleSelector<StanceMethod>
                  label="Stance method"
                  value={stanceMethod}
                  onChange={onStanceMethodChange}
                  options={[
                    { id: 'nli', label: 'NLI', sub: 'quick classifier', disabled: !supportedStanceMethods.includes('nli') || useChunking },
                    { id: 'llm', label: 'LLM', sub: 'context-aware · default', disabled: !supportedStanceMethods.includes('llm') || !llmAgreementAvailable },
                  ]}
                />
                {useChunking && (
                  <p style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: 'var(--ink-mute)', marginTop: 8 }}>
                    Chunking requires <PersonaName persona="llm" />. <PersonaName persona="nli" /> is locked out for chunked retrieval.
                  </p>
                )}
                <SmallSwitch
                  label="Filter unrelated LLM results"
                  checked={llmLabelIrrelevant}
                  disabled={effectiveStanceMethod !== 'llm' || !llmAgreementAvailable}
                  onChange={onLlmLabelIrrelevantChange}
                  hint="When using the LLM, let it mark completely unrelated articles as not relevant so they stay out of the main results."
                />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                <SubEditorPortrait who={effectiveStanceMethod} />
                <ModalNameCard persona={effectiveStanceMethod} detail={subEditor.tool} desc={subEditor.desc} />
              </div>
            </div>
          </StepPanel>

          <StepPanel idx={4} title="Re-ranking" subtitle="The editor's weighting — set the priorities for the final order" stepRef={stepRefs[4]}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: 32, alignItems: 'start' }}>
              <div>
                <p style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.55, color: 'var(--ink-soft)', maxWidth: 520 }}>
                  Choose how much each factor matters when sorting the final results: relevance, stance agreement, and recency.
                </p>
                <WeightSliders weights={[
                  {
                    id: 'relevance',
                    label: 'Relevance',
                    desc: 'how close the article is to your topic',
                    value: topicWeight,
                    onChange: (next) => onWeightsChange({ topicWeight: next, stanceWeight, recencyWeight }),
                  },
                  {
                    id: 'agreement',
                    label: 'Stance agreement',
                    desc: 'how much it agrees with your claim',
                    value: stanceWeight,
                    onChange: (next) => onWeightsChange({ topicWeight, stanceWeight: next, recencyWeight }),
                  },
                  {
                    id: 'recency',
                    label: 'Recency',
                    desc: 'how recently it was published',
                    value: recencyWeight,
                    onChange: (next) => onWeightsChange({ topicWeight, stanceWeight, recencyWeight: next }),
                  },
                ]} />
              </div>
              <RerankPreview weights={{
                relevance: topicWeight,
                agreement: stanceWeight,
                recency: recencyWeight,
              }} />
            </div>
          </StepPanel>

          {/* Final summary + apply */}
          <div style={{ padding: '32px 40px 48px', borderTop: '1px solid var(--ink)' }}>
            <div className="tracker" style={{ color: 'var(--accent)', fontSize: 10 }}>The complete instrument</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 19, marginTop: 6, marginBottom: 12, lineHeight: 1.55, color: 'var(--ink)', maxWidth: 760 }}>
              {(() => {
                const unitNoun = useChunking ? 'top passages' : 'on-topic articles'
                const handoffPhrase = rerankSelectionMode === 'automatic'
                  ? <>handing off up to <strong>{rerankTopK}</strong> that score at least <strong>{autoRerankThreshold.toFixed(2)}</strong> on relevance</>
                  : <>handing off the strongest <strong>{rerankTopK}</strong></>
                const chunkingClause = useChunking
                  ? ' (each article cut on section breaks before being read)'
                  : ''
                return (
                  <>
                    Let <em><PersonaName persona={effectiveRetrievalModel} /></em> pull {unitNoun} from the archive{chunkingClause}, {handoffPhrase} to <em><PersonaName persona={effectiveStanceMethod} /></em> who marks stance agreement.
                    <span style={{ display: 'block', marginTop: 8, fontFamily: "'Special Elite', monospace", fontSize: 13, color: 'var(--ink-mute)' }}>
                      The editor then weights the running order — relevance <strong>{Math.round(topicWeight * 100)}</strong> · agreement <strong>{Math.round(stanceWeight * 100)}</strong> · recency <strong>{Math.round(recencyWeight * 100)}</strong>.
                    </span>
                  </>
                )
              })()}
            </div>
            <div style={{ marginTop: 22, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <button
                type="button"
                onClick={onResetDefaults}
                disabled={!onResetDefaults}
                style={{
                  background: 'transparent',
                  border: 0,
                  fontFamily: "'IM Fell DW Pica SC', serif",
                  fontSize: 10,
                  letterSpacing: '0.28em',
                  textTransform: 'uppercase',
                  color: 'var(--ink-mute)',
                  cursor: onResetDefaults ? 'pointer' : 'not-allowed',
                  padding: '6px 10px',
                }}
              >
                ↻ restore defaults
              </button>
              <button
                type="button"
                onClick={onApply}
                disabled={isApplying}
                style={{
                  background: settingsDirty ? 'var(--accent)' : 'var(--ink)',
                  color: 'var(--paper)',
                  border: `1px solid ${settingsDirty ? 'var(--accent)' : 'var(--ink)'}`,
                  padding: '12px 26px',
                  fontFamily: "'IM Fell DW Pica SC', serif",
                  fontSize: 12,
                  letterSpacing: '0.28em',
                  textTransform: 'uppercase',
                  cursor: isApplying ? 'wait' : 'pointer',
                  opacity: isApplying ? 0.72 : 1,
                  transition: 'background 160ms ease, border-color 160ms ease',
                }}
              >
                {isApplying ? 'applying…' : (settingsDirty ? 'apply changes →' : 'close ✕')}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SettingsTray
