import { useEffect, useMemo, useState, type FormEvent } from 'react'
import type {
  Article,
  QueryRewriteAlternative,
  ResultsOverview,
  ResultsOverviewArgument,
  ResultsOverviewSource,
  SimilarArticlesResponse,
  SvdLatentDimension,
  TypoCorrectionSuggestion,
} from './types'

export type SearchProgressLine = {
  label: string
  pct: number
  state: 'queued' | 'active' | 'done'
}

export type ResultsChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  source_indices?: number[] | null
  sources?: ResultsOverviewSource[] | null
}

export type ResultsFlowProps = {
  topic: string
  opinion: string
  inputMode: 'stance' | 'essay'
  essayText: string
  thesisSentence: string

  loading: boolean
  error: string | null
  emptyResultsMessage: string | null

  articles: Article[]
  dismissedIds: Set<string>
  onDismiss: (article: Article) => void
  onUndoDismiss: (article: Article) => void
  onApplyDismissals: () => void
  onBackToCompose: () => void

  // typo + rewrite
  typoCorrection: TypoCorrectionSuggestion | null
  onApplyTypoCorrection: (query: string) => void
  onSearchAnyway: () => void
  rewriteAlternatives: QueryRewriteAlternative[]
  rewriteLoading: boolean
  rewriteError: string | null
  onLoadRewrites: () => void
  onApplyRewrite: (alternative: QueryRewriteAlternative) => void

  // structured overview
  overview: ResultsOverview | null
  overviewDraft: string
  overviewLoading: boolean
  overviewError: string | null

  // similar articles
  similarSource: Article | null
  similarArticles: Article[]
  similarHasMore: boolean
  similarLoading: boolean
  similarError: string | null
  onFindSimilar: (article: Article) => void
  onLoadMoreSimilar: () => void
  onCloseSimilar: () => void

  // chat about results
  chatMessages: ResultsChatMessage[]
  chatInput: string
  chatLoading: boolean
  chatError: string | null
  onChatInputChange: (next: string) => void
  onSubmitChat: (event: FormEvent<HTMLFormElement>) => void

  // chat about article
  articleChatMessages: ResultsChatMessage[]
  articleChatInput: string
  articleChatLoading: boolean
  articleChatError: string | null
  articleChatArticleId: string | null
  onOpenArticleChat: (article: Article) => void
  onCloseArticleChat: () => void
  onArticleChatInputChange: (next: string) => void
  onSubmitArticleChat: (event: FormEvent<HTMLFormElement>) => void

  // search progress
  progressLines: SearchProgressLine[]
  progressMessage: string | null

  // svd
  querySvdDimensions: SvdLatentDimension[]
}

const SCATTER_SPOTS = [
  { x: 0, y: 0, r: -1.4 }, { x: 270, y: 18, r: 0.8 },
  { x: 540, y: 2, r: -0.6 }, { x: 10, y: 170, r: 1.0 },
  { x: 280, y: 188, r: -1.2 }, { x: 550, y: 174, r: 0.4 },
  { x: 140, y: 340, r: 1.6 }, { x: 410, y: 354, r: -0.8 },
] as const

function getArticleId(article: Article): string {
  return String(article.id)
}

function formatDate(value: string | null | undefined): string {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' })
}

function getSentColor(article: Article): string {
  const label = String(article.vader_sentiment?.label || '').toLocaleLowerCase()
  if (label === 'positive') return '#1a1a1a'
  if (label === 'negative') return '#7a1d1d'
  return '#9a9a92'
}

function getStanceCategory(article: Article): 'supports' | 'complicates' | 'neutral' {
  const label = String(article.stance_label || '').toLocaleLowerCase()
  if (label.includes('support') || label.includes('entail')) return 'supports'
  if (label.includes('contradict') || label.includes('against') || label.includes('oppos')) return 'complicates'
  return 'neutral'
}

function clampPct(value: number | null | undefined): number {
  if (value === undefined || value === null || Number.isNaN(value)) return 0
  return Math.max(0, Math.min(100, Math.round(value <= 1 ? value * 100 : value)))
}

function getTopicPct(article: Article): number {
  const candidate = article.topic_score_display ?? article.topic_score_normalized ?? article.topic_score
  return clampPct(candidate ?? null)
}

function getAgreementPct(article: Article): number {
  const candidate = article.stance_score_normalized ?? article.llm_agreement_score
  return clampPct(candidate ?? null)
}

function getRecencyPct(article: Article): number {
  return clampPct(article.recency_score_normalized ?? null)
}

function getCentralClaim(article: Article): string {
  const claim = article.central_claim_summary?.trim()
    || article.thesis_sentence?.trim()
    || article.summary?.trim()
    || ''
  return claim
}

function getDisplayKeywords(article: Article): string[] {
  const arr = (article.keywords ?? []).map(k => String(k).trim()).filter(Boolean)
  return arr.slice(0, 6)
}

function getOutlet(_article: Article): string {
  return 'The Guardian'
}

function getAuthorDisplay(article: Article): string {
  return article.author_display?.trim() || article.author_raw?.trim() || 'Unknown'
}

function PinnedSlip({ topic, opinion, mode, essayText, thesisSentence, animateOnMount }: {
  topic: string
  opinion: string
  mode: 'stance' | 'essay'
  essayText: string
  thesisSentence: string
  animateOnMount: boolean
}): JSX.Element {
  const [mounted, setMounted] = useState(!animateOnMount)
  useEffect(() => {
    if (!animateOnMount) return
    const id = requestAnimationFrame(() => setMounted(true))
    return () => cancelAnimationFrame(id)
  }, [animateOnMount])

  const today = useMemo(() => new Date().toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }), [])

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      transform: `translateY(${mounted ? 0 : 80}px)`,
      opacity: mounted ? 1 : 0,
      transition: 'transform 700ms cubic-bezier(.2,.7,.2,1.05), opacity 500ms',
    }}>
      <div style={{
        position: 'relative',
        width: 720,
        padding: '12px 22px 14px',
        background: '#fafaf7',
        border: '1px solid #1a1a1a',
        boxShadow: '0 1px 0 rgba(26,26,26,0.06)',
        fontFamily: "'IM Fell English', serif",
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: "'IM Fell DW Pica SC', serif",
          fontSize: 9,
          letterSpacing: '0.28em',
          textTransform: 'uppercase',
          color: '#6a6a62',
          borderBottom: '1px solid rgba(26,26,26,0.18)',
          paddingBottom: 6,
          marginBottom: 8,
        }}>
          <span>search slip · {today}</span>
          <span>filed by reader</span>
        </div>

        {mode === 'stance' ? (
          <>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
              <span style={{ fontStyle: 'italic', fontSize: 14, color: '#6a6a62', minWidth: 70, textAlign: 'right' }}>regarding</span>
              <span style={{
                fontFamily: "'Special Elite', monospace",
                fontSize: 17,
                color: '#1a1a1a',
                borderBottom: '1px solid #1a1a1a',
                paddingBottom: 1,
                flex: 1,
              }}>{topic || ' '}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, marginTop: 6 }}>
              <span style={{ fontStyle: 'italic', fontSize: 14, color: '#6a6a62', minWidth: 70, textAlign: 'right' }}>I believe</span>
              <span style={{
                fontFamily: "'Special Elite', monospace",
                fontSize: 17,
                color: '#1a1a1a',
                borderBottom: '1px solid #1a1a1a',
                paddingBottom: 1,
                flex: 1,
              }}>{opinion || ' '}</span>
            </div>
          </>
        ) : (
          <>
            {thesisSentence && (
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
                <span style={{ fontStyle: 'italic', fontSize: 14, color: '#6a6a62', minWidth: 70, textAlign: 'right' }}>thesis</span>
                <span style={{
                  fontFamily: "'Special Elite', monospace",
                  fontSize: 15,
                  color: '#1a1a1a',
                  borderBottom: '1px solid #1a1a1a',
                  paddingBottom: 1,
                  flex: 1,
                }}>{thesisSentence}</span>
              </div>
            )}
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, marginTop: 6 }}>
              <span style={{ fontStyle: 'italic', fontSize: 14, color: '#6a6a62', minWidth: 70, textAlign: 'right' }}>essay</span>
              <span style={{
                fontFamily: "'IM Fell English', serif",
                fontStyle: 'italic',
                fontSize: 13,
                color: '#3a3a36',
                flex: 1,
                lineHeight: 1.5,
              }}>{essayText.slice(0, 220)}{essayText.length > 220 ? '…' : ''}</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function RFSpinner(): JSX.Element {
  return (
    <span style={{ display: 'inline-block', width: 12, height: 12, position: 'relative' }}>
      <span style={{
        position: 'absolute',
        inset: 0,
        border: '1.5px solid rgba(26,26,26,0.18)',
        borderTopColor: '#7a1d1d',
        borderRadius: '50%',
        animation: 'rf-spin 900ms linear infinite',
      }} />
    </span>
  )
}

function RFProgressLine({ line }: { line: SearchProgressLine }): JSX.Element {
  const color = line.state === 'done' ? '#1a1a1a' : line.state === 'active' ? '#7a1d1d' : '#9a9a92'
  const symbol = line.state === 'done' ? '◼' : line.state === 'active' ? '◧' : '◻'
  const pct = clampPct(line.pct)
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', color }}>
        <span>{symbol} {line.label}</span>
        <span>{line.state === 'done' ? '✓' : line.state === 'queued' ? '— —' : `${pct}%`}</span>
      </div>
      <div style={{ height: 2, background: 'rgba(26,26,26,0.12)', position: 'relative', marginTop: 2 }}>
        <div style={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: `${pct}%`,
          background: color,
          opacity: line.state === 'queued' ? 0.3 : 1,
        }} />
      </div>
    </div>
  )
}

function RFScatterCard({
  article,
  spot,
  delay,
  onDismiss,
}: {
  article: Article
  spot: { x: number; y: number; r: number }
  delay: number
  onDismiss: () => void
}): JSX.Element {
  const [hover, setHover] = useState(false)
  const sentColor = getSentColor(article)
  const topicPct = getTopicPct(article)
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        position: 'absolute',
        left: spot.x,
        top: spot.y,
        width: 250,
        height: 116,
        padding: '12px 14px',
        background: '#fafaf7',
        border: '1px solid #1a1a1a',
        transform: `rotate(${spot.r}deg)`,
        animation: `rf-scatter 700ms ${delay}ms cubic-bezier(.2,.7,.2,1.05) both`,
        fontFamily: "'Old Standard TT', serif",
        boxShadow: '2px 4px 0 rgba(26,26,26,0.04)',
      }}
    >
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'baseline',
        fontFamily: "'IM Fell DW Pica SC', serif",
        fontSize: 9,
        letterSpacing: '0.24em',
        textTransform: 'uppercase',
        color: '#6a6a62',
      }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: sentColor, display: 'inline-block' }} />
          {getOutlet(article)}
        </span>
        <span style={{ color: '#1a1a1a', fontFamily: "'Special Elite', monospace", letterSpacing: '0.04em' }}>τ {topicPct}</span>
      </div>
      <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 16, lineHeight: 1.15, marginTop: 6, color: '#1a1a1a' }}>
        {article.title.length > 68 ? `${article.title.slice(0, 66)}…` : article.title}
      </div>
      <div style={{
        position: 'absolute',
        bottom: 10,
        left: 14,
        right: 14,
        fontFamily: "'IM Fell English', serif",
        fontStyle: 'italic',
        fontSize: 11,
        color: '#6a6a62',
        display: 'flex',
        justifyContent: 'space-between',
      }}>
        <span>by {getAuthorDisplay(article)}</span>
        <span>{formatDate(article.date)}</span>
      </div>
      {hover && (
        <button
          type="button"
          onClick={(event) => { event.stopPropagation(); onDismiss() }}
          style={{
            position: 'absolute',
            top: -10,
            right: -10,
            width: 24,
            height: 24,
            border: '1px solid #1a1a1a',
            background: '#fafaf7',
            cursor: 'pointer',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 10,
            color: '#7a1d1d',
          }}
          title="mark as not relevant"
        >×</button>
      )}
    </div>
  )
}

function RFMicroDial({ label, value, accent, active }: { label: string; value: number; accent?: boolean; active?: boolean }): JSX.Element {
  const fill = accent ? '#7a1d1d' : (active ? '#fafaf7' : '#1a1a1a')
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: "'Special Elite', monospace", fontSize: 9 }}>
      <span>{label}</span>
      <div style={{
        flex: 1,
        height: 3,
        background: active ? 'rgba(250,250,247,0.18)' : 'rgba(26,26,26,0.12)',
        position: 'relative',
      }}>
        <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${value}%`, background: fill }} />
      </div>
      <span style={{ width: 18, textAlign: 'right' }}>{value}</span>
    </div>
  )
}

function RFRankedRow({
  article,
  rank,
  active,
  onClick,
  onDismiss,
}: {
  article: Article
  rank: number
  active: boolean
  onClick: () => void
  onDismiss: () => void
}): JSX.Element {
  const [hover, setHover] = useState(false)
  const sentColor = getSentColor(article)
  const wordCount = article.word_count ?? null
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={onClick}
      style={{
        display: 'grid',
        gridTemplateColumns: '52px 1fr 96px 28px',
        gap: 14,
        alignItems: 'center',
        padding: '12px 4px',
        borderBottom: '1px solid rgba(26,26,26,0.18)',
        background: active ? '#1a1a1a' : 'transparent',
        color: active ? '#fafaf7' : '#1a1a1a',
        fontFamily: "'Old Standard TT', serif",
        cursor: 'pointer',
      }}
    >
      <div style={{
        fontFamily: "'IM Fell English', serif",
        fontSize: 26,
        lineHeight: 1,
        color: active ? '#fafaf7' : '#7a1d1d',
        textAlign: 'center',
      }}>{String(rank).padStart(2, '0')}</div>
      <div>
        <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 16, lineHeight: 1.15, display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: sentColor, display: 'inline-block', flexShrink: 0 }} />
          <span>{article.title}</span>
        </div>
        <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, opacity: 0.75, marginTop: 3 }}>
          {getAuthorDisplay(article)} · {getOutlet(article)} · {formatDate(article.date)}{wordCount ? ` · ${wordCount.toLocaleString()} words` : ''}
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <RFMicroDial label="A" value={getAgreementPct(article)} active={active} accent />
        <RFMicroDial label="T" value={getTopicPct(article)} active={active} />
        <RFMicroDial label="R" value={getRecencyPct(article)} active={active} />
      </div>
      <button
        type="button"
        onClick={(event) => { event.stopPropagation(); onDismiss() }}
        title="not relevant"
        style={{
          background: 'transparent',
          border: '1px solid',
          borderColor: active ? '#fafaf7' : '#1a1a1a',
          color: active ? '#fafaf7' : '#7a1d1d',
          cursor: 'pointer',
          width: 22,
          height: 22,
          opacity: hover || active ? 1 : 0.25,
          fontFamily: "'IM Fell DW Pica SC', serif",
          fontSize: 10,
        }}
      >×</button>
    </div>
  )
}

function RFMetric({ label, value, text, accent }: { label: string; value?: number; text?: string; accent?: boolean }): JSX.Element {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '6px 0',
      borderBottom: '1px solid rgba(26,26,26,0.18)',
    }}>
      <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#6a6a62' }}>{label}</span>
      {value !== undefined ? (
        <span style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 130 }}>
          <span style={{ flex: 1, height: 3, background: 'rgba(26,26,26,0.12)', position: 'relative' }}>
            <span style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: `${value}%`,
              background: accent ? '#7a1d1d' : '#1a1a1a',
            }} />
          </span>
          <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#1a1a1a' }}>{value}</span>
        </span>
      ) : (
        <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#1a1a1a' }}>{text}</span>
      )}
    </div>
  )
}

const SVD_RADAR_SIZE = 520
const SVD_RADAR_CENTER = SVD_RADAR_SIZE / 2
const SVD_RADAR_RADIUS = 158
const SVD_RADAR_LEVELS = 4

function clampSvdMagnitude(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0))
}

function getSvdMagnitude(d: SvdLatentDimension): number {
  const m = Math.abs(Number(d.magnitude))
  const v = Math.abs(Number(d.value))
  return Math.max(Number.isFinite(m) ? m : 0, Number.isFinite(v) ? v : 0)
}

function getSvdLabel(d: SvdLatentDimension): string {
  return (d.display_label || d.dimension_name || d.name || d.label
    || (Array.isArray(d.label_terms) ? d.label_terms.slice(0, 2).join(' / ') : '')
    || `Concept ${d.dimension_label}`).toString()
}

function VintageSvdRadar({
  articleDims,
  queryDims,
}: {
  articleDims: SvdLatentDimension[]
  queryDims: SvdLatentDimension[]
}): JSX.Element {
  const dims = articleDims.slice(0, 8)
  if (dims.length === 0) {
    return <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', color: '#6a6a62', padding: 24, textAlign: 'center' }}>No SVD concepts available for this proof.</div>
  }
  const queryByIndex = new Map<number, SvdLatentDimension>()
  for (const d of queryDims) {
    if (typeof d.dimension_index === 'number') queryByIndex.set(d.dimension_index, d)
  }
  const N = dims.length
  const angle = (i: number): number => -Math.PI / 2 + (i * 2 * Math.PI) / N
  const point = (i: number, scale: number): { x: number; y: number } => ({
    x: SVD_RADAR_CENTER + Math.cos(angle(i)) * SVD_RADAR_RADIUS * scale,
    y: SVD_RADAR_CENTER + Math.sin(angle(i)) * SVD_RADAR_RADIUS * scale,
  })
  const articleHull = dims.map((d, i) => {
    const p = point(i, clampSvdMagnitude(getSvdMagnitude(d)))
    return `${p.x.toFixed(1)},${p.y.toFixed(1)}`
  }).join(' ')
  const hasQuery = dims.every((d) => queryByIndex.has(d.dimension_index))
  const queryHull = hasQuery
    ? dims.map((d, i) => {
      const q = queryByIndex.get(d.dimension_index)
      const p = point(i, clampSvdMagnitude(q ? getSvdMagnitude(q) : 0))
      return `${p.x.toFixed(1)},${p.y.toFixed(1)}`
    }).join(' ')
    : ''

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 200px', gap: 16, alignItems: 'start', border: '1px solid #1a1a1a', background: '#fafaf7', padding: '12px 14px' }}>
      <svg width="100%" viewBox={`0 0 ${SVD_RADAR_SIZE} ${SVD_RADAR_SIZE + 24}`} style={{ display: 'block', overflow: 'visible' }}>
        {Array.from({ length: SVD_RADAR_LEVELS }).map((_, l) => {
          const scale = (l + 1) / SVD_RADAR_LEVELS
          const poly = dims.map((_d, i) => {
            const p = point(i, scale)
            return `${p.x.toFixed(1)},${p.y.toFixed(1)}`
          }).join(' ')
          return (
            <polygon
              key={l}
              points={poly}
              fill="none"
              stroke="#1a1a1a"
              strokeOpacity={0.18}
              strokeWidth={l === SVD_RADAR_LEVELS - 1 ? 1 : 0.5}
              strokeDasharray={l === SVD_RADAR_LEVELS - 1 ? '0' : '2 3'}
            />
          )
        })}
        {dims.map((_d, i) => {
          const p = point(i, 1)
          return <line key={i} x1={SVD_RADAR_CENTER} y1={SVD_RADAR_CENTER} x2={p.x} y2={p.y} stroke="#1a1a1a" strokeOpacity={0.22} strokeWidth={0.6} />
        })}
        {hasQuery && (
          <polygon points={queryHull} fill="rgba(122, 29, 29, 0.10)" stroke="#7a1d1d" strokeWidth={1.2} strokeDasharray="3 2" />
        )}
        <polygon points={articleHull} fill="rgba(26, 26, 26, 0.20)" stroke="#1a1a1a" strokeWidth={1.4} />
        {dims.map((d, i) => {
          const ap = point(i, clampSvdMagnitude(getSvdMagnitude(d)))
          const lp = point(i, 1.16)
          const dx = lp.x - SVD_RADAR_CENTER
          const anchor: 'start' | 'middle' | 'end' = Math.abs(dx) < 24 ? 'middle' : (dx < 0 ? 'end' : 'start')
          const aFill = (Number(d.value) >= 0) ? '#1a1a1a' : '#7a1d1d'
          return (
            <g key={i}>
              {Number(d.value) < 0 && <circle cx={ap.x} cy={ap.y} r={6} fill="none" stroke="#7a1d1d" strokeWidth={1} />}
              <circle cx={ap.x} cy={ap.y} r={3.5} fill={aFill} />
              {hasQuery && (() => {
                const q = queryByIndex.get(d.dimension_index)
                if (!q) return null
                const qp = point(i, clampSvdMagnitude(getSvdMagnitude(q)))
                return <circle cx={qp.x} cy={qp.y} r={3.5} fill="#fafaf7" stroke="#7a1d1d" strokeWidth={1.2} />
              })()}
              <text x={lp.x} y={lp.y + 4} fontFamily="'IM Fell English', serif" fontStyle="italic" fontSize="12" fill="#1a1a1a" textAnchor={anchor}>
                {getSvdLabel(d)}
              </text>
              <text x={lp.x} y={lp.y + 17} fontFamily="'Special Elite', monospace" fontSize="9" fill="#6a6a62" textAnchor={anchor}>
                a {Number(d.value) >= 0 ? '+' : ''}{Number(d.value).toFixed(2)}{hasQuery && (() => {
                  const q = queryByIndex.get(d.dimension_index)
                  if (!q) return ''
                  return ` · q ${Number(q.value) >= 0 ? '+' : ''}${Number(q.value).toFixed(2)}`
                })()}
              </text>
            </g>
          )
        })}
        <circle cx={SVD_RADAR_CENTER} cy={SVD_RADAR_CENTER} r={1.4} fill="#1a1a1a" />
      </svg>

      <div style={{ paddingTop: 6 }}>
        <div className="tracker" style={{ marginBottom: 8 }}>The legend</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span style={{ width: 18, height: 10, background: 'rgba(26,26,26,0.20)', border: '1px solid #1a1a1a', display: 'inline-block' }} />
          <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13 }}>this article</span>
        </div>
        {hasQuery && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <span style={{ width: 18, height: 10, background: 'rgba(122,29,29,0.10)', border: '1px dashed #7a1d1d', display: 'inline-block' }} />
            <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13 }}>your slip</span>
          </div>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#1a1a1a', display: 'inline-block', margin: '0 5px' }} />
          <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13 }}>positive pole</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <span style={{ width: 12, height: 12, border: '1px solid #7a1d1d', display: 'inline-block', borderRadius: '50%' }} />
          <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13 }}>negative pole</span>
        </div>
        <div style={{ height: 1, background: 'rgba(26,26,26,0.18)', margin: '12px 0 10px' }} />
        <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, lineHeight: 1.5, color: '#3a3a36' }}>
          Concentric rings mark magnitude in quartiles, 0 at the center, ±1 at the edge. Vertices = where each side stands on each concept.
        </div>
      </div>
    </div>
  )
}

function ChatBubble({
  message,
}: {
  message: ResultsChatMessage
}): JSX.Element {
  const isAssistant = message.role === 'assistant'
  return (
    <div style={{ marginBottom: 16, display: 'flex', flexDirection: 'column', alignItems: isAssistant ? 'flex-start' : 'flex-end' }}>
      <div style={{
        fontFamily: "'IM Fell DW Pica SC', serif",
        fontSize: 9,
        letterSpacing: '0.28em',
        textTransform: 'uppercase',
        color: '#6a6a62',
        marginBottom: 4,
      }}>
        {isAssistant ? 'editor hollis' : 'you'}
      </div>
      <div style={{
        maxWidth: '88%',
        padding: '10px 14px',
        background: isAssistant ? '#fafaf7' : '#1a1a1a',
        color: isAssistant ? '#1a1a1a' : '#fafaf7',
        border: '1px solid #1a1a1a',
        fontFamily: "'IM Fell English', serif",
        fontSize: 14,
        lineHeight: 1.5,
        whiteSpace: 'pre-wrap',
      }}>
        {message.content}
        {message.sources && message.sources.length > 0 && (
          <div style={{
            marginTop: 8,
            paddingTop: 8,
            borderTop: '1px dotted rgba(26,26,26,0.4)',
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
          }}>
            {message.sources.map(source => (
              <span key={source.result_index} style={{
                fontFamily: "'IM Fell DW Pica SC', serif",
                fontSize: 8,
                letterSpacing: '0.22em',
                textTransform: 'uppercase',
                border: '1px solid currentColor',
                padding: '2px 6px',
              }}>
                [{source.result_index}] {source.title.slice(0, 24)}{source.title.length > 24 ? '…' : ''}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function BroadsheetOverlay({
  article,
  rank,
  total,
  onClose,
  onDismiss,
  onChatThis,
  onFindSimilar,
}: {
  article: Article
  rank: number
  total: number
  onClose: () => void
  onDismiss: () => void
  onChatThis: () => void
  onFindSimilar: () => void
}): JSX.Element {
  const stance = getStanceCategory(article)
  const stanceLabel = stance === 'supports'
    ? 'supports your stance'
    : stance === 'complicates'
      ? 'complicates your stance'
      : 'neutral on your stance'
  const articleSvd = (article.svd_query_chart_dimensions ?? article.svd_chart_dimensions ?? article.svd_dimensions ?? []) as SvdLatentDimension[]
  const wordCount = article.word_count ?? null
  const sentiment = article.vader_sentiment
  return (
    <div onClick={onClose} style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(26,26,26,0.32)',
      display: 'flex',
      justifyContent: 'flex-end',
      zIndex: 30,
      animation: 'rf-fade 220ms ease-out both',
    }}>
      <div onClick={(event) => event.stopPropagation()} className="tray-scroll" style={{
        width: 760,
        height: '100%',
        background: '#fafaf7',
        borderLeft: '1px solid #1a1a1a',
        boxShadow: '-30px 0 60px rgba(26,26,26,0.18)',
        padding: '28px 40px 60px',
        animation: 'rf-slide 360ms cubic-bezier(.2,.7,.2,1.05) both',
        fontFamily: "'Old Standard TT', serif",
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 10, letterSpacing: '0.28em', textTransform: 'uppercase', color: '#6a6a62' }}>
          <span>broadsheet · the proof in full</span>
          <button type="button" onClick={onClose} style={{
            background: 'transparent',
            border: '1px solid #1a1a1a',
            padding: '6px 12px',
            fontFamily: 'inherit',
            fontSize: 'inherit',
            letterSpacing: 'inherit',
            textTransform: 'inherit',
            color: '#1a1a1a',
            cursor: 'pointer',
          }}>← back to the ledger</button>
        </div>

        <div style={{ borderTop: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a', textAlign: 'center', padding: '14px 0 10px', marginTop: 14 }}>
          <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 32, lineHeight: 1 }}>{getOutlet(article)}</div>
          <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 10, letterSpacing: '0.28em', textTransform: 'uppercase', color: '#6a6a62', marginTop: 6 }}>
            {formatDate(article.date)} · No. {String(rank).padStart(3, '0')} of {total}
          </div>
        </div>

        <div style={{ marginTop: 20 }}>
          <div className="tracker" style={{ color: 'var(--accent)' }}>
            on your topic · {stanceLabel}
          </div>
          <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 30, lineHeight: 1.08, marginTop: 8 }}>{article.title}</div>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            borderTop: '1px solid rgba(26,26,26,0.18)',
            borderBottom: '1px solid rgba(26,26,26,0.18)',
            padding: '6px 0',
            marginTop: 16,
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 10,
            letterSpacing: '0.24em',
            textTransform: 'uppercase',
            color: '#6a6a62',
          }}>
            <span>by {getAuthorDisplay(article)}</span>
            <span>{wordCount ? `${wordCount.toLocaleString()} words` : (article.character_count ? `${article.character_count.toLocaleString()} chars` : '')}</span>
          </div>
        </div>

        {getCentralClaim(article) && (
          <div style={{ marginTop: 22, padding: '14px 18px', borderTop: '2px solid #1a1a1a', borderBottom: '2px solid #1a1a1a' }}>
            <div className="tracker" style={{ color: 'var(--accent)' }}>the author's central claim</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 18, lineHeight: 1.45, fontStyle: 'italic', marginTop: 6 }}>
              {getCentralClaim(article)}
            </div>
            <div className="tracker" style={{ marginTop: 8 }}>summarized by the editor from the article body</div>
          </div>
        )}

        {(article.support_sentences && article.support_sentences.length > 0) && (
          <div style={{ marginTop: 20 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 8 }}>passages most relevant to your slip</div>
            {article.support_sentences.slice(0, 3).map((quote, i) => {
              const stanceColor = stance === 'supports' ? '#1a1a1a' : stance === 'complicates' ? '#7a1d1d' : '#9a9a92'
              return (
                <blockquote key={i} style={{
                  margin: '0 0 12px',
                  padding: '6px 0 6px 18px',
                  borderLeft: `2px solid ${stanceColor}`,
                  fontFamily: "'IM Fell English', serif",
                  fontStyle: 'italic',
                  fontSize: 16,
                  lineHeight: 1.5,
                  color: '#1a1a1a',
                }}>"{quote}"</blockquote>
              )
            })}
          </div>
        )}

        {(article.llm_relevant_paragraphs && article.llm_relevant_paragraphs.length > 0) && (
          <div style={{ marginTop: 20 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 8 }}>chunks the editor scored most relevant</div>
            {article.llm_relevant_paragraphs.slice(0, 3).map((para, i) => (
              <blockquote key={para.paragraph_id ?? i} style={{
                margin: '0 0 12px',
                padding: '6px 0 6px 18px',
                borderLeft: `2px solid #7a1d1d`,
                fontFamily: "'IM Fell English', serif",
                fontStyle: 'italic',
                fontSize: 15,
                lineHeight: 1.5,
                color: '#1a1a1a',
              }}>"{para.text}"</blockquote>
            ))}
          </div>
        )}

        <div style={{ marginTop: 22, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 28 }}>
          <div>
            <div className="tracker" style={{ marginBottom: 6, borderBottom: '1px solid #1a1a1a', paddingBottom: 6 }}>the editor's marks</div>
            <RFMetric label="topic relevance" value={getTopicPct(article)} />
            <RFMetric label="stance agreement" value={getAgreementPct(article)} accent />
            <RFMetric label="recency score" value={getRecencyPct(article)} />
            {sentiment && typeof sentiment.compound === 'number' && (
              <RFMetric
                label="sentiment compound"
                text={`${sentiment.compound >= 0 ? '+' : ''}${sentiment.compound.toFixed(2)} · ${sentiment.label}${sentiment.tone_strength ? ` · ${sentiment.tone_strength}` : ''}`}
              />
            )}
          </div>
          <div>
            <div className="tracker" style={{ marginBottom: 8, borderBottom: '1px solid #1a1a1a', paddingBottom: 6 }}>filed under</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {getDisplayKeywords(article).map(k => (
                <span key={k} style={{
                  border: '1px solid #1a1a1a',
                  padding: '3px 9px',
                  fontFamily: "'IM Fell DW Pica SC', serif",
                  fontSize: 9,
                  letterSpacing: '0.16em',
                  textTransform: 'uppercase',
                }}>{k}</span>
              ))}
              {getDisplayKeywords(article).length === 0 && (
                <span style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>
                  no tags filed
                </span>
              )}
            </div>
            <div style={{ marginTop: 14, fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62', lineHeight: 1.5 }}>
              {article.stance_method ? `agreement scored by ${article.stance_method.toUpperCase()}` : 'agreement scoring pending'}
            </div>
          </div>
        </div>

        {articleSvd.length > 0 && (
          <div style={{ marginTop: 28 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 6 }}>the latent dimensions · a compass of concepts</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#3a3a36', marginBottom: 8, lineHeight: 1.5 }}>
              Each spoke is a latent concept the SVD has surfaced from the morgue. The dark hull is <strong style={{ fontWeight: 400 }}>this article</strong>'s footprint; the oxblood hull is your <strong style={{ fontWeight: 400 }}>slip</strong>'s.
            </div>
            <VintageSvdRadar articleDims={articleSvd} queryDims={[]} />
          </div>
        )}

        <div style={{ marginTop: 30, paddingTop: 18, borderTop: '1px solid #1a1a1a', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          <button type="button" onClick={onChatThis} style={{
            background: '#1a1a1a',
            color: '#fafaf7',
            border: '1px solid #1a1a1a',
            padding: '12px 14px',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 9,
            letterSpacing: '0.26em',
            textTransform: 'uppercase',
            cursor: 'pointer',
          }}>↳ ask the editor about this article</button>
          <button type="button" onClick={onFindSimilar} style={{
            background: 'transparent',
            color: '#1a1a1a',
            border: '1px solid #1a1a1a',
            padding: '12px 14px',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 9,
            letterSpacing: '0.26em',
            textTransform: 'uppercase',
            cursor: 'pointer',
          }}>⌕ find similar proofs</button>
          <button type="button" onClick={onDismiss} style={{
            background: 'transparent',
            color: '#7a1d1d',
            border: '1px solid #7a1d1d',
            padding: '12px 14px',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 9,
            letterSpacing: '0.26em',
            textTransform: 'uppercase',
            cursor: 'pointer',
          }}>× mark not relevant</button>
        </div>

        {article.url && (
          <div style={{ marginTop: 16, textAlign: 'center' }}>
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontFamily: "'IM Fell DW Pica SC', serif",
                fontSize: 9,
                letterSpacing: '0.28em',
                textTransform: 'uppercase',
                color: '#6a6a62',
                borderBottom: '1px solid #6a6a62',
                paddingBottom: 1,
                textDecoration: 'none',
              }}
            >↗ open at source · {getOutlet(article)}</a>
          </div>
        )}
      </div>
    </div>
  )
}

function SimilarOverlay({
  source,
  similar,
  loading,
  hasMore,
  error,
  onClose,
  onLoadMore,
  onOpen,
}: {
  source: Article
  similar: Article[]
  loading: boolean
  hasMore: boolean
  error: string | null
  onClose: () => void
  onLoadMore: () => void
  onOpen: (article: Article) => void
}): JSX.Element {
  const sharedKeywords = (source.keywords ?? []).slice(0, 6)
  return (
    <div onClick={onClose} style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(26,26,26,0.42)',
      zIndex: 60,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      animation: 'rf-fade 220ms ease-out both',
    }}>
      <div onClick={(event) => event.stopPropagation()} style={{
        width: 760,
        maxHeight: '90%',
        background: '#fafaf7',
        border: '1px solid #1a1a1a',
        boxShadow: '0 24px 60px rgba(26,26,26,0.36)',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: "'Old Standard TT', serif",
        animation: 'rf-rise 360ms cubic-bezier(.2,.7,.2,1.05) both',
      }}>
        <div style={{ padding: '20px 28px 14px', borderBottom: '1px solid #1a1a1a' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <div className="tracker" style={{ color: 'var(--accent)' }}>from the morgue · neighbours of this proof</div>
            <button type="button" onClick={onClose} style={{
              background: 'transparent',
              border: '1px solid #1a1a1a',
              padding: '6px 12px',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 10,
              letterSpacing: '0.28em',
              textTransform: 'uppercase',
              color: '#1a1a1a',
              cursor: 'pointer',
            }}>close</button>
          </div>
          <div style={{ marginTop: 10, fontFamily: "'IM Fell English', serif", fontSize: 16, lineHeight: 1.4 }}>
            proofs that share latitude with <em style={{ fontStyle: 'italic' }}>"{source.title}"</em>
          </div>
          <div style={{ marginTop: 6, fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>
            measured by SVD cosine on the article body · {getAuthorDisplay(source)} · {getOutlet(source)}
          </div>
          {sharedKeywords.length > 0 && (
            <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {sharedKeywords.map(k => (
                <span key={k} style={{
                  border: '1px solid #1a1a1a',
                  padding: '3px 9px',
                  fontFamily: "'IM Fell DW Pica SC', serif",
                  fontSize: 9,
                  letterSpacing: '0.16em',
                  textTransform: 'uppercase',
                }}>{k}</span>
              ))}
            </div>
          )}
        </div>

        <div className="tray-scroll" style={{ flex: 1 }}>
          {error && (
            <div style={{
              padding: '14px 28px',
              fontFamily: "'IM Fell English', serif",
              fontStyle: 'italic',
              color: '#7a1d1d',
            }}>{error}</div>
          )}
          {similar.length === 0 && !loading && !error && (
            <div style={{
              padding: '24px 28px',
              fontFamily: "'IM Fell English', serif",
              fontStyle: 'italic',
              color: '#6a6a62',
            }}>No close neighbours surfaced from the morgue yet.</div>
          )}
          {similar.map((article) => {
            const sentColor = getSentColor(article)
            return (
              <div
                key={getArticleId(article)}
                onClick={() => onOpen(article)}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '76px 1fr 110px',
                  gap: 18,
                  alignItems: 'center',
                  padding: '14px 28px',
                  borderBottom: '1px solid rgba(26,26,26,0.18)',
                  cursor: 'pointer',
                }}
              >
                <div style={{
                  fontFamily: "'IM Fell English', serif",
                  fontSize: 22,
                  lineHeight: 1,
                  color: '#7a1d1d',
                  textAlign: 'center',
                }}>
                  {clampPct(article.score ?? null)}
                </div>
                <div>
                  <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 16, lineHeight: 1.2, display: 'flex', gap: 8, alignItems: 'baseline' }}>
                    <span style={{ width: 7, height: 7, borderRadius: '50%', background: sentColor, display: 'inline-block', flexShrink: 0 }} />
                    <span>{article.title}</span>
                  </div>
                  <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62', marginTop: 3 }}>
                    {getAuthorDisplay(article)} · {getOutlet(article)} · {formatDate(article.date)}
                  </div>
                  <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {(article.keywords ?? []).slice(0, 4).map(k => (
                      <span key={k} style={{
                        background: '#1a1a1a',
                        color: '#fafaf7',
                        padding: '2px 7px',
                        fontFamily: "'IM Fell DW Pica SC', serif",
                        fontSize: 8,
                        letterSpacing: '0.18em',
                        textTransform: 'uppercase',
                      }}>{k}</span>
                    ))}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <span className="tracker" style={{ borderBottom: '1px solid #1a1a1a', paddingBottom: 2, color: '#1a1a1a' }}>read the proof →</span>
                </div>
              </div>
            )
          })}
        </div>

        <div style={{
          padding: '12px 28px',
          borderTop: '1px solid #1a1a1a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span className="tracker">
            {similar.length} {similar.length === 1 ? 'proof' : 'proofs'} · sorted by cosine similarity
          </span>
          <button
            type="button"
            onClick={onLoadMore}
            disabled={!hasMore || loading}
            style={{
              background: 'transparent',
              color: hasMore ? '#1a1a1a' : '#9a9a92',
              border: '1px solid',
              borderColor: hasMore ? '#1a1a1a' : '#9a9a92',
              padding: '7px 14px',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 9,
              letterSpacing: '0.26em',
              textTransform: 'uppercase',
              cursor: hasMore && !loading ? 'pointer' : 'not-allowed',
            }}
          >{loading ? 'loading…' : hasMore ? 'load more →' : 'no more'}</button>
        </div>
      </div>
    </div>
  )
}

function ResultsChatOverlay({
  total,
  messages,
  input,
  loading,
  error,
  onClose,
  onInputChange,
  onSubmit,
}: {
  total: number
  messages: ResultsChatMessage[]
  input: string
  loading: boolean
  error: string | null
  onClose: () => void
  onInputChange: (next: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}): JSX.Element {
  const presets = [
    'Why are some authors ranked above others?',
    'Which results disagree with each other most sharply?',
    'Summarize the strongest argument against my claim.',
  ]
  return (
    <div onClick={onClose} style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(26,26,26,0.32)',
      zIndex: 40,
      display: 'flex',
      justifyContent: 'flex-end',
      animation: 'rf-fade 220ms ease-out both',
    }}>
      <div onClick={(event) => event.stopPropagation()} style={{
        width: 560,
        height: '100%',
        background: '#fafaf7',
        borderLeft: '1px solid #1a1a1a',
        animation: 'rf-slide 360ms cubic-bezier(.2,.7,.2,1.05) both',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: "'Old Standard TT', serif",
      }}>
        <div style={{
          padding: '20px 24px 14px',
          borderBottom: '1px solid #1a1a1a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'baseline',
        }}>
          <div>
            <div className="tracker" style={{ color: 'var(--accent)' }}>at the editor's desk</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 24, marginTop: 2 }}>Editor Hollis</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>has read all {total} proofs · cites by [number]</div>
          </div>
          <button type="button" onClick={onClose} style={{
            background: 'transparent',
            border: '1px solid #1a1a1a',
            padding: '6px 12px',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 10,
            letterSpacing: '0.28em',
            textTransform: 'uppercase',
            color: '#1a1a1a',
            cursor: 'pointer',
          }}>close</button>
        </div>

        <div className="tray-scroll" style={{ flex: 1, padding: '20px 24px' }}>
          {messages.length === 0 && !loading && (
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 14, color: '#6a6a62', marginBottom: 14 }}>
              Good afternoon. I have your proofs in front of me. Ask whatever you like — about the running order, the dissent, or any single proof.
            </div>
          )}
          {messages.map((message) => (
            <ChatBubble key={message.id} message={message} />
          ))}
          {loading && messages.length > 0 && messages[messages.length - 1].content === '' && (
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#6a6a62', display: 'flex', gap: 8, alignItems: 'center' }}>
              <RFSpinner /> Hollis is at the lectern…
            </div>
          )}
          {error && (
            <div style={{ marginTop: 12, fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#7a1d1d' }}>{error}</div>
          )}
          <div style={{ marginTop: 16 }}>
            <div className="tracker" style={{ marginBottom: 6 }}>ready-made questions</div>
            {presets.map(q => (
              <button
                key={q}
                type="button"
                onClick={() => onInputChange(q)}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 12px',
                  marginBottom: 6,
                  background: 'transparent',
                  border: '1px solid rgba(26,26,26,0.4)',
                  cursor: 'pointer',
                  fontFamily: "'IM Fell English', serif",
                  fontStyle: 'italic',
                  fontSize: 13,
                  color: '#1a1a1a',
                }}
              >{q}</button>
            ))}
          </div>
        </div>

        <form onSubmit={onSubmit} style={{ padding: '14px 24px', borderTop: '1px solid #1a1a1a', display: 'flex', gap: 10 }}>
          <input
            value={input}
            onChange={(event) => onInputChange(event.target.value)}
            placeholder="ask the editor…"
            style={{
              flex: 1,
              padding: '10px 12px',
              fontFamily: "'Special Elite', monospace",
              fontSize: 13,
              color: '#1a1a1a',
              border: '1px solid #1a1a1a',
              background: '#fafaf7',
              outline: 'none',
            }}
          />
          <button
            type="submit"
            disabled={loading || input.trim() === ''}
            style={{
              background: '#1a1a1a',
              color: '#fafaf7',
              border: '1px solid #1a1a1a',
              padding: '8px 18px',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 10,
              letterSpacing: '0.28em',
              textTransform: 'uppercase',
              cursor: loading || input.trim() === '' ? 'not-allowed' : 'pointer',
              opacity: loading || input.trim() === '' ? 0.6 : 1,
            }}
          >send →</button>
        </form>
      </div>
    </div>
  )
}

function ArticleChatOverlay({
  article,
  messages,
  input,
  loading,
  error,
  onClose,
  onInputChange,
  onSubmit,
}: {
  article: Article
  messages: ResultsChatMessage[]
  input: string
  loading: boolean
  error: string | null
  onClose: () => void
  onInputChange: (next: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}): JSX.Element {
  const presets = [
    'Explain why this ranks where it does.',
    'What is the strongest counter-argument the author considers?',
    'Quote me a passage that complicates my claim.',
  ]
  return (
    <div onClick={onClose} style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(26,26,26,0.42)',
      zIndex: 50,
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      animation: 'rf-fade 220ms ease-out both',
    }}>
      <div onClick={(event) => event.stopPropagation()} style={{
        width: 640,
        maxHeight: '88%',
        background: '#fafaf7',
        border: '1px solid #1a1a1a',
        boxShadow: '0 24px 60px rgba(26,26,26,0.36)',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: "'Old Standard TT', serif",
      }}>
        <div style={{
          padding: '18px 22px',
          borderBottom: '1px solid #1a1a1a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'baseline',
          gap: 14,
        }}>
          <div>
            <div className="tracker" style={{ color: 'var(--accent)' }}>asking about a single proof</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 18, marginTop: 4, lineHeight: 1.2 }}>{article.title}</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>
              by {getAuthorDisplay(article)} · {getOutlet(article)}
            </div>
          </div>
          <button type="button" onClick={onClose} style={{
            background: 'transparent',
            border: '1px solid #1a1a1a',
            padding: '6px 12px',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 10,
            letterSpacing: '0.28em',
            textTransform: 'uppercase',
            color: '#1a1a1a',
            cursor: 'pointer',
          }}>close</button>
        </div>

        <div className="tray-scroll" style={{ flex: 1, padding: '18px 22px', minHeight: 200 }}>
          {messages.length === 0 && !loading && (
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#6a6a62', marginBottom: 12 }}>
              I've read this proof in full. What do you want to know?
            </div>
          )}
          {messages.map(message => (
            <ChatBubble key={message.id} message={message} />
          ))}
          {error && (
            <div style={{ marginTop: 12, fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#7a1d1d' }}>{error}</div>
          )}
          <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {presets.map(p => (
              <button
                key={p}
                type="button"
                onClick={() => onInputChange(p)}
                style={{
                  background: 'transparent',
                  border: '1px solid rgba(26,26,26,0.4)',
                  padding: '6px 10px',
                  fontFamily: "'IM Fell English', serif",
                  fontStyle: 'italic',
                  fontSize: 12,
                  cursor: 'pointer',
                  color: '#1a1a1a',
                }}
              >{p}</button>
            ))}
          </div>
        </div>

        <form onSubmit={onSubmit} style={{ padding: '12px 22px', borderTop: '1px solid #1a1a1a', display: 'flex', gap: 8 }}>
          <input
            value={input}
            onChange={(event) => onInputChange(event.target.value)}
            placeholder="explain the ranking, quote a passage…"
            style={{
              flex: 1,
              padding: '8px 12px',
              fontFamily: "'Special Elite', monospace",
              fontSize: 12,
              border: '1px solid #1a1a1a',
              background: '#fafaf7',
              outline: 'none',
              color: '#1a1a1a',
            }}
          />
          <button
            type="submit"
            disabled={loading || input.trim() === ''}
            style={{
              background: '#1a1a1a',
              color: '#fafaf7',
              border: '1px solid #1a1a1a',
              padding: '6px 16px',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 10,
              letterSpacing: '0.28em',
              textTransform: 'uppercase',
              cursor: loading || input.trim() === '' ? 'not-allowed' : 'pointer',
              opacity: loading || input.trim() === '' ? 0.6 : 1,
            }}
          >send →</button>
        </form>
      </div>
    </div>
  )
}

function OverviewItem({ item, sources }: { item: ResultsOverviewArgument; sources: ResultsOverviewSource[] }): JSX.Element {
  const indices = item.source_indices ?? []
  const sourceMap = new Map(sources.map(s => [s.result_index, s]))
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '36px 1fr',
      gap: 10,
      alignItems: 'baseline',
      paddingBottom: 6,
      borderBottom: '1px dotted rgba(26,26,26,0.2)',
    }}>
      <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 18, color: '#7a1d1d', fontStyle: 'italic' }}>"</span>
      <div>
        <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 14, lineHeight: 1.45, color: '#1a1a1a' }}>
          {item.argument}
        </div>
        <div style={{
          fontFamily: "'IM Fell DW Pica SC', serif",
          fontSize: 9,
          letterSpacing: '0.24em',
          textTransform: 'uppercase',
          color: '#6a6a62',
          marginTop: 4,
          display: 'flex',
          flexWrap: 'wrap',
          gap: 6,
        }}>
          {indices.map(i => {
            const source = sourceMap.get(i)
            if (!source) return null
            return (
              <span key={i} style={{ border: '1px solid currentColor', padding: '2px 6px' }}>[{i}] {source.title.slice(0, 24)}{source.title.length > 24 ? '…' : ''}</span>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export function ResultsFlow(props: ResultsFlowProps): JSX.Element {
  const {
    topic,
    opinion,
    inputMode,
    essayText,
    thesisSentence,
    loading,
    error,
    emptyResultsMessage,
    articles,
    dismissedIds,
    onDismiss,
    onApplyDismissals,
    onBackToCompose,
    typoCorrection,
    onApplyTypoCorrection,
    onSearchAnyway,
    rewriteAlternatives,
    rewriteLoading,
    rewriteError,
    onLoadRewrites,
    onApplyRewrite,
    overview,
    overviewDraft,
    overviewLoading,
    overviewError,
    similarSource,
    similarArticles,
    similarHasMore,
    similarLoading,
    similarError,
    onFindSimilar,
    onLoadMoreSimilar,
    onCloseSimilar,
    chatMessages,
    chatInput,
    chatLoading,
    chatError,
    onChatInputChange,
    onSubmitChat,
    articleChatMessages,
    articleChatInput,
    articleChatLoading,
    articleChatError,
    articleChatArticleId,
    onOpenArticleChat,
    onCloseArticleChat,
    onArticleChatInputChange,
    onSubmitArticleChat,
    progressLines,
    progressMessage,
    querySvdDimensions,
  } = props

  const [openId, setOpenId] = useState<string | null>(null)
  const [chatOpen, setChatOpen] = useState(false)

  const visibleArticles = useMemo(
    () => articles.filter(a => !dismissedIds.has(getArticleId(a))),
    [articles, dismissedIds],
  )

  const supporting = useMemo(
    () => visibleArticles.filter(a => getStanceCategory(a) === 'supports').slice(0, 3),
    [visibleArticles],
  )
  const opposing = useMemo(
    () => visibleArticles.filter(a => getStanceCategory(a) === 'complicates').slice(0, 3),
    [visibleArticles],
  )

  const stage: 1 | 2 | 3 = (() => {
    if (loading && articles.length === 0) return 1
    if (loading && articles.length > 0) return 2
    return 3
  })()

  const hasTypo = typoCorrection !== null
  const stageOneVisible = hasTypo || rewriteAlternatives.length > 0 || rewriteLoading
  const stageLabels = [
    { n: 1, title: 'Query pinned', caption: 'slip rises · typo & rewrite checked' },
    { n: 2, title: 'Topic relevance in', caption: 'proofs scatter on the desk' },
    { n: 3, title: 'Agreement in', caption: 'ranked ledger · structured overview · the editor' },
  ] as const

  const overviewSources = overview?.sources ?? []
  const articleChatTarget = articleChatArticleId
    ? articles.find(a => getArticleId(a) === articleChatArticleId) ?? null
    : null

  const openArticle = articles.find(a => getArticleId(a) === openId) ?? null
  const visibleStage: 1 | 2 | 3 = stageOneVisible ? 1 : stage

  return (
    <div className="stage-shell" style={{ position: 'relative' }}>
      <div className="top-rail">
        <button type="button" className="top-rail-brand" onClick={onBackToCompose}>hear! hear!</button>
        <div className="top-rail-links">
          <button type="button" onClick={onBackToCompose}>compose</button>
          <span style={{ color: '#1a1a1a' }}>the search · stage {visibleStage} of 3</span>
          <button type="button" onClick={onBackToCompose}>← back to compose</button>
        </div>
      </div>
      <div className="top-rule" />

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button type="button" onClick={onBackToCompose}>edit slip →</button>
        </div>
      )}

      {/* pinned slip */}
      <div style={{ padding: '18px 48px 0' }}>
        <PinnedSlip
          topic={topic}
          opinion={opinion}
          mode={inputMode}
          essayText={essayText}
          thesisSentence={thesisSentence}
          animateOnMount
        />
      </div>

      {/* step rail */}
      <div style={{
        margin: '14px 48px 0',
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        borderTop: '1px solid #1a1a1a',
        borderBottom: '1px solid #1a1a1a',
      }}>
        {stageLabels.map((s, i) => {
          const active = visibleStage === s.n
          const passed = visibleStage > s.n
          return (
            <div
              key={s.n}
              style={{
                background: active ? '#1a1a1a' : 'transparent',
                color: active ? '#fafaf7' : (passed ? '#1a1a1a' : '#6a6a62'),
                borderLeft: i === 0 ? 0 : '1px solid #1a1a1a',
                padding: '10px 14px',
                fontFamily: "'IM Fell DW Pica SC', serif",
              }}
            >
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 10,
                letterSpacing: '0.24em',
                textTransform: 'uppercase',
              }}>
                <span style={{ opacity: 0.7 }}>{passed ? '✓' : `0${s.n}`}</span>
                <span>{s.title}</span>
              </div>
              <div style={{
                fontFamily: "'IM Fell English', serif",
                fontStyle: 'italic',
                fontSize: 12,
                marginTop: 2,
                opacity: 0.85,
                textTransform: 'none',
                letterSpacing: 'normal',
              }}>{s.caption}</div>
            </div>
          )
        })}
      </div>

      {/* body */}
      <div style={{ position: 'relative', padding: '24px 48px 0', minHeight: 540 }}>
        {/* Stage 1: typo + rewrite */}
        {stageOneVisible && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 36, paddingTop: 8 }}>
            <div>
              <div className="tracker" style={{ color: 'var(--accent)', fontSize: 10, letterSpacing: '0.32em' }}>the proofreader's mark</div>
              {hasTypo ? (
                <>
                  <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 17, lineHeight: 1.55, marginTop: 8 }}>
                    A small thing — the slip read "<span style={{ position: 'relative', borderBottom: '2px wavy #7a1d1d', paddingBottom: 2 }}>{typoCorrection.query}</span>". The morgue suggests a more usual spelling. Shall I correct it?
                  </div>
                  <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 0, border: '1px solid #1a1a1a' }}>
                    {typoCorrection.options.map((option, i) => (
                      <button
                        key={option.query}
                        type="button"
                        onClick={() => onApplyTypoCorrection(option.query)}
                        style={{
                          padding: '10px 14px',
                          textAlign: 'left',
                          cursor: 'pointer',
                          background: 'transparent',
                          color: '#1a1a1a',
                          border: 0,
                          borderTop: i === 0 ? 0 : '1px solid #1a1a1a',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          fontFamily: "'Old Standard TT', serif",
                        }}
                      >
                        <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 14 }}>{option.query}</span>
                        <span className="tracker">
                          {option.distance !== null && option.distance !== undefined ? `edit-distance ${option.distance}` : ''}
                          {option.df ? ` · seen ${option.df.toLocaleString()}×` : ''}
                        </span>
                      </button>
                    ))}
                    <button
                      type="button"
                      onClick={onSearchAnyway}
                      style={{
                        padding: '10px 14px',
                        textAlign: 'left',
                        cursor: 'pointer',
                        background: 'transparent',
                        color: '#7a1d1d',
                        border: 0,
                        borderTop: '1px solid #1a1a1a',
                        fontFamily: "'IM Fell DW Pica SC', serif",
                        fontSize: 10,
                        letterSpacing: '0.28em',
                        textTransform: 'uppercase',
                      }}
                    >search as written →</button>
                  </div>
                </>
              ) : (
                <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 14, color: '#6a6a62', marginTop: 8 }}>
                  No spelling marks against your slip — proceeding to scoring.
                </div>
              )}
            </div>

            <div style={{ borderLeft: '1px solid rgba(26,26,26,0.12)', paddingLeft: 28 }}>
              <div className="tracker" style={{ color: 'var(--accent)', fontSize: 10, letterSpacing: '0.32em' }}>the rewrite desk · suggested by the editor</div>
              <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 14, color: '#3a3a36', marginTop: 6, marginBottom: 10 }}>
                Three alternative searches that may surface a richer running order. Click one to file it instead.
              </div>
              {rewriteAlternatives.length === 0 && !rewriteLoading && !rewriteError && (
                <button type="button" onClick={onLoadRewrites} className="btn-stamp">ask the editor for rewrites</button>
              )}
              {rewriteLoading && (
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10, fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#6a6a62' }}>
                  <RFSpinner /><span>· · · Mrs. Calder is at the type case</span>
                </div>
              )}
              {rewriteError && (
                <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#7a1d1d' }}>{rewriteError}</div>
              )}
              {rewriteAlternatives.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 0, border: '1px solid #1a1a1a' }}>
                  {rewriteAlternatives.map((r, i) => (
                    <button
                      key={`${r.topic}-${r.opinion}`}
                      type="button"
                      onClick={() => onApplyRewrite(r)}
                      style={{
                        padding: '12px 14px',
                        textAlign: 'left',
                        cursor: 'pointer',
                        background: 'transparent',
                        color: '#1a1a1a',
                        border: 0,
                        borderTop: i === 0 ? 0 : '1px solid #1a1a1a',
                        fontFamily: "'Old Standard TT', serif",
                      }}
                    >
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 14, opacity: 0.78 }}>regarding {r.topic} · I believe {r.opinion}</div>
                      <div style={{ fontFamily: "'Special Elite', monospace", fontSize: 13, marginTop: 4 }}>↦ {r.query}</div>
                      {r.rationale && (
                        <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 11, marginTop: 4, opacity: 0.65 }}>{r.rationale}</div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Stage 2: scatter while loading */}
        {!stageOneVisible && stage <= 2 && (
          <div style={{ display: 'grid', gridTemplateColumns: '820px 1fr', gap: 32 }}>
            <div style={{ position: 'relative', height: 470 }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontFamily: "'IM Fell DW Pica SC', serif",
                fontSize: 9,
                letterSpacing: '0.28em',
                textTransform: 'uppercase',
                color: '#6a6a62',
                marginBottom: 10,
              }}>
                <span>the desk · proofs that cleared topic-relevance</span>
                <span>{visibleArticles.length} of {articles.length} retained{dismissedIds.size ? ` · ${dismissedIds.size} dismissed` : ''}</span>
              </div>
              <div style={{ position: 'relative', width: 820, height: 440 }}>
                {visibleArticles.slice(0, 8).map((article, i) => (
                  <RFScatterCard
                    key={getArticleId(article)}
                    article={article}
                    spot={SCATTER_SPOTS[i % SCATTER_SPOTS.length]}
                    delay={i * 80}
                    onDismiss={() => onDismiss(article)}
                  />
                ))}
              </div>
            </div>
            <div style={{ paddingTop: 4, borderLeft: '1px solid rgba(26,26,26,0.12)', paddingLeft: 28 }}>
              <div className="tracker" style={{ color: 'var(--accent)', fontSize: 10, letterSpacing: '0.28em', marginBottom: 10 }}>from editor hollis, at the lectern</div>
              <div style={{
                fontFamily: "'IM Fell English', serif",
                fontStyle: 'italic',
                fontSize: 18,
                lineHeight: 1.55,
                color: '#1a1a1a',
              }}>
                {progressMessage || 'Here are some relevant articles, dear reader. I am presently reranking them by how closely they agree with you.'}
                {progressMessage && (
                  <span style={{ display: 'block', marginTop: 6, fontFamily: "'Special Elite', monospace", fontSize: 12, fontStyle: 'normal', color: '#6a6a62' }}>(this can take around 30 seconds)</span>
                )}
              </div>
              <div style={{ marginTop: 22, display: 'flex', alignItems: 'center', gap: 10, fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#6a6a62' }}>
                <RFSpinner /><span>· · · {progressMessage ?? 'scoring agreement'}</span>
              </div>
              <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 4, fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#3a3a36' }}>
                {progressLines.map(line => (
                  <RFProgressLine key={line.label} line={line} />
                ))}
              </div>
              {dismissedIds.size > 0 && (
                <div style={{ marginTop: 18, display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>
                  <span>{dismissedIds.size} dismissed — pull next-best?</span>
                  <button type="button" onClick={onApplyDismissals} className="btn-stamp" style={{ padding: '6px 12px', fontSize: 9 }}>refresh</button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Stage 3: structured overview + ledger */}
        {!stageOneVisible && stage === 3 && (
          <div style={{ display: 'grid', gridTemplateColumns: '520px 1fr', gap: 36 }}>
            <div className="tray-scroll" style={{ display: 'flex', flexDirection: 'column', gap: 14, maxHeight: 'calc(100dvh - 360px)', paddingRight: 4 }}>
              <div className="tracker" style={{ color: 'var(--accent)', fontSize: 10, letterSpacing: '0.32em' }}>editor hollis returns · the structured overview</div>
              {overviewLoading && !overview && (
                <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 14, color: '#6a6a62', display: 'flex', alignItems: 'center', gap: 10 }}>
                  <RFSpinner /> <span>{overviewDraft || 'composing the brief…'}</span>
                </div>
              )}
              {overviewError && (
                <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#7a1d1d' }}>{overviewError}</div>
              )}
              {overview && (
                <>
                  <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.5, color: '#1a1a1a' }}>
                    {overview.overview}
                  </div>
                  {(overview.supporting_arguments && overview.supporting_arguments.length > 0) && (
                    <div style={{ borderTop: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a', padding: '10px 0' }}>
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 18, color: '#1a1a1a', marginBottom: 8 }}>Authors who support you</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {overview.supporting_arguments.map((item, i) => (
                          <OverviewItem key={i} item={item} sources={overviewSources} />
                        ))}
                      </div>
                    </div>
                  )}
                  {(overview.opposing_arguments && overview.opposing_arguments.length > 0) && (
                    <div style={{ borderTop: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a', padding: '10px 0' }}>
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 18, color: 'var(--accent)', marginBottom: 8 }}>Authors who challenge you</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {overview.opposing_arguments.map((item, i) => (
                          <OverviewItem key={i} item={item} sources={overviewSources} />
                        ))}
                      </div>
                    </div>
                  )}
                  {overview.caveat && (
                    <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#6a6a62' }}>
                      {overview.caveat}
                    </div>
                  )}
                </>
              )}
              {!overview && !overviewLoading && !overviewError && (supporting.length > 0 || opposing.length > 0) && (
                <>
                  {supporting.length > 0 && (
                    <div style={{ borderTop: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a', padding: '10px 0' }}>
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 18, color: '#1a1a1a', marginBottom: 8 }}>Authors who support you</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {supporting.map(article => (
                          <div
                            key={getArticleId(article)}
                            style={{ display: 'grid', gridTemplateColumns: '36px 1fr', gap: 10, alignItems: 'baseline', paddingBottom: 6, borderBottom: '1px dotted rgba(26,26,26,0.2)' }}
                          >
                            <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 18, color: '#7a1d1d', fontStyle: 'italic' }}>"</span>
                            <div>
                              <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 14, lineHeight: 1.45, color: '#1a1a1a' }}>{getCentralClaim(article) || article.title}</div>
                              <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: '#6a6a62', marginTop: 4, display: 'flex', justifyContent: 'space-between' }}>
                                <span>— {getAuthorDisplay(article)} · {getOutlet(article)}</span>
                                <button
                                  type="button"
                                  onClick={() => setOpenId(getArticleId(article))}
                                  style={{
                                    background: 'transparent',
                                    border: 0,
                                    padding: 0,
                                    cursor: 'pointer',
                                    fontFamily: 'inherit',
                                    fontSize: 'inherit',
                                    letterSpacing: 'inherit',
                                    textTransform: 'inherit',
                                    color: '#1a1a1a',
                                    borderBottom: '1px solid #1a1a1a',
                                  }}
                                >read the proof →</button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {opposing.length > 0 && (
                    <div style={{ borderTop: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a', padding: '10px 0' }}>
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 18, color: 'var(--accent)', marginBottom: 8 }}>Authors who challenge you</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {opposing.map(article => (
                          <div
                            key={getArticleId(article)}
                            style={{ display: 'grid', gridTemplateColumns: '36px 1fr', gap: 10, alignItems: 'baseline', paddingBottom: 6, borderBottom: '1px dotted rgba(26,26,26,0.2)' }}
                          >
                            <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 18, color: '#7a1d1d', fontStyle: 'italic' }}>"</span>
                            <div>
                              <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 14, lineHeight: 1.45, color: '#1a1a1a' }}>{getCentralClaim(article) || article.title}</div>
                              <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.24em', textTransform: 'uppercase', color: '#6a6a62', marginTop: 4, display: 'flex', justifyContent: 'space-between' }}>
                                <span>— {getAuthorDisplay(article)} · {getOutlet(article)}</span>
                                <button
                                  type="button"
                                  onClick={() => setOpenId(getArticleId(article))}
                                  style={{
                                    background: 'transparent',
                                    border: 0,
                                    padding: 0,
                                    cursor: 'pointer',
                                    fontFamily: 'inherit',
                                    fontSize: 'inherit',
                                    letterSpacing: 'inherit',
                                    textTransform: 'inherit',
                                    color: '#1a1a1a',
                                    borderBottom: '1px solid #1a1a1a',
                                  }}
                                >read the proof →</button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}

              {visibleArticles.length > 0 && (
                <div>
                  <div className="tracker" style={{ borderBottom: '1px solid #1a1a1a', paddingBottom: 4, marginBottom: 6 }}>sources cited above</div>
                  <div style={{ fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#3a3a36', lineHeight: 1.7 }}>
                    {visibleArticles.slice(0, 12).map((article, i) => (
                      <div key={getArticleId(article)}>[{String(i + 1).padStart(2, '0')}] {getAuthorDisplay(article)} — <em style={{ fontStyle: 'italic' }}>{getOutlet(article)}</em>, {formatDate(article.date)}</div>
                    ))}
                  </div>
                </div>
              )}

              <button
                type="button"
                onClick={() => setChatOpen(true)}
                style={{
                  marginTop: 4,
                  background: '#1a1a1a',
                  color: '#fafaf7',
                  border: '1px solid #1a1a1a',
                  padding: '12px 18px',
                  fontFamily: "'IM Fell DW Pica SC', serif",
                  fontSize: 10,
                  letterSpacing: '0.32em',
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  gap: 12,
                }}
              >
                <span>↳ ask the editor about these results</span>
                <span>→</span>
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
                <div className="tracker" style={{ fontSize: 10, letterSpacing: '0.32em' }}>the ledger · ranked 01—{String(visibleArticles.length).padStart(2, '0')}</div>
                <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>click for the broadsheet</div>
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: '52px 1fr 96px 28px',
                gap: 14,
                alignItems: 'center',
                fontFamily: "'IM Fell DW Pica SC', serif",
                fontSize: 8,
                letterSpacing: '0.24em',
                textTransform: 'uppercase',
                color: '#9a9a92',
                padding: '4px 4px',
                borderTop: '1.5px solid #1a1a1a',
                borderBottom: '1px solid rgba(26,26,26,0.18)',
              }}>
                <span>rank</span><span>title · author</span><span>marks</span><span></span>
              </div>
              <div className="tray-scroll" style={{ maxHeight: 'calc(100dvh - 360px)' }}>
                {visibleArticles.length === 0 && (
                  <div style={{ padding: 24, fontFamily: "'IM Fell English', serif", fontStyle: 'italic', color: '#6a6a62' }}>
                    {emptyResultsMessage || 'No proofs cleared the bench.'}
                  </div>
                )}
                {visibleArticles.map((article, i) => (
                  <RFRankedRow
                    key={getArticleId(article)}
                    article={article}
                    rank={i + 1}
                    active={openId === getArticleId(article)}
                    onClick={() => setOpenId(getArticleId(article))}
                    onDismiss={() => onDismiss(article)}
                  />
                ))}
              </div>
              {dismissedIds.size > 0 && (
                <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>
                  <span>
                    {dismissedIds.size} {dismissedIds.size === 1 ? 'proof was' : 'proofs were'} marked not relevant.
                  </span>
                  <button type="button" onClick={onApplyDismissals} className="btn-stamp" style={{ padding: '4px 10px', fontSize: 9 }}>pull next-best</button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* footer */}
      <div className="footer-rail" style={{ marginTop: 32 }}>
        <span>the search · after press</span>
        <span>{topic ? `${topic} · ${visibleArticles.length} of ${articles.length} retained` : `${visibleArticles.length} retained${dismissedIds.size ? ` · ${dismissedIds.size} dismissed` : ''}`}</span>
        <span>guardian opinion · indexed</span>
      </div>

      {/* overlays */}
      {openArticle && (
        <BroadsheetOverlay
          article={openArticle}
          rank={visibleArticles.findIndex(a => getArticleId(a) === openId) + 1}
          total={visibleArticles.length}
          onClose={() => setOpenId(null)}
          onDismiss={() => { onDismiss(openArticle); setOpenId(null) }}
          onChatThis={() => onOpenArticleChat(openArticle)}
          onFindSimilar={() => onFindSimilar(openArticle)}
        />
      )}
      {chatOpen && (
        <ResultsChatOverlay
          total={visibleArticles.length}
          messages={chatMessages}
          input={chatInput}
          loading={chatLoading}
          error={chatError}
          onClose={() => setChatOpen(false)}
          onInputChange={onChatInputChange}
          onSubmit={onSubmitChat}
        />
      )}
      {articleChatTarget && (
        <ArticleChatOverlay
          article={articleChatTarget}
          messages={articleChatMessages}
          input={articleChatInput}
          loading={articleChatLoading}
          error={articleChatError}
          onClose={onCloseArticleChat}
          onInputChange={onArticleChatInputChange}
          onSubmit={onSubmitArticleChat}
        />
      )}
      {similarSource && (
        <SimilarOverlay
          source={similarSource}
          similar={similarArticles}
          loading={similarLoading}
          hasMore={similarHasMore}
          error={similarError}
          onClose={onCloseSimilar}
          onLoadMore={onLoadMoreSimilar}
          onOpen={(article) => {
            // open it inside the same flow if it exists in articles
            const id = getArticleId(article)
            const existing = articles.find(a => getArticleId(a) === id)
            if (existing) {
              onCloseSimilar()
              setOpenId(id)
            } else if (article.url) {
              window.open(article.url, '_blank', 'noopener')
            }
          }}
        />
      )}

      {/* unused import quieting */}
      <div style={{ display: 'none' }}>{querySvdDimensions.length}{(undefined as unknown as SimilarArticlesResponse | undefined)?.results?.length}</div>
    </div>
  )
}

export default ResultsFlow
