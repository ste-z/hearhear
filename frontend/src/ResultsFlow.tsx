import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import type {
  Article,
  EssayClaimCandidate,
  QueryRewriteAlternative,
  ResultsOverview,
  ResultsOverviewArgument,
  ResultsOverviewSource,
  SvdLatentDimension,
  TypoCorrectionSuggestion,
} from './types'
import PersonaName from './PersonaName'

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
  attachments?: Array<{ articleId: string; resultIndex: number; title: string }> | null
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
  llmLabelIrrelevant: boolean
  stage: 'stage1' | 'stage2' | 'stage3'
  dismissedIds: Set<string>
  onDismiss: (article: Article) => void
  onUndoDismiss: (article: Article) => void
  onApplyDismissals: () => void
  onBackToCompose: () => void
  onOpenAbout: () => void
  onOpenMethod: () => void

  // typo + rewrite
  typoCorrection: TypoCorrectionSuggestion | null
  onApplyTypoCorrection: (query: string) => void
  onSearchAnyway: () => void
  rewriteAlternatives: QueryRewriteAlternative[]
  rewriteLoading: boolean
  rewriteError: string | null
  onLoadRewrites: () => void
  onApplyRewrite: (alternative: QueryRewriteAlternative) => void

  // essay-mode stage 1 (NLI thesis picker)
  essayCandidates: EssayClaimCandidate[]
  selectedThesisId: string | null
  onSelectThesisCandidate: (id: string) => void
  thesisMode: 'candidate' | 'custom'
  onThesisModeChange: (mode: 'candidate' | 'custom') => void
  customThesis: string
  onCustomThesisChange: (value: string) => void
  onConfirmEssayThesis: () => void

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
  chatAttachedIds: string[]
  onToggleChatAttachment: (id: string) => void
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
  querySvdCorpusChartDimensions: SvdLatentDimension[]

  // The active retrieval model — used to attribute the stage-2 "compositor
  // is finding articles" sticky note to the correct persona.
  effectiveRetrievalModel: 'tfidf' | 'svd' | 'minilm'

  // explain ranking
  rankingExplanations: Record<string, { loading: boolean; explanation: string | null; error: string | null }>
  onExplainRanking: (article: Article, rank: number) => void
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

function isLlmIrrelevantArticle(article: Article): boolean {
  return article.stance_method === 'llm' && article.llm_irrelevant === true
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

/**
 * SortFlight — Stage 2 → Stage 3 transition animation.
 *
 * The ledger underneath is rendered fully (rank · marks · dismiss columns)
 * with empty article-info slots. The cards fly in from their scatter
 * positions and land into those empty slots, then fade out so the ledger
 * settles into its flat row style.
 */
function SortFlight({ articles }: { articles: Article[] }): JSX.Element {
  const [moved, setMoved] = useState(false)
  const [shed, setShed] = useState(false)
  useEffect(() => {
    let raf2 = 0
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setMoved(true))
    })
    const t = setTimeout(() => setShed(true), 1050)
    return () => {
      cancelAnimationFrame(raf1)
      if (raf2) cancelAnimationFrame(raf2)
      clearTimeout(t)
    }
  }, [])

  // Layout constants — must track the ledger panel's actual layout.
  // Caption row "the ledger · ranked NN—NN" with marginBottom: 6.
  const CAPTION_H = 26
  // Column header row "rank | title · author | marks | x" with 1.5 + 1 px borders.
  const TABLE_HEAD_H = 28
  // Each RFRankedRow: padding 12*2 + ~36px content ≈ 60px pitch.
  const ROW_H = 60
  // Top inset inside a row before the article-info cell visual content.
  const ROW_PAD_TOP = 8
  // Article-info cell's left edge inside the row grid (52px rank + 14px gap).
  const SLOT_X = 66

  return (
    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, pointerEvents: 'none', zIndex: 12 }}>
      {articles.slice(0, 8).map((article, i) => {
        const spot = SCATTER_SPOTS[i % SCATTER_SPOTS.length]
        const sentColor = getSentColor(article)
        const fromX = spot.x
        const fromY = CAPTION_H + spot.y
        const toX = SLOT_X
        const toY = CAPTION_H + TABLE_HEAD_H + i * ROW_H + ROW_PAD_TOP
        return (
          <div
            key={getArticleId(article)}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: 280,
              padding: '8px 12px',
              background: shed ? 'transparent' : '#fafaf7',
              border: shed ? '1px solid transparent' : '1px solid #1a1a1a',
              boxShadow: shed ? 'none' : '2px 4px 0 rgba(26,26,26,0.06)',
              fontFamily: "'Old Standard TT', serif",
              transform: moved
                ? `translate(${toX}px, ${toY}px) rotate(0deg)`
                : `translate(${fromX}px, ${fromY}px) rotate(${spot.r}deg)`,
              transformOrigin: 'top left',
              transition: 'transform 0.95s cubic-bezier(.2,.7,.2,1.05), background 0.45s ease, border-color 0.45s ease, box-shadow 0.45s ease, opacity 0.4s ease 0.1s',
              opacity: shed ? 0 : 1,
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
              <span style={{ color: '#1a1a1a', fontFamily: "'Special Elite', monospace" }}>τ {getTopicPct(article)}</span>
            </div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 14, lineHeight: 1.2, marginTop: 4, color: '#1a1a1a' }}>
              {article.title.length > 64 ? `${article.title.slice(0, 62)}…` : article.title}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function ExpandableChart({
  title,
  children,
}: {
  title: string
  children: React.ReactNode
}): JSX.Element {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open) return undefined
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open])

  const expandedChart = open && typeof document !== 'undefined'
    ? createPortal(
        <div
          onClick={() => setOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 140,
            background: 'rgba(26,26,26,0.78)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'stretch',
            padding: 0,
          }}
        >
          <div onClick={(event) => event.stopPropagation()} style={{
            background: '#fafaf7',
            border: 'none',
            padding: '28px 48px 40px',
            width: '100vw',
            height: '100vh',
            overflowY: 'auto',
            boxShadow: '0 24px 60px rgba(0,0,0,0.4)',
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18, flexShrink: 0 }}>
              <div className="tracker" style={{ color: 'var(--accent)', fontSize: 12, letterSpacing: '0.34em' }}>{title}</div>
              <button type="button" onClick={() => setOpen(false)} style={{
                background: 'transparent',
                border: '1px solid #1a1a1a',
                padding: '8px 16px',
                fontFamily: "'IM Fell DW Pica SC', serif",
                fontSize: 11,
                letterSpacing: '0.28em',
                textTransform: 'uppercase',
                color: '#1a1a1a',
                cursor: 'pointer',
              }}>close x</button>
            </div>
            <div className="expanded-chart-content" style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              {children}
            </div>
          </div>
        </div>,
        document.body,
      )
    : null

  return (
    <div style={{ position: 'relative' }}>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title="Expand chart"
        aria-label="Expand chart"
        style={{
          position: 'absolute',
          top: 8,
          right: 8,
          zIndex: 5,
          width: 24,
          height: 24,
          background: '#fafaf7',
          border: '1px solid #1a1a1a',
          cursor: 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 0,
          color: '#1a1a1a',
        }}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
          <path d="M2 5V2H5" />
          <path d="M10 7V10H7" />
          <path d="M2 2L5 5" />
          <path d="M10 10L7 7" />
        </svg>
      </button>
      {children}
      {expandedChart}
    </div>
  )
}

function StickyNote({
  children,
  rotation = -1.4,
  pinColor = '#7a1d1d',
  background = '#fdf6c9',
  maxWidth = 360,
  style,
}: {
  children: React.ReactNode
  rotation?: number
  pinColor?: string
  background?: string
  maxWidth?: number | string
  style?: React.CSSProperties
}): JSX.Element {
  return (
    <div style={{
      position: 'relative',
      maxWidth,
      padding: '20px 22px 18px',
      background,
      backgroundImage: `radial-gradient(ellipse at 30% 18%, rgba(255,255,255,0.6), transparent 55%), radial-gradient(ellipse at 80% 90%, rgba(180,150,80,0.18), transparent 60%)`,
      transform: `rotate(${rotation}deg)`,
      boxShadow: '0 6px 14px rgba(26,26,26,0.16), 0 1px 0 rgba(26,26,26,0.06) inset',
      fontFamily: "'IM Fell English', serif",
      color: '#1a1a1a',
      ...style,
    }}>
      <span style={{
        position: 'absolute',
        top: -8,
        left: '50%',
        transform: 'translateX(-50%)',
        width: 14,
        height: 14,
        borderRadius: '50%',
        background: `radial-gradient(circle at 35% 30%, #d4444c 0%, ${pinColor} 70%, #5a1419 100%)`,
        boxShadow: '0 2px 4px rgba(0,0,0,0.4), inset 0 -1px 2px rgba(0,0,0,0.4)',
      }} />
      {children}
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
        // Smooth "shuffle" when the rerank reorders the visible articles —
        // each card slides to its new scatter spot rather than jumping.
        transition: 'left 600ms cubic-bezier(.2,.7,.2,1.05), top 600ms cubic-bezier(.2,.7,.2,1.05), transform 600ms cubic-bezier(.2,.7,.2,1.05)',
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
  hideArticleInfo = false,
}: {
  article: Article
  rank: number
  active: boolean
  onClick: () => void
  onDismiss: () => void
  hideArticleInfo?: boolean
}): JSX.Element {
  const [hover, setHover] = useState(false)
  const sentColor = getSentColor(article)
  const wordCount = article.word_count ?? null
  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={hideArticleInfo ? undefined : onClick}
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
        cursor: hideArticleInfo ? 'default' : 'pointer',
      }}
    >
      <div style={{
        fontFamily: "'IM Fell English', serif",
        fontSize: 26,
        lineHeight: 1,
        color: active ? '#fafaf7' : '#7a1d1d',
        textAlign: 'center',
      }}>{String(rank).padStart(2, '0')}</div>
      {hideArticleInfo ? (
        <div style={{
          minHeight: 36,
          border: '1px dashed rgba(26,26,26,0.22)',
          background: 'rgba(26,26,26,0.025)',
          opacity: 0.85,
        }} aria-hidden />
      ) : (
        <div>
          <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 16, lineHeight: 1.15, display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: sentColor, display: 'inline-block', flexShrink: 0 }} />
            <span>{article.title}</span>
          </div>
          <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, opacity: 0.75, marginTop: 3 }}>
            {getAuthorDisplay(article)} · {getOutlet(article)} · {formatDate(article.date)}{wordCount ? ` · ${wordCount.toLocaleString()} words` : ''}
          </div>
        </div>
      )}
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

/**
 * Word-wrap a label to N lines of approximately maxChars each. Returns the
 * full label split at word boundaries — never truncates with an ellipsis.
 * Used by the latent-dimension charts so long concept labels wrap onto
 * multiple lines instead of being clipped.
 */
function wrapLabelLines(label: string, maxChars: number): string[] {
  if (!label) return ['']
  const words = label.split(/\s+/).filter(Boolean)
  if (words.length === 0) return [label]
  const lines: string[] = []
  let current = ''
  for (const w of words) {
    if (!current) {
      current = w
    } else if (current.length + 1 + w.length <= maxChars) {
      current += ' ' + w
    } else {
      lines.push(current)
      current = w
    }
  }
  if (current) lines.push(current)
  return lines
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
    return <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', color: '#6a6a62', padding: 24, textAlign: 'center' }}>No SVD concepts available for this article.</div>
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
    <div className="svd-radar-panel" style={{ display: 'grid', gridTemplateColumns: '1fr 200px', gap: 16, alignItems: 'start', border: '1px solid #1a1a1a', background: '#fafaf7', padding: '12px 14px', overflow: 'hidden' }}>
      <svg className="svd-radar-svg" width="100%" viewBox={`-60 -10 ${SVD_RADAR_SIZE + 120} ${SVD_RADAR_SIZE + 60}`} style={{ display: 'block', overflow: 'visible' }}>
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
              {(() => {
                const lines = wrapLabelLines(getSvdLabel(d), 22)
                const valueY = lp.y + 4 + lines.length * 14
                return (
                  <>
                    <text x={lp.x} y={lp.y + 4} fontFamily="'IM Fell English', serif" fontStyle="italic" fontSize="12" fill="#1a1a1a" textAnchor={anchor}>
                      {lines.map((line, idx) => (
                        <tspan key={idx} x={lp.x} dy={idx === 0 ? 0 : '1.15em'}>{line}</tspan>
                      ))}
                    </text>
                    <text x={lp.x} y={valueY} fontFamily="'Special Elite', monospace" fontSize="9" fill="#6a6a62" textAnchor={anchor}>
                      a {Number(d.value) >= 0 ? '+' : ''}{Number(d.value).toFixed(2)}{hasQuery && (() => {
                        const q = queryByIndex.get(d.dimension_index)
                        if (!q) return ''
                        return ` · q ${Number(q.value) >= 0 ? '+' : ''}${Number(q.value).toFixed(2)}`
                      })()}
                    </text>
                  </>
                )
              })()}
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

function VintageSvdConceptBars({
  dimensions,
  comparisonDimensions,
}: {
  dimensions: SvdLatentDimension[]
  comparisonDimensions?: SvdLatentDimension[]
}): JSX.Element {
  const dims = dimensions.slice(0, 10)
  if (dims.length === 0) {
    return (
      <div style={{ border: '1px solid #1a1a1a', background: '#fafaf7', padding: '12px 14px', fontFamily: "'IM Fell English', serif", fontStyle: 'italic', color: '#6a6a62' }}>
        No article-specific SVD concepts available.
      </div>
    )
  }
  const comparisonByIndex = new Map<number, SvdLatentDimension>()
  for (const d of comparisonDimensions ?? []) {
    if (typeof d.dimension_index === 'number') comparisonByIndex.set(d.dimension_index, d)
  }
  const hasComparison = dims.some((d) => comparisonByIndex.has(d.dimension_index))
  // Find max magnitude across all dims (article + comparison) for consistent scale.
  let maxMag = 0
  for (const d of dims) {
    maxMag = Math.max(maxMag, Math.abs(Number(d.value)), Math.abs(Number(d.magnitude)))
    const c = comparisonByIndex.get(d.dimension_index)
    if (c) maxMag = Math.max(maxMag, Math.abs(Number(c.value)), Math.abs(Number(c.magnitude)))
  }
  if (maxMag <= 0) maxMag = 1

  const renderBar = (d: SvdLatentDimension, accent: 'article' | 'query'): JSX.Element => {
    const value = Number(d.value)
    const widthPct = Math.min(100, (Math.abs(value) / maxMag) * 100)
    const color = accent === 'query' ? '#7a1d1d' : '#1a1a1a'
    return (
      <div style={{ display: 'flex', height: 14, border: '1px solid #1a1a1a', position: 'relative', background: '#fafaf7' }}>
        <div style={{ flex: 1, position: 'relative', borderRight: '1px solid rgba(26,26,26,0.4)' }}>
          {value < 0 && (
            <span style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: `${widthPct}%`, background: color, opacity: accent === 'query' ? 0.5 : 0.85 }} />
          )}
        </div>
        <div style={{ flex: 1, position: 'relative' }}>
          {value >= 0 && (
            <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${widthPct}%`, background: color, opacity: accent === 'query' ? 0.5 : 0.85 }} />
          )}
        </div>
      </div>
    )
  }

  return (
    <div style={{ border: '1px solid #1a1a1a', background: '#fafaf7', padding: '12px 14px' }}>
      {hasComparison && (
        <div style={{ display: 'flex', gap: 14, marginBottom: 10, fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.22em', textTransform: 'uppercase', color: 'var(--ink-mute)' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 14, height: 8, background: '#1a1a1a' }} />article
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 14, height: 8, background: '#7a1d1d', opacity: 0.7 }} />your slip
          </span>
        </div>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.2em', textTransform: 'uppercase', color: 'var(--ink-mute)', marginBottom: 6 }}>
        <span>negative pole</span>
        <span style={{ borderLeft: '1px solid rgba(26,26,26,0.4)', height: 12 }} />
        <span>positive pole</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {dims.map((d) => {
          const c = comparisonByIndex.get(d.dimension_index)
          return (
            <div key={d.dimension_index} style={{ display: 'grid', gridTemplateColumns: '180px 1fr 110px', gap: 12, alignItems: 'center' }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 13, fontStyle: 'italic', lineHeight: 1.25, wordBreak: 'break-word' }}>
                  {getSvdLabel(d)}
                </div>
                {Array.isArray(d.label_terms) && d.label_terms.length > 0 && (
                  <div style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 8, letterSpacing: '0.18em', textTransform: 'uppercase', color: '#9a9a92', marginTop: 2, lineHeight: 1.4, wordBreak: 'break-word' }}>
                    {d.label_terms.slice(0, 3).join(' · ')}
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {renderBar(d, 'article')}
                {hasComparison && c && renderBar(c, 'query')}
                {hasComparison && !c && (
                  <div style={{ height: 14, border: '1px dashed rgba(26,26,26,0.3)', background: 'rgba(26,26,26,0.04)' }} />
                )}
              </div>
              <div style={{ fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#1a1a1a', textAlign: 'right' }}>
                <div>a {Number(d.value) >= 0 ? '+' : ''}{Number(d.value).toFixed(2)}</div>
                {hasComparison && (
                  <div style={{ color: c ? '#7a1d1d' : '#9a9a92', opacity: c ? 1 : 0.5 }}>
                    q {c ? `${Number(c.value) >= 0 ? '+' : ''}${Number(c.value).toFixed(2)}` : 'n/a'}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

type CitationRenderContext = {
  sourceMap: Map<number, ResultsOverviewSource>
  onOpenSource?: (resultIndex: number) => void
  seenCitations: Set<number>
}

function normalizeSourceIndices(values: readonly number[] | null | undefined): number[] {
  const normalized: number[] = []
  const seen = new Set<number>()
  for (const value of values ?? []) {
    const index = Number(value)
    if (!Number.isInteger(index) || index <= 0 || seen.has(index)) continue
    seen.add(index)
    normalized.push(index)
  }
  return normalized
}

function parseCitationNumbers(raw: string): number[] {
  return normalizeSourceIndices(raw.split(',').map(part => Number(part.trim())))
}

function buildSourceMap(sources: ResultsOverviewSource[] | null | undefined): Map<number, ResultsOverviewSource> {
  return new Map((sources ?? []).map(source => [source.result_index, source]))
}

function truncateReferenceTitle(title: string): string {
  return `${title.slice(0, 24)}${title.length > 24 ? '...' : ''}`
}

function CitationMark({
  index,
  source,
  onOpenSource,
  showTitle = false,
}: {
  index: number
  source?: ResultsOverviewSource
  onOpenSource?: (resultIndex: number) => void
  showTitle?: boolean
}): JSX.Element {
  const label = `[${index}]`
  const text = showTitle && source?.title ? `${label} ${truncateReferenceTitle(source.title)}` : label
  const title = source?.title ? `${label} ${source.title}` : onOpenSource ? `Open source ${label}` : `Source ${label}`
  const style = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    maxWidth: '100%',
    border: '1px solid currentColor',
    borderRadius: 0,
    background: 'transparent',
    color: 'inherit',
    padding: showTitle ? '2px 6px' : '1px 5px',
    margin: 0,
    fontFamily: "'IM Fell DW Pica SC', serif",
    fontSize: showTitle ? 8 : 9,
    letterSpacing: showTitle ? '0.18em' : '0.12em',
    lineHeight: 1.25,
    textTransform: 'uppercase',
    verticalAlign: 'baseline',
    textAlign: 'left' as const,
    cursor: onOpenSource ? 'pointer' : 'default',
  }

  if (onOpenSource) {
    return (
      <button
        type="button"
        title={title}
        aria-label={title}
        onClick={() => onOpenSource(index)}
        style={style}
      >
        {text}
      </button>
    )
  }
  return (
    <span title={title} style={style}>
      {text}
    </span>
  )
}

function SourceCitationList({
  indices,
  sourceMap,
  onOpenSource,
  showTitle = false,
}: {
  indices: readonly number[] | null | undefined
  sourceMap: Map<number, ResultsOverviewSource>
  onOpenSource?: (resultIndex: number) => void
  showTitle?: boolean
}): JSX.Element | null {
  const normalized = normalizeSourceIndices(indices)
  if (normalized.length === 0) return null
  return (
    <div style={{
      marginTop: 6,
      display: 'flex',
      flexWrap: 'wrap',
      gap: 5,
      color: 'inherit',
    }}>
      {normalized.map(index => (
        <CitationMark
          key={index}
          index={index}
          source={sourceMap.get(index)}
          onOpenSource={onOpenSource}
          showTitle={showTitle}
        />
      ))}
    </div>
  )
}

function renderTextWithCitations(text: string, context: CitationRenderContext, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const citationPattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = citationPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index))
    }

    const indices = parseCitationNumbers(match[1])
    if (indices.length === 0) {
      nodes.push(match[0])
    } else {
      indices.forEach(index => context.seenCitations.add(index))
      nodes.push(
        <span
          key={`${keyPrefix}-cite-${match.index}`}
          style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 4, margin: '0 2px', verticalAlign: 'baseline' }}
        >
          {indices.map(index => (
            <CitationMark
              key={index}
              index={index}
              source={context.sourceMap.get(index)}
              onOpenSource={context.onOpenSource}
            />
          ))}
        </span>,
      )
    }
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex))
  }
  return nodes
}

function isSafeMarkdownUrl(url: string): boolean {
  return url.startsWith('https://') || url.startsWith('http://') || url.startsWith('mailto:')
}

function renderInlineMarkdown(text: string, context: CitationRenderContext, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const tokenPattern = /(\[[^\]\n]+\]\((?:https?:\/\/|mailto:)[^)]+\)|`[^`\n]+`|\*\*[^*\n]+?\*\*|__[^_\n]+?__|\*[^*\n]+?\*|_[^_\n]+?_)/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = tokenPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(...renderTextWithCitations(text.slice(lastIndex, match.index), context, `${keyPrefix}-t${lastIndex}`))
    }

    const token = match[0]
    const key = `${keyPrefix}-md-${match.index}`
    const linkMatch = token.match(/^\[([^\]\n]+)\]\(([^)]+)\)$/)
    if (linkMatch) {
      const [, label, url] = linkMatch
      if (isSafeMarkdownUrl(url)) {
        nodes.push(
          <a key={key} href={url} target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'underline' }}>
            {renderInlineMarkdown(label, context, `${key}-label`)}
          </a>,
        )
      } else {
        nodes.push(...renderTextWithCitations(label, context, `${key}-unsafe-link`))
      }
    } else if (token.startsWith('`') && token.endsWith('`')) {
      nodes.push(
        <code key={key} style={{
          fontFamily: "'Special Elite', monospace",
          fontSize: '0.9em',
          border: '1px solid currentColor',
          padding: '0 3px',
        }}>
          {token.slice(1, -1)}
        </code>,
      )
    } else if ((token.startsWith('**') && token.endsWith('**')) || (token.startsWith('__') && token.endsWith('__'))) {
      nodes.push(
        <strong key={key}>
          {renderInlineMarkdown(token.slice(2, -2), context, `${key}-strong`)}
        </strong>,
      )
    } else if ((token.startsWith('*') && token.endsWith('*')) || (token.startsWith('_') && token.endsWith('_'))) {
      nodes.push(
        <em key={key}>
          {renderInlineMarkdown(token.slice(1, -1), context, `${key}-em`)}
        </em>,
      )
    } else {
      nodes.push(...renderTextWithCitations(token, context, key))
    }

    lastIndex = match.index + token.length
  }

  if (lastIndex < text.length) {
    nodes.push(...renderTextWithCitations(text.slice(lastIndex), context, `${keyPrefix}-t${lastIndex}`))
  }
  return nodes
}

function MarkdownText({
  text,
  sources = [],
  fallbackSourceIndices = [],
  onOpenSource,
}: {
  text: string
  sources?: ResultsOverviewSource[] | null
  fallbackSourceIndices?: readonly number[] | null
  onOpenSource?: (resultIndex: number) => void
}): JSX.Element {
  const context: CitationRenderContext = {
    sourceMap: buildSourceMap(sources),
    onOpenSource,
    seenCitations: new Set<number>(),
  }
  const blocks: ReactNode[] = []
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  let index = 0

  const readParagraph = (): string => {
    const paragraphLines: string[] = []
    while (index < lines.length) {
      const line = lines[index]
      if (line.trim() === '') break
      if (/^#{1,3}\s+/.test(line) || /^\s*(?:[-*]|\d+\.)\s+/.test(line)) break
      paragraphLines.push(line.trim())
      index += 1
    }
    return paragraphLines.join(' ')
  }

  while (index < lines.length) {
    const line = lines[index]
    if (line.trim() === '') {
      index += 1
      continue
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/)
    if (heading) {
      const level = heading[1].length
      blocks.push(
        <div
          key={`heading-${index}`}
          style={{
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: level === 1 ? '1.04em' : level === 2 ? '0.98em' : '0.92em',
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            lineHeight: 1.35,
          }}
        >
          {renderInlineMarkdown(heading[2], context, `heading-${index}`)}
        </div>,
      )
      index += 1
      continue
    }

    const unordered = line.match(/^\s*[-*]\s+(.+)$/)
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/)
    if (unordered || ordered) {
      const orderedList = Boolean(ordered)
      const items: string[] = []
      while (index < lines.length) {
        const nextLine = lines[index]
        const itemMatch = orderedList
          ? nextLine.match(/^\s*\d+\.\s+(.+)$/)
          : nextLine.match(/^\s*[-*]\s+(.+)$/)
        if (!itemMatch) break
        items.push(itemMatch[1].trim())
        index += 1
      }
      const ListTag = orderedList ? 'ol' : 'ul'
      blocks.push(
        <ListTag key={`list-${index}`} style={{ margin: 0, paddingLeft: 18, display: 'grid', gap: 3 }}>
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>
              {renderInlineMarkdown(item, context, `list-${index}-${itemIndex}`)}
            </li>
          ))}
        </ListTag>,
      )
      continue
    }

    const paragraph = readParagraph()
    if (paragraph) {
      blocks.push(
        <p key={`paragraph-${index}`} style={{ margin: 0 }}>
          {renderInlineMarkdown(paragraph, context, `paragraph-${index}`)}
        </p>,
      )
    } else {
      index += 1
    }
  }

  const unseenFallbackIndices = normalizeSourceIndices(fallbackSourceIndices)
    .filter(sourceIndex => !context.seenCitations.has(sourceIndex))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {blocks}
      <SourceCitationList
        indices={unseenFallbackIndices}
        sourceMap={context.sourceMap}
        onOpenSource={onOpenSource}
        showTitle
      />
    </div>
  )
}

function ChatBubble({
  message,
  onOpenSource,
}: {
  message: ResultsChatMessage
  onOpenSource?: (resultIndex: number) => void
}): JSX.Element {
  const isAssistant = message.role === 'assistant'
  const fallbackSourceIndices = message.source_indices ?? message.sources?.map(source => source.result_index) ?? []
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
        {isAssistant ? 'ai assistant editor' : 'you'}
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
        whiteSpace: 'normal',
      }}>
        {message.attachments && message.attachments.length > 0 && (
          <div style={{
            marginBottom: 8,
            paddingBottom: 8,
            borderBottom: '1px dotted rgba(255,255,255,0.4)',
            display: 'flex',
            flexWrap: 'wrap',
            gap: 6,
          }}>
            {message.attachments.map(att => (
              <span key={att.articleId} style={{
                fontFamily: "'IM Fell DW Pica SC', serif",
                fontSize: 8,
                letterSpacing: '0.22em',
                textTransform: 'uppercase',
                border: '1px solid currentColor',
                padding: '2px 6px',
                opacity: 0.85,
              }}>
                ↳ [{att.resultIndex}] {att.title.slice(0, 28)}{att.title.length > 28 ? '…' : ''}
              </span>
            ))}
          </div>
        )}
        {message.content !== '' && (
          <MarkdownText
            text={message.content}
            sources={message.sources ?? []}
            fallbackSourceIndices={fallbackSourceIndices}
            onOpenSource={onOpenSource}
          />
        )}
        {/* Blinking caret while the assistant streams. The placeholder
            assistant message starts with an empty string and grows as SSE
            chunks arrive. */}
        {isAssistant && message.content === '' && (
          <span aria-hidden style={{
            display: 'inline-block',
            width: '0.55ch',
            height: '0.95em',
            verticalAlign: '-0.1em',
            background: '#1a1a1a',
            animation: 'caret-blink 1s steps(1) infinite',
          }} />
        )}
      </div>
    </div>
  )
}

function BroadsheetOverlay({
  article,
  rank,
  total,
  querySvdDimensions,
  querySvdCorpusChartDimensions,
  onClose,
  onDismiss,
  onChatThis,
  onFindSimilar,
  onExplainRanking,
  explainState,
}: {
  article: Article
  rank: number
  total: number
  querySvdDimensions: SvdLatentDimension[]
  querySvdCorpusChartDimensions: SvdLatentDimension[]
  onClose: () => void
  onDismiss: () => void
  onChatThis: () => void
  onFindSimilar: () => void
  onExplainRanking?: () => void
  explainState?: { loading: boolean; explanation: string | null; error: string | null }
}): JSX.Element {
  const stance = getStanceCategory(article)
  const articleQuerySvd = (article.svd_query_chart_dimensions ?? []) as SvdLatentDimension[]
  const articleCorpusSvd = (article.svd_chart_dimensions ?? []) as SvdLatentDimension[]
  const articleOwnSvd = (article.svd_dimensions ?? []) as SvdLatentDimension[]
  const articleQueryOwnSvd = (article.svd_article_query_dimensions ?? []) as SvdLatentDimension[]
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
        width: 'min(960px, 92vw)',
        height: '100%',
        background: '#fafaf7',
        borderLeft: '1px solid #1a1a1a',
        boxShadow: '-30px 0 60px rgba(26,26,26,0.18)',
        padding: '28px 48px 60px',
        animation: 'rf-slide 360ms cubic-bezier(.2,.7,.2,1.05) both',
        fontFamily: "'Old Standard TT', serif",
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 10, letterSpacing: '0.28em', textTransform: 'uppercase', color: '#6a6a62' }}>
          <span>broadsheet · the article in full</span>
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
          <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 30, lineHeight: 1.08 }}>{article.title}</div>
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

        {/* Most important quotes — the article's pre-extracted claim/support
            sentences from the bootstrap claim database. These are inherent to
            the article, independent of the slip. The blockquote rule already
            telegraphs that this is a quotation, so no surrounding "…". */}
        {(article.support_sentences && article.support_sentences.length > 0) && (
          <div style={{ marginTop: 20 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 8 }}>most important quotes</div>
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
                }}>{quote}</blockquote>
              )
            })}
          </div>
        )}

        {/* Passages most relevant to your slip — only when the search ran in
            chunking mode. Two backend lists can overlap here:
              - topic_relevant_chunks: scored by the retrieval model
              - llm_relevant_paragraphs: re-scored by the LLM sub-editor; has
                both topic_score and agreement_score on each chunk
            We dedupe by paragraph_id / chunk_id so the same passage isn't
            listed twice, and surface BOTH scores when available. */}
        {(() => {
          const chunks = article.topic_relevant_chunks ?? []
          const llmParas = article.llm_relevant_paragraphs ?? []
          if (chunks.length === 0 && llmParas.length === 0) return null
          type Passage = { id: string; text: string; topicScore: number | null; agreementScore: number | null; rank: number | null }
          const merged = new Map<string, Passage>()
          for (const c of chunks) {
            const id = String(c.chunk_id ?? c.chunk_index ?? c.text.slice(0, 64))
            merged.set(id, {
              id,
              text: c.text,
              topicScore: typeof c.topic_score === 'number' ? c.topic_score : null,
              agreementScore: null,
              rank: typeof c.chunk_rank === 'number' ? c.chunk_rank : null,
            })
          }
          for (const p of llmParas) {
            const id = String(p.paragraph_id ?? p.paragraph_index ?? p.text.slice(0, 64))
            const existing = merged.get(id)
            const topicScore = typeof p.topic_score === 'number' ? p.topic_score : existing?.topicScore ?? null
            const agreementScore = typeof p.agreement_score === 'number' ? p.agreement_score : existing?.agreementScore ?? null
            const rank = typeof p.topic_chunk_rank === 'number' ? p.topic_chunk_rank : existing?.rank ?? null
            merged.set(id, { id, text: p.text, topicScore, agreementScore, rank })
          }
          const passages = Array.from(merged.values()).slice(0, 4)
          return (
            <div style={{ marginTop: 20 }}>
              <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 8 }}>passages most relevant to your slip</div>
              {passages.map((p, i) => {
                const tPct = clampPct(p.topicScore)
                const aPct = clampPct(p.agreementScore)
                const rankLabel = p.rank ?? (i + 1)
                return (
                  <blockquote key={p.id} style={{
                    margin: '0 0 12px',
                    padding: '8px 14px 10px 18px',
                    borderLeft: '2px solid #1a1a1a',
                    background: 'rgba(26,26,26,0.02)',
                    fontFamily: "'IM Fell English', serif",
                    fontStyle: 'italic',
                    fontSize: 15,
                    lineHeight: 1.5,
                    color: '#1a1a1a',
                  }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'baseline',
                      fontFamily: "'IM Fell DW Pica SC', serif",
                      fontSize: 9,
                      letterSpacing: '0.24em',
                      textTransform: 'uppercase',
                      color: '#6a6a62',
                      marginBottom: 6,
                      fontStyle: 'normal',
                    }}>
                      <span>chunk {rankLabel}</span>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 14, letterSpacing: 0, textTransform: 'none', fontFamily: "'Special Elite', monospace", fontSize: 11 }}>
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ width: 50, height: 3, background: 'rgba(26,26,26,0.12)', position: 'relative' }}>
                            <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${tPct}%`, background: '#1a1a1a' }} />
                          </span>
                          <span style={{ color: '#1a1a1a' }}>τ {tPct}</span>
                        </span>
                        {p.agreementScore !== null && (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ width: 50, height: 3, background: 'rgba(26,26,26,0.12)', position: 'relative' }}>
                              <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${aPct}%`, background: '#7a1d1d' }} />
                            </span>
                            <span style={{ color: '#7a1d1d' }}>A {aPct}</span>
                          </span>
                        )}
                      </span>
                    </div>
                    {p.text}
                  </blockquote>
                )
              })}
            </div>
          )
        })()}

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

        {articleQuerySvd.length > 0 && (
          <div style={{ marginTop: 28 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 6 }}>the latent dimensions · query-anchored compass</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#3a3a36', marginBottom: 8, lineHeight: 1.5 }}>
              Each spoke is one of the concepts your slip activated most strongly. The dark hull is this article's footprint on those same concepts; the oxblood hull is your slip's. Where the hulls part company, the author is travelling somewhere you did not ask to go.
            </div>
            <ExpandableChart title="query-anchored compass">
              <VintageSvdRadar articleDims={articleQuerySvd} queryDims={querySvdDimensions} />
            </ExpandableChart>
          </div>
        )}

        {articleCorpusSvd.length > 0 && (
          <div style={{ marginTop: 28 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 6 }}>the latent dimensions · the archive's broad concepts</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#3a3a36', marginBottom: 8, lineHeight: 1.5 }}>
              The same article, plotted against the archive's ten broadest concepts. Useful for seeing how its focus differs from the rest of the running order.
            </div>
            <ExpandableChart title="the archive's broad concepts">
              <VintageSvdRadar articleDims={articleCorpusSvd} queryDims={querySvdCorpusChartDimensions} />
            </ExpandableChart>
          </div>
        )}

        {articleOwnSvd.length > 0 && (
          <div style={{ marginTop: 28 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 6 }}>top concepts for this article</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#3a3a36', marginBottom: 8, lineHeight: 1.5 }}>
              Longer bars mean the article is more strongly associated with that concept. Direction (left = negative pole, right = positive pole) marks which side of the dimension the article lands on — opposite poles are different but related themes.
            </div>
            <ExpandableChart title="top concepts for this article">
              <VintageSvdConceptBars
                dimensions={articleOwnSvd}
                comparisonDimensions={articleQueryOwnSvd}
              />
            </ExpandableChart>
          </div>
        )}

        {/* Sentiment details */}
        {sentiment && (
          <div style={{ marginTop: 28 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 6 }}>sentiment · vader</div>
            <div style={{ border: '1px solid #1a1a1a', background: '#fafaf7', padding: '12px 14px' }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 18, fontFamily: "'Special Elite', monospace", fontSize: 12, color: '#1a1a1a' }}>
                <span><strong>compound</strong> {sentiment.compound >= 0 ? '+' : ''}{sentiment.compound.toFixed(2)}</span>
                <span><strong>label</strong> {sentiment.label}</span>
                {sentiment.tone_strength && <span><strong>tone</strong> {sentiment.tone_strength}</span>}
              </div>
              {/* Negative · neutral · positive ratio bar — three segments
                  whose widths are the VADER neg/neu/pos proportions. Mirrors
                  what main shipped under .sentiment-meter / .sentiment-breakdown. */}
              {(typeof sentiment.negative === 'number' || typeof sentiment.neutral === 'number' || typeof sentiment.positive === 'number') && (() => {
                const neg = Math.max(0, Number(sentiment.negative) || 0)
                const neu = Math.max(0, Number(sentiment.neutral) || 0)
                const pos = Math.max(0, Number(sentiment.positive) || 0)
                const total = neg + neu + pos || 1
                const negPct = (neg / total) * 100
                const neuPct = (neu / total) * 100
                const posPct = (pos / total) * 100
                const fmt = (v: number): string => `${(v * 100).toFixed(0)}%`
                return (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ display: 'flex', height: 10, border: '1px solid #1a1a1a', overflow: 'hidden' }}>
                      <div style={{ width: `${negPct}%`, background: '#7a1d1d' }} title={`negative ${fmt(neg)}`} />
                      <div style={{ width: `${neuPct}%`, background: '#cdcabd' }} title={`neutral ${fmt(neu)}`} />
                      <div style={{ width: `${posPct}%`, background: '#1a1a1a' }} title={`positive ${fmt(pos)}`} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.22em', textTransform: 'uppercase', color: '#6a6a62' }}>
                      <span style={{ color: '#7a1d1d' }}>negative {fmt(neg)}</span>
                      <span>neutral {fmt(neu)}</span>
                      <span style={{ color: '#1a1a1a' }}>positive {fmt(pos)}</span>
                    </div>
                  </div>
                )
              })()}
              {sentiment.text_scores && (
                <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10 }}>
                  {(['title', 'summary', 'article'] as const).map(k => {
                    const s = sentiment.text_scores?.[k]
                    if (!s || typeof s.compound !== 'number') return null
                    return (
                      <div key={k} style={{ borderTop: '1px solid rgba(26,26,26,0.18)', paddingTop: 6, fontFamily: "'IM Fell English', serif", fontSize: 12 }}>
                        <div className="tracker" style={{ marginBottom: 2 }}>{k}</div>
                        <div style={{ fontFamily: "'Special Elite', monospace" }}>{s.compound >= 0 ? '+' : ''}{s.compound.toFixed(2)} · {s.label}</div>
                      </div>
                    )
                  })}
                </div>
              )}
              {sentiment.snippets && (sentiment.snippets.positive?.length || sentiment.snippets.negative?.length) && (
                <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                  <div>
                    <div className="tracker" style={{ color: '#1a1a1a', marginBottom: 6 }}>most positive lines</div>
                    {(sentiment.snippets.positive ?? []).slice(0, 3).map((sn, i) => (
                      <blockquote key={i} style={{
                        margin: '0 0 8px',
                        padding: '4px 0 4px 12px',
                        borderLeft: '2px solid #1a1a1a',
                        fontFamily: "'IM Fell English', serif",
                        fontStyle: 'italic',
                        fontSize: 13,
                        color: '#1a1a1a',
                        lineHeight: 1.5,
                      }}>{sn.text} <span style={{ fontFamily: "'Special Elite', monospace", fontStyle: 'normal', fontSize: 10, color: '#6a6a62', whiteSpace: 'nowrap' }}>{sn.compound >= 0 ? '+' : ''}{sn.compound.toFixed(2)}</span></blockquote>
                    ))}
                    {(sentiment.snippets.positive ?? []).length === 0 && (
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>none</div>
                    )}
                  </div>
                  <div>
                    <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 6 }}>most negative lines</div>
                    {(sentiment.snippets.negative ?? []).slice(0, 3).map((sn, i) => (
                      <blockquote key={i} style={{
                        margin: '0 0 8px',
                        padding: '4px 0 4px 12px',
                        borderLeft: '2px solid #7a1d1d',
                        fontFamily: "'IM Fell English', serif",
                        fontStyle: 'italic',
                        fontSize: 13,
                        color: '#1a1a1a',
                        lineHeight: 1.5,
                      }}>{sn.text} <span style={{ fontFamily: "'Special Elite', monospace", fontStyle: 'normal', fontSize: 10, color: '#6a6a62', whiteSpace: 'nowrap' }}>{sn.compound >= 0 ? '+' : ''}{sn.compound.toFixed(2)}</span></blockquote>
                    ))}
                    {(sentiment.snippets.negative ?? []).length === 0 && (
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>none</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Explain ranking */}
        {onExplainRanking && (
          <div style={{ marginTop: 24 }}>
            <div className="tracker" style={{ color: 'var(--accent)', marginBottom: 6 }}>why is this ranked here?</div>
            {explainState?.loading || explainState?.explanation ? (
              <div style={{
                border: '1px solid #1a1a1a',
                padding: '12px 14px',
                background: '#fafaf7',
                fontFamily: "'IM Fell English', serif",
                fontStyle: 'italic',
                fontSize: 14,
                lineHeight: 1.5,
                color: '#1a1a1a',
                whiteSpace: 'pre-wrap',
              }}>
                {explainState.explanation || (explainState.loading && !explainState.explanation
                  ? 'the AI Assistant Editor is reviewing the marks…'
                  : '')}
                {/* Caret blinks while content streams in. */}
                {explainState.loading && (
                  <span aria-hidden style={{
                    display: 'inline-block',
                    width: '0.55ch',
                    height: '0.95em',
                    marginLeft: 2,
                    verticalAlign: '-0.1em',
                    background: '#1a1a1a',
                    animation: 'caret-blink 1s steps(1) infinite',
                  }} />
                )}
              </div>
            ) : (
              <button type="button" onClick={onExplainRanking} style={{
                background: 'transparent',
                color: '#1a1a1a',
                border: '1px solid #1a1a1a',
                padding: '8px 16px',
                fontFamily: "'IM Fell DW Pica SC', serif",
                fontSize: 10,
                letterSpacing: '0.28em',
                textTransform: 'uppercase',
                cursor: 'pointer',
              }}>↳ ask the AI Assistant Editor to explain this ranking</button>
            )}
            {explainState?.error && (
              <div style={{ marginTop: 8, fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#7a1d1d' }}>{explainState.error}</div>
            )}
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
          }}>⌕ find similar articles</button>
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
            <div className="tracker" style={{ color: 'var(--accent)' }}>from the archive · neighbours of this article</div>
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
            articles that share latitude with <em style={{ fontStyle: 'italic' }}>"{source.title}"</em>
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
            }}>No close neighbours surfaced from the archive yet.</div>
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
                  <span className="tracker" style={{ borderBottom: '1px solid #1a1a1a', paddingBottom: 2, color: '#1a1a1a' }}>read the article →</span>
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
            {similar.length} {similar.length === 1 ? 'article' : 'articles'} · sorted by cosine similarity
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
  attachableArticles,
  attachedIds,
  onToggleAttachment,
  onOpenSource,
}: {
  total: number
  messages: ResultsChatMessage[]
  input: string
  loading: boolean
  error: string | null
  onClose: () => void
  onInputChange: (next: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  attachableArticles: Array<{ id: string; rank: number; title: string }>
  attachedIds: string[]
  onToggleAttachment: (id: string) => void
  onOpenSource?: (resultIndex: number) => void
}): JSX.Element {
  const [pickerOpen, setPickerOpen] = useState(false)
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
            <div className="tracker" style={{ color: 'var(--accent)' }}>ask the editor</div>
            {/* Chat is an AI-assistant feature, separate from the persona
                stance-scorer. Plain label, no portrait. */}
            <div style={{ fontFamily: "'IM Fell English', serif", fontSize: 24, marginTop: 2 }}>AI Assistant Editor</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>has read all {total} articles · cites by [number]</div>
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
              Good afternoon. I have your articles in front of me. Ask whatever you like — about the running order, the dissent, or any single article.
            </div>
          )}
          {messages.map((message) => (
            <ChatBubble key={message.id} message={message} onOpenSource={onOpenSource} />
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

        {/* Attachment chips + picker */}
        {(attachedIds.length > 0 || pickerOpen) && (
          <div style={{ padding: '10px 24px', borderTop: '1px solid rgba(26,26,26,0.18)' }}>
            {attachedIds.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: pickerOpen ? 10 : 0 }}>
                {attachedIds.map(id => {
                  const a = attachableArticles.find(x => x.id === id)
                  if (!a) return null
                  return (
                    <span key={id} style={{
                      display: 'inline-flex', alignItems: 'center', gap: 6,
                      border: '1px solid #1a1a1a', padding: '3px 8px',
                      fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.18em', textTransform: 'uppercase',
                      background: '#1a1a1a', color: '#fafaf7',
                    }}>
                      [{String(a.rank).padStart(2, '0')}] {a.title.slice(0, 28)}{a.title.length > 28 ? '…' : ''}
                      <button type="button" onClick={() => onToggleAttachment(id)} style={{ background: 'transparent', border: 0, color: '#fafaf7', cursor: 'pointer', padding: 0, fontSize: 11 }}>×</button>
                    </span>
                  )
                })}
              </div>
            )}
            {pickerOpen && (
              <div style={{ maxHeight: 180, overflowY: 'auto', border: '1px solid rgba(26,26,26,0.18)', padding: '6px' }}>
                {attachableArticles.map(a => {
                  const isAttached = attachedIds.includes(a.id)
                  return (
                    <button
                      key={a.id}
                      type="button"
                      onClick={() => onToggleAttachment(a.id)}
                      style={{
                        display: 'flex', width: '100%', alignItems: 'baseline', gap: 8, padding: '4px 6px',
                        background: isAttached ? '#1a1a1a' : 'transparent',
                        color: isAttached ? '#fafaf7' : '#1a1a1a',
                        border: 0, cursor: 'pointer', textAlign: 'left',
                        fontFamily: "'IM Fell English', serif", fontSize: 13,
                      }}
                    >
                      <span style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.18em', minWidth: 24 }}>{String(a.rank).padStart(2, '0')}</span>
                      <span style={{ flex: 1, lineHeight: 1.3 }}>{a.title}</span>
                      <span style={{ fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.18em', textTransform: 'uppercase' }}>{isAttached ? 'attached' : '+ attach'}</span>
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        )}

        <form onSubmit={onSubmit} style={{ padding: '14px 24px', borderTop: '1px solid #1a1a1a', display: 'flex', gap: 10, alignItems: 'center' }}>
          <button
            type="button"
            onClick={() => setPickerOpen(p => !p)}
            title="Attach an article to your question"
            style={{
              background: pickerOpen ? '#1a1a1a' : 'transparent',
              color: pickerOpen ? '#fafaf7' : '#1a1a1a',
              border: '1px solid #1a1a1a',
              padding: '8px 10px',
              cursor: 'pointer',
              fontFamily: "'IM Fell DW Pica SC', serif",
              fontSize: 9,
              letterSpacing: '0.24em',
              textTransform: 'uppercase',
            }}
          >+ ref</button>
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
            <div className="tracker" style={{ color: 'var(--accent)' }}>asking about a single article</div>
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
              I've read this article in full. What do you want to know?
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

function OverviewItem({
  item,
  sources,
  onOpenSource,
}: {
  item: ResultsOverviewArgument
  sources: ResultsOverviewSource[]
  onOpenSource: (resultIndex: number) => void
}): JSX.Element {
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
          <MarkdownText
            text={item.argument}
            sources={sources}
            fallbackSourceIndices={item.source_indices ?? []}
            onOpenSource={onOpenSource}
          />
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
    llmLabelIrrelevant,
    dismissedIds,
    onDismiss,
    onApplyDismissals,
    onBackToCompose,
    onOpenAbout,
    onOpenMethod,
    typoCorrection,
    onApplyTypoCorrection,
    onSearchAnyway,
    rewriteAlternatives,
    rewriteLoading,
    rewriteError,
    onLoadRewrites,
    onApplyRewrite,
    essayCandidates,
    selectedThesisId,
    onSelectThesisCandidate,
    thesisMode,
    onThesisModeChange,
    customThesis,
    onCustomThesisChange,
    onConfirmEssayThesis,
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
    chatAttachedIds,
    onToggleChatAttachment,
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
    querySvdCorpusChartDimensions,
    effectiveRetrievalModel,
    stage,
    rankingExplanations,
    onExplainRanking,
  } = props

  const [openId, setOpenId] = useState<string | null>(null)
  const [chatOpen, setChatOpen] = useState(false)
  const [sortPhase, setSortPhase] = useState<'idle' | 'sorting' | 'done'>('idle')
  const prevStageRef = useRef<'stage1' | 'stage2' | 'stage3'>(stage)
  useEffect(() => {
    if (stage === 'stage3' && prevStageRef.current === 'stage2' && articles.length > 0) {
      setSortPhase('sorting')
      const timer = setTimeout(() => setSortPhase('done'), 1500)
      prevStageRef.current = stage
      return () => clearTimeout(timer)
    }
    if (stage !== 'stage3') setSortPhase('idle')
    prevStageRef.current = stage
    return undefined
  }, [stage, articles.length])

  const visibleArticles = useMemo(
    () => articles.filter(a => !dismissedIds.has(getArticleId(a)) && !(llmLabelIrrelevant && isLlmIrrelevantArticle(a))),
    [articles, dismissedIds, llmLabelIrrelevant],
  )

  const llmIrrelevantArticles = useMemo(
    () => llmLabelIrrelevant
      ? articles.filter(a => isLlmIrrelevantArticle(a) && !dismissedIds.has(getArticleId(a)))
      : [],
    [articles, dismissedIds, llmLabelIrrelevant],
  )

  /**
   * Lock the 8 cards on the desk in stage 2.
   *
   * The bug: topic_results SSE arrives → top-8 by topic-relevance scatters.
   * When the rerank JSON arrives, the new top-8 by combined-score may not
   * be the *same* 8 IDs — articles that were rank 5–8 by topic can drop to
   * rank 9+ after stance scoring, replaced by articles that climbed up.
   * Slicing `visibleArticles.slice(0, 8)` then changes membership, React
   * unmounts the dropped cards and remounts the new ones, and the user
   * sees the desk briefly empty out / repopulate.
   *
   * Fix: when the desk first becomes non-empty, freeze its 8 IDs. On every
   * subsequent render we keep the *same* IDs but re-order them by their
   * position in the current `visibleArticles` — so the cards reshuffle
   * smoothly via the CSS left/top transition instead of remounting.
   */
  const scatteredIdsRef = useRef<string[] | null>(null)
  // Reset the lock whenever stage 2 ends (search restart, dismiss-and-refetch).
  useEffect(() => {
    if (stage !== 'stage2') {
      scatteredIdsRef.current = null
    }
  }, [stage])

  const scatterArticles = useMemo<Article[]>(() => {
    if (visibleArticles.length === 0) return []
    if (scatteredIdsRef.current === null) {
      // First time the desk has cards — freeze the top-8 by current order.
      const seed = visibleArticles.slice(0, 8)
      scatteredIdsRef.current = seed.map(a => getArticleId(a))
      return seed
    }
    // Subsequent renders — keep the same IDs, but reorder them to match
    // the new ranking from `visibleArticles`. Drop any frozen IDs that
    // dropped out of the visible set entirely (e.g. user dismissed it).
    const lockedIds = scatteredIdsRef.current
    const rankIndex = new Map<string, number>()
    visibleArticles.forEach((a, i) => rankIndex.set(getArticleId(a), i))
    const byId = new Map(visibleArticles.map(a => [getArticleId(a), a]))
    const stillVisible = lockedIds.filter(id => byId.has(id))
    stillVisible.sort((a, b) => (rankIndex.get(a) ?? 1e9) - (rankIndex.get(b) ?? 1e9))
    // If a frozen card was dismissed, refill the slot from the next-best
    // visible article (by current ranking) so the desk stays at 8.
    const stillVisibleSet = new Set(stillVisible)
    if (stillVisible.length < 8) {
      for (const a of visibleArticles) {
        if (stillVisible.length >= 8) break
        const id = getArticleId(a)
        if (stillVisibleSet.has(id)) continue
        stillVisible.push(id)
        stillVisibleSet.add(id)
      }
      scatteredIdsRef.current = [...stillVisible]
    } else if (stillVisible.length !== lockedIds.length) {
      scatteredIdsRef.current = [...stillVisible]
    }
    return stillVisible.map(id => byId.get(id)!).filter(Boolean) as Article[]
  }, [visibleArticles])

  const supporting = useMemo(
    () => visibleArticles.filter(a => getStanceCategory(a) === 'supports').slice(0, 3),
    [visibleArticles],
  )
  const opposing = useMemo(
    () => visibleArticles.filter(a => getStanceCategory(a) === 'complicates').slice(0, 3),
    [visibleArticles],
  )

  const stageNumber: 1 | 2 | 3 = stage === 'stage1' ? 1 : stage === 'stage2' ? 2 : 3
  const isStage1 = stage === 'stage1'
  const isStage2 = stage === 'stage2'
  const isStage3 = stage === 'stage3'
  const hasTypo = typoCorrection !== null
  const stageLabels = [
    { n: 1, title: 'Proofreading your slip', caption: 'spelling & rewrite check' },
    { n: 2, title: 'Topic Relevance', caption: 'articles scatter on the desk' },
    { n: 3, title: 'Stance Agreement', caption: 'ranked ledger · structured overview · the editor' },
  ] as const

  const overviewSources = overview?.sources ?? []
  const articleChatTarget = articleChatArticleId
    ? articles.find(a => getArticleId(a) === articleChatArticleId) ?? null
    : null

  const openArticle = articles.find(a => getArticleId(a) === openId) ?? null
  const openResultSource = (resultIndex: number): void => {
    const article = visibleArticles[resultIndex - 1]
    if (article) setOpenId(getArticleId(article))
  }

  return (
    <div className="stage-shell" style={{ position: 'relative' }}>
      {/* Same nav bar as the landing page (search active, about, method).
          The "back to compose" affordance lives at the bottom of the page
          alongside the about/method pages — and the pen icon next to the
          pinned slip already lets the user jump back to edit the query. */}
      <div className="top-rail">
        <button type="button" className="top-rail-brand" onClick={onBackToCompose}>hear! hear!</button>
        <div className="top-rail-links">
          <button type="button" className="active">search</button>
          <button type="button" onClick={onOpenAbout}>about</button>
          <button type="button" onClick={onOpenMethod}>method</button>
        </div>
      </div>
      <div className="top-rule" />

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button type="button" onClick={onBackToCompose}>edit slip →</button>
        </div>
      )}

      {/* pinned slip — with a pen-icon edit button to jump back to the search page */}
      <div style={{ padding: '18px 48px 0', position: 'relative' }}>
        <PinnedSlip
          topic={topic}
          opinion={opinion}
          mode={inputMode}
          essayText={essayText}
          thesisSentence={thesisSentence}
          animateOnMount
        />
        <button
          type="button"
          onClick={onBackToCompose}
          title="Edit slip"
          aria-label="Edit slip"
          style={{
            position: 'absolute',
            right: 'calc((100% - 720px) / 2 - 36px)',
            top: 24,
            background: '#fafaf7',
            border: '1px solid #1a1a1a',
            cursor: 'pointer',
            width: 28,
            height: 28,
            padding: 0,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#1a1a1a',
            zIndex: 4,
          }}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 1.5L14.5 5L5 14.5H1.5V11L11 1.5Z" />
            <path d="M9.5 3L13 6.5" />
          </svg>
        </button>
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
          const active = stageNumber === s.n
          const passed = stageNumber > s.n
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

      {/* body — flex-1 so it fills available height. Stage 1 needs outer scroll
          (typo/rewrite or thesis-picker can be tall); stages 2 & 3 own their scroll
          inside the panels (ledger, overview), so the outer pane stays fixed. */}
      <div className="tray-scroll" style={{ position: 'relative', padding: '24px 48px 0', flex: 1, minHeight: 0, overflowY: isStage1 ? 'auto' : 'hidden' }}>
        {/* Stage 1: typo + rewrite (stance) OR thesis picker (essay/NLI) */}
        {isStage1 && inputMode === 'essay' && (
          <div>
            <div className="tracker" style={{ color: 'var(--accent)', fontSize: 10, letterSpacing: '0.32em' }}>the editor's lectern · pick the thesis</div>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.55, color: '#3a3a36', marginTop: 8, marginBottom: 14, maxWidth: 720 }}>
              The NLI sub-editor (<PersonaName persona="nli" />) marks one sentence at a time, so we need a single thesis sentence to score against each article. Pick the candidate the editor pulled from your essay, or write your own.
            </div>

            <div style={{ display: 'flex', gap: 0, marginBottom: 14, fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 10, letterSpacing: '0.28em', textTransform: 'uppercase' }}>
              <button type="button" onClick={() => onThesisModeChange('candidate')} disabled={essayCandidates.length === 0} style={{
                padding: '8px 18px',
                border: '1px solid #1a1a1a',
                background: thesisMode === 'candidate' ? '#1a1a1a' : 'transparent',
                color: thesisMode === 'candidate' ? '#fafaf7' : '#1a1a1a',
                fontFamily: 'inherit', fontSize: 'inherit', letterSpacing: 'inherit', textTransform: 'inherit',
                cursor: essayCandidates.length === 0 ? 'not-allowed' : 'pointer',
                opacity: essayCandidates.length === 0 ? 0.5 : 1,
              }}>extracted candidates</button>
              <button type="button" onClick={() => onThesisModeChange('custom')} style={{
                padding: '8px 18px',
                border: '1px solid #1a1a1a',
                borderLeft: 0,
                background: thesisMode === 'custom' ? '#1a1a1a' : 'transparent',
                color: thesisMode === 'custom' ? '#fafaf7' : '#1a1a1a',
                fontFamily: 'inherit', fontSize: 'inherit', letterSpacing: 'inherit', textTransform: 'inherit',
                cursor: 'pointer',
              }}>write a custom thesis</button>
            </div>

            {thesisMode === 'candidate' && (
              <>
                {loading && essayCandidates.length === 0 && (
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10, fontFamily: "'Special Elite', monospace", fontSize: 12, color: '#6a6a62' }}>
                    <RFSpinner /><span>· · · the editor is reading your essay</span>
                  </div>
                )}
                {essayCandidates.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 0, border: '1px solid #1a1a1a', maxWidth: 920 }}>
                    {essayCandidates.map((c, i) => {
                      const active = selectedThesisId === c.sentence_id
                      return (
                        <button
                          key={c.sentence_id}
                          type="button"
                          onClick={() => onSelectThesisCandidate(c.sentence_id)}
                          style={{
                            padding: '12px 16px',
                            textAlign: 'left',
                            cursor: 'pointer',
                            background: active ? '#1a1a1a' : 'transparent',
                            color: active ? '#fafaf7' : '#1a1a1a',
                            border: 0,
                            borderTop: i === 0 ? 0 : '1px solid #1a1a1a',
                            display: 'flex',
                            gap: 14,
                            alignItems: 'baseline',
                            fontFamily: "'Old Standard TT', serif",
                          }}
                        >
                          <span style={{
                            fontFamily: "'IM Fell English', serif",
                            fontStyle: 'italic',
                            fontSize: 18,
                            color: active ? '#fafaf7' : '#7a1d1d',
                            minWidth: 22,
                          }}>{i + 1}.</span>
                          <span style={{ fontFamily: "'IM Fell English', serif", fontSize: 16, lineHeight: 1.45, flex: 1 }}>
                            {c.sentence}
                          </span>
                          {typeof c.score === 'number' && (
                            <span className="tracker" style={{ color: active ? 'rgba(250,250,247,0.7)' : 'var(--ink-mute)' }}>
                              {(c.score * 100).toFixed(0)}%
                            </span>
                          )}
                        </button>
                      )
                    })}
                  </div>
                )}
                {!loading && essayCandidates.length === 0 && (
                  <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 14, color: '#6a6a62' }}>
                    No clear thesis sentences could be pulled from this essay. Switch to "write a custom thesis" instead.
                  </div>
                )}
              </>
            )}

            {thesisMode === 'custom' && (
              <div style={{ maxWidth: 920 }}>
                <div className="tracker" style={{ marginBottom: 6 }}>your thesis sentence</div>
                <textarea
                  value={customThesis}
                  onChange={(event) => onCustomThesisChange(event.target.value)}
                  placeholder="Type the single sentence you want each article scored against."
                  style={{
                    width: '100%',
                    minHeight: 90,
                    padding: '10px 14px',
                    border: '1px solid #1a1a1a',
                    background: '#fafaf7',
                    fontFamily: "'Special Elite', monospace",
                    fontSize: 14,
                    lineHeight: 1.5,
                    color: '#1a1a1a',
                    outline: 'none',
                    resize: 'vertical',
                  }}
                />
              </div>
            )}

            {/* Confirmation gate */}
            <div style={{ marginTop: 28, paddingTop: 16, borderTop: '1px solid rgba(26,26,26,0.18)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#6a6a62' }}>
                {loading
                  ? 'reading your essay…'
                  : 'pick a thesis above, then send the slip across the bench.'}
              </div>
              <button
                type="button"
                onClick={onConfirmEssayThesis}
                disabled={loading || (thesisMode === 'candidate' ? !selectedThesisId : customThesis.trim() === '')}
                style={{
                  background: '#1a1a1a',
                  color: '#fafaf7',
                  border: '1px solid #1a1a1a',
                  padding: '10px 22px',
                  fontFamily: "'IM Fell DW Pica SC', serif",
                  fontSize: 11,
                  letterSpacing: '0.32em',
                  textTransform: 'uppercase',
                  cursor: (loading || (thesisMode === 'candidate' ? !selectedThesisId : customThesis.trim() === '')) ? 'not-allowed' : 'pointer',
                  opacity: (loading || (thesisMode === 'candidate' ? !selectedThesisId : customThesis.trim() === '')) ? 0.55 : 1,
                }}
              >
                continue · score the articles →
              </button>
            </div>
          </div>
        )}

        {isStage1 && inputMode === 'stance' && (
          <div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 36, paddingTop: 8 }}>
            <div>
              <div className="tracker" style={{ color: 'var(--accent)', fontSize: 10, letterSpacing: '0.32em' }}>the proofreader's mark</div>
              {loading && !hasTypo ? (
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10, marginTop: 10, fontFamily: "'Special Elite', monospace", fontSize: 12, color: '#6a6a62' }}>
                  <RFSpinner /><span>· · · cross-checking the archive's vocabulary</span>
                </div>
              ) : hasTypo ? (
                <>
                  <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 17, lineHeight: 1.55, marginTop: 8 }}>
                    {(() => {
                      // Highlight each misspelled term in the query.
                      const terms = (typoCorrection.highlighted_terms ?? []).map(t => t.toLocaleLowerCase())
                      const tokens = typoCorrection.query.split(/(\s+)/)
                      return tokens.map((tok, i) => {
                        const lower = tok.toLocaleLowerCase().replace(/[^\p{L}\p{N}'-]/gu, '')
                        const isTypo = lower && terms.includes(lower)
                        return isTypo
                          ? <span key={i} style={{ borderBottom: '2px wavy #7a1d1d', paddingBottom: 2 }}>{tok}</span>
                          : <span key={i}>{tok}</span>
                      })
                    })()}
                  </div>
                  <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 14, color: '#6a6a62', marginTop: 8, marginBottom: 12 }}>
                    {(typoCorrection.highlighted_terms?.length ?? 0) > 1
                      ? `${typoCorrection.highlighted_terms?.length} words look mistyped. The archive suggests:`
                      : 'A small thing — the archive suggests a more usual spelling.'}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 0, border: '1px solid #1a1a1a' }}>
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
                          gap: 12,
                          fontFamily: "'Old Standard TT', serif",
                        }}
                      >
                        <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 14 }}>{option.query}</span>
                        <span className="tracker" style={{ flexShrink: 0 }}>
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
                  No typo found.
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
                  {/* Rewrite alternatives are LLM-generated. The AI assistant
                      is a separate role from any of the persona compositors
                      / sub-editors, so it gets a plain label with no portrait. */}
                  <RFSpinner /><span>· · · the AI Assistant Editor is rewriting your slip</span>
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
                      <div style={{ display: 'flex', gap: 14, alignItems: 'baseline', fontFamily: "'IM Fell English', serif" }}>
                        <span style={{ fontStyle: 'italic', fontSize: 13, color: '#6a6a62', minWidth: 70, textAlign: 'right' }}>regarding</span>
                        <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 14, color: '#1a1a1a' }}>{r.topic}</span>
                      </div>
                      <div style={{ display: 'flex', gap: 14, alignItems: 'baseline', marginTop: 4, fontFamily: "'IM Fell English', serif" }}>
                        <span style={{ fontStyle: 'italic', fontSize: 13, color: '#6a6a62', minWidth: 70, textAlign: 'right' }}>I believe</span>
                        <span style={{ fontFamily: "'Special Elite', monospace", fontSize: 14, color: '#1a1a1a' }}>{r.opinion}</span>
                      </div>
                      {r.rationale && (
                        <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, marginTop: 6, paddingLeft: 84, color: 'var(--ink-mute)' }}>{r.rationale}</div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Stage 1 confirmation gate — proceeds to stage 2 with whatever query the user has */}
          <div style={{ marginTop: 28, paddingTop: 16, borderTop: '1px solid rgba(26,26,26,0.18)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#6a6a62' }}>
              {loading
                ? 'the editor is checking your slip…'
                : (hasTypo
                  ? 'apply a correction above, or proceed with the slip as written.'
                  : 'when you are ready, send the slip across the bench.')}
            </div>
            <button
              type="button"
              onClick={onSearchAnyway}
              disabled={loading}
              style={{
                background: '#1a1a1a',
                color: '#fafaf7',
                border: '1px solid #1a1a1a',
                padding: '10px 22px',
                fontFamily: "'IM Fell DW Pica SC', serif",
                fontSize: 11,
                letterSpacing: '0.32em',
                textTransform: 'uppercase',
                cursor: loading ? 'wait' : 'pointer',
                opacity: loading ? 0.55 : 1,
              }}
            >
              {hasTypo ? 'search as written →' : 'continue · score the articles →'}
            </button>
          </div>
          </div>
        )}

        {/* Stage 2: scatter (after stage 1 confirmed, before ledger) */}
        {isStage2 && (
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
                <span>the desk · articles that cleared topic-relevance</span>
                <span>{visibleArticles.length} of {articles.length} retained{dismissedIds.size ? ` · ${dismissedIds.size} dismissed` : ''}</span>
              </div>
              <div style={{ position: 'relative', width: 820, height: 440 }}>
                {scatterArticles.map((article, i) => (
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
            <div style={{ paddingTop: 12, paddingLeft: 28, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22 }}>
              {/* Two notes for stage 2:
                  - While the retrieval pipeline is still running we have no
                    cards yet, so the Compositor is on stage telling the
                    reader articles are being pulled from the archive.
                  - Once articles are in, the LLM sub-editor takes over and
                    explains it's now reranking by stance agreement. */}
              {articles.length === 0 ? (
                <StickyNote rotation={-1.5} maxWidth={360}>
                  <div className="tracker" style={{ color: 'var(--accent)', fontSize: 9, letterSpacing: '0.26em', marginBottom: 8 }}>from <PersonaName persona={effectiveRetrievalModel} /></div>
                  <div style={{
                    fontFamily: "'IM Fell English', serif",
                    fontStyle: 'italic',
                    fontSize: 16,
                    lineHeight: 1.55,
                    color: '#1a1a1a',
                  }}>
                    <PersonaName persona={effectiveRetrievalModel} /> is finding articles that are on topic. One moment.
                  </div>
                </StickyNote>
              ) : (
                <StickyNote rotation={-1.5} maxWidth={360}>
                  <div className="tracker" style={{ color: 'var(--accent)', fontSize: 9, letterSpacing: '0.26em', marginBottom: 8 }}>from <PersonaName persona="llm" /></div>
                  <div style={{
                    fontFamily: "'IM Fell English', serif",
                    fontStyle: 'italic',
                    fontSize: 16,
                    lineHeight: 1.55,
                    color: '#1a1a1a',
                  }}>
                    Here are some relevant articles, dear reader. I am presently <span style={{ borderBottom: '1.5px dotted #7a1d1d' }}>reranking</span> them by how closely they agree with you.
                    <br /><br />Give me one second.
                    <span style={{ display: 'block', marginTop: 6, fontFamily: "'Special Elite', monospace", fontSize: 12, fontStyle: 'normal', color: '#6a6a62' }}>(actually, about 30 seconds)</span>
                  </div>
                </StickyNote>
              )}
              <div style={{ width: '100%', maxWidth: 360, display: 'flex', alignItems: 'center', gap: 10, fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#6a6a62' }}>
                <RFSpinner /><span>· · · {progressMessage ?? 'scoring agreement with the sub-editor'}</span>
              </div>
              <div style={{ width: '100%', maxWidth: 360, display: 'flex', flexDirection: 'column', gap: 4, fontFamily: "'Special Elite', monospace", fontSize: 11, color: '#3a3a36' }}>
                {progressLines.map(line => (
                  <RFProgressLine key={line.label} line={line} />
                ))}
              </div>
              {dismissedIds.size > 0 && (
                <div style={{ width: '100%', maxWidth: 360, display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 12, color: '#6a6a62' }}>
                  <span>{dismissedIds.size} dismissed — pull next-best?</span>
                  <button type="button" onClick={onApplyDismissals} className="btn-stamp" style={{ padding: '6px 12px', fontSize: 9 }}>refresh</button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Stage 3: ledger (LEFT) + Editor Overview sticky (RIGHT) */}
        {isStage3 && (
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 460px', gap: 36, position: 'relative', height: '100%', minHeight: 0, alignItems: 'stretch' }}>
            <div style={{ order: 2, display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0 }}>
              {dismissedIds.size > 0 && (
                <div style={{
                  border: '2px solid #1a1a1a',
                  background: '#1a1a1a',
                  color: '#fafaf7',
                  padding: '12px 14px',
                  boxShadow: '0 10px 22px rgba(26,26,26,0.20)',
                  transform: 'rotate(0.35deg)',
                }}>
                  <div className="tracker" style={{ color: 'rgba(250,250,247,0.72)', fontSize: 9, letterSpacing: '0.26em', marginBottom: 5 }}>not relevant marked</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 14, alignItems: 'center' }}>
                    <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 15, lineHeight: 1.35 }}>
                      {dismissedIds.size} {dismissedIds.size === 1 ? 'article was' : 'articles were'} set aside.
                    </div>
                    <button
                      type="button"
                      onClick={onApplyDismissals}
                      style={{
                        flexShrink: 0,
                        background: '#fafaf7',
                        color: '#1a1a1a',
                        border: '1px solid #fafaf7',
                        padding: '8px 12px',
                        fontFamily: "'IM Fell DW Pica SC', serif",
                        fontSize: 10,
                        letterSpacing: '0.24em',
                        textTransform: 'uppercase',
                        cursor: 'pointer',
                        boxShadow: '2px 2px 0 rgba(250,250,247,0.26)',
                      }}
                    >pull next-best →</button>
                  </div>
                </div>
              )}
              {llmIrrelevantArticles.length > 0 && (
                <details style={{
                  borderTop: '1px solid #1a1a1a',
                  borderBottom: '1px solid #1a1a1a',
                  padding: '9px 0',
                  color: '#6a6a62',
                }}>
                  <summary style={{
                    cursor: 'pointer',
                    listStyle: 'none',
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: 12,
                    alignItems: 'baseline',
                    fontFamily: "'IM Fell DW Pica SC', serif",
                    fontSize: 9,
                    letterSpacing: '0.22em',
                    textTransform: 'uppercase',
                    color: '#7a1d1d',
                  }}>
                    <span>AI marked not relevant</span>
                    <span>{llmIrrelevantArticles.length}</span>
                  </summary>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
                    {llmIrrelevantArticles.slice(0, 8).map(article => (
                      <div
                        key={getArticleId(article)}
                        style={{
                          fontFamily: "'IM Fell English', serif",
                          fontStyle: 'italic',
                          fontSize: 12,
                          lineHeight: 1.35,
                          color: '#3a3a36',
                          borderTop: '1px dotted rgba(26,26,26,0.22)',
                          paddingTop: 6,
                        }}
                      >
                        {article.title}
                      </div>
                    ))}
                    {llmIrrelevantArticles.length > 8 && (
                      <div style={{ fontFamily: "'Special Elite', monospace", fontSize: 10, color: '#6a6a62' }}>
                        +{llmIrrelevantArticles.length - 8} more
                      </div>
                    )}
                  </div>
                </details>
              )}
            {/* Editor Overview */}
            <div className="tray-scroll" style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 14,
              flex: 1,
              minHeight: 0,
              padding: '24px 22px 22px',
              background: '#fdf6c9',
              backgroundImage: 'radial-gradient(ellipse at 30% 18%, rgba(255,255,255,0.6), transparent 55%), radial-gradient(ellipse at 80% 90%, rgba(180,150,80,0.18), transparent 60%)',
              boxShadow: '0 6px 14px rgba(26,26,26,0.16), 0 1px 0 rgba(26,26,26,0.06) inset',
              transform: 'rotate(-0.4deg)',
              position: 'relative',
            }}>
              {/* Tape strip at top */}
              <div style={{ position: 'absolute', top: -14, left: '50%', transform: 'translateX(-50%) rotate(-2deg)', width: 96, height: 22, background: 'rgba(214, 196, 130, 0.7)', boxShadow: '0 2px 4px rgba(26,26,26,0.18)' }} aria-hidden />
              <div className="tracker" style={{ color: 'var(--accent)', fontSize: 10, letterSpacing: '0.32em' }}>AI Assistant Editor Overview</div>
              {overviewLoading && !overview && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontFamily: "'IM Fell DW Pica SC', serif", fontSize: 9, letterSpacing: '0.26em', textTransform: 'uppercase', color: '#6a6a62' }}>
                    <RFSpinner /> <span>{overviewDraft ? 'the AI assistant editor is dictating…' : 'composing the brief…'}</span>
                  </div>
                  {overviewDraft && (
                    <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.5, color: '#1a1a1a' }}>
                      <MarkdownText
                        text={overviewDraft}
                        sources={overviewSources}
                        onOpenSource={openResultSource}
                      />
                      <span aria-hidden style={{
                        display: 'inline-block',
                        width: '0.55ch',
                        height: '0.95em',
                        marginLeft: 2,
                        verticalAlign: '-0.1em',
                        background: '#1a1a1a',
                        animation: 'caret-blink 1s steps(1) infinite',
                      }} />
                    </div>
                  )}
                </div>
              )}
              {overviewError && (
                <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#7a1d1d' }}>{overviewError}</div>
              )}
              {overview && (
                <>
                  <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 16, lineHeight: 1.5, color: '#1a1a1a' }}>
                    <MarkdownText
                      text={overview.overview}
                      sources={overviewSources}
                      onOpenSource={openResultSource}
                    />
                  </div>
                  {(overview.supporting_arguments && overview.supporting_arguments.length > 0) && (
                    <div style={{ borderTop: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a', padding: '10px 0' }}>
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 18, color: '#1a1a1a', marginBottom: 8 }}>Authors who support you</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {overview.supporting_arguments.map((item, i) => (
                          <OverviewItem
                            key={i}
                            item={item}
                            sources={overviewSources}
                            onOpenSource={openResultSource}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                  {(overview.opposing_arguments && overview.opposing_arguments.length > 0) && (
                    <div style={{ borderTop: '1px solid #1a1a1a', borderBottom: '1px solid #1a1a1a', padding: '10px 0' }}>
                      <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 18, color: 'var(--accent)', marginBottom: 8 }}>Authors who challenge you</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {overview.opposing_arguments.map((item, i) => (
                          <OverviewItem
                            key={i}
                            item={item}
                            sources={overviewSources}
                            onOpenSource={openResultSource}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                  {overview.caveat && (
                    <div style={{ fontFamily: "'IM Fell English', serif", fontStyle: 'italic', fontSize: 13, color: '#6a6a62' }}>
                      <MarkdownText
                        text={overview.caveat}
                        sources={overviewSources}
                        onOpenSource={openResultSource}
                      />
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
                                >read the article →</button>
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
                                >read the article →</button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}

            </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', position: 'relative', minHeight: 0 }}>
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
              <div className="tray-scroll" style={{ flex: 1, minHeight: 0, paddingBottom: 24 }}>
                {visibleArticles.length === 0 && (
                  <div style={{ padding: 24, fontFamily: "'IM Fell English', serif", fontStyle: 'italic', color: '#6a6a62' }}>
                    {llmIrrelevantArticles.length > 0
                      ? 'The AI marked every candidate article as not relevant.'
                      : (emptyResultsMessage || 'No articles cleared the bench.')}
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
                    hideArticleInfo={sortPhase === 'sorting'}
                  />
                ))}
              </div>
              {/* Cards fly in from scatter positions and land in the empty
                  article-info slots above. Rendered inside the ledger panel
                  (which has position:relative) so coordinates are panel-relative
                  and include the caption + table-header offsets. */}
              {sortPhase === 'sorting' && (
                <SortFlight articles={visibleArticles} />
              )}
            </div>
          </div>
        )}
      </div>

      {/* Floating "ask the editor" button — visible once we're in stage 3 */}
      {isStage3 && visibleArticles.length > 0 && !chatOpen && (
        <button
          type="button"
          onClick={() => setChatOpen(true)}
          style={{
            position: 'absolute',
            right: 32,
            bottom: 56,
            zIndex: 20,
            background: '#1a1a1a',
            color: '#fafaf7',
            border: '1px solid #1a1a1a',
            padding: '14px 20px',
            fontFamily: "'IM Fell DW Pica SC', serif",
            fontSize: 11,
            letterSpacing: '0.32em',
            textTransform: 'uppercase',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            boxShadow: '0 14px 30px rgba(26,26,26,0.20)',
          }}
        >
          <span>↳ ask the editor</span>
          <span>→</span>
        </button>
      )}

      {/* footer — retained-count summary + a "← compose" stamp button to
          mirror the bottom affordance on the about / method pages. */}
      <div style={{
        flexShrink: 0,
        padding: '8px 48px 12px',
        textAlign: 'center',
        fontFamily: "'IM Fell DW Pica SC', serif",
        fontSize: 9,
        letterSpacing: '0.28em',
        textTransform: 'uppercase',
        color: 'var(--ink-mute)',
      }}>
        {topic
          ? `${topic} · ${visibleArticles.length} of ${articles.length} retained${dismissedIds.size ? ` · ${dismissedIds.size} dismissed` : ''}`
          : `${visibleArticles.length} retained${dismissedIds.size ? ` · ${dismissedIds.size} dismissed` : ''}`}
      </div>
      <div style={{ flexShrink: 0, padding: '4px 48px 12px', display: 'flex', justifyContent: 'center' }}>
        <button type="button" onClick={onBackToCompose} className="btn-stamp" style={{ padding: '6px 14px' }}>← compose</button>
      </div>

      {/* overlays */}
      {openArticle && (() => {
        const rank = visibleArticles.findIndex(a => getArticleId(a) === openId) + 1
        return (
          <BroadsheetOverlay
            article={openArticle}
            rank={rank}
            total={visibleArticles.length}
            querySvdDimensions={querySvdDimensions}
            querySvdCorpusChartDimensions={querySvdCorpusChartDimensions}
            onClose={() => setOpenId(null)}
            onDismiss={() => { onDismiss(openArticle); setOpenId(null) }}
            onChatThis={() => onOpenArticleChat(openArticle)}
            onFindSimilar={() => onFindSimilar(openArticle)}
            onExplainRanking={() => onExplainRanking(openArticle, rank)}
            explainState={rankingExplanations[String(openArticle.id)]}
          />
        )
      })()}
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
          attachableArticles={visibleArticles.map((a, i) => ({ id: getArticleId(a), rank: i + 1, title: a.title }))}
          attachedIds={chatAttachedIds}
          onToggleAttachment={onToggleChatAttachment}
          onOpenSource={(resultIndex) => {
            openResultSource(resultIndex)
            setChatOpen(false)
          }}
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

    </div>
  )
}

export default ResultsFlow
