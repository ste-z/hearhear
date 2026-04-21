import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type SyntheticEvent,
} from 'react'
import './App.css'
import {
  Article,
  ArticleSearchResponse,
  EssayClaimCandidate,
  EssayClaimCandidateResponse,
  EssayTextExtractionResponse,
  LlmRelevantParagraph,
  RetrievalModel,
  SvdLatentDimension,
} from './types'
import Chat from './Chat'

type InputMode = 'stance' | 'essay'
type IntroStage = 0 | 1 | 2
type EssayStep = 1 | 2
type EssayThesisMode = 'candidate' | 'custom'
type RerankSelectionMode = 'manual' | 'automatic'
type StanceMethod = 'nli' | 'llm'
type ChunkingMode = 'none' | 'paragraph' | 'semantic'
type FrontendChunkingMode = Exclude<ChunkingMode, 'paragraph'>
type SettingsFocusTarget = 'topic-relevance' | 'agreement-scorer'
type SvdDimensionLabelMap = Record<number, string>
type SvdChartSeriesRole = 'article' | 'query'
type TopicFeedbackSearchOptions = {
  topicFeedbackIrrelevantArticleIds?: string[]
  markTopicFeedbackApplied?: boolean
}

type ConfigResponse = {
  use_llm: boolean
  default_retrieval_model?: string | null
  default_normalize_topic_scores?: boolean | null
  default_stance_method?: string | null
  default_use_chunking?: boolean | null
  default_chunking_mode?: string | null
  supported_chunking_modes?: string[] | null
  supported_stance_methods?: string[] | null
  llm_agreement_available?: boolean | null
  default_rerank_selection_mode?: string | null
  default_auto_rerank_thresholds?: Partial<Record<RetrievalModel, number | null>> | null
  max_auto_rerank_candidates?: number | null
  supported_retrieval_models?: string[] | null
  supported_rerank_selection_modes?: string[] | null
  min_article_year?: number | null
  max_article_year?: number | null
}

type ApiErrorPayload = {
  error?: string
}

const introTopicSequence = [
  'climate',
  'immigration',
  'minimum wage',
] as const

type IntroTopic = (typeof introTopicSequence)[number]

const introClaimsByTopic: Record<IntroTopic, readonly string[]> = {
  climate: [
    'cut emissions',
    'expand clean energy',
    'hold polluters accountable',
  ],
  immigration: [
    'protect asylum rights',
    'expand legal pathways',
    'support new arrivals',
  ],
  'minimum wage': [
    'wages should rise',
    'pay should track inflation',
    'work should pay enough',
  ],
}

const finalIntroTopic = introTopicSequence[introTopicSequence.length - 1]
const introClaimSequence = introClaimsByTopic[finalIntroTopic]
const landingSeenStorageKey = 'hearhear.hasSeenLanding'
const defaultSupportedRetrievalModels: RetrievalModel[] = ['svd', 'tfidf']
const defaultRerankSelectionMode: RerankSelectionMode = 'automatic'
const defaultStanceMethod: StanceMethod = 'nli'
const defaultSupportedStanceMethods: StanceMethod[] = ['nli', 'llm']
const defaultChunkingMode: FrontendChunkingMode = 'none'
const defaultSupportedChunkingModes: FrontendChunkingMode[] = ['none', 'semantic']
const defaultAutoRerankThresholds: Record<RetrievalModel, number> = {
  tfidf: 0.3,
  svd: 0.6,
}
const defaultMaxAutoRerankCandidates = 100

const isRetrievalModel = (value: unknown): value is RetrievalModel => (
  value === 'tfidf' || value === 'svd'
)

const normalizeRetrievalModels = (value: unknown): RetrievalModel[] => {
  if (!Array.isArray(value)) return defaultSupportedRetrievalModels

  const filtered = value.filter(isRetrievalModel)
  const unique = Array.from(new Set(filtered))
  return unique.length > 0 ? unique : defaultSupportedRetrievalModels
}

const isRerankSelectionMode = (value: unknown): value is RerankSelectionMode => (
  value === 'manual' || value === 'automatic'
)

const isStanceMethod = (value: unknown): value is StanceMethod => (
  value === 'nli' || value === 'llm'
)

const isFrontendChunkingMode = (value: unknown): value is FrontendChunkingMode => (
  value === 'none' || value === 'semantic'
)

const normalizeStanceMethods = (value: unknown): StanceMethod[] => {
  if (!Array.isArray(value)) return defaultSupportedStanceMethods

  const filtered = value.filter(isStanceMethod)
  const unique = Array.from(new Set(filtered))
  return unique.length > 0 ? unique : defaultSupportedStanceMethods
}

const normalizeChunkingModes = (value: unknown): FrontendChunkingMode[] => {
  if (!Array.isArray(value)) return defaultSupportedChunkingModes

  const filtered = value.filter(isFrontendChunkingMode)
  const unique = Array.from(new Set(filtered))
  return unique.length > 0 ? unique : defaultSupportedChunkingModes
}

const clampAutoRerankThreshold = (value: number): number => (
  Math.max(0, Math.min(1, value))
)

const normalizeAutoRerankThresholds = (
  value: unknown,
): Record<RetrievalModel, number> => {
  const nextThresholds = { ...defaultAutoRerankThresholds }
  if (!value || typeof value !== 'object') {
    return nextThresholds
  }

  for (const model of defaultSupportedRetrievalModels) {
    const candidateValue = (value as Partial<Record<RetrievalModel, unknown>>)[model]
    if (typeof candidateValue === 'number' && Number.isFinite(candidateValue)) {
      nextThresholds[model] = clampAutoRerankThreshold(candidateValue)
    } else if (typeof candidateValue === 'string' && candidateValue.trim() !== '') {
      const parsed = Number(candidateValue)
      if (Number.isFinite(parsed)) {
        nextThresholds[model] = clampAutoRerankThreshold(parsed)
      }
    }
  }

  return nextThresholds
}

const normalizeConfigYear = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isInteger(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isInteger(parsed)) return parsed
  }
  return null
}

const clampYear = (value: number, minYear: number, maxYear: number): number => (
  Math.min(maxYear, Math.max(minYear, Math.round(value)))
)

const hasSeenLanding = (): boolean => {
  if (typeof window === 'undefined') return false

  try {
    return window.localStorage.getItem(landingSeenStorageKey) === 'true'
  } catch {
    return false
  }
}

const markLandingAsSeen = (): void => {
  if (typeof window === 'undefined') return

  try {
    window.localStorage.setItem(landingSeenStorageKey, 'true')
  } catch {
    // Ignore storage failures and fall back to the current in-memory session.
  }
}

const summarizeApiText = (value: string, maxLength = 180): string => (
  value.replace(/\s+/g, ' ').trim().slice(0, maxLength)
)

const getArticleIdKey = (article: Pick<Article, 'id'>): string => String(article.id)

const readApiJson = async <T,>(response: Response): Promise<T> => {
  const rawText = await response.text()
  let payload: unknown = null

  if (rawText) {
    try {
      payload = JSON.parse(rawText)
    } catch {
      payload = rawText
    }
  }

  const apiError = (
    payload &&
    typeof payload === 'object' &&
    'error' in payload &&
    typeof (payload as ApiErrorPayload).error === 'string'
  )
    ? (payload as ApiErrorPayload).error
    : null

  if (!response.ok) {
    if (apiError) {
      throw new Error(apiError)
    }

    if (typeof payload === 'string') {
      const snippet = summarizeApiText(payload)
      if (snippet.startsWith('<')) {
        throw new Error(
          `The server returned an HTML error page (${response.status}) instead of JSON. Check the server logs or try a smaller PDF.`,
        )
      }
      throw new Error(snippet || `Request failed (${response.status})`)
    }

    throw new Error(`Request failed (${response.status})`)
  }

  if (typeof payload === 'string') {
    const snippet = summarizeApiText(payload)
    if (snippet.startsWith('<')) {
      throw new Error('The server returned HTML instead of JSON.')
    }
    throw new Error(snippet || 'The server returned text instead of JSON.')
  }

  return (payload ?? null) as T
}

const normalizeArticleSearchResponse = (
  payload: Article[] | ArticleSearchResponse | null,
): {
  articles: Article[]
  querySvdCorpusChartDimensions: SvdLatentDimension[]
  querySvdDimensions: SvdLatentDimension[]
  emptyResultsMessage: string | null
} => {
  if (Array.isArray(payload)) {
    return {
      articles: payload,
      querySvdCorpusChartDimensions: [],
      querySvdDimensions: [],
      emptyResultsMessage: null,
    }
  }

  const results = Array.isArray(payload?.results) ? payload.results : []
  const querySvdCorpusChartDimensions = Array.isArray(payload?.query_svd_corpus_chart_dimensions)
    ? payload.query_svd_corpus_chart_dimensions
    : []
  const querySvdDimensions = Array.isArray(payload?.query_svd_dimensions)
    ? payload.query_svd_dimensions
    : []
  const emptyResultsMessage = typeof payload?.empty_results_message === 'string'
    ? payload.empty_results_message
    : null

  return {
    articles: results,
    querySvdCorpusChartDimensions,
    querySvdDimensions,
    emptyResultsMessage,
  }
}

const SVD_RADAR_SIZE = 420
const SVD_RADAR_CENTER = SVD_RADAR_SIZE / 2
const SVD_RADAR_RADIUS = 104
const SVD_RADAR_LEVELS = 4

const clampSvdMagnitude = (value: number): number => (
  Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0))
)

const getSvdMagnitude = (dimension?: SvdLatentDimension | null): number => {
  if (!dimension) return 0
  const magnitude = Math.abs(Number(dimension.magnitude))
  const valueMagnitude = Math.abs(Number(dimension.value))
  return Math.max(
    Number.isFinite(magnitude) ? magnitude : 0,
    Number.isFinite(valueMagnitude) ? valueMagnitude : 0,
  )
}

const getMaxSvdMagnitude = (
  dimensionGroups: Array<SvdLatentDimension[] | null | undefined>,
): number | null => {
  let maxMagnitude = 0
  for (const dimensions of dimensionGroups) {
    for (const dimension of dimensions ?? []) {
      maxMagnitude = Math.max(maxMagnitude, getSvdMagnitude(dimension))
    }
  }
  return maxMagnitude > 0 ? maxMagnitude : null
}

const scaleSvdMagnitude = (
  dimension: SvdLatentDimension,
  maxMagnitude?: number | null,
): number => {
  const magnitude = getSvdMagnitude(dimension)
  if (!maxMagnitude || !Number.isFinite(maxMagnitude) || maxMagnitude <= 0) {
    return clampSvdMagnitude(magnitude)
  }
  return clampSvdMagnitude(magnitude / maxMagnitude)
}

const formatSvdValue = (value: number): string => (
  `${value >= 0 ? '+' : ''}${value.toFixed(3)}`
)

const formatThresholdValue = (value: number): string => value.toFixed(2)

const buildSvdDimensionLookup = (
  dimensions?: SvdLatentDimension[] | null,
): Map<number, SvdLatentDimension> => {
  const lookup = new Map<number, SvdLatentDimension>()
  for (const dimension of dimensions ?? []) {
    if (typeof dimension.dimension_index === 'number') {
      lookup.set(dimension.dimension_index, dimension)
    }
  }
  return lookup
}

const getSvdAnchor = (x: number): 'start' | 'middle' | 'end' => {
  if (x < SVD_RADAR_CENTER - 18) return 'end'
  if (x > SVD_RADAR_CENTER + 18) return 'start'
  return 'middle'
}

const getSvdPoint = (
  index: number,
  total: number,
  radius: number,
): { x: number, y: number } => {
  const angle = (-Math.PI / 2) + ((Math.PI * 2 * index) / total)
  return {
    x: SVD_RADAR_CENTER + (Math.cos(angle) * radius),
    y: SVD_RADAR_CENTER + (Math.sin(angle) * radius),
  }
}

const buildSvdLabelLines = (dimension: SvdLatentDimension): string[] => {
  const firstTerms = dimension.label_terms.slice(0, 3).join(' · ')
  const secondTerms = dimension.label_terms.slice(3, 5).join(' · ')
  return [
    `Concept ${dimension.dimension_label}`,
    firstTerms,
    secondTerms,
  ].filter(Boolean)
}

const buildSvdDisplayLabelLines = (
  dimension: SvdLatentDimension,
  dimensionLabels?: SvdDimensionLabelMap | null,
): string[] => {
  const llmLabel = dimensionLabels?.[dimension.dimension_index]?.trim()
  if (!llmLabel) return buildSvdLabelLines(dimension)

  const firstTerms = dimension.label_terms.slice(0, 3).join(' · ')
  return [
    llmLabel,
    firstTerms,
  ].filter(Boolean)
}

function SvdRadarChart(
  {
    dimensions,
    comparisonDimensions = null,
    dimensionLabels = null,
    primaryLabel = 'Article',
    primaryRole = 'article',
    comparisonLabel = 'Query',
    maxMagnitude = null,
    ariaLabel = 'Radar chart of SVD concepts',
    caption = 'Radius shows absolute loading, while filled and hollow points preserve the signed concept direction.',
    emptyCopy = 'No SVD concepts are available yet.',
  }: {
    dimensions: SvdLatentDimension[]
    comparisonDimensions?: SvdLatentDimension[] | null
    dimensionLabels?: SvdDimensionLabelMap | null
    primaryLabel?: string
    primaryRole?: SvdChartSeriesRole
    comparisonLabel?: string
    maxMagnitude?: number | null
    ariaLabel?: string
    caption?: string
    emptyCopy?: string
  },
): JSX.Element {
  const chartDimensions = dimensions.slice(0, 10)
  if (chartDimensions.length === 0) {
    return (
      <div className="svd-radar-shell">
        <p className="svd-radar-caption">{emptyCopy}</p>
      </div>
    )
  }

  const comparisonByDimension = buildSvdDimensionLookup(comparisonDimensions)
  const hasComparisonSeries = chartDimensions.every(
    (dimension) => comparisonByDimension.has(dimension.dimension_index),
  )

  const areaPoints = chartDimensions
    .map((dimension, index) => {
      const radius = scaleSvdMagnitude(dimension, maxMagnitude) * SVD_RADAR_RADIUS
      const point = getSvdPoint(index, chartDimensions.length, radius)
      return `${point.x},${point.y}`
    })
    .join(' ')
  const comparisonAreaPoints = hasComparisonSeries
    ? chartDimensions
      .map((dimension, index) => {
        const comparisonDimension = comparisonByDimension.get(dimension.dimension_index)
        const radius = comparisonDimension
          ? scaleSvdMagnitude(comparisonDimension, maxMagnitude) * SVD_RADAR_RADIUS
          : 0
        const point = getSvdPoint(index, chartDimensions.length, radius)
        return `${point.x},${point.y}`
      })
      .join(' ')
    : ''

  const renderRadarPoint = (
    dimension: SvdLatentDimension,
    index: number,
    role: SvdChartSeriesRole,
  ): JSX.Element => {
    const pointRadius = scaleSvdMagnitude(dimension, maxMagnitude) * SVD_RADAR_RADIUS
    const point = getSvdPoint(index, chartDimensions.length, pointRadius)
    return (
      <circle
        key={`${role}-point-${dimension.dimension_index}`}
        className={`svd-radar-point ${role} ${dimension.pole}`}
        cx={point.x}
        cy={point.y}
        r={role === 'query' ? 3.9 : 4.6}
      />
    )
  }

  return (
    <div className="svd-radar-shell">
      <div className="svd-radar-legend" aria-label="Radar chart legend">
        <span className="svd-radar-legend-item">
          <span className={`svd-radar-legend-line ${primaryRole}`} aria-hidden="true" />
          {primaryLabel}
        </span>
        {hasComparisonSeries && (
          <span className="svd-radar-legend-item">
            <span className="svd-radar-legend-line query" aria-hidden="true" />
            {comparisonLabel}
          </span>
        )}
        <span className="svd-radar-legend-item">
          <span className="svd-radar-legend-point positive" aria-hidden="true" />
          Positive activation
        </span>
        <span className="svd-radar-legend-item">
          <span className="svd-radar-legend-point negative" aria-hidden="true" />
          Negative activation
        </span>
      </div>
      <svg
        className="svd-radar"
        viewBox={`0 0 ${SVD_RADAR_SIZE} ${SVD_RADAR_SIZE}`}
        role="img"
        aria-label={ariaLabel}
      >
        {Array.from({ length: SVD_RADAR_LEVELS }, (_, levelIndex) => {
          const scale = (levelIndex + 1) / SVD_RADAR_LEVELS
          const points = chartDimensions
            .map((_dimension, index) => {
              const point = getSvdPoint(
                index,
                chartDimensions.length,
                SVD_RADAR_RADIUS * scale,
              )
              return `${point.x},${point.y}`
            })
            .join(' ')

          return (
            <polygon
              key={`ring-${scale}`}
              className="svd-radar-ring"
              points={points}
            />
          )
        })}

        <polygon className={`svd-radar-area ${primaryRole}`} points={areaPoints} />
        {hasComparisonSeries && (
          <polygon className="svd-radar-area query" points={comparisonAreaPoints} />
        )}

        {chartDimensions.map((dimension, index) => {
          const axisPoint = getSvdPoint(index, chartDimensions.length, SVD_RADAR_RADIUS)
          const labelPoint = getSvdPoint(index, chartDimensions.length, SVD_RADAR_RADIUS + 30)
          const labelLines = buildSvdDisplayLabelLines(dimension, dimensionLabels)
          const anchor = getSvdAnchor(labelPoint.x)
          const labelStartY = labelPoint.y - ((labelLines.length - 1) * 9)
          const comparisonDimension = comparisonByDimension.get(dimension.dimension_index)

          return (
            <g key={`axis-${dimension.dimension_index}`}>
              <line
                className="svd-radar-axis"
                x1={SVD_RADAR_CENTER}
                y1={SVD_RADAR_CENTER}
                x2={axisPoint.x}
                y2={axisPoint.y}
              />
              {renderRadarPoint(dimension, index, primaryRole)}
              {hasComparisonSeries && comparisonDimension && renderRadarPoint(
                comparisonDimension,
                index,
                'query',
              )}
              <text
                className="svd-radar-label"
                x={labelPoint.x}
                y={labelStartY}
                textAnchor={anchor}
              >
                {labelLines.map((line, lineIndex) => (
                  <tspan
                    key={`${dimension.dimension_index}-${lineIndex}`}
                    x={labelPoint.x}
                    dy={lineIndex === 0 ? 0 : 12}
                    className={lineIndex === 0 ? 'svd-radar-label-index' : undefined}
                  >
                    {line}
                  </tspan>
                ))}
              </text>
            </g>
          )
        })}
      </svg>
      <p className="svd-radar-caption">{caption}</p>
    </div>
  )
}

const renderSvdConceptBarTrack = (
  dimension: SvdLatentDimension,
  role: SvdChartSeriesRole,
  maxMagnitude?: number | null,
): JSX.Element => {
  const widthPercent = `${scaleSvdMagnitude(dimension, maxMagnitude) * 100}%`

  return (
    <div className={`svd-concept-bar-track ${role}`} aria-hidden="true">
      <div className="svd-concept-bar-half negative">
        {dimension.value < 0 && (
          <span
            className={`svd-concept-bar-fill ${role}`}
            style={{ width: widthPercent }}
          />
        )}
      </div>
      <div className="svd-concept-bar-half positive">
        {dimension.value >= 0 && (
          <span
            className={`svd-concept-bar-fill ${role}`}
            style={{ width: widthPercent }}
          />
        )}
      </div>
      <span className="svd-concept-bar-zero" />
    </div>
  )
}

function SvdConceptBarChart(
  {
    dimensions,
    comparisonDimensions = null,
    dimensionLabels = null,
    primaryLabel = 'Article',
    primaryRole = 'article',
    comparisonLabel = 'Query',
    maxMagnitude = null,
    emptyCopy = 'No article-specific SVD concepts are available yet.',
  }: {
    dimensions: SvdLatentDimension[]
    comparisonDimensions?: SvdLatentDimension[] | null
    dimensionLabels?: SvdDimensionLabelMap | null
    primaryLabel?: string
    primaryRole?: SvdChartSeriesRole
    comparisonLabel?: string
    maxMagnitude?: number | null
    emptyCopy?: string
  },
): JSX.Element {
  const chartDimensions = dimensions.slice(0, 10)
  const comparisonByDimension = buildSvdDimensionLookup(comparisonDimensions)
  const hasComparisonSeries = chartDimensions.some(
    (dimension) => comparisonByDimension.has(dimension.dimension_index),
  )

  if (chartDimensions.length === 0) {
    return (
      <div className="svd-concept-bar-chart">
        <div className="svd-concept-bar-empty">
          {emptyCopy}
        </div>
      </div>
    )
  }

  return (
    <div className="svd-concept-bar-chart">
      {hasComparisonSeries && (
        <div className="svd-concept-legend" aria-label="Bar chart legend">
          <span className="svd-concept-legend-item">
            <span className={`svd-concept-legend-swatch ${primaryRole}`} aria-hidden="true" />
            {primaryLabel}
          </span>
          <span className="svd-concept-legend-item">
            <span className="svd-concept-legend-swatch query" aria-hidden="true" />
            {comparisonLabel}
          </span>
        </div>
      )}
      <div className="svd-concept-axis-row" aria-hidden="true">
        <div className="svd-concept-axis-copy" />
        <div className="svd-concept-axis">
          <span className="svd-concept-axis-label negative">Negative</span>
          <span className="svd-concept-axis-label positive">Positive</span>
          <span className="svd-concept-axis-center" />
        </div>
        <div className="svd-concept-axis-value" />
      </div>

      <div className="svd-concept-bar-list">
        {chartDimensions.map((dimension) => {
          const comparisonDimension = comparisonByDimension.get(dimension.dimension_index)

          return (
            <div
              key={`concept-bar-${dimension.dimension_index}`}
              className="svd-concept-bar-row"
            >
              <div className="svd-concept-bar-copy">
                <div className="svd-dimension-title">
                  {dimensionLabels?.[dimension.dimension_index]?.trim() || `Concept ${dimension.dimension_label}`}
                </div>
                <div className="svd-dimension-terms">
                  {dimension.label_text}
                </div>
              </div>

              <div className="svd-concept-bar-track-stack">
                {renderSvdConceptBarTrack(dimension, primaryRole, maxMagnitude)}
                {hasComparisonSeries && (
                  comparisonDimension
                    ? renderSvdConceptBarTrack(comparisonDimension, 'query', maxMagnitude)
                    : <div className="svd-concept-bar-track missing" aria-hidden="true" />
                )}
              </div>

              <div className="svd-concept-bar-value-block">
                <span className={`svd-dimension-value ${primaryRole}`}>
                  {formatSvdValue(dimension.value)}
                </span>
                {hasComparisonSeries && (
                  <span className={`svd-dimension-value query ${comparisonDimension ? '' : 'missing'}`}>
                    {comparisonDimension ? formatSvdValue(comparisonDimension.value) : 'n/a'}
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function YearRangeSlider(
  {
    minYear,
    maxYear,
    startYear,
    endYear,
    disabled = false,
    onStartYearChange,
    onEndYearChange,
  }: {
    minYear: number
    maxYear: number
    startYear: number
    endYear: number
    disabled?: boolean
    onStartYearChange: (nextYear: number) => void
    onEndYearChange: (nextYear: number) => void
  },
): JSX.Element {
  const sliderShellRef = useRef<HTMLDivElement | null>(null)
  const [draggingThumb, setDraggingThumb] = useState<'start' | 'end' | null>(null)
  const yearSpan = Math.max(0, maxYear - minYear)
  const startPercent = yearSpan === 0 ? 0 : (((startYear - minYear) / yearSpan) * 100)
  const endPercent = yearSpan === 0 ? 100 : (((endYear - minYear) / yearSpan) * 100)

  const resolveYearFromClientX = (clientX: number): number => {
    const sliderBounds = sliderShellRef.current?.getBoundingClientRect()
    if (!sliderBounds || sliderBounds.width <= 0 || yearSpan === 0) {
      return startYear
    }
    const relativeX = Math.min(sliderBounds.width, Math.max(0, clientX - sliderBounds.left))
    const nextPercent = relativeX / sliderBounds.width
    const nextYear = minYear + Math.round(nextPercent * yearSpan)
    return clampYear(nextYear, minYear, maxYear)
  }

  const applyDraggedYear = (clientX: number, thumb: 'start' | 'end'): void => {
    const nextYear = resolveYearFromClientX(clientX)
    if (thumb === 'start') {
      onStartYearChange(Math.min(nextYear, endYear))
      return
    }
    onEndYearChange(Math.max(nextYear, startYear))
  }

  useEffect(() => {
    if (!draggingThumb || disabled) {
      return
    }

    const handlePointerMove = (event: PointerEvent): void => {
      applyDraggedYear(event.clientX, draggingThumb)
    }

    const stopDragging = (): void => {
      setDraggingThumb(null)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', stopDragging)
    window.addEventListener('pointercancel', stopDragging)

    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', stopDragging)
      window.removeEventListener('pointercancel', stopDragging)
    }
  }, [disabled, draggingThumb, endYear, maxYear, minYear, onEndYearChange, onStartYearChange, startYear, yearSpan])

  const beginDrag = (
    thumb: 'start' | 'end',
    event: ReactPointerEvent<HTMLButtonElement | HTMLDivElement>,
  ): void => {
    if (disabled) return
    event.preventDefault()
    event.stopPropagation()
    setDraggingThumb(thumb)
    applyDraggedYear(event.clientX, thumb)
  }

  const handleTrackPointerDown = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (disabled) return
    const nextYear = resolveYearFromClientX(event.clientX)
    const nearestThumb = Math.abs(nextYear - startYear) <= Math.abs(nextYear - endYear)
      ? 'start'
      : 'end'
    beginDrag(nearestThumb, event)
  }

  const nudgeThumb = (thumb: 'start' | 'end', delta: number): void => {
    if (thumb === 'start') {
      onStartYearChange(clampYear(startYear + delta, minYear, endYear))
      return
    }
    onEndYearChange(clampYear(endYear + delta, startYear, maxYear))
  }

  const handleThumbKeyDown = (
    thumb: 'start' | 'end',
    event: ReactKeyboardEvent<HTMLButtonElement>,
  ): void => {
    if (disabled) return

    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
      event.preventDefault()
      nudgeThumb(thumb, -1)
      return
    }

    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
      event.preventDefault()
      nudgeThumb(thumb, 1)
      return
    }

    if (event.key === 'Home') {
      event.preventDefault()
      if (thumb === 'start') {
        onStartYearChange(minYear)
      } else {
        onEndYearChange(startYear)
      }
      return
    }

    if (event.key === 'End') {
      event.preventDefault()
      if (thumb === 'start') {
        onStartYearChange(endYear)
      } else {
        onEndYearChange(maxYear)
      }
    }
  }

  return (
    <div className={`year-range-slider-shell ${disabled ? 'disabled' : ''}`}>
      <div
        ref={sliderShellRef}
        className="year-range-slider-track"
        onPointerDown={handleTrackPointerDown}
        role="presentation"
      >
        <span className="year-range-track" aria-hidden="true" />
        <span
          className="year-range-track active"
          aria-hidden="true"
          style={{
            left: `${startPercent}%`,
            width: `${Math.max(0, endPercent - startPercent)}%`,
          }}
        />
        <button
          type="button"
          className={`year-range-handle start ${draggingThumb === 'start' ? 'dragging' : ''}`}
          style={{ left: `calc(${startPercent}% - 10px)` }}
          onPointerDown={(event) => beginDrag('start', event)}
          onKeyDown={(event) => handleThumbKeyDown('start', event)}
          disabled={disabled}
          role="slider"
          aria-label="Start year"
          aria-valuemin={minYear}
          aria-valuemax={endYear}
          aria-valuenow={startYear}
          aria-valuetext={String(startYear)}
        />
        <button
          type="button"
          className={`year-range-handle end ${draggingThumb === 'end' ? 'dragging' : ''}`}
          style={{ left: `calc(${endPercent}% - 10px)` }}
          onPointerDown={(event) => beginDrag('end', event)}
          onKeyDown={(event) => handleThumbKeyDown('end', event)}
          disabled={disabled}
          role="slider"
          aria-label="End year"
          aria-valuemin={startYear}
          aria-valuemax={maxYear}
          aria-valuenow={endYear}
          aria-valuetext={String(endYear)}
        />
      </div>
    </div>
  )
}

type RankingWeightBoundary = 'topic-recency' | 'recency-agreement'

const clampUnit = (value: number): number => (
  Math.min(1, Math.max(0, value))
)

const formatWeightShare = (value: number): string => `${Math.round(clampUnit(value) * 100)}%`

const parseWeightPercentInput = (value: string, fallback: number): number => {
  if (value.trim() === '') return fallback
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.round(Math.min(100, Math.max(0, parsed)))
}

const rebalanceWeightShares = (
  target: 'topic' | 'recency' | 'agreement',
  nextTargetShare: number,
  currentShares: {
    topic: number
    recency: number
    agreement: number
  },
): {
  topic: number
  recency: number
  agreement: number
} => {
  const nextTarget = clampUnit(nextTargetShare)
  const remainder = 1 - nextTarget
  const otherKeys = (['topic', 'recency', 'agreement'] as const).filter(key => key !== target)
  const otherTotal = otherKeys.reduce((sum, key) => sum + currentShares[key], 0)
  const firstOther = otherKeys[0]
  const secondOther = otherKeys[1]
  const nextShares = {
    topic: currentShares.topic,
    recency: currentShares.recency,
    agreement: currentShares.agreement,
  }

  nextShares[target] = nextTarget
  if (otherTotal > 0) {
    nextShares[firstOther] = remainder * (currentShares[firstOther] / otherTotal)
    nextShares[secondOther] = remainder * (currentShares[secondOther] / otherTotal)
  } else {
    nextShares[firstOther] = remainder / 2
    nextShares[secondOther] = remainder / 2
  }

  const roundedTopic = Number(nextShares.topic.toFixed(3))
  const roundedRecency = Number(nextShares.recency.toFixed(3))
  const roundedAgreement = Number(Math.max(0, 1 - roundedTopic - roundedRecency).toFixed(3))
  return {
    topic: roundedTopic,
    recency: roundedRecency,
    agreement: roundedAgreement,
  }
}

function RankingWeightSlider(
  {
    topicWeight,
    recencyWeight,
    agreementWeight,
    onChange,
  }: {
    topicWeight: number
    recencyWeight: number
    agreementWeight: number
    onChange: (nextWeights: {
      topicWeight: number
      recencyWeight: number
      agreementWeight: number
    }) => void
  },
): JSX.Element {
  const sliderRef = useRef<HTMLDivElement | null>(null)
  const [draggingBoundary, setDraggingBoundary] = useState<RankingWeightBoundary | null>(null)
  const safeTopicWeight = Math.max(0, Number.isFinite(topicWeight) ? topicWeight : 0)
  const safeRecencyWeight = Math.max(0, Number.isFinite(recencyWeight) ? recencyWeight : 0)
  const safeAgreementWeight = Math.max(0, Number.isFinite(agreementWeight) ? agreementWeight : 0)
  const totalWeight = safeTopicWeight + safeRecencyWeight + safeAgreementWeight
  const topicShare = totalWeight > 0 ? safeTopicWeight / totalWeight : 0.4
  const recencyShare = totalWeight > 0 ? safeRecencyWeight / totalWeight : 0.2
  const agreementShare = totalWeight > 0 ? safeAgreementWeight / totalWeight : 0.4
  const firstBoundary = clampUnit(topicShare)
  const secondBoundary = clampUnit(topicShare + recencyShare)

  const publishShares = (
    nextTopicShare: number,
    nextRecencyShare: number,
    nextAgreementShare: number,
  ): void => {
    const rounded = (value: number): number => Number(clampUnit(value).toFixed(3))
    onChange({
      topicWeight: rounded(nextTopicShare),
      recencyWeight: rounded(nextRecencyShare),
      agreementWeight: rounded(nextAgreementShare),
    })
  }

  const handleDirectShareChange = (
    target: 'topic' | 'recency' | 'agreement',
    rawValue: string,
  ): void => {
    const fallbackPercent = Math.round(
      (target === 'topic'
        ? topicShare
        : target === 'recency'
          ? recencyShare
          : agreementShare) * 100,
    )
    const nextPercent = parseWeightPercentInput(rawValue, fallbackPercent)
    const nextShares = rebalanceWeightShares(target, nextPercent / 100, {
      topic: topicShare,
      recency: recencyShare,
      agreement: agreementShare,
    })
    publishShares(nextShares.topic, nextShares.recency, nextShares.agreement)
  }

  const resolveShareFromClientX = (clientX: number): number => {
    const bounds = sliderRef.current?.getBoundingClientRect()
    if (!bounds || bounds.width <= 0) {
      return firstBoundary
    }
    const relativeX = Math.min(bounds.width, Math.max(0, clientX - bounds.left))
    return clampUnit(relativeX / bounds.width)
  }

  const applyBoundaryShare = (
    rawShare: number,
    boundary: RankingWeightBoundary,
  ): void => {
    if (boundary === 'topic-recency') {
      const nextFirstBoundary = Math.min(secondBoundary, clampUnit(rawShare))
      publishShares(
        nextFirstBoundary,
        secondBoundary - nextFirstBoundary,
        1 - secondBoundary,
      )
      return
    }

    const nextSecondBoundary = Math.max(firstBoundary, clampUnit(rawShare))
    publishShares(
      firstBoundary,
      nextSecondBoundary - firstBoundary,
      1 - nextSecondBoundary,
    )
  }

  useEffect(() => {
    if (!draggingBoundary) {
      return
    }

    const handlePointerMove = (event: PointerEvent): void => {
      applyBoundaryShare(resolveShareFromClientX(event.clientX), draggingBoundary)
    }

    const stopDragging = (): void => {
      setDraggingBoundary(null)
    }

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', stopDragging)
    window.addEventListener('pointercancel', stopDragging)

    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', stopDragging)
      window.removeEventListener('pointercancel', stopDragging)
    }
  }, [draggingBoundary, firstBoundary, secondBoundary])

  const beginDrag = (
    boundary: RankingWeightBoundary,
    event: ReactPointerEvent<HTMLButtonElement | HTMLDivElement>,
  ): void => {
    event.preventDefault()
    event.stopPropagation()
    setDraggingBoundary(boundary)
    applyBoundaryShare(resolveShareFromClientX(event.clientX), boundary)
  }

  const handleTrackPointerDown = (event: ReactPointerEvent<HTMLDivElement>): void => {
    const share = resolveShareFromClientX(event.clientX)
    const nearestBoundary = Math.abs(share - firstBoundary) <= Math.abs(share - secondBoundary)
      ? 'topic-recency'
      : 'recency-agreement'
    beginDrag(nearestBoundary, event)
  }

  const nudgeBoundary = (boundary: RankingWeightBoundary, delta: number): void => {
    if (boundary === 'topic-recency') {
      applyBoundaryShare(firstBoundary + delta, boundary)
      return
    }
    applyBoundaryShare(secondBoundary + delta, boundary)
  }

  const handleBoundaryKeyDown = (
    boundary: RankingWeightBoundary,
    event: ReactKeyboardEvent<HTMLButtonElement>,
  ): void => {
    const step = event.shiftKey ? 0.1 : 0.05
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
      event.preventDefault()
      nudgeBoundary(boundary, -step)
      return
    }

    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
      event.preventDefault()
      nudgeBoundary(boundary, step)
      return
    }

    if (event.key === 'Home') {
      event.preventDefault()
      applyBoundaryShare(boundary === 'topic-recency' ? 0 : firstBoundary, boundary)
      return
    }

    if (event.key === 'End') {
      event.preventDefault()
      applyBoundaryShare(boundary === 'topic-recency' ? secondBoundary : 1, boundary)
    }
  }

  return (
    <div className="ranking-weight-slider">
      <div
        ref={sliderRef}
        className="ranking-weight-track"
        onPointerDown={handleTrackPointerDown}
        role="presentation"
      >
        <span
          className="ranking-weight-segment topic"
          style={{ left: 0, width: `${firstBoundary * 100}%` }}
          aria-hidden="true"
        />
        <span
          className="ranking-weight-segment recency"
          style={{
            left: `${firstBoundary * 100}%`,
            width: `${Math.max(0, secondBoundary - firstBoundary) * 100}%`,
          }}
          aria-hidden="true"
        />
        <span
          className="ranking-weight-segment agreement"
          style={{
            left: `${secondBoundary * 100}%`,
            width: `${Math.max(0, 1 - secondBoundary) * 100}%`,
          }}
          aria-hidden="true"
        />
        <button
          type="button"
          className={`ranking-weight-handle topic-recency ${draggingBoundary === 'topic-recency' ? 'dragging' : ''}`}
          style={{ left: `calc(${firstBoundary * 100}% - 11px)` }}
          onPointerDown={(event) => beginDrag('topic-recency', event)}
          onKeyDown={(event) => handleBoundaryKeyDown('topic-recency', event)}
          role="slider"
          aria-label="Boundary between topic and recency weight"
          aria-valuemin={0}
          aria-valuemax={Math.round(secondBoundary * 100)}
          aria-valuenow={Math.round(firstBoundary * 100)}
          aria-valuetext={`Topic ${formatWeightShare(topicShare)}, recency ${formatWeightShare(recencyShare)}`}
        />
        <button
          type="button"
          className={`ranking-weight-handle recency-agreement ${draggingBoundary === 'recency-agreement' ? 'dragging' : ''}`}
          style={{ left: `calc(${secondBoundary * 100}% - 11px)` }}
          onPointerDown={(event) => beginDrag('recency-agreement', event)}
          onKeyDown={(event) => handleBoundaryKeyDown('recency-agreement', event)}
          role="slider"
          aria-label="Boundary between recency and agreement weight"
          aria-valuemin={Math.round(firstBoundary * 100)}
          aria-valuemax={100}
          aria-valuenow={Math.round(secondBoundary * 100)}
          aria-valuetext={`Recency ${formatWeightShare(recencyShare)}, agreement ${formatWeightShare(agreementShare)}`}
        />
      </div>
      <div className="ranking-weight-legend">
        <div className="ranking-weight-legend-item topic">
          <span className="ranking-weight-swatch" aria-hidden="true" />
          <label>
            <input
              type="number"
              min="0"
              max="100"
              step="1"
              value={Math.round(topicShare * 100)}
              onChange={(event) => handleDirectShareChange('topic', event.target.value)}
              aria-label="Topic weight percentage"
            />
            <span>%</span>
          </label>
          <span>Topic</span>
        </div>
        <div className="ranking-weight-legend-item recency">
          <span className="ranking-weight-swatch" aria-hidden="true" />
          <label>
            <input
              type="number"
              min="0"
              max="100"
              step="1"
              value={Math.round(recencyShare * 100)}
              onChange={(event) => handleDirectShareChange('recency', event.target.value)}
              aria-label="Recency weight percentage"
            />
            <span>%</span>
          </label>
          <span>Recency</span>
        </div>
        <div className="ranking-weight-legend-item agreement">
          <span className="ranking-weight-swatch" aria-hidden="true" />
          <label>
            <input
              type="number"
              min="0"
              max="100"
              step="1"
              value={Math.round(agreementShare * 100)}
              onChange={(event) => handleDirectShareChange('agreement', event.target.value)}
              aria-label="Agreement weight percentage"
            />
            <span>%</span>
          </label>
          <span>Agreement</span>
        </div>
      </div>
    </div>
  )
}

function App(): JSX.Element {
  const hasSeenLandingRef = useRef<boolean>(hasSeenLanding())
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [introSequenceKey, setIntroSequenceKey] = useState<number>(0)
  const [introStage, setIntroStage] = useState<IntroStage>(hasSeenLandingRef.current ? 2 : 0)
  const [typedTopic, setTypedTopic] = useState<string>(
    hasSeenLandingRef.current ? finalIntroTopic : '',
  )
  const [typedClaim, setTypedClaim] = useState<string>(
    hasSeenLandingRef.current ? introClaimSequence[introClaimSequence.length - 1] : '',
  )
  const [inputMode, setInputMode] = useState<InputMode>('stance')
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [topic, setTopic] = useState<string>('')
  const [opinion, setOpinion] = useState<string>('')
  const [topicWeight, setTopicWeight] = useState<number>(0.4)
  const [stanceWeight, setStanceWeight] = useState<number>(0.4)
  const [recencyWeight, setRecencyWeight] = useState<number>(0.2)
  const [rerankTopK, setRerankTopK] = useState<number>(20)
  const [rerankSelectionMode, setRerankSelectionMode] = useState<RerankSelectionMode>(defaultRerankSelectionMode)
  const [stanceMethod, setStanceMethod] = useState<StanceMethod>(defaultStanceMethod)
  const [chunkingMode, setChunkingMode] = useState<FrontendChunkingMode>(defaultChunkingMode)
  const [supportedStanceMethods, setSupportedStanceMethods] = useState<StanceMethod[]>(
    defaultSupportedStanceMethods,
  )
  const [supportedChunkingModes, setSupportedChunkingModes] = useState<FrontendChunkingMode[]>(
    defaultSupportedChunkingModes,
  )
  const [llmAgreementAvailable, setLlmAgreementAvailable] = useState<boolean>(false)
  const [autoRerankThresholds, setAutoRerankThresholds] = useState<Record<RetrievalModel, number>>(
    defaultAutoRerankThresholds,
  )
  const [maxAutoRerankCandidates, setMaxAutoRerankCandidates] = useState<number>(
    defaultMaxAutoRerankCandidates,
  )
  const [normalizeTopicScores, setNormalizeTopicScores] = useState<boolean>(false)
  const [retrievalModel, setRetrievalModel] = useState<RetrievalModel>('svd')
  const [minArticleYear, setMinArticleYear] = useState<number | null>(null)
  const [maxArticleYear, setMaxArticleYear] = useState<number | null>(null)
  const [yearStart, setYearStart] = useState<number | null>(null)
  const [yearEnd, setYearEnd] = useState<number | null>(null)
  const [yearStartInput, setYearStartInput] = useState<string>('')
  const [yearEndInput, setYearEndInput] = useState<string>('')
  const [supportedRetrievalModels, setSupportedRetrievalModels] = useState<RetrievalModel[]>(
    defaultSupportedRetrievalModels,
  )
  const [articles, setArticles] = useState<Article[]>([])
  const [topicFeedbackIrrelevantArticleIds, setTopicFeedbackIrrelevantArticleIds] = useState<string[]>([])
  const [appliedTopicFeedbackArticleIds, setAppliedTopicFeedbackArticleIds] = useState<string[]>([])
  const [querySvdCorpusChartDimensions, setQuerySvdCorpusChartDimensions] = useState<SvdLatentDimension[]>([])
  const [querySvdDimensions, setQuerySvdDimensions] = useState<SvdLatentDimension[]>([])
  const [svdRankingExplanations, setSvdRankingExplanations] = useState<Record<string, {
    loading: boolean
    error: string | null
    explanation: string | null
  }>>({})
  const [svdDimensionLabelStates, setSvdDimensionLabelStates] = useState<Record<string, {
    loading: boolean
    error: string | null
    labels: SvdDimensionLabelMap
  }>>({})
  const [isImportingPdf, setIsImportingPdf] = useState<boolean>(false)
  const [importedPdfName, setImportedPdfName] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [emptyResultsMessage, setEmptyResultsMessage] = useState<string | null>(null)
  const [isAboutOpen, setIsAboutOpen] = useState<boolean>(false)
  const [activeAboutTab, setActiveAboutTab] = useState<InputMode>('stance')
  const [isFilterOpen, setIsFilterOpen] = useState<boolean>(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false)
  const [settingsFocusTarget, setSettingsFocusTarget] = useState<SettingsFocusTarget | null>(null)
  const [essayCandidates, setEssayCandidates] = useState<EssayClaimCandidate[]>([])
  const [essayPreparedText, setEssayPreparedText] = useState<string>('')
  const [selectedEssayCandidateId, setSelectedEssayCandidateId] = useState<string | null>(null)
  const [essayCustomThesis, setEssayCustomThesis] = useState<string>('')
  const [essayThesisMode, setEssayThesisMode] = useState<EssayThesisMode>('candidate')
  const [essayActiveStep, setEssayActiveStep] = useState<EssayStep>(1)
  const essayOptionsRef = useRef<HTMLDivElement | null>(null)
  const settingsScrollPaneRef = useRef<HTMLDivElement | null>(null)
  const topicSettingsRef = useRef<HTMLDivElement | null>(null)
  const agreementSettingsRef = useRef<HTMLDivElement | null>(null)
  const resultsSectionRef = useRef<HTMLDivElement | null>(null)
  const touchStartYRef = useRef<number | null>(null)
  const lastAppliedYearRangeRef = useRef<{ yearStart: number | null, yearEnd: number | null } | null>(null)
  const [isSearchStageVisible, setIsSearchStageVisible] = useState<boolean>(hasSeenLandingRef.current)
  const [hasSubmittedSearch, setHasSubmittedSearch] = useState<boolean>(false)
  const useChunking = chunkingMode !== 'none'

  useEffect(() => {
    let isActive = true

    const loadConfig = async (): Promise<void> => {
      try {
        const response = await fetch('/api/config')
        const data = await readApiJson<ConfigResponse>(response)
        if (!isActive) return
        setUseLlm(Boolean(data.use_llm))
        const supportedModels = normalizeRetrievalModels(data.supported_retrieval_models)
        const supportedAgreementMethods = normalizeStanceMethods(data.supported_stance_methods)
        const supportedChunkingOptions = normalizeChunkingModes(data.supported_chunking_modes)
        const preferredModel = isRetrievalModel(data.default_retrieval_model)
          ? data.default_retrieval_model
          : supportedModels[0]
        const resolvedModel = supportedModels.includes(preferredModel)
          ? preferredModel
          : supportedModels[0]
        const preferredStanceMethod = isStanceMethod(data.default_stance_method)
          ? data.default_stance_method
          : defaultStanceMethod
        const resolvedStanceMethod = supportedAgreementMethods.includes(preferredStanceMethod)
          ? preferredStanceMethod
          : supportedAgreementMethods[0]
        const preferredChunkingMode = isFrontendChunkingMode(data.default_chunking_mode)
          ? data.default_chunking_mode
          : (data.default_use_chunking ? 'semantic' : defaultChunkingMode)
        const resolvedChunkingMode = supportedChunkingOptions.includes(preferredChunkingMode)
          ? preferredChunkingMode
          : (supportedChunkingOptions.includes(defaultChunkingMode)
            ? defaultChunkingMode
            : supportedChunkingOptions[0])
        const nextMinArticleYear = normalizeConfigYear(data.min_article_year)
        const nextMaxArticleYear = normalizeConfigYear(data.max_article_year)
        setSupportedRetrievalModels(supportedModels)
        setSupportedStanceMethods(supportedAgreementMethods)
        setSupportedChunkingModes(supportedChunkingOptions)
        setLlmAgreementAvailable(Boolean(data.llm_agreement_available))
        setRetrievalModel(currentModel => (
          supportedModels.includes(currentModel) ? currentModel : resolvedModel
        ))
        setStanceMethod(currentMethod => (
          supportedAgreementMethods.includes(currentMethod) ? currentMethod : resolvedStanceMethod
        ))
        setChunkingMode(currentMode => (
          supportedChunkingOptions.includes(currentMode) ? currentMode : resolvedChunkingMode
        ))
        setRerankSelectionMode(currentMode => (
          isRerankSelectionMode(data.default_rerank_selection_mode)
            ? data.default_rerank_selection_mode
            : currentMode
        ))
        setAutoRerankThresholds(normalizeAutoRerankThresholds(data.default_auto_rerank_thresholds))
        if (
          typeof data.max_auto_rerank_candidates === 'number'
          && Number.isFinite(data.max_auto_rerank_candidates)
          && data.max_auto_rerank_candidates > 0
        ) {
          setMaxAutoRerankCandidates(Math.round(data.max_auto_rerank_candidates))
        }
        if (
          nextMinArticleYear !== null &&
          nextMaxArticleYear !== null &&
          nextMinArticleYear <= nextMaxArticleYear
        ) {
          setMinArticleYear(nextMinArticleYear)
          setMaxArticleYear(nextMaxArticleYear)
          setYearStart(currentYear => (
            currentYear === null
              ? nextMinArticleYear
              : clampYear(currentYear, nextMinArticleYear, nextMaxArticleYear)
          ))
          setYearEnd(currentYear => (
            currentYear === null
              ? nextMaxArticleYear
              : clampYear(currentYear, nextMinArticleYear, nextMaxArticleYear)
          ))
        }
        setNormalizeTopicScores(Boolean(data.default_normalize_topic_scores))
      } catch (configError) {
        console.error('Config request failed:', configError)
        if (!isActive) return
        setUseLlm(false)
        setRerankSelectionMode(defaultRerankSelectionMode)
        setStanceMethod(defaultStanceMethod)
        setChunkingMode(defaultChunkingMode)
        setSupportedStanceMethods(defaultSupportedStanceMethods)
        setSupportedChunkingModes(defaultSupportedChunkingModes)
        setLlmAgreementAvailable(false)
        setAutoRerankThresholds(defaultAutoRerankThresholds)
        setMaxAutoRerankCandidates(defaultMaxAutoRerankCandidates)
        setError(
          configError instanceof Error
            ? configError.message
            : 'Failed to load app configuration.',
        )
      }
    }

    void loadConfig()

    return () => {
      isActive = false
    }
  }, [])

  useEffect(() => {
    if (!hasSeenLandingRef.current) {
      markLandingAsSeen()
      hasSeenLandingRef.current = true
    }
  }, [])

  useEffect(() => {
    if (isSearchStageVisible) {
      setIntroStage(2)
      setTypedTopic(finalIntroTopic)
      setTypedClaim(introClaimSequence[introClaimSequence.length - 1])
      return
    }

    let isCancelled = false
    const timeoutIds: number[] = []

    const wait = async (ms: number): Promise<void> => {
      await new Promise<void>((resolve) => {
        const timeoutId = window.setTimeout(resolve, ms)
        timeoutIds.push(timeoutId)
      })
    }

    const runIntroSequence = async (): Promise<void> => {
      const prefersReducedMotion = (
        typeof window.matchMedia === 'function' &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches
      )

      setIntroStage(0)
      setTypedTopic('')
      setTypedClaim('')

      if (prefersReducedMotion) {
        setTypedTopic(finalIntroTopic)
        setIntroStage(1)
        setTypedClaim(introClaimSequence[introClaimSequence.length - 1])
        await wait(150)
        if (isCancelled) return
        setIntroStage(2)
        return
      }

      const runTypewriterSequence = async (
        items: readonly string[],
        setValue: (value: string) => void,
        timing: {
          typeDelay: number
          finalTypeDelay?: number
          pauseBeforeDelete: number
          deleteDelay: number
          pauseBeforeNext: number
          pauseAfterFinal: number
        },
      ): Promise<void> => {
        for (let itemIndex = 0; itemIndex < items.length; itemIndex += 1) {
          const item = items[itemIndex]
          const isLastItem = itemIndex === items.length - 1

          for (let charIndex = 1; charIndex <= item.length; charIndex += 1) {
            if (isCancelled) return
            setValue(item.slice(0, charIndex))
            await wait(isLastItem ? (timing.finalTypeDelay ?? timing.typeDelay) : timing.typeDelay)
          }

          if (isLastItem) {
            await wait(timing.pauseAfterFinal)
            return
          }

          await wait(timing.pauseBeforeDelete)
          if (isCancelled) return

          for (let charIndex = item.length - 1; charIndex >= 0; charIndex -= 1) {
            if (isCancelled) return
            setValue(item.slice(0, charIndex))
            await wait(timing.deleteDelay)
          }

          await wait(timing.pauseBeforeNext)
          if (isCancelled) return
        }
      }

      await runTypewriterSequence(introTopicSequence, setTypedTopic, {
        typeDelay: 55,
        finalTypeDelay: 65,
        pauseBeforeDelete: 250,
        deleteDelay: 36,
        pauseBeforeNext: 120,
        pauseAfterFinal: 420,
      })
      if (isCancelled) return

      setIntroStage(1)
      await wait(220)
      if (isCancelled) return

      await runTypewriterSequence(introClaimSequence, setTypedClaim, {
        typeDelay: 42,
        finalTypeDelay: 48,
        pauseBeforeDelete: 340,
        deleteDelay: 24,
        pauseBeforeNext: 120,
        pauseAfterFinal: 420,
      })
      if (isCancelled) return

      setIntroStage(2)
    }

    void runIntroSequence()

    return () => {
      isCancelled = true
      timeoutIds.forEach((timeoutId) => window.clearTimeout(timeoutId))
    }
  }, [introSequenceKey, isSearchStageVisible])

  useEffect(() => {
    if (inputMode !== 'stance') {
      return
    }
    setArticles([])
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setEmptyResultsMessage(null)
    setError(null)
    setHasSubmittedSearch(false)
  }, [chunkingMode, inputMode, opinion, recencyWeight, rerankTopK, stanceMethod, stanceWeight, topic, topicWeight])

  useEffect(() => {
    setArticles([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setEmptyResultsMessage(null)
    setError(null)
    setHasSubmittedSearch(false)
  }, [
    autoRerankThresholds,
    inputMode,
    recencyWeight,
    rerankSelectionMode,
    rerankTopK,
    retrievalModel,
    stanceMethod,
    stanceWeight,
    topicWeight,
    chunkingMode,
  ])

  useEffect(() => {
    if (inputMode !== 'essay') {
      return
    }
    setEssayCandidates([])
    setEssayPreparedText('')
    setSelectedEssayCandidateId(null)
    setEssayCustomThesis('')
    setEssayThesisMode('candidate')
    setEssayActiveStep(1)
    setArticles([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setEmptyResultsMessage(null)
    setError(null)
    setHasSubmittedSearch(false)
  }, [inputMode, searchTerm])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      setIsAboutOpen(false)
      setIsFilterOpen(false)
      setIsSettingsOpen(false)
      setSettingsFocusTarget(null)
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    if (!isSettingsOpen || settingsFocusTarget === null || typeof window === 'undefined') {
      return
    }

    const frameId = window.requestAnimationFrame(() => {
      const target = settingsFocusTarget === 'topic-relevance'
        ? topicSettingsRef.current
        : agreementSettingsRef.current
      const pane = settingsScrollPaneRef.current
      if (!target || !pane) return

      const paneRect = pane.getBoundingClientRect()
      const targetRect = target.getBoundingClientRect()
      pane.scrollTo({
        top: pane.scrollTop + targetRect.top - paneRect.top - 6,
        behavior: 'smooth',
      })
      target.focus({ preventScroll: true })
    })

    return () => window.cancelAnimationFrame(frameId)
  }, [isSettingsOpen, settingsFocusTarget])

  const activateSearchStage = (scrollTop = false): void => {
    if (scrollTop && typeof window !== 'undefined') {
      window.scrollTo({
        top: 0,
        behavior: 'smooth',
      })
    }
    setIsSearchStageVisible(true)
  }

  const returnToLanding = (): void => {
    if (typeof window !== 'undefined') {
      window.scrollTo({
        top: 0,
        behavior: 'auto',
      })
    }

    if (typeof document !== 'undefined') {
      document.body.style.overflow = ''
    }

    touchStartYRef.current = null
    setHasSubmittedSearch(false)
    setIsSearchStageVisible(false)
    setIntroStage(0)
    setTypedTopic('')
    setTypedClaim('')
    setIntroSequenceKey(currentKey => currentKey + 1)
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (isSearchStageVisible || introStage < 2) return

    const isAtTop = (): boolean => window.scrollY <= 4

    const handleWheel = (event: WheelEvent): void => {
      if (!isAtTop() || event.deltaY <= 0) return
      event.preventDefault()
      activateSearchStage()
    }

    const handleTouchStart = (event: TouchEvent): void => {
      touchStartYRef.current = event.touches[0]?.clientY ?? null
    }

    const handleTouchMove = (event: TouchEvent): void => {
      if (!isAtTop()) return
      const startY = touchStartYRef.current
      const currentY = event.touches[0]?.clientY ?? null
      if (startY === null || currentY === null) return
      if (startY - currentY < 18) return
      event.preventDefault()
      activateSearchStage()
    }

    const handleSearchTransitionKey = (event: KeyboardEvent): void => {
      if (!isAtTop()) return
      if (!['ArrowDown', 'PageDown', ' '].includes(event.key)) return
      event.preventDefault()
      activateSearchStage()
    }

    window.addEventListener('wheel', handleWheel, { passive: false })
    window.addEventListener('touchstart', handleTouchStart, { passive: true })
    window.addEventListener('touchmove', handleTouchMove, { passive: false })
    window.addEventListener('keydown', handleSearchTransitionKey)

    return () => {
      window.removeEventListener('wheel', handleWheel)
      window.removeEventListener('touchstart', handleTouchStart)
      window.removeEventListener('touchmove', handleTouchMove)
      window.removeEventListener('keydown', handleSearchTransitionKey)
    }
  }, [introStage, isSearchStageVisible])

  useEffect(() => {
    if (typeof document === 'undefined') return

    const previousOverflow = document.body.style.overflow

    if (isSearchStageVisible && inputMode === 'stance' && !hasSubmittedSearch) {
      document.body.style.overflow = 'hidden'
      return () => {
        document.body.style.overflow = previousOverflow
      }
    }

    document.body.style.overflow = previousOverflow

    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [hasSubmittedSearch, inputMode, isSearchStageVisible])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!hasSubmittedSearch) return

    let secondFrameId = 0

    const frameId = window.requestAnimationFrame(() => {
      secondFrameId = window.requestAnimationFrame(() => {
        scrollToNode(resultsSectionRef.current)
      })
    })

    return () => {
      window.cancelAnimationFrame(frameId)
      if (secondFrameId) {
        window.cancelAnimationFrame(secondFrameId)
      }
    }
  }, [hasSubmittedSearch])

  const trimmedEssayText = searchTerm.trim()
  const trimmedTopic = topic.trim()
  const trimmedOpinion = opinion.trim()
  const trimmedCustomEssayThesis = essayCustomThesis.trim()
  const canSearchStance = inputMode === 'stance' && trimmedTopic !== '' && trimmedOpinion !== ''
  const canAnalyzeEssay = inputMode === 'essay' && trimmedEssayText !== ''
  const selectedEssayCandidate = useMemo(
    () => essayCandidates.find(candidate => candidate.sentence_id === selectedEssayCandidateId) ?? null,
    [essayCandidates, selectedEssayCandidateId],
  )
  const resolvedEssayThesis = essayThesisMode === 'custom'
    ? trimmedCustomEssayThesis
    : (selectedEssayCandidate?.sentence.trim() ?? '')
  const resolvedEssayThesisId = essayThesisMode === 'candidate'
    ? selectedEssayCandidate?.sentence_id ?? null
    : null
  const effectiveStanceMethod: StanceMethod = useChunking ? 'llm' : stanceMethod
  const isLlmAgreementSelected = effectiveStanceMethod === 'llm'
  const shouldUseEssayThesisStep = !isLlmAgreementSelected
  const canSubmitEssay = Boolean(
    essayPreparedText.trim() && (isLlmAgreementSelected || resolvedEssayThesis),
  )
  const isEssayStepTwoAvailable = essayPreparedText.trim() !== ''
  const isUsingCustomEssayThesis = essayThesisMode === 'custom'
  const essayWorkflowStep = shouldUseEssayThesisStep && isEssayStepTwoAvailable ? essayActiveStep : 1
  const canUseSvd = supportedRetrievalModels.includes('svd')
  const canUseTfidf = supportedRetrievalModels.includes('tfidf')
  const canUseNliAgreement = supportedStanceMethods.includes('nli')
  const canUseLlmAgreement = supportedStanceMethods.includes('llm') && llmAgreementAvailable
  const canUseChunking = canUseLlmAgreement && supportedChunkingModes.includes('semantic')
  const canToggleSvd = canUseSvd && canUseTfidf
  const currentAutoRerankThreshold = autoRerankThresholds[retrievalModel]
  const availableYears = useMemo(() => {
    if (minArticleYear === null || maxArticleYear === null || minArticleYear > maxArticleYear) {
      return []
    }
    return Array.from(
      { length: (maxArticleYear - minArticleYear) + 1 },
      (_value, index) => minArticleYear + index,
    )
  }, [maxArticleYear, minArticleYear])
  const hasAvailableYearBounds = availableYears.length > 0
  const resolvedYearStart = yearStart ?? minArticleYear
  const resolvedYearEnd = yearEnd ?? maxArticleYear
  const yearRangeSpan = hasAvailableYearBounds && minArticleYear !== null && maxArticleYear !== null
    ? maxArticleYear - minArticleYear
    : 0
  const hasYearRangeSelection = resolvedYearStart !== null && resolvedYearEnd !== null
  const isYearFilterActive = (
    hasYearRangeSelection &&
    minArticleYear !== null &&
    maxArticleYear !== null &&
    (resolvedYearStart !== minArticleYear || resolvedYearEnd !== maxArticleYear)
  )
  const activeYearRangeLabel = !hasYearRangeSelection
    ? 'All years'
    : (resolvedYearStart === resolvedYearEnd
      ? `${resolvedYearStart}`
      : `${resolvedYearStart}-${resolvedYearEnd}`)
  const yearRangeSummary = !isYearFilterActive || !hasYearRangeSelection
    ? ''
    : (resolvedYearStart === resolvedYearEnd
      ? ` from ${resolvedYearStart}`
      : ` from ${resolvedYearStart} to ${resolvedYearEnd}`)

  const formatDate = (isoDate: string | null): string => {
    if (!isoDate) return 'Unknown date'
    const parsed = new Date(isoDate)
    if (Number.isNaN(parsed.getTime())) return 'Unknown date'
    return parsed.toLocaleDateString()
  }

  const clampUnitScore = (value?: number | null): number | null => {
    if (value === undefined || value === null || Number.isNaN(value)) return null
    return Math.max(0, Math.min(1, value))
  }

  const formatPercent = (value?: number | null): string => {
    const normalized = clampUnitScore(value)
    if (normalized === null) return 'n/a'
    return `${Math.round(normalized * 100)}%`
  }

  const getMeterWidth = (value?: number | null): string => {
    const normalized = clampUnitScore(value)
    return `${Math.round((normalized ?? 0) * 100)}%`
  }

  const renderMetricInfo = (
    label: string,
    tooltipId: string,
    explanation: string,
  ): JSX.Element => (
    <span className="metric-info-wrap">
      <button
        type="button"
        className="metric-info-button"
        aria-label={`Explain ${label.toLowerCase()}`}
        aria-describedby={tooltipId}
      >
        i
      </button>
      <span id={tooltipId} role="tooltip" className="metric-info-tooltip">
        {explanation}
      </span>
    </span>
  )

  const parseTopKInput = (value: string, fallback: number): number => {
    if (value.trim() === '') return fallback
    const parsed = Number(value)
    if (Number.isNaN(parsed)) return fallback
    return Math.min(100, Math.max(1, Math.round(parsed)))
  }

  const parseAutoRerankThresholdInput = (value: string, fallback: number): number => {
    if (value.trim() === '') return fallback
    const parsed = Number(value)
    if (Number.isNaN(parsed)) return fallback
    return clampAutoRerankThreshold(parsed)
  }

  const updateCurrentAutoRerankThreshold = (value: string): void => {
    const nextThreshold = parseAutoRerankThresholdInput(
      value,
      autoRerankThresholds[retrievalModel],
    )
    setAutoRerankThresholds(currentThresholds => ({
      ...currentThresholds,
      [retrievalModel]: nextThreshold,
    }))
  }

  const handleYearStartChange = (value: string): void => {
    const nextStart = Number(value)
    if (Number.isNaN(nextStart)) return
    const boundedStart = (
      minArticleYear !== null && maxArticleYear !== null
        ? clampYear(nextStart, minArticleYear, maxArticleYear)
        : Math.round(nextStart)
    )
    setYearStart(boundedStart)
    setYearEnd(currentEnd => (
      currentEnd === null || currentEnd < boundedStart ? boundedStart : currentEnd
    ))
  }

  const handleYearEndChange = (value: string): void => {
    const nextEnd = Number(value)
    if (Number.isNaN(nextEnd)) return
    const boundedEnd = (
      minArticleYear !== null && maxArticleYear !== null
        ? clampYear(nextEnd, minArticleYear, maxArticleYear)
        : Math.round(nextEnd)
    )
    setYearEnd(boundedEnd)
    setYearStart(currentStart => (
      currentStart === null || currentStart > boundedEnd ? boundedEnd : currentStart
    ))
  }

  const handleYearStartInputChange = (value: string): void => {
    setYearStartInput(value.replace(/[^\d]/g, ''))
  }

  const handleYearEndInputChange = (value: string): void => {
    setYearEndInput(value.replace(/[^\d]/g, ''))
  }

  const commitYearStartInput = (): void => {
    const normalizedValue = yearStartInput.trim()
    if (normalizedValue === '') {
      if (minArticleYear !== null) {
        handleYearStartChange(String(minArticleYear))
      }
      return
    }
    handleYearStartChange(normalizedValue)
  }

  const commitYearEndInput = (): void => {
    const normalizedValue = yearEndInput.trim()
    if (normalizedValue === '') {
      if (maxArticleYear !== null) {
        handleYearEndChange(String(maxArticleYear))
      }
      return
    }
    handleYearEndChange(normalizedValue)
  }

  const scrollToNode = (node: HTMLDivElement | null): void => {
    if (typeof window === 'undefined' || !node) return

    const target = node.querySelector<HTMLElement>('.results-paper') ?? node
    const targetTop = window.scrollY + target.getBoundingClientRect().top - 16

    window.scrollTo({
      top: Math.max(0, targetTop),
      behavior: 'smooth',
    })
  }

  const handleEssaySearch = (value: string): void => {
    setInputMode('essay')
    setImportedPdfName(null)
    setEssayCandidates([])
    setEssayPreparedText('')
    setSelectedEssayCandidateId(null)
    setEssayCustomThesis('')
    setEssayThesisMode('candidate')
    setSearchTerm(value)
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setEssayActiveStep(1)
    activateSearchStage(true)
  }

  const handleImportPdf = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0] ?? null
    event.target.value = ''
    if (!file || isImportingPdf) return

    setIsImportingPdf(true)
    setError(null)
    setArticles([])
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setHasSubmittedSearch(false)
    setEssayCandidates([])
    setSelectedEssayCandidateId(null)
    setEssayCustomThesis('')
    setEssayThesisMode('candidate')
    setEssayPreparedText('')
    setEssayActiveStep(1)

    try {
      const formData = new FormData()
      formData.append('pdf', file)

      const response = await fetch('/api/essay/extract-text', {
        method: 'POST',
        body: formData,
      })
      const data = await readApiJson<EssayTextExtractionResponse>(response)
      const extractedText = String(data.essay_text || '').trim()

      if (!extractedText) {
        throw new Error("We couldn't read text from that PDF. Try another file or paste the essay manually.")
      }

      setSearchTerm(extractedText)
      setImportedPdfName(file.name)
    } catch (fetchError) {
      console.error('PDF text extraction failed:', fetchError)
      setImportedPdfName(null)
      setError(fetchError instanceof Error ? fetchError.message : 'PDF text extraction failed.')
    } finally {
      setIsImportingPdf(false)
    }
  }

  const handleSubmitStance = async (
    options: TopicFeedbackSearchOptions = {},
  ): Promise<void> => {
    if (!canSearchStance || loading) return
    const feedbackArticleIds = options.topicFeedbackIrrelevantArticleIds ?? topicFeedbackIrrelevantArticleIds

    lastAppliedYearRangeRef.current = {
      yearStart: resolvedYearStart,
      yearEnd: resolvedYearEnd,
    }
    setHasSubmittedSearch(true)
    if (typeof document !== 'undefined') {
      document.body.style.overflow = ''
    }
    setLoading(true)
    setError(null)
    setArticles([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setEmptyResultsMessage(null)

    try {
      const response = await fetch('/api/articles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode: 'stance',
          topic: trimmedTopic,
          opinion: trimmedOpinion,
          topic_weight: topicWeight,
          stance_weight: stanceWeight,
          recency_weight: recencyWeight,
          top_k: rerankTopK,
          normalize_topic_scores: normalizeTopicScores,
          stance_method: stanceMethod,
          use_chunking: useChunking,
          chunking_mode: chunkingMode,
          retrieval_model: retrievalModel,
          rerank_selection_mode: rerankSelectionMode,
          rerank_threshold: currentAutoRerankThreshold,
          year_start: resolvedYearStart,
          year_end: resolvedYearEnd,
          topic_feedback_irrelevant_article_ids: feedbackArticleIds,
        }),
      })

      const data = await readApiJson<Article[] | ArticleSearchResponse>(response)
      const normalized = normalizeArticleSearchResponse(data)
      setArticles(normalized.articles)
      setQuerySvdCorpusChartDimensions(normalized.querySvdCorpusChartDimensions)
      setQuerySvdDimensions(normalized.querySvdDimensions)
      setEmptyResultsMessage(normalized.emptyResultsMessage)
      if (options.markTopicFeedbackApplied) {
        setAppliedTopicFeedbackArticleIds(feedbackArticleIds)
      }
    } catch (fetchError) {
      console.error('Search request failed:', fetchError)
      setArticles([])
      setQuerySvdCorpusChartDimensions([])
      setQuerySvdDimensions([])
      setEmptyResultsMessage(null)
      setError(fetchError instanceof Error ? fetchError.message : 'Search request failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleAnalyzeEssay = async (): Promise<void> => {
    if (!canAnalyzeEssay || loading) return

    setLoading(true)
    setError(null)
    setArticles([])
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setEmptyResultsMessage(null)

    try {
      const formData = new FormData()
      formData.append('mode', 'essay')
      formData.append('q', trimmedEssayText)
      formData.append('candidate_top_n', '5')

      const response = await fetch('/api/essay/claim-candidates', {
        method: 'POST',
        body: formData,
      })
      const data = await readApiJson<EssayClaimCandidateResponse>(response)
      const nextEssayText = data.essay_text || trimmedEssayText
      const nextCandidates = data.candidates || []
      const nextSelectedCandidateId = (
        nextCandidates.find(candidate => candidate.sentence_id === selectedEssayCandidateId)?.sentence_id
        ?? nextCandidates[0]?.sentence_id
        ?? null
      )
      const shouldKeepCustomSelection = (
        essayThesisMode === 'custom' &&
        trimmedCustomEssayThesis !== ''
      )

      setEssayPreparedText(nextEssayText)
      setEssayCandidates(nextCandidates)
      setSelectedEssayCandidateId(nextSelectedCandidateId)
      setEssayThesisMode(
        shouldKeepCustomSelection ? 'custom' : (nextCandidates.length > 0 ? 'candidate' : 'custom'),
      )
      setEssayActiveStep(2)
    } catch (fetchError) {
      console.error('Essay analysis failed:', fetchError)
      setEssayCandidates([])
      setSelectedEssayCandidateId(null)
      setEssayThesisMode('candidate')
      setEssayPreparedText('')
      setEssayActiveStep(1)
      setQuerySvdCorpusChartDimensions([])
      setQuerySvdDimensions([])
      setError(fetchError instanceof Error ? fetchError.message : 'Essay analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  const submitEssaySearch = async (
    essayText: string,
    thesisSentence: string,
    thesisId: string | null,
    options: TopicFeedbackSearchOptions = {},
  ): Promise<void> => {
    const nextEssayText = essayText.trim()
    const nextThesisSentence = thesisSentence.trim()
    if (!nextEssayText || loading) return
    if (!isLlmAgreementSelected && !nextThesisSentence) return
    const feedbackArticleIds = options.topicFeedbackIrrelevantArticleIds ?? topicFeedbackIrrelevantArticleIds

    lastAppliedYearRangeRef.current = {
      yearStart: resolvedYearStart,
      yearEnd: resolvedYearEnd,
    }
    setHasSubmittedSearch(true)
    if (typeof document !== 'undefined') {
      document.body.style.overflow = ''
    }
    setLoading(true)
    setError(null)
    setArticles([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setEmptyResultsMessage(null)

    try {
      const response = await fetch('/api/articles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode: 'essay',
          q: nextEssayText,
          selected_thesis_id: thesisId,
          selected_thesis_sentence: nextThesisSentence,
          topic_weight: topicWeight,
          stance_weight: stanceWeight,
          recency_weight: recencyWeight,
          top_k: rerankTopK,
          normalize_topic_scores: normalizeTopicScores,
          stance_method: stanceMethod,
          use_chunking: useChunking,
          chunking_mode: chunkingMode,
          retrieval_model: retrievalModel,
          rerank_selection_mode: rerankSelectionMode,
          rerank_threshold: currentAutoRerankThreshold,
          year_start: resolvedYearStart,
          year_end: resolvedYearEnd,
          topic_feedback_irrelevant_article_ids: feedbackArticleIds,
        }),
      })

      const data = await readApiJson<Article[] | ArticleSearchResponse>(response)
      const normalized = normalizeArticleSearchResponse(data)
      setArticles(normalized.articles)
      setQuerySvdCorpusChartDimensions(normalized.querySvdCorpusChartDimensions)
      setQuerySvdDimensions(normalized.querySvdDimensions)
      setEmptyResultsMessage(normalized.emptyResultsMessage)
      if (options.markTopicFeedbackApplied) {
        setAppliedTopicFeedbackArticleIds(feedbackArticleIds)
      }
    } catch (fetchError) {
      console.error('Essay search failed:', fetchError)
      setArticles([])
      setQuerySvdCorpusChartDimensions([])
      setQuerySvdDimensions([])
      setEmptyResultsMessage(null)
      setError(fetchError instanceof Error ? fetchError.message : 'Essay search failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmitEssay = async (): Promise<void> => {
    if (!canSubmitEssay || loading) return
    await submitEssaySearch(
      essayPreparedText,
      resolvedEssayThesis,
      resolvedEssayThesisId,
    )
  }

  const handleSubmitEssayFromDraft = async (): Promise<void> => {
    if (!canAnalyzeEssay || loading || !isLlmAgreementSelected) return

    const nextEssayText = trimmedEssayText
    setEssayPreparedText(nextEssayText)
    setEssayCandidates([])
    setSelectedEssayCandidateId(null)
    setEssayCustomThesis('')
    setEssayThesisMode('candidate')
    setEssayActiveStep(1)
    await submitEssaySearch(nextEssayText, '', null)
  }

  useEffect(() => {
    if (isFilterOpen || loading || !hasSubmittedSearch) {
      return
    }

    const lastAppliedYearRange = lastAppliedYearRangeRef.current
    if (
      lastAppliedYearRange &&
      lastAppliedYearRange.yearStart === resolvedYearStart &&
      lastAppliedYearRange.yearEnd === resolvedYearEnd
    ) {
      return
    }

    if (inputMode === 'stance') {
      if (!canSearchStance) return
      void handleSubmitStance()
      return
    }

    if (!canSubmitEssay) return
    void handleSubmitEssay()
  }, [
    canSearchStance,
    canSubmitEssay,
    hasSubmittedSearch,
    inputMode,
    isFilterOpen,
    loading,
    resolvedYearEnd,
    resolvedYearStart,
  ])

  useEffect(() => {
    setYearStartInput(resolvedYearStart === null ? '' : String(resolvedYearStart))
  }, [resolvedYearStart])

  useEffect(() => {
    setYearEndInput(resolvedYearEnd === null ? '' : String(resolvedYearEnd))
  }, [resolvedYearEnd])

  const scrollEssayOptions = (direction: 'left' | 'right'): void => {
    const container = essayOptionsRef.current
    if (!container) return
    const amount = Math.max(240, Math.round(container.clientWidth * 0.7))
    container.scrollBy({
      left: direction === 'left' ? -amount : amount,
      behavior: 'smooth',
    })
  }

  const openSettings = (): void => {
    setSettingsFocusTarget(null)
    setIsSettingsOpen(true)
  }

  const closeSettings = (): void => {
    setIsSettingsOpen(false)
    setSettingsFocusTarget(null)
  }

  const openSettingsAt = (target: SettingsFocusTarget): void => {
    setSettingsFocusTarget(target)
    setIsSettingsOpen(true)
  }

  const handleTopicSearchModeChange = (nextModel: RetrievalModel): void => {
    if (!supportedRetrievalModels.includes(nextModel)) return
    setRetrievalModel(nextModel)
  }

  const handleAgreementSearchModeChange = (nextMethod: StanceMethod): void => {
    if (!supportedStanceMethods.includes(nextMethod)) return
    if (nextMethod === 'llm') {
      if (!canUseLlmAgreement) return
      setStanceMethod('llm')
      return
    }

    if (!canUseNliAgreement) return
    setChunkingMode('none')
    setStanceMethod('nli')
  }

  const openAbout = (tab: InputMode = inputMode): void => {
    setActiveAboutTab(tab)
    setIsAboutOpen(true)
  }
  const showScoreGrid = (article: Article): boolean => (
    article.combined_score != null ||
    article.stance_score_normalized != null ||
    article.topic_score_display != null ||
    article.topic_score != null ||
    article.topic_score_normalized != null ||
    article.recency_score_normalized != null
  )
  const getTopicMatchScore = (article: Article): number | null => (
    article.topic_score_display
    ?? article.topic_score
    ?? article.topic_score_normalized
    ?? null
  )
  const hasSvdExplainability = (article: Article): boolean => (
    (Array.isArray(article.svd_query_chart_dimensions) && article.svd_query_chart_dimensions.length > 0) ||
    (Array.isArray(article.svd_chart_dimensions) && article.svd_chart_dimensions.length > 0) ||
    (Array.isArray(article.svd_dimensions) && article.svd_dimensions.length > 0)
  )
  const getSvdDimensionsForLabeling = (article: Article): SvdLatentDimension[] => {
    const seen = new Set<number>()
    const dimensions = [
      ...(article.svd_query_chart_dimensions ?? []),
      ...(article.svd_chart_dimensions ?? []),
      ...(article.svd_dimensions ?? []),
    ]

    return dimensions.filter((dimension) => {
      if (seen.has(dimension.dimension_index)) return false
      seen.add(dimension.dimension_index)
      return true
    })
  }
  const hasStanceSignals = (article: Article): boolean => (
    article.stance_entailment_prob != null ||
    article.stance_neutral_prob != null ||
    article.stance_contradiction_prob != null
  )
  const isLlmIrrelevantArticle = (article: Article): boolean => (
    article.stance_method === 'llm' && article.llm_irrelevant === true
  )
  const hasLlmRelevantParagraphs = (article: Article): boolean => (
    Array.isArray(article.llm_relevant_paragraphs) &&
    article.llm_relevant_paragraphs.length > 0
  )
  const getLlmChunkNoun = (article: Article, plural = true): string => {
    if (article.llm_chunking_mode === 'semantic') {
      return plural ? 'semantic chunks' : 'Semantic chunk'
    }
    return plural ? 'paragraphs' : 'Paragraph'
  }
  const getParagraphEvidenceHint = (article: Article): string => {
    const relatedCount = article.llm_related_chunk_count ?? article.llm_relevant_paragraphs?.length ?? 0
    const totalCount = article.llm_chunk_count ?? 0
    const chunkNoun = getLlmChunkNoun(article)
    if (relatedCount > 0 && totalCount > 0) {
      return `Expand to inspect the strongest ${Math.min(relatedCount, article.llm_relevant_paragraphs?.length ?? relatedCount)} of ${relatedCount} related ${chunkNoun}.`
    }
    return `Expand to inspect the ${chunkNoun} behind this LLM score.`
  }
  const paragraphKey = (
    article: Article,
    paragraph: LlmRelevantParagraph,
    index: number,
  ): string => (
    `${article.id}-paragraph-${paragraph.paragraph_id ?? paragraph.paragraph_index ?? index}`
  )
  const topicFeedbackIrrelevantIdSet = useMemo(
    () => new Set(topicFeedbackIrrelevantArticleIds),
    [topicFeedbackIrrelevantArticleIds],
  )
  const appliedTopicFeedbackIdSet = useMemo(
    () => new Set(appliedTopicFeedbackArticleIds),
    [appliedTopicFeedbackArticleIds],
  )
  const pendingTopicFeedbackArticleIds = useMemo(
    () => topicFeedbackIrrelevantArticleIds.filter(articleId => !appliedTopicFeedbackIdSet.has(articleId)),
    [appliedTopicFeedbackIdSet, topicFeedbackIrrelevantArticleIds],
  )
  const isTopicFeedbackIrrelevantArticle = (article: Article): boolean => (
    topicFeedbackIrrelevantIdSet.has(getArticleIdKey(article))
  )
  const visibleArticles = articles.filter(article => !isLlmIrrelevantArticle(article))
  const topicFeedbackIrrelevantArticles = visibleArticles.filter(isTopicFeedbackIrrelevantArticle)
  const activeVisibleArticles = visibleArticles.filter(article => !isTopicFeedbackIrrelevantArticle(article))
  const queryTopRadarMaxMagnitude = getMaxSvdMagnitude([
    querySvdDimensions,
    ...activeVisibleArticles.map(article => article.svd_query_chart_dimensions),
  ])
  const sharedCorpusRadarMaxMagnitude = getMaxSvdMagnitude([
    querySvdCorpusChartDimensions,
    ...activeVisibleArticles.map(article => article.svd_chart_dimensions),
  ])
  const articleConceptBarMaxMagnitude = getMaxSvdMagnitude([
    ...activeVisibleArticles.map(article => article.svd_dimensions),
    ...activeVisibleArticles.map(article => article.svd_article_query_dimensions),
  ])
  const queryConceptBarMaxMagnitude = getMaxSvdMagnitude([querySvdDimensions])
  const llmIrrelevantArticles = articles.filter(isLlmIrrelevantArticle)
  const canExplainRanking = useLlm === true && retrievalModel === 'svd'
  const shouldShowEssayShortcut = useLlm && inputMode === 'essay' && !isSearchStageVisible
  const pendingTopicFeedbackCount = pendingTopicFeedbackArticleIds.length

  const handleMarkTopicIrrelevant = (article: Article): void => {
    const articleId = getArticleIdKey(article)
    setTopicFeedbackIrrelevantArticleIds(currentIds => (
      currentIds.includes(articleId) ? currentIds : [...currentIds, articleId]
    ))
  }

  const handleUndoTopicIrrelevant = (article: Article): void => {
    const articleId = getArticleIdKey(article)
    setTopicFeedbackIrrelevantArticleIds(currentIds => (
      currentIds.filter(currentId => currentId !== articleId)
    ))
    setAppliedTopicFeedbackArticleIds(currentIds => (
      currentIds.filter(currentId => currentId !== articleId)
    ))
  }

  const handleRefreshTopicFeedback = async (): Promise<void> => {
    if (loading || pendingTopicFeedbackCount === 0) return
    const feedbackArticleIds = [...topicFeedbackIrrelevantArticleIds]

    if (inputMode === 'stance') {
      await handleSubmitStance({
        topicFeedbackIrrelevantArticleIds: feedbackArticleIds,
        markTopicFeedbackApplied: true,
      })
      return
    }

    await submitEssaySearch(
      essayPreparedText,
      resolvedEssayThesis,
      resolvedEssayThesisId,
      {
        topicFeedbackIrrelevantArticleIds: feedbackArticleIds,
        markTopicFeedbackApplied: true,
      },
    )
  }

  const getMatchSummary = (article: Article): string => {
    const hasWeightedRecency = (article.recency_weight ?? recencyWeight) > 0

    if (article.stance_score_normalized === undefined || article.stance_score_normalized === null) {
      return hasWeightedRecency
        ? 'This article ranked mainly on subject overlap and publish-date recency because no clear claim comparison was available yet.'
        : 'This article ranked mainly on subject overlap because no clear claim comparison was available yet.'
    }

    const normalized = article.stance_label?.toLowerCase() ?? ''
    const recencyNote = hasWeightedRecency
      ? ' Its final rank also reflects how recently it was published.'
      : ''

    if (normalized.includes('support') || normalized.includes('entail')) {
      return `This article stays on your topic and likely supports your position.${recencyNote}`
    }
    if (normalized.includes('contradict')) {
      return `This article stays on your topic but likely argues against your position.${recencyNote}`
    }
    if (normalized.includes('neutral')) {
      return `This article stays on your topic, but its position looks mixed or unclear.${recencyNote}`
    }
    return `This article is on your topic and was compared against your statement.${recencyNote}`
  }

  const getOverviewHint = (article: Article): string => {
    const hasThesis = Boolean(article.thesis_sentence)
    const hasSupport = Boolean(article.support_sentences && article.support_sentences.length > 0)

    if (hasThesis && hasSupport) {
      return 'Expand to see the thesis and support sentences.'
    }
    if (hasThesis) {
      return 'Expand to see the thesis sentence.'
    }
    if (hasSupport) {
      return 'Expand to see the support sentences.'
    }
    return 'Expand to see the article overview.'
  }

  const getRankingExplanationKey = (article: Article): string => String(article.id)

  const getRankingExplanationState = (article: Article): {
    loading: boolean
    error: string | null
    explanation: string | null
  } => svdRankingExplanations[getRankingExplanationKey(article)] ?? {
    loading: false,
    error: null,
    explanation: null,
  }

  const getSvdDimensionLabelState = (article: Article): {
    loading: boolean
    error: string | null
    labels: SvdDimensionLabelMap
  } => svdDimensionLabelStates[getRankingExplanationKey(article)] ?? {
    loading: false,
    error: null,
    labels: {},
  }

  const getRankingExplanationQueryText = (): string => {
    if (inputMode === 'stance') {
      const trimmedTopic = topic.trim()
      const trimmedOpinion = opinion.trim()
      if (trimmedTopic || trimmedOpinion) {
        return [
          trimmedTopic ? `Topic: ${trimmedTopic}` : null,
          trimmedOpinion ? `Opinion: ${trimmedOpinion}` : null,
        ].filter(Boolean).join('\n')
      }
      return ''
    }
    return essayPreparedText.trim() || searchTerm.trim()
  }

  const setRankingExplanationState = (
    article: Article,
    nextState: Partial<{ loading: boolean; error: string | null; explanation: string | null }>,
  ): void => {
    const key = getRankingExplanationKey(article)
    setSvdRankingExplanations((prev) => ({
      ...prev,
      [key]: {
        ...(prev[key] ?? {
          loading: false,
          error: null,
          explanation: null,
        }),
        ...nextState,
      },
    }))
  }

  const setSvdDimensionLabelState = (
    article: Article,
    nextState: Partial<{ loading: boolean; error: string | null; labels: SvdDimensionLabelMap }>,
  ): void => {
    const key = getRankingExplanationKey(article)
    setSvdDimensionLabelStates((prev) => ({
      ...prev,
      [key]: {
        ...(prev[key] ?? {
          loading: false,
          error: null,
          labels: {},
        }),
        ...nextState,
      },
    }))
  }

  const requestSvdDimensionLabels = async (article: Article): Promise<void> => {
    const currentState = getSvdDimensionLabelState(article)
    if (currentState.loading || Object.keys(currentState.labels).length > 0) return

    const dimensions = getSvdDimensionsForLabeling(article)
    if (dimensions.length === 0) return

    if (useLlm !== true) {
      setSvdDimensionLabelState(article, {
        error: 'LLM labels are turned off in the backend config.',
      })
      return
    }

    setSvdDimensionLabelState(article, {
      loading: true,
      error: null,
    })

    try {
      const response = await fetch('/api/llm/svd-dimension-labels', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dimensions,
        }),
      })

      const data = await readApiJson<{ labels?: Array<{ dimension_index?: number; label?: string }> }>(response)
      const nextLabels: SvdDimensionLabelMap = {}
      for (const item of data.labels ?? []) {
        if (typeof item.dimension_index !== 'number') continue
        const label = String(item.label || '').trim()
        if (label) {
          nextLabels[item.dimension_index] = label
        }
      }
      if (Object.keys(nextLabels).length === 0) {
        throw new Error('No concept labels were returned.')
      }

      setSvdDimensionLabelState(article, {
        loading: false,
        error: null,
        labels: nextLabels,
      })
    } catch (fetchError) {
      setSvdDimensionLabelState(article, {
        loading: false,
        error: fetchError instanceof Error ? fetchError.message : 'SVD concept labeling failed.',
      })
    }
  }

  const handleSvdDisclosureToggle = (
    article: Article,
    event: SyntheticEvent<HTMLDetailsElement>,
  ): void => {
    if (!event.currentTarget.open) return
    void requestSvdDimensionLabels(article)
  }

  const handleExplainRanking = async (article: Article, rank: number): Promise<void> => {
    setRankingExplanationState(article, {
      loading: true,
      error: null,
      explanation: null,
    })

    const queryText = getRankingExplanationQueryText()
    if (!queryText) {
      setRankingExplanationState(article, {
        loading: false,
        error: 'No searchable query is available for this result.',
      })
      return
    }

    try {
      const response = await fetch('/api/llm/explain-ranking', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: queryText,
          article,
          position: rank,
          query_svd_dimensions: querySvdDimensions,
        }),
      })

      const data = await readApiJson<{ explanation?: string }>(response)
      if (!data || typeof data.explanation !== 'string' || data.explanation.trim() === '') {
        throw new Error('Invalid response from the ranking explanation API.')
      }

      setRankingExplanationState(article, {
        loading: false,
        explanation: data.explanation.trim(),
      })
    } catch (fetchError) {
      setRankingExplanationState(article, {
        loading: false,
        error: fetchError instanceof Error ? fetchError.message : 'Ranking explanation failed.',
      })
    }
  }

  const getSvdExplainabilityHint = (article: Article): string => {
    const queryChartCount = article.svd_query_chart_dimensions?.length ?? 0
    const sharedDimensionCount = article.svd_chart_dimensions?.length ?? 0
    const articleDimensionCount = article.svd_dimensions?.length ?? 0
    if (queryChartCount > 0 && sharedDimensionCount > 0 && articleDimensionCount > 0) {
      return `Expand to compare this article against the query concepts, the corpus concepts, and its top ${articleDimensionCount} signed concepts.`
    }
    if (queryChartCount > 0 && articleDimensionCount > 0) {
      return `Expand to compare this article against the query concepts and inspect its top ${articleDimensionCount} signed concepts.`
    }
    if (sharedDimensionCount > 0 && articleDimensionCount > 0) {
      return `Expand to compare this article on ${sharedDimensionCount} corpus concepts and inspect its top ${articleDimensionCount} signed concepts.`
    }
    if (queryChartCount > 0) {
      return `Expand to compare this article against the query's top ${queryChartCount} concepts.`
    }
    if (sharedDimensionCount > 0) {
      return `Expand to compare this article on ${sharedDimensionCount} shared corpus concepts.`
    }
    if (articleDimensionCount > 0) {
      return `Expand to inspect this article's top ${articleDimensionCount} signed latent concepts.`
    }
    return 'Expand to inspect the latent concepts behind this match.'
  }

  const resultsDescription = useMemo(() => {
    if (loading) {
      return `Ranking Guardian opinion pieces${yearRangeSummary} with your current search settings.`
    }

    if (error) {
      return 'Something interrupted the search. Adjust the prompt above or try again.'
    }

    if (!hasSubmittedSearch) {
      return inputMode === 'stance'
        ? 'Submit a topic and stance above to open a page of supporting, opposing, and neutral perspectives.'
        : (isLlmAgreementSelected
          ? 'Paste an essay, and your ranked Guardian matches will appear here.'
          : 'Paste an essay, choose or write its thesis, and your ranked Guardian matches will appear here.')
    }

    if (articles.length === 0) {
      if (emptyResultsMessage) {
        return emptyResultsMessage
      }
      return (
        `No matching articles came back${yearRangeSummary} this time. `
        + 'Try broadening the topic, sharpening the claim, or widening the year range.'
      )
    }

    const hiddenCopy = llmIrrelevantArticles.length > 0
      ? ` ${llmIrrelevantArticles.length} ${llmIrrelevantArticles.length === 1 ? 'article is' : 'articles are'} hidden as unrelated.`
      : ''
    const feedbackCopy = topicFeedbackIrrelevantArticles.length > 0
      ? ` ${topicFeedbackIrrelevantArticles.length} ${topicFeedbackIrrelevantArticles.length === 1 ? 'article is' : 'articles are'} collapsed as not relevant.`
      : ''
    const appliedFeedbackCopy = appliedTopicFeedbackArticleIds.length > 0
      ? ` Topic feedback is active for ${appliedTopicFeedbackArticleIds.length} ${appliedTopicFeedbackArticleIds.length === 1 ? 'article' : 'articles'}.`
      : ''
    return (
      `${activeVisibleArticles.length} Guardian opinion ${activeVisibleArticles.length === 1 ? 'piece' : 'pieces'}`
      + `${yearRangeSummary} ranked with your current search settings.${hiddenCopy}${feedbackCopy}${appliedFeedbackCopy}`
    )
  }, [
    activeVisibleArticles.length,
    appliedTopicFeedbackArticleIds.length,
    articles.length,
    emptyResultsMessage,
    error,
    hasSubmittedSearch,
    inputMode,
    isLlmAgreementSelected,
    llmIrrelevantArticles.length,
    loading,
    topicFeedbackIrrelevantArticles.length,
    yearRangeSummary,
  ])

  return (
    <div className="experience-shell">
      <div
        className={[
          'intro-screen',
          'landing-section',
          isSearchStageVisible ? 'search-active' : '',
          inputMode === 'essay' ? 'essay-mode' : 'stance-mode',
        ].filter(Boolean).join(' ')}
      >
        <div className={`intro-shell ${isSearchStageVisible ? 'search-active' : ''}`}>
          <div className={`search-chrome ${isSearchStageVisible ? 'visible' : ''}`}>
            <div className="top-nav" aria-label="Page navigation">
              <div className="top-nav-spacer" aria-hidden="true" />
              <div className="top-nav-actions">
                <button
                  type="button"
                  className="top-nav-button"
                  onClick={returnToLanding}
                >
                  Home
                </button>
                <button
                  type="button"
                  className="top-nav-button"
                  onClick={() => openAbout()}
                >
                  About
                </button>
              </div>
            </div>

            <div className="search-header-block">
              <div className="hero-copy">
                <h1>hear! hear!</h1>
                <h2>Find your voice in Guardian opinion articles</h2>
              </div>

              <div className="mode-switch" role="tablist" aria-label="Search mode">
                <button
                  type="button"
                  className={`mode-pill ${inputMode === 'stance' ? 'active' : ''}`}
                  onClick={() => setInputMode('stance')}
                >
                  Topic + Stance
                </button>
                <button
                  type="button"
                  className={`mode-pill ${inputMode === 'essay' ? 'active' : ''}`}
                  onClick={() => setInputMode('essay')}
                >
                  Essay
                </button>
              </div>

              <div className="top-search-mode-toolbar" aria-label="Search scoring modes">
                <div className="top-search-mode-group">
                  <div className="top-search-mode-heading">
                    <span>Topic Relevance Search Mode</span>
                    <button
                      type="button"
                      className="top-search-mode-help"
                      onClick={() => openSettingsAt('topic-relevance')}
                      aria-label="Open topic relevance mode settings"
                    >
                      ?
                    </button>
                  </div>
                  <div className="top-search-mode-segments" role="group" aria-label="Topic relevance search mode">
                    <button
                      type="button"
                      className={`top-search-mode-segment ${retrievalModel === 'tfidf' ? 'active' : ''}`}
                      aria-pressed={retrievalModel === 'tfidf'}
                      onClick={() => handleTopicSearchModeChange('tfidf')}
                      disabled={!canUseTfidf}
                    >
                      Lexical
                    </button>
                    <button
                      type="button"
                      className={`top-search-mode-segment ${retrievalModel === 'svd' ? 'active' : ''}`}
                      aria-pressed={retrievalModel === 'svd'}
                      onClick={() => handleTopicSearchModeChange('svd')}
                      disabled={!canUseSvd}
                    >
                      Semantic
                    </button>
                  </div>
                </div>

                <div className="top-search-mode-group">
                  <div className="top-search-mode-heading">
                    <span>Stance Agreement Search Mode</span>
                    <button
                      type="button"
                      className="top-search-mode-help"
                      onClick={() => openSettingsAt('agreement-scorer')}
                      aria-label="Open stance agreement mode settings"
                    >
                      ?
                    </button>
                  </div>
                  <div className="top-search-mode-segments" role="group" aria-label="Stance agreement search mode">
                    <button
                      type="button"
                      className={`top-search-mode-segment ${effectiveStanceMethod === 'nli' ? 'active' : ''}`}
                      aria-pressed={effectiveStanceMethod === 'nli'}
                      onClick={() => handleAgreementSearchModeChange('nli')}
                      disabled={!canUseNliAgreement}
                    >
                      NLI
                    </button>
                    <button
                      type="button"
                      className={`top-search-mode-segment ${effectiveStanceMethod === 'llm' ? 'active' : ''}`}
                      aria-pressed={effectiveStanceMethod === 'llm'}
                      onClick={() => handleAgreementSearchModeChange('llm')}
                      disabled={!canUseLlmAgreement}
                    >
                      LLM
                    </button>
                  </div>
                </div>
              </div>

              {inputMode === 'essay' && isSearchStageVisible && shouldUseEssayThesisStep && (
                <div
                  className="essay-progress-shell"
                  aria-label={`Essay workflow step ${essayWorkflowStep} of 2`}
                >
                  <div className="essay-progress-bar" aria-hidden="true">
                    <span className={`essay-progress-segment ${essayWorkflowStep === 1 ? 'active' : 'complete'}`} />
                    <span className={`essay-progress-segment ${essayWorkflowStep === 2 ? 'active' : (isEssayStepTwoAvailable ? 'complete' : '')}`} />
                  </div>

                  <div className="essay-progress-steps">
                    <button
                      type="button"
                      className={`essay-progress-step ${essayWorkflowStep === 1 ? 'active' : 'complete'}`}
                      onClick={() => setEssayActiveStep(1)}
                      aria-current={essayWorkflowStep === 1 ? 'step' : undefined}
                    >
                      <span className="essay-progress-number">1</span>
                      <div className="essay-progress-copy">
                        <span className="essay-progress-title">Add your essay</span>
                        <span className="essay-progress-note">Paste text or import from a PDF.</span>
                      </div>
                    </button>

                    <button
                      type="button"
                      className={`essay-progress-step ${essayWorkflowStep === 2
                        ? 'active'
                        : (isEssayStepTwoAvailable ? 'available' : 'disabled')
                        }`}
                      onClick={() => {
                        if (isEssayStepTwoAvailable) {
                          setEssayActiveStep(2)
                        }
                      }}
                      disabled={!isEssayStepTwoAvailable}
                      aria-current={essayWorkflowStep === 2 ? 'step' : undefined}
                    >
                      <span className="essay-progress-number">2</span>
                      <div className="essay-progress-copy">
                        <span className="essay-progress-title">Choose the thesis</span>
                        <span className="essay-progress-note">
                          {isEssayStepTwoAvailable
                            ? 'Pick a sentence or write your own thesis.'
                            : 'Extract thesis options to unlock this step.'}
                        </span>
                      </div>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className={`landing-prompt-shell ${(inputMode === 'essay' && isSearchStageVisible) ? 'hidden' : ''}`}>
            <div
              className={`intro-line visible ${introStage > 0 ? 'done' : ''}`}
              role="text"
              aria-label={`Regarding ${typedTopic || trimmedTopic || introTopicSequence[introTopicSequence.length - 1]}`}
            >
              <span className="intro-line-label">Regarding</span>
              {isSearchStageVisible && inputMode === 'stance' ? (
                <span className="intro-inline-form-slot">
                  <span className="intro-inline-input-wrap">
                    <input
                      type="text"
                      value={topic}
                      onChange={(event) => setTopic(event.target.value)}
                      placeholder="type your topic"
                      aria-label="Topic"
                    />
                  </span>
                </span>
              ) : (
                <span className="intro-typewriter-slot" aria-hidden="true">
                  <span className="intro-typewriter-value">{typedTopic || '\u00A0'}</span>
                </span>
              )}
            </div>

            <div
              className={[
                'intro-line',
                introStage >= 1 ? 'visible' : '',
                introStage > 1 ? 'done' : '',
              ].filter(Boolean).join(' ')}
              role="text"
              aria-label={`I believe ${typedClaim || trimmedOpinion || introClaimSequence[introClaimSequence.length - 1]}`}
            >
              <span className="intro-line-label">I believe</span>
              {isSearchStageVisible && inputMode === 'stance' ? (
                <span className="intro-inline-form-slot">
                  <span className="intro-inline-input-wrap">
                    <input
                      type="text"
                      value={opinion}
                      onChange={(event) => setOpinion(event.target.value)}
                      placeholder="type your stance"
                      aria-label="Opinion"
                    />
                  </span>
                </span>
              ) : (
                <span className="intro-typewriter-slot" aria-hidden="true">
                  <span className="intro-typewriter-value">{typedClaim || '\u00A0'}</span>
                </span>
              )}
            </div>
          </div>

          {inputMode === 'essay' && isSearchStageVisible && (
            <div className="essay-panel landing-essay-panel">
              {essayWorkflowStep === 1 && (
                <>

                  <div className="essay-intake-panel">
                    <label className="essay-intake-line essay-intake-text-line">
                      <span className="essay-intake-label">Essay</span>
                      <span className="essay-intake-field essay-intake-text-field">
                        <textarea
                          id="search-input"
                          placeholder="Paste an essay, paper, or op-ed..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          rows={6}
                          aria-label="Essay or search phrase"
                        />
                      </span>
                    </label>

                    <div className="essay-intake-tools">
                      <p className="essay-upload-hint">
                        {importedPdfName
                          ? `Text imported from ${importedPdfName}. You can keep editing it here before ${isLlmAgreementSelected ? 'searching' : 'extracting thesis options'}.`
                          : 'Have a PDF already? Upload it and we’ll drop the extracted text into this editor so you can revise it.'}
                      </p>
                      <label
                        htmlFor="pdf-upload"
                        className={`essay-upload-trigger ${isImportingPdf ? 'disabled' : ''}`}
                        aria-disabled={isImportingPdf}
                      >
                        {isImportingPdf ? 'reading PDF...' : (importedPdfName ? 'replace with another PDF' : 'Import from PDF')}
                      </label>
                      <input
                        id="pdf-upload"
                        className="pdf-upload-input"
                        type="file"
                        accept="application/pdf"
                        onChange={handleImportPdf}
                        disabled={isImportingPdf}
                      />
                    </div>
                  </div>

                  <div className="essay-actions essay-intake-actions">
                    <button
                      type="button"
                      className="primary-action-button"
                      onClick={isLlmAgreementSelected ? handleSubmitEssayFromDraft : handleAnalyzeEssay}
                      disabled={!canAnalyzeEssay || loading}
                    >
                      {isLlmAgreementSelected
                        ? (loading ? 'Searching...' : 'Search')
                        : ((loading && essayWorkflowStep === 1) ? 'Extracting thesis...' : 'Extract thesis options')}
                    </button>
                  </div>
                </>
              )}

              {isEssayStepTwoAvailable && essayWorkflowStep === 2 && (
                <div className="essay-candidate-panel">

                  <div className="essay-option-strip">
                    <div className="essay-option-strip-header">
                      <div>
                        <p className="essay-option-strip-title">Thesis options</p>
                        <p className="essay-option-strip-note">
                          {essayCandidates.length > 0
                            ? 'Select a suggestion or enter your own thesis.'
                            : 'Type your own thesis to continue.'}
                        </p>
                      </div>
                      <div className="essay-option-strip-controls">
                        {essayCandidates.length > 1 && (
                          <div className="essay-option-arrow-group">
                            <button
                              type="button"
                              className="essay-option-arrow"
                              onClick={() => scrollEssayOptions('left')}
                              aria-label="Scroll thesis options left"
                            >
                              {'<'}
                            </button>
                            <button
                              type="button"
                              className="essay-option-arrow"
                              onClick={() => scrollEssayOptions('right')}
                              aria-label="Scroll thesis options right"
                            >
                              {'>'}
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                    {essayCandidates.length > 0 ? (
                      <div className="essay-candidate-grid" ref={essayOptionsRef}>
                        {essayCandidates.map((candidate, index) => {
                          const isSelected = (
                            essayThesisMode === 'candidate' &&
                            candidate.sentence_id === selectedEssayCandidateId
                          )
                          return (
                            <button
                              key={candidate.sentence_id}
                              type="button"
                              className={`candidate-card ${isSelected ? 'selected' : ''}`}
                              onClick={() => {
                                setEssayThesisMode('candidate')
                                setSelectedEssayCandidateId(candidate.sentence_id)
                              }}
                              style={{ animationDelay: `${index * 70}ms` }}
                            >
                              <div className="candidate-card-header">
                                <span className="candidate-rank">Option {index + 1}</span>
                                {isSelected && <span className="candidate-selected-badge">Selected</span>}
                              </div>
                              <p className="candidate-sentence">{candidate.sentence}</p>
                            </button>
                          )
                        })}
                      </div>
                    ) : (
                      <p className="essay-option-empty">
                        We couldn&apos;t identify a clear thesis sentence from the draft, but you can
                        still enter the statement you want us to use.
                      </p>
                    )}

                    <div
                      className={`essay-custom-thesis-card ${isUsingCustomEssayThesis ? 'selected' : ''}`}
                      onClick={() => setEssayThesisMode('custom')}
                    >
                      <div className="candidate-card-header">
                        <span className="candidate-rank">Your thesis</span>
                        {isUsingCustomEssayThesis && (
                          <span className="candidate-selected-badge">Selected</span>
                        )}
                      </div>
                      <label className="essay-custom-thesis-field">
                        <span className="sr-only">Enter your thesis statement</span>
                        <textarea
                          value={essayCustomThesis}
                          onChange={(event) => {
                            setEssayCustomThesis(event.target.value)
                            setEssayThesisMode('custom')
                          }}
                          onFocus={() => setEssayThesisMode('custom')}
                          rows={3}
                          placeholder="Type the sentence you consider to be your thesis statement..."
                          aria-label="Custom thesis statement"
                        />
                      </label>
                    </div>
                  </div>

                  <div className="essay-submit-panel">
                    <div>
                      <p className="essay-submit-eyebrow">Selected thesis</p>
                      <p className="essay-submit-copy">
                        {resolvedEssayThesis || 'Choose a sentence above or type your own thesis below.'}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="primary-action-button"
                      onClick={handleSubmitEssay}
                      disabled={!canSubmitEssay || loading}
                    >
                      {(loading && essayWorkflowStep === 2) ? 'Searching...' : 'Search'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          <div className={`stance-actions landing-stance-actions ${isSearchStageVisible ? 'visible' : ''}`}>
            {inputMode === 'stance' && (
              <button
                type="button"
                className="primary-action-button"
                onClick={() => void handleSubmitStance()}
                disabled={!canSearchStance || loading}
              >
                Search
              </button>
            )}
            <button
              type="button"
              className="utility-pill"
              onClick={() => setIsFilterOpen(true)}
            >
              {`Filter (${activeYearRangeLabel})`}
            </button>
            <button
              type="button"
              className="utility-pill"
              onClick={openSettings}
            >
              Settings
            </button>
          </div>

          <div className={`intro-cta ${introStage >= 2 ? 'visible' : ''} ${isSearchStageVisible ? 'hidden' : ''}`}>
            <button
              type="button"
              className="intro-scroll-cue"
              onClick={() => activateSearchStage()}
              aria-label="Reveal search"
            >
              <span className="intro-cue-text">find your voice</span>
              <span className="intro-cue-arrow" aria-hidden="true">↓</span>
            </button>
          </div>
        </div>
      </div>

      {hasSubmittedSearch && (
        <div
          ref={resultsSectionRef}
          className="results-paper-section visible"
        >
          {!loading && !error && querySvdCorpusChartDimensions.length > 0 && (
            <div className="results-query-concepts-shell">
              <div className="results-query-concepts-wrap">
                <details className="content-disclosure results-query-concepts-disclosure">
                  <summary className="content-disclosure-summary">
                    <span className="content-disclosure-copy">
                      <span className="content-disclosure-title">Query concepts</span>
                      <span className="content-disclosure-hint">
                        Expand to inspect how your query loads onto the top 10 corpus concepts.
                      </span>
                    </span>
                    <span className="content-disclosure-status" aria-hidden="true" />
                  </summary>

                  <div className="results-query-concepts">
                    <div className="results-query-concepts-copy">
                      <p className="results-query-concepts-eyebrow">Query concepts</p>
                      <h3>Query on top 10 corpus concepts</h3>
                      <p>
                        This radar chart uses the shared top 10 corpus-level concepts and shows how strongly your query loads on those same axes.
                      </p>
                    </div>
                    <SvdRadarChart
                      dimensions={querySvdCorpusChartDimensions}
                      primaryLabel="Query"
                      primaryRole="query"
                      maxMagnitude={sharedCorpusRadarMaxMagnitude}
                      ariaLabel="Radar chart of your query across the top 10 corpus-level SVD concepts"
                      caption="These axes are fixed to the first 10 corpus-level concepts, so this gives a corpus-frame view of the query before you compare it with individual articles."
                      emptyCopy="No corpus-level SVD concept view is available for this query yet."
                    />
                    {querySvdDimensions.length > 0 && (
                      <div className="results-query-concepts-bar-block">
                        <div className="results-query-concepts-copy">
                          <p className="results-query-concepts-eyebrow">Query top concepts</p>
                          <h3>Top 10 concepts from your query</h3>
                          <p>
                            This bar chart keeps the query&apos;s own top 10 concepts and shows whether each concept loads positively or negatively.
                          </p>
                        </div>
                        <SvdConceptBarChart
                          dimensions={querySvdDimensions}
                          primaryLabel="Query"
                          primaryRole="query"
                          maxMagnitude={queryConceptBarMaxMagnitude}
                          emptyCopy="No query-specific SVD concepts are available yet."
                        />
                      </div>
                    )}
                  </div>
                </details>
              </div>
            </div>
          )}

          <div className="results-paper">
            <div className="results-paper-header">
              <p className="results-paper-eyebrow">Results</p>
              <h2>Guardian opinion matches</h2>
              <p className="results-paper-copy">{resultsDescription}</p>
            </div>

            {!loading && !error && pendingTopicFeedbackCount > 0 && (
              <div className="topic-feedback-refresh-bar" role="status" aria-live="polite">
                <div className="topic-feedback-refresh-copy">
                  <p className="topic-feedback-refresh-eyebrow">Topic feedback</p>
                  <p>
                    {`${pendingTopicFeedbackCount} ${pendingTopicFeedbackCount === 1 ? 'article is' : 'articles are'} marked not relevant.`}
                  </p>
                </div>
                <button
                  type="button"
                  className="topic-feedback-refresh-button"
                  onClick={handleRefreshTopicFeedback}
                  disabled={loading}
                >
                  Refresh
                </button>
              </div>
            )}

            {loading && (
              <div className="results-thinking-card" role="status" aria-live="polite">
                <p className="results-thinking-label">Thinking</p>
                <div className="results-thinking-dots" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}

            {!loading && error && (
              <div className="results-empty-card error">
                <p>{error}</p>
              </div>
            )}

            {!loading && !error && articles.length > 0 && (
              <div id="answer-box">
                {visibleArticles.map((article) => {
                  const articleTooltipBase = String(article.id).replace(/[^a-zA-Z0-9_-]/g, '-')
                  const articleRecencyWeight = article.recency_weight ?? recencyWeight
                  const svdDimensionLabelState = getSvdDimensionLabelState(article)
                  const isMarkedNotRelevant = isTopicFeedbackIrrelevantArticle(article)
                  const activeArticleRank = activeVisibleArticles.findIndex(activeArticle => (
                    getArticleIdKey(activeArticle) === getArticleIdKey(article)
                  )) + 1

                  if (isMarkedNotRelevant) {
                    return (
                      <article key={article.id} className="article-item topic-feedback-collapsed-article">
                        <div className="topic-feedback-collapsed-copy">
                          <p className="topic-feedback-state">Marked not relevant</p>
                          <h3 className="article-title">
                            <a href={article.url} target="_blank" rel="noreferrer">{article.title}</a>
                          </h3>
                          <p className="article-meta">
                            {article.author_display || article.author_raw || 'Unknown author'} | {formatDate(article.date)}
                          </p>
                        </div>
                        <button
                          type="button"
                          className="topic-feedback-undo-button"
                          onClick={() => handleUndoTopicIrrelevant(article)}
                        >
                          Undo
                        </button>
                      </article>
                    )
                  }

                  return (
                    <article key={article.id} className="article-item">
                      <p className="article-meta">
                        {article.author_display || article.author_raw || 'Unknown author'} | {formatDate(article.date)}
                      </p>

                      <h3 className="article-title">
                        <a href={article.url} target="_blank" rel="noreferrer">{article.title}</a>
                      </h3>

                      <p className="article-summary">{article.summary}</p>

                      {article.central_claim_summary && (
                        <div className="claim-band">
                          <span className="claim-band-label">Author&apos;s claim</span>
                          <p>{article.central_claim_summary}</p>
                        </div>
                      )}

                      {showScoreGrid(article) && (
                        <div className="match-panel">
                          <div className="match-panel-header">
                            {!canExplainRanking && (
                              <div>
                                <div className="match-panel-eyebrow">Why it ranked here</div>
                                <div className="match-panel-summary">{getMatchSummary(article)}</div>
                              </div>
                            )}



                            {canExplainRanking && (
                              <div className="match-panel-actions">
                                <p></p>
                                <button
                                  type="button"
                                  className="explain-ranking-button"
                                  onClick={() => handleExplainRanking(article, activeArticleRank)}
                                  disabled={getRankingExplanationState(article).loading}
                                >
                                  {getRankingExplanationState(article).loading ? 'Explaining…' : 'Explain ranking'}
                                </button>
                              </div>
                            )}
                          </div>

                          {canExplainRanking && (
                            <div className="ranking-explanation-shell" aria-live="polite">
                              {getRankingExplanationState(article).error && (
                                <div className="ranking-explanation-error">
                                  {getRankingExplanationState(article).error}
                                </div>
                              )}
                              {getRankingExplanationState(article).explanation && (
                                <div className="ranking-explanation-text">
                                  <p>{getRankingExplanationState(article).explanation}</p>
                                </div>
                              )}
                            </div>
                          )}

                          <div className="match-score-stack">
                            <div className="match-metric-card overall">
                              <div className="match-metric-header">
                                <div className="match-metric-heading">
                                  <div className="match-metric-label">Overall match</div>
                                  {renderMetricInfo(
                                    'Overall match',
                                    `${articleTooltipBase}-overall-help`,
                                    articleRecencyWeight > 0
                                      ? 'Final ranking after combining topic match, agreement, and recency.'
                                      : 'Final ranking after combining topic match and agreement.',
                                  )}
                                </div>
                                <div className="match-metric-value">{formatPercent(article.combined_score)}</div>
                              </div>
                              <div className="match-meter" aria-hidden="true">
                                <span
                                  className="match-meter-fill overall"
                                  style={{ width: getMeterWidth(article.combined_score) }}
                                />
                              </div>
                            </div>

                            <div className="match-input-grid">
                              <div className="match-metric-card source">
                                <div className="match-metric-header">
                                  <div className="match-metric-heading">
                                    <div className="match-metric-label">Topic match</div>
                                    {renderMetricInfo(
                                      'Topic match',
                                      `${articleTooltipBase}-topic-help`,
                                      'How closely the article matches your subject in the first text pass. This can use either raw retrieval similarity or within-result normalization from Settings.',
                                    )}
                                  </div>
                                  <div className="match-metric-value">{formatPercent(getTopicMatchScore(article))}</div>
                                </div>
                                <div className="match-meter" aria-hidden="true">
                                  <span
                                    className="match-meter-fill topic"
                                    style={{ width: getMeterWidth(getTopicMatchScore(article)) }}
                                  />
                                </div>
                              </div>

                              {articleRecencyWeight > 0 && (
                                <div className="match-metric-card source">
                                  <div className="match-metric-header">
                                    <div className="match-metric-heading">
                                      <div className="match-metric-label">Recency</div>
                                      {renderMetricInfo(
                                        'Recency',
                                        `${articleTooltipBase}-recency-help`,
                                        'How strongly the article benefits from being more recently published.',
                                      )}
                                    </div>
                                    <div className="match-metric-value">{formatPercent(article.recency_score_normalized)}</div>
                                  </div>
                                  <div className="match-meter" aria-hidden="true">
                                    <span
                                      className="match-meter-fill recency"
                                      style={{ width: getMeterWidth(article.recency_score_normalized) }}
                                    />
                                  </div>
                                </div>
                              )}

                              <div className="agreement-branch" tabIndex={0}>
                                <div className="match-metric-card source agreement">
                                  <div className="match-metric-header">
                                    <div className="match-metric-heading">
                                      <div className="match-metric-label">Agreement</div>
                                      {renderMetricInfo(
                                        'Agreement',
                                        `${articleTooltipBase}-agreement-help`,
                                        'How closely the article\'s main claim seems to align with your view.',
                                      )}
                                    </div>
                                    <div className="match-metric-value">{formatPercent(article.stance_score_normalized)}</div>
                                  </div>
                                  <div className="match-meter" aria-hidden="true">
                                    <span
                                      className="match-meter-fill stance"
                                      style={{ width: getMeterWidth(article.stance_score_normalized) }}
                                    />
                                  </div>
                                </div>

                                {hasStanceSignals(article) && (
                                  <div className="agreement-hover-panel">
                                    <div className="agreement-hover-title">Agreement is based on</div>
                                    <div className="stance-read-panel">
                                      <div className="stance-read-grid">
                                        <div className="stance-read-row">
                                          <div className="stance-read-label">Supports your view</div>
                                          <div className="stance-read-bar" aria-hidden="true">
                                            <span
                                              className="stance-read-fill support"
                                              style={{ width: getMeterWidth(article.stance_entailment_prob) }}
                                            />
                                          </div>
                                          <div className="stance-read-value">{formatPercent(article.stance_entailment_prob)}</div>
                                        </div>

                                        <div className="stance-read-row">
                                          <div className="stance-read-label">Mixed or unclear</div>
                                          <div className="stance-read-bar" aria-hidden="true">
                                            <span
                                              className="stance-read-fill neutral"
                                              style={{ width: getMeterWidth(article.stance_neutral_prob) }}
                                            />
                                          </div>
                                          <div className="stance-read-value">{formatPercent(article.stance_neutral_prob)}</div>
                                        </div>

                                        <div className="stance-read-row">
                                          <div className="stance-read-label">Pushes back</div>
                                          <div className="stance-read-bar" aria-hidden="true">
                                            <span
                                              className="stance-read-fill contradict"
                                              style={{ width: getMeterWidth(article.stance_contradiction_prob) }}
                                            />
                                          </div>
                                          <div className="stance-read-value">{formatPercent(article.stance_contradiction_prob)}</div>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {hasLlmRelevantParagraphs(article) && (
                        <details className="content-disclosure paragraph-evidence-disclosure">
                          <summary className="content-disclosure-summary">
                            <span className="content-disclosure-copy">
                              <span className="content-disclosure-title">{`Relevant ${getLlmChunkNoun(article)}`}</span>
                              <span className="content-disclosure-hint">{getParagraphEvidenceHint(article)}</span>
                            </span>
                            <span className="content-disclosure-status" aria-hidden="true" />
                          </summary>

                          <div className="paragraph-evidence-list">
                            {(article.llm_relevant_paragraphs ?? []).map((paragraph, index) => (
                              <div key={paragraphKey(article, paragraph, index)} className="paragraph-evidence-item">
                                <div className="paragraph-evidence-header">
                                  <span>{`${getLlmChunkNoun(article, false)} ${(paragraph.paragraph_index ?? index) + 1}`}</span>
                                  <strong>{formatPercent(paragraph.agreement_score)}</strong>
                                </div>
                                <p>{paragraph.text}</p>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}

                      {hasSvdExplainability(article) && (
                        <details
                          className="content-disclosure svd-explainability-disclosure"
                          onToggle={(event) => handleSvdDisclosureToggle(article, event)}
                        >
                          <summary className="content-disclosure-summary">
                            <span className="content-disclosure-copy">
                              <span className="content-disclosure-title">Latent concepts</span>
                              <span className="content-disclosure-hint">{getSvdExplainabilityHint(article)}</span>
                            </span>
                            <span className="content-disclosure-status" aria-hidden="true" />
                          </summary>

                          <div className="svd-explainability-panel">
                            {(svdDimensionLabelState.loading || svdDimensionLabelState.error) && (
                              <p
                                className={`svd-section-copy ${svdDimensionLabelState.loading ? 'svd-label-loading' : ''}`}
                                data-text={svdDimensionLabelState.loading ? 'Labeling latent concepts with the LLM...' : undefined}
                              >
                                {svdDimensionLabelState.loading
                                  ? 'Labeling latent concepts with the LLM...'
                                  : svdDimensionLabelState.error}
                              </p>
                            )}

                            {Array.isArray(article.svd_query_chart_dimensions) && article.svd_query_chart_dimensions.length > 0 && (
                              <div className="svd-chart-section">
                                <div className="svd-section-copy-block">
                                  <div className="svd-section-title">Query top 10 concepts</div>
                                  <p className="svd-section-copy">
                                    This radar uses the 10 concepts most activated by the query, then shows how strongly this article aligns with each one.
                                  </p>
                                </div>
                                <SvdRadarChart
                                  dimensions={article.svd_query_chart_dimensions}
                                  comparisonDimensions={querySvdDimensions}
                                  dimensionLabels={svdDimensionLabelState.labels}
                                  primaryLabel="Article"
                                  comparisonLabel="Query"
                                  maxMagnitude={queryTopRadarMaxMagnitude}
                                  ariaLabel="Radar chart of this article measured on the query's top 10 SVD concepts"
                                  caption="The axes are the 10 concepts most strongly activated by your query. Article and query lines show their loadings on those same concepts."
                                  emptyCopy="No query-anchored SVD concepts are available for this article yet."
                                />
                              </div>
                            )}

                            {Array.isArray(article.svd_chart_dimensions) && article.svd_chart_dimensions.length > 0 && (
                              <div className="svd-chart-section">
                                <div className="svd-section-copy-block">
                                  <div className="svd-section-title">Shared top 10 corpus concepts</div>
                                  <p className="svd-section-copy">
                                    This radar plots every article against the same 10 broad corpus concepts, so differences in shape reflect differences in topic emphasis.
                                  </p>
                                </div>
                                <SvdRadarChart
                                  dimensions={article.svd_chart_dimensions}
                                  comparisonDimensions={querySvdCorpusChartDimensions}
                                  dimensionLabels={svdDimensionLabelState.labels}
                                  primaryLabel="Article"
                                  comparisonLabel="Query"
                                  maxMagnitude={sharedCorpusRadarMaxMagnitude}
                                  ariaLabel="Radar chart of this article across the shared top 10 corpus-level SVD concepts"
                                  caption="Article and query lines use the same 10 broad concept axes. Points farther from the center show stronger connections; filled points are positive and hollow points are negative."
                                  emptyCopy="No shared corpus-level SVD concepts are available for this article yet."
                                />
                              </div>
                            )}

                            {(
                              Array.isArray(article.svd_dimensions) && article.svd_dimensions.length > 0
                            ) && (
                                <div className="svd-dimension-section">
                                  <div className="svd-section-copy-block">
                                    <div className="svd-section-title">Top concepts for this article</div>
                                    <p className="svd-section-copy">
                                      These bars show the article&apos;s top 10 concepts overall. Concepts extend left for negative loadings and right for positive loadings.
                                    </p>
                                  </div>

                                  <SvdConceptBarChart
                                    dimensions={article.svd_dimensions ?? []}
                                    comparisonDimensions={article.svd_article_query_dimensions ?? []}
                                    dimensionLabels={svdDimensionLabelState.labels}
                                    primaryLabel="Article"
                                    comparisonLabel="Query"
                                    maxMagnitude={articleConceptBarMaxMagnitude}
                                  />
                                </div>
                              )}
                          </div>
                        </details>
                      )}

                      {(article.thesis_sentence || (article.support_sentences && article.support_sentences.length > 0)) && (
                        <details className="content-disclosure">
                          <summary className="content-disclosure-summary">
                            <span className="content-disclosure-copy">
                              <span className="content-disclosure-title">Overview</span>
                              <span className="content-disclosure-hint">{getOverviewHint(article)}</span>
                            </span>
                            <span className="content-disclosure-status" aria-hidden="true" />
                          </summary>
                          <div className="sentence-block">
                            {article.thesis_sentence && (
                              <div className="overview-group">
                                <div className="overview-label">Thesis sentence</div>
                                <p>{article.thesis_sentence}</p>
                              </div>
                            )}

                            {article.support_sentences && article.support_sentences.length > 0 && (
                              <div className="overview-group">
                                <div className="overview-label">Support sentences</div>
                                <ul className="sentence-list">
                                  {article.support_sentences.map((sentence, index) => (
                                    <li key={`${article.id}-support-${index}`}>{sentence}</li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </details>
                      )}

                      {showScoreGrid(article) && !article.central_claim_summary && (
                        <p className="claim-missing">
                          No extracted central claim is available for this article yet, so it stayed in the ranking based on topic or essay relevance alone.
                        </p>
                      )}

                      {article.keywords && article.keywords.length > 0 && (
                        <div className="article-footer-row">
                          <div className="keyword-block">
                            <p>Keywords</p>
                            <div className="keyword-list">
                              {article.keywords.map((kw, index) => (
                                <span key={`${article.id}-keyword-${index}`} className="keyword-chip">{kw}</span>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="topic-feedback-action-row">
                        <button
                          type="button"
                          className="topic-feedback-button"
                          onClick={() => handleMarkTopicIrrelevant(article)}
                          aria-label={`Mark ${article.title} as not relevant`}
                        >
                          Mark as not relevant
                        </button>
                      </div>
                    </article>
                  )
                })}

                {llmIrrelevantArticles.length > 0 && (
                  <details className="content-disclosure irrelevant-results-disclosure">
                    <summary className="content-disclosure-summary">
                      <span className="content-disclosure-copy">
                        <span className="content-disclosure-title">Hidden as unrelated</span>
                        <span className="content-disclosure-hint">
                          {`${llmIrrelevantArticles.length} ${llmIrrelevantArticles.length === 1 ? 'article was' : 'articles were'} marked completely unrelated by the LLM. Expand to inspect.`}
                        </span>
                      </span>
                      <span className="content-disclosure-status" aria-hidden="true" />
                    </summary>

                    <div className="irrelevant-results-list">
                      {llmIrrelevantArticles.map((article) => (
                        <article key={`${article.id}-irrelevant`} className="article-item irrelevant-article-item">
                          <p className="article-meta">
                            {article.author_display || article.author_raw || 'Unknown author'} | {formatDate(article.date)}
                          </p>

                          <h3 className="article-title">
                            <a href={article.url} target="_blank" rel="noreferrer">{article.title}</a>
                          </h3>

                          <p className="article-summary">{article.summary}</p>

                          {article.central_claim_summary && (
                            <div className="claim-band">
                              <span className="claim-band-label">Author&apos;s claim</span>
                              <p>{article.central_claim_summary}</p>
                            </div>
                          )}
                        </article>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}

            {!loading && !error && articles.length === 0 && (
              <div className="results-empty-card searched">
                <p>
                  {emptyResultsMessage || `No matching articles were returned${yearRangeSummary}. Try broadening the topic, making the stance more explicit, or widening the year range.`}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {shouldShowEssayShortcut && <Chat onSearchTerm={handleEssaySearch} />}

      {isAboutOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setIsAboutOpen(false)}
          role="presentation"
        >
          <div
            className="modal-card about-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="about-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <h3 id="about-modal-title">About</h3>
              </div>
              <button
                type="button"
                className="modal-close"
                onClick={() => setIsAboutOpen(false)}
                aria-label="Close About popup"
              >
                Close
              </button>
            </div>
            <div className="about-tablist" role="tablist" aria-label="About search modes">
              <button
                type="button"
                role="tab"
                aria-selected={activeAboutTab === 'stance'}
                className={`about-tab ${activeAboutTab === 'stance' ? 'active' : ''}`}
                onClick={() => setActiveAboutTab('stance')}
              >
                Topic and Stance Search
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeAboutTab === 'essay'}
                className={`about-tab ${activeAboutTab === 'essay' ? 'active' : ''}`}
                onClick={() => setActiveAboutTab('essay')}
              >
                Essay-Guided Search
              </button>
            </div>
            <div className="modal-stage-list">
              {activeAboutTab === 'stance' ? (
                <>
                  <section className="about-section">
                    <p className="about-section-label">Stage 1</p>
                    <p className="modal-copy">
                      <strong>Stage 1: Topic relevance.</strong> We first identify articles that are
                      relevant to your topic. To do this, we compute the similarity between your
                      input and each Guardian article using the retrieval representation selected
                      in topic relevance mode: either Lexical TF-IDF term vectors or Semantic
                      truncated-SVD latent dimensions, both compared with cosine similarity. This
                      helps us find articles that discuss similar themes and keywords.
                    </p>
                  </section>
                  <section className="about-section">
                    <p className="about-section-label">Stage 2</p>
                    <p className="modal-copy">
                      <strong>Stage 2: Stance relevance.</strong> From the candidate articles identified
                      in Stage 1, we then rank them based on how they relate to your opinion.
                      The Agreement scorer in Settings can use either DeBERTa Natural Language
                      Inference (NLI) over each extracted article claim or Spark LLM scoring over
                      retrieved article context. The model estimates whether each article supports,
                      contradicts, or is neutral toward your stance. If you raise the recency
                      weight in Settings, newer publication dates also contribute to the final
                      ranking.
                    </p>
                  </section>
                </>
              ) : (
                <>
                  <section className="about-section">
                    <p className="about-section-label">Stage 1</p>
                    <p className="modal-copy">
                      <strong>Stage 1: Essay thesis detection.</strong> We first split your essay into
                      individual sentences using our sentence segmentation pipeline. Then we use a
                      DeBERTa Natural Language Inference (NLI) model to compare each sentence against
                      the hypothesis, &ldquo;This sentence is the author&apos;s main claim.&rdquo; This gives
                      each sentence a claimness score, and we present the top options so you can
                      choose the sentence that best represents your essay&apos;s central thesis, or
                      enter your own thesis wording when you want to override the suggestions.
                      When the LLM Agreement scorer is selected, this step is skipped and the
                      full essay is used for agreement scoring.
                    </p>
                  </section>
                  <section className="about-section">
                    <p className="about-section-label">Stage 2</p>
                    <p className="modal-copy">
                      <strong>Stage 2: Topic relevance.</strong> We identify articles that are relevant
                      to your essay as a whole. To
                      do this, we compute the similarity between your full essay and each Guardian
                      article using the retrieval representation selected in topic relevance mode:
                      either Lexical TF-IDF term vectors or Semantic truncated-SVD latent dimensions,
                      both compared with cosine similarity. This surfaces articles that discuss
                      similar themes, issues, and vocabulary.
                    </p>
                  </section>
                  <section className="about-section">
                    <p className="about-section-label">Stage 3</p>
                    <p className="modal-copy">
                      <strong>Stage 3: Agreement relevance.</strong> From the candidate articles identified
                      in Stage 2, we then rank them based on how they relate to your selected thesis
                      for NLI or your full essay for LLM.
                      The Agreement scorer in Settings can use either DeBERTa NLI over each
                      extracted article claim or Spark LLM scoring over retrieved article context.
                      The model estimates whether each article supports, contradicts, or is neutral toward your position.
                      If you raise the recency weight in Settings, newer publication dates also
                      contribute to the final ranking.
                    </p>
                  </section>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {isFilterOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setIsFilterOpen(false)}
          role="presentation"
        >
          <div
            className="modal-card filter-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="filter-settings-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <p className="modal-eyebrow">Filter</p>
                <h3 id="filter-settings-title">Article year range</h3>
              </div>
              <button
                type="button"
                className="modal-close"
                onClick={() => setIsFilterOpen(false)}
                aria-label="Close filter popup"
              >
                Close
              </button>
            </div>
            <div className="modal-settings-grid">
              <div className="weight-card full-row">
                <span>Year range</span>
                <div className="year-range-summary-grid" aria-live="polite">
                  <label className="year-range-value-card year-range-input-card">
                    <span>From</span>
                    <input
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      value={yearStartInput}
                      onChange={(event) => handleYearStartInputChange(event.target.value)}
                      onBlur={commitYearStartInput}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.currentTarget.blur()
                        }
                      }}
                      disabled={!hasAvailableYearBounds}
                      aria-label="Start year value"
                    />
                  </label>
                  <label className="year-range-value-card year-range-input-card">
                    <span>To</span>
                    <input
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      value={yearEndInput}
                      onChange={(event) => handleYearEndInputChange(event.target.value)}
                      onBlur={commitYearEndInput}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.currentTarget.blur()
                        }
                      }}
                      disabled={!hasAvailableYearBounds}
                      aria-label="End year value"
                    />
                  </label>
                </div>
                {minArticleYear !== null && maxArticleYear !== null && resolvedYearStart !== null && resolvedYearEnd !== null && (
                  <YearRangeSlider
                    minYear={minArticleYear}
                    maxYear={maxArticleYear}
                    startYear={resolvedYearStart}
                    endYear={resolvedYearEnd}
                    disabled={!hasAvailableYearBounds || yearRangeSpan === 0}
                    onStartYearChange={(nextYear) => handleYearStartChange(String(nextYear))}
                    onEndYearChange={(nextYear) => handleYearEndChange(String(nextYear))}
                  />
                )}
                <div className="year-range-scale" aria-hidden="true">
                  <span>{minArticleYear ?? '—'}</span>
                  <span>{maxArticleYear ?? '—'}</span>
                </div>
                <p className="setting-help-text">
                  Only return articles published within the selected year range.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {isSettingsOpen && (
        <div
          className="modal-backdrop"
          onClick={closeSettings}
          role="presentation"
        >
          <div
            className="modal-card settings-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="search-settings-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <p className="modal-eyebrow">Settings</p>
                <h3 id="search-settings-title">Search settings</h3>
              </div>
              <button
                type="button"
                className="modal-close"
                onClick={closeSettings}
                aria-label="Close settings popup"
              >
                Close
              </button>
            </div>
            <div className="settings-scroll-pane" ref={settingsScrollPaneRef}>
              <section className="settings-stage-section">
                <div className="settings-stage-heading">
                  <span>Stage 1</span>
                  <h4>Topic retrieval</h4>
                </div>
                <div className="modal-settings-grid">
                  <div
                    className="weight-card full-row settings-selection-card settings-scroll-target"
                    ref={topicSettingsRef}
                    tabIndex={-1}
                  >
                    <span>Topic relevance mode</span>
                    <div className="retrieval-model-grid">
                      {canUseTfidf && (
                        <button
                          type="button"
                          className={`retrieval-model-button ${retrievalModel === 'tfidf' ? 'active' : ''}`}
                          onClick={() => setRetrievalModel('tfidf')}
                          disabled={!canToggleSvd}
                        >
                          <strong>Lexical</strong>
                          <p>Pure TF-IDF term matching with cosine similarity.</p>
                        </button>
                      )}
                      {canUseSvd && (
                        <button
                          type="button"
                          className={`retrieval-model-button ${retrievalModel === 'svd' ? 'active' : ''}`}
                          onClick={() => setRetrievalModel('svd')}
                          disabled={!canToggleSvd}
                        >
                          <strong>Semantic</strong>
                          <p>SVD on TF-IDF to compare articles in latent concept space.</p>
                        </button>
                      )}
                    </div>
                    <p className="setting-help-text">
                      {retrievalModel === 'svd'
                        ? 'Semantic mode uses truncated-SVD dimensions from the TF-IDF matrix before cosine retrieval.'
                        : 'Lexical mode uses the original TF-IDF term vectors before cosine retrieval.'}
                    </p>
                  </div>

                  <div className="weight-card full-row settings-toggle-card">
                    <div className="settings-toggle-row">
                      <div className="settings-toggle-copy">
                        <span>Normalize topic relevance</span>
                        <p className="setting-help-text">
                          When on, the strongest retrieved article becomes 100% topic match within the current result set. When off, the app uses the raw retrieval similarity instead.
                        </p>
                      </div>
                      <button
                        type="button"
                        className={`settings-switch-button ${normalizeTopicScores ? 'active' : ''}`}
                        aria-pressed={normalizeTopicScores}
                        onClick={() => setNormalizeTopicScores(current => !current)}
                      >
                        <span className="settings-switch-label">
                          {normalizeTopicScores ? 'Normalized' : 'Raw'}
                        </span>
                        <span className="retrieval-toggle-switch" aria-hidden="true">
                          <span className="retrieval-toggle-thumb" />
                        </span>
                      </button>
                    </div>
                  </div>

                  <div className="weight-card full-row settings-selection-card">
                    <span>Candidate selection</span>
                    <div className="retrieval-model-grid">
                      <button
                        type="button"
                        className={`retrieval-model-button ${rerankSelectionMode === 'manual' ? 'active' : ''}`}
                        onClick={() => setRerankSelectionMode('manual')}
                      >
                        <strong>Manual K</strong>
                        <p>Use a fixed number of top topic matches.</p>
                      </button>
                      <button
                        type="button"
                        className={`retrieval-model-button ${rerankSelectionMode === 'automatic' ? 'active' : ''}`}
                        onClick={() => setRerankSelectionMode('automatic')}
                      >
                        <strong>Automatic threshold</strong>
                        <p>Only move articles forward when their raw topic relevance clears a threshold.</p>
                      </button>
                    </div>
                    {rerankSelectionMode === 'manual' ? (
                      <label className="settings-inline-field settings-range-field">
                        <div className="settings-range-header">
                          <span>Top K</span>
                          <strong className="settings-range-value">{rerankTopK}</strong>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="100"
                          step="1"
                          value={rerankTopK}
                          onChange={(e) => setRerankTopK(parseTopKInput(e.target.value, rerankTopK))}
                        />
                        <div className="settings-range-scale" aria-hidden="true">
                          <span>1</span>
                          <span>100</span>
                        </div>
                      </label>
                    ) : (
                      <label className="settings-inline-field settings-range-field">
                        <div className="settings-range-header">
                          <span>{`Topic relevance threshold (${retrievalModel === 'svd' ? 'Semantic' : 'Lexical'})`}</span>
                          <strong className="settings-range-value">{formatThresholdValue(currentAutoRerankThreshold)}</strong>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={currentAutoRerankThreshold}
                          onChange={(e) => updateCurrentAutoRerankThreshold(e.target.value)}
                        />
                        <div className="settings-range-scale" aria-hidden="true">
                          <span>0.00</span>
                          <span>1.00</span>
                        </div>
                      </label>
                    )}
                    <p className="setting-help-text">
                      {rerankSelectionMode === 'manual'
                        ? 'How many top retrieval matches move into the agreement reranking stage.'
                        : `Articles at or above this raw topic relevance threshold move into the next stage, with at most ${maxAutoRerankCandidates} articles reranked.`}
                    </p>
                  </div>
                </div>
              </section>

              <section className="settings-stage-section">
                <div className="settings-stage-heading">
                  <span>Stage 2</span>
                  <h4>Agreement scoring</h4>
                </div>
                <div className="modal-settings-grid">
                  <div className="weight-card full-row settings-toggle-card">
                    <div className="settings-toggle-row">
                      <div className="settings-toggle-copy">
                        <span>Semantic chunking</span>
                        <p className="setting-help-text">
                          When on, Spark scores semantic chunks, averages the related chunk scores, and hides articles with no related chunks.
                        </p>
                      </div>
                      <button
                        type="button"
                        className={`settings-switch-button ${useChunking ? 'active' : ''}`}
                        aria-pressed={useChunking}
                        onClick={() => {
                          if (!canUseChunking) return
                          setChunkingMode(currentMode => {
                            const nextMode = currentMode === 'semantic' ? 'none' : 'semantic'
                            if (nextMode === 'semantic') {
                              setStanceMethod('llm')
                            }
                            return nextMode
                          })
                        }}
                        disabled={!canUseChunking}
                      >
                        <span className="settings-switch-label">
                          {useChunking ? 'Semantic' : 'Article-level'}
                        </span>
                        <span className="retrieval-toggle-switch" aria-hidden="true">
                          <span className="retrieval-toggle-thumb" />
                        </span>
                      </button>
                    </div>
                    {!llmAgreementAvailable && supportedChunkingModes.includes('semantic') && (
                      <p className="setting-help-text">Add SPARK_API_KEY or API_KEY to enable semantic chunking.</p>
                    )}
                  </div>

                  <div
                    className="weight-card full-row settings-selection-card settings-scroll-target"
                    ref={agreementSettingsRef}
                    tabIndex={-1}
                  >
                    <span>Agreement scorer</span>
                    <div className="retrieval-model-grid">
                      {canUseNliAgreement && (
                        <button
                          type="button"
                          className={`retrieval-model-button ${stanceMethod === 'nli' ? 'active' : ''}`}
                          onClick={() => {
                            if (!useChunking) {
                              setStanceMethod('nli')
                            }
                          }}
                          disabled={useChunking}
                        >
                          <strong>NLI</strong>
                          <p>{useChunking ? 'Disabled while chunking is on.' : 'Compare the selected thesis or stance against each extracted article claim with DeBERTa.'}</p>
                        </button>
                      )}
                      {supportedStanceMethods.includes('llm') && (
                        <button
                          type="button"
                          className={`retrieval-model-button ${stanceMethod === 'llm' ? 'active' : ''}`}
                          onClick={() => {
                            if (canUseLlmAgreement) {
                              setStanceMethod('llm')
                            }
                          }}
                          disabled={!canUseLlmAgreement}
                        >
                          <strong>LLM RAG</strong>
                          <p>Send retrieved article context and your stance or full essay to Spark for a 0-1 agreement score.</p>
                        </button>
                      )}
                    </div>
                    <p className="setting-help-text">
                      {stanceMethod === 'llm'
                        ? (useChunking
                          ? 'The final agreement meter averages Spark scores across semantic chunks the LLM marks relevant.'
                          : 'The final agreement meter comes from Spark scoring the retrieved articles against your stance, thesis, or full essay.')
                        : 'The final agreement meter comes from local NLI over extracted article claims.'}
                      {!llmAgreementAvailable && supportedStanceMethods.includes('llm')
                        ? ' Add SPARK_API_KEY or API_KEY to enable the LLM scorer.'
                        : ''}
                    </p>
                  </div>
                </div>
              </section>

              <section className="settings-stage-section">
                <div className="settings-stage-heading">
                  <span>Final ranking</span>
                  <h4>Score weights</h4>
                </div>
                <div className="modal-settings-grid">
                  <div className="weight-card full-row weights-group-card">
                    <span>Final ranking weights</span>
                    <RankingWeightSlider
                      topicWeight={topicWeight}
                      recencyWeight={recencyWeight}
                      agreementWeight={stanceWeight}
                      onChange={(nextWeights) => {
                        setTopicWeight(nextWeights.topicWeight)
                        setRecencyWeight(nextWeights.recencyWeight)
                        setStanceWeight(nextWeights.agreementWeight)
                      }}
                    />
                    <div className="parameter-help-list">
                      <p className="parameter-help-item">
                        <strong>Topic weight:</strong> how much the final score prioritizes whole-text topical similarity.
                      </p>
                      <p className="parameter-help-item">
                        <strong>Agreement weight:</strong> how much the final score prioritizes whether the article aligns with your stance, thesis, or essay.
                      </p>
                      <p className="parameter-help-item">
                        <strong>Recency weight:</strong> how much the final score rewards newer publication dates.
                      </p>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
