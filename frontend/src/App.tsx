import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
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
  QueryHelpResponse,
  QueryRewriteAlternative,
  ResultsChatResponse,
  ResultsOverview,
  RetrievalModel,
  SimilarArticlesResponse,
  SvdLatentDimension,
  TypoCorrectionSuggestion,
} from './types'
import Chat from './Chat'
import AboutMethodFlow from './AboutMethodFlow'

type InputMode = 'stance' | 'essay'
type TopNavPage = 'home' | 'search' | 'about'
type AboutSection = 'overview' | 'team' | 'method'
type IntroStage = 0 | 1 | 2
type EssayStep = 1 | 2
type EssayThesisMode = 'candidate' | 'custom'
type RerankSelectionMode = 'manual' | 'automatic'
type StanceMethod = 'nli' | 'llm'
type ChunkingMode = 'none' | 'paragraph' | 'semantic'
type FrontendChunkingMode = Exclude<ChunkingMode, 'paragraph'>
type LengthFilterUnit = 'characters' | 'words' | 'reading_time'
type SettingsFocusTarget = 'retrieval-granularity' | 'topic-relevance' | 'agreement-scorer'
type QueryAssistMode = 'menu' | 'rewrite' | 'suggestions'
type SvdDimensionLabelMap = Record<number, string>
type SvdChartSeriesRole = 'article' | 'query'
type TopicFeedbackSearchOptions = {
  topicFeedbackIrrelevantArticleIds?: string[]
  markTopicFeedbackApplied?: boolean
  topicOverride?: string
  skipTypoCorrection?: boolean
}
type SearchFocusWordSnapshot = {
  text: string
  startX: number
  startY: number
  driftX: number
  driftY: number
  fontFamily: string
  fontSize: number
  fontWeight: string
  lineHeight: number
}
type SearchFocusSnapshot = {
  key: number
  text: string
  mode: InputMode
  words: SearchFocusWordSnapshot[]
  clearing: boolean
}
type ResultsChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  attachments?: ResultsChatAttachment[] | null
  source_indices?: number[] | null
  sources?: ResultsOverviewSource[] | null
}
type ResultsChatAttachment = {
  articleId: string
  resultIndex: number
  title: string
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
  default_chunk_auto_rerank_thresholds?: Partial<Record<RetrievalModel, number | null>> | null
  default_chunk_candidate_top_k?: number | null
  default_chunk_article_top_k?: number | null
  max_auto_rerank_candidates?: number | null
  max_chunk_candidate_top_k?: number | null
  supported_retrieval_models?: string[] | null
  supported_rerank_selection_modes?: string[] | null
  min_article_year?: number | null
  max_article_year?: number | null
  min_article_characters?: number | null
  max_article_characters?: number | null
  min_article_words?: number | null
  max_article_words?: number | null
  min_article_reading_minutes?: number | null
  max_article_reading_minutes?: number | null
}

type ApiErrorPayload = {
  error?: string
}

const introTopicSequence = [
  'climate',
  'immigration',
  'minimum wage',
] as const

const aboutOverviewParagraphs = [
  'hear! hear! is a research and writing companion designed to help users explore Guardian opinion articles in relation to their own ideas. Rather than only matching simple keywords, the app combines topic relevance and stance-aware ranking so readers can quickly find articles that support, complicate, or challenge a position they care about.',
  'The project is built to make argument discovery more transparent and useful for writing. Whether someone starts with a short claim or a full draft essay, hear! hear! surfaces meaningful viewpoints, explains how results are ranked, and helps users understand the broader conversation before they refine a thesis, gather evidence, or revise an argument.',
] as const

const aboutTeamMembers = [
  'Ashali Sharma',
  'Jonathan Scardon',
  'Steven Zhou',
  'Nuo Cen',
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
const defaultSupportedRetrievalModels: RetrievalModel[] = ['svd', 'minilm', 'tfidf']
const defaultRerankSelectionMode: RerankSelectionMode = 'automatic'
const defaultStanceMethod: StanceMethod = 'nli'
const defaultSupportedStanceMethods: StanceMethod[] = ['nli', 'llm']
const defaultChunkingMode: FrontendChunkingMode = 'none'
const defaultSupportedChunkingModes: FrontendChunkingMode[] = ['none', 'semantic']
const defaultAutoRerankThresholds: Record<RetrievalModel, number> = {
  tfidf: 0.3,
  svd: 0.6,
  minilm: 0.4,
}
const defaultChunkAutoRerankThresholds: Record<RetrievalModel, number> = {
  tfidf: 0.12,
  svd: 0.35,
  minilm: 0.45,
}
const defaultMaxAutoRerankCandidates = 100
const defaultChunkCandidateTopK = 100
const defaultChunkArticleTopK = 5
const defaultMaxChunkCandidateTopK = 500
const similarArticlesPageSize = 5
const searchFocusMinimumMs = 0
const searchFocusClearMs = 850
const searchFocusMaxWords = 34

const isRetrievalModel = (value: unknown): value is RetrievalModel => (
  value === 'tfidf' || value === 'svd' || value === 'minilm'
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

const resolvePreferredStanceMethod = (
  preferredMethod: StanceMethod,
  supportedMethods: StanceMethod[],
  llmAvailable: boolean,
): StanceMethod => {
  const fallbackMethod = supportedMethods.includes('nli')
    ? 'nli'
    : (supportedMethods[0] ?? defaultSupportedStanceMethods[0])

  if (preferredMethod === 'llm') {
    return llmAvailable && supportedMethods.includes('llm') ? 'llm' : fallbackMethod
  }

  if (supportedMethods.includes(preferredMethod)) {
    return preferredMethod
  }

  return llmAvailable && supportedMethods.includes('llm')
    ? 'llm'
    : fallbackMethod
}

const resolvePreferredChunkingMode = (
  preferredMode: FrontendChunkingMode,
  supportedModes: FrontendChunkingMode[],
  llmAvailable: boolean,
): FrontendChunkingMode => {
  const fallbackMode = supportedModes.includes('none')
    ? 'none'
    : (supportedModes[0] ?? defaultChunkingMode)

  if (preferredMode === 'semantic') {
    return llmAvailable && supportedModes.includes('semantic') ? 'semantic' : fallbackMode
  }

  return supportedModes.includes(preferredMode) ? preferredMode : fallbackMode
}

const clampAutoRerankThreshold = (value: number): number => (
  Math.max(0, Math.min(1, value))
)

const normalizeAutoRerankThresholds = (
  value: unknown,
  defaults: Record<RetrievalModel, number> = defaultAutoRerankThresholds,
): Record<RetrievalModel, number> => {
  const nextThresholds = { ...defaults }
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

const normalizeConfigInteger = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isInteger(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    if (Number.isInteger(parsed)) return parsed
  }
  return null
}

const normalizeConfigYear = normalizeConfigInteger

const clampWholeNumber = (value: number, minValue: number, maxValue: number): number => (
  Math.min(maxValue, Math.max(minValue, Math.round(value)))
)

const clampYear = (value: number, minYear: number, maxYear: number): number => (
  clampWholeNumber(value, minYear, maxYear)
)

const clampCharacterCount = (value: number, minValue: number, maxValue: number): number => (
  clampWholeNumber(value, minValue, maxValue)
)

const clampWordCount = (value: number, minValue: number, maxValue: number): number => (
  clampWholeNumber(value, minValue, maxValue)
)

const clampReadingMinutes = (value: number, minValue: number, maxValue: number): number => (
  clampWholeNumber(value, minValue, maxValue)
)

const formatCharacterCount = (value: number): string => value.toLocaleString()
const formatWordCount = (value: number): string => value.toLocaleString()
const formatReadingMinutes = (value: number): string => value.toLocaleString()

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

const searchFocusWordPattern = /[\p{L}\p{N}]+(?:[-'][\p{L}\p{N}]+)*/gu

const getSearchFocusWords = (value: string): string[] => {
  const words = [...value.matchAll(searchFocusWordPattern)]
    .map(match => match[0])
    .filter(Boolean)
    .slice(0, searchFocusMaxWords)

  return words.length > 0 ? words : ['Searching']
}

const getCanvasTextMeasure = (): CanvasRenderingContext2D | null => {
  if (typeof document === 'undefined') return null
  const canvas = document.createElement('canvas')
  return canvas.getContext('2d')
}

const getElementFont = (element: HTMLElement): {
  font: string
  fontFamily: string
  fontSize: number
  fontWeight: string
  lineHeight: number
} => {
  const style = window.getComputedStyle(element)
  const fontSize = Number.parseFloat(style.fontSize) || 24
  const lineHeight = style.lineHeight === 'normal'
    ? fontSize * 1.08
    : (Number.parseFloat(style.lineHeight) || fontSize * 1.08)
  const fontWeight = style.fontWeight || '400'
  const fontFamily = style.fontFamily || 'Kanit, sans-serif'
  const font = [
    style.fontStyle && style.fontStyle !== 'normal' ? style.fontStyle : '',
    style.fontVariant && style.fontVariant !== 'normal' ? style.fontVariant : '',
    fontWeight,
    `${fontSize}px`,
    fontFamily,
  ].filter(Boolean).join(' ')

  return {
    font,
    fontFamily,
    fontSize,
    fontWeight,
    lineHeight,
  }
}

const measureSearchFocusSourceWords = (
  text: string,
  element: HTMLElement | null,
  options: {
    maxWords?: number
    wrap?: boolean
  } = {},
): SearchFocusWordSnapshot[] => {
  const words = getSearchFocusWords(text).slice(0, options.maxWords ?? searchFocusMaxWords)
  if (!element || words.length === 0 || typeof window === 'undefined') return []

  const rect = element.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return []

  const style = window.getComputedStyle(element)
  const paddingLeft = Number.parseFloat(style.paddingLeft) || 0
  const paddingTop = Number.parseFloat(style.paddingTop) || 0
  const paddingRight = Number.parseFloat(style.paddingRight) || 0
  const font = getElementFont(element)
  const measure = getCanvasTextMeasure()
  if (measure) {
    measure.font = font.font
  }

  const measureText = (value: string): number => (
    measure ? measure.measureText(value).width : value.length * font.fontSize * 0.56
  )
  const spaceWidth = measureText(' ')
  const maxLineWidth = Math.max(24, rect.width - paddingLeft - paddingRight)
  const shouldWrap = Boolean(options.wrap)
  const baselineTop = rect.top + (shouldWrap ? paddingTop : Math.max(0, (rect.height - font.lineHeight) / 2))
  let cursorX = 0
  let lineIndex = 0

  return words.map((word) => {
    const wordWidth = measureText(word)
    if (shouldWrap && cursorX > 0 && cursorX + wordWidth > maxLineWidth) {
      cursorX = 0
      lineIndex += 1
    }

    const startX = rect.left + paddingLeft + cursorX
    const startY = baselineTop + lineIndex * font.lineHeight
    cursorX += wordWidth + spaceWidth

    return {
      text: word,
      startX,
      startY,
      driftX: 0,
      driftY: 0,
      fontFamily: font.fontFamily,
      fontSize: font.fontSize,
      fontWeight: font.fontWeight,
      lineHeight: font.lineHeight,
    }
  })
}

const addSearchFocusDrift = (words: SearchFocusWordSnapshot[]): SearchFocusWordSnapshot[] => {
  if (words.length === 0 || typeof window === 'undefined') return words

  const centers = words.map(word => ({
    x: word.startX + (word.text.length * word.fontSize * 0.28),
    y: word.startY + word.lineHeight / 2,
  }))
  const minX = Math.min(...centers.map(center => center.x))
  const maxX = Math.max(...centers.map(center => center.x))
  const minY = Math.min(...centers.map(center => center.y))
  const maxY = Math.max(...centers.map(center => center.y))
  const centerX = (minX + maxX) / 2
  const centerY = (minY + maxY) / 2
  const viewportWidth = window.innerWidth || 1024
  const viewportHeight = window.innerHeight || 768
  const padding = Math.min(96, Math.max(28, viewportWidth * 0.04))
  const maxRightScale = maxX === centerX ? 2.4 : (viewportWidth - padding - centerX) / (maxX - centerX)
  const maxLeftScale = minX === centerX ? 2.4 : (centerX - padding) / (centerX - minX)
  const maxBottomScale = maxY === centerY ? 2.4 : (viewportHeight - padding - centerY) / (maxY - centerY)
  const maxTopScale = minY === centerY ? 2.4 : (centerY - padding) / (centerY - minY)
  const boundedScale = Math.min(maxRightScale, maxLeftScale, maxBottomScale, maxTopScale)
  const finalScale = Math.max(1.5, Math.min(3.85, boundedScale * 1.18))
  const driftScale = finalScale - 1

  return words.map((word, index) => {
    const center = centers[index]
    let vectorX = center.x - centerX
    let vectorY = center.y - centerY

    if (Math.abs(vectorX) < 4 && Math.abs(vectorY) < 4) {
      const angle = (index / Math.max(1, words.length)) * Math.PI * 2
      vectorX = Math.cos(angle) * 34
      vectorY = Math.sin(angle) * 34
    }

    return {
      ...word,
      driftX: vectorX * driftScale,
      driftY: vectorY * driftScale,
    }
  })
}

const buildFallbackSearchFocusWords = (
  text: string,
  mode: InputMode,
): SearchFocusWordSnapshot[] => {
  if (typeof window === 'undefined') return []

  const words = getSearchFocusWords(text)
  const fontSize = mode === 'essay'
    ? Math.max(18, Math.min(34, window.innerWidth * 0.033))
    : Math.max(24, Math.min(56, window.innerWidth * 0.052))
  const lineHeight = fontSize * 1.05
  const rowWidth = Math.min(window.innerWidth - 48, Math.max(320, words.length * fontSize * 2.6))
  let x = (window.innerWidth - rowWidth) / 2
  let y = window.innerHeight / 2 - lineHeight / 2

  const positionedWords = words.map((word) => {
    const wordWidth = word.length * fontSize * 0.56
    if (x + wordWidth > window.innerWidth - 24) {
      x = (window.innerWidth - rowWidth) / 2
      y += lineHeight * 1.24
    }

    const snapshot: SearchFocusWordSnapshot = {
      text: word,
      startX: x,
      startY: y,
      driftX: 0,
      driftY: 0,
      fontFamily: "'Kanit', sans-serif",
      fontSize,
      fontWeight: '400',
      lineHeight,
    }
    x += wordWidth + fontSize * 0.38
    return snapshot
  })

  return addSearchFocusDrift(positionedWords)
}

const getArticleIdKey = (article: Pick<Article, 'id'>): string => String(article.id)
const typoTokenPattern = /[\p{L}]+(?:[-'][\p{L}]+)*/gu
const avoidWordTokenPattern = /[\p{L}\p{N}_]+(?:[-'][\p{L}\p{N}_]+)*/gu

const normalizeTypoTerm = (value: string): string => value.trim().toLocaleLowerCase()

const parseWordsToAvoid = (value: string): string[] => {
  const seen = new Set<string>()
  const words: string[] = []

  for (const match of value.matchAll(avoidWordTokenPattern)) {
    const word = match[0].replace(/^[_'-]+|[_'-]+$/g, '').toLocaleLowerCase()
    if (!word || seen.has(word)) continue
    words.push(word)
    seen.add(word)
  }

  return words
}

const normalizeTypoCorrection = (value: unknown): TypoCorrectionSuggestion | null => {
  if (!value || typeof value !== 'object') return null

  const rawSuggestion = value as Partial<TypoCorrectionSuggestion>
  const query = typeof rawSuggestion.query === 'string' ? rawSuggestion.query : ''
  const highlightedTerms = Array.isArray(rawSuggestion.highlighted_terms)
    ? rawSuggestion.highlighted_terms
      .filter((term): term is string => typeof term === 'string' && term.trim() !== '')
      .map(normalizeTypoTerm)
    : []
  const options = Array.isArray(rawSuggestion.options)
    ? rawSuggestion.options
      .filter(option => option && typeof option.query === 'string' && option.query.trim() !== '')
      .map(option => ({
        query: option.query.trim(),
        label: typeof option.label === 'string' && option.label.trim() !== ''
          ? option.label.trim()
          : option.query.trim(),
        replacements: option.replacements ?? null,
        distance: typeof option.distance === 'number' ? option.distance : null,
        df: typeof option.df === 'number' ? option.df : null,
      }))
    : []

  if (!query.trim() || highlightedTerms.length === 0 || options.length === 0) {
    return null
  }

  return {
    query,
    highlighted_terms: Array.from(new Set(highlightedTerms)),
    options,
    corrections: Array.isArray(rawSuggestion.corrections) ? rawSuggestion.corrections : null,
  }
}

const normalizeQueryRewriteAlternatives = (value: unknown): QueryRewriteAlternative[] => {
  if (!Array.isArray(value)) return []

  const alternatives: QueryRewriteAlternative[] = []
  const seen = new Set<string>()
  for (const item of value) {
    if (!item || typeof item !== 'object') continue
    const rawAlternative = item as Partial<QueryRewriteAlternative>
    const topic = typeof rawAlternative.topic === 'string' ? rawAlternative.topic.trim() : ''
    const opinion = typeof rawAlternative.opinion === 'string' ? rawAlternative.opinion.trim() : ''
    const query = typeof rawAlternative.query === 'string' ? rawAlternative.query.trim() : ''
    if (!topic || !opinion || !query) continue

    const key = `${topic.toLocaleLowerCase()}\u0000${opinion.toLocaleLowerCase()}`
    if (seen.has(key)) continue

    alternatives.push({
      topic,
      opinion,
      query,
      rationale: typeof rawAlternative.rationale === 'string' && rawAlternative.rationale.trim() !== ''
        ? rawAlternative.rationale.trim()
        : null,
    })
    seen.add(key)
    if (alternatives.length >= 3) break
  }

  return alternatives
}

const normalizeQueryImproveSuggestions = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []

  const suggestions: string[] = []
  const seen = new Set<string>()
  for (const item of value) {
    const suggestion = typeof item === 'string' ? item.trim() : ''
    const key = suggestion.toLocaleLowerCase()
    if (!suggestion || seen.has(key)) continue
    suggestions.push(suggestion)
    seen.add(key)
    if (suggestions.length >= 6) break
  }

  return suggestions
}

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

type ResultsOverviewSource = NonNullable<ResultsOverview['sources']>[number]
type ResultsOverviewEvidence = NonNullable<ResultsOverview['key_evidence']>[number]
type ResultsOverviewArgument = NonNullable<ResultsOverview['supporting_arguments']>[number]
type ResultsOverviewCitationSegment =
  | { type: 'text'; value: string }
  | { type: 'sources'; sourceIndices: number[] }

const normalizeResultsOverviewSourceIndices = (value: unknown): number[] => {
  const rawValues = Array.isArray(value) ? value : [value]
  const seen = new Set<number>()
  const indices: number[] = []

  rawValues.forEach((rawValue) => {
    const index = Number(rawValue)
    if (!Number.isInteger(index) || index < 1 || seen.has(index)) return
    indices.push(index)
    seen.add(index)
  })

  return indices
}

const resultsOverviewCitationPattern = /\[\s*(?:#|\d+(?:\s*,\s*\d+)*)\s*\]/g

const parseResultsOverviewCitationIndices = (value: string): number[] => (
  normalizeResultsOverviewSourceIndices(value.match(/\d+/g) ?? [])
)

const getResultsOverviewCitationSegments = (
  value: string,
): {
  segments: ResultsOverviewCitationSegment[]
  sourceIndices: number[]
} => {
  const segments: ResultsOverviewCitationSegment[] = []
  const sourceIndices: number[] = []
  const seenSourceIndices = new Set<number>()
  let currentIndex = 0

  for (const match of value.matchAll(resultsOverviewCitationPattern)) {
    const matchIndex = match.index ?? 0
    const textBeforeCitation = value.slice(currentIndex, matchIndex).replace(/\s+$/u, '')
    if (textBeforeCitation) {
      segments.push({ type: 'text', value: textBeforeCitation })
    }

    const citationSourceIndices = parseResultsOverviewCitationIndices(match[0])
    if (citationSourceIndices.length > 0) {
      segments.push({ type: 'sources', sourceIndices: citationSourceIndices })
    }

    citationSourceIndices.forEach((sourceIndex) => {
      if (seenSourceIndices.has(sourceIndex)) return
      sourceIndices.push(sourceIndex)
      seenSourceIndices.add(sourceIndex)
    })
    currentIndex = matchIndex + match[0].length
  }

  const textAfterCitations = value.slice(currentIndex)
  if (textAfterCitations) {
    segments.push({ type: 'text', value: textAfterCitations })
  }

  return { segments, sourceIndices }
}

const normalizeResultsOverviewEvidence = (
  value: unknown,
  maxItems = 5,
): ResultsOverviewEvidence[] => {
  if (!Array.isArray(value)) return []

  return value.slice(0, maxItems).reduce<ResultsOverviewEvidence[]>((items, item) => {
    if (typeof item === 'string') {
      const evidence = item.trim()
      if (evidence) {
        items.push({ evidence, source_indices: [] })
      }
      return items
    }

    if (!item || typeof item !== 'object') return items

    const rawItem = item as Partial<ResultsOverviewEvidence> & {
      text?: unknown
      claim?: unknown
      sources?: unknown
      result_indices?: unknown
    }
    const evidence = typeof rawItem.evidence === 'string'
      ? rawItem.evidence.trim()
      : typeof rawItem.text === 'string'
        ? rawItem.text.trim()
        : typeof rawItem.claim === 'string'
          ? rawItem.claim.trim()
          : ''

    if (!evidence) return items

    items.push({
      evidence,
      source_indices: normalizeResultsOverviewSourceIndices(
        rawItem.source_indices ?? rawItem.sources ?? rawItem.result_indices,
      ),
    })
    return items
  }, [])
}

const normalizeResultsOverviewArguments = (
  value: unknown,
): ResultsOverviewArgument[] => {
  if (!Array.isArray(value)) return []

  return value.slice(0, 3).reduce<ResultsOverviewArgument[]>((items, item) => {
    if (typeof item === 'string') {
      const argument = item.trim()
      if (argument) {
        items.push({ argument, source_indices: [], evidence: [] })
      }
      return items
    }

    if (!item || typeof item !== 'object') return items

    const rawItem = item as Partial<ResultsOverviewArgument> & {
      claim?: unknown
      point?: unknown
      sources?: unknown
      result_indices?: unknown
      key_evidence?: unknown
    }
    const argument = typeof rawItem.argument === 'string'
      ? rawItem.argument.trim()
      : typeof rawItem.claim === 'string'
        ? rawItem.claim.trim()
        : typeof rawItem.point === 'string'
          ? rawItem.point.trim()
          : ''

    if (!argument) return items

    items.push({
      argument,
      source_indices: normalizeResultsOverviewSourceIndices(
        rawItem.source_indices ?? rawItem.sources ?? rawItem.result_indices,
      ),
      evidence: normalizeResultsOverviewEvidence(rawItem.evidence ?? rawItem.key_evidence, 3),
    })
    return items
  }, [])
}

const normalizeResultsOverviewSources = (value: unknown): ResultsOverviewSource[] => {
  if (!Array.isArray(value)) return []

  return value.reduce<ResultsOverviewSource[]>((sources, source) => {
    if (!source || typeof source !== 'object') return sources

    const rawSource = source as Partial<ResultsOverviewSource>
    const resultIndex = Number(rawSource.result_index)
    const title = typeof rawSource.title === 'string' ? rawSource.title.trim() : ''
    if (!Number.isInteger(resultIndex) || resultIndex < 1 || !title) return sources

    sources.push({
      result_index: resultIndex,
      title,
      url: typeof rawSource.url === 'string' ? rawSource.url : null,
      article_id: rawSource.article_id ?? null,
    })
    return sources
  }, [])
}

const easeOutCubic = (progress: number): number => 1 - (1 - progress) ** 3

let resultsOverviewScrollAnimationFrame: number | null = null

const cancelResultsOverviewScrollAnimation = (): void => {
  if (typeof window === 'undefined' || resultsOverviewScrollAnimationFrame === null) return

  window.cancelAnimationFrame(resultsOverviewScrollAnimationFrame)
  resultsOverviewScrollAnimationFrame = null
}

const animateWindowScrollTo = (targetTop: number, durationMs = 380): void => {
  if (typeof window === 'undefined' || typeof document === 'undefined') return

  cancelResultsOverviewScrollAnimation()

  const startTop = window.scrollY
  const scrollHeight = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
  const maxTop = Math.max(0, scrollHeight - window.innerHeight)
  const resolvedTargetTop = Math.max(0, Math.min(targetTop, maxTop))
  const distance = resolvedTargetTop - startTop

  if (Math.abs(distance) < 1) {
    window.scrollTo({ top: resolvedTargetTop, behavior: 'auto' })
    return
  }

  const startTime = window.performance.now()

  const step = (currentTime: number): void => {
    const elapsed = currentTime - startTime
    const progress = Math.min(1, elapsed / durationMs)
    const nextTop = startTop + distance * easeOutCubic(progress)

    window.scrollTo({ top: nextTop, behavior: 'auto' })

    if (progress < 1) {
      resultsOverviewScrollAnimationFrame = window.requestAnimationFrame(step)
    } else {
      resultsOverviewScrollAnimationFrame = null
    }
  }

  resultsOverviewScrollAnimationFrame = window.requestAnimationFrame(step)
}

const scrollToResultsOverviewSource = (resultIndex: number): boolean => {
  if (typeof window === 'undefined' || typeof document === 'undefined') return false

  const target = document.getElementById(`result-${resultIndex}`)
  if (!target) return false

  const targetTop = window.scrollY + target.getBoundingClientRect().top - 16
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  if (prefersReducedMotion) {
    cancelResultsOverviewScrollAnimation()
    window.scrollTo({ top: Math.max(0, targetTop), behavior: 'auto' })
  } else {
    animateWindowScrollTo(targetTop)
  }

  const nextHash = `#result-${resultIndex}`
  if (window.location.hash !== nextHash) {
    window.history.pushState(null, '', `${window.location.pathname}${window.location.search}${nextHash}`)
  }

  return true
}

const handleResultsOverviewSourceClick = (
  event: ReactMouseEvent<HTMLAnchorElement>,
  resultIndex: number,
): void => {
  if (!scrollToResultsOverviewSource(resultIndex)) return

  event.preventDefault()
}

const ResultsOverviewSources = ({
  sourceIndices,
  sources,
}: {
  sourceIndices?: number[] | null
  sources?: ResultsOverviewSource[] | null
}): JSX.Element | null => {
  const indices = normalizeResultsOverviewSourceIndices(sourceIndices)
  if (indices.length === 0) return null

  const sourceByIndex = new Map((sources ?? []).map(source => [source.result_index, source]))
  const resolvedSources = indices.map(index => sourceByIndex.get(index) ?? {
    result_index: index,
    title: `Result ${index}`,
    url: null,
    article_id: null,
  })

  return (
    <span className="results-overview-sources" aria-label="Sources">
      {resolvedSources.map((source) => {
        const label = `Result ${source.result_index}`
        return (
          <a
            key={source.result_index}
            className="results-overview-source-chip"
            href={`#result-${source.result_index}`}
            title={source.title}
            onClick={(event) => handleResultsOverviewSourceClick(event, source.result_index)}
          >
            {label}
          </a>
        )
      })}
    </span>
  )
}

const ResultsOverviewCitedText = ({
  text,
  sourceIndices,
  sources,
}: {
  text: string
  sourceIndices?: number[] | null
  sources?: ResultsOverviewSource[] | null
}): JSX.Element => {
  const { segments, sourceIndices: inlineSourceIndices } = getResultsOverviewCitationSegments(text)
  const inlineSourceIndexSet = new Set(inlineSourceIndices)
  const trailingSourceIndices = normalizeResultsOverviewSourceIndices(sourceIndices)
    .filter(sourceIndex => !inlineSourceIndexSet.has(sourceIndex))
  const renderedSegments = segments.length > 0 ? segments : [{ type: 'text' as const, value: text }]

  return (
    <>
      {renderedSegments.map((segment, index) => {
        if (segment.type === 'sources') {
          return (
            <ResultsOverviewSources
              key={`source-${index}-${segment.sourceIndices.join('-')}`}
              sourceIndices={segment.sourceIndices}
              sources={sources}
            />
          )
        }

        return <span key={`text-${index}`}>{segment.value}</span>
      })}
      <ResultsOverviewSources sourceIndices={trailingSourceIndices} sources={sources} />
    </>
  )
}

const ResultsOverviewArgumentList = ({
  title,
  items,
  sources,
}: {
  title: string
  items?: ResultsOverviewArgument[] | null
  sources?: ResultsOverviewSource[] | null
}): JSX.Element | null => {
  if (!Array.isArray(items) || items.length === 0) return null

  return (
    <div className="results-overview-argument-group">
      <h4>{title}</h4>
      <ul className="results-overview-argument-list">
        {items.map((item, index) => (
          <li key={`${title}-${index}-${item.argument}`}>
            <div className="results-overview-argument-main">
              <span>
                <ResultsOverviewCitedText
                  text={item.argument}
                  sourceIndices={item.source_indices}
                  sources={sources}
                />
              </span>
            </div>
            {Array.isArray(item.evidence) && item.evidence.length > 0 && (
              <ul className="results-overview-evidence-list nested">
                {item.evidence.map((evidence, evidenceIndex) => (
                  <li key={`${item.argument}-evidence-${evidenceIndex}`}>
                    <span>
                      <ResultsOverviewCitedText
                        text={evidence.evidence}
                        sourceIndices={evidence.source_indices}
                        sources={sources}
                      />
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

const normalizeArticleSearchResponse = (
  payload: Article[] | ArticleSearchResponse | null,
): {
  articles: Article[]
  querySvdCorpusChartDimensions: SvdLatentDimension[]
  querySvdDimensions: SvdLatentDimension[]
  emptyResultsMessage: string | null
  typoCorrection: TypoCorrectionSuggestion | null
} => {
  if (Array.isArray(payload)) {
    return {
      articles: payload,
      querySvdCorpusChartDimensions: [],
      querySvdDimensions: [],
      emptyResultsMessage: null,
      typoCorrection: null,
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
  const typoCorrection = normalizeTypoCorrection(payload?.typo_suggestion)

  return {
    articles: results,
    querySvdCorpusChartDimensions,
    querySvdDimensions,
    emptyResultsMessage,
    typoCorrection,
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

const formatVaderScore = (value: number): string => (
  `${value >= 0 ? '+' : ''}${value.toFixed(2)}`
)

const formatSentimentPercent = (value: number): string => `${Math.round(clampSvdMagnitude(value) * 100)}%`

const formatSentimentLabel = (value?: string | null): string => {
  const normalized = String(value || '').trim()
  if (!normalized) return 'Neutral'
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

const formatToneStrength = (value?: string | null): string => (
  formatSentimentLabel(value)
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

function TwoHandleRangeSlider(
  {
    minValue,
    maxValue,
    startValue,
    endValue,
    disabled = false,
    startAriaLabel,
    endAriaLabel,
    formatValue = (value: number): string => String(value),
    onStartValueChange,
    onEndValueChange,
  }: {
    minValue: number
    maxValue: number
    startValue: number
    endValue: number
    disabled?: boolean
    startAriaLabel: string
    endAriaLabel: string
    formatValue?: (value: number) => string
    onStartValueChange: (nextValue: number) => void
    onEndValueChange: (nextValue: number) => void
  },
): JSX.Element {
  const sliderShellRef = useRef<HTMLDivElement | null>(null)
  const [draggingThumb, setDraggingThumb] = useState<'start' | 'end' | null>(null)
  const valueSpan = Math.max(0, maxValue - minValue)
  const startPercent = valueSpan === 0 ? 0 : (((startValue - minValue) / valueSpan) * 100)
  const endPercent = valueSpan === 0 ? 100 : (((endValue - minValue) / valueSpan) * 100)

  const resolveValueFromClientX = (clientX: number): number => {
    const sliderBounds = sliderShellRef.current?.getBoundingClientRect()
    if (!sliderBounds || sliderBounds.width <= 0 || valueSpan === 0) {
      return startValue
    }
    const relativeX = Math.min(sliderBounds.width, Math.max(0, clientX - sliderBounds.left))
    const nextPercent = relativeX / sliderBounds.width
    const nextValue = minValue + Math.round(nextPercent * valueSpan)
    return clampWholeNumber(nextValue, minValue, maxValue)
  }

  const applyDraggedValue = (clientX: number, thumb: 'start' | 'end'): void => {
    const nextValue = resolveValueFromClientX(clientX)
    if (thumb === 'start') {
      onStartValueChange(Math.min(nextValue, endValue))
      return
    }
    onEndValueChange(Math.max(nextValue, startValue))
  }

  useEffect(() => {
    if (!draggingThumb || disabled) {
      return
    }

    const handlePointerMove = (event: PointerEvent): void => {
      applyDraggedValue(event.clientX, draggingThumb)
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
  }, [
    disabled,
    draggingThumb,
    endValue,
    maxValue,
    minValue,
    onEndValueChange,
    onStartValueChange,
    startValue,
    valueSpan,
  ])

  const beginDrag = (
    thumb: 'start' | 'end',
    event: ReactPointerEvent<HTMLButtonElement | HTMLDivElement>,
  ): void => {
    if (disabled) return
    event.preventDefault()
    event.stopPropagation()
    setDraggingThumb(thumb)
    applyDraggedValue(event.clientX, thumb)
  }

  const handleTrackPointerDown = (event: ReactPointerEvent<HTMLDivElement>): void => {
    if (disabled) return
    const nextValue = resolveValueFromClientX(event.clientX)
    const nearestThumb = Math.abs(nextValue - startValue) <= Math.abs(nextValue - endValue)
      ? 'start'
      : 'end'
    beginDrag(nearestThumb, event)
  }

  const nudgeThumb = (thumb: 'start' | 'end', delta: number): void => {
    if (thumb === 'start') {
      onStartValueChange(clampWholeNumber(startValue + delta, minValue, endValue))
      return
    }
    onEndValueChange(clampWholeNumber(endValue + delta, startValue, maxValue))
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
        onStartValueChange(minValue)
      } else {
        onEndValueChange(startValue)
      }
      return
    }

    if (event.key === 'End') {
      event.preventDefault()
      if (thumb === 'start') {
        onStartValueChange(endValue)
      } else {
        onEndValueChange(maxValue)
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
          aria-label={startAriaLabel}
          aria-valuemin={minValue}
          aria-valuemax={endValue}
          aria-valuenow={startValue}
          aria-valuetext={formatValue(startValue)}
        />
        <button
          type="button"
          className={`year-range-handle end ${draggingThumb === 'end' ? 'dragging' : ''}`}
          style={{ left: `calc(${endPercent}% - 10px)` }}
          onPointerDown={(event) => beginDrag('end', event)}
          onKeyDown={(event) => handleThumbKeyDown('end', event)}
          disabled={disabled}
          role="slider"
          aria-label={endAriaLabel}
          aria-valuemin={startValue}
          aria-valuemax={maxValue}
          aria-valuenow={endValue}
          aria-valuetext={formatValue(endValue)}
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

const FloatingSearchFocus = ({
  mode,
  words,
  clearing,
}: {
  mode: InputMode
  words: SearchFocusWordSnapshot[]
  clearing: boolean
}): JSX.Element => {
  const densityClass = words.length > 22
    ? 'dense'
    : (words.length > 14 ? 'balanced' : 'spacious')

  return (
    <section
      className={`search-focus-overlay ${clearing ? 'clearing' : ''}`}
      role="status"
      aria-live="polite"
      aria-label={`${mode === 'essay' ? 'Essay search' : 'Stance search'} in progress`}
    >
      <div className="search-focus-glow" aria-hidden="true" />
      <div className={`search-focus-field ${densityClass}`} aria-hidden="true">
        {words.map((word, index) => {
          const duration = 12.4 + (index % 5) * 0.34

          return (
            <span
              key={`${word.text}-${index}`}
              className="search-focus-word"
              style={{
                left: `${word.startX}px`,
                top: `${word.startY}px`,
                fontSize: `${word.fontSize}px`,
                fontWeight: word.fontWeight,
                lineHeight: `${word.lineHeight}px`,
                '--focus-font-family': word.fontFamily,
                '--drift-x': `${word.driftX}px`,
                '--drift-y': `${word.driftY}px`,
                '--word-delay': `${index * 18}ms`,
                '--word-duration': `${duration}s`,
              } as CSSProperties}
            >
              {word.text}
            </span>
          )
        })}
      </div>
    </section>
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
  const [autoChunkRerankThresholds, setAutoChunkRerankThresholds] = useState<Record<RetrievalModel, number>>(
    defaultChunkAutoRerankThresholds,
  )
  const [maxAutoRerankCandidates, setMaxAutoRerankCandidates] = useState<number>(
    defaultMaxAutoRerankCandidates,
  )
  const [chunkCandidateTopK, setChunkCandidateTopK] = useState<number>(defaultChunkCandidateTopK)
  const [chunkArticleTopK, setChunkArticleTopK] = useState<number>(defaultChunkArticleTopK)
  const [maxChunkCandidateTopK, setMaxChunkCandidateTopK] = useState<number>(
    defaultMaxChunkCandidateTopK,
  )
  const [normalizeTopicScores, setNormalizeTopicScores] = useState<boolean>(false)
  const [retrievalModel, setRetrievalModel] = useState<RetrievalModel>('svd')
  const [minArticleYear, setMinArticleYear] = useState<number | null>(null)
  const [maxArticleYear, setMaxArticleYear] = useState<number | null>(null)
  const [minArticleCharacters, setMinArticleCharacters] = useState<number | null>(null)
  const [maxArticleCharacters, setMaxArticleCharacters] = useState<number | null>(null)
  const [minArticleWords, setMinArticleWords] = useState<number | null>(null)
  const [maxArticleWords, setMaxArticleWords] = useState<number | null>(null)
  const [minArticleReadingMinutes, setMinArticleReadingMinutes] = useState<number | null>(null)
  const [maxArticleReadingMinutes, setMaxArticleReadingMinutes] = useState<number | null>(null)
  const [yearStart, setYearStart] = useState<number | null>(null)
  const [yearEnd, setYearEnd] = useState<number | null>(null)
  const [yearStartInput, setYearStartInput] = useState<string>('')
  const [yearEndInput, setYearEndInput] = useState<string>('')
  const [lengthFilterUnit, setLengthFilterUnit] = useState<LengthFilterUnit>('characters')
  const [characterStart, setCharacterStart] = useState<number | null>(null)
  const [characterEnd, setCharacterEnd] = useState<number | null>(null)
  const [characterStartInput, setCharacterStartInput] = useState<string>('')
  const [characterEndInput, setCharacterEndInput] = useState<string>('')
  const [wordStart, setWordStart] = useState<number | null>(null)
  const [wordEnd, setWordEnd] = useState<number | null>(null)
  const [wordStartInput, setWordStartInput] = useState<string>('')
  const [wordEndInput, setWordEndInput] = useState<string>('')
  const [readingTimeStart, setReadingTimeStart] = useState<number | null>(null)
  const [readingTimeEnd, setReadingTimeEnd] = useState<number | null>(null)
  const [readingTimeStartInput, setReadingTimeStartInput] = useState<string>('')
  const [readingTimeEndInput, setReadingTimeEndInput] = useState<string>('')
  const [wordsToAvoid, setWordsToAvoid] = useState<string[]>([])
  const [wordsToAvoidDraft, setWordsToAvoidDraft] = useState<string>('')
  const [supportedRetrievalModels, setSupportedRetrievalModels] = useState<RetrievalModel[]>(
    defaultSupportedRetrievalModels,
  )
  const [articles, setArticles] = useState<Article[]>([])
  const [resultsOverview, setResultsOverview] = useState<ResultsOverview | null>(null)
  const [resultsOverviewLoading, setResultsOverviewLoading] = useState<boolean>(false)
  const [resultsOverviewError, setResultsOverviewError] = useState<string | null>(null)
  const [resultsChatMessages, setResultsChatMessages] = useState<ResultsChatMessage[]>([])
  const [resultsChatInput, setResultsChatInput] = useState<string>('')
  const [resultsChatArticleIds, setResultsChatArticleIds] = useState<string[]>([])
  const [resultsChatLoading, setResultsChatLoading] = useState<boolean>(false)
  const [resultsChatError, setResultsChatError] = useState<string | null>(null)
  const [isResultsChatMinimized, setIsResultsChatMinimized] = useState<boolean>(true)
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
  const [searchFocusSnapshot, setSearchFocusSnapshot] = useState<SearchFocusSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [emptyResultsMessage, setEmptyResultsMessage] = useState<string | null>(null)
  const [typoCorrection, setTypoCorrection] = useState<TypoCorrectionSuggestion | null>(null)
  const [isQueryAssistOpen, setIsQueryAssistOpen] = useState<boolean>(false)
  const [queryAssistMode, setQueryAssistMode] = useState<QueryAssistMode>('menu')
  const [queryAssistLoading, setQueryAssistLoading] = useState<boolean>(false)
  const [queryAssistError, setQueryAssistError] = useState<string | null>(null)
  const [queryRewriteOptions, setQueryRewriteOptions] = useState<QueryRewriteAlternative[]>([])
  const [queryImproveSuggestions, setQueryImproveSuggestions] = useState<string[]>([])
  const [activeTopNavPage, setActiveTopNavPage] = useState<TopNavPage>(
    hasSeenLandingRef.current ? 'search' : 'home',
  )
  const [activeAboutSection, setActiveAboutSection] = useState<AboutSection>('overview')
  const [activeAboutMethodTab, setActiveAboutMethodTab] = useState<InputMode>('stance')
  const [isFilterOpen, setIsFilterOpen] = useState<boolean>(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false)
  const [similarArticleSource, setSimilarArticleSource] = useState<Article | null>(null)
  const [similarArticles, setSimilarArticles] = useState<Article[]>([])
  const [similarArticlesOffset, setSimilarArticlesOffset] = useState<number>(0)
  const [similarArticlesHasMore, setSimilarArticlesHasMore] = useState<boolean>(false)
  const [similarArticlesLoading, setSimilarArticlesLoading] = useState<boolean>(false)
  const [similarArticlesError, setSimilarArticlesError] = useState<string | null>(null)
  const [settingsFocusTarget, setSettingsFocusTarget] = useState<SettingsFocusTarget | null>(null)
  const [essayCandidates, setEssayCandidates] = useState<EssayClaimCandidate[]>([])
  const [essayPreparedText, setEssayPreparedText] = useState<string>('')
  const [selectedEssayCandidateId, setSelectedEssayCandidateId] = useState<string | null>(null)
  const [essayCustomThesis, setEssayCustomThesis] = useState<string>('')
  const [essayThesisMode, setEssayThesisMode] = useState<EssayThesisMode>('candidate')
  const [essayActiveStep, setEssayActiveStep] = useState<EssayStep>(1)
  const essayOptionsRef = useRef<HTMLDivElement | null>(null)
  const topicPromptLabelRef = useRef<HTMLSpanElement | null>(null)
  const opinionPromptLabelRef = useRef<HTMLSpanElement | null>(null)
  const topicInputRef = useRef<HTMLInputElement | null>(null)
  const opinionInputRef = useRef<HTMLInputElement | null>(null)
  const essayTextAreaRef = useRef<HTMLTextAreaElement | null>(null)
  const essaySubmitCopyRef = useRef<HTMLParagraphElement | null>(null)
  const settingsScrollPaneRef = useRef<HTMLDivElement | null>(null)
  const retrievalGranularitySettingsRef = useRef<HTMLDivElement | null>(null)
  const topicSettingsRef = useRef<HTMLDivElement | null>(null)
  const agreementSettingsRef = useRef<HTMLDivElement | null>(null)
  const resultsSectionRef = useRef<HTMLDivElement | null>(null)
  const touchStartYRef = useRef<number | null>(null)
  const resultsOverviewRequestIdRef = useRef<number>(0)
  const queryAssistRequestIdRef = useRef<number>(0)
  const searchFocusKeyRef = useRef<number>(0)
  const searchFocusStartedAtRef = useRef<number>(0)
  const searchFocusTimeoutRef = useRef<number | null>(null)
  const lastAppliedFiltersRef = useRef<{
    yearStart: number | null
    yearEnd: number | null
    lengthFilterUnit: LengthFilterUnit
    characterStart: number | null
    characterEnd: number | null
    wordStart: number | null
    wordEnd: number | null
    readingTimeStart: number | null
    readingTimeEnd: number | null
    wordsToAvoidKey: string
  } | null>(null)
  const skipNextStanceResetRef = useRef<boolean>(false)
  const [isSearchStageVisible, setIsSearchStageVisible] = useState<boolean>(hasSeenLandingRef.current)
  const [hasSubmittedSearch, setHasSubmittedSearch] = useState<boolean>(false)
  const [shouldScrollToResults, setShouldScrollToResults] = useState<boolean>(true)
  const useChunking = chunkingMode !== 'none'
  const isSearchPageActive = activeTopNavPage === 'search'
  const isAboutPageActive = activeTopNavPage === 'about'
  const isSearchChromeVisible = isSearchPageActive || isAboutPageActive

  const clearSearchFocusTimer = (): void => {
    if (typeof window !== 'undefined' && searchFocusTimeoutRef.current !== null) {
      window.clearTimeout(searchFocusTimeoutRef.current)
    }
    searchFocusTimeoutRef.current = null
  }

  const startSearchFocus = (
    text: string,
    mode: InputMode,
    positionedWords: SearchFocusWordSnapshot[] = [],
  ): void => {
    clearSearchFocusTimer()
    const nextKey = searchFocusKeyRef.current + 1
    const fallbackText = mode === 'essay'
      ? 'Searching this essay'
      : 'Regarding this topic I believe this stance'
    const resolvedText = summarizeApiText(text, 220) || fallbackText
    const resolvedWords = positionedWords.length > 0
      ? addSearchFocusDrift(positionedWords)
      : buildFallbackSearchFocusWords(resolvedText, mode)

    searchFocusKeyRef.current = nextKey
    searchFocusStartedAtRef.current = Date.now()
    setSearchFocusSnapshot({
      key: nextKey,
      text: resolvedText,
      mode,
      words: resolvedWords,
      clearing: false,
    })
  }

  const finishSearchFocus = (): void => {
    const activeKey = searchFocusKeyRef.current
    const elapsedMs = Date.now() - searchFocusStartedAtRef.current
    const remainingMs = Math.max(0, searchFocusMinimumMs - elapsedMs)

    if (typeof window === 'undefined') {
      setSearchFocusSnapshot(null)
      return
    }

    clearSearchFocusTimer()
    searchFocusTimeoutRef.current = window.setTimeout(() => {
      setSearchFocusSnapshot(currentSnapshot => (
        currentSnapshot?.key === activeKey
          ? { ...currentSnapshot, clearing: true }
          : currentSnapshot
      ))
      searchFocusTimeoutRef.current = window.setTimeout(() => {
        setSearchFocusSnapshot(currentSnapshot => (
          currentSnapshot?.key === activeKey ? null : currentSnapshot
        ))
        searchFocusTimeoutRef.current = null
      }, searchFocusClearMs)
    }, remainingMs)
  }

  useEffect(() => (
    () => clearSearchFocusTimer()
  ), [])

  const buildStanceSearchFocusWords = (
    nextTopic: string,
    nextOpinion: string,
  ): SearchFocusWordSnapshot[] => {
    const topicLabelWords = measureSearchFocusSourceWords(
      'Regarding',
      topicPromptLabelRef.current,
      { maxWords: 1 },
    )
    const topicWords = measureSearchFocusSourceWords(
      nextTopic,
      topicInputRef.current,
      { maxWords: 14 },
    )
    const opinionLabelWords = measureSearchFocusSourceWords(
      'I believe',
      opinionPromptLabelRef.current,
      { maxWords: 2 },
    )
    const remainingWordCount = Math.max(
      4,
      searchFocusMaxWords - topicLabelWords.length - topicWords.length - opinionLabelWords.length,
    )
    const opinionWords = measureSearchFocusSourceWords(
      nextOpinion,
      opinionInputRef.current,
      { maxWords: remainingWordCount },
    )

    return [
      ...topicLabelWords,
      ...topicWords,
      ...opinionLabelWords,
      ...opinionWords,
    ].slice(0, searchFocusMaxWords)
  }

  const buildEssaySearchFocusWords = (
    nextEssayText: string,
    nextThesisSentence: string,
  ): SearchFocusWordSnapshot[] => {
    const sourceElement = nextThesisSentence
      ? essaySubmitCopyRef.current
      : essayTextAreaRef.current
    const sourceText = nextThesisSentence || summarizeApiText(nextEssayText, 170)

    return measureSearchFocusSourceWords(
      sourceText,
      sourceElement,
      {
        maxWords: searchFocusMaxWords,
        wrap: true,
      },
    )
  }

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
        const llmAvailable = Boolean(data.llm_agreement_available)
        const preferredModel = isRetrievalModel(data.default_retrieval_model)
          ? data.default_retrieval_model
          : supportedModels[0]
        const resolvedModel = supportedModels.includes(preferredModel)
          ? preferredModel
          : supportedModels[0]
        const preferredStanceMethod = isStanceMethod(data.default_stance_method)
          ? data.default_stance_method
          : 'llm'
        const resolvedStanceMethod = resolvePreferredStanceMethod(
          preferredStanceMethod,
          supportedAgreementMethods,
          llmAvailable,
        )
        const preferredChunkingMode = isFrontendChunkingMode(data.default_chunking_mode)
          ? data.default_chunking_mode
          : (data.default_use_chunking ? 'semantic' : defaultChunkingMode)
        const resolvedChunkingMode = resolvePreferredChunkingMode(
          preferredChunkingMode,
          supportedChunkingOptions,
          llmAvailable,
        )
        const nextMinArticleYear = normalizeConfigYear(data.min_article_year)
        const nextMaxArticleYear = normalizeConfigYear(data.max_article_year)
        const nextMinArticleCharacters = normalizeConfigInteger(data.min_article_characters)
        const nextMaxArticleCharacters = normalizeConfigInteger(data.max_article_characters)
        const nextMinArticleWords = normalizeConfigInteger(data.min_article_words)
        const nextMaxArticleWords = normalizeConfigInteger(data.max_article_words)
        const nextMinArticleReadingMinutes = normalizeConfigInteger(data.min_article_reading_minutes)
        const nextMaxArticleReadingMinutes = normalizeConfigInteger(data.max_article_reading_minutes)
        setSupportedRetrievalModels(supportedModels)
        setSupportedStanceMethods(supportedAgreementMethods)
        setSupportedChunkingModes(supportedChunkingOptions)
        setLlmAgreementAvailable(llmAvailable)
        setRetrievalModel(currentModel => (
          supportedModels.includes(currentModel) ? currentModel : resolvedModel
        ))
        setStanceMethod(resolvedStanceMethod)
        setChunkingMode(resolvedChunkingMode)
        setRerankSelectionMode(currentMode => (
          isRerankSelectionMode(data.default_rerank_selection_mode)
            ? data.default_rerank_selection_mode
            : currentMode
        ))
        setAutoRerankThresholds(normalizeAutoRerankThresholds(data.default_auto_rerank_thresholds))
        setAutoChunkRerankThresholds(normalizeAutoRerankThresholds(
          data.default_chunk_auto_rerank_thresholds,
          defaultChunkAutoRerankThresholds,
        ))
        if (
          typeof data.max_auto_rerank_candidates === 'number'
          && Number.isFinite(data.max_auto_rerank_candidates)
          && data.max_auto_rerank_candidates > 0
        ) {
          setMaxAutoRerankCandidates(Math.round(data.max_auto_rerank_candidates))
        }
        if (
          typeof data.max_chunk_candidate_top_k === 'number'
          && Number.isFinite(data.max_chunk_candidate_top_k)
          && data.max_chunk_candidate_top_k > 0
        ) {
          setMaxChunkCandidateTopK(Math.round(data.max_chunk_candidate_top_k))
        }
        if (
          typeof data.default_chunk_candidate_top_k === 'number'
          && Number.isFinite(data.default_chunk_candidate_top_k)
          && data.default_chunk_candidate_top_k > 0
        ) {
          setChunkCandidateTopK(Math.round(data.default_chunk_candidate_top_k))
        }
        if (
          typeof data.default_chunk_article_top_k === 'number'
          && Number.isFinite(data.default_chunk_article_top_k)
          && data.default_chunk_article_top_k > 0
        ) {
          setChunkArticleTopK(Math.round(data.default_chunk_article_top_k))
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
        if (
          nextMinArticleCharacters !== null &&
          nextMaxArticleCharacters !== null &&
          nextMinArticleCharacters <= nextMaxArticleCharacters
        ) {
          setMinArticleCharacters(nextMinArticleCharacters)
          setMaxArticleCharacters(nextMaxArticleCharacters)
          setCharacterStart(currentCharacters => (
            currentCharacters === null
              ? nextMinArticleCharacters
              : clampCharacterCount(currentCharacters, nextMinArticleCharacters, nextMaxArticleCharacters)
          ))
          setCharacterEnd(currentCharacters => (
            currentCharacters === null
              ? nextMaxArticleCharacters
              : clampCharacterCount(currentCharacters, nextMinArticleCharacters, nextMaxArticleCharacters)
          ))
        }
        if (
          nextMinArticleWords !== null &&
          nextMaxArticleWords !== null &&
          nextMinArticleWords <= nextMaxArticleWords
        ) {
          setMinArticleWords(nextMinArticleWords)
          setMaxArticleWords(nextMaxArticleWords)
          setWordStart(currentWords => (
            currentWords === null
              ? nextMinArticleWords
              : clampWordCount(currentWords, nextMinArticleWords, nextMaxArticleWords)
          ))
          setWordEnd(currentWords => (
            currentWords === null
              ? nextMaxArticleWords
              : clampWordCount(currentWords, nextMinArticleWords, nextMaxArticleWords)
          ))
        }
        if (
          nextMinArticleReadingMinutes !== null &&
          nextMaxArticleReadingMinutes !== null &&
          nextMinArticleReadingMinutes <= nextMaxArticleReadingMinutes
        ) {
          setMinArticleReadingMinutes(nextMinArticleReadingMinutes)
          setMaxArticleReadingMinutes(nextMaxArticleReadingMinutes)
          setReadingTimeStart(currentMinutes => (
            currentMinutes === null
              ? nextMinArticleReadingMinutes
              : clampReadingMinutes(currentMinutes, nextMinArticleReadingMinutes, nextMaxArticleReadingMinutes)
          ))
          setReadingTimeEnd(currentMinutes => (
            currentMinutes === null
              ? nextMaxArticleReadingMinutes
              : clampReadingMinutes(currentMinutes, nextMinArticleReadingMinutes, nextMaxArticleReadingMinutes)
          ))
        }
        setNormalizeTopicScores(Boolean(data.default_normalize_topic_scores))
      } catch (configError) {
        console.error('Config request failed:', configError)
        if (!isActive) return
        setUseLlm(false)
        setRerankSelectionMode(defaultRerankSelectionMode)
        setStanceMethod(resolvePreferredStanceMethod(defaultStanceMethod, defaultSupportedStanceMethods, false))
        setChunkingMode(resolvePreferredChunkingMode(defaultChunkingMode, defaultSupportedChunkingModes, false))
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
    if (activeTopNavPage !== 'home') {
      return
    }

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
  }, [activeTopNavPage, introSequenceKey, isSearchStageVisible])

  useEffect(() => {
    if (inputMode !== 'stance') {
      return
    }
    if (skipNextStanceResetRef.current) {
      skipNextStanceResetRef.current = false
      return
    }
    setArticles([])
    resetResultsOverview()
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setEmptyResultsMessage(null)
    setTypoCorrection(null)
    setError(null)
    setHasSubmittedSearch(false)
  }, [chunkArticleTopK, chunkCandidateTopK, chunkingMode, inputMode, opinion, recencyWeight, rerankTopK, stanceMethod, stanceWeight, topic, topicWeight])

  useEffect(() => {
    setArticles([])
    resetResultsOverview()
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setEmptyResultsMessage(null)
    setTypoCorrection(null)
    setError(null)
    setHasSubmittedSearch(false)
  }, [
    autoRerankThresholds,
    autoChunkRerankThresholds,
    chunkArticleTopK,
    chunkCandidateTopK,
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
    queryAssistRequestIdRef.current += 1
    if (inputMode !== 'stance') {
      setIsQueryAssistOpen(false)
    }
    setQueryAssistLoading(false)
    setQueryAssistError(null)
    setQueryRewriteOptions([])
    setQueryImproveSuggestions([])
    setQueryAssistMode('menu')
  }, [inputMode, opinion, retrievalModel, topic])

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
    resetResultsOverview()
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setEmptyResultsMessage(null)
    setTypoCorrection(null)
    setError(null)
    setHasSubmittedSearch(false)
  }, [inputMode, searchTerm])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      setIsFilterOpen(false)
      setIsSettingsOpen(false)
      setIsQueryAssistOpen(false)
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
      const target = settingsFocusTarget === 'retrieval-granularity'
        ? retrievalGranularitySettingsRef.current
        : (settingsFocusTarget === 'topic-relevance'
          ? topicSettingsRef.current
          : agreementSettingsRef.current)
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
    setActiveTopNavPage('search')
    setIsSearchStageVisible(true)
  }

  const scrollPageToTop = (): void => {
    if (typeof window !== 'undefined') {
      window.scrollTo({
        top: 0,
        behavior: 'auto',
      })
    }
  }

  const showHomePage = (): void => {
    scrollPageToTop()
    if (typeof document !== 'undefined') {
      document.body.style.overflow = ''
    }

    clearSearchFocusTimer()
    touchStartYRef.current = null
    setSearchFocusSnapshot(null)
    setActiveTopNavPage('home')
    setHasSubmittedSearch(false)
    setIsSearchStageVisible(false)
    setIntroStage(0)
    setTypedTopic('')
    setTypedClaim('')
    setIntroSequenceKey(currentKey => currentKey + 1)
  }

  const showSearchPage = (): void => {
    activateSearchStage(true)
  }

  const openAboutPage = (): void => {
    scrollPageToTop()
    setActiveAboutMethodTab(inputMode)
    setActiveTopNavPage('about')
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (activeTopNavPage !== 'home') return
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
  }, [activeTopNavPage, introStage, isSearchStageVisible])

  useEffect(() => {
    if (typeof document === 'undefined') return

    const previousOverflow = document.body.style.overflow

    if (isSearchPageActive && isSearchStageVisible && inputMode === 'stance' && !hasSubmittedSearch) {
      document.body.style.overflow = 'hidden'
      return () => {
        document.body.style.overflow = previousOverflow
      }
    }

    document.body.style.overflow = previousOverflow

    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [hasSubmittedSearch, inputMode, isSearchPageActive, isSearchStageVisible])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!hasSubmittedSearch) return
    if (!shouldScrollToResults) return

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
  }, [hasSubmittedSearch, shouldScrollToResults])

  const trimmedEssayText = searchTerm.trim()
  const trimmedTopic = topic.trim()
  const trimmedOpinion = opinion.trim()
  const hasQueryAssistInput = trimmedTopic !== '' || trimmedOpinion !== ''
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
  const canUseMiniLm = supportedRetrievalModels.includes('minilm')
  const canUseTfidf = supportedRetrievalModels.includes('tfidf')
  const canUseNliAgreement = supportedStanceMethods.includes('nli')
  const canUseLlmAgreement = supportedStanceMethods.includes('llm') && llmAgreementAvailable
  const canUseChunking = canUseLlmAgreement && supportedChunkingModes.includes('semantic')
  const canUseLexicalRetrieval = canUseTfidf && !useChunking
  const firstSemanticRetrievalModel: RetrievalModel | null = canUseSvd
    ? 'svd'
    : (canUseMiniLm ? 'minilm' : null)
  const effectiveRetrievalModel: RetrievalModel = (
    useChunking && retrievalModel === 'tfidf' && firstSemanticRetrievalModel
      ? firstSemanticRetrievalModel
      : retrievalModel
  )
  const effectiveRetrievalLabel = (
    effectiveRetrievalModel === 'tfidf'
      ? 'Lexical'
      : (effectiveRetrievalModel === 'minilm' ? 'Enhanced Semantic' : 'Semantic')
  )
  const isLexicalSearchMode = !useChunking && retrievalModel === 'tfidf'
  const queryAssistDisabledReason = !hasQueryAssistInput
    ? 'Add a topic or stance to use AI query help.'
    : (useLlm !== true
      ? 'AI query help is turned off in the backend config.'
      : (!llmAgreementAvailable
        ? 'AI query help needs SPARK_API_KEY or API_KEY in your backend environment.'
        : ''))
  const canUseQueryAssist = queryAssistDisabledReason === ''
  const activeWordsToAvoid = isLexicalSearchMode ? wordsToAvoid : []
  const activeWordsToAvoidKey = activeWordsToAvoid.join('\u0000')
  const currentAutoRerankThreshold = (
    useChunking ? autoChunkRerankThresholds : autoRerankThresholds
  )[effectiveRetrievalModel]

  useEffect(() => {
    if (useChunking && retrievalModel === 'tfidf' && firstSemanticRetrievalModel) {
      setRetrievalModel(firstSemanticRetrievalModel)
    }
  }, [firstSemanticRetrievalModel, retrievalModel, useChunking])

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
  const yearRangeSummary = !isYearFilterActive || !hasYearRangeSelection
    ? ''
    : (resolvedYearStart === resolvedYearEnd
      ? ` from ${resolvedYearStart}`
      : ` from ${resolvedYearStart} to ${resolvedYearEnd}`)
  const hasAvailableCharacterBounds = (
    minArticleCharacters !== null &&
    maxArticleCharacters !== null &&
    minArticleCharacters <= maxArticleCharacters
  )
  const resolvedCharacterStart = characterStart ?? minArticleCharacters
  const resolvedCharacterEnd = characterEnd ?? maxArticleCharacters
  const characterRangeSpan = hasAvailableCharacterBounds && minArticleCharacters !== null && maxArticleCharacters !== null
    ? maxArticleCharacters - minArticleCharacters
    : 0
  const hasCharacterRangeSelection = resolvedCharacterStart !== null && resolvedCharacterEnd !== null
  const isCharacterFilterActive = (
    hasCharacterRangeSelection &&
    minArticleCharacters !== null &&
    maxArticleCharacters !== null &&
    (resolvedCharacterStart !== minArticleCharacters || resolvedCharacterEnd !== maxArticleCharacters)
  )
  const characterRangeSummary = !isCharacterFilterActive || !hasCharacterRangeSelection
    ? ''
    : (resolvedCharacterStart === resolvedCharacterEnd
      ? ` with ${formatCharacterCount(resolvedCharacterStart)} characters`
      : ` with ${formatCharacterCount(resolvedCharacterStart)} to ${formatCharacterCount(resolvedCharacterEnd)} characters`)
  const hasAvailableWordBounds = (
    minArticleWords !== null &&
    maxArticleWords !== null &&
    minArticleWords <= maxArticleWords
  )
  const resolvedWordStart = wordStart ?? minArticleWords
  const resolvedWordEnd = wordEnd ?? maxArticleWords
  const wordRangeSpan = hasAvailableWordBounds && minArticleWords !== null && maxArticleWords !== null
    ? maxArticleWords - minArticleWords
    : 0
  const hasWordRangeSelection = resolvedWordStart !== null && resolvedWordEnd !== null
  const isWordFilterActive = (
    hasWordRangeSelection &&
    minArticleWords !== null &&
    maxArticleWords !== null &&
    (resolvedWordStart !== minArticleWords || resolvedWordEnd !== maxArticleWords)
  )
  const wordRangeSummary = !isWordFilterActive || !hasWordRangeSelection
    ? ''
    : (resolvedWordStart === resolvedWordEnd
      ? ` with ${formatWordCount(resolvedWordStart)} words`
      : ` with ${formatWordCount(resolvedWordStart)} to ${formatWordCount(resolvedWordEnd)} words`)
  const hasAvailableReadingTimeBounds = (
    minArticleReadingMinutes !== null &&
    maxArticleReadingMinutes !== null &&
    minArticleReadingMinutes <= maxArticleReadingMinutes
  )
  const resolvedReadingTimeStart = readingTimeStart ?? minArticleReadingMinutes
  const resolvedReadingTimeEnd = readingTimeEnd ?? maxArticleReadingMinutes
  const readingTimeRangeSpan = hasAvailableReadingTimeBounds && minArticleReadingMinutes !== null && maxArticleReadingMinutes !== null
    ? maxArticleReadingMinutes - minArticleReadingMinutes
    : 0
  const hasReadingTimeRangeSelection = resolvedReadingTimeStart !== null && resolvedReadingTimeEnd !== null
  const isReadingTimeFilterActive = (
    hasReadingTimeRangeSelection &&
    minArticleReadingMinutes !== null &&
    maxArticleReadingMinutes !== null &&
    (resolvedReadingTimeStart !== minArticleReadingMinutes || resolvedReadingTimeEnd !== maxArticleReadingMinutes)
  )
  const readingTimeRangeSummary = !isReadingTimeFilterActive || !hasReadingTimeRangeSelection
    ? ''
    : (resolvedReadingTimeStart === resolvedReadingTimeEnd
      ? ` with a ${formatReadingMinutes(resolvedReadingTimeStart)} min read`
      : ` with a ${formatReadingMinutes(resolvedReadingTimeStart)} to ${formatReadingMinutes(resolvedReadingTimeEnd)} min read`)
  const avoidWordsFilterSummary = activeWordsToAvoid.length === 0
    ? ''
    : ` avoiding ${activeWordsToAvoid.length === 1 ? activeWordsToAvoid[0] : `${activeWordsToAvoid.length} words`}`
  const selectedLengthRangeSummary = lengthFilterUnit === 'reading_time'
    ? readingTimeRangeSummary
    : (lengthFilterUnit === 'words'
      ? wordRangeSummary
      : characterRangeSummary)
  const activeFilterSummary = `${yearRangeSummary}${selectedLengthRangeSummary}${avoidWordsFilterSummary}`
  const selectedLengthFilterHasBounds = lengthFilterUnit === 'reading_time'
    ? hasAvailableReadingTimeBounds
    : (lengthFilterUnit === 'words' ? hasAvailableWordBounds : hasAvailableCharacterBounds)
  const selectedLengthRangeSpan = lengthFilterUnit === 'reading_time'
    ? readingTimeRangeSpan
    : (lengthFilterUnit === 'words' ? wordRangeSpan : characterRangeSpan)
  const selectedLengthRangeStartInput = lengthFilterUnit === 'reading_time'
    ? readingTimeStartInput
    : (lengthFilterUnit === 'words' ? wordStartInput : characterStartInput)
  const selectedLengthRangeEndInput = lengthFilterUnit === 'reading_time'
    ? readingTimeEndInput
    : (lengthFilterUnit === 'words' ? wordEndInput : characterEndInput)
  const selectedLengthRangeMin = lengthFilterUnit === 'reading_time'
    ? minArticleReadingMinutes
    : (lengthFilterUnit === 'words' ? minArticleWords : minArticleCharacters)
  const selectedLengthRangeMax = lengthFilterUnit === 'reading_time'
    ? maxArticleReadingMinutes
    : (lengthFilterUnit === 'words' ? maxArticleWords : maxArticleCharacters)
  const selectedLengthRangeStart = lengthFilterUnit === 'reading_time'
    ? resolvedReadingTimeStart
    : (lengthFilterUnit === 'words' ? resolvedWordStart : resolvedCharacterStart)
  const selectedLengthRangeEnd = lengthFilterUnit === 'reading_time'
    ? resolvedReadingTimeEnd
    : (lengthFilterUnit === 'words' ? resolvedWordEnd : resolvedCharacterEnd)

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

  const parseChunkCandidateTopKInput = (value: string, fallback: number): number => {
    if (value.trim() === '') return fallback
    const parsed = Number(value)
    if (Number.isNaN(parsed)) return fallback
    return Math.min(maxChunkCandidateTopK, Math.max(25, Math.round(parsed)))
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
      currentAutoRerankThreshold,
    )
    if (useChunking) {
      setAutoChunkRerankThresholds(currentThresholds => ({
        ...currentThresholds,
        [effectiveRetrievalModel]: nextThreshold,
      }))
      return
    }
    setAutoRerankThresholds(currentThresholds => ({
      ...currentThresholds,
      [retrievalModel]: nextThreshold,
    }))
  }

  const renderTypoHighlightedQuery = (suggestion: TypoCorrectionSuggestion): Array<string | JSX.Element> => {
    const highlightedTerms = new Set(suggestion.highlighted_terms.map(normalizeTypoTerm))
    const nodes: Array<string | JSX.Element> = []
    let lastIndex = 0
    let tokenIndex = 0

    for (const match of suggestion.query.matchAll(typoTokenPattern)) {
      const token = match[0]
      const index = match.index ?? 0
      if (index > lastIndex) {
        nodes.push(suggestion.query.slice(lastIndex, index))
      }

      if (highlightedTerms.has(normalizeTypoTerm(token))) {
        nodes.push(
          <span key={`typo-${tokenIndex}`} className="typo-incorrect-word">
            {token}
          </span>,
        )
      } else {
        nodes.push(token)
      }
      lastIndex = index + token.length
      tokenIndex += 1
    }

    if (lastIndex < suggestion.query.length) {
      nodes.push(suggestion.query.slice(lastIndex))
    }

    return nodes.length > 0 ? nodes : [suggestion.query]
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

  const handleCharacterStartChange = (value: string): void => {
    const nextStart = Number(value)
    if (Number.isNaN(nextStart)) return
    const boundedStart = (
      minArticleCharacters !== null && maxArticleCharacters !== null
        ? clampCharacterCount(nextStart, minArticleCharacters, maxArticleCharacters)
        : Math.round(nextStart)
    )
    setCharacterStart(boundedStart)
    setCharacterEnd(currentEnd => (
      currentEnd === null || currentEnd < boundedStart ? boundedStart : currentEnd
    ))
  }

  const handleCharacterEndChange = (value: string): void => {
    const nextEnd = Number(value)
    if (Number.isNaN(nextEnd)) return
    const boundedEnd = (
      minArticleCharacters !== null && maxArticleCharacters !== null
        ? clampCharacterCount(nextEnd, minArticleCharacters, maxArticleCharacters)
        : Math.round(nextEnd)
    )
    setCharacterEnd(boundedEnd)
    setCharacterStart(currentStart => (
      currentStart === null || currentStart > boundedEnd ? boundedEnd : currentStart
    ))
  }

  const handleCharacterStartInputChange = (value: string): void => {
    setCharacterStartInput(value.replace(/[^\d]/g, ''))
  }

  const handleCharacterEndInputChange = (value: string): void => {
    setCharacterEndInput(value.replace(/[^\d]/g, ''))
  }

  const handleWordStartChange = (value: string): void => {
    const nextStart = Number(value)
    if (Number.isNaN(nextStart)) return
    const boundedStart = (
      minArticleWords !== null && maxArticleWords !== null
        ? clampWordCount(nextStart, minArticleWords, maxArticleWords)
        : Math.round(nextStart)
    )
    setWordStart(boundedStart)
    setWordEnd(currentEnd => (
      currentEnd === null || currentEnd < boundedStart ? boundedStart : currentEnd
    ))
  }

  const handleWordEndChange = (value: string): void => {
    const nextEnd = Number(value)
    if (Number.isNaN(nextEnd)) return
    const boundedEnd = (
      minArticleWords !== null && maxArticleWords !== null
        ? clampWordCount(nextEnd, minArticleWords, maxArticleWords)
        : Math.round(nextEnd)
    )
    setWordEnd(boundedEnd)
    setWordStart(currentStart => (
      currentStart === null || currentStart > boundedEnd ? boundedEnd : currentStart
    ))
  }

  const handleWordStartInputChange = (value: string): void => {
    setWordStartInput(value.replace(/[^\d]/g, ''))
  }

  const handleWordEndInputChange = (value: string): void => {
    setWordEndInput(value.replace(/[^\d]/g, ''))
  }

  const handleReadingTimeStartChange = (value: string): void => {
    const nextStart = Number(value)
    if (Number.isNaN(nextStart)) return
    const boundedStart = (
      minArticleReadingMinutes !== null && maxArticleReadingMinutes !== null
        ? clampReadingMinutes(nextStart, minArticleReadingMinutes, maxArticleReadingMinutes)
        : Math.round(nextStart)
    )
    setReadingTimeStart(boundedStart)
    setReadingTimeEnd(currentEnd => (
      currentEnd === null || currentEnd < boundedStart ? boundedStart : currentEnd
    ))
  }

  const handleReadingTimeEndChange = (value: string): void => {
    const nextEnd = Number(value)
    if (Number.isNaN(nextEnd)) return
    const boundedEnd = (
      minArticleReadingMinutes !== null && maxArticleReadingMinutes !== null
        ? clampReadingMinutes(nextEnd, minArticleReadingMinutes, maxArticleReadingMinutes)
        : Math.round(nextEnd)
    )
    setReadingTimeEnd(boundedEnd)
    setReadingTimeStart(currentStart => (
      currentStart === null || currentStart > boundedEnd ? boundedEnd : currentStart
    ))
  }

  const handleReadingTimeStartInputChange = (value: string): void => {
    setReadingTimeStartInput(value.replace(/[^\d]/g, ''))
  }

  const handleReadingTimeEndInputChange = (value: string): void => {
    setReadingTimeEndInput(value.replace(/[^\d]/g, ''))
  }

  const addWordsToAvoid = (value: string): void => {
    const nextWords = parseWordsToAvoid(value)
    if (nextWords.length === 0) return

    setWordsToAvoid(currentWords => {
      const seen = new Set(currentWords)
      const mergedWords = [...currentWords]
      for (const word of nextWords) {
        if (seen.has(word)) continue
        mergedWords.push(word)
        seen.add(word)
      }
      return mergedWords
    })
    setWordsToAvoidDraft('')
  }

  const removeWordToAvoid = (word: string): void => {
    setWordsToAvoid(currentWords => currentWords.filter(currentWord => currentWord !== word))
  }

  const handleWordsToAvoidKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>): void => {
    if (!isLexicalSearchMode) return

    if (event.key === 'Enter') {
      event.preventDefault()
      addWordsToAvoid(wordsToAvoidDraft)
      return
    }

    if (event.key === 'Backspace' && wordsToAvoidDraft === '') {
      setWordsToAvoid(currentWords => currentWords.slice(0, -1))
    }
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

  const commitCharacterStartInput = (): void => {
    const normalizedValue = characterStartInput.trim()
    if (normalizedValue === '') {
      if (minArticleCharacters !== null) {
        handleCharacterStartChange(String(minArticleCharacters))
      }
      return
    }
    handleCharacterStartChange(normalizedValue)
  }

  const commitCharacterEndInput = (): void => {
    const normalizedValue = characterEndInput.trim()
    if (normalizedValue === '') {
      if (maxArticleCharacters !== null) {
        handleCharacterEndChange(String(maxArticleCharacters))
      }
      return
    }
    handleCharacterEndChange(normalizedValue)
  }

  const commitWordStartInput = (): void => {
    const normalizedValue = wordStartInput.trim()
    if (normalizedValue === '') {
      if (minArticleWords !== null) {
        handleWordStartChange(String(minArticleWords))
      }
      return
    }
    handleWordStartChange(normalizedValue)
  }

  const commitWordEndInput = (): void => {
    const normalizedValue = wordEndInput.trim()
    if (normalizedValue === '') {
      if (maxArticleWords !== null) {
        handleWordEndChange(String(maxArticleWords))
      }
      return
    }
    handleWordEndChange(normalizedValue)
  }

  const commitReadingTimeStartInput = (): void => {
    const normalizedValue = readingTimeStartInput.trim()
    if (normalizedValue === '') {
      if (minArticleReadingMinutes !== null) {
        handleReadingTimeStartChange(String(minArticleReadingMinutes))
      }
      return
    }
    handleReadingTimeStartChange(normalizedValue)
  }

  const commitReadingTimeEndInput = (): void => {
    const normalizedValue = readingTimeEndInput.trim()
    if (normalizedValue === '') {
      if (maxArticleReadingMinutes !== null) {
        handleReadingTimeEndChange(String(maxArticleReadingMinutes))
      }
      return
    }
    handleReadingTimeEndChange(normalizedValue)
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

  const resetResultsOverview = (): void => {
    resultsOverviewRequestIdRef.current += 1
    setResultsOverview(null)
    setResultsOverviewError(null)
    setResultsOverviewLoading(false)
    resetResultsChat()
  }

  const resetResultsChat = (): void => {
    setResultsChatMessages([])
    setResultsChatInput('')
    setResultsChatArticleIds([])
    setResultsChatError(null)
    setResultsChatLoading(false)
    setIsResultsChatMinimized(true)
  }

  const requestResultsOverview = async (
    query: string,
    nextArticles: Article[],
    mode: InputMode,
  ): Promise<void> => {
    if (!query.trim() || nextArticles.length === 0) {
      resetResultsOverview()
      return
    }

    if (useLlm !== true) {
      setResultsOverview(null)
      setResultsOverviewError('AI overview is turned off in the backend config.')
      setResultsOverviewLoading(false)
      return
    }

    if (!llmAgreementAvailable) {
      setResultsOverview(null)
      setResultsOverviewError('AI overview needs SPARK_API_KEY or API_KEY in your backend environment.')
      setResultsOverviewLoading(false)
      return
    }

    const requestId = resultsOverviewRequestIdRef.current + 1
    resultsOverviewRequestIdRef.current = requestId
    setResultsOverview(null)
    setResultsOverviewError(null)
    setResultsOverviewLoading(true)

    try {
      const response = await fetch('/api/llm/results-overview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          mode,
          articles: nextArticles.slice(0, 10),
        }),
      })

      const data = await readApiJson<ResultsOverview>(response)
      if (resultsOverviewRequestIdRef.current !== requestId) return
      if (!data || typeof data.overview !== 'string' || data.overview.trim() === '') {
        throw new Error('Invalid response from the results overview API.')
      }

      setResultsOverview({
        overview: data.overview.trim(),
        key_points: Array.isArray(data.key_points)
          ? data.key_points.map(point => String(point).trim()).filter(Boolean).slice(0, 4)
          : [],
        supporting_arguments: normalizeResultsOverviewArguments(data.supporting_arguments),
        opposing_arguments: normalizeResultsOverviewArguments(data.opposing_arguments),
        key_evidence: normalizeResultsOverviewEvidence(data.key_evidence),
        sources: normalizeResultsOverviewSources(data.sources),
        caveat: typeof data.caveat === 'string' ? data.caveat.trim() : '',
      })
      setResultsOverviewError(null)
    } catch (fetchError) {
      if (resultsOverviewRequestIdRef.current !== requestId) return
      console.error('Results overview failed:', fetchError)
      setResultsOverview(null)
      setResultsOverviewError(fetchError instanceof Error ? fetchError.message : 'Results overview failed.')
    } finally {
      if (resultsOverviewRequestIdRef.current === requestId) {
        setResultsOverviewLoading(false)
      }
    }
  }

  const buildResultsChatQuery = (): string => {
    if (inputMode === 'essay') {
      return [
        resolvedEssayThesis ? `Thesis: ${resolvedEssayThesis}` : null,
        essayPreparedText.trim() ? `Essay: ${essayPreparedText.trim()}` : null,
      ].filter(Boolean).join('\n')
    }

    return [
      topic.trim() ? `Topic: ${topic.trim()}` : null,
      opinion.trim() ? `Stance: ${opinion.trim()}` : null,
    ].filter(Boolean).join('\n')
  }

  const handleSubmitResultsChat = async (event: FormEvent<HTMLFormElement>): Promise<void> => {
    event.preventDefault()

    const question = resultsChatInput.trim()
    if (!question || resultsChatLoading) return

    if (useLlm !== true) {
      setResultsChatError('Results chat is turned off in the backend config.')
      return
    }

    if (!llmAgreementAvailable) {
      setResultsChatError('Results chat needs SPARK_API_KEY or API_KEY in your backend environment.')
      return
    }

    const resultArticles = getResultsChatContextArticles()
    if (resultArticles.length === 0) {
      setResultsChatError('There are no results available to chat about.')
      return
    }

    const userAttachments = resultsChatAttachments.length > 0 ? resultsChatAttachments : null
    const userMessage: ResultsChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: question,
      attachments: userAttachments,
    }
    const nextMessages = [...resultsChatMessages, userMessage]
    setResultsChatMessages(nextMessages)
    setResultsChatInput('')
    if (userAttachments) {
      setResultsChatArticleIds([])
    }
    setResultsChatError(null)
    setResultsChatLoading(true)

    try {
      const response = await fetch('/api/llm/results-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          query: buildResultsChatQuery(),
          mode: inputMode,
          article_scope: resultsChatAttachments.length > 0 ? 'selected' : 'top_results',
          articles: resultArticles,
          history: resultsChatMessages.slice(-6).map(message => ({
            role: message.role,
            content: message.content,
          })),
        }),
      })

      const data = await readApiJson<ResultsChatResponse>(response)
      if (!data || typeof data.answer !== 'string' || data.answer.trim() === '') {
        throw new Error('Invalid response from the results chat API.')
      }

      setResultsChatMessages([
        ...nextMessages,
        {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: data.answer.trim(),
          source_indices: normalizeResultsOverviewSourceIndices(data.source_indices),
          sources: normalizeResultsOverviewSources(data.sources),
        },
      ])
    } catch (fetchError) {
      console.error('Results chat failed:', fetchError)
      setResultsChatMessages(nextMessages)
      setResultsChatError(fetchError instanceof Error ? fetchError.message : 'Results chat failed.')
    } finally {
      setResultsChatLoading(false)
    }
  }

  const handleToggleQueryAssist = (): void => {
    if (!canUseQueryAssist) return
    setIsQueryAssistOpen(currentOpen => !currentOpen)
  }

  const requestQueryAssist = async (nextMode: Exclude<QueryAssistMode, 'menu'>): Promise<void> => {
    if (!canUseQueryAssist || queryAssistLoading) return

    const requestId = queryAssistRequestIdRef.current + 1
    queryAssistRequestIdRef.current = requestId
    setQueryAssistMode(nextMode)
    setQueryAssistLoading(true)
    setQueryAssistError(null)
    setQueryRewriteOptions([])
    setQueryImproveSuggestions([])

    try {
      const response = await fetch('/api/llm/query-help', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: nextMode === 'rewrite' ? 'rewrite' : 'suggest',
          topic: trimmedTopic,
          opinion: trimmedOpinion,
          retrieval_model: effectiveRetrievalModel,
        }),
      })

      const data = await readApiJson<QueryHelpResponse>(response)
      if (queryAssistRequestIdRef.current !== requestId) return

      if (nextMode === 'rewrite') {
        const alternatives = normalizeQueryRewriteAlternatives(data.alternatives)
        if (alternatives.length === 0) {
          throw new Error('No query alternatives were returned.')
        }
        setQueryRewriteOptions(alternatives)
        return
      }

      const suggestions = normalizeQueryImproveSuggestions(data.suggestions)
      if (suggestions.length === 0) {
        throw new Error('No query suggestions were returned.')
      }
      setQueryImproveSuggestions(suggestions)
    } catch (fetchError) {
      if (queryAssistRequestIdRef.current !== requestId) return
      console.error('Query help failed:', fetchError)
      setQueryAssistError(fetchError instanceof Error ? fetchError.message : 'Query help failed.')
    } finally {
      if (queryAssistRequestIdRef.current === requestId) {
        setQueryAssistLoading(false)
      }
    }
  }

  const handleApplyQueryRewrite = (alternative: QueryRewriteAlternative): void => {
    setTopic(alternative.topic)
    setOpinion(alternative.opinion)
    setTypoCorrection(null)
    setIsQueryAssistOpen(false)
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
    setTypoCorrection(null)
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
    resetResultsOverview()
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setTypoCorrection(null)
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
    const nextTopic = (options.topicOverride ?? trimmedTopic).trim()
    if (inputMode !== 'stance' || !nextTopic || !trimmedOpinion || loading) return
    const feedbackArticleIds = options.topicFeedbackIrrelevantArticleIds ?? topicFeedbackIrrelevantArticleIds

    lastAppliedFiltersRef.current = {
      yearStart: resolvedYearStart,
      yearEnd: resolvedYearEnd,
      lengthFilterUnit,
      characterStart: resolvedCharacterStart,
      characterEnd: resolvedCharacterEnd,
      wordStart: resolvedWordStart,
      wordEnd: resolvedWordEnd,
      readingTimeStart: resolvedReadingTimeStart,
      readingTimeEnd: resolvedReadingTimeEnd,
      wordsToAvoidKey: activeWordsToAvoidKey,
    }
    setHasSubmittedSearch(true)
    setShouldScrollToResults(false)
    if (typeof document !== 'undefined') {
      document.body.style.overflow = ''
    }
    startSearchFocus(
      [
        nextTopic ? `Regarding ${nextTopic}` : null,
        trimmedOpinion ? `I believe ${trimmedOpinion}` : null,
      ].filter(Boolean).join(' '),
      'stance',
      buildStanceSearchFocusWords(nextTopic, trimmedOpinion),
    )
    setLoading(true)
    setError(null)
    setArticles([])
    resetResultsOverview()
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setEmptyResultsMessage(null)
    setTypoCorrection(null)

    try {
      const response = await fetch('/api/articles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode: 'stance',
          topic: nextTopic,
          opinion: trimmedOpinion,
          topic_weight: topicWeight,
          stance_weight: stanceWeight,
          recency_weight: recencyWeight,
          top_k: rerankTopK,
          normalize_topic_scores: normalizeTopicScores,
          stance_method: stanceMethod,
          use_chunking: useChunking,
          chunking_mode: chunkingMode,
          chunk_candidate_top_k: chunkCandidateTopK,
          chunk_article_top_k: chunkArticleTopK,
          retrieval_model: effectiveRetrievalModel,
          rerank_selection_mode: rerankSelectionMode,
          rerank_threshold: currentAutoRerankThreshold,
          year_start: resolvedYearStart,
          year_end: resolvedYearEnd,
          character_start: lengthFilterUnit === 'characters' ? resolvedCharacterStart : null,
          character_end: lengthFilterUnit === 'characters' ? resolvedCharacterEnd : null,
          word_start: lengthFilterUnit === 'words' ? resolvedWordStart : null,
          word_end: lengthFilterUnit === 'words' ? resolvedWordEnd : null,
          reading_time_start: lengthFilterUnit === 'reading_time' ? resolvedReadingTimeStart : null,
          reading_time_end: lengthFilterUnit === 'reading_time' ? resolvedReadingTimeEnd : null,
          words_to_avoid: activeWordsToAvoid,
          topic_feedback_irrelevant_article_ids: feedbackArticleIds,
          skip_typo_correction: Boolean(options.skipTypoCorrection),
        }),
      })

      const data = await readApiJson<Article[] | ArticleSearchResponse>(response)
      const normalized = normalizeArticleSearchResponse(data)
      setArticles(normalized.articles)
      setQuerySvdCorpusChartDimensions(normalized.querySvdCorpusChartDimensions)
      setQuerySvdDimensions(normalized.querySvdDimensions)
      setEmptyResultsMessage(normalized.emptyResultsMessage)
      setTypoCorrection(normalized.typoCorrection)
      setShouldScrollToResults(true)
      if (options.markTopicFeedbackApplied) {
        setAppliedTopicFeedbackArticleIds(feedbackArticleIds)
      }
      void requestResultsOverview(
        [
          nextTopic ? `Topic: ${nextTopic}` : null,
          trimmedOpinion ? `Stance: ${trimmedOpinion}` : null,
        ].filter(Boolean).join('\n'),
        normalized.articles,
        'stance',
      )
    } catch (fetchError) {
      console.error('Search request failed:', fetchError)
      setArticles([])
      resetResultsOverview()
      setQuerySvdCorpusChartDimensions([])
      setQuerySvdDimensions([])
      setEmptyResultsMessage(null)
      setTypoCorrection(null)
      setShouldScrollToResults(true)
      setError(fetchError instanceof Error ? fetchError.message : 'Search request failed.')
    } finally {
      setLoading(false)
      finishSearchFocus()
    }
  }

  const handleAnalyzeEssay = async (): Promise<void> => {
    if (!canAnalyzeEssay || loading) return

    setLoading(true)
    setError(null)
    setArticles([])
    resetResultsOverview()
    setTopicFeedbackIrrelevantArticleIds([])
    setAppliedTopicFeedbackArticleIds([])
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setEmptyResultsMessage(null)
    setTypoCorrection(null)

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
      setTypoCorrection(null)
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

    lastAppliedFiltersRef.current = {
      yearStart: resolvedYearStart,
      yearEnd: resolvedYearEnd,
      lengthFilterUnit,
      characterStart: resolvedCharacterStart,
      characterEnd: resolvedCharacterEnd,
      wordStart: resolvedWordStart,
      wordEnd: resolvedWordEnd,
      readingTimeStart: resolvedReadingTimeStart,
      readingTimeEnd: resolvedReadingTimeEnd,
      wordsToAvoidKey: activeWordsToAvoidKey,
    }
    setHasSubmittedSearch(true)
    setShouldScrollToResults(false)
    if (typeof document !== 'undefined') {
      document.body.style.overflow = ''
    }
    startSearchFocus(
      [
        nextThesisSentence ? `Thesis ${nextThesisSentence}` : 'Essay topic',
        summarizeApiText(nextEssayText, 170),
      ].filter(Boolean).join(' '),
      'essay',
      buildEssaySearchFocusWords(nextEssayText, nextThesisSentence),
    )
    setLoading(true)
    setError(null)
    setArticles([])
    resetResultsOverview()
    setQuerySvdCorpusChartDimensions([])
    setQuerySvdDimensions([])
    setEmptyResultsMessage(null)
    setTypoCorrection(null)

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
          chunk_candidate_top_k: chunkCandidateTopK,
          chunk_article_top_k: chunkArticleTopK,
          retrieval_model: effectiveRetrievalModel,
          rerank_selection_mode: rerankSelectionMode,
          rerank_threshold: currentAutoRerankThreshold,
          year_start: resolvedYearStart,
          year_end: resolvedYearEnd,
          character_start: lengthFilterUnit === 'characters' ? resolvedCharacterStart : null,
          character_end: lengthFilterUnit === 'characters' ? resolvedCharacterEnd : null,
          word_start: lengthFilterUnit === 'words' ? resolvedWordStart : null,
          word_end: lengthFilterUnit === 'words' ? resolvedWordEnd : null,
          reading_time_start: lengthFilterUnit === 'reading_time' ? resolvedReadingTimeStart : null,
          reading_time_end: lengthFilterUnit === 'reading_time' ? resolvedReadingTimeEnd : null,
          words_to_avoid: activeWordsToAvoid,
          topic_feedback_irrelevant_article_ids: feedbackArticleIds,
        }),
      })

      const data = await readApiJson<Article[] | ArticleSearchResponse>(response)
      const normalized = normalizeArticleSearchResponse(data)
      setArticles(normalized.articles)
      setQuerySvdCorpusChartDimensions(normalized.querySvdCorpusChartDimensions)
      setQuerySvdDimensions(normalized.querySvdDimensions)
      setEmptyResultsMessage(normalized.emptyResultsMessage)
      setTypoCorrection(normalized.typoCorrection)
      setShouldScrollToResults(true)
      if (options.markTopicFeedbackApplied) {
        setAppliedTopicFeedbackArticleIds(feedbackArticleIds)
      }
      void requestResultsOverview(
        [
          nextThesisSentence ? `Thesis: ${nextThesisSentence}` : null,
          nextEssayText ? `Essay: ${nextEssayText}` : null,
        ].filter(Boolean).join('\n'),
        normalized.articles,
        'essay',
      )
    } catch (fetchError) {
      console.error('Essay search failed:', fetchError)
      setArticles([])
      resetResultsOverview()
      setQuerySvdCorpusChartDimensions([])
      setQuerySvdDimensions([])
      setEmptyResultsMessage(null)
      setTypoCorrection(null)
      setShouldScrollToResults(true)
      setError(fetchError instanceof Error ? fetchError.message : 'Essay search failed.')
    } finally {
      setLoading(false)
      finishSearchFocus()
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

  const handleApplyTypoCorrection = (correctedQuery: string): void => {
    const nextTopic = correctedQuery.trim()
    if (!nextTopic || loading) return
    skipNextStanceResetRef.current = true
    setTopic(nextTopic)
    void handleSubmitStance({ topicOverride: nextTopic })
  }

  const handleSearchAnyway = (): void => {
    if (loading) return
    void handleSubmitStance({ skipTypoCorrection: true })
  }

  useEffect(() => {
    if (isFilterOpen || loading || !hasSubmittedSearch) {
      return
    }

    const lastAppliedFilters = lastAppliedFiltersRef.current
    if (
      lastAppliedFilters &&
      lastAppliedFilters.yearStart === resolvedYearStart &&
      lastAppliedFilters.yearEnd === resolvedYearEnd &&
      lastAppliedFilters.lengthFilterUnit === lengthFilterUnit &&
      lastAppliedFilters.characterStart === resolvedCharacterStart &&
      lastAppliedFilters.characterEnd === resolvedCharacterEnd &&
      lastAppliedFilters.wordStart === resolvedWordStart &&
      lastAppliedFilters.wordEnd === resolvedWordEnd &&
      lastAppliedFilters.readingTimeStart === resolvedReadingTimeStart &&
      lastAppliedFilters.readingTimeEnd === resolvedReadingTimeEnd &&
      lastAppliedFilters.wordsToAvoidKey === activeWordsToAvoidKey
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
    activeWordsToAvoidKey,
    hasSubmittedSearch,
    inputMode,
    isFilterOpen,
    lengthFilterUnit,
    loading,
    resolvedCharacterEnd,
    resolvedCharacterStart,
    resolvedReadingTimeEnd,
    resolvedReadingTimeStart,
    resolvedWordEnd,
    resolvedWordStart,
    resolvedYearEnd,
    resolvedYearStart,
  ])

  useEffect(() => {
    setYearStartInput(resolvedYearStart === null ? '' : String(resolvedYearStart))
  }, [resolvedYearStart])

  useEffect(() => {
    setYearEndInput(resolvedYearEnd === null ? '' : String(resolvedYearEnd))
  }, [resolvedYearEnd])

  useEffect(() => {
    setCharacterStartInput(resolvedCharacterStart === null ? '' : String(resolvedCharacterStart))
  }, [resolvedCharacterStart])

  useEffect(() => {
    setCharacterEndInput(resolvedCharacterEnd === null ? '' : String(resolvedCharacterEnd))
  }, [resolvedCharacterEnd])

  useEffect(() => {
    setWordStartInput(resolvedWordStart === null ? '' : String(resolvedWordStart))
  }, [resolvedWordStart])

  useEffect(() => {
    setWordEndInput(resolvedWordEnd === null ? '' : String(resolvedWordEnd))
  }, [resolvedWordEnd])

  useEffect(() => {
    setReadingTimeStartInput(resolvedReadingTimeStart === null ? '' : String(resolvedReadingTimeStart))
  }, [resolvedReadingTimeStart])

  useEffect(() => {
    setReadingTimeEndInput(resolvedReadingTimeEnd === null ? '' : String(resolvedReadingTimeEnd))
  }, [resolvedReadingTimeEnd])

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
    if (useChunking && nextModel === 'tfidf') return
    setRetrievalModel(nextModel)
  }

  const handleChunkingModeChange = (nextUseChunking: boolean): void => {
    if (nextUseChunking) {
      if (!canUseChunking) return
      setChunkingMode('semantic')
      setStanceMethod('llm')
      return
    }

    setChunkingMode('none')
  }

  const handleAgreementSearchModeChange = (nextMethod: StanceMethod): void => {
    if (!supportedStanceMethods.includes(nextMethod)) return
    if (nextMethod === 'llm') {
      if (!canUseLlmAgreement) return
      setStanceMethod('llm')
      return
    }

    if (useChunking) return
    if (!canUseNliAgreement) return
    setStanceMethod('nli')
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
    void article
    return ''
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
  const getVisibleArticleResultIndex = (article: Article): number => {
    const index = visibleArticles.findIndex(visibleArticle => (
      getArticleIdKey(visibleArticle) === getArticleIdKey(article)
    ))
    return index >= 0 ? index + 1 : 1
  }
  const topicFeedbackIrrelevantArticles = visibleArticles.filter(isTopicFeedbackIrrelevantArticle)
  const activeVisibleArticles = visibleArticles.filter(article => !isTopicFeedbackIrrelevantArticle(article))
  const selectedResultsChatArticles = resultsChatArticleIds
    .map(articleId => visibleArticles.find(article => getArticleIdKey(article) === articleId))
    .filter((article): article is Article => Boolean(article))
  const resultsChatAttachments = selectedResultsChatArticles.map((article) => ({
    articleId: getArticleIdKey(article),
    resultIndex: getVisibleArticleResultIndex(article),
    title: article.title || 'Untitled article',
  }))
  const getResultsChatContextArticles = (): Article[] => {
    const contextArticles = selectedResultsChatArticles.length > 0
      ? selectedResultsChatArticles
      : visibleArticles.slice(0, 10)

    return contextArticles.map(article => ({
      ...article,
      result_index: getVisibleArticleResultIndex(article),
    }))
  }
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
  const llmIrrelevantArticles = articles.filter(isLlmIrrelevantArticle)
  const canExplainRanking = useLlm === true && llmAgreementAvailable
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

  const closeSimilarArticles = (): void => {
    setSimilarArticleSource(null)
    setSimilarArticles([])
    setSimilarArticlesOffset(0)
    setSimilarArticlesHasMore(false)
    setSimilarArticlesLoading(false)
    setSimilarArticlesError(null)
  }

  const fetchSimilarArticles = async (
    article: Article,
    offset = 0,
  ): Promise<void> => {
    const isLoadingMore = offset > 0
    setSimilarArticleSource(article)
    setSimilarArticlesLoading(true)
    setSimilarArticlesError(null)
    if (!isLoadingMore) {
      setSimilarArticles([])
      setSimilarArticlesOffset(0)
      setSimilarArticlesHasMore(false)
    }

    try {
      const response = await fetch('/api/articles/similar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          article_id: article.id,
          limit: similarArticlesPageSize,
          offset,
          year_start: resolvedYearStart,
          year_end: resolvedYearEnd,
          character_start: lengthFilterUnit === 'characters' ? resolvedCharacterStart : null,
          character_end: lengthFilterUnit === 'characters' ? resolvedCharacterEnd : null,
          word_start: lengthFilterUnit === 'words' ? resolvedWordStart : null,
          word_end: lengthFilterUnit === 'words' ? resolvedWordEnd : null,
          reading_time_start: lengthFilterUnit === 'reading_time' ? resolvedReadingTimeStart : null,
          reading_time_end: lengthFilterUnit === 'reading_time' ? resolvedReadingTimeEnd : null,
        }),
      })
      const data = await readApiJson<SimilarArticlesResponse>(response)
      const nextResults = Array.isArray(data.results) ? data.results : []
      setSimilarArticles(currentArticles => {
        if (!isLoadingMore) return nextResults
        const seen = new Set(currentArticles.map(getArticleIdKey))
        const appended = nextResults.filter(result => !seen.has(getArticleIdKey(result)))
        return [...currentArticles, ...appended]
      })
      setSimilarArticlesOffset(
        typeof data.next_offset === 'number'
          ? data.next_offset
          : offset + nextResults.length,
      )
      setSimilarArticlesHasMore(Boolean(data.has_more))
    } catch (similarError) {
      console.error('Similar article search failed:', similarError)
      setSimilarArticlesError(
        similarError instanceof Error
          ? similarError.message
          : 'Similar article search failed.',
      )
    } finally {
      setSimilarArticlesLoading(false)
    }
  }

  const handleFindSimilarArticles = (article: Article): void => {
    void fetchSimilarArticles(article, 0)
  }

  const handleAskAiAboutArticle = (article: Article): void => {
    const articleId = getArticleIdKey(article)
    setResultsChatArticleIds(currentIds => (
      currentIds.includes(articleId) ? currentIds : [...currentIds, articleId]
    ))
    setResultsChatError(null)
    setIsResultsChatMinimized(false)
  }

  const handleRemoveResultsChatArticle = (articleId: string): void => {
    setResultsChatArticleIds(currentIds => currentIds.filter(currentId => currentId !== articleId))
  }

  const handleClearResultsChatArticles = (): void => {
    setResultsChatArticleIds([])
  }

  const handleLoadMoreSimilarArticles = (): void => {
    if (!similarArticleSource || similarArticlesLoading) return
    void fetchSimilarArticles(similarArticleSource, similarArticlesOffset)
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

  const requestSvdDimensionLabels = async (article: Article): Promise<SvdDimensionLabelMap> => {
    const currentState = getSvdDimensionLabelState(article)
    if (currentState.loading || Object.keys(currentState.labels).length > 0) {
      return currentState.labels
    }

    const dimensions = getSvdDimensionsForLabeling(article)
    if (dimensions.length === 0) return currentState.labels

    if (useLlm !== true) {
      setSvdDimensionLabelState(article, {
        error: 'LLM labels are turned off in the backend config.',
      })
      return currentState.labels
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
      return nextLabels
    } catch (fetchError) {
      setSvdDimensionLabelState(article, {
        loading: false,
        error: fetchError instanceof Error ? fetchError.message : 'SVD concept labeling failed.',
      })
      return currentState.labels
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
      let svdDimensionLabels: SvdDimensionLabelMap = {}
      if (effectiveRetrievalModel === 'svd' && hasSvdExplainability(article)) {
        const currentLabelState = getSvdDimensionLabelState(article)
        svdDimensionLabels = Object.keys(currentLabelState.labels).length > 0
          ? currentLabelState.labels
          : await requestSvdDimensionLabels(article)
      }

      const response = await fetch('/api/llm/explain-ranking', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: queryText,
          article,
          position: rank,
          retrieval_model: effectiveRetrievalModel,
          chunking_mode: chunkingMode,
          query_svd_dimensions: querySvdDimensions,
          svd_dimension_labels: svdDimensionLabels,
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

  const resultsDescription = useMemo(() => {
    if (loading) {
      return `Ranking Guardian opinion pieces${activeFilterSummary} with your current search settings.`
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
        `No matching articles came back${activeFilterSummary} this time. `
        + 'Try broadening the topic, sharpening the claim, or widening the filters.'
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
      + `${activeFilterSummary} ranked with your current search settings.${hiddenCopy}${feedbackCopy}${appliedFeedbackCopy}`
    )
  }, [
    activeVisibleArticles.length,
    activeFilterSummary,
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
  ])

  return (
    <div className="experience-shell">
      <div
        className={[
          'intro-screen',
          'landing-section',
          isSearchChromeVisible ? 'search-active' : '',
          inputMode === 'essay' ? 'essay-mode' : 'stance-mode',
        ].filter(Boolean).join(' ')}
      >
        <div className={`intro-shell ${isSearchChromeVisible ? 'search-active' : ''} ${isAboutPageActive ? 'about-active' : ''}`}>
          <div className={`top-nav page-top-nav ${isAboutPageActive ? 'about-active' : ''}`} aria-label="Page navigation">
            <div className="top-nav-spacer" aria-hidden="true" />
            <div className="top-nav-actions">
              <button
                type="button"
                className={`top-nav-button ${activeTopNavPage === 'home' ? 'active' : ''}`}
                onClick={showHomePage}
                aria-pressed={activeTopNavPage === 'home'}
              >
                Home
              </button>
              <button
                type="button"
                className={`top-nav-button ${isSearchPageActive ? 'active' : ''}`}
                onClick={showSearchPage}
                aria-pressed={isSearchPageActive}
              >
                Search
              </button>
              <button
                type="button"
                className={`top-nav-button ${isAboutPageActive ? 'active' : ''}`}
                onClick={openAboutPage}
                aria-pressed={isAboutPageActive}
              >
                About
              </button>
            </div>
          </div>

          <div className={`search-chrome ${isSearchChromeVisible ? 'visible' : ''} ${isAboutPageActive ? 'about-active' : ''}`}>
            {isAboutPageActive ? (
              <section className="about-page" aria-labelledby="about-page-title">
                <div className="about-page-card">
                  <div className="about-page-header">
                    <p className="about-page-eyebrow">About</p>
                    <h2 id="about-page-title">About hear! hear!</h2>
                    <p className="about-page-subtitle">
                      Learn what the project is for, who built it, and how the search pipeline works.
                    </p>
                  </div>

                  <div className="about-tablist" role="tablist" aria-label="About sections">
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeAboutSection === 'overview'}
                      className={`about-tab ${activeAboutSection === 'overview' ? 'active' : ''}`}
                      onClick={() => setActiveAboutSection('overview')}
                    >
                      Overview
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeAboutSection === 'team'}
                      className={`about-tab ${activeAboutSection === 'team' ? 'active' : ''}`}
                      onClick={() => setActiveAboutSection('team')}
                    >
                      Team
                    </button>
                    <button
                      type="button"
                      role="tab"
                      aria-selected={activeAboutSection === 'method'}
                      className={`about-tab ${activeAboutSection === 'method' ? 'active' : ''}`}
                      onClick={() => setActiveAboutSection('method')}
                    >
                      Method
                    </button>
                  </div>

                  <div className="about-page-panel">
                    {activeAboutSection === 'overview' && (
                      <div className="about-page-stack">
                        {aboutOverviewParagraphs.map((paragraph) => (
                          <p key={paragraph} className="about-page-copy">
                            {paragraph}
                          </p>
                        ))}
                      </div>
                    )}

                    {activeAboutSection === 'team' && (
                      <ul className="about-team-grid" aria-label="Team members">
                        {aboutTeamMembers.map((member) => (
                          <li key={member} className="about-team-card">
                            <p className="about-team-label">Team Member</p>
                            <h3>{member}</h3>
                          </li>
                        ))}
                      </ul>
                    )}

                    {activeAboutSection === 'method' && (
                      <div className="about-page-stack">
                        <AboutMethodFlow mode={activeAboutMethodTab} onModeChange={setActiveAboutMethodTab} />
                      </div>
                    )}
                  </div>
                </div>
              </section>
            ) : (
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

                <div className="top-search-controls" aria-label="Search controls">
                  <div className="top-search-mode-group top-search-mode-group-compact">
                    <div className="top-search-mode-heading">
                      <span>Search Granularity</span>
                      <button
                        type="button"
                        className="top-search-mode-help"
                        onClick={() => openSettingsAt('retrieval-granularity')}
                        aria-label="Open search granularity settings"
                      >
                        ?
                      </button>
                    </div>
                    <div className="top-search-mode-segments" role="group" aria-label="Search granularity">
                      <button
                        type="button"
                        className={`top-search-mode-segment ${!useChunking ? 'active' : ''}`}
                        aria-pressed={!useChunking}
                        onClick={() => handleChunkingModeChange(false)}
                      >
                        Article
                      </button>
                      <button
                        type="button"
                        className={`top-search-mode-segment ${useChunking ? 'active' : ''}`}
                        aria-pressed={useChunking}
                        onClick={() => handleChunkingModeChange(true)}
                        disabled={!canUseChunking}
                      >
                        Chunks
                      </button>
                    </div>
                  </div>

                  <div className="top-search-mode-toolbar" aria-label="Search methods">
                    <div className="top-search-mode-group">
                      <div className="top-search-mode-heading">
                        <span>Topic Relevance Search Method</span>
                        <button
                          type="button"
                          className="top-search-mode-help"
                          onClick={() => openSettingsAt('topic-relevance')}
                          aria-label="Open topic relevance settings"
                        >
                          ?
                        </button>
                      </div>
                      <div className="top-search-mode-segments" role="group" aria-label="Topic relevance search method">
                        <button
                          type="button"
                          className={`top-search-mode-segment ${!useChunking && retrievalModel === 'tfidf' ? 'active' : ''}`}
                          aria-pressed={!useChunking && retrievalModel === 'tfidf'}
                          onClick={() => handleTopicSearchModeChange('tfidf')}
                          disabled={!canUseLexicalRetrieval}
                        >
                          Lexical
                        </button>
                        <button
                          type="button"
                          className={`top-search-mode-segment ${effectiveRetrievalModel === 'svd' ? 'active' : ''}`}
                          aria-pressed={effectiveRetrievalModel === 'svd'}
                          onClick={() => handleTopicSearchModeChange('svd')}
                          disabled={!canUseSvd}
                        >
                          Semantic
                        </button>
                        <button
                          type="button"
                          className={`top-search-mode-segment ${effectiveRetrievalModel === 'minilm' ? 'active' : ''}`}
                          aria-pressed={effectiveRetrievalModel === 'minilm'}
                          onClick={() => handleTopicSearchModeChange('minilm')}
                          disabled={!canUseMiniLm}
                        >
                          Enhanced
                        </button>
                      </div>
                    </div>

                    <div className="top-search-mode-group">
                      <div className="top-search-mode-heading">
                        <span>Stance Agreement Search Method</span>
                        <button
                          type="button"
                          className="top-search-mode-help"
                          onClick={() => openSettingsAt('agreement-scorer')}
                          aria-label="Open stance agreement settings"
                        >
                          ?
                        </button>
                      </div>
                      <div className="top-search-mode-segments" role="group" aria-label="Stance agreement search method">
                        <button
                          type="button"
                          className={`top-search-mode-segment ${effectiveStanceMethod === 'nli' ? 'active' : ''}`}
                          aria-pressed={effectiveStanceMethod === 'nli'}
                          onClick={() => handleAgreementSearchModeChange('nli')}
                          disabled={!canUseNliAgreement || useChunking}
                        >
                          Fast
                        </button>
                        <button
                          type="button"
                          className={`top-search-mode-segment ${effectiveStanceMethod === 'llm' ? 'active' : ''}`}
                          aria-pressed={effectiveStanceMethod === 'llm'}
                          onClick={() => handleAgreementSearchModeChange('llm')}
                          disabled={!canUseLlmAgreement}
                        >
                          Enhanced
                        </button>
                      </div>
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
            )}
          </div>

          {!isAboutPageActive && (
            <>
              <div className={`landing-prompt-shell ${(inputMode === 'essay' && isSearchStageVisible) ? 'hidden' : ''}`}>
            <div
              className={`intro-line visible ${introStage > 0 ? 'done' : ''}`}
              role="text"
              aria-label={`Regarding ${typedTopic || trimmedTopic || introTopicSequence[introTopicSequence.length - 1]}`}
            >
              <span ref={topicPromptLabelRef} className="intro-line-label">Regarding</span>
              {isSearchStageVisible && inputMode === 'stance' ? (
                <span className="intro-inline-form-slot">
                  <span className="intro-inline-input-wrap">
                    <input
                      ref={topicInputRef}
                      type="text"
                      value={topic}
                      onChange={(event) => setTopic(event.target.value)}
                      placeholder="type your topic"
                      aria-label="Topic"
                    />
                  </span>
                  {!loading && typoCorrection && (
                    <span className="inline-typo-popover" role="status" aria-live="polite">
                      <span className="typo-suggestion-copy">
                        <span className="typo-suggestion-query">
                          {renderTypoHighlightedQuery(typoCorrection)}
                        </span>
                        <span className="typo-suggestion-label">did you mean:</span>
                      </span>
                      <span className="typo-suggestion-options">
                        {typoCorrection.options.map((option) => (
                          <button
                            key={option.query}
                            type="button"
                            className="typo-suggestion-option"
                            onClick={() => handleApplyTypoCorrection(option.query)}
                          >
                            {option.label || option.query}
                          </button>
                        ))}
                        <button
                          type="button"
                          className="typo-suggestion-option search-anyway"
                          onClick={handleSearchAnyway}
                        >
                          Search anyway
                        </button>
                      </span>
                    </span>
                  )}
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
              <span ref={opinionPromptLabelRef} className="intro-line-label">I believe</span>
              {isSearchStageVisible && inputMode === 'stance' ? (
                <span className="intro-inline-form-slot">
                  <span className="intro-inline-input-wrap">
                    <input
                      ref={opinionInputRef}
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

            {isSearchStageVisible && inputMode === 'stance' && (
              <div className="query-assist-shell">
                <button
                  type="button"
                  className="query-assist-link"
                  onClick={handleToggleQueryAssist}
                  disabled={!canUseQueryAssist}
                  aria-expanded={isQueryAssistOpen}
                  title={queryAssistDisabledReason || undefined}
                >
                  Help me improve my query with AI.
                </button>
              </div>
            )}
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
                          ref={essayTextAreaRef}
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
                      <p ref={essaySubmitCopyRef} className="essay-submit-copy">
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
              Filter
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
            </>
          )}
        </div>
      </div>

      {isSearchPageActive && (
        <>
          {searchFocusSnapshot && (
            <FloatingSearchFocus
              key={searchFocusSnapshot.key}
              mode={searchFocusSnapshot.mode}
              words={searchFocusSnapshot.words}
              clearing={searchFocusSnapshot.clearing}
            />
          )}

          {hasSubmittedSearch && (
            <div
              ref={resultsSectionRef}
              className="results-paper-section visible"
            >
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

            {loading && !searchFocusSnapshot && (
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
              <>
                {(resultsOverviewLoading || resultsOverview || resultsOverviewError) && (
                  <section className="results-overview-card" aria-live="polite">
                    <div className="results-overview-header">
                      <p className="results-overview-eyebrow">AI overview</p>
                      {resultsOverviewLoading && (
                        <div className="results-overview-spinner" aria-hidden="true">
                          <span />
                          <span />
                          <span />
                        </div>
                      )}
                    </div>

                    {resultsOverviewLoading && (
                      <p className="results-overview-copy">Reading the result set for agreement patterns, shared claims, and differences.</p>
                    )}

                    {!resultsOverviewLoading && resultsOverview && (
                      <>
                        <p className="results-overview-copy">
                          <ResultsOverviewCitedText
                            text={resultsOverview.overview}
                            sources={resultsOverview.sources}
                          />
                        </p>
                        {Array.isArray(resultsOverview.key_points) && resultsOverview.key_points.length > 0 && (
                          <ul className="results-overview-list">
                            {resultsOverview.key_points.map((point) => (
                              <li key={point}>
                                <ResultsOverviewCitedText
                                  text={point}
                                  sources={resultsOverview.sources}
                                />
                              </li>
                            ))}
                          </ul>
                        )}
                        <div className="results-overview-argument-grid">
                          <ResultsOverviewArgumentList
                            title="Supports you"
                            items={resultsOverview.supporting_arguments}
                            sources={resultsOverview.sources}
                          />
                          <ResultsOverviewArgumentList
                            title="Challenges you"
                            items={resultsOverview.opposing_arguments}
                            sources={resultsOverview.sources}
                          />
                        </div>
                        {Array.isArray(resultsOverview.key_evidence) && resultsOverview.key_evidence.length > 0 && (
                          <div className="results-overview-evidence-group">
                            <h4>Key evidence</h4>
                            <ul className="results-overview-evidence-list">
                              {resultsOverview.key_evidence.map((evidence, evidenceIndex) => (
                                <li key={`overview-evidence-${evidenceIndex}-${evidence.evidence}`}>
                                  <span>
                                    <ResultsOverviewCitedText
                                      text={evidence.evidence}
                                      sourceIndices={evidence.source_indices}
                                      sources={resultsOverview.sources}
                                    />
                                  </span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {resultsOverview.caveat && (
                          <p className="results-overview-caveat">
                            <ResultsOverviewCitedText
                              text={resultsOverview.caveat}
                              sources={resultsOverview.sources}
                            />
                          </p>
                        )}
                      </>
                    )}

                    {!resultsOverviewLoading && !resultsOverview && resultsOverviewError && (
                      <p className="results-overview-error">{resultsOverviewError}</p>
                    )}
                  </section>
                )}

                <div id="answer-box">
                  {visibleArticles.map((article, visibleArticleIndex) => {
                  const articleTooltipBase = String(article.id).replace(/[^a-zA-Z0-9_-]/g, '-')
                  const articleRecencyWeight = article.recency_weight ?? recencyWeight
                  const svdDimensionLabelState = getSvdDimensionLabelState(article)
                  const isMarkedNotRelevant = isTopicFeedbackIrrelevantArticle(article)
                  const visibleResultIndex = visibleArticleIndex + 1
                  const activeArticleRank = activeVisibleArticles.findIndex(activeArticle => (
                    getArticleIdKey(activeArticle) === getArticleIdKey(article)
                  )) + 1
                  const isArticleAttachedToResultsChat = resultsChatArticleIds.includes(getArticleIdKey(article))

                  if (isMarkedNotRelevant) {
                    return (
                      <article
                        key={article.id}
                        id={`result-${visibleResultIndex}`}
                        className="article-item topic-feedback-collapsed-article"
                      >
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
                    <article
                      key={article.id}
                      id={`result-${visibleResultIndex}`}
                      className="article-item"
                    >
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
                                  {getRankingExplanationState(article).loading ? 'Explaining…' : 'Explain ranking with AI'}
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
                                    This chart shows the main ideas in your query, and how strongly this article relates to each of them.
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
                                    This chart compares the article to a common set of broad topics used across all results, so you can see how its focus differs from others.
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
                                      Longer bars mean the article is more strongly associated with that concept.
                                    </p>
                                    <p className='svd-section-copy'>
                                      The direction (left vs. right) shows which side of the concept the article falls on. SVD dimensions capture contrasts between related themes, so opposite directions correspond to different but related topics.
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

                      {article.vader_sentiment && (
                        <details className="content-disclosure sentiment-disclosure">
                          <summary className="content-disclosure-summary">
                            <span className="content-disclosure-copy">
                              <span className="content-disclosure-title">Sentiment</span>
                            </span>
                            <span className="content-disclosure-status" aria-hidden="true" />
                          </summary>
                          <div className="sentiment-panel">
                            <div className="sentiment-score-row">
                              <div className="sentiment-score-copy">
                                <span className={`sentiment-label ${article.vader_sentiment.label}`}>
                                  {formatSentimentLabel(article.vader_sentiment.label)}
                                </span>
                                {article.vader_sentiment.tone_strength && (
                                  <span className={`sentiment-strength ${article.vader_sentiment.tone_strength}`}>
                                    {formatToneStrength(article.vader_sentiment.tone_strength)} tone
                                  </span>
                                )}
                              </div>
                              <span className="sentiment-compound">
                                {formatVaderScore(article.vader_sentiment.compound)}
                              </span>
                            </div>
                            <div
                              className="sentiment-meter"
                              aria-label={`VADER sentiment: ${article.vader_sentiment.label}, compound ${formatVaderScore(article.vader_sentiment.compound)}`}
                            >
                              <span
                                className="sentiment-meter-segment negative"
                                style={{ width: formatSentimentPercent(article.vader_sentiment.negative) }}
                              />
                              <span
                                className="sentiment-meter-segment neutral"
                                style={{ width: formatSentimentPercent(article.vader_sentiment.neutral) }}
                              />
                              <span
                                className="sentiment-meter-segment positive"
                                style={{ width: formatSentimentPercent(article.vader_sentiment.positive) }}
                              />
                            </div>
                            <div className="sentiment-breakdown">
                              <span
                                className="sentiment-breakdown-segment negative"
                                style={{ width: formatSentimentPercent(article.vader_sentiment.negative) }}
                                aria-label={`Negative ${formatSentimentPercent(article.vader_sentiment.negative)}`}
                              >
                                <span className="sentiment-breakdown-text" aria-hidden="true">
                                  Negative {formatSentimentPercent(article.vader_sentiment.negative)}
                                </span>
                              </span>
                              <span
                                className="sentiment-breakdown-segment neutral"
                                style={{ width: formatSentimentPercent(article.vader_sentiment.neutral) }}
                                aria-label={`Neutral ${formatSentimentPercent(article.vader_sentiment.neutral)}`}
                              >
                                <span className="sentiment-breakdown-text" aria-hidden="true">
                                  Neutral {formatSentimentPercent(article.vader_sentiment.neutral)}
                                </span>
                              </span>
                              <span
                                className="sentiment-breakdown-segment positive"
                                style={{ width: formatSentimentPercent(article.vader_sentiment.positive) }}
                                aria-label={`Positive ${formatSentimentPercent(article.vader_sentiment.positive)}`}
                              >
                                <span className="sentiment-breakdown-text" aria-hidden="true">
                                  Positive {formatSentimentPercent(article.vader_sentiment.positive)}
                                </span>
                              </span>
                            </div>

                            {article.vader_sentiment.text_scores && (
                              <div className="sentiment-comparison">
                                <div className="sentiment-section-title">Title vs full article</div>
                                <div className="sentiment-source-grid">
                                  {[
                                    { label: 'Title', score: article.vader_sentiment.text_scores.title },
                                    { label: 'Summary', score: article.vader_sentiment.text_scores.summary },
                                    { label: 'Full article', score: article.vader_sentiment.text_scores.article },
                                  ].map((item) => {
                                    if (!item.score) return null
                                    return (
                                      <div key={item.label} className="sentiment-source-card">
                                        <span>{item.label}</span>
                                        <strong className={item.score.label}>
                                          {formatVaderScore(item.score.compound)}
                                        </strong>
                                        <small>{formatSentimentLabel(item.score.label)}</small>
                                      </div>
                                    )
                                  })}
                                </div>
                              </div>
                            )}

                            {(
                              (article.vader_sentiment.snippets?.negative?.length ?? 0) > 0 ||
                              (article.vader_sentiment.snippets?.positive?.length ?? 0) > 0
                            ) && (
                                <div className="sentiment-evidence">
                                  {[
                                    {
                                      key: 'negative',
                                      title: 'Most negative sentences',
                                      snippets: article.vader_sentiment.snippets?.negative ?? [],
                                    },
                                    {
                                      key: 'positive',
                                      title: 'Most positive sentences',
                                      snippets: article.vader_sentiment.snippets?.positive ?? [],
                                    },
                                  ].map((group) => (
                                    group.snippets.length > 0 && (
                                      <div key={group.key} className={`sentiment-evidence-group ${group.key}`}>
                                        <div className="sentiment-section-title">{group.title}</div>
                                        <ol className="sentiment-snippet-list">
                                          {group.snippets.map((snippet, index) => (
                                            <li key={`${article.id}-sentiment-${group.key}-${index}`}>
                                              <span>{snippet.text}</span>
                                              <strong>{formatVaderScore(snippet.compound)}</strong>
                                            </li>
                                          ))}
                                        </ol>
                                      </div>
                                    )
                                  ))}
                                </div>
                              )}

                            <div className="sentiment-ranking-note">
                              Display only. Not used for ranking.
                            </div>
                          </div>
                        </details>
                      )}

                      {(article.thesis_sentence || (article.support_sentences && article.support_sentences.length > 0) || (article.keywords && article.keywords.length > 0)) && (
                        <details className="content-disclosure">
                          <summary className="content-disclosure-summary">
                            <span className="content-disclosure-copy">
                              <span className="content-disclosure-title">Overview</span>
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

                            {article.keywords && article.keywords.length > 0 && (
                              <div className="overview-group">
                                <div className="overview-label">Keywords</div>
                                <div className="keyword-list">
                                  {article.keywords.map((kw, index) => (
                                    <span key={`${article.id}-keyword-${index}`} className="keyword-chip">{kw}</span>
                                  ))}
                                </div>
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

                      <div className="topic-feedback-action-row">
                        <button
                          type="button"
                          className="topic-feedback-button"
                          onClick={() => handleMarkTopicIrrelevant(article)}
                          aria-label={`Mark ${article.title} as not relevant`}
                        >
                          Mark as not relevant
                        </button>
                        <button
                          type="button"
                          className="topic-feedback-button similar"
                          onClick={() => handleFindSimilarArticles(article)}
                          disabled={similarArticlesLoading && getArticleIdKey(similarArticleSource ?? article) === getArticleIdKey(article)}
                          aria-label={`Find articles similar to ${article.title}`}
                        >
                          {similarArticlesLoading && getArticleIdKey(similarArticleSource ?? article) === getArticleIdKey(article)
                            ? 'Finding similar...'
                            : 'Find similar articles'}
                        </button>
                        <button
                          type="button"
                          className="topic-feedback-button ask-ai"
                          onClick={() => handleAskAiAboutArticle(article)}
                          aria-pressed={isArticleAttachedToResultsChat}
                          aria-label={`Ask AI about ${article.title}`}
                        >
                          Ask AI
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
              </>
            )}

            {!loading && !error && articles.length === 0 && (
              <div className="results-empty-card searched">
                <p>
                  {emptyResultsMessage || `No matching articles were returned${activeFilterSummary}. Try broadening the topic, making the stance more explicit, or widening the filters.`}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {shouldShowEssayShortcut && <Chat onSearchTerm={handleEssaySearch} />}

      {isQueryAssistOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setIsQueryAssistOpen(false)}
          role="presentation"
        >
          <div
            className="modal-card query-assist-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="query-assist-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <p className="modal-eyebrow">AI query help</p>
                <h3 id="query-assist-modal-title">Improve your query</h3>
              </div>
              <button
                type="button"
                className="modal-close"
                onClick={() => setIsQueryAssistOpen(false)}
                aria-label="Close AI query help popup"
              >
                Close
              </button>
            </div>

            <div className="query-assist-modal-body">
              {queryAssistMode === 'menu' && (
                <div className="query-assist-choice-grid">
                  <button
                    type="button"
                    className="query-assist-choice"
                    onClick={() => void requestQueryAssist('rewrite')}
                    disabled={queryAssistLoading}
                  >
                    Let the AI rewrite the query for me
                  </button>
                  <button
                    type="button"
                    className="query-assist-choice"
                    onClick={() => void requestQueryAssist('suggestions')}
                    disabled={queryAssistLoading}
                  >
                    Tell me how to improve my query
                  </button>
                </div>
              )}

              {queryAssistLoading && (
                <div className="query-assist-thinking-card" role="status" aria-live="polite">
                  <p className="results-thinking-label">Thinking</p>
                  <div className="results-thinking-dots" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </div>
                  <p className="query-assist-thinking-copy">Improving your query...</p>
                </div>
              )}

              {!queryAssistLoading && queryAssistError && (
                <div className="query-assist-error" role="alert">
                  {queryAssistError}
                </div>
              )}

              {!queryAssistLoading && queryAssistMode === 'rewrite' && queryRewriteOptions.length > 0 && (
                <div className="query-rewrite-list">
                  {queryRewriteOptions.map((alternative, index) => (
                    <button
                      key={`${alternative.topic}-${alternative.opinion}`}
                      type="button"
                      className="query-rewrite-option"
                      onClick={() => handleApplyQueryRewrite(alternative)}
                    >
                      <span className="query-rewrite-option-label">{`Option ${index + 1}`}</span>
                      <span className="query-rewrite-query">{alternative.query}</span>
                      {alternative.rationale && (
                        <span className="query-rewrite-rationale">{alternative.rationale}</span>
                      )}
                    </button>
                  ))}
                </div>
              )}

              {!queryAssistLoading && queryAssistMode === 'suggestions' && queryImproveSuggestions.length > 0 && (
                <ul className="query-suggestion-list">
                  {queryImproveSuggestions.map((suggestion) => (
                    <li key={suggestion}>{suggestion}</li>
                  ))}
                </ul>
              )}
            </div>

            {queryAssistMode !== 'menu' && !queryAssistLoading && (
              <button
                type="button"
                className="query-assist-back"
                onClick={() => {
                  setQueryAssistMode('menu')
                  setQueryAssistError(null)
                  setQueryRewriteOptions([])
                  setQueryImproveSuggestions([])
                }}
              >
                Back
              </button>
            )}
          </div>
        </div>
      )}

      {isSearchPageActive && hasSubmittedSearch && !loading && !error && articles.length > 0 && (
        <aside
          className={`results-chat-popout ${isResultsChatMinimized ? 'minimized' : 'open'}`}
          aria-label="Ask questions about these results"
        >
          {isResultsChatMinimized ? (
            <button
              type="button"
              className="results-chat-launcher"
              onClick={() => setIsResultsChatMinimized(false)}
              aria-label="Open results chat"
            >
              <span className="results-chat-launcher-mark" aria-hidden="true">?</span>
              <span className="results-chat-launcher-copy">
                <span>Ask AI</span>
                {resultsChatMessages.length > 0 && (
                  <span>{`${resultsChatMessages.length} messages`}</span>
                )}
              </span>
            </button>
          ) : (
            <section className="results-chat-card">
              <div className="results-chat-header">
                <div>
                  <p className="results-overview-eyebrow">Ask AI about the results</p>
                  <h3>{resultsChatAttachments.length > 0 ? 'Question selected articles' : 'Question the retrieved articles'}</h3>
                </div>
                <button
                  type="button"
                  className="results-chat-minimize-button"
                  onClick={() => setIsResultsChatMinimized(true)}
                  aria-label="Minimize results chat"
                >
                  _
                </button>
              </div>

              <div className="results-chat-body">
                {resultsChatMessages.length > 0 ? (
                  <div className="results-chat-thread" aria-live="polite">
                    {resultsChatMessages.map((message) => (
                      <div
                        key={message.id}
                        className={`results-chat-message ${message.role}`}
                      >
                        <p>{message.content}</p>
                        {message.attachments && message.attachments.length > 0 && (
                          <div className="results-chat-message-attachments" aria-label="Attached articles">
                            {message.attachments.map((attachment) => (
                              <a
                                key={`${message.id}-${attachment.articleId}`}
                                className="results-chat-message-attachment"
                                href={`#result-${attachment.resultIndex}`}
                                title={attachment.title}
                                onClick={(event) => handleResultsOverviewSourceClick(event, attachment.resultIndex)}
                              >
                                {`Result ${attachment.resultIndex}: ${attachment.title}`}
                              </a>
                            ))}
                          </div>
                        )}
                        {message.role === 'assistant' && (
                          <ResultsOverviewSources
                            sourceIndices={message.source_indices}
                            sources={message.sources}
                          />
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="results-chat-empty">
                    Ask what the results agree on, where they split, or which pieces best support your claim.
                  </p>
                )}
              </div>

              {resultsChatAttachments.length > 0 && (
                <div className="results-chat-attachments" aria-label="Article attachments">
                  <div className="results-chat-attachments-header">
                    <span>
                      {`${resultsChatAttachments.length} ${resultsChatAttachments.length === 1 ? 'article' : 'articles'} attached`}
                    </span>
                    <button
                      type="button"
                      className="results-chat-attachment-clear-button"
                      onClick={handleClearResultsChatArticles}
                    >
                      Clear
                    </button>
                  </div>
                  <div className="results-chat-attachment-list">
                    {resultsChatAttachments.map((attachment) => (
                      <div key={attachment.articleId} className="results-chat-attachment-chip">
                        <a
                          href={`#result-${attachment.resultIndex}`}
                          title={attachment.title}
                          onClick={(event) => handleResultsOverviewSourceClick(event, attachment.resultIndex)}
                        >
                          {`Result ${attachment.resultIndex}: ${attachment.title}`}
                        </a>
                        <button
                          type="button"
                          className="results-chat-attachment-remove-button"
                          onClick={() => handleRemoveResultsChatArticle(attachment.articleId)}
                          aria-label={`Remove ${attachment.title} from AI chat`}
                        >
                          x
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {resultsChatError && (
                <p className="results-overview-error">{resultsChatError}</p>
              )}

              <form className="results-chat-form" onSubmit={handleSubmitResultsChat}>
                <input
                  type="text"
                  value={resultsChatInput}
                  onChange={(event) => setResultsChatInput(event.target.value)}
                  placeholder={resultsChatAttachments.length > 0 ? 'Ask about attached articles...' : 'Ask about these results...'}
                  aria-label="Ask a question about the current results"
                  disabled={resultsChatLoading}
                />
                <button
                  type="submit"
                  disabled={resultsChatLoading || resultsChatInput.trim() === ''}
                >
                  {resultsChatLoading ? 'Asking...' : 'Ask'}
                </button>
              </form>
            </section>
          )}
        </aside>
      )}
            </>
          )}

      {similarArticleSource && (
        <div
          className="modal-backdrop"
          onClick={closeSimilarArticles}
          role="presentation"
        >
          <div
            className="modal-card similar-articles-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="similar-articles-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <p className="modal-eyebrow">SVD similarity</p>
                <h3 id="similar-articles-modal-title">Similar articles</h3>
              </div>
              <button
                type="button"
                className="modal-close"
                onClick={closeSimilarArticles}
                aria-label="Close similar articles popup"
              >
                Close
              </button>
            </div>

            <div className="similar-articles-source">
              <span>Source</span>
              <strong>{similarArticleSource.title}</strong>
            </div>

            {similarArticlesError && (
              <div className="similar-articles-error" role="alert">
                {similarArticlesError}
              </div>
            )}

            {similarArticlesLoading && similarArticles.length === 0 && (
              <div className="similar-articles-empty" role="status" aria-live="polite">
                Finding similar articles...
              </div>
            )}

            {!similarArticlesLoading && !similarArticlesError && similarArticles.length === 0 && (
              <div className="similar-articles-empty">
                No similar SVD articles were found for this result.
              </div>
            )}

            {similarArticles.length > 0 && (
              <div className="similar-articles-list">
                {similarArticles.map((similarArticle, index) => (
                  <article
                    key={`${similarArticleSource.id}-similar-${similarArticle.id}`}
                    className="similar-article-item"
                  >
                    <div className="similar-article-rank">{index + 1}</div>
                    <div className="similar-article-copy">
                      <div className="similar-article-header">
                        <h4>
                          <a href={similarArticle.url} target="_blank" rel="noreferrer">
                            {similarArticle.title}
                          </a>
                        </h4>
                        <span>{formatPercent(similarArticle.score)}</span>
                      </div>
                      <p className="similar-article-meta">
                        {similarArticle.author_display || similarArticle.author_raw || 'Unknown author'} | {formatDate(similarArticle.date)}
                      </p>
                      <p className="similar-article-summary">{similarArticle.summary}</p>
                    </div>
                  </article>
                ))}
              </div>
            )}

            {(similarArticlesHasMore || similarArticlesLoading) && similarArticles.length > 0 && (
              <button
                type="button"
                className="similar-articles-load-more"
                onClick={handleLoadMoreSimilarArticles}
                disabled={similarArticlesLoading}
              >
                {similarArticlesLoading ? 'Loading...' : 'Load more'}
              </button>
            )}
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
                <h3 id="filter-settings-title">Article filters</h3>
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
                  <TwoHandleRangeSlider
                    minValue={minArticleYear}
                    maxValue={maxArticleYear}
                    startValue={resolvedYearStart}
                    endValue={resolvedYearEnd}
                    disabled={!hasAvailableYearBounds || yearRangeSpan === 0}
                    startAriaLabel="Start year"
                    endAriaLabel="End year"
                    onStartValueChange={(nextYear) => handleYearStartChange(String(nextYear))}
                    onEndValueChange={(nextYear) => handleYearEndChange(String(nextYear))}
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
              <div className="weight-card full-row">
                <span>Article length</span>
                <div className="length-filter-tabs" role="tablist" aria-label="Article length unit">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={lengthFilterUnit === 'characters'}
                    className={`length-filter-tab ${lengthFilterUnit === 'characters' ? 'active' : ''}`}
                    onClick={() => setLengthFilterUnit('characters')}
                  >
                    Characters
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={lengthFilterUnit === 'words'}
                    className={`length-filter-tab ${lengthFilterUnit === 'words' ? 'active' : ''}`}
                    onClick={() => setLengthFilterUnit('words')}
                  >
                    Words
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={lengthFilterUnit === 'reading_time'}
                    className={`length-filter-tab ${lengthFilterUnit === 'reading_time' ? 'active' : ''}`}
                    onClick={() => setLengthFilterUnit('reading_time')}
                  >
                    Reading time
                  </button>
                </div>
                <div className="year-range-summary-grid" aria-live="polite">
                  <label className="year-range-value-card year-range-input-card">
                    <span>
                      {lengthFilterUnit === 'reading_time'
                        ? 'Min min'
                        : (lengthFilterUnit === 'words' ? 'Min words' : 'Min chars')}
                    </span>
                    <input
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      value={selectedLengthRangeStartInput}
                      onChange={(event) => {
                        if (lengthFilterUnit === 'reading_time') {
                          handleReadingTimeStartInputChange(event.target.value)
                          return
                        }
                        if (lengthFilterUnit === 'words') {
                          handleWordStartInputChange(event.target.value)
                          return
                        }
                        handleCharacterStartInputChange(event.target.value)
                      }}
                      onBlur={
                        lengthFilterUnit === 'reading_time'
                          ? commitReadingTimeStartInput
                          : (lengthFilterUnit === 'words' ? commitWordStartInput : commitCharacterStartInput)
                      }
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.currentTarget.blur()
                        }
                      }}
                      disabled={!selectedLengthFilterHasBounds}
                      aria-label={`Minimum article length in ${lengthFilterUnit === 'reading_time' ? 'reading minutes' : lengthFilterUnit}`}
                    />
                  </label>
                  <label className="year-range-value-card year-range-input-card">
                    <span>
                      {lengthFilterUnit === 'reading_time'
                        ? 'Max min'
                        : (lengthFilterUnit === 'words' ? 'Max words' : 'Max chars')}
                    </span>
                    <input
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      value={selectedLengthRangeEndInput}
                      onChange={(event) => {
                        if (lengthFilterUnit === 'reading_time') {
                          handleReadingTimeEndInputChange(event.target.value)
                          return
                        }
                        if (lengthFilterUnit === 'words') {
                          handleWordEndInputChange(event.target.value)
                          return
                        }
                        handleCharacterEndInputChange(event.target.value)
                      }}
                      onBlur={
                        lengthFilterUnit === 'reading_time'
                          ? commitReadingTimeEndInput
                          : (lengthFilterUnit === 'words' ? commitWordEndInput : commitCharacterEndInput)
                      }
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          event.currentTarget.blur()
                        }
                      }}
                      disabled={!selectedLengthFilterHasBounds}
                      aria-label={`Maximum article length in ${lengthFilterUnit === 'reading_time' ? 'reading minutes' : lengthFilterUnit}`}
                    />
                  </label>
                </div>
                {selectedLengthRangeMin !== null && selectedLengthRangeMax !== null && selectedLengthRangeStart !== null && selectedLengthRangeEnd !== null && (
                  <TwoHandleRangeSlider
                    minValue={selectedLengthRangeMin}
                    maxValue={selectedLengthRangeMax}
                    startValue={selectedLengthRangeStart}
                    endValue={selectedLengthRangeEnd}
                    disabled={!selectedLengthFilterHasBounds || selectedLengthRangeSpan === 0}
                    startAriaLabel="Minimum article length"
                    endAriaLabel="Maximum article length"
                    formatValue={(value) => (
                      lengthFilterUnit === 'reading_time'
                        ? `${formatReadingMinutes(value)} min`
                        : lengthFilterUnit === 'words'
                        ? `${formatWordCount(value)} words`
                        : `${formatCharacterCount(value)} characters`
                    )}
                    onStartValueChange={(nextValue) => {
                      if (lengthFilterUnit === 'reading_time') {
                        handleReadingTimeStartChange(String(nextValue))
                        return
                      }
                      if (lengthFilterUnit === 'words') {
                        handleWordStartChange(String(nextValue))
                        return
                      }
                      handleCharacterStartChange(String(nextValue))
                    }}
                    onEndValueChange={(nextValue) => {
                      if (lengthFilterUnit === 'reading_time') {
                        handleReadingTimeEndChange(String(nextValue))
                        return
                      }
                      if (lengthFilterUnit === 'words') {
                        handleWordEndChange(String(nextValue))
                        return
                      }
                      handleCharacterEndChange(String(nextValue))
                    }}
                  />
                )}
                <div className="year-range-scale" aria-hidden="true">
                  <span>
                    {selectedLengthRangeMin === null
                      ? '—'
                      : (lengthFilterUnit === 'reading_time'
                        ? formatReadingMinutes(selectedLengthRangeMin)
                        : lengthFilterUnit === 'words'
                        ? formatWordCount(selectedLengthRangeMin)
                        : formatCharacterCount(selectedLengthRangeMin))}
                  </span>
                  <span>
                    {selectedLengthRangeMax === null
                      ? '—'
                      : (lengthFilterUnit === 'reading_time'
                        ? formatReadingMinutes(selectedLengthRangeMax)
                        : lengthFilterUnit === 'words'
                        ? formatWordCount(selectedLengthRangeMax)
                        : formatCharacterCount(selectedLengthRangeMax))}
                  </span>
                </div>
                <p className="setting-help-text">
                  {lengthFilterUnit === 'reading_time'
                    ? 'Only return articles whose estimated reading time falls within the selected range, using 250 words per minute.'
                    : lengthFilterUnit === 'words'
                    ? 'Only return articles whose full body text falls within the selected word-count range.'
                    : 'Only return articles whose full body text falls within the selected character range.'}
                </p>
              </div>
              <div
                className={`weight-card full-row lexical-filter-card ${!isLexicalSearchMode ? 'disabled' : ''}`}
                aria-disabled={!isLexicalSearchMode}
              >
                <span>Words to avoid</span>
                <div
                  className={`year-range-value-card year-range-input-card avoid-words-input-card ${!isLexicalSearchMode ? 'disabled' : ''}`}
                >
                  <span id="avoid-words-input-label">Word</span>
                  <div className="avoid-words-token-field">
                    {wordsToAvoid.map(word => (
                      <button
                        key={word}
                        type="button"
                        className="avoid-words-chip"
                        onClick={() => removeWordToAvoid(word)}
                        disabled={!isLexicalSearchMode}
                        aria-label={`Remove ${word} from words to avoid`}
                      >
                        {word}
                        <span className="avoid-words-chip-remove" aria-hidden="true">x</span>
                      </button>
                    ))}
                    <input
                      className="avoid-words-entry-input"
                      type="text"
                      value={wordsToAvoidDraft}
                      onChange={(event) => setWordsToAvoidDraft(event.target.value)}
                      onKeyDown={handleWordsToAvoidKeyDown}
                      disabled={!isLexicalSearchMode}
                      spellCheck={false}
                      placeholder={isLexicalSearchMode && wordsToAvoid.length === 0 ? 'Add word' : ''}
                      aria-labelledby="avoid-words-input-label"
                      aria-describedby="avoid-words-help"
                    />
                  </div>
                </div>
                <p id="avoid-words-help" className="setting-help-text">
                  {isLexicalSearchMode
                    ? 'Exclude tagged words from lexical TF-IDF results.'
                    : (useChunking
                      ? 'Unavailable while Search Granularity is set to Chunks.'
                      : 'Switch Topic Relevance Search Mode to Lexical to use this filter.')}
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
                  <span>Pipeline</span>
                  <h4>Search granularity</h4>
                </div>
                <div className="modal-settings-grid">
                  <div
                    className="weight-card full-row settings-selection-card settings-scroll-target"
                    ref={retrievalGranularitySettingsRef}
                    tabIndex={-1}
                  >
                    <span>Search Granularity</span>
                    <div className="retrieval-model-grid">
                      <button
                        type="button"
                        className={`retrieval-model-button ${!useChunking ? 'active' : ''}`}
                        onClick={() => handleChunkingModeChange(false)}
                      >
                        <strong>Article</strong>
                        <p>Search and score each article as a whole.</p>
                      </button>
                      <button
                        type="button"
                        className={`retrieval-model-button ${useChunking ? 'active' : ''}`}
                        onClick={() => handleChunkingModeChange(true)}
                        disabled={!canUseChunking}
                      >
                        <strong>Chunks</strong>
                        <p>
                          {canUseChunking
                            ? 'Search over semantic chunks in Stage 1, then send the top chunks to the LLM in Stage 2.'
                            : 'Add SPARK_API_KEY or API_KEY to enable chunk-level search.'}
                        </p>
                      </button>
                    </div>
                    {useChunking && (
                      <label className="settings-inline-field settings-range-field">
                        <div className="settings-range-header">
                          <span>Global chunk pool</span>
                          <strong className="settings-range-value">{chunkCandidateTopK}</strong>
                        </div>
                        <input
                          type="range"
                          min="25"
                          max={maxChunkCandidateTopK}
                          step="25"
                          value={chunkCandidateTopK}
                          onChange={(e) => setChunkCandidateTopK(parseChunkCandidateTopKInput(e.target.value, chunkCandidateTopK))}
                        />
                        <div className="settings-range-scale" aria-hidden="true">
                          <span>25</span>
                          <span>{maxChunkCandidateTopK}</span>
                        </div>
                        <p className="setting-help-text">
                          Search this many top chunks globally before grouping them back into articles. The LLM receives the top {chunkArticleTopK} chunks per article.
                        </p>
                      </label>
                    )}
                  </div>
                </div>
              </section>

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
                    <span>Search Method</span>
                    <div className="retrieval-model-grid">
                      {canUseTfidf && (
                        <button
                          type="button"
                          className={`retrieval-model-button ${!useChunking && retrievalModel === 'tfidf' ? 'active' : ''}`}
                          onClick={() => handleTopicSearchModeChange('tfidf')}
                          disabled={!canUseLexicalRetrieval}
                        >
                          <strong>Lexical</strong>
                          <p>{useChunking ? 'Disabled while Search Granularity is set to Chunks.' : 'Pure TF-IDF term matching with cosine similarity.'}</p>
                        </button>
                      )}
                      {canUseSvd && (
                        <button
                          type="button"
                          className={`retrieval-model-button ${effectiveRetrievalModel === 'svd' ? 'active' : ''}`}
                          onClick={() => handleTopicSearchModeChange('svd')}
                          disabled={!canUseSvd}
                        >
                          <strong>Semantic</strong>
                          <p>Truncated-SVD on TF-IDF to compare articles in latent concept space with cosine similarity.</p>
                        </button>
                      )}
                      {canUseMiniLm && (
                        <button
                          type="button"
                          className={`retrieval-model-button ${effectiveRetrievalModel === 'minilm' ? 'active' : ''}`}
                          onClick={() => handleTopicSearchModeChange('minilm')}
                          disabled={!canUseMiniLm}
                        >
                          <strong>Enhanced Semantic</strong>
                          <p>MiniLM dense embeddings over semantic chunks or pooled article embeddings with cosine similarity.</p>
                        </button>
                      )}
                    </div>
                
                  </div>

                  <div className="weight-card full-row settings-toggle-card">
                    <div className="settings-toggle-row">
                      <div className="settings-toggle-copy">
                        <span>Match Score Display</span>
                        <p className="setting-help-text">
                          Raw: Show the original similarity score without scaling. 
                        </p>
                        <p className="setting-help-text">
                          Relative: Show results scaled within this search — the best match appears as 100%.
                        </p>
                      </div>
                      <button
                        type="button"
                        className={`settings-switch-button ${normalizeTopicScores ? 'active' : ''}`}
                        aria-pressed={normalizeTopicScores}
                        onClick={() => setNormalizeTopicScores(current => !current)}
                      >
                        <span className="settings-switch-label">
                          {normalizeTopicScores ? 'Relative' : 'Raw'}
                        </span>
                        <span className="retrieval-toggle-switch" aria-hidden="true">
                          <span className="retrieval-toggle-thumb" />
                        </span>
                      </button>
                    </div>
                  </div>

                </div>
              </section>

              <section className="settings-stage-section">
                <div className="settings-stage-heading">
                  <span>Stage 2</span>
                  <h4>Agreement scoring</h4>
                </div>
                <div className="modal-settings-grid">
                  <div className="weight-card full-row settings-selection-card">
                    <span>Candidate selection</span>
                    <div className="retrieval-model-grid">
                      <button
                        type="button"
                        className={`retrieval-model-button ${rerankSelectionMode === 'manual' ? 'active' : ''}`}
                        onClick={() => setRerankSelectionMode('manual')}
                      >
                        <strong>Fixed Number</strong>
                        <p>Always use a set number of top matches.</p>
                      </button>
                      <button
                        type="button"
                        className={`retrieval-model-button ${rerankSelectionMode === 'automatic' ? 'active' : ''}`}
                        onClick={() => setRerankSelectionMode('automatic')}
                      >
                        <strong>Smart Filter</strong>
                        <p>
                          {useChunking
                            ? 'Only chunks that are at least this relevant will enter the article pool.'
                            : 'Only articles that are at least this relevant will be considered for deeper analysis.'}
                        </p>
                      </button>
                    </div>
                    {rerankSelectionMode === 'manual' ? (
                      <label className="settings-inline-field settings-range-field">
                        <div className="settings-range-header">
                          <span>Number of top matches </span>
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
                          <span>{`Relevance Sensitivity (${effectiveRetrievalLabel})`}</span>
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
                        ? (useChunking
                          ? 'How many aggregated articles move from chunk retrieval into the agreement reranking stage.'
                          : 'How many top retrieval matches move into the agreement reranking stage.')
                        : (useChunking
                          ? `Chunks at or above this raw relevance threshold are pooled first, capped by the global chunk pool, then grouped into articles for agreement reranking.`
                          : `Articles at or above this raw topic relevance threshold move into the agreement reranking stage, with at most ${maxAutoRerankCandidates} articles reranked.`)}
                    </p>
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
                          className={`retrieval-model-button ${effectiveStanceMethod === 'nli' ? 'active' : ''}`}
                          onClick={() => {
                            if (!useChunking) {
                              setStanceMethod('nli')
                            }
                          }}
                          disabled={useChunking}
                        >
                          <strong>Fast</strong>
                          <p>{useChunking ? 'Disabled while Search Granularity is set to Chunks.' : 'Uses Natural Language Inference (DeBERTa) to compare your thesis with the article.'}</p>
                        </button>
                      )}
                      {supportedStanceMethods.includes('llm') && (
                        <button
                          type="button"
                          className={`retrieval-model-button ${effectiveStanceMethod === 'llm' ? 'active' : ''}`}
                          onClick={() => {
                            if (canUseLlmAgreement) {
                              setStanceMethod('llm')
                            }
                          }}
                          disabled={!canUseLlmAgreement}
                        >
                          <strong>Enhanced</strong>
                          <p>Uses an LLM (gpt-oss-20b) to compare your full essay with the article.</p>
                        </button>
                      )}
                    </div>
          
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
                        <strong>Topic match:</strong> How closely the article matches your topic.
                      </p>
                      <p className="parameter-help-item">
                        <strong>Recency:</strong> How recent the article is.
                      </p>
                      <p className="parameter-help-item">
                        <strong>Agreement:</strong> How much the article supports or challenges your viewpoint.
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
