import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react'
import './App.css'
import {
  Article,
  EssayClaimCandidate,
  EssayClaimCandidateResponse,
  EssayTextExtractionResponse,
} from './types'
import Chat from './Chat'

type InputMode = 'stance' | 'essay'
type IntroStage = 0 | 1 | 2
type EssayStep = 1 | 2

type ConfigResponse = {
  use_llm: boolean
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

const summarizeApiText = (value: string, maxLength = 180): string => (
  value.replace(/\s+/g, ' ').trim().slice(0, maxLength)
)

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

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [introSequenceKey, setIntroSequenceKey] = useState<number>(0)
  const [introStage, setIntroStage] = useState<IntroStage>(0)
  const [typedTopic, setTypedTopic] = useState<string>('')
  const [typedClaim, setTypedClaim] = useState<string>('')
  const [inputMode, setInputMode] = useState<InputMode>('stance')
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [topic, setTopic] = useState<string>('')
  const [opinion, setOpinion] = useState<string>('')
  const [topicWeight, setTopicWeight] = useState<number>(0.5)
  const [stanceWeight, setStanceWeight] = useState<number>(0.5)
  const [rerankTopK, setRerankTopK] = useState<number>(20)
  const [articles, setArticles] = useState<Article[]>([])
  const [isImportingPdf, setIsImportingPdf] = useState<boolean>(false)
  const [importedPdfName, setImportedPdfName] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [isAboutOpen, setIsAboutOpen] = useState<boolean>(false)
  const [activeAboutTab, setActiveAboutTab] = useState<InputMode>('stance')
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false)
  const [essayCandidates, setEssayCandidates] = useState<EssayClaimCandidate[]>([])
  const [essayPreparedText, setEssayPreparedText] = useState<string>('')
  const [selectedEssayCandidateId, setSelectedEssayCandidateId] = useState<string | null>(null)
  const [essayActiveStep, setEssayActiveStep] = useState<EssayStep>(1)
  const essayOptionsRef = useRef<HTMLDivElement | null>(null)
  const resultsSectionRef = useRef<HTMLDivElement | null>(null)
  const touchStartYRef = useRef<number | null>(null)
  const [isSearchStageVisible, setIsSearchStageVisible] = useState<boolean>(false)
  const [hasSubmittedSearch, setHasSubmittedSearch] = useState<boolean>(false)

  useEffect(() => {
    let isActive = true

    const loadConfig = async (): Promise<void> => {
      try {
        const response = await fetch('/api/config')
        const data = await readApiJson<ConfigResponse>(response)
        if (!isActive) return
        setUseLlm(Boolean(data.use_llm))
      } catch (configError) {
        console.error('Config request failed:', configError)
        if (!isActive) return
        setUseLlm(false)
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
  }, [introSequenceKey])

  useEffect(() => {
    if (inputMode !== 'stance') {
      return
    }
    setArticles([])
    setError(null)
    setHasSubmittedSearch(false)
  }, [inputMode, opinion, rerankTopK, stanceWeight, topic, topicWeight])

  useEffect(() => {
    if (inputMode !== 'essay') {
      return
    }
    setEssayCandidates([])
    setEssayPreparedText('')
    setSelectedEssayCandidateId(null)
    setEssayActiveStep(1)
    setArticles([])
    setError(null)
    setHasSubmittedSearch(false)
  }, [inputMode, searchTerm])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      setIsAboutOpen(false)
      setIsSettingsOpen(false)
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

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

    const frameId = window.requestAnimationFrame(() => {
      scrollToNode(resultsSectionRef.current)
    })

    return () => window.cancelAnimationFrame(frameId)
  }, [hasSubmittedSearch])

  const trimmedEssayText = searchTerm.trim()
  const trimmedTopic = topic.trim()
  const trimmedOpinion = opinion.trim()
  const canSearchStance = inputMode === 'stance' && trimmedTopic !== '' && trimmedOpinion !== ''
  const canAnalyzeEssay = inputMode === 'essay' && trimmedEssayText !== ''
  const selectedEssayCandidate = useMemo(
    () => essayCandidates.find(candidate => candidate.sentence_id === selectedEssayCandidateId) ?? null,
    [essayCandidates, selectedEssayCandidateId],
  )
  const canSubmitEssay = Boolean(essayPreparedText && selectedEssayCandidate)
  const isEssayStepTwoAvailable = essayCandidates.length > 0
  const essayWorkflowStep = isEssayStepTwoAvailable ? essayActiveStep : 1

  const formatDate = (isoDate: string | null): string => {
    if (!isoDate) return 'Unknown date'
    const parsed = new Date(isoDate)
    if (Number.isNaN(parsed.getTime())) return 'Unknown date'
    return parsed.toLocaleDateString()
  }

  const formatScore = (value?: number | null): string => {
    if (value === undefined || value === null || Number.isNaN(value)) return 'n/a'
    return value.toFixed(3)
  }

  const formatPercent = (value?: number | null): string => {
    if (value === undefined || value === null || Number.isNaN(value)) return 'n/a'
    return `${Math.round(value * 100)}%`
  }

  const parseWeightInput = (value: string, fallback: number): number => {
    if (value.trim() === '') return fallback
    const parsed = Number(value)
    if (Number.isNaN(parsed) || parsed < 0) return fallback
    return parsed
  }

  const parseTopKInput = (value: string, fallback: number): number => {
    if (value.trim() === '') return fallback
    const parsed = Number(value)
    if (Number.isNaN(parsed)) return fallback
    return Math.min(100, Math.max(1, Math.round(parsed)))
  }

  const scrollToNode = (node: HTMLDivElement | null): void => {
    node?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    })
  }

  const handleEssaySearch = (value: string): void => {
    setInputMode('essay')
    setImportedPdfName(null)
    setSearchTerm(value)
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
    setHasSubmittedSearch(false)
    setEssayCandidates([])
    setSelectedEssayCandidateId(null)
    setEssayPreparedText('')

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

  const handleSubmitStance = async (): Promise<void> => {
    if (!canSearchStance || loading) return

    setHasSubmittedSearch(true)
    if (typeof document !== 'undefined') {
      document.body.style.overflow = ''
    }
    scrollToNode(resultsSectionRef.current)
    setLoading(true)
    setError(null)
    setArticles([])

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
          top_k: rerankTopK,
        }),
      })

      const data = await readApiJson<Article[]>(response)
      setArticles(Array.isArray(data) ? data : [])
    } catch (fetchError) {
      console.error('Search request failed:', fetchError)
      setArticles([])
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

      setEssayPreparedText(data.essay_text || trimmedEssayText)
      setEssayCandidates(data.candidates || [])
      setSelectedEssayCandidateId(data.candidates?.[0]?.sentence_id || null)
      setEssayActiveStep((data.candidates && data.candidates.length > 0) ? 2 : 1)
      if (!data.candidates || data.candidates.length === 0) {
        setError('No thesis were found. Try a longer essay or cleaner PDF text.')
      }
    } catch (fetchError) {
      console.error('Essay analysis failed:', fetchError)
      setEssayCandidates([])
      setSelectedEssayCandidateId(null)
      setEssayPreparedText('')
      setEssayActiveStep(1)
      setError(fetchError instanceof Error ? fetchError.message : 'Essay analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmitEssay = async (): Promise<void> => {
    if (!canSubmitEssay || loading || !selectedEssayCandidate) return

    setHasSubmittedSearch(true)
    if (typeof document !== 'undefined') {
      document.body.style.overflow = ''
    }
    scrollToNode(resultsSectionRef.current)
    setLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/articles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode: 'essay',
          q: essayPreparedText,
          selected_thesis_id: selectedEssayCandidate.sentence_id,
          selected_thesis_sentence: selectedEssayCandidate.sentence,
          topic_weight: topicWeight,
          stance_weight: stanceWeight,
          top_k: rerankTopK,
        }),
      })

      const data = await readApiJson<Article[]>(response)
      setArticles(Array.isArray(data) ? data : [])
    } catch (fetchError) {
      console.error('Essay search failed:', fetchError)
      setArticles([])
      setError(fetchError instanceof Error ? fetchError.message : 'Essay search failed.')
    } finally {
      setLoading(false)
    }
  }

  const scrollEssayOptions = (direction: 'left' | 'right'): void => {
    const container = essayOptionsRef.current
    if (!container) return
    const amount = Math.max(240, Math.round(container.clientWidth * 0.7))
    container.scrollBy({
      left: direction === 'left' ? -amount : amount,
      behavior: 'smooth',
    })
  }

  const openAbout = (tab: InputMode = inputMode): void => {
    setActiveAboutTab(tab)
    setIsAboutOpen(true)
  }
  const showScoreGrid = (article: Article): boolean => (
    article.combined_score !== undefined ||
    article.stance_score_normalized !== undefined ||
    article.topic_score_normalized !== undefined
  )
  const formatArticleLabel = (label?: string | null): string => (
    label ? label.replace(/_/g, ' ') : 'Relevant'
  )

  const getArticleTone = (label?: string | null): string => {
    const normalized = label?.toLowerCase() ?? ''

    if (normalized.includes('support') || normalized.includes('entail')) return 'support'
    if (normalized.includes('contradict')) return 'contradict'
    if (normalized.includes('neutral')) return 'neutral'
    return 'default'
  }

  const resultsDescription = useMemo(() => {
    if (loading) {
      return inputMode === 'stance'
        ? 'Reading across Guardian opinion pieces for topical fit and stance alignment.'
        : 'Ranking Guardian opinion pieces against the thesis you selected.'
    }

    if (error) {
      return 'Something interrupted the search. Adjust the prompt above or try again.'
    }

    if (!hasSubmittedSearch) {
      return inputMode === 'stance'
        ? 'Submit a topic and stance above to open a page of supporting, opposing, and neutral perspectives.'
        : 'Paste an essay, choose its thesis, and your ranked Guardian matches will appear here.'
    }

    if (articles.length === 0) {
      return 'No matching articles came back this time. Try broadening the topic or sharpening the claim.'
    }

    return inputMode === 'stance'
      ? `${articles.length} Guardian opinion ${articles.length === 1 ? 'piece' : 'pieces'} ranked by topic and stance alignment.`
      : `${articles.length} Guardian opinion ${articles.length === 1 ? 'piece' : 'pieces'} ranked against your selected thesis.`
  }, [articles.length, error, hasSubmittedSearch, inputMode, loading])

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

              {inputMode === 'essay' && isSearchStageVisible && (
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
                        <span className="essay-progress-note">Paste text or bring in a PDF.</span>
                      </div>
                    </button>

                    <button
                      type="button"
                      className={`essay-progress-step ${
                        essayWorkflowStep === 2
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
                            ? 'Pick the sentence that anchors the search.'
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
                          ? `Text imported from ${importedPdfName}. You can keep editing it here before extracting thesis options.`
                          : 'Have a PDF already? Upload it and we’ll drop the extracted text into this editor so you can revise it.'}
                      </p>
                      <label
                        htmlFor="pdf-upload"
                        className={`essay-upload-trigger ${isImportingPdf ? 'disabled' : ''}`}
                        aria-disabled={isImportingPdf}
                      >
                        {isImportingPdf ? 'reading PDF...' : (importedPdfName ? 'replace with another PDF' : 'fill editor from PDF')}
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
                      onClick={handleAnalyzeEssay}
                      disabled={!canAnalyzeEssay || loading}
                    >
                      {(loading && essayWorkflowStep === 1) ? 'Extracting thesis...' : 'Extract thesis options'}
                    </button>
                  </div>
                </>
              )}

              {isEssayStepTwoAvailable && essayWorkflowStep === 2 && (
                <div className="essay-candidate-panel">

                  <div className="essay-option-strip">
                    <div className="essay-option-strip-header">
                      <p className="essay-option-strip-title">Thesis options</p>
                      <div className="essay-option-strip-controls">
                        <p className="essay-option-strip-note">Scroll sideways to compare them.</p>
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
                      </div>
                    </div>
                    <div className="essay-candidate-grid" ref={essayOptionsRef}>
                      {essayCandidates.map((candidate, index) => {
                        const isSelected = candidate.sentence_id === selectedEssayCandidateId
                        return (
                          <button
                            key={candidate.sentence_id}
                            type="button"
                            className={`candidate-card ${isSelected ? 'selected' : ''}`}
                            onClick={() => setSelectedEssayCandidateId(candidate.sentence_id)}
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
                  </div>

                  <div className="essay-submit-panel">
                    <div>
                      <p className="essay-submit-eyebrow">Selected thesis</p>
                      <p className="essay-submit-copy">
                        {selectedEssayCandidate?.sentence || 'Choose a sentence above to continue.'}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="primary-action-button"
                      onClick={handleSubmitEssay}
                      disabled={!canSubmitEssay || loading}
                    >
                      {(loading && essayWorkflowStep === 2) ? 'Searching...' : 'Search with selected thesis'}
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
                onClick={handleSubmitStance}
                disabled={!canSearchStance || loading}
              >
                Search
              </button>
            )}
            <button
              type="button"
              className="utility-pill"
              onClick={() => setIsSettingsOpen(true)}
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
          <div className="results-paper">
            <div className="results-paper-header">
              <p className="results-paper-eyebrow">Results</p>
              <h2>Guardian opinion matches</h2>
              <p className="results-paper-copy">{resultsDescription}</p>
            </div>

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
                {articles.map((article) => (
                  <article key={article.id} className="article-item">
                    <div className="article-topline">
                      <span className={`article-kicker ${getArticleTone(article.stance_label)}`}>
                        {formatArticleLabel(article.stance_label)}
                      </span>
                      <p className="article-meta">
                        {article.author_display || article.author_raw || 'Unknown author'} | {formatDate(article.date)}
                      </p>
                    </div>

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
                      <div className="score-grid">
                        <div className="score-chip">
                          <span>Combined</span>
                          <strong>{formatScore(article.combined_score)}</strong>
                        </div>
                        <div className="score-chip">
                          <span>Topic</span>
                          <strong>{formatScore(article.topic_score_normalized)}</strong>
                        </div>
                        <div className="score-chip">
                          <span>Stance</span>
                          <strong>{formatScore(article.stance_score_normalized)}</strong>
                        </div>
                        <div className="score-chip">
                          <span>Label</span>
                          <strong>{formatArticleLabel(article.stance_label)}</strong>
                        </div>
                        <div className="score-chip">
                          <span>Entail</span>
                          <strong>{formatPercent(article.stance_entailment_prob)}</strong>
                        </div>
                        <div className="score-chip">
                          <span>Contradict</span>
                          <strong>{formatPercent(article.stance_contradiction_prob)}</strong>
                        </div>
                      </div>
                    )}

                    {article.thesis_sentence && (
                      <div className="sentence-block">
                        <h4>Thesis sentence</h4>
                        <p>{article.thesis_sentence}</p>
                      </div>
                    )}

                    {article.support_sentences && article.support_sentences.length > 0 && (
                      <div className="sentence-block">
                        <h4>Support sentences</h4>
                        <ul className="sentence-list">
                          {article.support_sentences.map((sentence, index) => (
                            <li key={`${article.id}-support-${index}`}>{sentence}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {showScoreGrid(article) && !article.central_claim_summary && (
                      <p className="claim-missing">
                        No extracted central claim is available for this article yet, so it stayed in the ranking based on topic or essay relevance alone.
                      </p>
                    )}

                    <div className="article-footer-row">
                      <div className="similarity-block">
                        <div className="similarity-title">Cosine similarity</div>
                        <div className="similarity-row">
                          <div className="similarity-bar">
                            <div
                              className="similarity-fill"
                              style={{ width: `${(article.score ?? 0) * 100}%` }}
                            />
                          </div>
                          <span className="similarity-label">
                            {formatPercent(article.score)}
                          </span>
                        </div>
                      </div>

                      {article.keywords && article.keywords.length > 0 && (
                        <div className="keyword-block">
                          <p>Keywords</p>
                          <div className="keyword-list">
                            {article.keywords.map((kw, index) => (
                              <span key={`${article.id}-keyword-${index}`} className="keyword-chip">{kw}</span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            )}

            {!loading && !error && articles.length === 0 && (
              <div className="results-empty-card searched">
                <p>
                  No matching articles were returned. Try broadening the topic or making the stance more explicit.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {useLlm && <Chat onSearchTerm={handleEssaySearch} />}

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
                      input and each Guardian article using TF-IDF (Term Frequency-Inverse Document
                      Frequency) representations combined with cosine similarity. This helps us find
                      articles that discuss similar themes and keywords.
                    </p>
                  </section>
                  <section className="about-section">
                    <p className="about-section-label">Stage 2</p>
                    <p className="modal-copy">
                      <strong>Stage 2: Stance relevance.</strong> From the top {rerankTopK}{' '}
                      {rerankTopK === 1 ? 'article' : 'articles'} identified in Stage 1, we then rank
                      them based on how they relate to your opinion. We use a Natural Language
                      Inference (NLI) model, DeBERTa (Decoding-enhanced BERT with disentangled
                      attention), to compare your claim with each article&apos;s central argument
                      (extracted using an LLM). The model estimates whether each article supports,
                      contradicts, or is neutral toward your stance, and we rank the results
                      accordingly.
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
                      choose the sentence that best represents your essay&apos;s central thesis.
                    </p>
                  </section>
                  <section className="about-section">
                    <p className="about-section-label">Stage 2</p>
                    <p className="modal-copy">
                      <strong>Stage 2: Topic relevance.</strong> After you select the best thesis
                      sentence, we identify articles that are relevant to your essay as a whole. To
                      do this, we compute the similarity between your full essay and each Guardian
                      article using TF-IDF (Term Frequency-Inverse Document Frequency)
                      representations combined with cosine similarity. This surfaces articles that
                      discuss similar themes, issues, and vocabulary.
                    </p>
                  </section>
                  <section className="about-section">
                    <p className="about-section-label">Stage 3</p>
                    <p className="modal-copy">
                      <strong>Stage 3: Thesis relevance.</strong> From the top {rerankTopK}{' '}
                      {rerankTopK === 1 ? 'article' : 'articles'} identified in Stage 2, we then rank
                      them based on how they relate to your selected thesis. We use a DeBERTa NLI
                      model to compare your chosen thesis sentence with each article&apos;s central
                      argument, which was extracted beforehand using an LLM. The model estimates
                      whether each article supports, contradicts, or is neutral toward your thesis,
                      and we rank the results accordingly.
                    </p>
                  </section>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {isSettingsOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setIsSettingsOpen(false)}
          role="presentation"
        >
          <div
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="search-settings-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <p className="modal-eyebrow">Settings</p>
                <h3 id="search-settings-title">Reranking settings</h3>
              </div>
              <button
                type="button"
                className="modal-close"
                onClick={() => setIsSettingsOpen(false)}
                aria-label="Close settings popup"
              >
                Close
              </button>
            </div>
            <div className="modal-settings-grid">
              <label className="weight-card full-row">
                <span>Top K</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  step="1"
                  value={rerankTopK}
                  onChange={(e) => setRerankTopK(parseTopKInput(e.target.value, rerankTopK))}
                />
                <p className="setting-help-text">
                  How many TF-IDF matches move into the NLI reranking stage.
                </p>
              </label>
              <div className="weight-card full-row weights-group-card">
                <span>Weights</span>
                <div className="weight-pair-grid">
                  <label className="paired-weight-field">
                    <span>Topic / essay weight</span>
                    <input
                      type="number"
                      min="0"
                      step="0.05"
                      value={topicWeight}
                      onChange={(e) => setTopicWeight(parseWeightInput(e.target.value, topicWeight))}
                    />
                  </label>
                  <label className="paired-weight-field">
                    <span>Stance / thesis weight</span>
                    <input
                      type="number"
                      min="0"
                      step="0.05"
                      value={stanceWeight}
                      onChange={(e) => setStanceWeight(parseWeightInput(e.target.value, stanceWeight))}
                    />
                  </label>
                </div>
                <div className="parameter-help-list">
                  <p className="parameter-help-item">
                    <strong>Topic / essay weight:</strong> how much the final score prioritizes whole-text topical similarity.
                  </p>
                  <p className="parameter-help-item">
                    <strong>Stance / thesis weight:</strong> how much the final score prioritizes whether the selected claim aligns with an article&apos;s central claim.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
