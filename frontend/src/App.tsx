import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { Article, EssayClaimCandidate, EssayClaimCandidateResponse } from './types'
import Chat from './Chat'

type InputMode = 'stance' | 'essay'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [inputMode, setInputMode] = useState<InputMode>('stance')
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [topic, setTopic] = useState<string>('')
  const [opinion, setOpinion] = useState<string>('')
  const [topicWeight, setTopicWeight] = useState<number>(0.5)
  const [stanceWeight, setStanceWeight] = useState<number>(0.5)
  const [rerankTopK, setRerankTopK] = useState<number>(20)
  const [articles, setArticles] = useState<Article[]>([])
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [isAboutOpen, setIsAboutOpen] = useState<boolean>(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false)
  const [essayCandidates, setEssayCandidates] = useState<EssayClaimCandidate[]>([])
  const [essayPreparedText, setEssayPreparedText] = useState<string>('')
  const [selectedEssayCandidateId, setSelectedEssayCandidateId] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => setUseLlm(data.use_llm))
  }, [])

  useEffect(() => {
    if (inputMode !== 'stance') {
      return
    }

    const trimmedTopic = topic.trim()
    const trimmedOpinion = opinion.trim()
    if (trimmedTopic === '' || trimmedOpinion === '') {
      setArticles([])
      setError(null)
      setLoading(false)
      return
    }

    const controller = new AbortController()
    const timeoutId = setTimeout(async () => {
      setLoading(true)
      setError(null)

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
          signal: controller.signal,
        })

        const data = await response.json()
        if (!response.ok) {
          throw new Error(data?.error || `Request failed (${response.status})`)
        }
        setArticles(Array.isArray(data) ? data : [])
      } catch (fetchError) {
        if (fetchError instanceof DOMException && fetchError.name === 'AbortError') return
        console.error('Search request failed:', fetchError)
        setArticles([])
        setError(fetchError instanceof Error ? fetchError.message : 'Search request failed.')
      } finally {
        setLoading(false)
      }
    }, 300)

    return () => {
      controller.abort()
      clearTimeout(timeoutId)
    }
  }, [inputMode, opinion, rerankTopK, stanceWeight, topic, topicWeight])

  useEffect(() => {
    if (inputMode !== 'essay') {
      return
    }
    setEssayCandidates([])
    setEssayPreparedText('')
    setSelectedEssayCandidateId(null)
    setArticles([])
    setError(null)
  }, [inputMode, pdfFile, searchTerm])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      setIsAboutOpen(false)
      setIsSettingsOpen(false)
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const trimmedEssayText = searchTerm.trim()
  const canAnalyzeEssay = inputMode === 'essay' && (trimmedEssayText !== '' || !!pdfFile)
  const selectedEssayCandidate = useMemo(
    () => essayCandidates.find(candidate => candidate.sentence_id === selectedEssayCandidateId) ?? null,
    [essayCandidates, selectedEssayCandidateId],
  )
  const canSubmitEssay = Boolean(essayPreparedText && selectedEssayCandidate)

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

  const handleEssaySearch = (value: string): void => {
    setInputMode('essay')
    setPdfFile(null)
    setSearchTerm(value)
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
      if (pdfFile) {
        formData.append('pdf', pdfFile)
      }

      const response = await fetch('/api/essay/claim-candidates', {
        method: 'POST',
        body: formData,
      })
      const data: EssayClaimCandidateResponse & { error?: string } = await response.json()
      if (!response.ok) {
        throw new Error(data?.error || `Request failed (${response.status})`)
      }

      setEssayPreparedText(data.essay_text || trimmedEssayText)
      setEssayCandidates(data.candidates || [])
      setSelectedEssayCandidateId(data.candidates?.[0]?.sentence_id || null)
      if (!data.candidates || data.candidates.length === 0) {
        setError('No claim candidates were found. Try a longer essay or cleaner PDF text.')
      }
    } catch (fetchError) {
      console.error('Essay analysis failed:', fetchError)
      setEssayCandidates([])
      setSelectedEssayCandidateId(null)
      setEssayPreparedText('')
      setError(fetchError instanceof Error ? fetchError.message : 'Essay analysis failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmitEssay = async (): Promise<void> => {
    if (!canSubmitEssay || loading || !selectedEssayCandidate) return

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

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error || `Request failed (${response.status})`)
      }
      setArticles(Array.isArray(data) ? data : [])
    } catch (fetchError) {
      console.error('Essay search failed:', fetchError)
      setArticles([])
      setError(fetchError instanceof Error ? fetchError.message : 'Essay search failed.')
    } finally {
      setLoading(false)
    }
  }

  const resultsLabel = useMemo(() => {
    if (inputMode === 'stance') {
      return 'Top topic matches reranked by claim-level stance alignment'
    }
    if (articles.length > 0) {
      return 'Essay matches reranked by your selected thesis sentence'
    }
    if (essayCandidates.length > 0) {
      return 'Choose the sentence that best states your essay’s central thesis'
    }
    return 'Paste or upload an essay, then identify its central thesis before searching'
  }, [articles.length, essayCandidates.length, inputMode])

  const aboutTitle = inputMode === 'stance' ? 'Two-stage stance search' : 'Essay-guided search'
  const showScoreGrid = (article: Article): boolean => (
    article.combined_score !== undefined ||
    article.stance_score_normalized !== undefined ||
    article.topic_score_normalized !== undefined
  )

  if (useLlm === null) return <></>

  return (
    <div className={`full-body-container ${useLlm ? 'llm-mode' : ''}`}>
      <div className="top-text">
        <div className="hero-copy">
          <h1>hear! hear!</h1>
          <h2>Find your voice in Guardian opinion articles</h2>
          <p className="hero-subtitle">
            Search by topic and stance, or let the app help isolate the central thesis of a full
            essay before reranking articles against that selected claim.
          </p>
        </div>

        <div className="mode-switch" role="tablist" aria-label="Search mode">
          <button
            type="button"
            className={`mode-pill ${inputMode === 'stance' ? 'active' : ''}`}
            onClick={() => setInputMode('stance')}
          >
            Topic + Opinion
          </button>
          <button
            type="button"
            className={`mode-pill ${inputMode === 'essay' ? 'active' : ''}`}
            onClick={() => setInputMode('essay')}
          >
            Essay
          </button>
        </div>

        {inputMode === 'stance' ? (
          <div className="stance-panel">
            <div className="stance-prompt-card">
              <label className="stance-prompt-line">
                <span className="stance-prefix">Regarding</span>
                <input
                  type="text"
                  placeholder="Climate protest, immigration, housing policy..."
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  aria-label="Topic"
                />
              </label>
              <label className="stance-prompt-line opinion-line">
                <span className="stance-prefix">I believe</span>
                <textarea
                  placeholder="Governments should invest far more in public transit than highway expansion."
                  value={opinion}
                  onChange={(e) => setOpinion(e.target.value)}
                  rows={4}
                  aria-label="Opinion"
                />
              </label>
            </div>
          </div>
        ) : (
          <div className="essay-panel">
            <div className="essay-stage-card">
              <div className="essay-stage-number">1</div>
              <div className="essay-stage-copy">
                <h3>Add your essay</h3>
                <p>Paste text or upload a PDF. We’ll sentence-split it and score which lines look most thesis-like.</p>
              </div>
            </div>

            <div className="stance-prompt-card essay-prompt-card">
              <label className="stance-prompt-line opinion-line">
                <span className="stance-prefix">Essay text</span>
                <textarea
                  id="search-input"
                  placeholder="Paste an essay, paper, op-ed..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  rows={5}
                  aria-label="Essay or search phrase"
                />
              </label>
            </div>

            <div className="essay-divider" aria-hidden="true">
              <span>OR</span>
            </div>

            <div className="stance-prompt-card essay-upload-card">
              <div className="essay-upload-header">
                <span className="stance-prefix">Upload PDF</span>
                <p className="essay-upload-copy">
                  Use a PDF instead of pasted text. The extracted text will be used for thesis detection and article matching.
                </p>
              </div>
              <div className="essay-upload-controls">
                <label htmlFor="pdf-upload" className="pdf-upload-button">
                  Choose PDF
                </label>
                <input
                  id="pdf-upload"
                  className="pdf-upload-input"
                  type="file"
                  accept="application/pdf"
                  onChange={(e) => {
                    const file = e.target.files?.[0] ?? null
                    setPdfFile(file)
                  }}
                />
                {pdfFile && (
                  <span className="pdf-file-chip">
                    {pdfFile.name}
                    <button
                      type="button"
                      onClick={() => setPdfFile(null)}
                      className="clear-pdf-button"
                    >
                      Remove
                    </button>
                  </span>
                )}
              </div>
            </div>

            <div className="essay-actions">
              <button
                type="button"
                className="primary-action-button"
                onClick={handleAnalyzeEssay}
                disabled={!canAnalyzeEssay || loading}
              >
                Find thesis candidates
              </button>
              <p className="essay-action-help">
                We score each sentence against the hypothesis “This sentence is the author&apos;s main claim.”
              </p>
            </div>

            {essayCandidates.length > 0 && (
              <div className="essay-candidate-panel">
                <div className="essay-stage-card">
                  <div className="essay-stage-number">2</div>
                  <div className="essay-stage-copy">
                    <h3>Pick the central thesis</h3>
                    <p>Select the one sentence that best captures the essay’s main claim. You can change your mind before submitting.</p>
                  </div>
                </div>

                <div className="essay-candidate-grid">
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
                          <span className="candidate-rank">Candidate {index + 1}</span>
                          <span className="candidate-score">
                            Claimness {formatScore(candidate.claim_score_normalized)}
                          </span>
                        </div>
                        <p className="candidate-sentence">{candidate.sentence}</p>
                        <div className="candidate-metrics">
                          <span>Entail {formatPercent(candidate.entailment_prob)}</span>
                          <span>Neutral {formatPercent(candidate.neutral_prob)}</span>
                          <span>Contradict {formatPercent(candidate.contradiction_prob)}</span>
                        </div>
                      </button>
                    )
                  })}
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
                    className="primary-action-button secondary-action"
                    onClick={handleSubmitEssay}
                    disabled={!canSubmitEssay || loading}
                  >
                    Rank articles with selected thesis
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="stance-actions">
          <button
            type="button"
            className="utility-pill"
            onClick={() => setIsAboutOpen(true)}
          >
            About
          </button>
          <button
            type="button"
            className="utility-pill"
            onClick={() => setIsSettingsOpen(true)}
          >
            Search settings
          </button>
        </div>
      </div>

      <div className="results-header">
        <p className="results-label">{resultsLabel}</p>
        {loading && <p className="results-status">Working...</p>}
        {error && <p className="results-status error">{error}</p>}
      </div>

      <div id="answer-box">
        {articles.map((article) => (
          <div key={article.id} className="article-item">
            <h3 className="article-title">
              <a href={article.url} target="_blank" rel="noreferrer">{article.title}</a>
            </h3>

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
                  <strong>{article.stance_label || 'unavailable'}</strong>
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

            <p className="article-summary">{article.summary}</p>

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
                No LLM-coded central claim is available for this article yet, so it stayed in the
                ranking based on essay/topic relevance alone.
              </p>
            )}

            <p className="article-meta">
              {article.author_display || article.author_raw || 'Unknown author'} | {formatDate(article.date)}
            </p>
          </div>
        ))}
      </div>

      {useLlm && <Chat onSearchTerm={handleEssaySearch} />}

      {isAboutOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setIsAboutOpen(false)}
          role="presentation"
        >
          <div
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="about-reranking-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <p className="modal-eyebrow">About</p>
                <h3 id="about-reranking-title">{aboutTitle}</h3>
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
            <div className="modal-stage-list">
              {inputMode === 'stance' ? (
                <>
                  <p className="modal-copy">
                    <strong>Stage 1:</strong> TF-IDF retrieves the top {rerankTopK}{' '}
                    {rerankTopK === 1 ? 'article' : 'articles'} that are most relevant to your topic.
                  </p>
                  <p className="modal-copy">
                    <strong>Stage 2:</strong> NLI reranks those {rerankTopK}{' '}
                    {rerankTopK === 1 ? 'article' : 'articles'} by comparing your opinion with each
                    article&apos;s LLM-coded central claim to estimate whether the article supports,
                    contradicts, or stays neutral toward your stance.
                  </p>
                </>
              ) : (
                <>
                  <p className="modal-copy">
                    <strong>Stage 1:</strong> the essay is sentence-split and each sentence is scored
                    for “claimness” against the hypothesis that it is the author&apos;s main claim.
                  </p>
                  <p className="modal-copy">
                    <strong>Stage 2:</strong> you choose the best thesis candidate, then the whole essay
                    retrieves top TF-IDF matches and the selected thesis sentence reranks those articles
                    with NLI against each article&apos;s LLM-coded central claim.
                  </p>
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
