import { useEffect, useMemo, useRef, useState } from 'react'
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
  const essayOptionsRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => setUseLlm(data.use_llm))
  }, [])

  useEffect(() => {
    if (inputMode !== 'stance') {
      return
    }
    setArticles([])
    setError(null)
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
  const trimmedTopic = topic.trim()
  const trimmedOpinion = opinion.trim()
  const canSearchStance = inputMode === 'stance' && trimmedTopic !== '' && trimmedOpinion !== ''
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

  const handleSubmitStance = async (): Promise<void> => {
    if (!canSearchStance || loading) return

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

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data?.error || `Request failed (${response.status})`)
      }
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
        setError('No thesis options were found. Try a longer essay or cleaner PDF text.')
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

  const scrollEssayOptions = (direction: 'left' | 'right'): void => {
    const container = essayOptionsRef.current
    if (!container) return
    const amount = Math.max(240, Math.round(container.clientWidth * 0.7))
    container.scrollBy({
      left: direction === 'left' ? -amount : amount,
      behavior: 'smooth',
    })
  }

  const resultsLabel = useMemo(() => {
    if (inputMode === 'stance') {
      return articles.length > 0 ? 'Top matches by topic and claim-level stance alignment' : ''
    }
    if (articles.length > 0) {
      return 'Essay matches reranked by your selected thesis sentence'
    }
    if (essayCandidates.length > 0) {
      return 'Choose the sentence that best states your essay’s central thesis'
    }
    return 'Paste or upload an essay, then identify its central thesis before searching'
  }, [articles.length, essayCandidates.length, inputMode])

  const aboutTitle = inputMode === 'stance' ? 'Topic and Stance Search' : 'Essay-guided search'
  const showResultsHeader = Boolean(resultsLabel || loading || error)
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
            Either describe your stance on a topic or paste an essay, and we&apos;ll find you relevant articles that support, contradict, or neutrally discuss your position. 
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
                  placeholder="'Governments should welcome refugees.', 'Climate protests are effective,' 'Housing policy needs reform' ..."
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
                <p>Paste text or upload a PDF. We'll score lines which look most thesis-like.</p>
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
                  Use a PDF instead of pasted text. 
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
                Find thesis options
              </button>
            </div>

            {essayCandidates.length > 0 && (
              <div className="essay-candidate-panel">
                <div className="essay-stage-card">
                  <div className="essay-stage-number">2</div>
                  <div className="essay-stage-copy">
                    <h3>Pick the central thesis</h3>
                    <p>Select the one sentence that best captures the essay's main claim or thesis statement.</p>
                  </div>
                </div>

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
                    Rank articles with selected thesis
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="stance-actions">
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

      {showResultsHeader && (
        <div className="results-header">
          {resultsLabel && <p className="results-label">{resultsLabel}</p>}
          {loading && <p className="results-status">Working...</p>}
          {error && <p className="results-status error">{error}</p>}
        </div>
      )}

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
                    <strong>Stage 1: Topic relevance.</strong> We first identify articles that are
                    relevant to your topic. To do this, we compute the similarity between your
                    input and each Guardian article using TF-IDF (Term Frequency-Inverse Document
                    Frequency) representations combined with cosine similarity. This helps us find
                    articles that discuss similar themes and keywords.
                  </p>
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
                </>
              ) : (
                <>
                  <p className="modal-copy">
                    <strong>Stage 1: Essay thesis detection.</strong> We first split your essay into
                    individual sentences using our sentence segmentation pipeline. Then we use a
                    DeBERTa Natural Language Inference (NLI) model to compare each sentence against
                    the hypothesis, &ldquo;This sentence is the author&apos;s main claim.&rdquo; This gives
                    each sentence a claimness score, and we present the top options so you can
                    choose the sentence that best represents your essay&apos;s central thesis.
                  </p>
                  <p className="modal-copy">
                    <strong>Stage 2: Topic relevance.</strong> After you select the best thesis
                    sentence, we identify articles that are relevant to your essay as a whole. To
                    do this, we compute the similarity between your full essay and each Guardian
                    article using TF-IDF (Term Frequency-Inverse Document Frequency)
                    representations combined with cosine similarity. This surfaces articles that
                    discuss similar themes, issues, and vocabulary.
                  </p>
                  <p className="modal-copy">
                    <strong>Stage 3: Thesis relevance.</strong> From the top {rerankTopK}{' '}
                    {rerankTopK === 1 ? 'article' : 'articles'} identified in Stage 2, we then rank
                    them based on how they relate to your selected thesis. We use a DeBERTa NLI
                    model to compare your chosen thesis sentence with each article&apos;s central
                    argument, which was extracted beforehand using an LLM. The model estimates
                    whether each article supports, contradicts, or is neutral toward your thesis,
                    and we rank the results accordingly.
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
