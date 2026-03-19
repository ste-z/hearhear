import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { Article } from './types'
import Chat from './Chat'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [articles, setArticles] = useState<Article[]>([])
  const [pdfFile, setPdfFile] = useState<File | null>(null) //pdf upload constant
  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => setUseLlm(data.use_llm))
  }, [])

  useEffect(() => {
    const trimmed = searchTerm.trim()
    if (trimmed === '' && !pdfFile) {
      setArticles([])
      return
    }

    const controller = new AbortController()
    const timeoutId = setTimeout(async () => {
      try {
        // const response = await fetch('/api/articles', {
        //   method: 'POST',
        //   headers: {
        //     'Content-Type': 'application/json',
        //   },
        //   body: JSON.stringify({ q: trimmed }),
        //   signal: controller.signal,
        // })

        //changed for pdf 
        const formData = new FormData()
        formData.append('q', trimmed)
        if (pdfFile) {
          formData.append('pdf', pdfFile)
        }

        const response = await fetch('/api/articles', {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        })
        const data: Article[] = await response.json()
        setArticles(data)
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return
        console.error('Search request failed:', error)
      }
    }, 300)

    return () => {
      controller.abort()
      clearTimeout(timeoutId)
    }
  }, [searchTerm, pdfFile]) //changed for pdffile compatible

  const formatDate = (isoDate: string | null): string => {
    if (!isoDate) return 'Unknown date'
    const parsed = new Date(isoDate)
    if (Number.isNaN(parsed.getTime())) return 'Unknown date'
    return parsed.toLocaleDateString()
  }

  const handleSearch = (value: string): void => {
    setSearchTerm(value)
  }

  if (useLlm === null) return <></>

  return (
    <div className={`full-body-container ${useLlm ? 'llm-mode' : ''}`}>
      {/* Search bar (always shown) */}
      <div className="top-text">
        <div>
          <h1>hear! hear!</h1>
          <h2>Find your voice in Guardian opinion articles</h2>
        </div>
        <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
          <img src={SearchIcon} alt="search" />
          <textarea
            id="search-input"
            placeholder="Search Guardian opinion articles or paste a paragraph..."
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
            rows={3}
          />
        </div>
        <div className="pdf-upload-row">
          <label htmlFor="pdf-upload" className="pdf-upload-label">
            Upload PDF
          </label>
          <input
            id="pdf-upload"
            type="file"
            accept="application/pdf"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null
              setPdfFile(file)
            }}
          />
          {pdfFile && (
            <span className="pdf-file-name">
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

      {/* Search results (always shown) */}
      <div id="answer-box">
        {articles.map((article) => (
          <div key={article.id} className="article-item">
            <h3 className="article-title">
              <a href={article.url} target="_blank" rel="noreferrer">{article.title}</a>
            </h3>
            <p className="article-summary">{article.summary}</p>
            <p className="article-meta">
              {article.author_display || article.author_raw || 'Unknown author'} | {formatDate(article.date)}
            </p>
          </div>
        ))}
      </div>

      {/* Chat (only when USE_LLM = True in routes.py) */}
      {useLlm && <Chat onSearchTerm={handleSearch} />}
    </div>
  )
}

export default App
