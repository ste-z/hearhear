import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { Article } from './types'
import Chat from './Chat'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [articles, setArticles] = useState<Article[]>([])

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => setUseLlm(data.use_llm))
  }, [])

  const formatDate = (isoDate: string | null): string => {
    if (!isoDate) return 'Unknown date'
    const parsed = new Date(isoDate)
    if (Number.isNaN(parsed.getTime())) return 'Unknown date'
    return parsed.toLocaleDateString()
  }

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)
    if (value.trim() === '') { setArticles([]); return }
    const response = await fetch(`/api/articles?q=${encodeURIComponent(value)}`)
    const data: Article[] = await response.json()
    setArticles(data)
  }

  if (useLlm === null) return <></>

  return (
    <div className={`full-body-container ${useLlm ? 'llm-mode' : ''}`}>
      {/* Search bar (always shown) */}
      <div className="top-text">
        <div className="google-colors">
          <h1 id="google-4">4</h1>
          <h1 id="google-3">3</h1>
          <h1 id="google-0-1">0</h1>
          <h1 id="google-0-2">0</h1>
        </div>
        <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Search Guardian opinion articles"
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
          />
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
      {useLlm && <Chat onSearchQuery={handleSearch} />}
    </div>
  )
}

export default App
