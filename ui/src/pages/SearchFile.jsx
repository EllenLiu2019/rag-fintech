import { useState, useEffect } from 'react'
import { apiBaseUrl } from '../../config/config'
import BackIcon from '../components/icons/BackIcon'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import './SearchFile.css'

function SearchFile({ fileInfo, onBack }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [filters, setFilters] = useState({}) // 存储筛选值
  const [selectedFilters, setSelectedFilters] = useState({}) // 存储用户勾选的 filters 状态
  const [availableFilters, setAvailableFilters] = useState([])

  // 初始化过滤器：从 summary 中提取可用的 metadata
  useEffect(() => {
    if (fileInfo?.summary?.metadata) {
      const metadata = fileInfo.summary.metadata
      setAvailableFilters(Object.keys(metadata))
      
      // 预填 filters 和 checkbox 状态
      const initialFilters = {}
      const initialSelected = {}
      const validKeys = [] // 存储有效（非列表值）的 key

      Object.entries(metadata).forEach(([key, value]) => {
        // 排除列表类型的值
        if (Array.isArray(value)) {
          return
        }

        validKeys.push(key)
        initialSelected[key] = false // 默认不选

        // 只预填简单类型的值
        if (typeof value === 'string' || typeof value === 'number') {
           initialFilters[key] = String(value)
        } else if (value !== null) {
           // 尝试提取嵌套对象中的 name 字段
           if (value.name) {
             initialFilters[key] = value.name
           } else if (value.value) {
             initialFilters[key] = value.value
           }
        }
      })
      
      setAvailableFilters(validKeys)
      setFilters(initialFilters)
      setSelectedFilters(initialSelected)
    } else if (fileInfo?.summary?.metadata_keys) {
      setAvailableFilters(fileInfo.summary.metadata_keys)
      // 只有 keys 时，初始化 selected 状态
      const initialSelected = {}
      fileInfo.summary.metadata_keys.forEach(key => {
        initialSelected[key] = true
      })
      setSelectedFilters(initialSelected)
    }
  }, [fileInfo])

  const handleSearch = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError('')
    setResults([])

    try {
      // 构建过滤器：只包含被勾选且有值的项
      const activeFilters = {}
      Object.entries(filters).forEach(([key, value]) => {
        // 只有当复选框被选中，且值不为空时，才加入过滤条件
        if (selectedFilters[key] && value && value.trim()) {
          activeFilters[key] = value.trim()
        }
      })

      const response = await fetch(`${apiBaseUrl}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          kb_id: "default_kb", // TODO: 动态获取或配置
          top_k: 5,
          filters: activeFilters
        })
      })

      if (!response.ok) {
        throw new Error('搜索请求失败')
      }

      const data = await response.json()
      setResults(data.results || [])
    } catch (err) {
      console.error('Search failed:', err)
      setError(err.message || '搜索失败')
    } finally {
      setLoading(false)
    }
  }

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const handleCheckboxChange = (key) => {
    setSelectedFilters(prev => ({
      ...prev,
      [key]: !prev[key]
    }))
  }

  return (
    <div className="search-file-container">
      <header className="search-header">
        <button onClick={onBack} className="back-button" title="返回">
          <BackIcon size={18} color="var(--accent-cyan, #5be9ff)" />
        </button>
        <h2>Similarity Search</h2>
      </header>

      <div className="search-content">
        {/* 左侧控制面板 */}
        <div className="left-control-panel">
          {/* File Info 简略版 */}
          <div className="control-card file-summary-card">
            <h3>Target File</h3>
            <div className="file-name">{fileInfo?.filename || 'Unknown'}</div>
          </div>

          {/* Metadata Filters */}
          <div className="control-card filters-card">
            <h3>Metadata Filters</h3>
            <div className="filters-list">
              {availableFilters.length > 0 ? (
                availableFilters.map(key => (
                  <div key={key} className="filter-item">
                    <div className="filter-header">
                      <input
                        type="checkbox"
                        id={`filter-${key}`}
                        checked={!!selectedFilters[key]}
                        onChange={() => handleCheckboxChange(key)}
                        className="filter-checkbox"
                      />
                      <label htmlFor={`filter-${key}`} className="filter-label">{key}:</label>
                    </div>
                    <input 
                      type="text" 
                      className="filter-input"
                      placeholder="Any"
                      value={filters[key] || ''}
                      onChange={(e) => handleFilterChange(key, e.target.value)}
                      disabled={!selectedFilters[key]}
                    />
                  </div>
                ))
              ) : (
                <div className="no-filters">No metadata keys available</div>
              )}
            </div>
          </div>

          {/* Retrieval Strategy (预留) */}
          <div className="control-card strategy-card">
            <h3>Retrieval Strategy</h3>
            <div className="strategy-option">
              <label>
                <input type="radio" name="strategy" value="vector" defaultChecked />
                Vector Search
              </label>
            </div>
            <div className="strategy-option disabled">
              <label>
                <input type="radio" name="strategy" value="hybrid" disabled />
                Hybrid Search (Coming Soon)
              </label>
            </div>
          </div>
        </div>

        {/* 右侧交互区 */}
        <div className="right-interaction-area">
          {/* 顶部搜索栏 */}
          <div className="search-bar-container">
            <input 
              type="text" 
              className="search-input" 
              placeholder="Enter your question about the document..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            />
            <button 
              className="search-button" 
              onClick={handleSearch}
              disabled={loading}
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* 结果展示区 */}
          <div className="results-area">
            {error && <div className="error-message">❌ {error}</div>}
            
            {results.length > 0 ? (
              <div className="results-list">
                <div className="results-count">Found {results.length} results</div>
                {results.map((result, index) => (
                  <div key={index} className="result-card">
                    <div className="result-header">
                      <span className="result-score">Score: {result.score ? result.score.toFixed(4) : 'N/A'}</span>
                      {result.pol_num && <span className="result-tag">Policy: {result.pol_num}</span>}
                    </div>
                    <div className="result-text">
                      <ReactMarkdown rehypePlugins={[rehypeRaw]}>
                        {result.text}
                      </ReactMarkdown>
                    </div>
                    {/* Metadata 详情 (可折叠) */}
                    <div className="result-metadata">
                      <details>
                        <summary>View Metadata</summary>
                        <pre>{JSON.stringify(result.metadata, null, 2)}</pre>
                      </details>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              !loading && <div className="empty-state">Enter a query to start searching</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default SearchFile

