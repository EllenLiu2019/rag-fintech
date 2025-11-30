import { useState, useEffect } from 'react'
import { apiBaseUrl } from '../../config/config'
import BackIcon from '../components/icons/BackIcon'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import './ChatQA.css'

function ChatQA({ fileInfo, onBack }) {
  // 消息列表状态
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // 左侧栏状态
  const [conversations, setConversations] = useState([
    { id: 'conv_1', title: 'New Conversation', preview: '', messageCount: 0 }
  ])
  const [currentConversationId, setCurrentConversationId] = useState('conv_1')

  // Filters状态
  const [filters, setFilters] = useState({})
  const [selectedFilters, setSelectedFilters] = useState({})
  const [availableFilters, setAvailableFilters] = useState([])

  // Model Settings状态
  const [model, setModel] = useState('DeepSeek-R1')
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(2000)

  // 右侧栏状态
  const [retrievedChunks, setRetrievedChunks] = useState([])
  const [activeTab, setActiveTab] = useState('source') // 'source' or 'graph'

  // 初始化Filters（从fileInfo.summary中提取）
  useEffect(() => {
    if (fileInfo?.summary?.metadata_keys) {
      setAvailableFilters(fileInfo.summary.metadata_keys)
      const initialSelected = {}
      fileInfo.summary.metadata_keys.forEach(key => {
        initialSelected[key] = false
      })
      setSelectedFilters(initialSelected)
    }
  }, [fileInfo])

  // 发送消息
  const handleSendMessage = async () => {
    if (!inputValue.trim() || loading) return

    const userMessage = {
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toISOString()
    }

    // 添加到消息列表
    setMessages(prev => [...prev, userMessage])
    const currentInput = inputValue.trim()
    setInputValue('')
    setLoading(true)
    setError('')
    setRetrievedChunks([]) // 清空之前的检索结果

    try {
      // 构建过滤器：只包含被勾选且有值的项
      const activeFilters = {}
      Object.entries(filters).forEach(([key, value]) => {
        if (selectedFilters[key] && value && value.trim()) {
          activeFilters[key] = value.trim()
        }
      })

      // 构建请求体
      const requestBody = {
        query: currentInput,
        kb_id: "default_kb", // TODO: 可以从 fileInfo 或其他地方获取
        conversation_history: messages.filter(m => m.role === 'user' || m.role === 'assistant').map(m => ({
          role: m.role,
          content: m.content
        })),
        filters: activeFilters,
        stream: false, // 暂时使用非流式，后续可以实现流式
        generation_config: {
          top_k: 5, // 可以从 UI 配置中获取
          temperature: temperature,
          max_tokens: maxTokens
        }
      }

      console.log('Sending chat request:', requestBody)

      // 调用后端 API
      const response = await fetch(`${apiBaseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.message || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // 格式化检索结果并更新右侧栏
      if (data.sources && data.sources.length > 0) {
        const formattedChunks = data.sources.map((source, idx) => {
          // 优先使用 doc_id，如果没有则使用文件名，最后使用 Unknown
          const sourceName = source.doc_id || fileInfo?.filename || 'Unknown'
          
          return {
            index: source.index || idx + 1,
            text: source.text || '',
            score: source.score || 0.0,
            source: sourceName,
            page: source.metadata?.page || source.page || 'N/A',
            metadata: source.metadata || {},
            policy_number: source.policy_number,
            holder_name: source.holder_name,
            insured_name: source.insured_name,
            doc_id: source.doc_id,
            referenced: true // 标记为被引用
          }
        })
        setRetrievedChunks(formattedChunks)
      } else {
        // 如果没有检索结果，清空右侧栏
        setRetrievedChunks([])
      }

      // 构建 AI 消息
      const aiMessage = {
        role: 'assistant',
        content: data.answer || 'No answer provided',
        reasoning: data.reasoning || null,
        sources: data.sources || [],
        tokens: data.tokens || 0,
        timestamp: new Date().toISOString()
      }

      setMessages(prev => [...prev, aiMessage])

      // 更新会话预览
      if (conversations.length > 0 && conversations[0].id === currentConversationId) {
        setConversations(prev => prev.map(conv => 
          conv.id === currentConversationId 
            ? { 
                ...conv, 
                preview: currentInput.substring(0, 50),
                messageCount: messages.length + 2 // user + assistant
              }
            : conv
        ))
      }

    } catch (err) {
      console.error('发送消息失败:', err)
      setError(err.message || '发送消息失败')
      
      // 添加错误消息到对话中
      const errorMessage = {
        role: 'assistant',
        content: `抱歉，处理您的问题时发生错误: ${err.message}`,
        reasoning: '请求处理失败',
        sources: [],
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  // 处理键盘事件
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // 处理Filter变化
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
    <div className="chat-qa-container">
      {/* Header */}
      <header className="chat-header">
        <button onClick={onBack} className="back-button" title="返回">
          <BackIcon size={18} color="var(--accent-cyan, #5be9ff)" />
        </button>
        <h2>Intelligent Q&A</h2>
      </header>

      {/* 三栏布局 */}
      <div className="chat-layout">
        {/* 左侧栏 (20%) */}
        <div className="left-sidebar">
          {/* Chat History */}
          <div className="sidebar-section chat-history-section">
            <h3>💬 Chat History</h3>
            <div className="conversation-list">
              {conversations.map(conv => (
                <div
                  key={conv.id}
                  className={`conversation-item ${currentConversationId === conv.id ? 'active' : ''}`}
                  onClick={() => setCurrentConversationId(conv.id)}
                >
                  <div className="conv-icon">📝</div>
                  <div className="conv-info">
                    <div className="conv-title">{conv.title}</div>
                    <div className="conv-preview">{conv.preview || 'No messages yet'}</div>
                    <div className="conv-meta">{conv.messageCount} messages</div>
                  </div>
                </div>
              ))}
              <button className="new-chat-button">
                + New Chat
              </button>
            </div>
          </div>

          {/* Filters */}
          <div className="sidebar-section filters-section">
            <h3>🔍 Filters</h3>
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

          {/* Model Settings */}
          <div className="sidebar-section model-settings-section">
            <h3>⚙️ Model Settings</h3>
            <div className="settings-content">
              <div className="setting-item">
                <label>Model:</label>
                <select
                  className="model-select"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  <option value="DeepSeek-R1">DeepSeek-R1</option>
                  <option value="GPT-4o">GPT-4o</option>
                  <option value="Claude-3.5">Claude-3.5</option>
                </select>
              </div>

              <div className="setting-item">
                <label>Temperature: {temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="temperature-slider"
                />
              </div>

              <div className="setting-item">
                <label>Max Tokens: {maxTokens}</label>
                <input
                  type="number"
                  min="1000"
                  max="4000"
                  step="500"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="tokens-input"
                />
              </div>

              <div className="setting-item checkbox-item">
                <label>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="setting-checkbox"
                  />
                  Enable Reasoning
                </label>
              </div>

              <div className="setting-item checkbox-item">
                <label>
                  <input
                    type="checkbox"
                    defaultChecked
                    className="setting-checkbox"
                  />
                  Stream Output
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* 中间栏 (50%) - Chat Interface */}
        <div className="center-panel">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="empty-chat-state">
                <div className="empty-icon">💬</div>
                <h3>Start a conversation</h3>
                <p>Ask me anything about your documents...</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-header">
                    <span className="message-role">
                      {message.role === 'user' ? '👤 You' : '🤖 AI Assistant'}
                    </span>
                    <span className="message-time">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </span>
                  </div>

                  {message.role === 'assistant' && message.reasoning && (
                    <details className="reasoning-block">
                      <summary>🧠 Thinking Process (click to expand)</summary>
                      <div className="reasoning-content">
                        {message.reasoning}
                      </div>
                    </details>
                  )}

                  <div className="message-content">
                    <ReactMarkdown rehypePlugins={[rehypeRaw]}>
                      {message.content}
                    </ReactMarkdown>
                  </div>

                  {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                    <div className="message-footer">
                      <span className="sources-info">
                        📊 {message.sources.length} sources • ⏱️ 1.2s • 🎯 Confidence: 95%
                      </span>
                    </div>
                  )}
                </div>
              ))
            )}

            {loading && (
              <div className="message assistant loading">
                <div className="message-header">
                  <span className="message-role">🤖 AI Assistant</span>
                </div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Box */}
          <div className="chat-input-container">
            <div className="input-wrapper">
              <textarea
                className="chat-input"
                placeholder="Ask me anything about your documents..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={loading}
              />
              <button
                className="send-button"
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || loading}
              >
                Send
              </button>
            </div>
            <div className="input-hint">
              Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line
            </div>
          </div>

          {error && (
            <div className="error-message">
              ❌ {error}
            </div>
          )}
        </div>

        {/* 右侧栏 (30%) - Evidence Board */}
        <div className="right-sidebar">
          <div className="evidence-tabs">
            <button
              className={`tab-button ${activeTab === 'source' ? 'active' : ''}`}
              onClick={() => setActiveTab('source')}
            >
              Source Context
            </button>
            <button
              className={`tab-button ${activeTab === 'graph' ? 'active' : ''}`}
              onClick={() => setActiveTab('graph')}
            >
              Knowledge Graph
            </button>
          </div>

          <div className="evidence-content">
            {activeTab === 'source' ? (
              <div className="source-context-tab">
                <div className="chunks-header">
                  <h4>📊 Retrieved Chunks</h4>
                  <select className="sort-select">
                    <option>Relevance</option>
                    <option>Score</option>
                    <option>Date</option>
                  </select>
                </div>

                {retrievedChunks.length > 0 ? (
                  <div className="chunks-list">
                    {retrievedChunks.map((chunk, index) => (
                      <div key={index} className="chunk-card">
                        <div className="chunk-header">
                          <span className="chunk-number">📄 Chunk #{index + 1}</span>
                          <span className="chunk-score" style={{
                            backgroundColor: chunk.score >= 0.9 ? 'rgba(0, 255, 157, 0.2)' :
                                           chunk.score >= 0.8 ? 'rgba(91, 233, 255, 0.2)' :
                                           chunk.score >= 0.7 ? 'rgba(255, 215, 0, 0.2)' :
                                           'rgba(255, 111, 111, 0.2)',
                            color: chunk.score >= 0.9 ? '#00ff9d' :
                                   chunk.score >= 0.8 ? '#5be9ff' :
                                   chunk.score >= 0.7 ? '#ffd700' : '#ff6f6f'
                          }}>
                            Score: {chunk.score?.toFixed(2) || 'N/A'}
                          </span>
                        </div>
                        <div className="chunk-source">
                          Source: {chunk.source || 'Unknown'} (p.{chunk.page || 'N/A'})
                        </div>
                        <div className="chunk-text">
                          <ReactMarkdown rehypePlugins={[rehypeRaw]}>
                            {chunk.text || 'No content'}
                          </ReactMarkdown>
                        </div>
                        {chunk.referenced && (
                          <div className="chunk-reference">
                            [{index + 1}] Referenced in answer
                          </div>
                        )}
                        {chunk.metadata && (
                          <details className="chunk-metadata">
                            <summary>Metadata</summary>
                            <pre>{JSON.stringify(chunk.metadata, null, 2)}</pre>
                          </details>
                        )}
                        <button className="view-document-button">
                          View Full Document
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-chunks">
                    <p>No chunks retrieved yet.</p>
                    <p className="hint">Send a message to see retrieved context here.</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="knowledge-graph-tab">
                <div className="coming-soon">
                  <div className="coming-soon-icon">🕸️</div>
                  <h4>Knowledge Graph</h4>
                  <p>Coming Soon</p>
                  <div className="coming-soon-features">
                    <p>Future features:</p>
                    <ul>
                      <li>Entity relationship graph</li>
                      <li>Policy connection network</li>
                      <li>Interactive visualization</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatQA

