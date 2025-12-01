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
  const [streamOutput, setStreamOutput] = useState(true) // 默认启用流式输出
  const [enableReasoning, setEnableReasoning] = useState(true)

  // 右侧栏状态
  const [retrievedChunks, setRetrievedChunks] = useState([])
  const [activeTab, setActiveTab] = useState('source') // 'source' or 'graph'

  // 重试机制状态
  const [retryCount, setRetryCount] = useState(0)
  const [isRetrying, setIsRetrying] = useState(false)

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

  // 判断错误是否可重试
  const isRetryableError = (error) => {
    // 网络错误
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      return true
    }
    // 超时错误
    if (error.name === 'AbortError') {
      return true
    }
    // HTTP 5xx 服务器错误
    if (error.message.includes('status: 5')) {
      return true
    }
    // 连接中断
    if (error.message.includes('connection') || error.message.includes('network')) {
      return true
    }
    // HTTP 429 Too Many Requests (速率限制)
    if (error.message.includes('status: 429')) {
      return true
    }
    return false
  }

  // 获取用户友好的错误消息
  const getUserFriendlyErrorMessage = (error, retryCount, maxRetries) => {
    if (error.message.includes('status: 429')) {
      return '请求过于频繁，请稍后再试'
    }
    if (error.message.includes('status: 5')) {
      return '服务器暂时不可用，正在重试...'
    }
    if (error.name === 'TypeError' || error.message.includes('network')) {
      return `网络连接不稳定，正在重试 (${retryCount}/${maxRetries})...`
    }
    return `处理请求时发生错误: ${error.message}`
  }

  // 计算重试延迟（指数退避策略）
  const getRetryDelay = (retryCount) => {
    // 基础延迟 1 秒，每次重试翻倍，最大 10 秒
    const baseDelay = 1000
    const maxDelay = 10000
    const delay = Math.min(baseDelay * Math.pow(2, retryCount), maxDelay)
    // 添加随机抖动，避免惊群效应
    const jitter = Math.random() * 0.3 * delay
    return delay + jitter
  }

  // 核心流式处理逻辑（提取为独立函数便于重试）
  const executeStreamRequest = async (requestBody, accumulators) => {
    const response = await fetch(`${apiBaseUrl}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let streamCompletedNormally = false // Track if stream ended with [DONE]

    while (true) {
      const { done, value } = await reader.read()
      
      if (done) {
        // Stream closed - check if it completed normally
        if (!streamCompletedNormally) {
          // Stream closed unexpectedly (e.g., due to error)
          // This allows retry mechanism to trigger
          throw new Error('Stream closed unexpectedly - connection may have failed')
        }
        break
      }

      // 解码数据块
      buffer += decoder.decode(value, { stream: true })
      
      // 按行分割
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // 保留最后一个不完整的行

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          
          if (data === '[DONE]') {
            // Stream completed normally
            streamCompletedNormally = true
            setMessages(prev => {
              const updated = [...prev]
              const lastMsg = updated[updated.length - 1]
              if (lastMsg && lastMsg.streaming) {
                lastMsg.streaming = false
                lastMsg.sources = accumulators.retrievedSources
              }
              return updated
            })
            continue
          }

          try {
            const event = JSON.parse(data)
            
            if (event.type === 'chunks') {
              // 接收到检索结果
              const formattedChunks = event.data.map((source, idx) => ({
                index: source.index || idx + 1,
                text: source.text || '',
                score: source.score || 0.0,
                source: source.doc_id || fileInfo?.filename || 'Unknown',
                page: source.metadata?.page || source.page || 'N/A',
                metadata: source.metadata || {},
                policy_number: source.policy_number,
                holder_name: source.holder_name,
                insured_name: source.insured_name,
                doc_id: source.doc_id,
                referenced: true
              }))
              setRetrievedChunks(formattedChunks)
              accumulators.retrievedSources = event.data
            } 
            else if (event.type === 'reasoning') {
              // 接收推理内容
              if (enableReasoning) {
                accumulators.accumulatedReasoning += event.data
                setMessages(prev => {
                  const updated = [...prev]
                  const lastMsg = updated[updated.length - 1]
                  if (lastMsg && lastMsg.streaming) {
                    lastMsg.reasoning = accumulators.accumulatedReasoning
                    lastMsg.reasoningComplete = event.done || false
                  }
                  return updated
                })
              }
            }
            else if (event.type === 'token') {
              // 接收内容 token
              accumulators.accumulatedAnswer += event.data
              setMessages(prev => {
                const updated = [...prev]
                const lastMsg = updated[updated.length - 1]
                if (lastMsg && lastMsg.streaming) {
                  lastMsg.content = accumulators.accumulatedAnswer
                }
                return updated
              })
            }
            else if (event.type === 'done') {
              // Stream completed normally (done event received)
              streamCompletedNormally = true
              setMessages(prev => {
                const updated = [...prev]
                const lastMsg = updated[updated.length - 1]
                if (lastMsg && lastMsg.streaming) {
                  lastMsg.content = event.data.answer || accumulators.accumulatedAnswer
                  lastMsg.streaming = false
                  lastMsg.sources = accumulators.retrievedSources
                  lastMsg.tokens = event.data.tokens || 0
                }
                return updated
              })
            }
            else if (event.type === 'error') {
              throw new Error(event.data.message || 'Stream error')
            }
          } catch (parseError) {
            console.error('Failed to parse SSE event:', parseError, data)
          }
        }
      }
    }
  }

  // 流式发送消息（带重试机制）
  const handleSendMessageStream = async () => {
    if (!inputValue.trim() || loading) return

    const MAX_RETRIES = 3
    let currentRetry = 0

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
    setRetryCount(0)
    setIsRetrying(false)

    // 创建占位消息用于流式更新
    const placeholderMessage = {
      role: 'assistant',
      content: '',
      reasoning: enableReasoning ? '' : null,
      reasoningComplete: false,
      sources: [],
      tokens: 0,
      timestamp: new Date().toISOString(),
      streaming: true
    }
    setMessages(prev => [...prev, placeholderMessage])

    // 构建过滤器
    const activeFilters = {}
    Object.entries(filters).forEach(([key, value]) => {
      if (selectedFilters[key] && value && value.trim()) {
        activeFilters[key] = value.trim()
      }
    })

    // 构建请求体
    const requestBody = {
      query: currentInput,
      kb_id: "default_kb",
      conversation_history: messages.filter(m => m.role === 'user' || m.role === 'assistant').map(m => ({
        role: m.role,
        content: m.content
      })),
      filters: activeFilters,
      stream: true,
      generation_config: {
        top_k: 5,
        temperature: temperature,
        max_tokens: maxTokens
      }
    }

    // 重试循环
    while (currentRetry <= MAX_RETRIES) {
      try {
        console.log(`Sending stream chat request (attempt ${currentRetry + 1}/${MAX_RETRIES + 1}):`, requestBody)

        // 用于累积完整内容的对象（每次重试都重置）
        const accumulators = {
          accumulatedAnswer: '',
          accumulatedReasoning: '',
          retrievedSources: []
        }

        // 执行流式请求
        await executeStreamRequest(requestBody, accumulators)

        // 成功完成，跳出重试循环
        console.log('Stream request completed successfully')
        setRetryCount(0)
        setIsRetrying(false)
        setError('') // Clear error message on success
        
        // 更新会话预览
        if (conversations.length > 0 && conversations[0].id === currentConversationId) {
          setConversations(prev => prev.map(conv => 
            conv.id === currentConversationId 
              ? { 
                  ...conv, 
                  preview: currentInput.substring(0, 50),
                  messageCount: messages.length + 2
                }
              : conv
          ))
        }
        
        // 成功后退出循环
        break

      } catch (err) {
        console.error(`Stream request failed (attempt ${currentRetry + 1}):`, err)
        
        // 判断是否应该重试
        const shouldRetry = isRetryableError(err) && currentRetry < MAX_RETRIES
        
        if (shouldRetry) {
          currentRetry++
          setRetryCount(currentRetry)
          setIsRetrying(true)
          
          // 显示重试提示
          const retryMessage = getUserFriendlyErrorMessage(err, currentRetry, MAX_RETRIES)
          setError(retryMessage)
          
          // 计算延迟
          const delay = getRetryDelay(currentRetry - 1)
          console.log(`Retrying in ${Math.round(delay)}ms...`)
          
          // 更新占位消息显示重试状态
          setMessages(prev => {
            const updated = [...prev]
            const lastMsg = updated[updated.length - 1]
            if (lastMsg && lastMsg.streaming) {
              lastMsg.content = `⚠️ ${retryMessage}`
              lastMsg.retrying = true
            }
            return updated
          })
          
          // 等待后重试
          await new Promise(resolve => setTimeout(resolve, delay))
          
          // 清空占位消息内容，准备下次尝试
          setMessages(prev => {
            const updated = [...prev]
            const lastMsg = updated[updated.length - 1]
            if (lastMsg && lastMsg.streaming) {
              lastMsg.content = ''
              lastMsg.reasoning = enableReasoning ? '' : null
              lastMsg.retrying = false
            }
            return updated
          })
          
        } else {
          // 不可重试或达到最大重试次数
          console.error('Max retries reached or non-retryable error:', err)
          
          const finalErrorMessage = currentRetry >= MAX_RETRIES
            ? `尝试了 ${MAX_RETRIES} 次后仍然失败，请检查网络连接或稍后再试`
            : `抱歉，处理您的问题时发生错误: ${err.message}`
          
          setError(finalErrorMessage)
          
          // 移除占位消息并添加错误消息
          setMessages(prev => {
            const updated = prev.filter(m => !m.streaming)
            return [...updated, {
              role: 'assistant',
              content: finalErrorMessage,
              reasoning: '请求处理失败',
              sources: [],
              timestamp: new Date().toISOString(),
              isError: true
            }]
          })
          
          break
        }
      }
    }

    // 清理状态
    setLoading(false)
    setRetryCount(0)
    setIsRetrying(false)
    // Note: Error message is cleared in success branch (line 313)
    // If we reach here after failure, error message should remain for user feedback
  }

  // 非流式发送消息
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
      if (streamOutput) {
        handleSendMessageStream()
      } else {
        handleSendMessage()
      }
    }
  }

  // 发送消息的统一入口
  const handleSend = () => {
    if (streamOutput) {
      handleSendMessageStream()
    } else {
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
                    checked={enableReasoning}
                    onChange={(e) => setEnableReasoning(e.target.checked)}
                    className="setting-checkbox"
                  />
                  Enable Reasoning
                </label>
              </div>

              <div className="setting-item checkbox-item">
                <label>
                  <input
                    type="checkbox"
                    checked={streamOutput}
                    onChange={(e) => setStreamOutput(e.target.checked)}
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

                  {message.role === 'assistant' && message.reasoning !== null && (
                    <details className="reasoning-block" open={message.streaming && !message.reasoningComplete}>
                      <summary>
                        🧠 Thinking Process 
                        {message.streaming && !message.reasoningComplete && <span className="thinking-indicator"> (thinking...)</span>}
                      </summary>
                      <div className="reasoning-content">
                        {message.reasoning}
                        {message.streaming && !message.reasoningComplete && (
                          <span className="cursor-blink">▋</span>
                        )}
                      </div>
                    </details>
                  )}

                  <div className="message-content">
                    {message.retrying ? (
                      <div className="retry-indicator">
                        <div className="retry-spinner"></div>
                        <span className="retry-text">{message.content}</span>
                      </div>
                    ) : message.streaming && !message.content ? (
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    ) : (
                      <>
                        <ReactMarkdown rehypePlugins={[rehypeRaw]}>
                          {message.content}
                        </ReactMarkdown>
                        {message.streaming && message.content && !message.retrying && (
                          <span className="cursor-blink">▋</span>
                        )}
                      </>
                    )}
                  </div>

                  {message.role === 'assistant' && (
                    <div className="message-footer">
                      <span className="sources-info">
                        {message.sources && message.sources.length > 0 && (
                          <>📊 {message.sources.length} sources • </>
                        )}
                        {message.streaming ? (
                          <span className="tokens-calculating">🔢 calculating tokens...</span>
                        ) : message.tokens && message.tokens > 0 ? (
                          <>🔢 {message.tokens.toLocaleString()} tokens</>
                        ) : null}
                      </span>
                    </div>
                  )}
                </div>
              ))
            )}

            {loading && !messages.some(m => m.streaming) && (
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
                onClick={handleSend}
                disabled={!inputValue.trim() || loading}
              >
                {streamOutput ? '🔄 Stream' : '📤 Send'}
              </button>
            </div>
            <div className="input-hint">
              Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line
            </div>
          </div>

          {error && (
            <div className={`error-message ${isRetrying ? 'retry-warning' : ''}`}>
              {isRetrying ? '⚠️' : '❌'} {error}
              {isRetrying && retryCount > 0 && (
                <span className="retry-count-badge">重试 {retryCount}/3</span>
              )}
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

