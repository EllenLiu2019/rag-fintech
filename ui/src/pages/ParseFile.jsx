import { useEffect, useState, useRef, useCallback } from 'react'
import { apiBaseUrl } from '../../config/config'
import BackIcon from '../components/icons/BackIcon'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import './ParseFile.css'

function ParseFile({ fileInfo, onBack, onSearch, onChat, onSubmitClaim }) {
  const [content, setContent] = useState('')
  const [parsedFileInfo, setParsedFileInfo] = useState(null) 
  const [loadingParsed, setLoadingParsed] = useState(true) // 只控制解析内容的加载状态
  const [error, setError] = useState('')
  const [pdfLoadError, setPdfLoadError] = useState(false)
  const [jobStatus, setJobStatus] = useState(null) // 任务状态
  const [progressMessage, setProgressMessage] = useState('') // 进度消息
  const pollingIntervalRef = useRef(null) // 轮询定时器
  const hasFetchedRef = useRef(false) // 防止重复请求
  const filenameRef = useRef(null) // 跟踪已请求的文件名

  // 获取文件内容
  const fetchFileContent = useCallback(async () => {
    if (!fileInfo?.filename) {
      setError('未找到文件信息')
      setLoadingParsed(false)
      return
    }

    // 防止重复调用：如果已经获取过相同文件的内容，直接返回
    if (hasFetchedRef.current && filenameRef.current === fileInfo.filename) {
      return
    }

    try {
      setLoadingParsed(true)
      setError('')
      setProgressMessage('正在加载文件内容...')

      // 标记为已获取，防止重复调用
      hasFetchedRef.current = true
      filenameRef.current = fileInfo.filename

      // 检查 doc_id 是否存在
      if (!fileInfo.doc_id) {
        throw new Error('文档ID缺失，无法获取解析后的文件内容。请重新上传文件。')
      }

      const url = `${apiBaseUrl}/api/parsed-file?filename=${encodeURIComponent(fileInfo.filename)}&doc_id=${encodeURIComponent(fileInfo.doc_id)}&doc_type=${fileInfo.doc_type}`
      const response = await fetch(url)
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.message || '获取文件内容失败')
      }

      const data = await response.json()
      
      // 处理解析结果的元信息
      if (data.business_data || data.confidence || data.document_id) {
        setParsedFileInfo({
          business_data: data.business_data || {},
          confidence: data.confidence,
          document_id: data.document_id,
        })
      }
      
      // 处理页面内容
      if (data.pages && Array.isArray(data.pages)) {
        // 如果是 Document 对象数组，提取文本内容
        const textContent = data.pages
          .map(doc => (typeof doc === 'string' ? doc : doc.text || doc.content || ''))
          .join('\n\n')
        setContent(textContent || '文件内容为空')
      } else if (data.content || data.text) {
        setContent(data.content || data.text)
      } else {
        setContent('文件内容为空')
      }
      
      // 成功获取内容后，清除任务状态，让内容正常显示
      setJobStatus(null)
      setProgressMessage('')
    } catch (err) {
      console.error('获取文件内容失败:', err)
      setError(err.message || '获取文件内容失败')
      hasFetchedRef.current = false // 失败时重置，允许重试
      filenameRef.current = null
    } finally {
      setLoadingParsed(false)
    }
  }, [fileInfo?.filename, fileInfo?.doc_id, fileInfo?.doc_type])

  // 轮询任务状态
  useEffect(() => {
    if (fileInfo?.task_id) {
      // 如果文件名改变，重置状态
      if (filenameRef.current !== fileInfo.filename) {
        hasFetchedRef.current = false
        filenameRef.current = fileInfo.filename
      }
      
      if (!hasFetchedRef.current) {
        const pollJobStatus = async () => {
          try {
            const response = await fetch(`${apiBaseUrl}/api/process/${fileInfo.task_id}`)
            if (!response.ok) {
              throw new Error('获取任务状态失败')
            }

            const jobData = await response.json()
            setJobStatus(jobData)

            // 更新进度消息
            if (jobData.step && jobData.step > 0) {
              setProgressMessage(`processing... step ${jobData.step}/7, ${jobData.message}`)
            }

            // 检查任务状态
            if (jobData.status === 'finished') {
              // 任务完成，停止轮询并获取文件内容
              if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current)
                pollingIntervalRef.current = null
              }
              // 只有在还没有获取过内容时才调用 fetchFileContent
              // fetchFileContent 内部已经有防重复调用的逻辑
              if (!hasFetchedRef.current || filenameRef.current !== fileInfo.filename) {
                fetchFileContent()
              }
            } else if (jobData.status === 'failed') {
              // 任务失败
              if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current)
                pollingIntervalRef.current = null
              }
              setError(jobData.error || '文件处理失败')
              setLoadingParsed(false)
            } else if (jobData.status === 'not_found') {
              // 但文件可能已经解析过，尝试直接获取解析后的文件内容
              if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current)
                pollingIntervalRef.current = null
              }
              // 尝试获取解析后的文件内容
              // 如果文件已经解析过，即使任务信息丢失也能正常显示
              if (!hasFetchedRef.current || filenameRef.current !== fileInfo.filename) {
                fetchFileContent().catch((err) => {
                  // 如果获取失败，再显示错误
                  console.error('获取解析文件失败:', err)
                  setError('任务不存在，且无法获取解析后的文件内容')
                  setLoadingParsed(false)
                })
              }
            }
            // queued, started 状态继续轮询
          } catch (err) {
            console.error('轮询任务状态失败:', err)
            // 轮询失败不中断，继续尝试
          }
        }

        // 立即执行一次
        pollJobStatus()

        // 每 2 秒轮询一次
        pollingIntervalRef.current = setInterval(pollJobStatus, 2000)

          // 清理函数
          return () => {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current)
              pollingIntervalRef.current = null
            }
          }
      }
    } else if (fileInfo?.filename && !fileInfo?.task_id) {
      // 没有 task_id，直接获取文件内容（兼容旧逻辑）
      // console.log('No task_id, fetching file content directly') // Debug log
      if (!hasFetchedRef.current || filenameRef.current !== fileInfo.filename) {
        hasFetchedRef.current = true
        filenameRef.current = fileInfo.filename
        fetchFileContent()
      }
    }
  }, [fileInfo?.task_id, fileInfo?.filename, fetchFileContent])

  return (
    <div className="file-parse-container">
      <header className="parse-header">
        <button onClick={onBack} className="back-button" title="返回上传">
          <BackIcon size={18} color="var(--accent-cyan, #5be9ff)" />
        </button>
      </header>

      <div className="parse-content">
        {/* 左侧面板：包含文件信息和功能面板 */}
        <div className="left-panel">
          {/* 文件信息卡片 */}
          <div className="info-panel">
            <div className="file-info-card">
              <h2>File Info</h2>
              <div className="file-info-list">
                <div className="file-info-item">
                  <span className="label">file name:</span>
                  <span className="value">{fileInfo?.filename || '未知'}</span>
                </div>
                <div className="file-info-item">
                  <span className="label">size:</span>
                  <span className="value">{fileInfo?.size ? `${(fileInfo.size / 1024).toFixed(2)} KB` : '未知'}</span>
                </div>
              </div>
            </div>

            {/* 新增：解析结果摘要卡片 */}
            {parsedFileInfo && (
              <div className="parsing-result-card">
                <h2>Parsing Result</h2>
                <div className="file-info-list">
                   <div className="file-info-item">
                    <span className="label">Confidence:</span>
                    <span className="value" style={{ color: '#00ff9d' }}>
                      {parsedFileInfo.confidence ? `${(parsedFileInfo.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="file-info-item">
                    <span className="label">Business Data Keys:</span>
                    <div className="tags-container">
                      {parsedFileInfo.business_data && Object.keys(parsedFileInfo.business_data).length > 0 ? (
                        Object.keys(parsedFileInfo.business_data).map((key, index) => (
                          <span key={index} className="meta-tag">{key}</span>
                        ))
                      ) : (
                        <span className="value">-</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
          {/* 功能面板 */}
          <div className="function-panel">
            <button 
              className="function-button" 
              onClick={() => onSearch({ 
                ...fileInfo, 
                ...parsedFileInfo  // ← 展开 parsedFileInfo，让 business_data 在顶层
              })}
            >
              Similarity Search
            </button>
            <button 
              className="function-button" 
              onClick={() => onChat && onChat({ 
                ...fileInfo, 
                ...parsedFileInfo  // ← 同样处理
              })}
            >
              Intelligent Q&A
            </button>
            <button 
              className="function-button" 
              onClick={() => {
                const combinedInfo = { 
                  ...fileInfo, 
                  ...parsedFileInfo 
                }
                console.log('Combined info:', combinedInfo)
                onSubmitClaim(combinedInfo)
              }}
            >
              Submit Claim
            </button>
          </div>
        </div>

        {/* 文件内容展示区：分为左右两部分 */}
        <div className="content-panel">
          {/* 左侧：PDF 原始内容 - 只要文件信息存在就显示，不等待任务完成 */}
          <div className="pdf-original-panel">
            <div className="panel-header">
              <h3>original file</h3>
            </div>
            <div className="pdf-display">
              {fileInfo?.filename?.toLowerCase().endsWith('.pdf') ? (
                pdfLoadError ? (
                  <div className="pdf-placeholder">
                    <p>⚠️ PDF 加载失败</p>
                    <p style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                      请检查文件是否存在或网络连接是否正常
                    </p>
                  </div>
                ) : (
                  <iframe
                    src={`${apiBaseUrl}/api/original-file?filename=${encodeURIComponent(fileInfo.filename)}${fileInfo.doc_id ? `&doc_id=${encodeURIComponent(fileInfo.doc_id)}` : ''}&doc_type=${fileInfo.doc_type || (fileInfo.doc_id?.startsWith('claim_') ? 'claim' : 'policy')}`}
                    title="PDF Viewer"
                    className="pdf-iframe"
                    allow="fullscreen"
                    onError={() => {
                      setPdfLoadError(true)
                      console.error('PDF 加载失败')
                    }}
                    onLoad={() => {
                      setPdfLoadError(false)
                    }}
                  />
                )
              ) : (
                <div className="pdf-placeholder">
                  <p>非 PDF 文件，无法显示原始内容</p>
                </div>
              )}
            </div>
          </div>

          {/* 右侧：解析后的内容 - 根据任务状态显示加载或内容 */}
          <div className="parsed-content-panel">
            <div className="panel-header">
              <h3>parsed file</h3>
            </div>
            <div className="content-display">
              {error ? (
                <div className="error-state">
                  <p>❌ {error}</p>
                </div>
              ) : loadingParsed || (jobStatus && jobStatus.status !== 'finished' && jobStatus.status !== 'not_found' && !content) ? (
                <div className="loading-state">
                  <div className="spinner"></div>
                  <p>{progressMessage || (jobStatus?.status === 'queued' ? '任务已提交，等待处理...' : 
                      jobStatus?.status === 'started' ? '正在处理文件...' : '正在解析文件内容...')}</p>
                  {jobStatus && (
                    <div style={{ marginTop: '1rem', fontSize: '0.9rem', color: '#888' }}>
                      <div>状态: {jobStatus.status}</div>
                      {jobStatus.step > 0 && <div>step: {jobStatus.step}/7, {jobStatus.message}</div>}
                    </div>
                  )}
                </div>
              ) : (
                <ReactMarkdown
                  rehypePlugins={[rehypeRaw]}
                  className="markdown-content"
                >
                  {content || '文件内容为空'}
                </ReactMarkdown>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ParseFile

