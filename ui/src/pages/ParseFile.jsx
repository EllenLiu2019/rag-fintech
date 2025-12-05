import { useEffect, useState, useRef } from 'react'
import { apiBaseUrl } from '../../config/config'
import BackIcon from '../components/icons/BackIcon'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import './ParseFile.css'

function ParseFile({ fileInfo, onBack, onSearch, onChat }) {
  const [content, setContent] = useState('')
  const [summary, setSummary] = useState(null) // 新增 summary 状态
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [pdfLoadError, setPdfLoadError] = useState(false)
  const hasFetchedRef = useRef(false) // 防止重复请求
  const filenameRef = useRef(null) // 跟踪已请求的文件名

  useEffect(() => {
    // 获取文件内容
    const fetchFileContent = async () => {
      if (!fileInfo?.filename) {
        setError('未找到文件信息')
        setLoading(false)
        return
      }

      // 如果已经请求过相同的文件名，跳过
      if (hasFetchedRef.current && filenameRef.current === fileInfo.filename) {
        return
      }

      try {
        setLoading(true)
        setError('')
        hasFetchedRef.current = true
        filenameRef.current = fileInfo.filename

        // 调用后端 API 获取文件解析后的内容
        const response = await fetch(`${apiBaseUrl}/api/file-parsed?filename=${encodeURIComponent(fileInfo.filename)}`)
        
        if (!response.ok) {
          throw new Error('获取文件内容失败')
        }

        const data = await response.json()
        // 处理返回的数据：可能是 pages content/text 字符串
        if (data.summary) {
          setSummary(data.summary)
        }
        
        if (data.pages && Array.isArray(data.pages)) {
          // 如果是 Document 对象数组，提取文本内容
          const textContent = data.pages
            .map(doc => (typeof doc === 'string' ? doc : doc.text || doc.content || ''))
            .join('\n\n')
          setContent(textContent || '文件内容为空')
        } else {
          setContent(data.content || data.text || data.pages || '文件内容为空')
        }
      } catch (err) {
        console.error('获取文件内容失败:', err)
        setError(err.message || '获取文件内容失败')
        hasFetchedRef.current = false // 失败时重置，允许重试
      } finally {
        setLoading(false)
      }
    }

    fetchFileContent()
  }, [fileInfo?.filename]) // 只依赖 filename，而不是整个 fileInfo 对象

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
            {summary && (
              <div className="parsing-result-card">
                <h2>Parsing Result</h2>
                <div className="file-info-list">
                   <div className="file-info-item">
                    <span className="label">Confidence:</span>
                    <span className="value" style={{ color: '#00ff9d' }}>
                      {summary.overall_confidence ? `${(summary.overall_confidence * 100).toFixed(1)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="file-info-item">
                    <span className="label">Business Data Keys:</span>
                    <div className="tags-container">
                      {summary.business_data && Object.keys(summary.business_data).length > 0 ? (
                        Object.keys(summary.business_data).map((key, index) => (
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
            <button className="function-button">
              Chunk File
            </button>
            <button className="function-button">
              Embedding File
            </button>
            <button className="function-button">
              Indexing with Vector Database
            </button>
            <button className="function-button" onClick={() => onSearch({ ...fileInfo, summary })}>
              Similarity Search
            </button>
            <button className="function-button" onClick={() => onChat && onChat({ ...fileInfo, summary })}>
              Intelligent Q&A
            </button>
          </div>
        </div>

        {/* 文件内容展示区：分为左右两部分 */}
        <div className="content-panel">
          {loading ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>正在解析文件内容...</p>
            </div>
          ) : error ? (
            <div className="error-state">
              <p>❌ {error}</p>
            </div>
          ) : (
            <>
              {/* 左侧：PDF 原始内容 */}
              <div className="pdf-original-panel">
                <div className="panel-header">
                  <h3>original file</h3>
                </div>
                <div className="pdf-display">
                  {fileInfo?.content_type === 'application/pdf' ? (
                    pdfLoadError ? (
                      <div className="pdf-placeholder">
                        <p>⚠️ PDF 加载失败</p>
                        <p style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                          请检查文件是否存在或网络连接是否正常
                        </p>
                      </div>
                    ) : (
                      <iframe
                        src={`${apiBaseUrl}/api/file-original?filename=${encodeURIComponent(fileInfo.filename)}`}
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

              {/* 右侧：解析后的内容 */}
              <div className="parsed-content-panel">
                <div className="panel-header">
                  <h3>parsed file</h3>
                </div>
                <div className="content-display">
                  <ReactMarkdown
                    rehypePlugins={[rehypeRaw]}
                    className="markdown-content"
                  >
                    {content}
                  </ReactMarkdown>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default ParseFile

