import { useEffect, useState, useRef } from 'react'
import { apiBaseUrl } from '../../config/config'
import './ParseFile.css'

function ParseFile({ fileInfo, onBack }) {
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
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

        // 调用后端 API 获取文件内容
        const response = await fetch(`${apiBaseUrl}/api/file-content?filename=${encodeURIComponent(fileInfo.filename)}`)
        
        if (!response.ok) {
          throw new Error('获取文件内容失败')
        }

        const data = await response.json()
        setContent(data.content || data.text || '文件内容为空')
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
        <button onClick={onBack} className="back-button">
          ← 返回上传
        </button>
        <h2>文件解析</h2>
      </header>

      <div className="parse-content">
        {/* 第一部分：功能面板 */}
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
          <button className="function-button">
            Similarity Search
          </button>
        </div>

        {/* 第二部分：文件信息卡片 */}
        <div className="info-panel">
          <div className="file-info-card">
            <h2>文件信息</h2>
            <div className="file-info-list">
              <div className="file-info-item">
                <span className="label">文件名:</span>
                <span className="value">{fileInfo?.filename || '未知'}</span>
              </div>
              <div className="file-info-item">
                <span className="label">文件类型:</span>
                <span className="value">{fileInfo?.content_type || '未知'}</span>
              </div>
              <div className="file-info-item">
                <span className="label">文件大小:</span>
                <span className="value">{fileInfo?.size ? `${(fileInfo.size / 1024).toFixed(2)} KB` : '未知'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* 第三部分：文件内容展示区 */}
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
            <div className="content-display">
              <pre>{content}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ParseFile

