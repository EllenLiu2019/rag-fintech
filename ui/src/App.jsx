import { useEffect, useRef, useState } from 'react'
import './App.css'
import ragLogo from '/rag.svg'

const MAX_FILE_SIZE_MB = 3
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
const ACCEPTED_TYPES = '.txt,.pdf,.doc,.docx,.md'

// 示例文件信息
const SAMPLE_FILE = {
  name: 'policy_lite.pdf',
  size: 1.9 * 1024 * 1024, // 将在加载时获取
  type: 'application/pdf',
  path: '/data/policy_lite.pdf'
}

const formatBytes = (bytes) => {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  )
  const value = bytes / 1024 ** exponent
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`
}

function App() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('idle') // idle | ready | uploading | success | error
  const [feedback, setFeedback] = useState('')
  const [isDragActive, setIsDragActive] = useState(false)
  const fileInputRef = useRef(null)
  const uploadControllerRef = useRef(null)

  useEffect(() => {
    return () => {
      if (uploadControllerRef.current) {
        uploadControllerRef.current.abort()
      }
    }
  }, [])

  const resetFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleFileSelection = (selectedFile) => {
    if (!selectedFile) return

    if (selectedFile.size > MAX_FILE_SIZE_BYTES) {
      setStatus('error')
      setFeedback(`文件大小不能超过 ${MAX_FILE_SIZE_MB} MB`)
      setFile(null)
      resetFileInput()
      return
    }

    setFile(selectedFile)
    setStatus('ready')
    setFeedback('')
  }

  const handleFileInputChange = (event) => {
    const selected = event.target.files?.[0]
    handleFileSelection(selected)
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragActive(false)
    const dropped = event.dataTransfer?.files?.[0]
    handleFileSelection(dropped)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    setIsDragActive(true)
  }

  const handleDragLeave = (event) => {
    event.preventDefault()
    setIsDragActive(false)
  }

  const clearFile = () => {
    setFile(null)
    setStatus('idle')
    setFeedback('')
    resetFileInput()
  }

  // 通用的文件上传逻辑
  const uploadFile = async (fileToUpload) => {
    if (!fileToUpload || status === 'uploading') return

    setStatus('uploading')
    setFeedback('正在上传，请稍候...')

    const formData = new FormData()
    formData.append('file', fileToUpload)

    const controller = new AbortController()
    uploadControllerRef.current = controller

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      const payloadText = await response.text()

      if (!response.ok) {
        let errorMessage = '上传失败，请稍后重试'
        if (payloadText) {
          try {
            const parsed = JSON.parse(payloadText)
            if (parsed?.message) {
              errorMessage = parsed.message
            }
          } catch {
            errorMessage = payloadText
          }
        }
        throw new Error(errorMessage)
      }

      let successMessage = `文件 "${fileToUpload.name}" 上传成功`
      if (payloadText) {
        try {
          const parsed = JSON.parse(payloadText)
          if (parsed?.message) {
            successMessage = parsed.message
          }
        } catch {
          successMessage = payloadText || successMessage
        }
      }

      setStatus('success')
      setFeedback(successMessage)
      setFile(null)
      resetFileInput()
    } catch (error) {
      if (error.name === 'AbortError') {
        setStatus('idle')
        setFeedback('')
      } else {
        setStatus('error')
        setFeedback(error.message || '上传失败，请稍后重试')
      }
    } finally {
      uploadControllerRef.current = null
    }
  }

  // 上传用户选择的文件
  const handleFileUpload = async (e) => {
    e?.stopPropagation()
    if (!file) return
    await uploadFile(file)
  }

  // 上传示例文件
  const handleSampleFileUpload = async (e) => {
    e?.stopPropagation()
    if (status === 'uploading') return

    try {
      setStatus('uploading')
      setFeedback('正在加载示例文件...')

      const response = await fetch(SAMPLE_FILE.path)
      const blob = await response.blob()
      
      const sampleFile = new File([blob], SAMPLE_FILE.name, { type: SAMPLE_FILE.type })
      
      if (sampleFile.size > MAX_FILE_SIZE_BYTES) {
        setStatus('error')
        setFeedback(`示例文件大小超过 ${MAX_FILE_SIZE_MB} MB，请选择其他文件`)
        return
      }

      setFile(sampleFile)
      await uploadFile(sampleFile)
    } catch (error) {
      console.error('加载示例文件失败:', error)
      setStatus('error')
      setFeedback('无法加载示例文件，请手动选择文件')
    }
  }

  // 运行图标组件
  const RunIcon = () => (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M3 2L13 8L3 14V2Z" fill="currentColor"/>
    </svg>
  )

  const statusClass = {
    uploading: 'loading',
    success: 'success',
    error: 'error',
  }[status] || ''

  return (
    <div className="app">
      <main className="home-container">
        <div className="home-content">
          <section className="left-panel">
            <div className="hero-frame">
              <div className="brand-card">
                <img src={ragLogo} className="brand-logo" alt="RAG Logo" />
                <div>
                  <p className="brand-eyebrow">RAG OF INTELLIGENT INSIGHTS</p>
                  <h1>欢迎来到 RAG 智能洞察</h1>
                  <p className="brand-tagline">让结构化与非结构化知识实时对齐，驱动精准的金融分析与决策支持。</p>
                </div>
              </div>
              <div className="cover-frame"></div>
            </div>
          </section>
          <section className="control-panel">
            <header className="panel-header compact">
              <div>
                <p className="eyebrow">Upload · Inspect</p>
                <h2>让知识随时入库</h2>
                <p className="muted">
                  通过拖放或文件选择器上传文档，立即启动知识入库流程，为 RAG 提供实时支持。
                </p>
              </div>
            </header>

            <div className="upload-card">
              <label
                htmlFor="file-input"
                className={`upload-dropzone ${isDragActive ? 'drag-active' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
              >
                <div className="dropzone-visual">
                  <span className="arrow">↥</span>
                  <div>
                    <p>拖拽文件至此</p>
                    <small>支持 TXT / PDF / DOC / DOCX，单文件 ≤ {MAX_FILE_SIZE_MB} MB</small>
                  </div>
                </div>

                <button type="button" className="ghost-button">
                  浏览文件
                </button>
                <input
                  ref={fileInputRef}
                  id="file-input"
                  type="file"
                  accept={ACCEPTED_TYPES}
                  onChange={handleFileInputChange}
                />
              </label>

              <div className="file-summary">
                <div>
                  <p className="muted">当前文件</p>
                  <strong>{file ? file.name : '尚未选择'}</strong>
                </div>
                <div>
                  <p className="muted">文件大小</p>
                  <strong>{file ? formatBytes(file.size) : '--'}</strong>
                </div>
                <button
                  type="button"
                  className="ghost-button subtle"
                  onClick={clearFile}
                  disabled={!file && status !== 'success'}
                >
                  清除
                </button>
              </div>
            </div>

            <div className="file-list">
              {file ? (
                <div className="file-row">
                  <div>
                    <p className="file-name">{file.name}</p>
                    <small className="muted">
                      {formatBytes(file.size)} · {file.type || '未知类型'}
                    </small>
                  </div>
                  <button
                    type="button"
                    className="sample-upload-button"
                    onClick={handleFileUpload}
                    disabled={status === 'uploading'}
                  >
                    <RunIcon />
                  </button>
                </div>
              ) : (
                <>
                  <div className="muted" style={{ marginBottom: '0.75rem', fontSize: '0.85rem' }}>
                    快速开始 · Try with a sample file
                  </div>
                  <div className="file-row sample-file">
                    <div>
                      <p className="file-name">{SAMPLE_FILE.name}</p>
                      <small className="muted">
                        {formatBytes(SAMPLE_FILE.size)} · {SAMPLE_FILE.type || '未知类型'}
                      </small>
                    </div>
                    <button
                      type="button"
                      className="sample-upload-button"
                      onClick={handleSampleFileUpload}
                      disabled={status === 'uploading'}
                    >
                      <RunIcon />
                    </button>
                  </div>
                </>
              )}
            </div>

            {feedback && (
              <div className={`status-message ${statusClass}`} role="status" aria-live="polite">
                {feedback}
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
