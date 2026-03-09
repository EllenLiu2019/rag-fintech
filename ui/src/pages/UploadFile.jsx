import { useEffect, useRef, useState } from 'react'
import { useFileUpload } from '../components/useFileUpload'
import { FILE_UPLOAD_CONFIG, SAMPLE_FILE, TIMING, ROUTES } from '../constants/config'
import BrandCard from '../components/BrandCard'
import FileDropzone from '../components/FileDropzone'
import FileSummary from '../components/FileSummary'
import FileList from '../components/FileList'
import StatusMessage from '../components/StatusMessage'
import DocumentTypeSelector from '../components/DocumentTypeSelector'
import DocumentList from '../components/DocumentList'

/**
 * UploadFile 组件
 * 文件上传页面
 */
export const UploadFile = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null)
  const [isDragActive, setIsDragActive] = useState(false)
  const [docType, setDocType] = useState('policy')       // upload: 'policy' | 'claim'
  const [listDocType, setListDocType] = useState('all')   // list filter: 'all' | 'policy' | 'claim'
  const [viewMode, setViewMode] = useState('upload') // 'upload' | 'list'
  const [policyDocId, setPolicyDocId] = useState('')  // claim upload: linked policy doc id
  const [refreshTrigger, setRefreshTrigger] = useState(0)
  const fileInputRef = useRef(null)

  const {
    status,
    feedback,
    uploadFile,
    cancelUpload,
    resetStatus,
    setError,
    isUploading,
  } = useFileUpload({
    docType: docType,
    policyDocId: policyDocId,
    onSuccess: (uploadedFile, successMessage, responseData) => {
      const fileInfo = responseData
        ? {
            filename: responseData.file_name || uploadedFile.name,
            size: responseData.size || uploadedFile.size,
            task_id: responseData.task_id,
            doc_id: responseData.doc_id,
            doc_type: docType,
          }
        : {
            filename: uploadedFile.name,
            size: uploadedFile.size,
            task_id: null,
            doc_id: null,
            doc_type: docType,
          }

      setFile(null)
      resetFileInput()
      setRefreshTrigger(prev => prev + 1) // 触发列表刷新
      onUploadSuccess(fileInfo)
    },
    onError: (error) => {
      console.error('文件上传失败:', error)
    },
  })

  // 组件卸载时取消上传
  useEffect(() => {
    return () => {
      cancelUpload()
    }
  }, [cancelUpload])

  const resetFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleFileSelection = (selectedFile) => {
    if (!selectedFile) return

    if (selectedFile.size > FILE_UPLOAD_CONFIG.MAX_SIZE_BYTES) {
      setError(`文件大小不能超过 ${FILE_UPLOAD_CONFIG.MAX_SIZE_MB} MB`)
      setFile(null)
      resetFileInput()
      return
    }

    setFile(selectedFile)
    resetStatus()
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
    resetStatus()
    resetFileInput()
  }

  const handleFileUpload = async (e) => {
    e?.stopPropagation()
    if (!file || isUploading) return
    await uploadFile(file)
  }

  const handleSampleFileUpload = async (e) => {
    e?.stopPropagation()
    if (isUploading) return

    try {
      const response = await fetch(SAMPLE_FILE.path)
      const blob = await response.blob()

      const sampleFile = new File([blob], SAMPLE_FILE.name, { type: SAMPLE_FILE.type })

      if (sampleFile.size > FILE_UPLOAD_CONFIG.MAX_SIZE_BYTES) {
        setError(`示例文件大小超过 ${FILE_UPLOAD_CONFIG.MAX_SIZE_MB} MB，请选择其他文件`)
        return
      }

      setFile(sampleFile)
      await uploadFile(sampleFile, '正在加载示例文件...')
    } catch (error) {
      console.error('加载示例文件失败:', error)
      setError('无法加载示例文件，请手动选择文件')
    }
  }

  const handleSelectDocument = (document) => {
    const fileInfo = {
      filename: document.file_name,
      size: document.size,
      doc_id: document.doc_id,
      doc_type: document.doc_type,
    }
    onUploadSuccess(fileInfo)
  }

  return (
    <main className="home-container">
      <div className="home-content">
        <section className="left-panel">
          <div className="hero-frame">
            <BrandCard />
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
            <div className="header-actions">
              <div className="view-mode-toggle">
                <button
                  type="button"
                  className={`mode-button ${viewMode === 'upload' ? 'active' : ''}`}
                  onClick={() => setViewMode('upload')}
                >
                  上传
                </button>
                <button
                  type="button"
                  className={`mode-button ${viewMode === 'list' ? 'active' : ''}`}
                  onClick={() => setViewMode('list')}
                >
                  文档列表
                </button>
              </div>
            </div>
          </header>

          {viewMode === 'upload' && (
            <div className="doc-type-selector-wrapper">
              <DocumentTypeSelector
                value={docType}
                onChange={setDocType}
                options={['policy', 'claim']}
              />
            </div>
          )}

          {viewMode === 'list' && (
            <div className="doc-type-selector-wrapper">
              <DocumentTypeSelector
                value={listDocType}
                onChange={setListDocType}
                options={['all', 'policy', 'claim']}
              />
            </div>
          )}

          {viewMode === 'upload' && docType === 'claim' && (
            <div className="policy-doc-id-wrapper">
              <label className="policy-doc-id-label">关联保单文档ID</label>
              <input
                type="text"
                className="policy-doc-id-input"
                value={policyDocId}
                onChange={(e) => setPolicyDocId(e.target.value)}
                placeholder="请输入保单文档ID，例如 policy_0308125419_88c9d1"
              />
            </div>
          )}

          {viewMode === 'upload' && (
            <>
              <div className="upload-card">
                <FileDropzone
                  isDragActive={isDragActive}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onFileInputChange={handleFileInputChange}
                  fileInputRef={fileInputRef}
                />

                <FileSummary file={file} status={status} onClear={clearFile} />
              </div>

              <FileList
                file={file}
                isUploading={isUploading}
                onFileUpload={handleFileUpload}
                onSampleFileUpload={handleSampleFileUpload}
              />

              <StatusMessage feedback={feedback} status={status} />
            </>
          )}

          {viewMode === 'list' && (
            <DocumentList
              docTypeFilter={listDocType}
              onSelectDocument={handleSelectDocument}
              refreshTrigger={refreshTrigger}
            />
          )}
        </section>
      </div>
    </main>
  )
}

export default UploadFile

