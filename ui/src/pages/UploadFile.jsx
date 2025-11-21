import { useEffect, useRef, useState } from 'react'
import { useFileUpload } from '../components/useFileUpload'
import { FILE_UPLOAD_CONFIG, SAMPLE_FILE, TIMING, ROUTES } from '../constants/config'
import BrandCard from '../components/BrandCard'
import FileDropzone from '../components/FileDropzone'
import FileSummary from '../components/FileSummary'
import FileList from '../components/FileList'
import StatusMessage from '../components/StatusMessage'

/**
 * UploadFile 组件
 * 文件上传页面
 */
export const UploadFile = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null)
  const [isDragActive, setIsDragActive] = useState(false)
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
    onSuccess: (uploadedFile, successMessage, responseData) => {
      // 上传成功后保存文件信息并跳转
      const fileInfo = responseData
        ? {
            filename: responseData.filename || uploadedFile.name,
            size: responseData.size || uploadedFile.size,
            content_type: responseData.content_type || uploadedFile.type,
          }
        : {
            filename: uploadedFile.name,
            size: uploadedFile.size,
            content_type: uploadedFile.type,
          }

      setFile(null)
      resetFileInput()

      // 延迟跳转，让用户看到成功消息
      setTimeout(() => {
        onUploadSuccess(fileInfo)
      }, TIMING.NAVIGATION_DELAY_MS)
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
          </header>

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
        </section>
      </div>
    </main>
  )
}

export default UploadFile

