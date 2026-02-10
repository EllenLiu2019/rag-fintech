import { useState, useEffect } from 'react'
import { apiBaseUrl } from '../../config/config'
import DocumentCard from './DocumentCard'

/**
 * DocumentList 组件
 * 文档列表，显示所有文档
 */
export const DocumentList = ({ 
  docTypeFilter = 'all', 
  onSelectDocument,
  onDeleteDocument,
  refreshTrigger 
}) => {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchDocuments()
  }, [docTypeFilter, refreshTrigger])

  const fetchDocuments = async () => {
    try {
      setLoading(true)
      setError('')

      const response = await fetch(`${apiBaseUrl}/api/documents?doc_type=${docTypeFilter}`)
      if (!response.ok) throw new Error('获取文档列表失败')
      const data = await response.json()
      setDocuments(data || [])
    } catch (err) {
      console.error('获取文档列表失败:', err)
      setError(err.message || '获取文档列表失败')
      setDocuments([])
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (document) => {
    if (!window.confirm(`确定要删除文档 "${document.file_name}" 吗？`)) {
      return
    }

    try {
      // TODO: implement DELETE /api/documents/{doc_id} endpoint
      alert('删除功能尚未实现')
      fetchDocuments()
    } catch (err) {
      console.error('删除文档失败:', err)
      alert('删除文档失败，请稍后重试')
    }
  }

  if (loading) {
    return (
      <div className="document-list-loading">
        <div className="spinner"></div>
        <p>加载文档列表...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="document-list-error">
        <p>❌ {error}</p>
        <button type="button" onClick={fetchDocuments}>重试</button>
      </div>
    )
  }

  if (documents.length === 0) {
    return (
      <div className="document-list-empty">
        <p>暂无文档</p>
        <small>上传文档后，它们会显示在这里</small>
      </div>
    )
  }

  return (
    <div className="document-list">
      {documents.map((doc) => (
        <DocumentCard
          key={doc.doc_id}
          document={doc}
          onSelect={onSelectDocument}
          onDelete={onDeleteDocument || handleDelete}
        />
      ))}
    </div>
  )
}

export default DocumentList

