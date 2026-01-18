import { formatBytes } from '../utils/formatBytes'

/**
 * DocumentCard 组件
 * 文档卡片，显示单个文档信息
 */
export const DocumentCard = ({ document, onSelect, onDelete }) => {
  const getStatusLabel = (status) => {
    const labels = {
      pending: '等待中',
      queued: '排队中',
      started: '处理中',
      finished: '已完成',
      failed: '失败',
    }
    return labels[status] || status
  }

  const getStatusClass = (status) => {
    const classes = {
      pending: 'status-pending',
      queued: 'status-queued',
      started: 'status-started',
      finished: 'status-finished',
      failed: 'status-failed',
    }
    return classes[status] || 'status-unknown'
  }

  const getTypeLabel = (docType) => {
    const labels = {
      policy: 'Policy',
      claim: 'Claim',
    }
    return labels[docType] || docType
  }

  return (
    <div className="document-card" onClick={() => onSelect && onSelect(document)}>
      <div className="document-card-header">
        <div className="document-info">
          <h4 className="document-name">{document.filename || document.file_name}</h4>
          <div className="document-meta">
            <span className={`type-badge type-${document.doc_type || 'unknown'}`}>
              {getTypeLabel(document.doc_type)}
            </span>
            {document.size && (
              <span className="document-size">{formatBytes(document.size)}</span>
            )}
          </div>
        </div>
        <div className={`status-badge ${getStatusClass(document.status)}`}>
          {getStatusLabel(document.status)}
        </div>
      </div>
      <div className="document-card-footer">
        <div className="document-time">
          {document.created_at && (
            <span>上传时间: {new Date(document.created_at).toLocaleString()}</span>
          )}
        </div>
        <div className="document-actions">
          {onSelect && (
            <button
              type="button"
              className="action-button"
              onClick={(e) => {
                e.stopPropagation()
                onSelect(document)
              }}
            >
              查看
            </button>
          )}
          {onDelete && (
            <button
              type="button"
              className="action-button delete"
              onClick={(e) => {
                e.stopPropagation()
                onDelete(document)
              }}
            >
              删除
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default DocumentCard

