/**
 * StatusMessage 组件
 * 显示上传状态消息
 */
export const StatusMessage = ({ feedback, status }) => {
  if (!feedback) return null

  const statusClass = {
    uploading: 'loading',
    success: 'success',
    error: 'error',
  }[status] || ''

  return (
    <div className={`status-message ${statusClass}`} role="status" aria-live="polite">
      {feedback}
    </div>
  )
}

export default StatusMessage

