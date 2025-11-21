import { formatBytes } from '../utils/formatBytes'

/**
 * FileSummary 组件
 * 显示当前选择的文件信息摘要
 */
export const FileSummary = ({ file, status, onClear }) => {
  return (
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
        onClick={onClear}
        disabled={!file && status !== 'success'}
      >
        清除
      </button>
    </div>
  )
}

export default FileSummary

