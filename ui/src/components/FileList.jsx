import { formatBytes } from '../utils/formatBytes'
import { FILE_UPLOAD_CONFIG, SAMPLE_FILE } from '../constants/config'
import { RunIcon } from './icons/RunIcon'

/**
 * FileList 组件
 * 显示文件列表，包括用户选择的文件和示例文件
 */
export const FileList = ({
  file,
  isUploading,
  onFileUpload,
  onSampleFileUpload,
}) => {
  if (file) {
    return (
      <div className="file-list">
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
            onClick={onFileUpload}
            disabled={isUploading}
          >
            <RunIcon />
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="file-list">
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
          onClick={onSampleFileUpload}
          disabled={isUploading}
        >
          <RunIcon />
        </button>
      </div>
    </div>
  )
}

export default FileList

