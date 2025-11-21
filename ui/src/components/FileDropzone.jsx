import { FILE_UPLOAD_CONFIG } from '../constants/config'

/**
 * FileDropzone 组件
 * 文件拖拽上传区域
 */
export const FileDropzone = ({
  isDragActive,
  onDrop,
  onDragOver,
  onDragLeave,
  onFileInputChange,
  fileInputRef,
}) => {
  return (
    <label
      htmlFor="file-input"
      className={`upload-dropzone ${isDragActive ? 'drag-active' : ''}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <div className="dropzone-visual">
        <span className="arrow">↥</span>
        <div>
          <p>拖拽文件至此</p>
          <small>支持 TXT / PDF / DOC / DOCX ≤ {FILE_UPLOAD_CONFIG.MAX_SIZE_MB} MB</small>
        </div>
      </div>

      <button type="button" className="ghost-button">
        浏览文件
      </button>
      <input
        ref={fileInputRef}
        id="file-input"
        type="file"
        accept={FILE_UPLOAD_CONFIG.ACCEPTED_TYPES}
        onChange={onFileInputChange}
      />
    </label>
  )
}

export default FileDropzone

