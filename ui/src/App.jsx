import { useState } from 'react'
import './App.css'
import { ROUTES } from './constants/config'
import UploadFile from './pages/UploadFile'
import ParseFile from './pages/ParseFile'

/**
 * App 组件
 * 应用主入口，负责页面路由和状态管理
 */
function App() {
  const [currentPage, setCurrentPage] = useState(ROUTES.UPLOAD)
  const [uploadedFileInfo, setUploadedFileInfo] = useState(null)

  // 处理上传成功后的页面跳转
  const handleUploadSuccess = (fileInfo) => {
    setUploadedFileInfo(fileInfo)
    setCurrentPage(ROUTES.PARSE)
  }

  // 返回文件上传页面
  const handleBackToUpload = () => {
    setCurrentPage(ROUTES.UPLOAD)
    setUploadedFileInfo(null)
  }

  // 根据当前页面渲染对应组件
  return (
    <div className="app">
      {currentPage === ROUTES.PARSE ? (
        <ParseFile fileInfo={uploadedFileInfo} onBack={handleBackToUpload} />
      ) : (
        <UploadFile onUploadSuccess={handleUploadSuccess} />
      )}
    </div>
  )
}

export default App
