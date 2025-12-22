import { useState } from 'react'
import './App.css'
import { ROUTES } from './constants/config'
import UploadFile from './pages/UploadFile'
import ParseFile from './pages/ParseFile'
import SearchFile from './pages/SearchFile'
import ChatQA from './pages/ChatQA'

/**
 * App 组件
 * 应用主入口，负责页面路由和状态管理
 */

// 开发模式：直接显示 ParseFile 组件（用于开发调试）
const DEV_MODE_SHOW_PARSE = false // 设置为 true 可直接显示 ParseFile 页面

// 模拟文件信息（用于开发调试）
const MOCK_FILE_INFO = {
  filename: 'policy_mini.pdf',
  content_type: 'application/pdf',
  size: 0, 
}

function App() {
  const [currentPage, setCurrentPage] = useState(DEV_MODE_SHOW_PARSE ? ROUTES.PARSE : ROUTES.UPLOAD)
  const [uploadedFileInfo, setUploadedFileInfo] = useState(DEV_MODE_SHOW_PARSE ? MOCK_FILE_INFO : null)

  // 处理上传成功后的页面跳转
  const handleUploadSuccess = (fileInfo) => {
    // fileInfo contains filename, size, content_type, task_id
    setUploadedFileInfo(fileInfo)
    setCurrentPage(ROUTES.PARSE)
  }

  // 返回文件上传页面
  const handleBackToUpload = () => {
    setCurrentPage(ROUTES.UPLOAD)
    setUploadedFileInfo(null)
  }

  // 跳转到搜索页面
  const handleGoToSearch = (detailedFileInfo) => {
    if (detailedFileInfo) {
      setUploadedFileInfo(curr => ({ ...curr, ...detailedFileInfo }))
    }
    setCurrentPage(ROUTES.SEARCH)
  }

  // 跳转到智能问答页面
  const handleGoToChat = (detailedFileInfo) => {
    if (detailedFileInfo) {
      setUploadedFileInfo(curr => ({ ...curr, ...detailedFileInfo }))
    }
    setCurrentPage(ROUTES.CHAT)
  }

  // 从搜索页面返回解析页面
  const handleBackToParse = () => {
    setCurrentPage(ROUTES.PARSE)
  }

  // 从智能问答页面返回解析页面
  const handleBackToParseFromChat = () => {
    setCurrentPage(ROUTES.PARSE)
  }

  // 根据当前页面渲染对应组件
  return (
    <div className="app">
      {currentPage === ROUTES.PARSE && (
        <ParseFile 
          fileInfo={uploadedFileInfo} 
          onBack={handleBackToUpload} 
          onSearch={handleGoToSearch}
          onChat={handleGoToChat}
        />
      )}
      {currentPage === ROUTES.SEARCH && (
        <SearchFile 
          fileInfo={uploadedFileInfo} 
          onBack={handleBackToParse} 
        />
      )}
      {currentPage === ROUTES.CHAT && (
        <ChatQA 
          fileInfo={uploadedFileInfo} 
          onBack={handleBackToParseFromChat} 
        />
      )}
      {currentPage === ROUTES.UPLOAD && (
        <UploadFile onUploadSuccess={handleUploadSuccess} />
      )}
    </div>
  )
}

export default App
