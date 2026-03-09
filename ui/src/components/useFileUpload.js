import { useRef, useState, useCallback } from 'react'
import { apiBaseUrl } from '../../config/config';

/**
 * 文件上传自定义 Hook
 * @param {Object} options - 配置选项
 * @param {string} options.uploadUrl - 上传接口地址，默认为 '/api/upload'
 * @param {Function} options.onSuccess - 上传成功回调
 * @param {Function} options.onError - 上传失败回调
 * @returns {Object} 返回状态和方法
 */
export const useFileUpload = ({
  uploadUrl = '/api/process',
  docType = 'policy', // 'policy' | 'claim'
  policyDocId = '',   // required when docType === 'claim'
  onSuccess,
  onError,
} = {}) => {
  const [status, setStatus] = useState('idle') // idle | uploading | success | error
  const [feedback, setFeedback] = useState('')
  const uploadControllerRef = useRef(null)

  /**
   * 上传文件
   * @param {File} fileToUpload - 要上传的文件
   * @param {string} loadingMessage - 加载时的提示信息
   */
  const uploadFile = useCallback(async (fileToUpload, loadingMessage = '正在上传，请稍候...') => {
    if (!fileToUpload) {
      setStatus('error')
      setFeedback('请选择要上传的文件')
      return
    }

    if (status === 'uploading') {
      return
    }

    setStatus('uploading')
    setFeedback(loadingMessage)

    const formData = new FormData()
    formData.append('file', fileToUpload)

    const controller = new AbortController()
    uploadControllerRef.current = controller

    try {
      // 根据 docType 选择正确的上传 URL
      let actualUrl = uploadUrl
      if (docType === 'claim') {
        actualUrl = `/api/claim/upload?policy_doc_id=${encodeURIComponent(policyDocId)}`
      } else {
        actualUrl = '/api/process'
      }
      const finalUrl = apiBaseUrl + actualUrl
      
      // 🔍 调试信息：检查请求参数
      console.group('🔍 文件上传调试信息')
      console.log('📤 准备上传文件:', {
        fileName: fileToUpload.name,
        fileSize: fileToUpload.size,
        fileType: fileToUpload.type,
        docType: docType,
        uploadUrl: actualUrl,
        apiBaseUrl: apiBaseUrl,
        fullUrl: finalUrl,
        method: 'POST'
      })
      console.log('📋 FormData 内容:', formData)
      console.log('🌐 开始发送请求到:', finalUrl)
      
      const response = await fetch(finalUrl, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      console.log('✅ 收到响应:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok,
        headers: Object.fromEntries(response.headers.entries())
      })

      const payloadText = await response.text()

      if (!response.ok) {
        let errorMessage = '上传失败，请稍后重试'
        if (payloadText) {
          try {
            const parsed = JSON.parse(payloadText)
            if (parsed?.message) {
              errorMessage = parsed.message
            }
          } catch {
            errorMessage = payloadText
          }
        }
        throw new Error(errorMessage)
      }

      let successMessage = `文件 "${fileToUpload.name}" 上传成功`
      let responseData = null
      
      if (payloadText) {
        try {
          const parsed = JSON.parse(payloadText)
          if (parsed?.message) {
            successMessage = parsed.message
          }
          responseData = parsed
        } catch {
          successMessage = payloadText || successMessage
        }
      }

      setStatus('success')
      setFeedback(successMessage)

      // 调用成功回调，传递文件信息和响应数据
      if (onSuccess) {
        onSuccess(fileToUpload, successMessage, responseData)
      }
    } catch (error) {
      // 🔍 详细的错误调试信息
      const actualUrl = docType === 'claim' ? '/api/claim/upload' : '/api/process'
      const finalUrl = apiBaseUrl + actualUrl
      console.error('❌ 上传失败，错误详情:', {
        name: error.name,
        message: error.message,
        stack: error.stack,
        fullUrl: finalUrl,
        apiBaseUrl: apiBaseUrl,
        uploadUrl: actualUrl,
        docType: docType,
        errorType: error.constructor.name
      })
      
      if (error.name === 'AbortError') {
        console.log('⚠️ 上传已取消')
        setStatus('idle')
        setFeedback('')
      } else {
        const errorMessage = error.message || '上传失败，请稍后重试'
        console.error('❌ 设置错误状态:', errorMessage)
        setStatus('error')
        setFeedback(errorMessage)

        // 调用失败回调
        if (onError) {
          onError(error, errorMessage)
        }
      }
    } finally {
      console.groupEnd()
      uploadControllerRef.current = null
    }
  }, [status, apiBaseUrl, uploadUrl, docType, policyDocId, onSuccess, onError])

  /**
   * 取消上传
   */
  const cancelUpload = useCallback(() => {
    if (uploadControllerRef.current) {
      uploadControllerRef.current.abort()
      setStatus('idle')
      setFeedback('')
    }
  }, [])

  /**
   * 重置状态
   */
  const resetStatus = useCallback(() => {
    setStatus('idle')
    setFeedback('')
  }, [])

  /**
   * 设置错误状态
   * @param {string} errorMessage - 错误消息
   */
  const setError = useCallback((errorMessage) => {
    setStatus('error')
    setFeedback(errorMessage)
    if (onError) {
      onError(new Error(errorMessage), errorMessage)
    }
  }, [onError])

  return {
    status,
    feedback,
    uploadFile,
    cancelUpload,
    resetStatus,
    setError,
    isUploading: status === 'uploading',
    isSuccess: status === 'success',
    isError: status === 'error',
  }
}

