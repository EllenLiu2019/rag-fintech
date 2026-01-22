import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { apiBaseUrl } from '../../config/config'
import BackIcon from '../components/icons/BackIcon'
import './ClaimProcess.css'

function ClaimProcess({ fileInfo, onBack }) {
  console.log('=== ClaimProcess Rendered ===')
  console.log('fileInfo:', fileInfo)
  console.log('document_id:', fileInfo?.document_id)
  console.log('doc_id:', fileInfo?.doc_id)
  console.log('business_data:', fileInfo?.business_data)
  
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [claimResult, setClaimResult] = useState(null)

  const handleSubmitClaim = async () => {
    console.log('=== Submit Claim Clicked ===')
    console.log('fileInfo:', fileInfo)
    
    const docId = fileInfo?.document_id || fileInfo?.doc_id
    console.log('docId:', docId)
    
    if (!docId) {
      console.error('No docId found')
      setError('缺少文档ID，无法提交理赔申请')
      return
    }

    setLoading(true)
    setError('')
    setClaimResult(null)

    try {
      const url = `${apiBaseUrl}/api/claim/submit`
      console.log('Submitting to:', url)
      console.log('Request body:', { doc_id: docId })
      
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: docId
        })
      })

      console.log('Response status:', response.status)
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.error('Error response:', errorData)
        throw new Error(errorData.detail || '理赔处理失败')
      }

      const result = await response.json()
      console.log('Result:', result)
      setClaimResult(result)
    } catch (err) {
      console.error('Submit claim failed:', err)
      setError(err.message || '理赔处理失败')
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status) => {
    const colors = {
      'eligible': '#00ff9d',
      'not_eligible': '#ff4444',
      'partial': '#ffaa00',
      'need_info': '#5be9ff',
      'under_review': '#888',
    }
    return colors[status] || '#888'
  }

  const getStatusText = (status) => {
    const texts = {
      'eligible': '符合理赔条件',
      'not_eligible': '不符合理赔条件',
      'partial': '部分符合',
      'need_info': '需要更多信息',
      'under_review': '审核中',
    }
    return texts[status] || status
  }

  return (
    <div className="claim-process-container">
      <header className="claim-header">
        <button onClick={onBack} className="back-button" title="返回">
          <BackIcon size={18} color="var(--accent-cyan, #5be9ff)" />
        </button>
        <h1>理赔处理</h1>
      </header>

      <div className="claim-content">
        {/* 左侧面板：文件信息和理赔数据 */}
        <div className="claim-left-panel">
          <div className="info-card">
            <h2>文件信息</h2>
            <div className="info-list">
              <div className="info-item">
                <span className="label">文件名:</span>
                <span className="value">{fileInfo?.filename || '未知'}</span>
              </div>
              <div className="info-item">
                <span className="label">文档ID:</span>
                <span className="value">{fileInfo?.document_id || fileInfo?.doc_id || '未知'}</span>
              </div>
            </div>
          </div>

          {fileInfo?.business_data && (
            <div className="info-card">
              <h2>理赔信息</h2>
              <div className="info-list">
                {fileInfo.business_data.patient_id && (
                  <div className="info-item">
                    <span className="label">患者ID:</span>
                    <span className="value">{fileInfo.business_data.patient_id}</span>
                  </div>
                )}
                {fileInfo.business_data.policy_doc_id && (
                  <div className="info-item">
                    <span className="label">保单ID:</span>
                    <span className="value">{fileInfo.business_data.policy_doc_id}</span>
                  </div>
                )}
                {fileInfo.business_data.claim_type && (
                  <div className="info-item">
                    <span className="label">理赔类型:</span>
                    <span className="value">{fileInfo.business_data.claim_type}</span>
                  </div>
                )}
                {fileInfo.business_data.medical_entities && (
                  <div className="info-item">
                    <span className="label">医疗实体:</span>
                    <span className="value">{fileInfo.business_data.medical_entities.length} 项</span>
                  </div>
                )}
              </div>
            </div>
          )}

          <button 
            className="submit-button" 
            onClick={() => {
              console.log('Button clicked!')
              handleSubmitClaim()
            }}
            disabled={loading || !(fileInfo?.document_id || fileInfo?.doc_id)}
          >
            {loading ? '处理中...' : '提交理赔申请'}
            {console.log('Button disabled:', loading || !(fileInfo?.document_id || fileInfo?.doc_id))}
          </button>
        </div>

        {/* 右侧面板：理赔结果 */}
        <div className="claim-right-panel">
          <div className="result-card">
            <h2>理赔结果</h2>
            
            {error && (
              <div className="error-message">
                <p>❌ {error}</p>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="spinner"></div>
                <p>正在处理理赔申请...</p>
              </div>
            )}

            {claimResult && !loading && (
              <div className="result-content">
                {/* 决策结果状态 */}
                <div className="result-section">
                  <h3>决策结果</h3>
                  <div className="status-row">
                    <div className="status-badge" style={{ backgroundColor: getStatusColor(claimResult.status) }}>
                      {getStatusText(claimResult.status)}
                    </div>
                  </div>
                </div>

                {/* 决策说明 (explanation) */}
                {claimResult.explanation && claimResult.explanation.trim() !== '' && (
                  <div className="result-section">
                    <h3>📝 决策说明</h3>
                    <div className="explanation-content">
                      <p>{claimResult.explanation}</p>
                    </div>
                  </div>
                )}

                {/* 建议 */}
                {claimResult.recommendations && claimResult.recommendations.length > 0 && (
                  <div className="result-section">
                    <h3>💡 建议</h3>
                    <ul className="recommendations-list">
                      {claimResult.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* 符合理赔条件 (coverage_evidence) */}
                {claimResult.eligible_items && claimResult.eligible_items.length > 0 && (
                  <div className="result-section">
                    <h3>✅ 符合理赔条件</h3>
                    <ul className="item-list eligible-list">
                      {claimResult.eligible_items.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* 不符合理赔条件 (exclusion_evidence) */}
                {claimResult.excluded_items && claimResult.excluded_items.length > 0 && (
                  <div className="result-section">
                    <h3>❌ 不符合理赔条件</h3>
                    <ul className="item-list excluded-list">
                      {claimResult.excluded_items.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* 匹配的条款 (Markdown 格式) */}
                {claimResult.matched_clauses && typeof claimResult.matched_clauses === 'string' && claimResult.matched_clauses.trim() !== '' && (
                  <div className="result-section">
                    <h3>📋 匹配的条款</h3>
                    <div className="markdown-content">
                      <ReactMarkdown>{claimResult.matched_clauses}</ReactMarkdown>
                    </div>
                  </div>
                )}

                

                {/* 推理过程 (reasoning) */}
                {claimResult.reasoning && claimResult.reasoning.trim() !== '' && (
                  <div className="result-section">
                    <h3>🔍 推理过程</h3>
                    <div className="reasoning-content">
                      <pre>{claimResult.reasoning}</pre>
                    </div>
                  </div>
                )}

                
              </div>
            )}

            {!claimResult && !loading && !error && (
              <div className="empty-state">
                <p>点击"提交理赔申请"按钮开始处理</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ClaimProcess

