import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { apiBaseUrl } from '../../config/config'
import BackIcon from '../components/icons/BackIcon'
import './ClaimProcess.css'

function ClaimProcess({ fileInfo, onBack }) {
  const [loading, setLoading] = useState(false)
  const [approving, setApproving] = useState(false)
  const [error, setError] = useState('')

  // Phase 1: AI review result (pending human confirmation)
  const [reviewData, setReviewData] = useState(null)
  // Editable decisions (one per entity)
  const [decisions, setDecisions] = useState([])
  // Phase 2: Final claim decision
  const [claimResult, setClaimResult] = useState(null)

  // ── Time Travel state ──
  const [historyMode, setHistoryMode] = useState(false)
  const [evaluations, setEvaluations] = useState([])
  const [selectedEval, setSelectedEval] = useState(null)
  const [checkpoints, setCheckpoints] = useState([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState(null)
  const [checkpointState, setCheckpointState] = useState(null)
  const [historyLoading, setHistoryLoading] = useState(false)

  // Phase 1: Submit claim → get AI candidates for review
  const handleSubmitClaim = async () => {
    const docId = fileInfo?.document_id || fileInfo?.doc_id
    if (!docId) {
      setError('缺少文档ID，无法提交理赔申请')
      return
    }

    setLoading(true)
    setError('')
    setReviewData(null)
    setClaimResult(null)
    setDecisions([])

    try {
      const response = await fetch(`${apiBaseUrl}/api/claim/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: docId })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || '理赔处理失败')
      }

      const result = await response.json()
      setReviewData(result)

      // Initialize editable decisions from AI output
      const initialDecisions = (result.pending_reviews || []).map(review => {
        // Merge agent_output from ALL interrupts (encode_graph + stage_graph)
        const mergedAgentOutput = {}
        for (const interrupt of (review.interrupts || [])) {
          Object.assign(mergedAgentOutput, interrupt.agent_output || {})
        }
        const encodeAgent = mergedAgentOutput.encode_agent || {}
        const alignAgent = mergedAgentOutput.align_agent || {}
        const stageAgent = mergedAgentOutput.stage_agent || {}

        // align_agent.tool_calls[0].output is a dict of ALL aligned concepts
        const allAlignedConcepts = alignAgent.tool_calls?.[0]?.output || {}
        // AI's best pick key
        const bestKey = alignAgent.agent_response?.best_matched_concept || ''
        // Best matched concept (or first available)
        const bestConcept = allAlignedConcepts[bestKey]
          || Object.values(allAlignedConcepts)[0]
          || {}

        return {
          entity_index: review.entity_index,
          entity_name: review.entity_name,
          icd_concept_code: bestConcept.icd_concept_code || '',
          icd_concept_name: bestConcept.icd_name || '',
          tnm_stage: stageAgent.step_output || '',
          // Keep all candidates for display
          _allAlignedConcepts: allAlignedConcepts,
          _bestKey: bestKey,
          _encodeAgent: encodeAgent,
          _stageAgent: stageAgent,
          _alignAgent: alignAgent,
        }
      })
      setDecisions(initialDecisions)
    } catch (err) {
      setError(err.message || '理赔处理失败')
    } finally {
      setLoading(false)
    }
  }

  // Phase 2: Approve with human decisions → get final result
  const handleApprove = async () => {
    if (!reviewData) return

    setApproving(true)
    setError('')

    try {
      const response = await fetch(`${apiBaseUrl}/api/claim/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: reviewData.doc_id,
          thread_ids: reviewData.thread_ids,
          decisions: decisions.map(d => ({
            icd_concept_code: d.icd_concept_code,
            icd_concept_name: d.icd_concept_name,
            tnm_stage: d.tnm_stage,
          }))
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || '理赔审批失败')
      }

      const result = await response.json()
      setClaimResult(result)
    } catch (err) {
      setError(err.message || '理赔审批失败')
    } finally {
      setApproving(false)
    }
  }

  // Update a single decision field
  const updateDecision = (index, field, value) => {
    setDecisions(prev => prev.map((d, i) =>
      i === index ? { ...d, [field]: value } : d
    ))
  }

  // ── Time Travel handlers ──

  const handleToggleHistory = async () => {
    if (historyMode) {
      // Exit history mode
      setHistoryMode(false)
      setEvaluations([])
      setSelectedEval(null)
      setCheckpoints([])
      setSelectedCheckpoint(null)
      setCheckpointState(null)
      return
    }

    const docId = fileInfo?.document_id || fileInfo?.doc_id
    if (!docId) return

    setHistoryMode(true)
    setHistoryLoading(true)
    setError('')
    try {
      const res = await fetch(`${apiBaseUrl}/api/claim/${docId}/evaluations`)
      if (!res.ok) throw new Error('Failed to load evaluation history')
      const data = await res.json()
      setEvaluations(data)
    } catch (err) {
      setError(err.message || 'Failed to load history')
    } finally {
      setHistoryLoading(false)
    }
  }

  const handleSelectEval = async (evalRecord) => {
    setSelectedEval(evalRecord)
    setCheckpoints([])
    setSelectedCheckpoint(null)
    setCheckpointState(null)
    setHistoryLoading(true)
    setError('')
    try {
      const res = await fetch(`${apiBaseUrl}/api/claim/checkpoints/${evalRecord.thread_id}`)
      if (!res.ok) throw new Error('Failed to load checkpoints')
      const data = await res.json()
      setCheckpoints(data)
    } catch (err) {
      setError(err.message || 'Failed to load checkpoints')
    } finally {
      setHistoryLoading(false)
    }
  }

  const handleSelectCheckpoint = async (cp) => {
    if (!selectedEval) return
    setSelectedCheckpoint(cp.checkpoint_id)
    setCheckpointState(null)
    setHistoryLoading(true)
    setError('')
    try {
      const res = await fetch(
        `${apiBaseUrl}/api/claim/state/${selectedEval.thread_id}?checkpoint_id=${cp.checkpoint_id}`
      )
      if (!res.ok) throw new Error('Failed to load state')
      const data = await res.json()
      setCheckpointState(data)
    } catch (err) {
      setError(err.message || 'Failed to load checkpoint state')
    } finally {
      setHistoryLoading(false)
    }
  }

  const getEvalStatusColor = (status) => {
    const colors = {
      pending: '#888',
      reviewing: '#5be9ff',
      approved: '#00ff9d',
      completed: '#00ff9d',
      rejected: '#ff4444',
    }
    return colors[status] || '#888'
  }

  const getEvalStatusText = (status) => {
    const texts = {
      pending: '待处理',
      reviewing: '审核中',
      approved: '已批准',
      completed: '已完成',
      rejected: '已拒绝',
    }
    return texts[status] || status
  }

  const getNodeLabel = (nextNodes) => {
    if (!nextNodes || nextNodes.length === 0) return 'END'
    const labels = {
      encode_graph: 'Encode Graph',
      stage_graph: 'Stage Graph',
      __end__: 'END',
    }
    return nextNodes.map(n => labels[n] || n).join(', ')
  }

  const formatTimestamp = (ts) => {
    if (!ts) return ''
    try {
      const d = new Date(ts)
      return d.toLocaleString('zh-CN', { hour12: false })
    } catch {
      return ts
    }
  }

  const truncateContent = (content, maxLen = 120) => {
    if (!content) return ''
    const text = typeof content === 'string' ? content : JSON.stringify(content)
    return text.length > maxLen ? text.slice(0, maxLen) + '...' : text
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

  const isProcessing = loading || approving

  return (
    <div className="claim-process-container">
      <header className="claim-header">
        <button onClick={onBack} className="back-button" title="返回">
          <BackIcon size={18} color="var(--accent-cyan, #5be9ff)" />
        </button>
        <h1>理赔处理</h1>
      </header>

      <div className="claim-content">
        {/* Left Panel */}
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

          {/* History toggle */}
          <button
            className={`history-toggle-btn ${historyMode ? 'active' : ''}`}
            onClick={handleToggleHistory}
            disabled={!(fileInfo?.document_id || fileInfo?.doc_id)}
          >
            {historyMode ? 'Exit History' : 'Evaluation History'}
          </button>

          {/* Show submit button only before review and not in history mode */}
          {!reviewData && !historyMode && (
            <button
              className="submit-button"
              onClick={handleSubmitClaim}
              disabled={isProcessing || !(fileInfo?.document_id || fileInfo?.doc_id)}
            >
              {loading ? '正在分析...' : '提交理赔申请'}
            </button>
          )}
        </div>

        {/* Right Panel */}
        <div className="claim-right-panel">
          <div className="result-card">
            {error && (
              <div className="error-message">
                <p>{error}</p>
              </div>
            )}

            {/* ── History Mode ── */}
            {historyMode && (
              <div className="history-panel">
                <h2>Evaluation History</h2>

                {historyLoading && !checkpointState && (
                  <div className="loading-state">
                    <div className="spinner"></div>
                    <p>Loading...</p>
                  </div>
                )}

                {/* Evaluation list */}
                {!selectedEval && evaluations.length > 0 && !historyLoading && (
                  <div className="eval-history-list">
                    {evaluations.map((ev) => (
                      <div
                        key={ev.id}
                        className="eval-history-card"
                        onClick={() => handleSelectEval(ev)}
                      >
                        <div className="eval-card-header">
                          <span className="eval-entity-name">{ev.entity_name || `Entity ${ev.entity_index}`}</span>
                          <span
                            className="eval-status-badge"
                            style={{ backgroundColor: getEvalStatusColor(ev.status) }}
                          >
                            {getEvalStatusText(ev.status)}
                          </span>
                        </div>
                        <div className="eval-card-meta">
                          <span className="eval-thread-id" title={ev.thread_id}>
                            Thread: {ev.thread_id.length > 30 ? '...' + ev.thread_id.slice(-24) : ev.thread_id}
                          </span>
                          <span className="eval-time">{formatTimestamp(ev.created_at)}</span>
                        </div>
                        {ev.human_decision && (
                          <div className="eval-card-decision">
                            <span>ICD: {ev.human_decision.icd_concept_code}</span>
                            <span>TNM: {ev.human_decision.tnm_stage}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {!selectedEval && evaluations.length === 0 && !historyLoading && (
                  <div className="empty-state"><p>No evaluation history found for this document.</p></div>
                )}

                {/* Checkpoint timeline + State inspector */}
                {selectedEval && (
                  <div className="checkpoint-section">
                    <button className="back-to-evals-btn" onClick={() => {
                      setSelectedEval(null)
                      setCheckpoints([])
                      setSelectedCheckpoint(null)
                      setCheckpointState(null)
                    }}>
                      Back to evaluations
                    </button>

                    <div className="selected-eval-header">
                      <span className="eval-entity-name">{selectedEval.entity_name || `Entity ${selectedEval.entity_index}`}</span>
                      <span
                        className="eval-status-badge"
                        style={{ backgroundColor: getEvalStatusColor(selectedEval.status) }}
                      >
                        {getEvalStatusText(selectedEval.status)}
                      </span>
                    </div>

                    <div className="checkpoint-content-row">
                      {/* Timeline column */}
                      <div className="checkpoint-timeline">
                        <h3>Checkpoints</h3>
                        {checkpoints.length === 0 && !historyLoading && (
                          <p className="no-checkpoints">No checkpoints found.</p>
                        )}
                        {checkpoints.map((cp, idx) => (
                          <div
                            key={cp.checkpoint_id}
                            className={`checkpoint-step ${selectedCheckpoint === cp.checkpoint_id ? 'active' : ''}`}
                            onClick={() => handleSelectCheckpoint(cp)}
                          >
                            <div className="step-indicator">
                              <div className={`step-dot ${cp.has_interrupt ? 'interrupt' : ''}`} />
                              {idx < checkpoints.length - 1 && <div className="step-connector" />}
                            </div>
                            <div className="step-info">
                              <div className="step-header">
                                <span className="step-number">Step {cp.step}</span>
                                <span className={`source-badge source-${cp.source}`}>{cp.source}</span>
                                {cp.has_interrupt && <span className="interrupt-badge">INTERRUPT</span>}
                              </div>
                              <div className="step-next">
                                Next: {getNodeLabel(cp.next)}
                              </div>
                              {cp.created_at && (
                                <div className="step-time">{formatTimestamp(cp.created_at)}</div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* State inspector column */}
                      <div className="state-inspector">
                        <h3>State Inspector</h3>
                        {historyLoading && selectedCheckpoint && (
                          <div className="loading-state">
                            <div className="spinner"></div>
                            <p>Loading state...</p>
                          </div>
                        )}

                        {!selectedCheckpoint && !historyLoading && (
                          <div className="empty-state"><p>Select a checkpoint to inspect its state.</p></div>
                        )}

                        {checkpointState && !historyLoading && (
                          <div className="state-content">
                            {/* Agent outputs */}
                            {checkpointState.agent_output_dict && Object.keys(checkpointState.agent_output_dict).length > 0 && (
                              <div className="state-section">
                                <h4>Agent Outputs</h4>
                                {Object.entries(checkpointState.agent_output_dict).map(([key, output]) => (
                                  <details key={key} className="agent-output-item">
                                    <summary>{key}</summary>
                                    <div className="agent-output-detail">
                                      {output.agent_response?.reasoning && (
                                        <div className="detail-row">
                                          <span className="detail-label">Reasoning:</span>
                                          <p className="detail-value">{output.agent_response.reasoning}</p>
                                        </div>
                                      )}
                                      {output.step_output && (
                                        <div className="detail-row">
                                          <span className="detail-label">Output:</span>
                                          <span className="detail-value">{typeof output.step_output === 'string' ? output.step_output : JSON.stringify(output.step_output)}</span>
                                        </div>
                                      )}
                                      {output.tool_calls && output.tool_calls.length > 0 && (
                                        <div className="detail-row">
                                          <span className="detail-label">Tool Calls:</span>
                                          <pre className="detail-pre">{JSON.stringify(output.tool_calls, null, 2)}</pre>
                                        </div>
                                      )}
                                    </div>
                                  </details>
                                ))}
                              </div>
                            )}

                            {/* Human decision */}
                            {checkpointState.human_decision && (checkpointState.human_decision.icd_concept_code || checkpointState.human_decision.tnm_stage) && (
                              <div className="state-section">
                                <h4>Human Decision</h4>
                                <div className="decision-summary">
                                  <div className="detail-row">
                                    <span className="detail-label">ICD Code:</span>
                                    <span className="detail-value">{checkpointState.human_decision.icd_concept_code || '-'}</span>
                                  </div>
                                  <div className="detail-row">
                                    <span className="detail-label">ICD Name:</span>
                                    <span className="detail-value">{checkpointState.human_decision.icd_concept_name || '-'}</span>
                                  </div>
                                  <div className="detail-row">
                                    <span className="detail-label">TNM Stage:</span>
                                    <span className="detail-value">{checkpointState.human_decision.tnm_stage || '-'}</span>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Messages */}
                            {checkpointState.messages && checkpointState.messages.length > 0 && (
                              <div className="state-section">
                                <h4>Messages ({checkpointState.messages.length})</h4>
                                <div className="messages-list">
                                  {checkpointState.messages.map((msg, idx) => (
                                    <details key={idx} className="message-item">
                                      <summary>
                                        <span className={`msg-type msg-type-${msg.type}`}>{msg.type}</span>
                                        {msg.name && <span className="msg-name">{msg.name}</span>}
                                        <span className="msg-preview">{truncateContent(msg.content, 60)}</span>
                                      </summary>
                                      <div className="msg-content">
                                        <pre>{typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2)}</pre>
                                        {msg.tool_calls && msg.tool_calls.length > 0 && (
                                          <div className="msg-tool-calls">
                                            <strong>Tool Calls:</strong>
                                            <pre>{JSON.stringify(msg.tool_calls, null, 2)}</pre>
                                          </div>
                                        )}
                                      </div>
                                    </details>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── Normal Mode ── */}
            {!historyMode && (
              <>
                {isProcessing && (
                  <div className="loading-state">
                    <div className="spinner"></div>
                    <p>{loading ? '正在进行医疗实体分析...' : '正在完成理赔审批...'}</p>
                  </div>
                )}

                {/* Phase 1: Human Review Form */}
                {reviewData && !claimResult && !isProcessing && (
                  <div className="result-content">
                    <h2>人工审核</h2>
                    <p className="review-hint">AI 已完成医疗实体分析，请确认或修改以下编码信息：</p>

                    {decisions.map((decision, index) => (
                      <div key={index} className="review-entity-card">
                        <h3>{decision.entity_name}</h3>

                        {/* All aligned concept candidates */}
                        {decision._allAlignedConcepts && Object.keys(decision._allAlignedConcepts).length > 0 && (
                          <div className="aligned-concepts-section">
                            <label className="section-label">ICD-SNOMED 对齐结果 (点击选择):</label>
                            <div className="aligned-concepts-list">
                              {Object.entries(decision._allAlignedConcepts).map(([key, concept]) => (
                                <div
                                  key={key}
                                  className={`aligned-concept-item ${
                                    decision.icd_concept_code === concept.icd_concept_code ? 'selected' : ''
                                  } ${key === decision._bestKey ? 'ai-recommended' : ''}`}
                                  onClick={() => {
                                    updateDecision(index, 'icd_concept_code', concept.icd_concept_code || '')
                                    updateDecision(index, 'icd_concept_name', concept.icd_name || '')
                                  }}
                                >
                                  {key === decision._bestKey && (
                                    <span className="ai-badge">AI</span>
                                  )}
                                  <div className="concept-row">
                                    <span className="concept-label">ICD:</span>
                                    <span className="concept-value">{concept.icd_name} ({concept.icd_concept_code})</span>
                                  </div>
                                  <div className="concept-row">
                                    <span className="concept-label">SNOMED:</span>
                                    <span className="concept-value">{concept.target_snomed_name} ({concept.target_snomed_concept_code})</span>
                                  </div>
                                  <div className="concept-row">
                                    <span className="concept-label">路径:</span>
                                    <span className="concept-value">{(concept.rel_types || []).join(' → ')} (长度: {concept.path_length})</span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Editable fields */}
                        <div className="review-fields-row">
                          <div className="review-field">
                            <label>ICD-10 编码:</label>
                            <input
                              type="text"
                              value={decision.icd_concept_code}
                              onChange={(e) => updateDecision(index, 'icd_concept_code', e.target.value)}
                              placeholder="例如: C73.x00"
                            />
                          </div>
                          <div className="review-field">
                            <label>ICD-10 名称:</label>
                            <input
                              type="text"
                              value={decision.icd_concept_name}
                              onChange={(e) => updateDecision(index, 'icd_concept_name', e.target.value)}
                              placeholder="例如: 甲状腺恶性肿瘤"
                            />
                          </div>
                          <div className="review-field">
                            <label>TNM 分期:</label>
                            <input
                              type="text"
                              value={decision.tnm_stage}
                              onChange={(e) => updateDecision(index, 'tnm_stage', e.target.value)}
                              placeholder="例如: I期"
                            />
                          </div>
                        </div>

                        {/* AI reasoning details */}
                        <details className="ai-reasoning-details">
                          <summary>查看 AI 分析详情</summary>
                          <div className="ai-reasoning-content">
                            {decision._encodeAgent?.agent_response?.reasoning && (
                              <div className="agent-output-block">
                                <h4>encode_agent (编码选择)</h4>
                                <p className="reasoning-text">{decision._encodeAgent.agent_response.reasoning}</p>
                              </div>
                            )}
                            {decision._alignAgent?.agent_response?.reasoning && (
                              <div className="agent-output-block">
                                <h4>align_agent (概念对齐)</h4>
                                <p className="reasoning-text">{decision._alignAgent.agent_response.reasoning}</p>
                              </div>
                            )}
                            {decision._stageAgent?.agent_response?.reasoning && (
                              <div className="agent-output-block">
                                <h4>stage_agent (TNM 分期)</h4>
                                <p className="reasoning-text">{decision._stageAgent.agent_response.reasoning}</p>
                              </div>
                            )}
                          </div>
                        </details>
                      </div>
                    ))}

                    <button
                      className="submit-button approve-button"
                      onClick={handleApprove}
                      disabled={approving || decisions.some(d => !d.icd_concept_code || !d.tnm_stage)}
                    >
                      {approving ? '处理中...' : '确认并完成审批'}
                    </button>
                  </div>
                )}

                {/* Phase 2: Final Claim Decision */}
                {claimResult && !isProcessing && (
                  <div className="result-content">
                    <h2>理赔结果</h2>

                    <div className="result-section">
                      <h3>决策结果</h3>
                      <div className="status-row">
                        <div className="status-badge" style={{ backgroundColor: getStatusColor(claimResult.status) }}>
                          {getStatusText(claimResult.status)}
                        </div>
                      </div>
                    </div>

                    {claimResult.explanation && claimResult.explanation.trim() !== '' && (
                      <div className="result-section">
                        <h3>决策说明</h3>
                        <div className="explanation-content">
                          <p>{claimResult.explanation}</p>
                        </div>
                      </div>
                    )}

                    {claimResult.recommendations && claimResult.recommendations.length > 0 && (
                      <div className="result-section">
                        <h3>建议</h3>
                        <ul className="recommendations-list">
                          {claimResult.recommendations.map((rec, index) => (
                            <li key={index}>{rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {claimResult.eligible_items && claimResult.eligible_items.length > 0 && (
                      <div className="result-section">
                        <h3>符合理赔条件</h3>
                        <ul className="item-list eligible-list">
                          {claimResult.eligible_items.map((item, index) => (
                            <li key={index}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {claimResult.excluded_items && claimResult.excluded_items.length > 0 && (
                      <div className="result-section">
                        <h3>不符合理赔条件</h3>
                        <ul className="item-list excluded-list">
                          {claimResult.excluded_items.map((item, index) => (
                            <li key={index}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {claimResult.matched_clauses && typeof claimResult.matched_clauses === 'string' && claimResult.matched_clauses.trim() !== '' && (
                      <div className="result-section">
                        <h3>匹配的条款</h3>
                        <div className="markdown-content">
                          <ReactMarkdown>{claimResult.matched_clauses}</ReactMarkdown>
                        </div>
                      </div>
                    )}

                    {claimResult.reasoning && claimResult.reasoning.trim() !== '' && (
                      <div className="result-section">
                        <h3>推理过程</h3>
                        <div className="reasoning-content">
                          <pre>{claimResult.reasoning}</pre>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {!reviewData && !claimResult && !isProcessing && !error && (
                  <div className="empty-state">
                    <p>点击"提交理赔申请"按钮开始处理</p>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ClaimProcess
