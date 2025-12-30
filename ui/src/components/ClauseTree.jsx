import { useState } from 'react'
import './ClauseTree.css'

function TreeNode({ node, level = 0, highlightedIds = new Set() }) {
  const [isExpanded, setIsExpanded] = useState(level < 2) // 默认展开前两层
  const hasChildren = node.children && node.children.length > 0
  // 排除根节点（ID 0）的高亮
  const isHighlighted = node.id > 0 && highlightedIds.has(node.id)
  const chunkCount = node.chunk_ids ? node.chunk_ids.length : 0

  return (
    <div className={`tree-node ${isHighlighted ? 'highlighted' : ''}`}>
      <div 
        className="tree-node-content"
        style={{ paddingLeft: `${level * 1.5}rem` }}
      >
        {hasChildren ? (
          <button
            className="tree-toggle"
            onClick={() => setIsExpanded(!isExpanded)}
            aria-label={isExpanded ? '折叠' : '展开'}
          >
            <span className={`tree-icon ${isExpanded ? 'expanded' : 'collapsed'}`}>
              ▶
            </span>
          </button>
        ) : (
          <span className="tree-icon leaf">•</span>
        )}
        
        <span className="tree-node-label">
          <span className="tree-node-title">{node.title || `节点 ${node.id}`}</span>
          <span className="tree-node-meta">
            <span className="tree-node-id">ID: {node.id}</span>
            {chunkCount > 0 && (
              <span className="tree-node-chunks">{chunkCount} chunks</span>
            )}
            {node.pages && node.pages.length > 0 && (
              <span className="tree-node-pages">
                页: {node.pages.length > 3 
                  ? `${node.pages.slice(0, 3).join(', ')}...` 
                  : node.pages.join(', ')}
              </span>
            )}
          </span>
        </span>
      </div>
      
      {hasChildren && isExpanded && (
        <div className="tree-children">
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              level={level + 1}
              highlightedIds={highlightedIds}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function ClauseTree({ clauseForest, highlightedClauseIds = [] }) {
  if (!clauseForest || !clauseForest.root) {
    return (
      <div className="clause-tree-empty">
        暂无条款森林数据
      </div>
    )
  }

  // 转换并过滤高亮 IDs，排除根节点 ID 0
  const highlightedIds = new Set(
    highlightedClauseIds
      .map(id => Number(id))
      .filter(id => !isNaN(id) && id > 0)
  )
  
  // Debug: 打印高亮 IDs
  if (highlightedIds.size > 0) {
    console.log('ClauseTree - Highlighted IDs set:', Array.from(highlightedIds))
  }

  return (
    <div className="clause-tree">
      <div className="clause-tree-header">
        <span className="clause-tree-stats">
          共 {clauseForest.clause_count || 0} 个条款，{clauseForest.node_count || 0} 个节点
        </span>
      </div>
      <div className="clause-tree-content">
        <TreeNode 
          node={clauseForest.root} 
          level={0}
          highlightedIds={highlightedIds}
        />
      </div>
    </div>
  )
}

export default ClauseTree

