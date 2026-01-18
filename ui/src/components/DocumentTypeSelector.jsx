/**
 * DocumentTypeSelector 组件
 * 文档类型选择器
 */
export const DocumentTypeSelector = ({ value, onChange, options = ['all', 'policy', 'claim'] }) => {
  const typeLabels = {
    all: 'All',
    policy: 'Policy',
    claim: 'Claim',
  }

  return (
    <div className="document-type-selector">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          className={`type-option ${value === option ? 'active' : ''}`}
          onClick={() => onChange(option)}
        >
          {typeLabels[option] || option}
        </button>
      ))}
    </div>
  )
}

export default DocumentTypeSelector

