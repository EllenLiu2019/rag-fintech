import ragLogo from '/rag.svg'
import './BrandCard.css'

/**
 * BrandCard 组件
 * 显示品牌信息和 Logo
 */
export const BrandCard = () => {
  return (
    <div className="brand-card">
      <img src={ragLogo} className="brand-logo" alt="RAG Logo" />
      <div>
        <p className="brand-eyebrow">RAG OF INTELLIGENT INSIGHTS</p>
        <h1>欢迎来到 RAG 智能洞察</h1>
        <p className="brand-tagline">
          让结构化与非结构化知识实时对齐，驱动精准的金融分析与决策支持。
        </p>
      </div>
    </div>
  )
}

export default BrandCard

