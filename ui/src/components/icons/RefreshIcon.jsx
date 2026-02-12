/**
 * RefreshIcon 组件
 * 用于显示刷新按钮的圆形箭头图标
 */
function RefreshIcon({ size = 14, color = 'currentColor' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ display: 'inline-block', verticalAlign: 'middle' }}
    >
      <path
        d="M4 4V10H10"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M4 10C4.5 7 6.5 4 10 4C14.5 4 18 7.5 18 12"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M20 20V14H14"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M20 14C19.5 17 17.5 20 14 20C9.5 20 6 16.5 6 12"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

export default RefreshIcon

