/**
 * 应用配置文件
 * 包含文件上传相关的常量配置
 */

export const FILE_UPLOAD_CONFIG = {
  MAX_SIZE_MB: 3,
  MAX_SIZE_BYTES: 3 * 1024 * 1024,
  ACCEPTED_TYPES: '.txt,.pdf,.doc,.docx',
}

export const SAMPLE_FILE = {
  name: 'policy_mini.pdf',
  size: 1.9 * 1024 * 1024,
  type: 'application/pdf',
  path: '/data/policy_mini.pdf',
}

export const ROUTES = {
  UPLOAD: 'upload',
  PARSE: 'parse',
  SEARCH: 'search',
}

export const TIMING = {
  NAVIGATION_DELAY_MS: 1000,
}

