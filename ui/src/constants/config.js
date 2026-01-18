/**
 * 应用配置文件
 * 包含文件上传相关的常量配置
 */

export const FILE_UPLOAD_CONFIG = {
  MAX_SIZE_MB: 4,
  MAX_SIZE_BYTES: 4 * 1024 * 1024,
  ACCEPTED_TYPES: '.txt,.pdf,.doc,.docx',
}

export const SAMPLE_FILE = {
  name: 'policy_base.pdf',
  size: 2.1 * 1024 * 1024,
  type: 'application/pdf',
  path: '/data/policy_base.pdf',
}

export const ROUTES = {
  UPLOAD: 'upload',
  PARSE: 'parse',
  SEARCH: 'search',
  CHAT: 'chat',
  CLAIM_UPLOAD: 'claim-upload',
}

export const TIMING = {
  NAVIGATION_DELAY_MS: 1000,
}

