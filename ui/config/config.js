// VITE_API_BASE_URL is injected at build time via `--mode production` + .env.production
// or by passing the env var directly: VITE_API_BASE_URL=http://x.x.x.x:8001 vite build
const apiBaseUrl =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.MODE === 'production' ? '' : 'http://localhost:8001');

export { apiBaseUrl };
export default { apiBaseUrl };