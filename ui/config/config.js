const apiBaseUrl =
  (import.meta.env.MODE === 'production' ? '' : 'http://localhost:8001');

export { apiBaseUrl };