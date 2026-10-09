import axios from 'axios'

const api = axios.create({
  baseURL: localStorage.getItem('memora-api-url') || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 120000,
})

api.interceptors.response.use(
  response => response,
  error => {
    const detail = error.response?.data?.detail || error.response?.data?.message
    error.userMessage = detail || (error.code === 'ECONNABORTED' ? 'The request took too long. Try again.' : 'Could not reach the MemoraAI API.')
    return Promise.reject(error)
  },
)

export const apiClient = {
  dashboard: () => api.get('/api/dashboard').then(r => r.data),
  status: () => api.get('/api/status').then(r => r.data),
  documents: (params = {}) => api.get('/api/documents', { params }).then(r => r.data),
  document: id => api.get(`/api/documents/${id}`).then(r => r.data),
  deleteDocument: id => api.delete(`/api/documents/${id}`).then(r => r.data),
  upload: (file, onUploadProgress) => {
    const body = new FormData()
    body.append('file', file)
    return api.post('/upload', body, { onUploadProgress }).then(r => r.data)
  },
  ask: question => api.post('/ask', { question }).then(r => r.data),
  memories: params => api.get('/api/memories', { params }).then(r => r.data),
  search: payload => api.post('/api/search', payload).then(r => r.data),
  timeline: params => api.get('/timeline/search', { params }).then(r => r.data),
  evaluations: () => api.get('/api/evaluations').then(r => r.data),
  evaluation: id => api.get(`/api/evaluations/${id}`).then(r => r.data),
}

export function errorMessage(error) { return error?.userMessage || error?.message || 'Something went wrong.' }
export function setApiBaseUrl(url) {
  api.defaults.baseURL = url.replace(/\/$/, '')
  localStorage.setItem('memora-api-url', api.defaults.baseURL)
}
