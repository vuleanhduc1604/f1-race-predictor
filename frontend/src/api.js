import axios from 'axios'

const client = axios.create({ baseURL: '/api' })

export const getYears = () => client.get('/years').then(r => r.data.years)

export const getEvents = (year) =>
  client.get('/events', { params: { year } }).then(r => r.data.events)

export const predictLiveStream = (year, event, onProgress) =>
  new Promise((resolve, reject) => {
    const url = `/api/predict?year=${year}&event=${encodeURIComponent(event)}`
    const es = new EventSource(url)
    es.addEventListener('progress', e => onProgress(JSON.parse(e.data).message))
    es.addEventListener('result', e => { es.close(); resolve(JSON.parse(e.data)) })
    es.addEventListener('fail', e => { es.close(); reject(new Error(JSON.parse(e.data).detail)) })
    es.onerror = () => { es.close(); reject(new Error('Connection to prediction server failed.')) }
  })

export const evaluate = (year) =>
  client.get('/evaluate', { params: { year } }).then(r => r.data)

export const getFeatureImportance = (topN = 25) =>
  client.get('/feature-importance', { params: { top_n: topN } }).then(r => r.data.features)
