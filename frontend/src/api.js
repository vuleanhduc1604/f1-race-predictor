import axios from 'axios'

const client = axios.create({ baseURL: '/api' })

export const getYears = () => client.get('/years').then(r => r.data.years)

export const getEvents = (year) =>
  client.get('/events', { params: { year } }).then(r => r.data.events)

export const predict = (year, event) =>
  client.get('/predict', { params: { year, event } }).then(r => r.data)

export const predictLive = (year, event) =>
  client.get('/predict-live', { params: { year, event } }).then(r => r.data)

export const evaluate = (year) =>
  client.get('/evaluate', { params: { year } }).then(r => r.data)

export const getFeatureImportance = (topN = 25) =>
  client.get('/feature-importance', { params: { top_n: topN } }).then(r => r.data.features)
