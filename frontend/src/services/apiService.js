import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/** Default timeout — cold hosts (e.g. Render free) often need >10s to respond. */
const DEFAULT_TIMEOUT_MS = 90000;
/** First request after idle — allow several minutes (demo / cold spin-up). */
const COLD_START_TIMEOUT_MS = 180000;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: DEFAULT_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
  },
});

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function withRetry(fn, { retries = 3, baseDelayMs = 800 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;

      const status = err?.status;
      const msg = String(err?.message || '');
      const retriable =
        err?.isNetworkError ||
        err?.isTimeout ||
        status === 408 ||
        status === 425 ||
        status === 429 ||
        status === 500 ||
        status === 502 ||
        status === 503 ||
        status === 504 ||
        msg.toLowerCase().includes('network error') ||
        msg.toLowerCase().includes('timeout') ||
        msg.toLowerCase().includes('waking up');

      if (!retriable || attempt === retries) break;

      const delay = Math.min(12000, baseDelayMs * Math.pow(2, attempt));
      await sleep(delay);
    }
  }
  throw lastErr;
}

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);

    if (error.code === 'ECONNABORTED' || (error.message && String(error.message).toLowerCase().includes('timeout'))) {
      const err = new Error('Request timed out - server may be cold (try again in a moment)');
      err.isTimeout = true;
      err.isNetworkError = true;
      throw err;
    }

    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || 'Server error';
      const err = new Error(message);
      err.status = error.response.status;
      throw err;
    } else if (error.request) {
      // Request was made but no response received
      const err = new Error('Network error - backend may be waking up');
      err.isNetworkError = true;
      throw err;
    } else {
      // Something else happened
      throw new Error('Request failed');
    }
  }
);

export const apiService = {
  /**
   * Single cheap request with a long timeout so the host wakes before heavier routes.
   * Call this before parallel dashboard fetches on initial load.
   */
  async wakeBackend() {
    const response = await withRetry(
      () => api.get('/health', { timeout: COLD_START_TIMEOUT_MS }),
      { retries: 12, baseDelayMs: 2000 }
    );
    return response.data;
  },

  // Health check
  async healthCheck() {
    const response = await withRetry(() => api.get('/health'), { retries: 2 });
    return response.data;
  },

  // Get shark tracks
  async getSharkTracks(filters = {}) {
    const params = new URLSearchParams();
    
    if (filters.species) params.append('species', filters.species);
    if (filters.limit) params.append('limit', filters.limit);
    if (filters.start_date) params.append('start_date', filters.start_date);
    if (filters.end_date) params.append('end_date', filters.end_date);
    
    const response = await withRetry(
      () => api.get('/shark-tracks', { params, timeout: COLD_START_TIMEOUT_MS }),
      { retries: 12, baseDelayMs: 2000 }
    );
    return response.data;
  },

  // Predict habitat suitability
  async predictHabitat(predictionData) {
    const response = await withRetry(() => api.post('/predict', predictionData), { retries: 1 });
    return response.data;
  },

  // Get model performance
  async getModelPerformance() {
    const response = await withRetry(() => api.get('/model-performance'), { retries: 2 });
    return response.data;
  },

  // Get species list
  async getSpecies() {
    const response = await withRetry(() => api.get('/species'), { retries: 2 });
    return response.data;
  },

  // Get dataset statistics
  async getStats() {
    const response = await withRetry(() => api.get('/stats'), { retries: 2 });
    return response.data;
  },

  // Batch predictions for multiple locations
  async batchPredict(locations) {
    const promises = locations.map(location => 
      this.predictHabitat(location).catch(error => ({
        ...location,
        error: error.message
      }))
    );
    
    const results = await Promise.all(promises);
    return results;
  },

  // Get predictions for a grid of locations
  async getGridPredictions(bounds, resolution = 0.1) {
    const { north, south, east, west } = bounds;
    const locations = [];
    
    // Create grid of points
    for (let lat = south; lat <= north; lat += resolution) {
      for (let lng = west; lng <= east; lng += resolution) {
        locations.push({
          latitude: lat,
          longitude: lng,
          datetime: new Date().toISOString(),
          sst: 20.0, // Default values - could be enhanced with real-time data
          chlorophyll_a: 0.5,
          primary_productivity: 0.5,
          ssh_anomaly: 0.0
        });
      }
    }
    
    // Limit grid size for performance
    if (locations.length > 100) {
      const step = Math.ceil(locations.length / 100);
      return this.batchPredict(locations.filter((_, index) => index % step === 0));
    }
    
    return this.batchPredict(locations);
  },

  // OpenAI-powered features
  async generateInsights(predictionData, sharkData) {
    const response = await api.post('/generate-insights', {
      prediction_data: predictionData,
      shark_data: sharkData
    });
    return response.data;
  },

  async askQuestion(questionData) {
    const response = await api.post('/ask-question', questionData);
    return response.data;
  },

  async generateReport(analysisData) {
    const response = await api.post('/generate-report', analysisData);
    return response.data;
  }
};

export default apiService;