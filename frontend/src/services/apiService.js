import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Handle different environments
const isDevelopment = process.env.NODE_ENV === 'development';
const isDocker = process.env.REACT_APP_API_URL && process.env.REACT_APP_API_URL.includes('backend');

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

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
    
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || 'Server error';
      throw new Error(message);
    } else if (error.request) {
      // Request was made but no response received
      throw new Error('Network error - please check your connection');
    } else {
      // Something else happened
      throw new Error('Request failed');
    }
  }
);

export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Get shark tracks
  async getSharkTracks(filters = {}) {
    const params = new URLSearchParams();
    
    if (filters.species) params.append('species', filters.species);
    if (filters.limit) params.append('limit', filters.limit);
    if (filters.start_date) params.append('start_date', filters.start_date);
    if (filters.end_date) params.append('end_date', filters.end_date);
    
    const response = await api.get('/shark-tracks', { params });
    return response.data;
  },

  // Predict habitat suitability
  async predictHabitat(predictionData) {
    const response = await api.post('/predict', predictionData);
    return response.data;
  },

  // Get model performance
  async getModelPerformance() {
    const response = await api.get('/model-performance');
    return response.data;
  },

  // Get species list
  async getSpecies() {
    const response = await api.get('/species');
    return response.data;
  },

  // Get dataset statistics
  async getStats() {
    const response = await api.get('/stats');
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
