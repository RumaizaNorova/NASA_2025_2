import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { apiService } from '../services/apiService';

const SharkContext = createContext();

const initialState = {
  sharkTracks: [],
  selectedShark: null,
  selectedSpecies: null,
  dateRange: {
    start: null,
    end: null
  },
  predictionData: null,
  modelPerformance: [],
  species: [],
  stats: null,
  loading: false,
  error: null
};

function sharkReducer(state, action) {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    
    case 'SET_SHARK_TRACKS':
      return { ...state, sharkTracks: action.payload, loading: false };
    
    case 'SET_SELECTED_SHARK':
      return { ...state, selectedShark: action.payload };
    
    case 'SET_SELECTED_SPECIES':
      return { ...state, selectedSpecies: action.payload };
    
    case 'SET_DATE_RANGE':
      return { ...state, dateRange: action.payload };
    
    case 'SET_PREDICTION_DATA':
      return { ...state, predictionData: action.payload };
    
    case 'SET_MODEL_PERFORMANCE':
      return { ...state, modelPerformance: action.payload };
    
    case 'SET_SPECIES':
      return { ...state, species: action.payload };
    
    case 'SET_STATS':
      return { ...state, stats: action.payload };
    
    default:
      return state;
  }
}

export function SharkProvider({ children }) {
  const [state, dispatch] = useReducer(sharkReducer, initialState);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      
      // Load data in parallel
      const [tracks, performance, species, stats] = await Promise.all([
        apiService.getSharkTracks({ limit: 1000 }).catch(err => {
          console.warn('Failed to load shark tracks:', err);
          return [];
        }),
        apiService.getModelPerformance().catch(err => {
          console.warn('Failed to load model performance:', err);
          return [];
        }),
        apiService.getSpecies().catch(err => {
          console.warn('Failed to load species:', err);
          return { species: [] };
        }),
        apiService.getStats().catch(err => {
          console.warn('Failed to load stats:', err);
          return null;
        })
      ]);
      
      dispatch({ type: 'SET_SHARK_TRACKS', payload: tracks });
      dispatch({ type: 'SET_MODEL_PERFORMANCE', payload: performance });
      dispatch({ type: 'SET_SPECIES', payload: species.species });
      dispatch({ type: 'SET_STATS', payload: stats });
      
    } catch (error) {
      console.error('Error loading initial data:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
    }
  };

  const loadSharkTracks = async (filters = {}) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      const tracks = await apiService.getSharkTracks(filters);
      dispatch({ type: 'SET_SHARK_TRACKS', payload: tracks });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });
    }
  };

  const predictHabitat = async (predictionData) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      const result = await apiService.predictHabitat(predictionData);
      dispatch({ type: 'SET_PREDICTION_DATA', payload: result });
      return result;
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });
      throw error;
    }
  };

  const value = {
    ...state,
    loadSharkTracks,
    predictHabitat,
    setSelectedShark: (shark) => dispatch({ type: 'SET_SELECTED_SHARK', payload: shark }),
    setSelectedSpecies: (species) => dispatch({ type: 'SET_SELECTED_SPECIES', payload: species }),
    setDateRange: (range) => dispatch({ type: 'SET_DATE_RANGE', payload: range }),
    clearError: () => dispatch({ type: 'SET_ERROR', payload: null })
  };

  return (
    <SharkContext.Provider value={value}>
      {children}
    </SharkContext.Provider>
  );
}

export function useShark() {
  const context = useContext(SharkContext);
  if (!context) {
    throw new Error('useShark must be used within a SharkProvider');
  }
  return context;
}