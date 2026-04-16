import React, { createContext, useContext, useReducer, useEffect, useRef, useCallback } from 'react';
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
  loading: true,
  error: null,
  /** False until the first successful shark track fetch — avoids showing "0 tracks" during cold start. */
  initialDataReady: false
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

    case 'SET_INITIAL_DATA_READY':
      return { ...state, initialDataReady: action.payload };
    
    default:
      return state;
  }
}

/** Max wall-clock time to wait for cold API (Render free can exceed 1–2 min). */
const INITIAL_LOAD_MAX_WALL_MS = 5 * 60 * 1000;
const MAX_INITIAL_ATTEMPTS = 50;

export function SharkProvider({ children }) {
  const [state, dispatch] = useReducer(sharkReducer, initialState);
  const loadAttemptRef = useRef(0);
  const cancelledRef = useRef(false);

  const loadInitialData = useCallback(async () => {
    const startedAt = Date.now();

    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    while (!cancelledRef.current) {
      const elapsed = Date.now() - startedAt;
      if (elapsed >= INITIAL_LOAD_MAX_WALL_MS) {
        dispatch({
          type: 'SET_ERROR',
          payload:
            'The data service did not respond in time (this can happen when the server was asleep). Tap Retry or refresh the page.',
        });
        dispatch({ type: 'SET_LOADING', payload: false });
        return;
      }

      loadAttemptRef.current += 1;
      const attempt = loadAttemptRef.current;

      if (attempt > MAX_INITIAL_ATTEMPTS) {
        dispatch({
          type: 'SET_ERROR',
          payload: 'Could not load shark data after several tries. Please check your connection and use Retry.',
        });
        dispatch({ type: 'SET_LOADING', payload: false });
        return;
      }

      try {
        dispatch({ type: 'SET_LOADING', payload: true });
        dispatch({ type: 'SET_ERROR', payload: null });

        await apiService.wakeBackend();
        if (cancelledRef.current) return;

        const tracks = await apiService.getSharkTracks({ limit: 1000 });
        if (cancelledRef.current) return;

        if (!Array.isArray(tracks)) {
          throw new Error('Unexpected response for shark tracks');
        }

        dispatch({ type: 'SET_SHARK_TRACKS', payload: tracks });
        dispatch({ type: 'SET_INITIAL_DATA_READY', payload: true });
        dispatch({ type: 'SET_LOADING', payload: false });
        loadAttemptRef.current = 0;

        void Promise.all([
          apiService.getModelPerformance().catch((err) => {
            console.warn('Failed to load model performance:', err);
            return [];
          }),
          apiService.getSpecies().catch((err) => {
            console.warn('Failed to load species:', err);
            return { species: [] };
          }),
          apiService.getStats().catch((err) => {
            console.warn('Failed to load stats:', err);
            return null;
          }),
        ]).then(([performance, species, stats]) => {
          if (cancelledRef.current) return;
          dispatch({ type: 'SET_MODEL_PERFORMANCE', payload: performance });
          dispatch({ type: 'SET_SPECIES', payload: species.species });
          dispatch({ type: 'SET_STATS', payload: stats });
        });

        return;
      } catch (error) {
        console.error('Error loading initial data:', error);
        if (cancelledRef.current) return;

        const wallLeft = INITIAL_LOAD_MAX_WALL_MS - (Date.now() - startedAt);
        if (wallLeft <= 0) {
          dispatch({
            type: 'SET_ERROR',
            payload:
              'The data service did not respond in time. Tap Retry or refresh — first load after idle can take a few minutes.',
          });
          dispatch({ type: 'SET_LOADING', payload: false });
          return;
        }

        const backoff = Math.min(20000, 2000 * Math.pow(1.25, attempt - 1));
        await sleep(Math.min(backoff, wallLeft));
      }
    }
  }, [dispatch]);

  const retryInitialLoad = useCallback(async () => {
    cancelledRef.current = false;
    loadAttemptRef.current = 0;
    dispatch({ type: 'SET_ERROR', payload: null });
    await loadInitialData();
  }, [dispatch, loadInitialData]);

  useEffect(() => {
    cancelledRef.current = false;
    loadInitialData();
    return () => {
      cancelledRef.current = true;
    };
  }, [loadInitialData]);

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
    retryInitialLoad,
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