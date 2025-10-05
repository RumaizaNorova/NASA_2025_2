import React, { createContext, useContext } from 'react';
import { apiService } from '../services/apiService';

const ApiContext = createContext();

export function ApiProvider({ children }) {
  const value = {
    apiService
  };

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  );
}

export function useApi() {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
}

