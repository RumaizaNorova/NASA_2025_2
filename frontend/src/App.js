import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MapComponent from './components/MapComponent';
import Dashboard from './components/Dashboard';
import Navigation from './components/Navigation';
import LoadingScreen from './components/LoadingScreen';
import { SharkProvider } from './context/SharkContext';
import { ApiProvider } from './context/ApiContext';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [currentView, setCurrentView] = useState('map');

  useEffect(() => {
    // Simulate loading time for smooth UX
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <ApiProvider>
      <SharkProvider>
        <div className="min-h-screen bg-gradient-to-br from-shark-950 via-ocean-950 to-shark-900">
          {/* Background pattern */}
          <div className="fixed inset-0 opacity-5">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.1"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')]"></div>
          </div>

          {/* Navigation */}
          <Navigation currentView={currentView} setCurrentView={setCurrentView} />

          {/* Main Content */}
          <main className="relative z-10">
            <AnimatePresence mode="wait">
              {currentView === 'map' && (
                <motion.div
                  key="map"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                  className="h-screen"
                >
                  <MapComponent />
                </motion.div>
              )}
              
              {currentView === 'dashboard' && (
                <motion.div
                  key="dashboard"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="h-screen overflow-y-auto"
                >
                  <Dashboard />
                </motion.div>
              )}
            </AnimatePresence>
          </main>

          {/* Footer */}
          <footer className="fixed bottom-0 left-0 right-0 z-20 p-4">
            <div className="text-center text-sm text-ocean-300 opacity-70">
              <p>🦈 Shark Habitat Prediction Dashboard | NASA Data Integration | AUC: 0.972</p>
            </div>
          </footer>
        </div>
      </SharkProvider>
    </ApiProvider>
  );
}

export default App;

