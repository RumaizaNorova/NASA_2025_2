import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MapComponent from './components/MapComponent';
import Dashboard from './components/Dashboard';
import Navigation from './components/Navigation';
import LoadingScreen from './components/LoadingScreen';
import { SharkProvider } from './context/SharkContext';
import { ApiProvider } from './context/ApiContext';
import './App.css';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [currentView, setCurrentView] = useState('map');
  const [showAIChat, setShowAIChat] = useState(false);

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
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent"></div>
          </div>

          {/* Navigation */}
          <Navigation 
            currentView={currentView} 
            setCurrentView={setCurrentView}
            showAIChat={showAIChat}
            setShowAIChat={setShowAIChat}
          />

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
              <p>🦈 Shark Habitat Prediction Dashboard | NASA Data Integration | GradientBoosting Model AUC: 0.972</p>
            </div>
          </footer>

          {/* Circular AI Assistant Indicator */}
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 1, duration: 0.5 }}
            onClick={() => setShowAIChat(true)}
            className="fixed bottom-6 right-6 z-30 w-14 h-14 bg-gradient-to-br from-purple-600 to-pink-600 rounded-full flex items-center justify-center shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110 border-2 border-purple-400/30"
          >
            <span className="text-2xl">🤖</span>
          </motion.button>
        </div>
      </SharkProvider>
    </ApiProvider>
  );
}

export default App;