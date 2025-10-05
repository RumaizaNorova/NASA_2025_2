import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Map, BarChart3, Bot } from 'lucide-react';
import OpenAIChat from './OpenAIChat';

const Navigation = ({ currentView, setCurrentView }) => {
  const [showAIChat, setShowAIChat] = useState(false);
  
  const navItems = [
    { id: 'map', label: 'Interactive Map', icon: Map },
    { id: 'dashboard', label: 'Analytics', icon: BarChart3 },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-30 bg-shark-900/90 backdrop-blur-md border-b border-shark-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="flex items-center space-x-3"
          >
            <div className="w-8 h-8 bg-gradient-to-br from-ocean-400 to-ocean-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">🦈</span>
            </div>
            <div>
              <h1 className="text-white font-semibold text-lg">Shark Habitat</h1>
              <p className="text-ocean-400 text-xs">NASA Data Integration</p>
            </div>
          </motion.div>

          {/* Navigation Items */}
          <div className="flex items-center space-x-1">
            {navItems.map((item, index) => {
              const Icon = item.icon;
              const isActive = currentView === item.id;
              
              return (
                <motion.button
                  key={item.id}
                  initial={{ y: -20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: index * 0.1, duration: 0.3 }}
                  onClick={() => setCurrentView(item.id)}
                  className={`relative px-4 py-2 rounded-lg transition-all duration-200 flex items-center space-x-2 ${
                    isActive
                      ? 'text-white bg-ocean-600/20 border border-ocean-500/30'
                      : 'text-ocean-300 hover:text-white hover:bg-shark-800/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                  
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute inset-0 bg-ocean-600/10 rounded-lg border border-ocean-500/20"
                      initial={false}
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    />
                  )}
                </motion.button>
              );
            })}
          </div>

          {/* AI Chat Button */}
          <motion.button
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            onClick={() => setShowAIChat(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600/20 to-pink-600/20 border border-purple-500/30 rounded-lg text-purple-400 hover:text-white hover:bg-purple-600/30 transition-all"
          >
            <Bot className="w-4 h-4" />
            <span className="text-sm font-medium">AI Assistant</span>
          </motion.button>
        </div>
      </div>

      {/* AI Chat Modal */}
      {showAIChat && (
        <OpenAIChat
          onClose={() => setShowAIChat(false)}
          predictionData={null}
          sharkData={null}
        />
      )}
    </nav>
  );
};

export default Navigation;
