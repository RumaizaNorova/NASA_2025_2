import React from 'react';
import { motion } from 'framer-motion';
import { Waves } from 'lucide-react';

const LoadingScreen = () => {
  return (
    <div className="fixed inset-0 bg-gradient-to-br from-shark-950 via-ocean-950 to-shark-900 flex items-center justify-center z-50">
      <div className="text-center">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <Waves className="w-24 h-24 text-ocean-400 mx-auto" />
        </motion.div>
        
        <motion.h1
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="text-4xl font-bold text-white mb-4"
        >
          Shark Habitat Prediction
        </motion.h1>
        
        <motion.p
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="text-ocean-300 text-lg mb-8"
        >
          Loading NASA satellite data and trained GradientBoosting model...
        </motion.p>
        
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100%" }}
          transition={{ delay: 0.6, duration: 1.5, ease: "easeInOut" }}
          className="h-1 bg-gradient-to-r from-ocean-500 to-ocean-300 rounded-full mx-auto max-w-md"
        />
        
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 0.5 }}
          className="mt-8 text-sm text-ocean-400"
        >
          <p>🦈 64,942 shark tracking records</p>
          <p>📡 MODIS-Aqua satellite data</p>
          <p>🤖 GradientBoosting model (AUC: 0.972)</p>
          <p>🎯 92.9% accuracy on test data</p>
        </motion.div>
      </div>
    </div>
  );
};

export default LoadingScreen;