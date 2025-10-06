import React from 'react';
import { motion } from 'framer-motion';
import { X, Fish, MapPin, Calendar, Activity } from 'lucide-react';

const SharkInfo = ({ shark, onClose }) => {
  if (!shark) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="glass-card rounded-xl p-4 space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Fish className="w-5 h-5 text-ocean-400" />
          <h3 className="text-white font-semibold">Shark Details</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-shark-700/50 rounded-lg transition-colors"
        >
          <X className="w-4 h-4 text-ocean-400" />
        </button>
      </div>

      {/* Shark Info */}
      <div className="space-y-3">
        <div className="flex items-center space-x-2">
          <span className="text-sm text-ocean-400 font-medium">Name:</span>
          <span className="text-white">{shark.name}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="text-sm text-ocean-400 font-medium">Species:</span>
          <span className="text-white">{shark.species}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <MapPin className="w-4 h-4 text-ocean-400" />
          <span className="text-sm text-ocean-400">Location:</span>
          <span className="text-white text-sm">
            {shark.latitude.toFixed(4)}, {shark.longitude.toFixed(4)}
          </span>
        </div>
        
        <div className="flex items-center space-x-2">
          <Calendar className="w-4 h-4 text-ocean-400" />
          <span className="text-sm text-ocean-400">Date:</span>
          <span className="text-white text-sm">
            {new Date(shark.datetime).toLocaleDateString()}
          </span>
        </div>
        
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-ocean-400" />
          <span className="text-sm text-ocean-400">Behavior:</span>
          <span className={`text-sm font-medium ${
            shark.foraging_behavior ? 'text-red-400' : 'text-green-400'
          }`}>
            {shark.foraging_behavior ? 'Foraging' : 'Not Foraging'}
          </span>
        </div>
        
        {shark.foraging_probability && (
          <div className="pt-2 border-t border-shark-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-ocean-400">Foraging Probability:</span>
              <span className="text-white font-medium">
                {(shark.foraging_probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-shark-700 rounded-full h-2">
              <div
                className="h-2 rounded-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"
                style={{ width: `${shark.foraging_probability * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default SharkInfo;