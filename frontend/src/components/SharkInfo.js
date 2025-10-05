import React from 'react';
import { motion } from 'framer-motion';
import { X, MapPin, Clock, Fish, Target, Activity } from 'lucide-react';
import { format } from 'date-fns';

const SharkInfo = ({ shark, onClose }) => {
  const getForagingColor = (foraging) => {
    return foraging === 1 
      ? 'text-red-400 bg-red-600/20 border-red-500/30'
      : 'text-green-400 bg-green-600/20 border-green-500/30';
  };

  const getForagingText = (foraging) => {
    return foraging === 1 ? 'Foraging' : 'Not Foraging';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="glass-card rounded-xl p-4 space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Fish className="w-5 h-5 text-ocean-400" />
          <h3 className="text-white font-semibold">{shark.name}</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-shark-700/50 rounded-lg transition-colors"
        >
          <X className="w-4 h-4 text-ocean-400" />
        </button>
      </div>

      {/* Species */}
      <div className="space-y-1">
        <label className="text-xs text-ocean-400 font-medium">Species</label>
        <div className="text-sm text-white">{shark.species}</div>
      </div>

      {/* Location */}
      <div className="space-y-1">
        <label className="text-xs text-ocean-400 font-medium">Location</label>
        <div className="flex items-center space-x-2 text-sm text-white">
          <MapPin className="w-3 h-3 text-ocean-400" />
          <span>{shark.latitude.toFixed(4)}, {shark.longitude.toFixed(4)}</span>
        </div>
      </div>

      {/* DateTime */}
      <div className="space-y-1">
        <label className="text-xs text-ocean-400 font-medium">Date & Time</label>
        <div className="flex items-center space-x-2 text-sm text-white">
          <Clock className="w-3 h-3 text-ocean-400" />
          <span>{format(new Date(shark.datetime), 'MMM dd, yyyy HH:mm')}</span>
        </div>
      </div>

      {/* Foraging Status */}
      <div className="space-y-1">
        <label className="text-xs text-ocean-400 font-medium">Foraging Status</label>
        <div className={`inline-flex items-center space-x-2 px-3 py-1 rounded-lg border ${getForagingColor(shark.foraging_behavior)}`}>
          <Activity className="w-3 h-3" />
          <span className="text-sm font-medium">{getForagingText(shark.foraging_behavior)}</span>
        </div>
      </div>

      {/* Prediction Probability */}
      {shark.foraging_probability !== null && (
        <div className="space-y-1">
          <label className="text-xs text-ocean-400 font-medium">Predicted Probability</label>
          <div className="flex items-center space-x-2">
            <div className="flex-1 bg-shark-700 rounded-full h-2">
              <div 
                className="h-2 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full"
                style={{ width: `${(shark.foraging_probability || 0) * 100}%` }}
              />
            </div>
            <span className="text-sm text-white font-medium">
              {((shark.foraging_probability || 0) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex space-x-2 pt-2">
        <button className="flex-1 p-2 bg-ocean-600/20 hover:bg-ocean-600/30 border border-ocean-500/30 rounded-lg text-ocean-300 hover:text-white transition-colors text-xs font-medium">
          View Track
        </button>
        <button className="flex-1 p-2 bg-shark-700/50 hover:bg-shark-600/50 border border-shark-600 rounded-lg text-ocean-300 hover:text-white transition-colors text-xs font-medium">
          Predict Here
        </button>
      </div>
    </motion.div>
  );
};

export default SharkInfo;

