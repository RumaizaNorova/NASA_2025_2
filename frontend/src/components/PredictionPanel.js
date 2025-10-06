import React from 'react';
import { motion } from 'framer-motion';
import { X, Zap, Target, TrendingUp, AlertCircle } from 'lucide-react';

const PredictionPanel = ({ prediction, onClose }) => {
  if (!prediction) return null;

  // Handle error case
  if (prediction.error) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="glass-card rounded-xl p-4 space-y-4"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <h3 className="text-white font-semibold">Prediction Error</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-shark-700/50 rounded-lg transition-colors"
          >
            <X className="w-4 h-4 text-ocean-400" />
          </button>
        </div>
        <p className="text-red-300 text-sm">{prediction.error}</p>
      </motion.div>
    );
  }

  const { foraging_probability, confidence, prediction: pred, features_used, model_info, location } = prediction;

  // Determine prediction color and text
  const getPredictionStyle = (prob) => {
    if (prob >= 0.7) {
      return {
        color: 'text-red-400',
        bg: 'bg-red-600/20',
        border: 'border-red-500/30',
        text: 'High Foraging Probability'
      };
    } else if (prob >= 0.4) {
      return {
        color: 'text-yellow-400',
        bg: 'bg-yellow-600/20',
        border: 'border-yellow-500/30',
        text: 'Medium Foraging Probability'
      };
    } else {
      return {
        color: 'text-green-400',
        bg: 'bg-green-600/20',
        border: 'border-green-500/30',
        text: 'Low Foraging Probability'
      };
    }
  };

  const style = getPredictionStyle(foraging_probability);

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
          <Zap className="w-5 h-5 text-ocean-400" />
          <h3 className="text-white font-semibold">Habitat Prediction</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-shark-700/50 rounded-lg transition-colors"
        >
          <X className="w-4 h-4 text-ocean-400" />
        </button>
      </div>

      {/* Location */}
      <div className="space-y-1">
        <label className="text-xs text-ocean-400 font-medium">Location</label>
        <div className="text-sm text-white">
          {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}
        </div>
      </div>

      {/* Prediction Result */}
      <div className={`p-3 rounded-lg border ${style.bg} ${style.border}`}>
        <div className="flex items-center justify-between mb-2">
          <span className={`text-sm font-medium ${style.color}`}>
            {style.text}
          </span>
          <span className={`text-lg font-bold ${style.color}`}>
            {(foraging_probability * 100).toFixed(1)}%
          </span>
        </div>
        
        {/* Probability Bar */}
        <div className="w-full bg-shark-700 rounded-full h-2 mb-2">
          <motion.div
            className="h-2 rounded-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"
            initial={{ width: 0 }}
            animate={{ width: `${foraging_probability * 100}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </div>
        
        <div className="text-xs text-ocean-300">
          Confidence: {(confidence * 100).toFixed(1)}%
        </div>
      </div>

      {/* Model Info */}
      <div className="space-y-2">
        <label className="text-xs text-ocean-400 font-medium">Model Information</label>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <div className="text-shark-400">Type</div>
            <div className="text-white font-medium">{model_info.model_type}</div>
          </div>
          <div>
            <div className="text-shark-400">AUC Score</div>
            <div className="text-white font-medium">{model_info.auc_score}</div>
          </div>
          <div>
            <div className="text-shark-400">Training Samples</div>
            <div className="text-white font-medium">{model_info.training_samples.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-shark-400">Features Used</div>
            <div className="text-white font-medium">{features_used.length}</div>
          </div>
        </div>
      </div>

      {/* Key Features */}
      <div className="space-y-2">
        <label className="text-xs text-ocean-400 font-medium">Key Features</label>
        <div className="space-y-1">
          {features_used.slice(0, 5).map((feature, index) => (
            <div key={index} className="flex items-center space-x-2 text-xs">
              <div className="w-1.5 h-1.5 bg-ocean-400 rounded-full"></div>
              <span className="text-ocean-300">{feature}</span>
            </div>
          ))}
          {features_used.length > 5 && (
            <div className="text-xs text-ocean-400">
              +{features_used.length - 5} more features
            </div>
          )}
        </div>
      </div>

      {/* Interpretation */}
      <div className="pt-3 border-t border-shark-700">
        <div className="flex items-start space-x-2">
          <AlertCircle className="w-4 h-4 text-ocean-400 mt-0.5 flex-shrink-0" />
          <div className="text-xs text-ocean-300">
            <div className="font-medium mb-1">Interpretation:</div>
            <div>
              {foraging_probability >= 0.7 
                ? "This location shows high probability of shark foraging activity based on environmental conditions and temporal factors."
                : foraging_probability >= 0.4
                ? "This location shows moderate probability of shark foraging activity."
                : "This location shows low probability of shark foraging activity."
              }
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-2 pt-2">
        <button className="flex-1 p-2 bg-ocean-600/20 hover:bg-ocean-600/30 border border-ocean-500/30 rounded-lg text-ocean-300 hover:text-white transition-colors text-xs font-medium">
          Save Prediction
        </button>
        <button className="flex-1 p-2 bg-shark-700/50 hover:bg-shark-600/50 border border-shark-600 rounded-lg text-ocean-300 hover:text-white transition-colors text-xs font-medium">
          Share Location
        </button>
      </div>
    </motion.div>
  );
};

export default PredictionPanel;