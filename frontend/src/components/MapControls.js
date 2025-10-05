import React from 'react';
import { motion } from 'framer-motion';
import { 
  Filter, 
  Play, 
  Pause, 
  RotateCcw, 
  Zap, 
  Target,
  Clock,
  MapPin
} from 'lucide-react';

const MapControls = ({
  selectedSpecies,
  setSelectedSpecies,
  showPredictions,
  setShowPredictions,
  isAnimating,
  setIsAnimating,
  animationSpeed,
  setAnimationSpeed,
  sharkTracks
}) => {
  // Get unique species from tracks
  const species = [...new Set(sharkTracks.map(track => track.species))];

  return (
    <div className="glass-card rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center space-x-2">
        <Target className="w-5 h-5 text-ocean-400" />
        <h3 className="text-white font-semibold">Map Controls</h3>
      </div>

      {/* Species Filter */}
      <div className="space-y-2">
        <label className="text-sm text-ocean-300 font-medium">Species Filter</label>
        <select
          value={selectedSpecies || ''}
          onChange={(e) => setSelectedSpecies(e.target.value || null)}
          className="w-full p-2 bg-shark-800/50 border border-shark-600 rounded-lg text-white text-sm focus:outline-none focus:border-ocean-500"
        >
          <option value="">All Species</option>
          {species.map(spec => (
            <option key={spec} value={spec}>{spec}</option>
          ))}
        </select>
      </div>

      {/* Prediction Toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Zap className="w-4 h-4 text-ocean-400" />
          <span className="text-sm text-ocean-300">Show Predictions</span>
        </div>
        <button
          onClick={() => setShowPredictions(!showPredictions)}
          className={`relative w-12 h-6 rounded-full transition-colors ${
            showPredictions ? 'bg-ocean-500' : 'bg-shark-600'
          }`}
        >
          <motion.div
            className="absolute top-1 w-4 h-4 bg-white rounded-full"
            animate={{ x: showPredictions ? 26 : 2 }}
            transition={{ type: "spring", stiffness: 500, damping: 30 }}
          />
        </button>
      </div>

      {/* Animation Controls */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Play className="w-4 h-4 text-ocean-400" />
            <span className="text-sm text-ocean-300">Animation</span>
          </div>
          <button
            onClick={() => setIsAnimating(!isAnimating)}
            className={`p-2 rounded-lg transition-colors ${
              isAnimating 
                ? 'bg-red-600/20 text-red-400 border border-red-500/30' 
                : 'bg-green-600/20 text-green-400 border border-green-500/30'
            }`}
          >
            {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
        </div>

        {/* Animation Speed */}
        {isAnimating && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-2"
          >
            <label className="text-sm text-ocean-300 font-medium">Speed</label>
            <input
              type="range"
              min="500"
              max="3000"
              step="100"
              value={animationSpeed}
              onChange={(e) => setAnimationSpeed(Number(e.target.value))}
              className="w-full h-2 bg-shark-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-ocean-400">
              <span>Slow</span>
              <span>Fast</span>
            </div>
          </motion.div>
        )}
      </div>

      {/* Reset View */}
      <button
        onClick={() => {
          // Reset to default view
          if (window.mapRef?.current) {
            window.mapRef.current.flyTo({
              center: [-80.0, 30.0],
              zoom: 6,
              duration: 1000
            });
          }
        }}
        className="w-full p-2 bg-shark-700/50 hover:bg-shark-600/50 border border-shark-600 rounded-lg text-ocean-300 hover:text-white transition-colors flex items-center justify-center space-x-2"
      >
        <RotateCcw className="w-4 h-4" />
        <span className="text-sm">Reset View</span>
      </button>

      {/* Stats */}
      <div className="pt-3 border-t border-shark-700">
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="text-center">
            <div className="text-ocean-400 font-semibold">
              {sharkTracks.length.toLocaleString()}
            </div>
            <div className="text-shark-400">Tracks</div>
          </div>
          <div className="text-center">
            <div className="text-ocean-400 font-semibold">
              {species.length}
            </div>
            <div className="text-shark-400">Species</div>
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="pt-3 border-t border-shark-700">
        <div className="text-xs text-ocean-400 space-y-1">
          <div className="flex items-center space-x-2">
            <MapPin className="w-3 h-3" />
            <span>Click map to predict habitat</span>
          </div>
          <div className="flex items-center space-x-2">
            <Target className="w-3 h-3" />
            <span>Click shark tracks for details</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MapControls;

