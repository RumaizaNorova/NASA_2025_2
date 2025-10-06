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
  MapPin,
  X,
  ChevronLeft,
  Globe,
  Map
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
  sharkTracks,
  onClose,
  currentFrame,
  totalFrames,
  mapStyle,
  setMapStyle
}) => {
  // Get unique species from tracks
  const species = [...new Set(sharkTracks.map(track => track.species))];

  return (
    <div className="glass-card rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Target className="w-5 h-5 text-ocean-400" />
          <h3 className="text-white font-semibold">Map Controls</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-shark-700/50 rounded-lg transition-colors text-shark-400 hover:text-white"
          title="Hide controls"
        >
          <X className="w-4 h-4" />
        </button>
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

      {/* Map Style Selector */}
      <div className="space-y-2">
        <label className="text-sm text-ocean-300 font-medium">Map Style</label>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setMapStyle('satellite')}
            className={`p-2 rounded-lg text-xs transition-colors flex items-center justify-center space-x-1 ${
              mapStyle === 'satellite' 
                ? 'bg-ocean-500 text-white' 
                : 'bg-shark-700/50 text-ocean-300 hover:bg-shark-600/50'
            }`}
          >
            <Map className="w-3 h-3" />
            <span>Map View</span>
          </button>
          <button
            onClick={() => setMapStyle('map')}
            className={`p-2 rounded-lg text-xs transition-colors flex items-center justify-center space-x-1 ${
              mapStyle === 'map' 
                ? 'bg-ocean-500 text-white' 
                : 'bg-shark-700/50 text-ocean-300 hover:bg-shark-600/50'
            }`}
          >
            <Globe className="w-3 h-3" />
            <span>Earth View</span>
          </button>
        </div>
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

        {/* Animation Speed - Always visible */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm text-ocean-300 font-medium">Speed</label>
            <span className="text-xs text-ocean-400">{animationSpeed}/30</span>
          </div>
          <input
            type="range"
            min="1"
            max="30"
            step="1"
            value={animationSpeed}
            onChange={(e) => setAnimationSpeed(Number(e.target.value))}
            className="w-full h-2 bg-shark-700 rounded-lg appearance-none cursor-pointer slider"
          />
          <div className="flex justify-between text-xs text-ocean-400">
            <span>Slow</span>
            <span className="text-red-400 font-semibold">Very Fast</span>
          </div>
          
          {/* Animation Progress */}
          {isAnimating && totalFrames > 0 && (
            <div className="text-xs text-ocean-400 text-center pt-2 border-t border-shark-700">
              <div className="mb-2">
                <Clock className="w-3 h-3 inline mr-1" />
                Frame {currentFrame + 1} / {totalFrames}
              </div>
              <div className="w-full bg-shark-700 rounded-full h-2">
                <div 
                  className="bg-ocean-500 h-2 rounded-full transition-all duration-200"
                  style={{ width: `${((currentFrame + 1) / totalFrames) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Reset View */}
      <button
        onClick={() => {
          // Reset to global view
          if (window.mapRef?.current) {
            window.mapRef.current.flyTo({
              center: [0.0, 0.0],
              zoom: 2,
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

      {/* Legend */}
      <div className="pt-3 border-t border-shark-700">
        <div className="text-xs text-ocean-300 font-medium mb-2">Shark Behavior</div>
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full border border-white"></div>
            <span className="text-xs text-ocean-400">Foraging</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full border border-white"></div>
            <span className="text-xs text-ocean-400">Not Foraging</span>
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