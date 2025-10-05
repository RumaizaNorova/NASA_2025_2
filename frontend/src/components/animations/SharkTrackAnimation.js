import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, RotateCcw, SkipForward, SkipBack } from 'lucide-react';

const SharkTrackAnimation = ({ 
  sharkTracks, 
  isPlaying, 
  onPlayPause, 
  speed, 
  onSpeedChange,
  onReset 
}) => {
  const [currentFrame, setCurrentFrame] = useState(0);
  const [animationProgress, setAnimationProgress] = useState(0);
  const animationRef = useRef(null);
  const intervalRef = useRef(null);

  // Sort tracks by datetime for animation
  const sortedTracks = sharkTracks
    .sort((a, b) => new Date(a.datetime) - new Date(b.datetime))
    .slice(0, 100); // Limit for performance

  const totalFrames = sortedTracks.length;

  useEffect(() => {
    if (isPlaying && totalFrames > 0) {
      intervalRef.current = setInterval(() => {
        setCurrentFrame(prev => {
          const next = prev + 1;
          if (next >= totalFrames) {
            onPlayPause(false); // Stop at end
            return totalFrames - 1;
          }
          return next;
        });
      }, speed);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, speed, totalFrames, onPlayPause]);

  useEffect(() => {
    setAnimationProgress((currentFrame / Math.max(totalFrames - 1, 1)) * 100);
  }, [currentFrame, totalFrames]);

  const handleReset = () => {
    setCurrentFrame(0);
    setAnimationProgress(0);
    onReset();
  };

  const handleSkipForward = () => {
    setCurrentFrame(prev => Math.min(prev + 10, totalFrames - 1));
  };

  const handleSkipBackward = () => {
    setCurrentFrame(prev => Math.max(prev - 10, 0));
  };

  const currentTrack = sortedTracks[currentFrame];
  const visibleTracks = sortedTracks.slice(0, currentFrame + 1);

  return (
    <div className="space-y-4">
      {/* Animation Controls */}
      <div className="glass-card rounded-xl p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold">Shark Movement Animation</h3>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-ocean-300">
              {currentFrame + 1} / {totalFrames}
            </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-4">
          <div className="w-full bg-shark-700 rounded-full h-2">
            <motion.div
              className="h-2 bg-gradient-to-r from-ocean-500 to-ocean-300 rounded-full"
              animate={{ width: `${animationProgress}%` }}
              transition={{ duration: 0.1 }}
            />
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center justify-center space-x-4">
          <button
            onClick={handleSkipBackward}
            className="p-2 bg-shark-700/50 hover:bg-shark-600/50 border border-shark-600 rounded-lg text-ocean-300 hover:text-white transition-colors"
            disabled={currentFrame === 0}
          >
            <SkipBack className="w-4 h-4" />
          </button>

          <button
            onClick={() => onPlayPause(!isPlaying)}
            className="p-3 bg-ocean-600 hover:bg-ocean-700 border border-ocean-500 rounded-lg text-white transition-colors"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>

          <button
            onClick={handleSkipForward}
            className="p-2 bg-shark-700/50 hover:bg-shark-600/50 border border-shark-600 rounded-lg text-ocean-300 hover:text-white transition-colors"
            disabled={currentFrame === totalFrames - 1}
          >
            <SkipForward className="w-4 h-4" />
          </button>

          <button
            onClick={handleReset}
            className="p-2 bg-shark-700/50 hover:bg-shark-600/50 border border-shark-600 rounded-lg text-ocean-300 hover:text-white transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Speed Control */}
        <div className="mt-4">
          <label className="text-sm text-ocean-300 font-medium">Animation Speed</label>
          <input
            type="range"
            min="100"
            max="2000"
            step="100"
            value={speed}
            onChange={(e) => onSpeedChange(Number(e.target.value))}
            className="w-full h-2 bg-shark-700 rounded-lg appearance-none cursor-pointer slider mt-2"
          />
          <div className="flex justify-between text-xs text-ocean-400 mt-1">
            <span>Slow</span>
            <span>Fast</span>
          </div>
        </div>
      </div>

      {/* Current Track Info */}
      {currentTrack && (
        <motion.div
          key={currentFrame}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-xl p-4"
        >
          <h4 className="text-white font-semibold mb-2">Current Track</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-ocean-400">Shark</div>
              <div className="text-white">{currentTrack.name}</div>
            </div>
            <div>
              <div className="text-ocean-400">Species</div>
              <div className="text-white">{currentTrack.species}</div>
            </div>
            <div>
              <div className="text-ocean-400">Location</div>
              <div className="text-white">
                {currentTrack.latitude.toFixed(4)}, {currentTrack.longitude.toFixed(4)}
              </div>
            </div>
            <div>
              <div className="text-ocean-400">Time</div>
              <div className="text-white">
                {new Date(currentTrack.datetime).toLocaleString()}
              </div>
            </div>
          </div>
          
          <div className="mt-3">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                currentTrack.foraging_behavior === 1 ? 'bg-red-500' : 'bg-green-500'
              }`}></div>
              <span className="text-sm text-ocean-300">
                {currentTrack.foraging_behavior === 1 ? 'Foraging' : 'Not Foraging'}
              </span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Animation Stats */}
      <div className="glass-card rounded-xl p-4">
        <h4 className="text-white font-semibold mb-3">Animation Statistics</h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-ocean-400">Total Tracks</div>
            <div className="text-white font-semibold">{totalFrames}</div>
          </div>
          <div>
            <div className="text-ocean-400">Progress</div>
            <div className="text-white font-semibold">{animationProgress.toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-ocean-400">Speed</div>
            <div className="text-white font-semibold">{speed}ms</div>
          </div>
          <div>
            <div className="text-ocean-400">Status</div>
            <div className="text-white font-semibold">
              {isPlaying ? 'Playing' : 'Paused'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SharkTrackAnimation;

