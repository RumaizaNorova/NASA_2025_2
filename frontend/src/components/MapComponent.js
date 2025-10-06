import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMapGL, { Source, Layer, Marker, Popup } from 'react-map-gl';
import { useShark } from '../context/SharkContext';
import { apiService } from '../services/apiService';
import MapControls from './MapControls';
import PredictionPanel from './PredictionPanel';
import SharkInfo from './SharkInfo';
import { 
  MapPin, 
  Zap, 
  Filter, 
  Play, 
  Pause, 
  RotateCcw,
  Layers,
  Target,
  AlertCircle
} from 'lucide-react';

const MapComponent = () => {
  const mapRef = useRef();
  const { sharkTracks, selectedShark, setSelectedShark, loading, error } = useShark();
  
  const [viewState, setViewState] = useState({
    longitude: 0.0,
    latitude: 0.0,
    zoom: 2 // Default zoom, will be updated by handleMapStyleChange
  });
  
  const [predictionData, setPredictionData] = useState(null);
  const [showPredictions, setShowPredictions] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(5);
  const [selectedSpecies, setSelectedSpecies] = useState(null);
  const [showControls, setShowControls] = useState(true);
  const [clickedLocation, setClickedLocation] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [currentAnimationTime, setCurrentAnimationTime] = useState(null);
  const [animationInterval, setAnimationInterval] = useState(null);
  const [animationTrail, setAnimationTrail] = useState([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [mapStyle, setMapStyle] = useState('satellite'); // Default to flat map view
  
  // Update zoom when switching map styles
  const handleMapStyleChange = (newStyle) => {
    setMapStyle(newStyle);
    // Adjust zoom based on style
    const newZoom = newStyle === 'map' ? 1.5 : 2;
    setViewState(prev => ({
      ...prev,
      zoom: newZoom
    }));
  };

  // Mapbox token from environment
  const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN;
  
  // Get map style URL
  const getMapStyle = () => {
    switch (mapStyle) {
      case 'satellite':
        return 'mapbox://styles/mapbox/satellite-streets-v12'; // Brighter satellite for flat map
      case 'map':
        return 'mapbox://styles/mapbox/satellite-streets-v12'; // Brighter satellite for globe
      default:
        return 'mapbox://styles/mapbox/satellite-streets-v12';
    }
  };
  
  // Debug logging
  console.log('Mapbox Token:', MAPBOX_TOKEN ? 'Present' : 'Missing');
  console.log('Token length:', MAPBOX_TOKEN ? MAPBOX_TOKEN.length : 0);
  console.log('Token starts with pk.:', MAPBOX_TOKEN ? MAPBOX_TOKEN.startsWith('pk.') : false);
  
  if (!MAPBOX_TOKEN) {
    console.error('Mapbox token is missing! Please set REACT_APP_MAPBOX_TOKEN environment variable.');
  } else if (!MAPBOX_TOKEN.startsWith('pk.')) {
    console.error('Invalid Mapbox token format! Token should start with "pk."');
  }

  // Handle map click for predictions
  const handleMapClick = useCallback(async (event) => {
    const { lngLat } = event;
    
    // Extract coordinates from lngLat object
    const longitude = lngLat.lng;
    const latitude = lngLat.lat;
    
    setClickedLocation({ longitude, latitude });
    setIsPredicting(true);
    
    try {
      const prediction = await apiService.predictHabitat({
        latitude,
        longitude,
        datetime: new Date().toISOString(),
        sst: 20.0, // Default values - could be enhanced with real-time data
        chlorophyll_a: 0.5,
        primary_productivity: 0.5,
        ssh_anomaly: 0.0
      });
      
      setPredictionData({
        ...prediction,
        location: { latitude, longitude }
      });
    } catch (error) {
      console.error('Prediction failed:', error);
      // Show error message to user
      setPredictionData({
        error: error.message,
        location: { latitude, longitude }
      });
    } finally {
      setIsPredicting(false);
    }
  }, []);

  // Cleanup animation interval on unmount
  useEffect(() => {
    return () => {
      if (animationInterval) {
        clearInterval(animationInterval);
      }
    };
  }, [animationInterval]);

  // Get sorted tracks for animation
  const getSortedTracks = useCallback(() => {
    if (sharkTracks.length === 0) return [];
    
    // Filter valid tracks and sort by datetime
    const validTracks = sharkTracks
      .filter(track => {
        const date = new Date(track.datetime);
        return !isNaN(date.getTime());
      })
      .sort((a, b) => new Date(a.datetime) - new Date(b.datetime));
    
    // Sample tracks to get better geographic distribution
    const totalTracks = validTracks.length;
    const sampleSize = Math.min(1000, totalTracks); // Increased from 500 to 1000
    
    if (totalTracks <= sampleSize) {
      return validTracks;
    }
    
    // Sample evenly across the time range to get geographic diversity
    const step = Math.floor(totalTracks / sampleSize);
    const sampledTracks = [];
    
    for (let i = 0; i < totalTracks; i += step) {
      if (sampledTracks.length < sampleSize) {
        sampledTracks.push(validTracks[i]);
      }
    }
    
    return sampledTracks;
  }, [sharkTracks]);

  // Get time range for animation
  const getTimeRange = useCallback(() => {
    const sortedTracks = getSortedTracks();
    if (sortedTracks.length === 0) return { min: null, max: null };
    
    const times = sortedTracks.map(track => new Date(track.datetime).getTime());
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    
    return {
      min: new Date(minTime),
      max: new Date(maxTime),
      totalFrames: sortedTracks.length
    };
  }, [getSortedTracks]);

  // Start animation
  const startAnimation = useCallback(() => {
    console.log('=== START ANIMATION CALLED ===');
    
    const timeRange = getTimeRange();
    if (!timeRange.min || !timeRange.max || !timeRange.totalFrames) {
      console.error('No valid time range for animation');
      return;
    }
    
    console.log('Starting animation with', timeRange.totalFrames, 'frames');
    
    // Reset animation state
    setCurrentFrame(0);
    setAnimationTrail([]);
    setIsAnimating(true);
    
    const interval = setInterval(() => {
      setCurrentFrame(prevFrame => {
        const nextFrame = prevFrame + 1;
        
        if (nextFrame >= timeRange.totalFrames) {
          clearInterval(interval);
          setIsAnimating(false);
          console.log('Animation completed');
          return timeRange.totalFrames - 1;
        }
        
        return nextFrame;
      });
    }, Math.max(10, 500 - (animationSpeed * 15))); // Speed control: 1-30 maps to 485-10ms (very fast)
    
    setAnimationInterval(interval);
  }, [getTimeRange, animationSpeed]);

  // Stop animation
  const stopAnimation = useCallback(() => {
    if (animationInterval) {
      clearInterval(animationInterval);
      setAnimationInterval(null);
    }
    setIsAnimating(false);
    setCurrentFrame(0);
    setAnimationTrail([]);
  }, [animationInterval]);

  // Toggle animation
  const toggleAnimation = useCallback(() => {
    console.log('=== TOGGLE ANIMATION CLICKED ===');
    console.log('Current isAnimating:', isAnimating);
    console.log('sharkTracks.length:', sharkTracks.length);
    
    if (isAnimating) {
      console.log('Stopping animation...');
      stopAnimation();
    } else {
      console.log('Starting animation...');
      startAnimation();
    }
  }, [isAnimating, startAnimation, stopAnimation]);

  // Create animation trail effect
  const createAnimationTrail = useCallback(() => {
    if (!isAnimating) return [];
    
    const sortedTracks = getSortedTracks();
    const trailLength = Math.min(20, Math.floor(sortedTracks.length / 10)); // Trail length based on data size
    
    // Get current position and recent positions for trail
    const currentTrack = sortedTracks[currentFrame];
    if (!currentTrack) return [];
    
    const trail = [];
    
    // Add current position (brightest)
    trail.push({
      ...currentTrack,
      opacity: 1.0,
      size: 8
    });
    
    // Add trail positions (fading)
    for (let i = 1; i <= trailLength; i++) {
      const trailIndex = currentFrame - i;
      if (trailIndex >= 0) {
        const trailTrack = sortedTracks[trailIndex];
        if (trailTrack) {
          trail.push({
            ...trailTrack,
            opacity: Math.max(0.1, 1.0 - (i / trailLength)),
            size: Math.max(4, 8 - (i * 0.2))
          });
        }
      }
    }
    
    return trail;
  }, [isAnimating, currentFrame, getSortedTracks]);

  // Filter shark tracks by species and time (for animation)
  const filteredTracks = useMemo(() => {
    let tracks = selectedSpecies 
      ? sharkTracks.filter(track => 
          track.species.toLowerCase().includes(selectedSpecies.toLowerCase())
        )
      : sharkTracks;

    // If animating, show trail effect
    if (isAnimating) {
      return createAnimationTrail();
    }

    return tracks;
  }, [sharkTracks, selectedSpecies, isAnimating, createAnimationTrail]);

  // Create shark track data for map
  const sharkTrackData = {
    type: 'FeatureCollection',
    features: filteredTracks.map(track => ({
      type: 'Feature',
      properties: {
        id: track.id,
        name: track.name,
        species: track.species,
        foraging: track.foraging_behavior,
        datetime: track.datetime,
        opacity: track.opacity || 0.8,
        size: track.size || 6
      },
      geometry: {
        type: 'Point',
        coordinates: [track.longitude, track.latitude]
      }
    }))
  };

  // Create prediction overlay data
  const predictionOverlayData = predictionData && !predictionData.error ? {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      properties: {
        probability: predictionData.foraging_probability,
        confidence: predictionData.confidence
      },
      geometry: {
        type: 'Point',
        coordinates: [
          predictionData.location.longitude,
          predictionData.location.latitude
        ]
      }
    }]
  } : null;

  return (
    <div className="relative w-full h-full">
      {/* Error Display */}
      {error && (
        <div className="absolute top-20 left-1/2 transform -translate-x-1/2 z-30">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-red-600/90 backdrop-blur-sm border border-red-500 rounded-lg p-4 text-white"
          >
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5" />
              <span>Error loading data: {error}</span>
            </div>
          </motion.div>
        </div>
      )}


      {/* Starfield Background for Globe View */}
      {mapStyle === 'map' && (
        <div 
          className="absolute inset-0 z-0"
          style={{
            background: `
              radial-gradient(ellipse at center, transparent 40%, #000 70%),
              radial-gradient(ellipse at 20% 30%, rgba(255,255,255,0.1) 1px, transparent 2px),
              radial-gradient(ellipse at 80% 20%, rgba(255,255,255,0.1) 1px, transparent 2px),
              radial-gradient(ellipse at 40% 80%, rgba(255,255,255,0.1) 1px, transparent 2px),
              radial-gradient(ellipse at 60% 70%, rgba(255,255,255,0.1) 1px, transparent 2px),
              radial-gradient(ellipse at 10% 60%, rgba(255,255,255,0.1) 1px, transparent 2px),
              radial-gradient(ellipse at 90% 40%, rgba(255,255,255,0.1) 1px, transparent 2px),
              radial-gradient(ellipse at 30% 10%, rgba(255,255,255,0.1) 1px, transparent 2px),
              radial-gradient(ellipse at 70% 90%, rgba(255,255,255,0.1) 1px, transparent 2px),
              linear-gradient(45deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)
            `,
            backgroundSize: '100% 100%, 200px 200px, 300px 300px, 150px 150px, 250px 250px, 180px 180px, 220px 220px, 160px 160px, 280px 280px, 100% 100%'
          }}
        />
      )}

      {/* Map */}
      {MAPBOX_TOKEN ? (
        <ReactMapGL
          ref={mapRef}
          {...viewState}
          onMove={evt => setViewState(evt.viewState)}
          onClick={handleMapClick}
          mapStyle={getMapStyle()}
          mapboxAccessToken={MAPBOX_TOKEN}
          style={{ width: '100%', height: '100%' }}
          cursor={isPredicting ? 'wait' : 'crosshair'}
          projection={mapStyle === 'map' ? 'globe' : 'mercator'}
        >
          {/* Shark tracks */}
          <Source id="shark-tracks" type="geojson" data={sharkTrackData}>
            <Layer
              id="shark-tracks-points"
              type="circle"
              paint={{
                'circle-radius': [
                  'case',
                  ['has', 'size'], ['get', 'size'],
                  ['case', ['==', ['get', 'foraging'], 1], 8, 6]
                ],
                'circle-color': [
                  'case',
                  ['==', ['get', 'foraging'], 1], '#ef4444', '#3b82f6'
                ],
                'circle-opacity': [
                  'case',
                  ['has', 'opacity'], ['get', 'opacity'], 0.8
                ],
                'circle-stroke-width': 2,
                'circle-stroke-color': '#ffffff',
                'circle-stroke-opacity': [
                  'case',
                  ['has', 'opacity'], ['get', 'opacity'], 0.8
                ]
              }}
            />
          </Source>

          {/* Prediction overlay */}
          {predictionOverlayData && (
            <Source id="prediction-overlay" type="geojson" data={predictionOverlayData}>
              <Layer
                id="prediction-points"
                type="circle"
                paint={{
                  'circle-radius': 12,
                  'circle-color': [
                    'interpolate',
                    ['linear'],
                    ['get', 'probability'],
                    0, '#1e40af',
                    0.5, '#f59e0b',
                    1, '#dc2626'
                  ],
                  'circle-opacity': 0.7,
                  'circle-stroke-width': 3,
                  'circle-stroke-color': '#ffffff',
                  'circle-stroke-opacity': 1
                }}
              />
            </Source>
          )}

          {/* Clicked location marker */}
          {clickedLocation && (
            <Marker
              longitude={clickedLocation.longitude}
              latitude={clickedLocation.latitude}
              anchor="center"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="w-4 h-4 bg-white rounded-full border-2 border-ocean-500 shadow-lg"
              />
            </Marker>
          )}
        </ReactMapGL>
      ) : (
        <div className="w-full h-full flex items-center justify-center bg-shark-900">
          <div className="text-center">
            <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
            <h3 className="text-white text-xl font-semibold mb-2">Mapbox Token Required</h3>
            <p className="text-ocean-300">
              Please set REACT_APP_MAPBOX_TOKEN environment variable to display the map.
            </p>
          </div>
        </div>
      )}

      {/* Controls */}
      <AnimatePresence>
        {showControls && (
          <motion.div
            initial={{ x: -300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -300, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute top-20 left-4 w-80 z-10"
          >
            <MapControls
              selectedSpecies={selectedSpecies}
              setSelectedSpecies={setSelectedSpecies}
              showPredictions={showPredictions}
              setShowPredictions={setShowPredictions}
              isAnimating={isAnimating}
              setIsAnimating={toggleAnimation}
              animationSpeed={animationSpeed}
              setAnimationSpeed={setAnimationSpeed}
              sharkTracks={sharkTracks}
              onClose={() => setShowControls(false)}
              currentFrame={currentFrame}
              totalFrames={getSortedTracks().length}
              mapStyle={mapStyle}
              setMapStyle={handleMapStyleChange}
            />
          </motion.div>
        )}
        
        {/* Show Controls Button - appears when controls are hidden */}
        {!showControls && (
          <motion.button
            initial={{ x: -50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: -50, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            onClick={() => setShowControls(true)}
            className="absolute top-20 left-4 z-10 p-3 bg-shark-900/80 backdrop-blur-sm border border-shark-700 rounded-lg text-white hover:bg-shark-800/80 transition-colors"
            title="Show map controls"
          >
            <Layers className="w-5 h-5" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Prediction Panel */}
      <AnimatePresence>
        {predictionData && (
          <motion.div
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 300, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute top-20 right-4 w-80 z-10"
          >
            <PredictionPanel
              prediction={predictionData}
              onClose={() => setPredictionData(null)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Shark Info Panel */}
      <AnimatePresence>
        {selectedShark && (
          <motion.div
            initial={{ y: 300, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 300, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute bottom-4 left-1/2 transform -translate-x-1/2 w-96 z-10"
          >
            <SharkInfo
              shark={selectedShark}
              onClose={() => setSelectedShark(null)}
            />
          </motion.div>
        )}
      </AnimatePresence>


      {/* Loading Overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-shark-900/50 backdrop-blur-sm flex items-center justify-center z-30"
          >
            <div className="glass-card rounded-lg p-6 text-center">
              <div className="animate-spin w-8 h-8 border-2 border-ocean-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              <p className="text-white">Loading shark data...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Prediction Loading */}
      <AnimatePresence>
        {isPredicting && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-30"
          >
            <div className="glass-card rounded-lg p-4 text-center">
              <div className="animate-spin w-6 h-6 border-2 border-ocean-500 border-t-transparent rounded-full mx-auto mb-2"></div>
              <p className="text-white text-sm">Predicting habitat...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default MapComponent;