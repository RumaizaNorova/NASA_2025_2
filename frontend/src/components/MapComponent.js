import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Map, { Source, Layer, Marker, Popup } from 'react-map-gl';
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
  Target
} from 'lucide-react';

const MapComponent = () => {
  const mapRef = useRef();
  const { sharkTracks, selectedShark, setSelectedShark, loading } = useShark();
  
  const [viewState, setViewState] = useState({
    longitude: -80.0,
    latitude: 30.0,
    zoom: 6
  });
  
  const [predictionData, setPredictionData] = useState(null);
  const [showPredictions, setShowPredictions] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1000);
  const [selectedSpecies, setSelectedSpecies] = useState(null);
  const [showControls, setShowControls] = useState(true);
  const [clickedLocation, setClickedLocation] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // Mapbox token from environment
  const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN;
  
  if (!MAPBOX_TOKEN) {
    console.error('Mapbox token is missing! Please set REACT_APP_MAPBOX_TOKEN environment variable.');
  }

  // Handle map click for predictions
  const handleMapClick = useCallback(async (event) => {
    const { lngLat } = event;
    const [longitude, latitude] = lngLat;
    
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
    } finally {
      setIsPredicting(false);
    }
  }, []);

  // Filter shark tracks by species
  const filteredTracks = selectedSpecies 
    ? sharkTracks.filter(track => 
        track.species.toLowerCase().includes(selectedSpecies.toLowerCase())
      )
    : sharkTracks;

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
        datetime: track.datetime
      },
      geometry: {
        type: 'Point',
        coordinates: [track.longitude, track.latitude]
      }
    }))
  };

  // Create prediction overlay data
  const predictionOverlayData = predictionData ? {
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
      {/* Map */}
      <Map
        ref={mapRef}
        {...viewState}
        onMove={evt => setViewState(evt.viewState)}
        onClick={handleMapClick}
        mapStyle="mapbox://styles/mapbox/satellite-streets-v12"
        mapboxAccessToken={MAPBOX_TOKEN}
        style={{ width: '100%', height: '100%' }}
        cursor={isPredicting ? 'wait' : 'crosshair'}
      >
        {/* Shark tracks */}
        <Source id="shark-tracks" type="geojson" data={sharkTrackData}>
          <Layer
            id="shark-tracks-points"
            type="circle"
            paint={{
              'circle-radius': [
                'case',
                ['==', ['get', 'foraging'], 1], 8, 6
              ],
              'circle-color': [
                'case',
                ['==', ['get', 'foraging'], 1], '#ef4444', '#3b82f6'
              ],
              'circle-opacity': 0.8,
              'circle-stroke-width': 2,
              'circle-stroke-color': '#ffffff',
              'circle-stroke-opacity': 0.8
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
      </Map>

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
              setIsAnimating={setIsAnimating}
              animationSpeed={animationSpeed}
              setAnimationSpeed={setAnimationSpeed}
              sharkTracks={sharkTracks}
            />
          </motion.div>
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

      {/* Toggle Controls Button */}
      <button
        onClick={() => setShowControls(!showControls)}
        className="absolute top-20 right-4 z-20 p-3 bg-shark-900/80 backdrop-blur-sm border border-shark-700 rounded-lg text-white hover:bg-shark-800/80 transition-colors"
      >
        <Layers className="w-5 h-5" />
      </button>

      {/* Loading Overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-shark-900/50 backdrop-blur-sm flex items-center justify-center z-30"
          >
            <div className="bg-shark-800/90 backdrop-blur-sm border border-shark-700 rounded-lg p-6 text-center">
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
            <div className="bg-shark-800/90 backdrop-blur-sm border border-shark-700 rounded-lg p-4 text-center">
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
