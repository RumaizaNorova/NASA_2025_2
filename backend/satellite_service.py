"""
Satellite Data Service for Shark Habitat Prediction
Provides realistic satellite data based on location and time patterns from training data
"""

import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors
import joblib
import os

logger = logging.getLogger(__name__)

class SatelliteDataService:
    """
    Service to provide satellite-derived environmental data for shark habitat prediction
    Uses training data patterns to generate realistic location-specific values
    """
    
    def __init__(self, training_data_path: str = "../integrated_data_full.csv"):
        """Initialize the satellite data service with training data patterns"""
        self.training_data_path = training_data_path
        self.training_data = None
        self.location_patterns = None
        self.temporal_patterns = None
        self.spatial_interpolator = None
        
        self._load_training_data()
        self._build_patterns()
    
    def _load_training_data(self):
        """Load and preprocess training data"""
        try:
            self.training_data = pd.read_csv(self.training_data_path)
            self.training_data['datetime'] = pd.to_datetime(self.training_data['datetime'])
            
            # Add temporal features
            self.training_data['month'] = self.training_data['datetime'].dt.month
            self.training_data['hour'] = self.training_data['datetime'].dt.hour
            self.training_data['day_of_year'] = self.training_data['datetime'].dt.dayofyear
            
            logger.info(f"Loaded training data: {len(self.training_data)} records")
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise
    
    def _build_patterns(self):
        """Build spatial and temporal patterns from training data"""
        if self.training_data is None:
            raise ValueError("Training data not loaded")
        
        # Create spatial patterns by location
        self.location_patterns = self.training_data.groupby(['latitude', 'longitude']).agg({
            'sst': ['mean', 'std'],
            'chlor_a': ['mean', 'std'],
            'primary_productivity': ['mean', 'std'],
            'ssh_anomaly': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        self.location_patterns.columns = ['latitude', 'longitude', 
                                        'sst_mean', 'sst_std',
                                        'chl_mean', 'chl_std',
                                        'pp_mean', 'pp_std',
                                        'ssh_mean', 'ssh_std']
        
        # Fill NaN values with global means
        column_mapping = {
            'sst_std': 'sst',
            'chl_std': 'chlor_a',
            'pp_std': 'primary_productivity',
            'ssh_std': 'ssh_anomaly'
        }
        
        for col in ['sst_std', 'chl_std', 'pp_std', 'ssh_std']:
            self.location_patterns[col] = self.location_patterns[col].fillna(
                self.training_data[column_mapping[col]].std()
            )
        
        # Create temporal patterns
        self.temporal_patterns = self.training_data.groupby(['month', 'hour']).agg({
            'sst': ['mean', 'std'],
            'chlor_a': ['mean', 'std'],
            'primary_productivity': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        self.temporal_patterns.columns = ['month', 'hour',
                                        'sst_temp_mean', 'sst_temp_std',
                                        'chl_temp_mean', 'chl_temp_std',
                                        'pp_temp_mean', 'pp_temp_std']
        
        # Build spatial interpolator for unknown locations
        self._build_spatial_interpolator()
        
        logger.info("Built spatial and temporal patterns from training data")
    
    def _build_spatial_interpolator(self):
        """Build spatial interpolator for unknown locations"""
        # Use NearestNeighbors for spatial interpolation
        coords = self.location_patterns[['latitude', 'longitude']].values
        self.spatial_interpolator = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self.spatial_interpolator.fit(coords)
    
    def get_satellite_data(self, latitude: float, longitude: float, 
                          datetime: datetime) -> Dict[str, float]:
        """
        Get satellite-derived environmental data for a specific location and time
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            datetime: Date and time
            
        Returns:
            Dictionary with satellite data values
        """
        try:
            # Check if we have exact location data
            exact_match = self.location_patterns[
                (abs(self.location_patterns['latitude'] - latitude) < 0.01) &
                (abs(self.location_patterns['longitude'] - longitude) < 0.01)
            ]
            
            if len(exact_match) > 0:
                # Use exact location data
                loc_data = exact_match.iloc[0]
                sst_base = loc_data['sst_mean']
                chl_base = loc_data['chl_mean']
                pp_base = loc_data['pp_mean']
                ssh_base = loc_data['ssh_mean']
                
                sst_std = loc_data['sst_std']
                chl_std = loc_data['chl_std']
                pp_std = loc_data['pp_std']
                ssh_std = loc_data['ssh_std']
                
            else:
                # Interpolate from nearby locations
                sst_base, chl_base, pp_base, ssh_base = self._interpolate_spatial_values(
                    latitude, longitude
                )
                
                # Use average standard deviations
                sst_std = self.location_patterns['sst_std'].mean()
                chl_std = self.location_patterns['chl_std'].mean()
                pp_std = self.location_patterns['pp_std'].mean()
                ssh_std = self.location_patterns['ssh_std'].mean()
            
            # Apply temporal variations
            month = datetime.month
            hour = datetime.hour
            
            # Get temporal patterns
            temp_match = self.temporal_patterns[
                (self.temporal_patterns['month'] == month) &
                (self.temporal_patterns['hour'] == hour)
            ]
            
            if len(temp_match) > 0:
                temp_data = temp_match.iloc[0]
                sst_temp_factor = temp_data['sst_temp_mean'] / self.training_data['sst'].mean()
                chl_temp_factor = temp_data['chl_temp_mean'] / self.training_data['chlor_a'].mean()
                pp_temp_factor = temp_data['pp_temp_mean'] / self.training_data['primary_productivity'].mean()
            else:
                # Use seasonal patterns
                sst_temp_factor = self._get_seasonal_factor(month, 'sst')
                chl_temp_factor = self._get_seasonal_factor(month, 'chlor_a')
                pp_temp_factor = self._get_seasonal_factor(month, 'primary_productivity')
            
            # Apply temporal adjustments
            sst = sst_base * sst_temp_factor
            chl = chl_base * chl_temp_factor
            pp = pp_base * pp_temp_factor
            
            # Add some realistic noise based on standard deviations
            sst += np.random.normal(0, sst_std * 0.3)
            chl += np.random.normal(0, chl_std * 0.3)
            pp += np.random.normal(0, pp_std * 0.3)
            ssh_anomaly = ssh_base + np.random.normal(0, ssh_std * 0.3)
            
            # Ensure realistic bounds
            sst = np.clip(sst, -2, 35)  # SST bounds
            chl = np.clip(chl, 0.01, 10)  # Chlorophyll bounds
            pp = np.clip(pp, 0.01, 5)  # Primary productivity bounds
            ssh_anomaly = np.clip(ssh_anomaly, -0.5, 0.5)  # SSH anomaly bounds
            
            return {
                'sst': float(sst),
                'chlorophyll_a': float(chl),
                'primary_productivity': float(pp),
                'ssh_anomaly': float(ssh_anomaly)
            }
            
        except Exception as e:
            logger.error(f"Error getting satellite data: {e}")
            # Return fallback values
            return {
                'sst': 20.0,
                'chlorophyll_a': 0.5,
                'primary_productivity': 0.5,
                'ssh_anomaly': 0.0
            }
    
    def _interpolate_spatial_values(self, latitude: float, longitude: float) -> Tuple[float, float, float, float]:
        """Interpolate spatial values for unknown locations"""
        try:
            # Find nearest neighbors
            distances, indices = self.spatial_interpolator.kneighbors([[latitude, longitude]])
            
            # Weighted average based on distance
            weights = 1.0 / (distances[0] + 1e-6)  # Add small value to avoid division by zero
            weights = weights / weights.sum()
            
            # Get values from nearest neighbors
            neighbor_data = self.location_patterns.iloc[indices[0]]
            
            sst = (neighbor_data['sst_mean'] * weights).sum()
            chl = (neighbor_data['chl_mean'] * weights).sum()
            pp = (neighbor_data['pp_mean'] * weights).sum()
            ssh = (neighbor_data['ssh_mean'] * weights).sum()
            
            return sst, chl, pp, ssh
            
        except Exception as e:
            logger.error(f"Error in spatial interpolation: {e}")
            # Return global means
            return (
                self.training_data['sst'].mean(),
                self.training_data['chlor_a'].mean(),
                self.training_data['primary_productivity'].mean(),
                self.training_data['ssh_anomaly'].mean()
            )
    
    def _get_seasonal_factor(self, month: int, variable: str) -> float:
        """Get seasonal factor for a variable based on month"""
        try:
            monthly_data = self.training_data.groupby('month')[variable].mean()
            global_mean = self.training_data[variable].mean()
            
            if month in monthly_data.index:
                return monthly_data[month] / global_mean
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error getting seasonal factor: {e}")
            return 1.0
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about the satellite data patterns"""
        if self.training_data is None:
            return {}
        
        return {
            'sst': {
                'mean': float(self.training_data['sst'].mean()),
                'std': float(self.training_data['sst'].std()),
                'min': float(self.training_data['sst'].min()),
                'max': float(self.training_data['sst'].max())
            },
            'chlorophyll_a': {
                'mean': float(self.training_data['chlor_a'].mean()),
                'std': float(self.training_data['chlor_a'].std()),
                'min': float(self.training_data['chlor_a'].min()),
                'max': float(self.training_data['chlor_a'].max())
            },
            'primary_productivity': {
                'mean': float(self.training_data['primary_productivity'].mean()),
                'std': float(self.training_data['primary_productivity'].std()),
                'min': float(self.training_data['primary_productivity'].min()),
                'max': float(self.training_data['primary_productivity'].max())
            },
            'ssh_anomaly': {
                'mean': float(self.training_data['ssh_anomaly'].mean()),
                'std': float(self.training_data['ssh_anomaly'].std()),
                'min': float(self.training_data['ssh_anomaly'].min()),
                'max': float(self.training_data['ssh_anomaly'].max())
            }
        }

# Global instance
satellite_service = None

def get_satellite_service() -> SatelliteDataService:
    """Get the global satellite service instance"""
    global satellite_service
    if satellite_service is None:
        satellite_service = SatelliteDataService()
    return satellite_service
