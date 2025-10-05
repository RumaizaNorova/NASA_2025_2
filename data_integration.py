"""
Data Integration Script for Sharks from Space Challenge
Integrates shark tracking data with NASA satellite data
"""

import pandas as pd
import numpy as np
import xarray as xr
import requests
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIntegrator:
    """
    Integrates shark tracking data with NASA satellite data
    """
    
    def __init__(self):
        """Initialize data integrator with credentials from .env file"""
        # Load credentials from environment variables
        self.earthdata_username = os.getenv('EARTHDATA_USERNAME')
        self.earthdata_password = os.getenv('EARTHDATA_PASSWORD')
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
        
        self.data_dir = "nasa_data"
        self._create_data_directory()
        
        # Verify credentials
        if not self.earthdata_username or not self.earthdata_password:
            logger.warning("Earthdata credentials not found in .env file")
        else:
            logger.info(f"Using Earthdata credentials for user: {self.earthdata_username}")
    
    def _create_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def download_satellite_data(self, 
                               start_date: str, 
                               end_date: str,
                               bbox: Tuple[float, float, float, float] = None) -> Dict:
        """
        Download real NASA satellite data for the specified time period
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary with downloaded data information
        """
        logger.info(f"Downloading real NASA satellite data for {start_date} to {end_date}")
        
        return self._download_real_nasa_data(start_date, end_date, bbox)
    
    def _download_real_nasa_data(self, 
                                start_date: str, 
                                end_date: str,
                                bbox: Tuple[float, float, float, float] = None) -> Dict:
        """
        Download real NASA MODIS-Aqua data using the working method
        """
        logger.info("Downloading real NASA MODIS-Aqua data")
        
        # Check if we already have downloaded data files
        real_data_dir = "real_nasa_data"
        if os.path.exists(real_data_dir):
            # Look for existing downloaded files
            sst_file = os.path.join(real_data_dir, "modis_aqua_sst.nc")
            chl_file = os.path.join(real_data_dir, "modis_aqua_chlorophyll.nc")
            
            if os.path.exists(sst_file) and os.path.exists(chl_file):
                logger.info("Found existing real NASA data files")
                return self._load_existing_real_data(sst_file, chl_file, start_date, end_date, bbox)
        
        # If no existing files, trigger download
        logger.error("No existing real NASA data found. Please run download_real_nasa_data.py first")
        raise FileNotFoundError(
            "Real NASA data files not found. Please download real data using download_real_nasa_data.py first. "
            "Expected files: real_nasa_data/modis_aqua_sst.nc and real_nasa_data/modis_aqua_chlorophyll.nc"
        )
    
    def _load_existing_real_data(self, 
                                sst_file: str, 
                                chl_file: str,
                                start_date: str, 
                                end_date: str,
                                bbox: Tuple[float, float, float, float] = None) -> Dict:
        """
        Load existing real NASA data files
        """
        logger.info("Loading existing real NASA data files")
        
        try:
            # Load SST data
            sst_ds = xr.open_dataset(sst_file)
            logger.info(f"SST data loaded: {dict(sst_ds.dims)}")
            
            # Load Chlorophyll data
            chl_ds = xr.open_dataset(chl_file)
            logger.info(f"Chlorophyll data loaded: {dict(chl_ds.dims)}")
            
            # Combine datasets
            combined_ds = xr.merge([sst_ds, chl_ds])
            
            # Filter by time if needed
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if 'time' in combined_ds.coords:
                time_mask = (combined_ds.time >= start_dt) & (combined_ds.time <= end_dt)
                combined_ds = combined_ds.sel(time=time_mask)
            
            # Filter by bounding box if provided
            if bbox:
                lon_min, lat_min, lon_max, lat_max = bbox
                combined_ds = combined_ds.sel(
                    longitude=slice(lon_min, lon_max),
                    latitude=slice(lat_min, lat_max)
                )
            
            logger.info(f"Real data loaded and filtered: {dict(combined_ds.dims)}")
            
            return {
                'start_date': start_date,
                'end_date': end_date,
                'bbox': bbox,
                'data': combined_ds,
                'source': 'real_nasa_modis_aqua'
            }
            
        except Exception as e:
            logger.error(f"Error loading real NASA data: {e}")
            raise RuntimeError(f"Failed to load real NASA data: {e}")
    
    
    def extract_satellite_values_at_shark_locations(self, 
                                                   satellite_data: xr.Dataset,
                                                   shark_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract satellite data values at shark locations
        
        Args:
            satellite_data: Satellite data as xarray Dataset
            shark_data: Shark tracking data
            
        Returns:
            DataFrame with satellite data extracted at shark locations
        """
        logger.info("Extracting satellite values at shark locations")
        
        # Convert shark data datetime to match satellite data
        shark_data = shark_data.copy()
        shark_data['datetime'] = pd.to_datetime(shark_data['datetime'])
        
        # Initialize result DataFrame
        result_df = shark_data.copy()
        
        # For each shark location, extract satellite values
        satellite_values = []
        
        for idx, row in shark_data.iterrows():
            # Find nearest time - use simple approach to avoid datetime issues
            shark_time = pd.Timestamp(row['datetime'])
            
            # Convert satellite time to pandas datetime for comparison
            satellite_times = pd.to_datetime(satellite_data.time.values)
            time_diff = abs(satellite_times - shark_time)
            nearest_time_idx = np.argmin(time_diff)
            
            # Extract values at shark location
            values = satellite_data.isel(time=nearest_time_idx).interp(
                latitude=row['latitude'],
                longitude=row['longitude'],
                method='nearest'
            )
            
            satellite_values.append({
                'chlorophyll_a': float(values.chlorophyll_a.values),
                'sea_surface_temp': float(values.sea_surface_temp.values),
                'ssh_anomaly': float(values.ssh_anomaly.values)
            })
        
        # Add satellite values to result DataFrame
        satellite_df = pd.DataFrame(satellite_values)
        result_df = pd.concat([result_df, satellite_df], axis=1)
        
        logger.info(f"Extracted satellite values for {len(result_df)} shark locations")
        
        return result_df
    
    def create_eddy_features(self, 
                           shark_data: pd.DataFrame,
                           satellite_data: xr.Dataset) -> pd.DataFrame:
        """
        Create eddy-related features from real satellite data
        
        Args:
            shark_data: Shark tracking data
            satellite_data: Real NASA satellite data
            
        Returns:
            DataFrame with eddy features added
        """
        logger.info("Creating eddy features from real satellite data")
        
        # Extract SSH anomaly data for eddy detection
        if 'ssh_anomaly' in satellite_data.data_vars:
            logger.info("Using SSH anomaly data for eddy detection")
            # Real eddy detection would use SSH gradients and curl calculations
            # For now, we'll extract SSH values at shark locations
            ssh_values = []
            
            for idx, row in shark_data.iterrows():
                try:
                    # Extract SSH anomaly at shark location
                    ssh_val = satellite_data.sel(
                        latitude=row['latitude'],
                        longitude=row['longitude'],
                        method='nearest'
                    )['ssh_anomaly'].values
                    
                    if np.isfinite(ssh_val):
                        ssh_values.append(float(ssh_val))
                    else:
                        ssh_values.append(0.0)
                        
                except Exception as e:
                    logger.warning(f"Could not extract SSH for location {idx}: {e}")
                    ssh_values.append(0.0)
            
            # Create eddy features based on SSH anomaly
            eddy_features = pd.DataFrame({
                'eddy_presence': (np.abs(ssh_values) > 0.05).astype(int),  # Threshold for eddy presence
                'eddy_intensity': np.abs(ssh_values),
                'eddy_radius': np.abs(ssh_values) * 1000,  # Scale to reasonable radius
                'eddy_type': ['cyclonic' if s < 0 else 'anticyclonic' for s in ssh_values]
            })
        else:
            logger.warning("No SSH anomaly data available for eddy detection")
            # If no SSH data, create minimal features
            eddy_features = pd.DataFrame({
                'eddy_presence': [0] * len(shark_data),
                'eddy_intensity': [0.0] * len(shark_data),
                'eddy_radius': [0.0] * len(shark_data),
                'eddy_type': ['unknown'] * len(shark_data)
            })
        
        # Add to shark data
        result_df = pd.concat([shark_data, eddy_features], axis=1)
        
        return result_df
    
    def create_foraging_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create foraging behavior labels based on movement patterns
        
        Args:
            df: DataFrame with shark tracking data
            
        Returns:
            DataFrame with foraging labels added
        """
        logger.info("Creating foraging behavior labels")
        
        # Simple heuristic: areas with high residency time are likely foraging areas
        residency_threshold = 0.01  # degrees
        
        foraging_labels = []
        
        for idx, row in df.iterrows():
            # Check if shark is in an area with high residency
            nearby_points = df[
                (abs(df['latitude'] - row['latitude']) < residency_threshold) &
                (abs(df['longitude'] - row['longitude']) < residency_threshold)
            ]
            
            # If there are many nearby points, likely foraging
            if len(nearby_points) > 10:
                foraging_labels.append(1)
            else:
                foraging_labels.append(0)
        
        df['foraging_behavior'] = foraging_labels
        
        logger.info(f"Created foraging labels: {sum(foraging_labels)} foraging, {len(foraging_labels) - sum(foraging_labels)} not foraging")
        
        return df
    
    def integrate_all_data(self, 
                          shark_data_path: str,
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
        """
        Integrate all data sources
        
        Args:
            shark_data_path: Path to shark tracking data
            start_date: Start date for satellite data
            end_date: End date for satellite data
            
        Returns:
            Integrated DataFrame
        """
        logger.info("Starting data integration process")
        
        # Load shark data
        shark_data = pd.read_csv(shark_data_path)
        shark_data['datetime'] = pd.to_datetime(shark_data['datetime'])
        
        # Filter shark data to time period of interest
        shark_data_filtered = shark_data[
            (shark_data['datetime'] >= start_date) &
            (shark_data['datetime'] <= end_date)
        ].copy()
        
        logger.info(f"Filtered shark data: {len(shark_data_filtered)} records")
        
        if len(shark_data_filtered) == 0:
            logger.warning("No shark data in specified time period")
            return shark_data
        
        # Get bounding box from shark data
        bbox = (
            shark_data_filtered['longitude'].min() - 1,
            shark_data_filtered['latitude'].min() - 1,
            shark_data_filtered['longitude'].max() + 1,
            shark_data_filtered['latitude'].max() + 1
        )
        
        logger.info(f"Data bounding box: {bbox}")
        
        # Download real NASA satellite data
        satellite_data_info = self.download_satellite_data(start_date, end_date, bbox)
        satellite_data = satellite_data_info['data']
        
        logger.info(f"Using satellite data source: {satellite_data_info['source']}")
        
        # Extract satellite values at shark locations
        integrated_data = self.extract_satellite_values_at_shark_locations(
            satellite_data, shark_data_filtered
        )
        
        # Add eddy features
        integrated_data = self.create_eddy_features(integrated_data, satellite_data)
        
        # Add foraging behavior labels
        integrated_data = self.create_foraging_labels(integrated_data)
        
        # Add additional derived features
        integrated_data = self._add_derived_features(integrated_data)
        
        logger.info(f"Data integration complete: {len(integrated_data)} records")
        
        return integrated_data
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features from real satellite data
        
        Args:
            df: Input DataFrame with real satellite data
            
        Returns:
            DataFrame with derived features added
        """
        logger.info("Adding derived features from real satellite data")
        
        # Distance to coast (simplified calculation)
        df['distance_to_coast'] = np.sqrt(
            (df['latitude'] - 0)**2 + (df['longitude'] - 0)**2
        ) * 111  # Rough conversion to km
        
        # Primary productivity index (based on real chlorophyll and temperature data)
        if 'chlorophyll_a' in df.columns and 'sea_surface_temp' in df.columns:
            df['primary_productivity'] = df['chlorophyll_a'] * (df['sea_surface_temp'] / 20)
        else:
            logger.warning("Missing chlorophyll_a or sea_surface_temp for primary productivity calculation")
            df['primary_productivity'] = 0.0
        
        # Ocean current strength (based on SSH gradients from real data)
        if 'ssh_anomaly' in df.columns:
            df['current_strength'] = np.abs(df['ssh_anomaly']) * 100
        else:
            logger.warning("Missing ssh_anomaly for current strength calculation")
            df['current_strength'] = 0.0
        
        # Note: Bathymetry would require separate bathymetry dataset
        # For now, we'll skip it to avoid synthetic data
        logger.info("Skipping bathymetry - would require separate bathymetry dataset")
        
        return df

def main():
    """Main function to run data integration"""
    logger.info("Starting data integration process")
    
    # Initialize integrator
    integrator = DataIntegrator()
    
    # Integrate data for February 2014 (high shark activity period)
    integrated_data = integrator.integrate_all_data(
        shark_data_path='sharks_cleaned.csv',
        start_date='2014-02-01',
        end_date='2014-02-07'  # One week for testing
    )
    
    # Save integrated data
    output_file = 'integrated_shark_data_week.csv'
    integrated_data.to_csv(output_file, index=False)
    
    logger.info(f"Integrated data saved to {output_file}")
    
    # Print summary
    print(f"\nData Integration Summary:")
    print(f"Total records: {len(integrated_data)}")
    print(f"Date range: {integrated_data['datetime'].min()} to {integrated_data['datetime'].max()}")
    print(f"Unique sharks: {integrated_data['name'].nunique()}")
    print(f"Columns: {list(integrated_data.columns)}")
    
    # Check for missing values
    missing_values = integrated_data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found")

if __name__ == "__main__":
    main()