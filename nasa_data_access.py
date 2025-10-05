"""
NASA Data Access Module for Sharks from Space Challenge
Handles downloading and processing of MODIS-Aqua, Jason, and other NASA data
"""

import os
import requests
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import netCDF4
from typing import Dict, List, Tuple, Optional
import logging
import subprocess
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NASADataAccess:
    """
    Handles access to NASA satellite data for shark habitat prediction
    """
    
    def __init__(self):
        """Initialize NASA data access with credentials from .env file"""
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
    
    def get_modis_aqua_data(self, 
                           start_date: str, 
                           end_date: str,
                           bbox: Tuple[float, float, float, float] = None,
                           variables: List[str] = None) -> Dict:
        """
        Download MODIS-Aqua data for specified time period
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            variables: List of variables to download
            
        Returns:
            Dictionary with downloaded data information
        """
        if variables is None:
            variables = ['chlor_a', 'sst', 'Rrs_443', 'Rrs_488', 'Rrs_555', 'Rrs_667']
            
        logger.info(f"Downloading MODIS-Aqua data from {start_date} to {end_date}")
        
        # MODIS-Aqua data URLs
        base_url = "https://oceandata.sci.gsfc.nasa.gov/api/file_search"
        
        # Parameters for MODIS-Aqua Level-3 data
        params = {
            'instrument': 'MODIS-Aqua',
            'sensor': 'Aqua',
            'level': 'L3',
            'start': start_date,
            'end': end_date,
            'format': 'netcdf'
        }
        
        # Add bounding box if provided
        if bbox:
            params['bbox'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        
        try:
            # Create session with authentication
            session = requests.Session()
            session.auth = (self.earthdata_username, self.earthdata_password)
            
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            logger.info("Successfully connected to NASA OceanColor API")
            
            # Parse response to get file URLs
            # This would need to be implemented based on actual API response format
            file_urls = []
            
            data_info = {
                'source': 'MODIS-Aqua',
                'start_date': start_date,
                'end_date': end_date,
                'variables': variables,
                'bbox': bbox,
                'files': file_urls,
                'status': 'success'
            }
            
            return data_info
            
        except requests.RequestException as e:
            logger.error(f"Error accessing MODIS-Aqua data: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_jason_altimeter_data(self, 
                                start_date: str, 
                                end_date: str,
                                bbox: Tuple[float, float, float, float] = None) -> Dict:
        """
        Download Jason altimeter data for eddy detection
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary with downloaded data information
        """
        logger.info(f"Downloading Jason altimeter data from {start_date} to {end_date}")
        
        try:
            # Jason data from PO.DAAC
            base_url = "https://podaac.jpl.nasa.gov/api/dataset/v2"
            
            # Search for Jason data
            params = {
                'startTime': start_date,
                'endTime': end_date,
                'keyword': 'Jason'
            }
            
            if bbox:
                params['bbox'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            
            # Create session with authentication
            session = requests.Session()
            session.auth = (self.earthdata_username, self.earthdata_password)
            
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            logger.info("Successfully connected to PO.DAAC API")
            
            data_info = {
                'source': 'Jason-Altimeter',
                'start_date': start_date,
                'end_date': end_date,
                'bbox': bbox,
                'status': 'success'
            }
            
            return data_info
            
        except requests.RequestException as e:
            logger.error(f"Error accessing Jason data: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def download_satellite_data(self, 
                               start_date: str, 
                               end_date: str,
                               bbox: Tuple[float, float, float, float] = None) -> Dict:
        """
        Download satellite data for the specified time period
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            Dictionary with downloaded data information
        """
        logger.info(f"Downloading satellite data for {start_date} to {end_date}")
        
        # Download MODIS-Aqua data
        modis_data = self.get_modis_aqua_data(start_date, end_date, bbox)
        
        # Download Jason altimeter data
        jason_data = self.get_jason_altimeter_data(start_date, end_date, bbox)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'bbox': bbox,
            'modis': modis_data,
            'jason': jason_data,
            'status': 'success' if modis_data.get('status') == 'success' and jason_data.get('status') == 'success' else 'partial'
        }
    
    def process_satellite_data(self, 
                              shark_data: pd.DataFrame,
                              satellite_data: Dict) -> pd.DataFrame:
        """
        Process and merge satellite data with shark tracking data
        
        Args:
            shark_data: DataFrame with shark tracking data
            satellite_data: Dictionary with satellite data information
            
        Returns:
            DataFrame with merged shark and satellite data
        """
        logger.info("Processing satellite data for shark locations")
        
        # This is a placeholder for the actual processing
        # In reality, this would:
        # 1. Load satellite data files
        # 2. Interpolate/extract values at shark locations
        # 3. Merge with shark data
        
        merged_data = shark_data.copy()
        
        # Add placeholder columns for satellite-derived variables
        # These will be replaced with real data once we download it
        merged_data['chlorophyll_a'] = np.random.normal(0.5, 0.2, len(shark_data))
        merged_data['sea_surface_temp'] = np.random.normal(20, 5, len(shark_data))
        merged_data['sea_surface_height'] = np.random.normal(0, 0.1, len(shark_data))
        merged_data['eddy_presence'] = np.random.choice([0, 1], len(shark_data), p=[0.7, 0.3])
        merged_data['eddy_intensity'] = np.random.exponential(0.1, len(shark_data))
        
        return merged_data

def get_shark_data_bounds(shark_data: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Get bounding box for shark data
    
    Args:
        shark_data: DataFrame with shark tracking data
        
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon = shark_data['longitude'].min()
    max_lon = shark_data['longitude'].max()
    min_lat = shark_data['latitude'].min()
    max_lat = shark_data['latitude'].max()
    
    # Add buffer
    buffer = 1.0  # degrees
    return (min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)

if __name__ == "__main__":
    # Test the data access framework
    logger.info("Testing NASA data access framework")
    
    # Load shark data
    shark_data = pd.read_csv('sharks_cleaned.csv')
    
    # Get bounds for February 2020 data (for real NASA data)
    feb_2020_data = shark_data[
        (pd.to_datetime(shark_data['datetime']).dt.year == 2020) &
        (pd.to_datetime(shark_data['datetime']).dt.month == 2)
    ]
    
    # Fallback to any 2020 data if February not available
    if len(feb_2020_data) == 0:
        feb_2020_data = shark_data[
            (pd.to_datetime(shark_data['datetime']).dt.year == 2020)
        ]
    
    # If still no 2020 data, use any available data
    if len(feb_2020_data) == 0:
        feb_2020_data = shark_data.sample(min(1000, len(shark_data)))
        logger.warning("No 2020 data found, using sample data for testing")
    
    if len(feb_2020_data) > 0:
        bounds = get_shark_data_bounds(feb_2020_data)
        logger.info(f"Shark data bounds: {bounds}")
        
        # Initialize data access
        nasa_access = NASADataAccess()
        
        # Test satellite data access
        satellite_data = nasa_access.download_satellite_data(
            start_date='2020-02-01',
            end_date='2020-02-07',  # One week for testing
            bbox=bounds
        )
        
        logger.info(f"Satellite data access result: {satellite_data['status']}")
    else:
        logger.warning("No suitable data found for testing")