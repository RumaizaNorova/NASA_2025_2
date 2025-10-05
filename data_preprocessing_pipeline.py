"""
Comprehensive Data Preprocessing Pipeline for Sharks from Space Challenge
Handles L3 binned NASA data and integrates with shark tracking data
"""

import pandas as pd
import numpy as np
import h5py
import xarray as xr
import netCDF4
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import os
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class L3BinnedDataProcessor:
    """
    Process NASA L3 binned data format
    """
    
    def __init__(self):
        self.data_dir = Path("nasa_data")
        
    def read_l3_binned_data(self, file_path: str) -> Dict:
        """
        Read L3 binned data from NetCDF file
        
        Args:
            file_path: Path to L3 binned NetCDF file
            
        Returns:
            Dictionary containing binned data
        """
        logger.info(f"Reading L3 binned data from {file_path}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Access the level-3_binned_data group
                data_group = f['level-3_binned_data']
                
                # Extract bin information
                bin_list = data_group['BinList'][:]  # (n_bins, 2) - [lat, lon] pairs
                bin_index = data_group['BinIndex'][:]  # (n_time_steps,) - indices into bin_list
                
                # Extract data variables
                data_vars = {}
                for var_name in ['sst', 'chlor_a']:  # Common variable names
                    if var_name in data_group:
                        data_vars[var_name] = data_group[var_name][:]
                    elif var_name == 'sst' and 'sst' in data_group:
                        data_vars['sst'] = data_group['sst'][:]
                    elif var_name == 'chlor_a' and 'chlor_a' in data_group:
                        data_vars['chlor_a'] = data_group['chlor_a'][:]
                
                # Get global attributes
                attrs = dict(f.attrs)
                
                logger.info(f"Loaded {len(bin_list)} bins with {len(bin_index)} time steps")
                logger.info(f"Data variables: {list(data_vars.keys())}")
                
                return {
                    'bin_list': bin_list,
                    'bin_index': bin_index,
                    'data_vars': data_vars,
                    'attrs': attrs,
                    'file_path': file_path
                }
                
        except Exception as e:
            logger.error(f"Error reading L3 binned data: {e}")
            raise
    
    def convert_binned_to_dataframe(self, binned_data: Dict) -> pd.DataFrame:
        """
        Convert L3 binned data to DataFrame format
        
        Args:
            binned_data: Dictionary from read_l3_binned_data
            
        Returns:
            DataFrame with columns: lat, lon, time_index, data_vars...
        """
        logger.info("Converting binned data to DataFrame")
        
        bin_list = binned_data['bin_list']
        bin_index = binned_data['bin_index']
        data_vars = binned_data['data_vars']
        
        # Create time index (simplified - assumes daily data)
        # In reality, you'd need to parse the actual time information
        time_coverage_start = binned_data['attrs'].get('time_coverage_start', b'2002-07-04T00:55:01.000Z')
        if isinstance(time_coverage_start, bytes):
            time_coverage_start = time_coverage_start.decode('utf-8')
        
        start_date = pd.to_datetime(time_coverage_start)
        time_indices = [start_date + timedelta(days=i) for i in range(len(bin_index))]
        
        # Create DataFrame
        records = []
        for time_idx, time_val in enumerate(time_indices):
            if time_idx < len(bin_index):
                # Handle numpy.void objects by converting to int
                bin_start = int(bin_index[time_idx]) if time_idx == 0 else int(bin_index[time_idx-1])
                bin_end = int(bin_index[time_idx])
                
                # Ensure valid range
                bin_start = max(0, bin_start)
                bin_end = min(len(bin_list), bin_end)
                
                for bin_idx in range(bin_start, bin_end):
                    if bin_idx < len(bin_list):
                        record = {
                            'latitude': float(bin_list[bin_idx, 0]),
                            'longitude': float(bin_list[bin_idx, 1]),
                            'datetime': time_val,
                            'time_index': time_idx
                        }
                        
                        # Add data variables
                        for var_name, var_data in data_vars.items():
                            if bin_idx < len(var_data):
                                record[var_name] = float(var_data[bin_idx]) if not np.isnan(var_data[bin_idx]) else np.nan
                        
                        records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Created DataFrame with {len(df)} records")
        
        return df

class SharkDataProcessor:
    """
    Process shark tracking data
    """
    
    def __init__(self):
        pass
    
    def load_shark_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess shark tracking data
        
        Args:
            file_path: Path to shark data CSV
            
        Returns:
            Processed shark DataFrame
        """
        logger.info(f"Loading shark data from {file_path}")
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Basic data quality checks
        logger.info(f"Original shark data: {len(df)} records")
        
        # Remove invalid coordinates
        valid_coords = (
            (df['latitude'] >= -90) & (df['latitude'] <= 90) &
            (df['longitude'] >= -180) & (df['longitude'] <= 180)
        )
        df = df[valid_coords].copy()
        
        logger.info(f"After coordinate validation: {len(df)} records")
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def create_foraging_labels(self, df: pd.DataFrame, 
                             residency_threshold: float = 0.01,
                             min_points: int = 10) -> pd.DataFrame:
        """
        Create foraging behavior labels based on movement patterns
        
        Args:
            df: Shark DataFrame
            residency_threshold: Distance threshold for residency (degrees)
            min_points: Minimum points to consider as foraging area
            
        Returns:
            DataFrame with foraging labels
        """
        logger.info("Creating foraging behavior labels")
        
        foraging_labels = []
        
        for idx, row in df.iterrows():
            # Find nearby points within residency threshold
            nearby_points = df[
                (abs(df['latitude'] - row['latitude']) < residency_threshold) &
                (abs(df['longitude'] - row['longitude']) < residency_threshold)
            ]
            
            # If there are many nearby points, likely foraging
            if len(nearby_points) >= min_points:
                foraging_labels.append(1)
            else:
                foraging_labels.append(0)
        
        df['foraging_behavior'] = foraging_labels
        
        foraging_count = sum(foraging_labels)
        logger.info(f"Created foraging labels: {foraging_count} foraging, {len(foraging_labels) - foraging_count} not foraging")
        
        return df

class DataIntegrator:
    """
    Integrate shark data with NASA satellite data
    """
    
    def __init__(self):
        self.l3_processor = L3BinnedDataProcessor()
        self.shark_processor = SharkDataProcessor()
        
    def extract_satellite_values_at_shark_locations(self, 
                                                   satellite_df: pd.DataFrame,
                                                   shark_df: pd.DataFrame,
                                                   time_tolerance_hours: int = 24,
                                                   spatial_tolerance_km: float = 10.0) -> pd.DataFrame:
        """
        Extract satellite data values at shark locations
        
        Args:
            satellite_df: Satellite data DataFrame
            shark_df: Shark tracking DataFrame
            time_tolerance_hours: Maximum time difference for matching
            spatial_tolerance_km: Maximum spatial distance for matching (km)
            
        Returns:
            DataFrame with satellite data at shark locations
        """
        logger.info("Extracting satellite values at shark locations")
        
        # Convert spatial tolerance to degrees (rough approximation)
        spatial_tolerance_deg = spatial_tolerance_km / 111.0  # 1 degree ≈ 111 km
        
        # Initialize result DataFrame
        result_df = shark_df.copy()
        
        # Add satellite data columns
        satellite_cols = [col for col in satellite_df.columns if col not in ['latitude', 'longitude', 'datetime']]
        for col in satellite_cols:
            result_df[col] = np.nan
        
        # For each shark location, find nearest satellite data
        matched_count = 0
        
        for idx, shark_row in shark_df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processing shark location {idx}/{len(shark_df)}")
            
            # Find satellite data within time and spatial tolerance
            time_diff = abs(satellite_df['datetime'] - shark_row['datetime'])
            spatial_diff = np.sqrt(
                (satellite_df['latitude'] - shark_row['latitude'])**2 +
                (satellite_df['longitude'] - shark_row['longitude'])**2
            )
            
            # Filter by tolerance
            valid_mask = (
                (time_diff <= timedelta(hours=time_tolerance_hours)) &
                (spatial_diff <= spatial_tolerance_deg)
            )
            
            valid_satellite = satellite_df[valid_mask]
            
            if len(valid_satellite) > 0:
                # Find closest match
                distances = np.sqrt(
                    (valid_satellite['latitude'] - shark_row['latitude'])**2 +
                    (valid_satellite['longitude'] - shark_row['longitude'])**2
                )
                closest_idx = distances.idxmin()
                closest_satellite = valid_satellite.loc[closest_idx]
                
                # Assign satellite values
                for col in satellite_cols:
                    if col in closest_satellite and not pd.isna(closest_satellite[col]):
                        result_df.at[idx, col] = closest_satellite[col]
                
                matched_count += 1
        
        logger.info(f"Matched {matched_count}/{len(shark_df)} shark locations with satellite data")
        
        return result_df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from the integrated data
        
        Args:
            df: Integrated DataFrame
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Creating derived features")
        
        # Distance to coast (simplified calculation)
        df['distance_to_coast'] = np.sqrt(
            (df['latitude'] - 0)**2 + (df['longitude'] - 0)**2
        ) * 111  # Rough conversion to km
        
        # Primary productivity index
        if 'sst' in df.columns and 'chlor_a' in df.columns:
            # Handle missing values
            sst_valid = df['sst'].fillna(df['sst'].median())
            chl_valid = df['chlor_a'].fillna(df['chlor_a'].median())
            df['primary_productivity'] = chl_valid * (sst_valid / 20.0)
        else:
            logger.warning("Missing SST or chlorophyll data for primary productivity calculation")
            df['primary_productivity'] = 0.0
        
        # Temperature gradient (if we have multiple SST measurements)
        if 'sst' in df.columns:
            df['sst_anomaly'] = df['sst'] - df['sst'].median()
        
        # Temporal features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        
        # Movement features (if we have multiple points per shark)
        df['movement_speed'] = 0.0  # Placeholder - would need to calculate from consecutive points
        
        return df
    
    def integrate_all_data(self, 
                          shark_data_path: str,
                          nasa_data_paths: Dict[str, str],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Integrate all data sources
        
        Args:
            shark_data_path: Path to shark data CSV
            nasa_data_paths: Dictionary mapping data types to file paths
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            output_path: Path to save integrated data
            
        Returns:
            Integrated DataFrame
        """
        logger.info("Starting comprehensive data integration")
        
        # Load shark data
        shark_df = self.shark_processor.load_shark_data(shark_data_path)
        
        # Filter shark data by date range if specified
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            shark_df = shark_df[
                (shark_df['datetime'] >= start_dt) &
                (shark_df['datetime'] <= end_dt)
            ].copy()
            logger.info(f"Filtered shark data to {len(shark_df)} records")
        
        # Load and process NASA data
        satellite_dataframes = []
        
        for data_type, file_path in nasa_data_paths.items():
            if os.path.exists(file_path):
                logger.info(f"Processing {data_type} data from {file_path}")
                
                # Read L3 binned data
                binned_data = self.l3_processor.read_l3_binned_data(file_path)
                
                # Convert to DataFrame
                sat_df = self.l3_processor.convert_binned_to_dataframe(binned_data)
                
                # Filter by date range if specified
                if start_date and end_date:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    sat_df = sat_df[
                        (sat_df['datetime'] >= start_dt) &
                        (sat_df['datetime'] <= end_dt)
                    ].copy()
                
                satellite_dataframes.append(sat_df)
                logger.info(f"Loaded {len(sat_df)} satellite records for {data_type}")
            else:
                logger.warning(f"NASA data file not found: {file_path}")
        
        # Combine satellite data
        if satellite_dataframes:
            # Merge all satellite DataFrames
            combined_satellite = satellite_dataframes[0]
            for sat_df in satellite_dataframes[1:]:
                combined_satellite = pd.merge(
                    combined_satellite, sat_df, 
                    on=['latitude', 'longitude', 'datetime'], 
                    how='outer', suffixes=('', '_dup')
                )
            
            logger.info(f"Combined satellite data: {len(combined_satellite)} records")
            
            # Extract satellite values at shark locations
            integrated_df = self.extract_satellite_values_at_shark_locations(
                combined_satellite, shark_df
            )
        else:
            logger.warning("No satellite data available, using shark data only")
            integrated_df = shark_df.copy()
        
        # Create foraging labels
        integrated_df = self.shark_processor.create_foraging_labels(integrated_df)
        
        # Create derived features
        integrated_df = self.create_derived_features(integrated_df)
        
        # Save integrated data
        if output_path:
            integrated_df.to_csv(output_path, index=False)
            logger.info(f"Saved integrated data to {output_path}")
        
        logger.info(f"Data integration complete: {len(integrated_df)} records")
        
        return integrated_df

def main():
    """Main function to run the preprocessing pipeline"""
    logger.info("Starting comprehensive data preprocessing pipeline")
    
    # Initialize integrator
    integrator = DataIntegrator()
    
    # Define NASA data paths
    nasa_data_paths = {
        'sst': 'nasa_data/modis_aqua/modis_aqua_sst.nc',
        'chlorophyll': 'nasa_data/modis_aqua/modis_aqua_chlorophyll.nc',
        'terra_sst': 'nasa_data/modis_terra/modis_terra_sst.nc'
    }
    
    # Integrate data for a specific time period (e.g., 2014-02-01 to 2014-02-07)
    integrated_data = integrator.integrate_all_data(
        shark_data_path='sharks_cleaned.csv',
        nasa_data_paths=nasa_data_paths,
        start_date='2014-02-01',
        end_date='2014-02-07',
        output_path='integrated_data_preprocessed.csv'
    )
    
    # Print summary
    print(f"\n=== DATA PREPROCESSING SUMMARY ===")
    print(f"Total records: {len(integrated_data):,}")
    print(f"Date range: {integrated_data['datetime'].min()} to {integrated_data['datetime'].max()}")
    print(f"Unique sharks: {integrated_data['name'].nunique()}")
    print(f"Columns: {list(integrated_data.columns)}")
    
    # Check data quality
    print(f"\n=== DATA QUALITY ===")
    missing_values = integrated_data.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  {col}: {count} ({count/len(integrated_data)*100:.1f}%)")
    else:
        print("No missing values found")
    
    # Check foraging behavior distribution
    if 'foraging_behavior' in integrated_data.columns:
        foraging_counts = integrated_data['foraging_behavior'].value_counts()
        print(f"\nForaging behavior distribution:")
        print(f"  Foraging: {foraging_counts.get(1, 0)} ({foraging_counts.get(1, 0)/len(integrated_data)*100:.1f}%)")
        print(f"  Not foraging: {foraging_counts.get(0, 0)} ({foraging_counts.get(0, 0)/len(integrated_data)*100:.1f}%)")
    
    # Check satellite data availability
    satellite_cols = ['sst', 'chlor_a', 'primary_productivity']
    print(f"\nSatellite data availability:")
    for col in satellite_cols:
        if col in integrated_data.columns:
            non_null_count = integrated_data[col].notna().sum()
            print(f"  {col}: {non_null_count} ({non_null_count/len(integrated_data)*100:.1f}%)")
    
    print(f"\n=== PREPROCESSING COMPLETE ===")
    print("Data is now ready for training!")

if __name__ == "__main__":
    main()
