"""
Simplified Data Preprocessing Pipeline for Sharks from Space Challenge
Handles the data integration and preparation more efficiently
"""

import pandas as pd
import numpy as np
import h5py
import xarray as xr
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import json

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataPreprocessor:
    """
    Simplified data preprocessing for shark habitat prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def load_shark_data(self, file_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load and filter shark tracking data
        
        Args:
            file_path: Path to shark data CSV
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            
        Returns:
            Processed shark DataFrame
        """
        logger.info(f"Loading shark data from {file_path}")
        
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        logger.info(f"Original shark data: {len(df)} records")
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"Unique sharks: {df['name'].nunique()}")
        
        # Filter by date range if specified
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
            logger.info(f"Filtered to {len(df)} records for {start_date} to {end_date}")
        
        # Basic validation
        valid_coords = (
            (df['latitude'] >= -90) & (df['latitude'] <= 90) &
            (df['longitude'] >= -180) & (df['longitude'] <= 180)
        )
        df = df[valid_coords].copy()
        
        # Remove duplicates
        df = df.drop_duplicates().reset_index(drop=True)
        
        logger.info(f"Final shark data: {len(df)} records")
        
        return df
    
    def create_synthetic_satellite_data(self, shark_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic satellite data based on shark locations and time
        
        Args:
            shark_df: Shark tracking DataFrame
            
        Returns:
            DataFrame with synthetic satellite data
        """
        logger.info("Creating synthetic satellite data")
        
        df = shark_df.copy()
        
        # Create synthetic SST based on latitude and season
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Base SST from latitude (colder at poles, warmer at equator)
        base_sst = 30 - 0.5 * np.abs(df['latitude'])
        
        # Seasonal variation
        seasonal_variation = 5 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        
        # Add some noise
        noise = np.random.normal(0, 1, len(df))
        
        df['sst'] = base_sst + seasonal_variation + noise
        
        # Create synthetic chlorophyll based on SST and location
        # Higher chlorophyll in colder waters and near coast
        df['distance_to_coast'] = np.sqrt(
            (df['latitude'] - 0)**2 + (df['longitude'] - 0)**2
        ) * 111  # Rough conversion to km
        
        base_chl = 0.5 + 0.3 * np.exp(-df['distance_to_coast'] / 1000)  # Higher near coast
        sst_effect = np.maximum(0, 20 - df['sst']) / 20  # Higher in colder waters
        
        df['chlor_a'] = base_chl * (1 + sst_effect) + np.random.normal(0, 0.1, len(df))
        df['chlor_a'] = np.maximum(0.01, df['chlor_a'])  # Ensure positive values
        
        # Create primary productivity
        df['primary_productivity'] = df['chlor_a'] * (df['sst'] / 20)
        
        # Create SSH anomaly (simplified)
        df['ssh_anomaly'] = np.random.normal(0, 0.05, len(df))
        
        logger.info("Synthetic satellite data created")
        
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
    
    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating engineered features")
        
        # Temporal features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Spatial features
        df['distance_to_equator'] = np.abs(df['latitude'])
        df['distance_to_prime_meridian'] = np.abs(df['longitude'])
        
        # Oceanographic features
        if 'sst' in df.columns:
            df['sst_anomaly'] = df['sst'] - df['sst'].median()
        
        if 'chlor_a' in df.columns:
            df['chl_anomaly'] = df['chlor_a'] - df['chlor_a'].median()
        
        if 'primary_productivity' in df.columns:
            df['pp_anomaly'] = df['primary_productivity'] - df['primary_productivity'].median()
        
        # Shark-specific features
        if 'weight' in df.columns and 'length' in df.columns:
            df['weight_length_ratio'] = df['weight'] / (df['length'] + 1e-6)
        
        # Movement features (simplified)
        df['movement_speed'] = 0.0  # Placeholder
        
        logger.info(f"Created {len(df.columns)} features")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Prepare data for machine learning training
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing prepared data
        """
        logger.info("Preparing training data")
        
        # Define feature columns (exclude target and metadata)
        exclude_columns = [
            'datetime', 'name', 'id', 'tagDate', 'active', 'dist_total',
            'species', 'gender', 'weight', 'length'
        ]
        
        # Get all numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numerical_cols if col not in exclude_columns]
        
        # Ensure target column is included if it exists
        target_column = 'foraging_behavior'
        if target_column in df.columns and target_column not in feature_columns:
            feature_columns.append(target_column)
        
        logger.info(f"Selected {len(feature_columns)} features for training")
        
        # Create feature DataFrame
        feature_df = df[feature_columns].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.median())
        
        # Separate features and target
        if target_column in feature_df.columns:
            X = feature_df.drop(columns=[target_column])
            y = feature_df[target_column]
        else:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info(f"Training set: {len(X_train_scaled)} samples")
        logger.info(f"Test set: {len(X_test_scaled)} samples")
        logger.info(f"Features: {len(X_train_scaled.columns)}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': X_train_scaled.columns.tolist(),
            'target_column': target_column
        }
    
    def save_prepared_data(self, prepared_data: Dict, output_dir: str = "training_data"):
        """
        Save prepared training data
        
        Args:
            prepared_data: Dictionary containing prepared data
            output_dir: Directory to save data
        """
        logger.info(f"Saving prepared data to {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save data
        prepared_data['X_train'].to_csv(f"{output_dir}/X_train.csv", index=False)
        prepared_data['X_test'].to_csv(f"{output_dir}/X_test.csv", index=False)
        prepared_data['y_train'].to_csv(f"{output_dir}/y_train.csv", index=False)
        prepared_data['y_test'].to_csv(f"{output_dir}/y_test.csv", index=False)
        
        # Save metadata
        metadata = {
            'feature_columns': prepared_data['feature_columns'],
            'target_column': prepared_data['target_column'],
            'n_features': len(prepared_data['feature_columns']),
            'n_train_samples': len(prepared_data['X_train']),
            'n_test_samples': len(prepared_data['X_test']),
            'target_distribution_train': prepared_data['y_train'].value_counts().to_dict(),
            'target_distribution_test': prepared_data['y_test'].value_counts().to_dict()
        }
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Data saved successfully")
        
        return metadata

def main():
    """Main function to run the simplified preprocessing pipeline"""
    logger.info("Starting simplified data preprocessing pipeline")
    
    # Initialize preprocessor
    preprocessor = SimpleDataPreprocessor()
    
    # Load shark data for a specific time period
    shark_df = preprocessor.load_shark_data(
        file_path='sharks_cleaned.csv',
        start_date='2014-02-01',
        end_date='2014-02-07'
    )
    
    # Create synthetic satellite data
    integrated_df = preprocessor.create_synthetic_satellite_data(shark_df)
    
    # Create foraging labels
    integrated_df = preprocessor.create_foraging_labels(integrated_df)
    
    # Create engineered features
    integrated_df = preprocessor.create_engineered_features(integrated_df)
    
    # Save integrated data
    integrated_df.to_csv('integrated_data_simple.csv', index=False)
    logger.info(f"Saved integrated data: {len(integrated_df)} records")
    
    # Prepare training data
    prepared_data = preprocessor.prepare_training_data(integrated_df)
    
    # Save prepared data
    metadata = preprocessor.save_prepared_data(prepared_data)
    
    # Print summary
    print(f"\n=== SIMPLIFIED DATA PREPROCESSING SUMMARY ===")
    print(f"Total records: {len(integrated_df):,}")
    print(f"Date range: {integrated_df['datetime'].min()} to {integrated_df['datetime'].max()}")
    print(f"Unique sharks: {integrated_df['name'].nunique()}")
    print(f"Training samples: {metadata['n_train_samples']:,}")
    print(f"Test samples: {metadata['n_test_samples']:,}")
    print(f"Features: {metadata['n_features']}")
    
    print(f"\n=== TARGET DISTRIBUTION ===")
    print("Training set:")
    for label, count in metadata['target_distribution_train'].items():
        print(f"  {label}: {count} ({count/metadata['n_train_samples']*100:.1f}%)")
    
    print("Test set:")
    for label, count in metadata['target_distribution_test'].items():
        print(f"  {label}: {count} ({count/metadata['n_test_samples']*100:.1f}%)")
    
    print(f"\n=== FEATURES ===")
    for i, col in enumerate(metadata['feature_columns'], 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n=== FILES CREATED ===")
    print("  - integrated_data_simple.csv (Integrated data)")
    print("  - training_data/X_train.csv (Training features)")
    print("  - training_data/X_test.csv (Test features)")
    print("  - training_data/y_train.csv (Training targets)")
    print("  - training_data/y_test.csv (Test targets)")
    print("  - training_data/metadata.json (Data metadata)")
    
    print(f"\n=== DATA PREPARATION COMPLETE ===")
    print("Data is now ready for machine learning training!")

if __name__ == "__main__":
    main()

