"""
Prepare Training Data for Shark Habitat Prediction Model
Handles data cleaning, feature engineering, and train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, List
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataPreparator:
    """
    Prepare data for machine learning training
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        self.target_column = 'foraging_behavior'
        
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and perform basic validation on the integrated data
        
        Args:
            data_path: Path to integrated data CSV
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"Unique sharks: {df['name'].nunique()}")
        
        # Basic validation
        required_columns = ['latitude', 'longitude', 'datetime', 'name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate records
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate records, removing them")
            df = df.drop_duplicates().reset_index(drop=True)
        
        # Check coordinate validity
        invalid_coords = (
            (df['latitude'] < -90) | (df['latitude'] > 90) |
            (df['longitude'] < -180) | (df['longitude'] > 180)
        )
        if invalid_coords.sum() > 0:
            logger.warning(f"Found {invalid_coords.sum()} invalid coordinates, removing them")
            df = df[~invalid_coords].reset_index(drop=True)
        
        logger.info(f"After validation: {len(df)} records")
        
        return df
    
    def create_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for training
        
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
        
        # Oceanographic features (if available)
        if 'sst' in df.columns:
            df['sst_anomaly'] = df['sst'] - df['sst'].median()
            df['sst_category'] = pd.cut(df['sst'], bins=5, labels=['very_cold', 'cold', 'moderate', 'warm', 'very_warm'])
        
        if 'chlor_a' in df.columns:
            df['chl_anomaly'] = df['chlor_a'] - df['chlor_a'].median()
            df['chl_category'] = pd.cut(df['chlor_a'], bins=5, labels=['very_low', 'low', 'moderate', 'high', 'very_high'])
        
        # Primary productivity features
        if 'primary_productivity' in df.columns:
            df['pp_anomaly'] = df['primary_productivity'] - df['primary_productivity'].median()
            df['pp_category'] = pd.cut(df['primary_productivity'], bins=5, labels=['very_low', 'low', 'moderate', 'high', 'very_high'])
        
        # Distance to coast (if not already calculated)
        if 'distance_to_coast' not in df.columns:
            df['distance_to_coast'] = np.sqrt(
                (df['latitude'] - 0)**2 + (df['longitude'] - 0)**2
            ) * 111  # Rough conversion to km
        
        # Shark-specific features
        if 'weight' in df.columns and 'length' in df.columns:
            df['weight_length_ratio'] = df['weight'] / (df['length'] + 1e-6)  # Avoid division by zero
        
        # Movement features (if we have multiple points per shark)
        df_sorted = df.sort_values(['name', 'datetime'])
        df_sorted['time_diff'] = df_sorted.groupby('name')['datetime'].diff()
        df_sorted['lat_diff'] = df_sorted.groupby('name')['latitude'].diff()
        df_sorted['lon_diff'] = df_sorted.groupby('name')['longitude'].diff()
        
        # Calculate movement speed (km/h)
        df_sorted['distance_moved'] = np.sqrt(
            (df_sorted['lat_diff'] * 111)**2 + (df_sorted['lon_diff'] * 111 * np.cos(np.radians(df_sorted['latitude'])))**2
        )
        df_sorted['movement_speed'] = df_sorted['distance_moved'] / (df_sorted['time_diff'].dt.total_seconds() / 3600 + 1e-6)
        
        # Add movement features back to original DataFrame
        df = df_sorted.copy()
        
        # Handle infinite values
        df['movement_speed'] = df['movement_speed'].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Created {len(df.columns)} features")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        # Check missing values
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            logger.info(f"Missing values found in {len(missing_cols)} columns:")
            for col, count in missing_cols.items():
                percentage = count / len(df) * 100
                logger.info(f"  {col}: {count} ({percentage:.1f}%)")
        
        # Handle missing values
        df_clean = df.copy()
        
        # For numerical columns, use median imputation
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # For categorical columns, use mode imputation
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
                logger.info(f"Filled {col} missing values with mode: {mode_val}")
        
        # Verify no missing values remain
        remaining_missing = df_clean.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Still have {remaining_missing} missing values after imputation")
        else:
            logger.info("All missing values handled successfully")
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features")
        
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in ['datetime', 'name']:  # Skip datetime and name columns
                # Use label encoding for categorical features
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                
                logger.info(f"Encoded {col} with {len(self.label_encoders[col].classes_)} categories")
        
        return df_encoded
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features for training
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with selected features
        """
        logger.info("Selecting features for training")
        
        # Define feature columns (exclude target and metadata)
        exclude_columns = [
            'datetime', 'name', 'id', 'tagDate', 'active', 'dist_total',
            'sst_category', 'chl_category', 'pp_category'  # Exclude categorical versions
        ]
        
        # Get all numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Select features
        self.feature_columns = [col for col in numerical_cols if col not in exclude_columns]
        
        # Ensure target column is included if it exists
        if self.target_column in df.columns and self.target_column not in self.feature_columns:
            self.feature_columns.append(self.target_column)
        
        logger.info(f"Selected {len(self.feature_columns)} features for training")
        logger.info(f"Features: {self.feature_columns}")
        
        # Create feature DataFrame
        feature_df = df[self.feature_columns].copy()
        
        return feature_df
    
    def scale_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale features for training
        
        Args:
            df: Input DataFrame
            fit_scaler: Whether to fit the scaler or use existing one
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Scaling features")
        
        df_scaled = df.copy()
        
        # Get feature columns (exclude target)
        feature_cols = [col for col in self.feature_columns if col != self.target_column]
        
        if fit_scaler:
            # Fit scaler on training data
            df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            logger.info("Fitted scaler on training data")
        else:
            # Transform using existing scaler
            df_scaled[feature_cols] = self.scaler.transform(df[feature_cols])
            logger.info("Transformed using existing scaler")
        
        return df_scaled
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data with test_size={test_size}")
        
        # Separate features and target
        feature_cols = [col for col in self.feature_columns if col != self.target_column]
        
        if self.target_column in df.columns:
            X = df[feature_cols]
            y = df[self.target_column]
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Training target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_training_data(self, data_path: str, 
                            test_size: float = 0.2,
                            random_state: int = 42,
                            output_dir: str = "training_data") -> Dict:
        """
        Complete pipeline to prepare training data
        
        Args:
            data_path: Path to integrated data
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            output_dir: Directory to save prepared data
            
        Returns:
            Dictionary containing prepared data and metadata
        """
        logger.info("Starting complete training data preparation pipeline")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Step 1: Load and validate data
        df = self.load_and_validate_data(data_path)
        
        # Step 2: Create engineered features
        df = self.create_feature_engineering(df)
        
        # Step 3: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Select features
        df = self.select_features(df)
        
        # Step 6: Split data
        X_train, X_test, y_train, y_test = self.split_data(df, test_size, random_state)
        
        # Step 7: Scale features
        X_train_scaled = self.scale_features(X_train, fit_scaler=True)
        X_test_scaled = self.scale_features(X_test, fit_scaler=False)
        
        # Step 8: Save prepared data
        X_train_scaled.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test_scaled.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train_scaled),
            'n_test_samples': len(X_test_scaled),
            'target_distribution_train': y_train.value_counts().to_dict(),
            'target_distribution_test': y_test.value_counts().to_dict(),
            'scaler_fitted': True,
            'label_encoders': list(self.label_encoders.keys())
        }
        
        # Save metadata
        import json
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training data preparation complete!")
        logger.info(f"Saved to {output_dir}/")
        logger.info(f"Training samples: {len(X_train_scaled)}")
        logger.info(f"Test samples: {len(X_test_scaled)}")
        logger.info(f"Features: {len(self.feature_columns)}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'metadata': metadata,
            'feature_columns': self.feature_columns
        }

def main():
    """Main function to prepare training data"""
    logger.info("Starting training data preparation")
    
    # Initialize preparator
    preparator = TrainingDataPreparator()
    
    # Prepare training data
    prepared_data = preparator.prepare_training_data(
        data_path='integrated_data_preprocessed.csv',
        test_size=0.2,
        random_state=42,
        output_dir='training_data'
    )
    
    # Print summary
    print(f"\n=== TRAINING DATA PREPARATION SUMMARY ===")
    print(f"Training samples: {prepared_data['metadata']['n_train_samples']:,}")
    print(f"Test samples: {prepared_data['metadata']['n_test_samples']:,}")
    print(f"Features: {prepared_data['metadata']['n_features']}")
    print(f"Feature columns: {prepared_data['feature_columns']}")
    
    print(f"\n=== TARGET DISTRIBUTION ===")
    print("Training set:")
    for label, count in prepared_data['metadata']['target_distribution_train'].items():
        print(f"  {label}: {count} ({count/prepared_data['metadata']['n_train_samples']*100:.1f}%)")
    
    print("Test set:")
    for label, count in prepared_data['metadata']['target_distribution_test'].items():
        print(f"  {label}: {count} ({count/prepared_data['metadata']['n_test_samples']*100:.1f}%)")
    
    print(f"\n=== DATA SAVED TO ===")
    print("training_data/X_train.csv - Training features")
    print("training_data/X_test.csv - Test features")
    print("training_data/y_train.csv - Training targets")
    print("training_data/y_test.csv - Test targets")
    print("training_data/metadata.json - Data metadata")
    
    print(f"\n=== TRAINING DATA READY ===")
    print("Data is now prepared and ready for machine learning training!")

if __name__ == "__main__":
    main()

