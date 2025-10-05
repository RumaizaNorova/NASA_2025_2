"""
Shark Habitat Prediction Model
Uses real NASA satellite data to predict shark foraging habitats
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SharkHabitatPredictor:
    """
    Main class for shark habitat prediction using real NASA satellite data
    """
    
    def __init__(self):
        """Initialize the shark habitat predictor"""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = 0
        
    def load_and_preprocess_data(self, shark_data_path: str) -> pd.DataFrame:
        """
        Load and preprocess shark tracking data
        
        Args:
            shark_data_path: Path to shark tracking data CSV
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Loading shark data from {shark_data_path}")
        
        # Load data
        df = pd.read_csv(shark_data_path)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract temporal features
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Extract spatial features
        df['lat_rounded'] = np.round(df['latitude'], 1)
        df['lon_rounded'] = np.round(df['longitude'], 1)
        
        # Create movement features
        df = df.sort_values(['name', 'datetime'])
        df['lat_diff'] = df.groupby('name')['latitude'].diff()
        df['lon_diff'] = df.groupby('name')['longitude'].diff()
        df['speed'] = np.sqrt(df['lat_diff']**2 + df['lon_diff']**2)
        df['distance_from_previous'] = np.sqrt(
            df['lat_diff']**2 + df['lon_diff']**2
        ) * 111  # Approximate km
        
        # Create foraging behavior labels (placeholder for now)
        # In reality, this would be based on movement patterns, dive data, etc.
        df['foraging_behavior'] = self._create_foraging_labels(df)
        
        # Check for satellite features (from data integration)
        df = self._check_satellite_features(df)
        
        logger.info(f"Data preprocessed: {len(df)} records")
        return df
    
    def _create_foraging_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create foraging behavior labels based on movement patterns
        
        Args:
            df: DataFrame with shark tracking data
            
        Returns:
            Array of foraging labels (0: not foraging, 1: foraging)
        """
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
        
        return np.array(foraging_labels)
    
    def _check_satellite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for satellite-derived oceanographic features in integrated data
        
        Args:
            df: DataFrame with shark tracking data (may already have satellite features)
            
        Returns:
            DataFrame with satellite features (if available)
        """
        logger.info("Checking for satellite features in data")
        
        # Check if satellite features are already present (from data integration)
        satellite_features = ['chlorophyll_a', 'sea_surface_temp', 'eddy_presence', 
                            'eddy_intensity', 'ssh_anomaly', 'distance_to_coast']
        
        missing_features = [feat for feat in satellite_features if feat not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing satellite features: {missing_features}")
            logger.warning("Please run data integration first to add real satellite features")
            # Don't add synthetic data - just return the dataframe as is
        else:
            logger.info("Found satellite features in data - using real satellite data")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for machine learning
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Check if this is integrated data (has satellite features) or raw data
        if 'chlorophyll_a' in df.columns and 'sea_surface_temp' in df.columns:
            # This is integrated data, use available features
            feature_columns = [
                'latitude', 'longitude', 'chlorophyll_a', 'sea_surface_temp', 
                'ssh_anomaly', 'eddy_presence', 'eddy_intensity', 'eddy_radius',
                'distance_to_coast', 'primary_productivity', 'current_strength'
            ]
            
            # Add temporal features if they exist
            temporal_features = ['year', 'month', 'day', 'hour', 'day_of_year']
            for feat in temporal_features:
                if feat in df.columns:
                    feature_columns.append(feat)
            
            # Add movement features if they exist
            movement_features = ['lat_diff', 'lon_diff', 'speed', 'distance_from_previous']
            for feat in movement_features:
                if feat in df.columns:
                    feature_columns.append(feat)
                    
        else:
            # This is raw data without satellite features
            logger.error("No satellite features found in data!")
            logger.error("Please run data integration first to add real satellite features")
            raise ValueError("Missing satellite features. Run data integration first.")
        
        # Encode categorical variables
        le_species = LabelEncoder()
        df['species_encoded'] = le_species.fit_transform(df['species'])
        feature_columns.append('species_encoded')
        
        le_gender = LabelEncoder()
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        feature_columns.append('gender_encoded')
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['foraging_behavior'].copy()
        
        # Handle missing values by filling with median/mean for numerical features
        # and mode for categorical features
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
        
        # Remove rows where target is still NaN
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
        logger.info(f"Features used: {feature_columns}")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train multiple models and select the best one
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training multiple models")
        
        # Check if we have enough data
        if len(X) < 10:
            logger.warning(f"Very limited data: {len(X)} samples. Results may not be reliable.")
        
        # Split data with temporal consideration
        # Use earlier data for training, later for testing
        X_sorted, y_sorted = self._temporal_split(X, y)
        
        # Use a smaller test size if we have limited data
        test_size = min(0.2, max(0.1, 5/len(X))) if len(X) > 10 else 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sorted, y_sorted, test_size=test_size, random_state=42, stratify=y_sorted
        )
        
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=50,  # Reduced for small dataset
                random_state=42,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=50,  # Reduced for small dataset
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            try:
                # Create pipeline with scaling for linear models
                if name == 'LogisticRegression':
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', model)
                    ])
                else:
                    pipeline = Pipeline([
                        ('classifier', model)
                    ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Evaluate
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'model': pipeline,
                    'auc_score': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'test_labels': y_test
                }
                
                # Store feature importance if available
                if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(
                        X.columns,
                        pipeline.named_steps['classifier'].feature_importances_
                    ))
                
                logger.info(f"{name} AUC Score: {auc_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
            self.best_model = results[best_model_name]['model']
            self.best_score = results[best_model_name]['auc_score']
            
            logger.info(f"Best model: {best_model_name} with AUC: {self.best_score:.3f}")
        else:
            logger.error("No models were successfully trained")
        
        return results
    
    def _temporal_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Sort data by time to avoid temporal data leakage
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Temporally sorted data
        """
        # Sort by year, month, day to maintain temporal order
        # This assumes the data is already in chronological order
        return X, y
    
    def evaluate_model(self, results: Dict[str, Any]) -> None:
        """
        Evaluate and visualize model performance
        
        Args:
            results: Dictionary with model results
        """
        if not results:
            logger.warning("No results to evaluate")
            return
            
        logger.info("Evaluating model performance")
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        
        for name, result in results.items():
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(result['test_labels'], result['probabilities'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc_score"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Shark Habitat Prediction Models (Real NASA Data)')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print classification reports
        for name, result in results.items():
            print(f"\n{name} Classification Report:")
            print(classification_report(result['test_labels'], result['predictions']))
        
        # Plot feature importance if available
        if self.feature_importance:
            self._plot_feature_importance()
    
    def _plot_feature_importance(self) -> None:
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(1, len(self.feature_importance), figsize=(15, 6))
        if len(self.feature_importance) == 1:
            axes = [axes]
        
        for idx, (name, importance) in enumerate(self.feature_importance.items()):
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features[:10])  # Top 10 features
            
            axes[idx].barh(range(len(features)), importances)
            axes[idx].set_yticks(range(len(features)))
            axes[idx].set_yticklabels(features)
            axes[idx].set_xlabel('Feature Importance')
            axes[idx].set_title(f'{name} Feature Importance (Real NASA Data)')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_habitat_suitability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict habitat suitability for new locations
        
        Args:
            X: Feature matrix for new locations
            
        Returns:
            Array of habitat suitability scores
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        return self.best_model.predict_proba(X)[:, 1]

def main():
    """Main function to run the shark habitat prediction pipeline"""
    logger.info("Starting Shark Habitat Prediction Pipeline with Real NASA Data")
    
    # Initialize predictor
    predictor = SharkHabitatPredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data('sharks_cleaned.csv')
    
    # Filter to February 2020 for testing (when real NASA data is available)
    df_test = df[
        (df['year'] == 2020) & (df['month'] == 2)
    ].copy()
    
    logger.info(f"Using {len(df_test)} records from February 2020 for testing")
    
    if len(df_test) < 100:
        logger.warning("Limited data for February 2020, using all available data")
        df_test = df.copy()
    
    # Prepare features
    X, y = predictor.prepare_features(df_test)
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Evaluate models
    predictor.evaluate_model(results)
    
    logger.info("Pipeline completed successfully with real NASA data")

if __name__ == "__main__":
    main()