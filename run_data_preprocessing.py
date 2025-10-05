"""
Run the complete data preprocessing pipeline
This script orchestrates the entire data preparation process
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_data_files():
    """Check if required data files exist"""
    logger.info("Checking for required data files...")
    
    required_files = {
        'shark_data': 'sharks_cleaned.csv',
        'nasa_sst': 'nasa_data/modis_aqua/modis_aqua_sst.nc',
        'nasa_chlorophyll': 'nasa_data/modis_aqua/modis_aqua_chlorophyll.nc',
        'nasa_terra_sst': 'nasa_data/modis_terra/modis_terra_sst.nc'
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
        else:
            logger.info(f"✓ Found {name}: {path}")
    
    if missing_files:
        logger.error("Missing required files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        return False
    
    logger.info("All required files found!")
    return True

def run_preprocessing_pipeline():
    """Run the data preprocessing pipeline"""
    logger.info("Starting data preprocessing pipeline...")
    
    try:
        # Import our preprocessing modules
        from data_preprocessing_pipeline import DataIntegrator
        from prepare_training_data import TrainingDataPreparator
        
        # Step 1: Data Integration
        logger.info("Step 1: Integrating shark and NASA data...")
        
        integrator = DataIntegrator()
        
        # Define NASA data paths
        nasa_data_paths = {
            'sst': 'nasa_data/modis_aqua/modis_aqua_sst.nc',
            'chlorophyll': 'nasa_data/modis_aqua/modis_aqua_chlorophyll.nc',
            'terra_sst': 'nasa_data/modis_terra/modis_terra_sst.nc'
        }
        
        # Integrate data for a specific time period
        # Using 2014-02-01 to 2014-02-07 for initial testing
        integrated_data = integrator.integrate_all_data(
            shark_data_path='sharks_cleaned.csv',
            nasa_data_paths=nasa_data_paths,
            start_date='2014-02-01',
            end_date='2014-02-07',
            output_path='integrated_data_preprocessed.csv'
        )
        
        logger.info(f"✓ Data integration complete: {len(integrated_data)} records")
        
        # Step 2: Training Data Preparation
        logger.info("Step 2: Preparing training data...")
        
        preparator = TrainingDataPreparator()
        
        prepared_data = preparator.prepare_training_data(
            data_path='integrated_data_preprocessed.csv',
            test_size=0.2,
            random_state=42,
            output_dir='training_data'
        )
        
        logger.info(f"✓ Training data preparation complete")
        
        # Step 3: Data Quality Report
        logger.info("Step 3: Generating data quality report...")
        
        generate_data_quality_report(integrated_data, prepared_data)
        
        logger.info("✓ Data preprocessing pipeline completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_data_quality_report(integrated_data, prepared_data):
    """Generate a comprehensive data quality report"""
    logger.info("Generating data quality report...")
    
    report = []
    report.append("=" * 60)
    report.append("DATA QUALITY REPORT")
    report.append("=" * 60)
    
    # Basic statistics
    report.append(f"\nBASIC STATISTICS:")
    report.append(f"Total records: {len(integrated_data):,}")
    report.append(f"Date range: {integrated_data['datetime'].min()} to {integrated_data['datetime'].max()}")
    report.append(f"Unique sharks: {integrated_data['name'].nunique()}")
    report.append(f"Species: {integrated_data['species'].nunique()}")
    
    # Missing values analysis
    report.append(f"\nMISSING VALUES ANALYSIS:")
    missing_values = integrated_data.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            percentage = count / len(integrated_data) * 100
            report.append(f"  {col}: {count} ({percentage:.1f}%)")
    else:
        report.append("  No missing values found")
    
    # Target distribution
    if 'foraging_behavior' in integrated_data.columns:
        report.append(f"\nTARGET DISTRIBUTION:")
        foraging_counts = integrated_data['foraging_behavior'].value_counts()
        for label, count in foraging_counts.items():
            percentage = count / len(integrated_data) * 100
            report.append(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Satellite data availability
    satellite_cols = ['sst', 'chlor_a', 'primary_productivity']
    report.append(f"\nSATELLITE DATA AVAILABILITY:")
    for col in satellite_cols:
        if col in integrated_data.columns:
            non_null_count = integrated_data[col].notna().sum()
            percentage = non_null_count / len(integrated_data) * 100
            report.append(f"  {col}: {non_null_count} ({percentage:.1f}%)")
    
    # Training data summary
    report.append(f"\nTRAINING DATA SUMMARY:")
    report.append(f"Training samples: {prepared_data['metadata']['n_train_samples']:,}")
    report.append(f"Test samples: {prepared_data['metadata']['n_test_samples']:,}")
    report.append(f"Features: {prepared_data['metadata']['n_features']}")
    
    # Feature importance (if available)
    if 'feature_columns' in prepared_data:
        report.append(f"\nFEATURE COLUMNS:")
        for i, col in enumerate(prepared_data['feature_columns'], 1):
            report.append(f"  {i:2d}. {col}")
    
    # Data quality issues
    report.append(f"\nDATA QUALITY ISSUES:")
    issues = []
    
    # Check for class imbalance
    if 'foraging_behavior' in integrated_data.columns:
        foraging_counts = integrated_data['foraging_behavior'].value_counts()
        if len(foraging_counts) > 1:
            imbalance_ratio = max(foraging_counts) / min(foraging_counts)
            if imbalance_ratio > 10:
                issues.append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f})")
            elif imbalance_ratio > 3:
                issues.append(f"Moderate class imbalance (ratio: {imbalance_ratio:.1f})")
    
    # Check for missing satellite data
    if 'sst' in integrated_data.columns:
        sst_coverage = integrated_data['sst'].notna().sum() / len(integrated_data)
        if sst_coverage < 0.5:
            issues.append(f"Low satellite data coverage ({sst_coverage:.1%})")
    
    if issues:
        for issue in issues:
            report.append(f"  ⚠️  {issue}")
    else:
        report.append("  ✓ No major data quality issues detected")
    
    # Recommendations
    report.append(f"\nRECOMMENDATIONS:")
    recommendations = []
    
    if 'foraging_behavior' in integrated_data.columns:
        foraging_counts = integrated_data['foraging_behavior'].value_counts()
        if len(foraging_counts) > 1:
            imbalance_ratio = max(foraging_counts) / min(foraging_counts)
            if imbalance_ratio > 3:
                recommendations.append("Consider using class balancing techniques (SMOTE, class weights)")
    
    if 'sst' in integrated_data.columns:
        sst_coverage = integrated_data['sst'].notna().sum() / len(integrated_data)
        if sst_coverage < 0.8:
            recommendations.append("Consider expanding time range or spatial coverage for better satellite data")
    
    recommendations.append("Consider cross-validation for robust model evaluation")
    recommendations.append("Monitor for data leakage between training and test sets")
    
    for i, rec in enumerate(recommendations, 1):
        report.append(f"  {i}. {rec}")
    
    report.append(f"\n" + "=" * 60)
    
    # Save report
    with open('data_quality_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    print('\n'.join(report))
    
    logger.info("Data quality report saved to data_quality_report.txt")

def main():
    """Main function to run the complete preprocessing pipeline"""
    print("=" * 60)
    print("SHARKS FROM SPACE - DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('sharks_cleaned.csv'):
        logger.error("sharks_cleaned.csv not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Step 1: Check data files
    if not check_data_files():
        logger.error("Missing required data files. Please ensure all data files are available.")
        sys.exit(1)
    
    # Step 2: Run preprocessing pipeline
    success = run_preprocessing_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - integrated_data_preprocessed.csv (Integrated shark + NASA data)")
        print("  - training_data/X_train.csv (Training features)")
        print("  - training_data/X_test.csv (Test features)")
        print("  - training_data/y_train.csv (Training targets)")
        print("  - training_data/y_test.csv (Test targets)")
        print("  - training_data/metadata.json (Data metadata)")
        print("  - data_quality_report.txt (Quality analysis)")
        
        print("\nNext steps:")
        print("  1. Review data_quality_report.txt for data quality insights")
        print("  2. Train machine learning models using the prepared data")
        print("  3. Evaluate model performance on the test set")
        print("  4. Iterate and improve based on results")
        
    else:
        print("\n" + "=" * 60)
        print("❌ DATA PREPROCESSING FAILED!")
        print("=" * 60)
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()

