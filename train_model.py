"""
Train Machine Learning Model on Prepared Shark Habitat Data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SharkHabitatModel:
    """
    Machine learning model for shark habitat prediction
    """
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
        }
        self.trained_models = {}
        self.results = {}
        
    def load_training_data(self, data_dir: str = "training_data"):
        """
        Load prepared training data
        
        Args:
            data_dir: Directory containing training data
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        logger.info(f"Loading training data from {data_dir}")
        
        # Load data
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        X_test = pd.read_csv(f"{data_dir}/X_test.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
        y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
        
        # Load metadata
        with open(f"{data_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded training data: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"Features: {metadata['n_features']}")
        
        return X_train, X_test, y_train, y_test, metadata
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        logger.info("Training models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            logger.info(f"✓ {name} trained successfully")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate trained models
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        logger.info("Evaluating models...")
        
        for name, model in self.trained_models.items():
            logger.info(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            logger.info(f"✓ {name} AUC: {auc_score:.3f}")
    
    def create_visualizations(self, X_test, y_test, output_dir: str = "results"):
        """
        Create visualizations of model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            output_dir: Directory to save visualizations
        """
        logger.info("Creating visualizations...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. ROC Curves
        plt.figure(figsize=(10, 8))
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc_score']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Shark Habitat Prediction')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrices
        fig, axes = plt.subplots(1, len(self.results), figsize=(5*len(self.results), 4))
        if len(self.results) == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(y_test, result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name}\nAUC: {result["auc_score"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance (for Random Forest)
        if 'RandomForest' in self.trained_models:
            feature_importance = self.trained_models['RandomForest'].feature_importances_
            feature_names = X_test.columns
            
            # Sort features by importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Plot top 15 features
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Most Important Features (Random Forest)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")
    
    def save_results(self, X_test, y_test, output_dir: str = "results"):
        """
        Save model results and metadata
        
        Args:
            X_test: Test features
            y_test: Test targets
            output_dir: Directory to save results
        """
        logger.info("Saving results...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save model performance summary
        summary = {}
        for name, result in self.results.items():
            summary[name] = {
                'auc_score': result['auc_score'],
                'classification_report': result['classification_report']
            }
        
        with open(f"{output_dir}/model_performance.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'actual': y_test,
            **{f"{name}_pred": result['predictions'] for name, result in self.results.items()},
            **{f"{name}_proba": result['probabilities'] for name, result in self.results.items()}
        })
        predictions_df.to_csv(f"{output_dir}/predictions.csv", index=False)
        
        logger.info(f"Results saved to {output_dir}/")
    
    def print_summary(self):
        """
        Print model performance summary
        """
        print("\n" + "="*60)
        print("SHARK HABITAT PREDICTION MODEL RESULTS")
        print("="*60)
        
        # Model performance
        print("\nMODEL PERFORMANCE:")
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  AUC Score: {result['auc_score']:.3f}")
            
            # Classification report
            report = result['classification_report']
            print(f"  Precision: {report['1']['precision']:.3f}")
            print(f"  Recall: {report['1']['recall']:.3f}")
            print(f"  F1-Score: {report['1']['f1-score']:.3f}")
        
        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['auc_score'])
        print(f"\nBEST MODEL: {best_model[0]} (AUC = {best_model[1]['auc_score']:.3f})")
        
        print("\n" + "="*60)

def main():
    """Main function to train and evaluate models"""
    logger.info("Starting shark habitat prediction model training")
    
    # Initialize model
    model = SharkHabitatModel()
    
    # Load training data
    X_train, X_test, y_train, y_test, metadata = model.load_training_data()
    
    # Train models
    model.train_models(X_train, y_train)
    
    # Evaluate models
    model.evaluate_models(X_test, y_test)
    
    # Create visualizations
    model.create_visualizations(X_test, y_test)
    
    # Save results
    model.save_results(X_test, y_test)
    
    # Print summary
    model.print_summary()
    
    print("\n=== TRAINING COMPLETE ===")
    print("Model training and evaluation completed successfully!")
    print("\nGenerated files:")
    print("  - results/model_performance.json (Performance metrics)")
    print("  - results/predictions.csv (Model predictions)")
    print("  - results/roc_curves.png (ROC curves)")
    print("  - results/confusion_matrices.png (Confusion matrices)")
    print("  - results/feature_importance.png (Feature importance)")
    
    print("\nNext steps:")
    print("  1. Review model performance and visualizations")
    print("  2. Analyze feature importance to understand key factors")
    print("  3. Consider model improvements (hyperparameter tuning, feature engineering)")
    print("  4. Validate on additional time periods or datasets")

if __name__ == "__main__":
    main()
