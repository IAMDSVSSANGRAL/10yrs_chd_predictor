import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging
from src.predictor.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_report_path: str = os.path.join("artifacts", "model_report.txt")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Train multiple models and select the best one
        """
        try:
            logging.info("Starting model training")

            # Define models to train
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=15, 
                    random_state=42,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5,
                    random_state=42
                ),
                'XGBoost': XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'AdaBoost': AdaBoostClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                ),
                'KNN': KNeighborsClassifier(n_neighbors=5)
            }

            model_results = {}
            
            logging.info(f"Training {len(models)} different models...")

            # Train and evaluate each model
            for model_name, model in models.items():
                logging.info(f"Training {model_name}...")
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_train_proba = model.predict_proba(X_train)[:, 1]
                    y_test_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_train_proba = None
                    y_test_proba = None
                
                # Evaluate
                train_metrics = self.evaluate_model(y_train, y_train_pred, y_train_proba)
                test_metrics = self.evaluate_model(y_test, y_test_pred, y_test_proba)
                
                model_results[model_name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'y_test_pred': y_test_pred,
                    'y_test_proba': y_test_proba
                }
                
                logging.info(f"{model_name} - Test F1: {test_metrics['f1_score']:.4f}, "
                           f"Test ROC-AUC: {test_metrics.get('roc_auc', 'N/A')}")

            # Select best model based on F1 score
            best_model_name = max(
                model_results.keys(), 
                key=lambda x: model_results[x]['test_metrics']['f1_score']
            )
            best_model_info = model_results[best_model_name]
            best_model = best_model_info['model']
            
            logging.info(f"Best model selected: {best_model_name}")

            # Save best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            # Generate and save model report
            self.generate_model_report(
                model_results, 
                best_model_name, 
                y_test, 
                best_model_info['y_test_pred']
            )

            return (
                self.model_trainer_config.trained_model_file_path,
                best_model_name,
                best_model_info['test_metrics'],
                model_results
            )

        except Exception as e:
            raise TenYearChdException(e, sys)

    def generate_model_report(self, model_results, best_model_name, y_test, y_test_pred):
        """
        Generate a detailed model training report
        """
        try:
            with open(self.model_trainer_config.model_report_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("MODEL TRAINING REPORT\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Best Model: {best_model_name}\n\n")
                
                f.write("-"*70 + "\n")
                f.write("ALL MODELS PERFORMANCE\n")
                f.write("-"*70 + "\n\n")
                
                for model_name, results in model_results.items():
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"{'='*50}\n\n")
                    
                    f.write("Training Metrics:\n")
                    for metric, value in results['train_metrics'].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    
                    f.write("\nTest Metrics:\n")
                    for metric, value in results['test_metrics'].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    
                    f.write("\n")
                
                # Confusion Matrix for best model
                f.write("\n" + "="*70 + "\n")
                f.write(f"CONFUSION MATRIX - {best_model_name}\n")
                f.write("="*70 + "\n")
                cm = confusion_matrix(y_test, y_test_pred)
                f.write(f"\n{cm}\n\n")
                
                # Classification Report
                f.write("\n" + "="*70 + "\n")
                f.write(f"CLASSIFICATION REPORT - {best_model_name}\n")
                f.write("="*70 + "\n\n")
                f.write(classification_report(y_test, y_test_pred))
                
            logging.info(f"Model report saved at: {self.model_trainer_config.model_report_path}")
            
        except Exception as e:
            logging.warning(f"Failed to generate model report: {str(e)}")