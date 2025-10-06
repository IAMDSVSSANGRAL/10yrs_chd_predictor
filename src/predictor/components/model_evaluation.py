import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import joblib
import json

from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging


@dataclass
class ModelEvaluationConfig:
    evaluation_report_path: str = os.path.join("artifacts", "evaluation_report.json")
    metrics_file_path: str = os.path.join("artifacts", "metrics.txt")


class ModelEvaluation:
    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics
        """
        try:
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
                'specificity': self.calculate_specificity(y_true, y_pred)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
                metrics['average_precision'] = float(average_precision_score(y_true, y_pred_proba))
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            }
            
            return metrics
            
        except Exception as e:
            raise TenYearChdException(e, sys)

    def calculate_specificity(self, y_true, y_pred):
        """
        Calculate specificity (True Negative Rate)
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp = cm[0, 0], cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return float(specificity)

    def evaluate_model(self, model_path, X_test, y_test):
        """
        Evaluate a trained model on test data
        """
        try:
            logging.info("Starting model evaluation")
            
            # Load model
            model = joblib.load(model_path)
            logging.info(f"Model loaded from {model_path}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Add sample size info
            metrics['test_samples'] = len(y_test)
            metrics['positive_samples'] = int(np.sum(y_test))
            metrics['negative_samples'] = int(len(y_test) - np.sum(y_test))
            
            # Save evaluation report as JSON
            os.makedirs(os.path.dirname(self.evaluation_config.evaluation_report_path), exist_ok=True)
            with open(self.evaluation_config.evaluation_report_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logging.info(f"Evaluation report saved at: {self.evaluation_config.evaluation_report_path}")
            
            # Save human-readable metrics
            self.save_metrics_report(metrics, y_test, y_pred)
            
            logging.info("Model evaluation completed")
            
            return metrics
            
        except Exception as e:
            raise TenYearChdException(e, sys)

    def save_metrics_report(self, metrics, y_test, y_pred):
        """
        Save human-readable metrics report
        """
        try:
            with open(self.evaluation_config.metrics_file_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("MODEL EVALUATION METRICS\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Test Samples: {metrics['test_samples']}\n")
                f.write(f"Positive Samples: {metrics['positive_samples']}\n")
                f.write(f"Negative Samples: {metrics['negative_samples']}\n\n")
                
                f.write("-"*70 + "\n")
                f.write("CLASSIFICATION METRICS\n")
                f.write("-"*70 + "\n\n")
                
                f.write(f"Accuracy:    {metrics['accuracy']:.4f}\n")
                f.write(f"Precision:   {metrics['precision']:.4f}\n")
                f.write(f"Recall:      {metrics['recall']:.4f}\n")
                f.write(f"F1-Score:    {metrics['f1_score']:.4f}\n")
                f.write(f"Specificity: {metrics['specificity']:.4f}\n")
                
                if 'roc_auc' in metrics:
                    f.write(f"ROC-AUC:     {metrics['roc_auc']:.4f}\n")
                if 'average_precision' in metrics:
                    f.write(f"Avg Precision: {metrics['average_precision']:.4f}\n")
                
                f.write("\n" + "-"*70 + "\n")
                f.write("CONFUSION MATRIX\n")
                f.write("-"*70 + "\n\n")
                cm = metrics['confusion_matrix']
                f.write(f"True Negative:  {cm['true_negative']}\n")
                f.write(f"False Positive: {cm['false_positive']}\n")
                f.write(f"False Negative: {cm['false_negative']}\n")
                f.write(f"True Positive:  {cm['true_positive']}\n\n")
                
                f.write("-"*70 + "\n")
                f.write("CLASSIFICATION REPORT\n")
                f.write("-"*70 + "\n\n")
                f.write(classification_report(y_test, y_pred))
                
            logging.info(f"Metrics report saved at: {self.evaluation_config.metrics_file_path}")
            
        except Exception as e:
            logging.warning(f"Failed to save metrics report: {str(e)}")

    def compare_with_baseline(self, metrics, baseline_f1=0.65):
        """
        Compare model performance with baseline
        """
        try:
            current_f1 = metrics['f1_score']
            
            if current_f1 >= baseline_f1:
                logging.info(f"✅ Model passed baseline check. F1: {current_f1:.4f} >= {baseline_f1:.4f}")
                return True
            else:
                logging.warning(f"❌ Model below baseline. F1: {current_f1:.4f} < {baseline_f1:.4f}")
                return False
                
        except Exception as e:
            raise TenYearChdException(e, sys)