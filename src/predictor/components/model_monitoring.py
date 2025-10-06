import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import json
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.predictor.exception import TenYearChdException
from src.predictor.logger import logging


@dataclass
class ModelMonitoringConfig:
    monitoring_log_path: str = os.path.join("artifacts", "monitoring", "model_metrics.json")
    drift_report_path: str = os.path.join("artifacts", "monitoring", "drift_report.txt")
    alert_threshold: float = 0.05  # 5% performance drop triggers alert


class ModelMonitoring:
    def __init__(self):
        self.monitoring_config = ModelMonitoringConfig()
        os.makedirs(os.path.dirname(self.monitoring_config.monitoring_log_path), exist_ok=True)

    def log_prediction_metrics(self, y_true, y_pred, y_pred_proba=None, model_version="v1"):
        """
        Log model performance metrics over time
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            metrics = {
                'timestamp': timestamp,
                'model_version': model_version,
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1_score': float(f1_score(y_true, y_pred)),
                'sample_size': len(y_true),
                'positive_rate': float(np.mean(y_true))
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
            
            # Append to log file
            if os.path.exists(self.monitoring_config.monitoring_log_path):
                with open(self.monitoring_config.monitoring_log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(metrics)
            
            with open(self.monitoring_config.monitoring_log_path, 'w') as f:
                json.dump(logs, f, indent=4)
            
            logging.info(f"Metrics logged: F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            raise TenYearChdException(e, sys)

    def detect_performance_drift(self, baseline_f1=0.70, threshold=0.05):
        """
        Detect if model performance has degraded
        """
        try:
            if not os.path.exists(self.monitoring_config.monitoring_log_path):
                logging.warning("No monitoring logs found")
                return False
            
            with open(self.monitoring_config.monitoring_log_path, 'r') as f:
                logs = json.load(f)
            
            if len(logs) < 2:
                logging.info("Not enough data for drift detection")
                return False
            
            recent_f1 = logs[-1]['f1_score']
            drift_detected = False
            alert_message = []
            
            # Check against baseline
            if recent_f1 < (baseline_f1 - threshold):
                drift_detected = True
                alert_message.append(
                    f"⚠️ Performance drift detected! "
                    f"Current F1: {recent_f1:.4f} vs Baseline: {baseline_f1:.4f}"
                )
            
            # Check recent trend (last 5 predictions)
            if len(logs) >= 5:
                recent_scores = [log['f1_score'] for log in logs[-5:]]
                avg_recent = np.mean(recent_scores)
                
                if avg_recent < (baseline_f1 - threshold):
                    drift_detected = True
                    alert_message.append(
                        f"⚠️ Recent performance trend is declining! "
                        f"Avg F1 (last 5): {avg_recent:.4f}"
                    )
            
            # Save drift report
            if drift_detected:
                with open(self.monitoring_config.drift_report_path, 'w') as f:
                    f.write("="*70 + "\n")
                    f.write("MODEL DRIFT ALERT\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("\n".join(alert_message))
                    f.write("\n\n" + "="*70 + "\n")
                
                for msg in alert_message:
                    logging.warning(msg)
            else:
                logging.info("✅ No performance drift detected")
            
            return drift_detected
            
        except Exception as e:
            raise TenYearChdException(e, sys)

    def check_data_drift(self, current_data, reference_stats_path=None):
        """
        Check for data drift by comparing distributions
        """
        try:
            logging.info("Checking for data drift...")
            
            if reference_stats_path and os.path.exists(reference_stats_path):
                with open(reference_stats_path, 'r') as f:
                    reference_stats = json.load(f)
                
                drift_detected = False
                drift_features = []
                
                # Check numerical features
                for col in current_data.select_dtypes(include=[np.number]).columns:
                    if col in reference_stats:
                        current_mean = current_data[col].mean()
                        ref_mean = reference_stats[col]['mean']
                        ref_std = reference_stats[col]['std']
                        
                        # Check if current mean is outside 2 std of reference
                        if abs(current_mean - ref_mean) > 2 * ref_std:
                            drift_detected = True
                            drift_features.append(col)
                
                if drift_detected:
                    logging.warning(f"⚠️ Data drift detected in features: {drift_features}")
                else:
                    logging.info("✅ No data drift detected")
                
                return drift_detected, drift_features
            else:
                logging.info("No reference statistics available for drift detection")
                return False, []
                
        except Exception as e:
            raise TenYearChdException(e, sys)

    def save_reference_stats(self, data, save_path=None):
        """
        Save reference statistics for future drift detection
        """
        try:
            if save_path is None:
                save_path = os.path.join("artifacts", "monitoring", "reference_stats.json")
            
            stats = {}
            for col in data.select_dtypes(include=[np.number]).columns:
                stats[col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median())
                }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(stats, f, indent=4)
            
            logging.info(f"Reference statistics saved at: {save_path}")
            
        except Exception as e:
            raise TenYearChdException(e, sys)