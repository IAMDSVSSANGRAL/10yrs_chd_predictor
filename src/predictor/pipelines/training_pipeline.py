import os
import sys
from src.predictor.logger import logging
from src.predictor.exception import TenYearChdException
from src.predictor.components.data_ingestion import DataIngestion
from src.predictor.components.data_validation import DataValidation
from src.predictor.components.data_transformation import DataTransformation
from src.predictor.components.model_trainer import ModelTrainer
from src.predictor.components.model_evaluation import ModelEvaluation
from src.predictor.components.model_monitoring import ModelMonitoring


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()
        self.model_monitoring = ModelMonitoring()

    def run_pipeline(self):
        """
        Run the complete training pipeline
        """
        try:
            logging.info("="*70)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("="*70)

            # Step 1: Data Ingestion
            logging.info("\n[STEP 1/6] Data Ingestion")
            logging.info("-"*70)
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"✅ Data Ingestion completed")
            logging.info(f"   Train: {train_data_path}")
            logging.info(f"   Test: {test_data_path}")

            # Step 2: Data Validation
            logging.info("\n[STEP 2/6] Data Validation")
            logging.info("-"*70)
            raw_data_path = self.data_ingestion.ingestion_config.raw_data_path
            validation_passed, report_path = self.data_validation.initiate_data_validation(raw_data_path)
            
            if not validation_passed:
                logging.warning(f"⚠️ Data validation issues found. Check: {report_path}")
            else:
                logging.info("✅ Data Validation passed")

            # Step 3: Data Transformation
            logging.info("\n[STEP 3/6] Data Transformation")
            logging.info("-"*70)
            (X_train_resampled, y_train_resampled, 
             X_test_processed, y_test, preprocessor_path) = self.data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )
            logging.info(f"✅ Data Transformation completed")
            logging.info(f"   Training samples: {X_train_resampled.shape[0]}")
            logging.info(f"   Test samples: {X_test_processed.shape[0]}")
            logging.info(f"   Features: {X_train_resampled.shape[1]}")

            # Step 4: Model Training
            logging.info("\n[STEP 4/6] Model Training")
            logging.info("-"*70)
            (model_path, best_model_name, 
             test_metrics, all_results) = self.model_trainer.train_model(
                X_train_resampled, y_train_resampled,
                X_test_processed, y_test
            )
            logging.info(f"✅ Model Training completed")
            logging.info(f"   Best Model: {best_model_name}")
            logging.info(f"   Test F1-Score: {test_metrics['f1_score']:.4f}")
            logging.info(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")

            # Step 5: Model Evaluation
            logging.info("\n[STEP 5/6] Model Evaluation")
            logging.info("-"*70)
            eval_metrics = self.model_evaluation.evaluate_model(
                model_path, X_test_processed, y_test
            )
            logging.info(f"✅ Model Evaluation completed")
            
            # Check against baseline
            baseline_passed = self.model_evaluation.compare_with_baseline(
                eval_metrics, baseline_f1=0.65
            )

            # Step 6: Initialize Monitoring
            logging.info("\n[STEP 6/6] Model Monitoring Setup")
            logging.info("-"*70)
            
            # Save reference statistics for drift detection
            import pandas as pd
            train_df = pd.read_csv(train_data_path)
            self.model_monitoring.save_reference_stats(train_df)
            
            # Log initial metrics
            import joblib
            model = joblib.load(model_path)
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
            
            self.model_monitoring.log_prediction_metrics(
                y_test, y_pred, y_pred_proba, model_version="v1.0"
            )
            logging.info(f"✅ Monitoring setup completed")

            # Final Summary
            logging.info("\n" + "="*70)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("="*70)
            logging.info(f"\n📊 Final Results:")
            logging.info(f"   Model: {best_model_name}")
            logging.info(f"   F1-Score: {test_metrics['f1_score']:.4f}")
            logging.info(f"   ROC-AUC: {test_metrics.get('roc_auc', 'N/A')}")
            logging.info(f"   Baseline Check: {'✅ PASSED' if baseline_passed else '❌ FAILED'}")
            logging.info(f"\n📁 Artifacts:")
            logging.info(f"   Model: {model_path}")
            logging.info(f"   Preprocessor: {preprocessor_path}")
            logging.info("="*70 + "\n")

            return {
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'best_model': best_model_name,
                'metrics': test_metrics,
                'baseline_passed': baseline_passed
            }

        except Exception as e:
            logging.error("❌ Training pipeline failed")
            raise TenYearChdException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run_pipeline()