from src.predictor.logger import logging
from src.predictor.exception import TenYearChdException
from src.predictor.components.data_ingestion import DataIngestion
from src.predictor.components.data_transformation import DataTransformation
from src.predictor.components.data_validation import DataValidation
from src.predictor.components.model_trainer import ModelTrainer
from src.predictor.components.model_evaluation import ModelEvaluation
from src.predictor.components.model_monitoring import ModelMonitoring
from src.predictor.utils import save_object
import os
import sys
import pandas as pd

if __name__ == "__main__":
    logging.info("="*70)
    logging.info("ML PIPELINE EXECUTION STARTED")
    logging.info("="*70)

    try:
        # -----------------------
        # Step 1: Data Ingestion
        # -----------------------
        logging.info("\n[STEP 1/6] Starting Data Ingestion...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"[OK] Data Ingestion completed")
        logging.info(f"     Train: {train_data_path}")
        logging.info(f"     Test: {test_data_path}")

        # -----------------------
        # Step 2: Data Validation
        # -----------------------
        logging.info("\n[STEP 2/6] Starting Data Validation...")
        data_validation = DataValidation()
        raw_data_path = data_ingestion.ingestion_config.raw_data_path
        validation_passed, report_path = data_validation.initiate_data_validation(raw_data_path)
        
        if not validation_passed:
            logging.warning(f"[WARNING] Data validation found issues. Check: {report_path}")
        else:
            logging.info(f"[OK] Data Validation passed")

        # -----------------------
        # Step 3: Data Transformation
        # -----------------------
        logging.info("\n[STEP 3/6] Starting Data Transformation...")
        data_transformation = DataTransformation(use_smoteenn=False)
        X_train_resampled, y_train_resampled, X_test_processed, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )

        logging.info("[OK] Data Transformation completed")
        logging.info(f"     Training samples: {X_train_resampled.shape[0]}")
        logging.info(f"     Test samples: {X_test_processed.shape[0]}")
        logging.info(f"     Features: {X_train_resampled.shape[1]}")
        logging.info(f"     Preprocessor saved at: {preprocessor_path}")

        # -----------------------
        # Step 4: Model Training
        # -----------------------
        logging.info("\n[STEP 4/6] Starting Model Training...")
        model_trainer = ModelTrainer()
        model_path, best_model_name, test_metrics, all_results = model_trainer.train_model(
            X_train_resampled, y_train_resampled, 
            X_test_processed, y_test
        )

        logging.info("[OK] Model Training completed")
        logging.info(f"     Best Model: {best_model_name}")
        logging.info(f"     Test F1-Score: {test_metrics['f1_score']:.4f}")
        logging.info(f"     Test Accuracy: {test_metrics['accuracy']:.4f}")
        if 'roc_auc' in test_metrics:
            logging.info(f"     Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logging.info(f"     Model saved at: {model_path}")

        # -----------------------
        # Step 5: Model Evaluation
        # -----------------------
        logging.info("\n[STEP 5/6] Starting Model Evaluation...")
        model_evaluation = ModelEvaluation()
        eval_metrics = model_evaluation.evaluate_model(
            model_path, X_test_processed, y_test
        )
        
        logging.info("[OK] Model Evaluation completed")
        logging.info(f"     Evaluation report saved")
        
        # Check against baseline
        baseline_passed = model_evaluation.compare_with_baseline(
            eval_metrics, baseline_f1=0.65
        )
        
        if baseline_passed:
            logging.info("     [OK] Model passed baseline check")
        else:
            logging.warning("     [WARNING] Model below baseline threshold")

        # -----------------------
        # Step 6: Model Monitoring Setup
        # -----------------------
        logging.info("\n[STEP 6/6] Setting up Model Monitoring...")
        model_monitoring = ModelMonitoring()
        
        # Save reference statistics
        train_df = pd.read_csv(train_data_path)
        model_monitoring.save_reference_stats(train_df)
        
        # Log initial metrics
        import joblib
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None
        
        initial_metrics = model_monitoring.log_prediction_metrics(
            y_test, y_pred, y_pred_proba, model_version="v1.0"
        )
        
        logging.info("[OK] Model Monitoring setup completed")
        logging.info(f"     Metrics logged for tracking")

        # -----------------------
        # FINAL SUMMARY
        # -----------------------
        logging.info("\n" + "="*70)
        logging.info("ML PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("="*70)
        logging.info("\nFINAL RESULTS:")
        logging.info(f"  Best Model: {best_model_name}")
        logging.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {test_metrics['precision']:.4f}")
        logging.info(f"  Recall: {test_metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")
        if 'roc_auc' in test_metrics:
            logging.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logging.info(f"\nBaseline Check: {'PASSED' if baseline_passed else 'FAILED'}")
        logging.info("\nARTIFACTS SAVED:")
        logging.info(f"  Model: {model_path}")
        logging.info(f"  Preprocessor: {preprocessor_path}")
        logging.info(f"  Validation Report: {report_path}")
        logging.info(f"  Model Report: {model_trainer.model_trainer_config.model_report_path}")
        logging.info(f"  Evaluation Report: {model_evaluation.evaluation_config.evaluation_report_path}")
        logging.info("="*70 + "\n")

        print("\n" + "🎉 SUCCESS! All pipeline steps completed.")
        print(f"📊 Best Model: {best_model_name} with F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"📁 Model saved at: {model_path}")

    except Exception as e:
        logging.error("="*70)
        logging.error("ML PIPELINE FAILED!")
        logging.error("="*70)
        raise TenYearChdException(e, sys)