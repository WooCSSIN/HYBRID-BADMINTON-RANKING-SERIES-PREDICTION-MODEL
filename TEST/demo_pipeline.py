"""
Demo Pipeline
=============

End-to-end demo chạy toàn bộ ML improvements pipeline.

Steps:
1. Feature Engineering
2. Model Training (Ensemble + Quantile)
3. Validation
4. Predictions with Confidence Intervals

Author: ML Improvements Project
Date: 2025-11-21
"""

from pathlib import Path
import pandas as pd
import sys

# Add TEST directory to path
BASE = Path(__file__).parent.parent
TEST_DIR = BASE / 'TEST'
sys.path.insert(0, str(TEST_DIR))

# Import modules
from importlib import import_module


def run_feature_engineering():
    """Step 1: Feature Engineering"""
    print("\n" + "=" * 80)
    print("STEP 1: FEATURE ENGINEERING")
    print("=" * 80)
    
    fe_module = import_module('1_feature_engineering')
    fe_module.main()


def run_validation():
    """Step 2: Validation Framework"""
    print("\n" + "=" * 80)
    print("STEP 2: VALIDATION FRAMEWORK")
    print("=" * 80)
    
    val_module = import_module('2_validation_framework')
    val_module.demo_validation()


def run_ensemble_training():
    """Step 3: Ensemble Model Training"""
    print("\n" + "=" * 80)
    print("STEP 3: ENSEMBLE MODEL TRAINING")
    print("=" * 80)
    
    ensemble_module = import_module('3_ensemble_models')
    ensemble_module.demo_ensemble()


def run_confidence_intervals():
    """Step 4: Confidence Intervals"""
    print("\n" + "=" * 80)
    print("STEP 4: CONFIDENCE INTERVALS")
    print("=" * 80)
    
    ci_module = import_module('4_confidence_intervals')
    ci_module.demo_confidence_intervals()


def main():
    """PIPELINE ĐƯỢC CHẠY HOÀN CHỈNH"""
    
    print("\n" + "=" * 80)
    print(" PIPELINE ĐƯỢC CHẠY HOÀN CHỈNH")
    print("=" * 80)
    print("\nBản demo này sẽ chạy tất cả các cải tiến theo trình tự:")
    print("1. Feature Engineering")
    print("2. Validation Framework")
    print("3. Ensemble Model Training")
    print("4. Confidence Intervals Prediction")
    print("\n⏱ Quá trình này có thể mất vài phút...\n")
    
    try:
        # Step 1: Feature Engineering
        run_feature_engineering()
        
        # Step 2: Validation
        run_validation()
        
        # Step 3: Ensemble Training
        run_ensemble_training()
        
        # Step 4: Confidence Intervals
        run_confidence_intervals()
        
        # Final summary
        print("\n" + "=" * 80)
        print(" PIPELINE thành công ")
        print("=" * 80)
        
        print("\n Generated Files:")
        print(f"    TEST/bwf_official_enhanced.csv - Enhanced dataset")
        print(f"    TEST/outputs/validation_results_MS.csv - Validation metrics")
        print(f"    TEST/outputs/validation_results.png - Validation plots")
        print(f"    TEST/models/ensemble_MS_lgbm.pkl - LightGBM model")
        print(f"    TEST/models/ensemble_MS_lstm.pt - LSTM model (if PyTorch available)")
        print(f"    TEST/outputs/predictions_with_ci_MS.csv - Predictions with CI")
        
        print("\n Next Steps:")
        print("   1. Xem lại các tệp đầu ra trong TEST/outputs/")
        print("   2. Kiểm tra số liệu hiệu suất mô hình")
        print("   3. Tích hợp các cải tiến vào tệp forecast_to_2035.py chính")
        print("   4. Kiểm tra trên các bản vẽ khác nhau (WS, MD, WD, XD)")
        
    except Exception as e:
        print(f"\n Pipeline thất bại với lỗi:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
