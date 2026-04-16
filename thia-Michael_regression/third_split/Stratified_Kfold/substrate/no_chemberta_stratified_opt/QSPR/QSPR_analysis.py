"""
ADVANCED QSPR ANALYSIS - MAIN EXECUTION SCRIPT
==============================================

Complete 7-step QSPR pipeline that imports functions and classes from:
  module/models_creation.py

This script:
>> Loads data and calculates descriptors
>> Performs advanced feature selection
>> Runs baseline comparison (dummy + target shuffling)
>> Optimizes hyperparameters (GridSearchCV)
>> Trains multiple ML algorithms
>> Compiles comprehensive results
>> Quantifies uncertainty & AOA analysis
>> Exports results to Excel

Location: QSPR/QSPR_analysis.py
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

import os
os.environ['SKLEARN_PARALLEL_WARNING'] = '0'

import warnings
warnings.filterwarnings("ignore")

class QSPRConfig:
    """Automatic search for config.yaml."""
    def __init__(self, config_name: str = "config.yaml"):
        self.config_path = self._find_config(config_name)
        self.config = self._load()
        #print(f" Config loaded: {self.config_path.absolute()}")
    
    def _find_config(self, config_name: str) -> Path:
        here = Path(__file__).parent / config_name
        if here.exists(): return here
        qspr_dir = Path(__file__).parent.parent / "QSPR" / config_name  
        if qspr_dir.exists(): return qspr_dir
        cwd = Path.cwd() / config_name
        if cwd.exists(): return cwd
        raise FileNotFoundError(f" config.yaml not found near {__file__}")

    def _load(self) -> Dict[str, Any]:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        if '.' not in key_path:
            return self.config.get(key_path, default)
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value or default

CONFIG = QSPRConfig()

TARGET = 'ee'

import sys
from pathlib import Path
from datetime import datetime

# Add module folder to path
module_path = Path(__file__).parent.parent / 'module'
sys.path.insert(0, str(module_path))

# Import from models_creation module
from models_creation import (
    select_features_advanced,
    prepare_model,
    print_feature_correlations,
    drop_high_nan_features,
    remove_low_variance_features,
    drop_highly_correlated_features,
    BaselineModels,
    HyperparameterOptimizer,
    RigorousValidator,

)

from hyperparameters import HyperparametersDisplay

from calculate_descriptors import prepare_data

import pandas as pd
import numpy as np
# =============================================================================
# DUAL WRITER: SAVES CONSOLE OUTPUT TO FILE
# =============================================================================

class DualWriter:
    """Dual output writer: console + log file simultaneously.
    
    Args:
        filename (str): Log file path for persistent output.
    
    Methods:
        write(msg): Write to both terminal and file
        flush(): Immediate flush for real-time logging
        close(): Close log file
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# =============================================================================
# COMPLETE ANALYSIS WORKFLOW CLASS
# =============================================================================

class AdvancedQSPRAnalysis:
    """Complete 7-step QSPR pipeline with chemical descriptors and ML.
    
    **Pipeline Steps**:
    1. Load data + compute descriptors (Mordred/Morgan/Sterimol/CircuS)
    2. Feature selection (consensus voting: Permutation/RFECV/Mutual Information on TRAIN only)
    3. Baseline comparisons (dummy + target shuffling)
    4. Hyperparameter optimization (GridSearchCV ×7 models)
    5. Train models + CV/bootstrap/AOA validation
    6. Compile results table (Excel export)
    7. Uncertainty quantification + statistical significance
    
    Args:
        data_frame (str): data path (SMILES + target)
        target_col (str): Target property column name
        descriptor_dir (str): Output directory for descriptors
        include_circus (bool): Add DOPtools CircuS fragments
        include_chyline (bool): Add DOPtools ChyLine fragments
    """
    
    def __init__(self, 
                data_frame: Optional[str] = None,
                target_col: Optional[str] = None,
                descriptor_dir: Optional[str] = None,
                include_circus: Optional[bool] = None,
                include_chyline: Optional[bool] = None):
        """Config-driven initialization."""

        # Config defaults
        self.data_frame = data_frame or CONFIG.get('data.file')
        self.target_col = target_col or CONFIG.get('data.target_col')
        self.descriptor_dir = descriptor_dir or CONFIG.get('paths.descriptors')

        self.name = 'models'
        self.mode = 'models_creation'
        
        desc_config = CONFIG.get('descriptors')

        circus_config = desc_config.get('circus', {})
        self.include_circus = circus_config.get('enabled', False)
        
        chyline_config = desc_config.get('chyline', {})
        self.include_chyline = chyline_config.get('enabled', False)
        
        self.results = {}

    
    def step1_load_and_prepare_data(self):
        """Step 1: Load CSVs + compute comprehensive chemical descriptors.
        
        Computes 5 descriptor families on combined train/test data:
        - Mordred: 1600+ 2D/3D physicochemical properties
        - Morgan FP: 2048-bit radius=2 fingerprints
        - Sterimol: L/B1/B5 steric parameters for substituents (R1, R2)
        - Electronic: Site-specific electronegativity, partial charges, and polarity gradients
        - ChemBERTa: 384-dim pretrained embeddings (optional)
        - CircuS/ChyLine: Fragment counts (circular/linear substructures)
        
        Returns:
            tuple: (train_df, test_df) with computed descriptors
        """
        print("\n" + "█"*75)
        print(" STEP 1: DATA LOADING & DESCRIPTOR CALCULATION ".center(73))
        print("█"*75)
        
        print(f"\nLoading data from: {self.data_frame}")
        print(f"Including CircuS descriptors: {self.include_circus}")
        print(f"Including ChyLine descriptors: {self.include_chyline}")
        
        self.train_data, self.test_data = prepare_data(
            data=self.data_frame,
            target_col=self.target_col,
            name=self.name,
            mode=self.mode,
            output_dir=self.descriptor_dir
            )
        # Assuming self.train_data and self.test_data are your DataFrames
        self.data = pd.concat([self.train_data, self.test_data], ignore_index=True) #keep original order
        print(f"\nOK: Data loaded: {self.data.shape[0]} samples × {self.data.shape[1]} features")
        print(f"  Descriptor breakdown:")
        print(f"    - Mordred: ~1600+ 2D/3D descriptors")
        print(f"    - Morgan FP: 2048-bit (radius=2)")
        print(f"    - Sterimol: L, B1, B5 parameters")
        print(f"    - Electronic: Electronegativity, Partial Charges, Polarity Diffs")
        print(f"    - ChemBERTa: 384-dim pretrained embeddings")
        if self.include_circus:
            print(f"    - CircuS: Fragment-count descriptors")
        if self.include_chyline:
            print(f"    - ChyLine: Linear fragment-count descriptors")
        print(f"\n Target range: [{self.data[self.target_col].min():.1f}, {self.data[self.target_col].max():.1f}]")
        print(f" Target std: {self.data[self.target_col].std():.2f}")
        print(f" Target mean: {self.data[self.target_col].mean():.2f}")
        
        return self

    
    def step2_feature_selection(self, method='consensus', n_features=20, mode='fast_aggr', backfill=False, **kwargs):
        """Step 2: 3-stage feature selection (TRAINING DATA ONLY).
    
        **Stage 1 - Pre-filtering** (Aggressive, unscaled):
        - Drop >20% NaN columns
        - Remove <1% variance features  
        - Eliminate |corr|>0.95 collinear pairs
        
        **Stage 2 - Advanced selection** (model-aware):
        | Method | Description |
        |--------|-------------|
        | consensus | ≥2 methods agree |
        | permutation | GB permutation importance (CV-based) |
        | rfecv | Recursive Elimination with Cross-Validation |
        | mi | mutual information |
        
        **Stage 3**: Print feature-target correlations (Pearson)
        
        Args:
            method (str): Selection method ('consensus', 'rfe', etc.)
            nfeatures (int): Target number of final features
        
        Returns:
            self: Updated with self.selected_features, self.correlation_df
        """
        print("\n" + "█"*75)
        print(" STEP 2: ADVANCED FEATURE SELECTION (CONSENSUS VOTING) ".center(73))
        print("█"*75)

        print(f"\nConfiguration:")
        print(f"├─ Method: {method}")
        print(f"├─ Target Features: {n_features}")
        if kwargs:
            print(f"└─ Tuning Params: {kwargs}")
        
        print("\nSelecting features using multiple methods:")
        if mode == "fast_aggr":
            print("├─ Permutation importance (Single Split)")
            print("├─ RFE (Recursive Elimination)")
            print("├─ Mutual Information")
        else:  # cv_safe
            print("├─ Permutation importance (5-Fold CV)")
            print("├─ RFECV (Recursive Elimination + CV)")
            print("├─ Mutual Information") #F-regression (Linear)
        print("└─ CONSENSUS (≥2 methods must agree)")
        
        X = self.data.drop(self.target_col, axis=1)
        
        train_data_filtered = self.train_data.copy()

        train_data = train_data_filtered
        train_data[self.target_col] = self.train_data[self.target_col]
        

        # ===== PRE-FILTERING PIPELINE (on training data only) =====
    
        print("\n" + "="*75)
        print("PRE-FILTERING PIPELINE (Training Data Only)")
        print("="*75)
        
        print(f"\nStarting with: {train_data.shape[0]} samples × {train_data.shape[1]-1} features")
        
        max_nan_ratio = CONFIG.get('filters.nan_threshold', 0.2)
        # Step 1: Drop high-NaN features
        train_data, cols1 = drop_high_nan_features(
            train_data,
            target_col=self.target_col,  # ← PROTECTS 'ee'
            max_nan_ratio=max_nan_ratio)
        
        threshold = CONFIG.get('filters.variance_threshold', 0.01)
        # Step 2: Remove low-variance features
        train_data, cols2 = remove_low_variance_features(
            train_data,
            target_col=self.target_col,  # ← PROTECTS 'ee'
            threshold=threshold)
        
        threshold = CONFIG.get('filters.correlation_threshold', 0.95)
        # Step 3: Drop highly correlated features
        train_data, cols3 = drop_highly_correlated_features(
            train_data,
            target_col=self.target_col,  # ← PROTECTS 'ee'
            threshold=threshold)

        
        print(f"\n[Summary] After all filters: {train_data.shape[0]} samples × {train_data.shape[1]-1} features")
        print(f"[Summary] Reduction: {X.shape[1]} → {train_data.shape[1]-1} ({100*(1 - (train_data.shape[1]-1)/X.shape[1]):.1f}%)")
        
        # ===== ADVANCED FEATURE SELECTION ON FILTERED DATA =====


        self.selected_features = select_features_advanced(train_data,             # <- ONLY TRAINING
                                                          self.target_col, 
                                                          method=method, 
                                                          n_features=n_features,
                                                          mode=mode,
                                                          backfill=backfill, 
                                                          **kwargs)
        
        self.method_name = method
        self.current_mode = mode
        self.backfill = backfill

        print(f"\n>> Selected features: {len(self.selected_features)}")
        print(f"  Removed: {self.data.shape[1] - len(self.selected_features) - 1}")
        print(f"  Ratio: {100 * len(self.selected_features) / (self.data.shape[1]-1):.1f}%")
        print(f"\n  Selected features list:")
        for idx, feat in enumerate(self.selected_features['feature'], 1):
            print(f"  {idx:2d}. {feat}")
        
        # Feature correlations
        print("\n" + "="*75)
        self.correlation_df = print_feature_correlations(
            train_data,
            self.selected_features, self.target_col
        )
        
        return self
    
    def step3_baseline_comparison(self):
        """Step 3: Statistical baselines using selected features.
    
        **Baselines**:
        1. **Dummy Mean**: Predicts training mean (absolute floor)
        2. **Target Shuffling**: 100x null hypothesis (strongest test) (done for final optimized model in step 5)
        
        Saves selected train/test data to parquet:
        ../Results/SelectedData/RS{rs}_{method}_{nfeat}_*.parquet
        
        Returns:
            dict: Baseline metrics (dummy_rmse, null_mean/std/distribution)
        """
        print("\n" + "█"*75)
        print(" STEP 3: BASELINE MODEL COMPARISONS ".center(73))
        print("█"*75)
        
        # Use SELECTED FEATURES ONLY (chosen from training data)
        selected_feature_list = list(self.selected_features['feature'])
        
        # Extract selected features from train/test sets (from Step 2)
        self.X_train_selected = self.train_data[selected_feature_list].fillna(self.train_data.median())
        self.X_test_selected = self.test_data[selected_feature_list].fillna(self.train_data.median()) #mean -> median

        self.X_train = self.X_train_selected
        self.X_test = self.X_test_selected

        self.y_train = self.train_data[self.target_col]
        self.y_test = self.test_data[self.target_col]

        # Define output directory and create it if it doesn't exist
        out_dir = Path(CONFIG.get('paths.results')) / "Selected_Data"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create a unique tag for the filename
        # NOTE: You must set self.current_method and self.current_nfeatures in your main loop
        tag = f"RS{self.random_state}_{self.current_method}_n{self.current_nfeatures}_{self.current_mode}_{self.backfill}"


        # Prepare DataFrames with features and target
        train_df_to_save = self.train_data[selected_feature_list].copy()
        train_df_to_save[TARGET] = self.y_train
        test_df_to_save = self.test_data[selected_feature_list].copy()
        test_df_to_save[TARGET] = self.y_test

        # Define file paths
        train_path = out_dir / f"{tag}_train_data.parquet"
        test_path = out_dir / f"{tag}_test_data.parquet"

        # Save to parquet
        train_df_to_save.to_parquet(train_path, index=True)
        test_df_to_save.to_parquet(test_path, index=True)

        print(f"\n[Data Saved] Training data -> {train_path}")
        print(f"[Data Saved] Test data     -> {test_path}")
        # =============================================================
        
        print(f"\nUsing SELECTED FEATURES only:")
        print(f"├─ Training set shape: {self.train_data.shape}")
        print(f"├─ Test set shape: {self.test_data.shape}")
        print(f"└─ Number of features: {self.train_data.shape}")
        print(f"\nNote: Features were selected from training data ONLY (Step 2)")
        print(f"      Test data was NOT used for feature selection\n")
        
        baselines = {}
        
        # Baseline 1: Dummy Mean
        print("1. Dummy Mean Predictor (predicts training mean):")
        dummy_rmse, dummy_r2 = BaselineModels.baseline_dummy_mean(
            self.y_train, self.y_test)
        baselines['dummy'] = {'RMSE': dummy_rmse, 'R2': dummy_r2}
        print(f"   RMSE: {dummy_rmse:.2f}")
        print(f"   R²: {dummy_r2:.4f}")
        self.baselines = baselines
 
        return baselines

    
    def step4_hyperparameter_optimization(self, algorithms=['RandomForestRegressor', 'XGBoost', 'GradientBoosting', 'Ridge', 'DecisionTree', 'KNN', 'SVR']):
        """Step 4: GridSearchCV hyperparameter optimization (7 models).
    
        **Models**:
        | Algorithm |
        |-----------|
        | RandomForest |
        | XGBoost |
        | GradientBoosting |
        | Ridge |
        | DecisionTree |
        | KNN |
        | SVR |
        
        Args:
            algorithms (list): ML models to optimize
        
        Returns:
            dict: {model_name: (best_params, best_score, n_combinations)}
        """ 
        print("\n" + "█"*75)
        print(" STEP 4: HYPERPARAMETER OPTIMIZATION (GRIDSEARCHCV) ".center(73))
        print("█"*75)
        
        opt_results = {}
        
        if 'RandomForestRegressor' in algorithms:
            print("\n1.  Optimizing Random Forest hyperparameters...")
            HyperparametersDisplay.print_hyperparameters('RANDOM_FOREST')
            rf_opt = HyperparameterOptimizer.optimize_random_forest(
                self.X_train, self.y_train, cv=5)
            opt_results['RandomForestRegressor'] = rf_opt
            
            print(f"    Best n_estimators: {rf_opt['best_params']['model__n_estimators']}")
            print(f"    Best max_depth: {rf_opt['best_params']['model__max_depth']}")
            print(f"    Best max_features: {rf_opt['best_params']['model__max_features']}")
            print(f"    Best bootstrap: {rf_opt['best_params']['model__bootstrap']}")
            print(f"    Best min_samples_split: {rf_opt['best_params']['model__min_samples_split']}")
            print(f"    Best min_samples_leaf: {rf_opt['best_params']['model__min_samples_leaf']}")
            print(f"    CV RMSE: {-rf_opt['best_score']:.4f}")
            print(f"    Combinations tested: {rf_opt['n_combinations']}")

        if 'XGBoost' in algorithms:
            print("\n2.  Optimizing XGBoost hyperparameters...")
            HyperparametersDisplay.print_hyperparameters('XGBOOST')
            xgb_opt = HyperparameterOptimizer.optimize_xgboost(
                self.X_train, self.y_train, cv=5)
            opt_results['XGBoost'] = xgb_opt
            
            print(f"    Best n_estimators: {xgb_opt['best_params']['model__n_estimators']}")
            print(f"    Best learning_rate: {xgb_opt['best_params']['model__learning_rate']}")
            print(f"    Best colsample_bytree: {xgb_opt['best_params']['model__colsample_bytree']}")
            print(f"    Best max_depth: {xgb_opt['best_params']['model__max_depth']}")
            print(f"    Best subsample: {xgb_opt['best_params']['model__subsample']}")
            print(f"    Best reg_alpha: {xgb_opt['best_params']['model__reg_alpha']}")
            print(f"    Best reg_lambda: {xgb_opt['best_params']['model__reg_lambda']}")
            print(f"    Best min_child_weight: {xgb_opt['best_params']['model__min_child_weight']}")
            print(f"    CV RMSE: {-xgb_opt['best_score']:.4f}")
            print(f"    Combinations tested: {xgb_opt['n_combinations']}")

        if 'GradientBoosting' in algorithms:
            print("\n3.  Optimizing Gradient Boosting hyperparameters...")
            HyperparametersDisplay.print_hyperparameters('GRADIENT_BOOSTING')
            gb_opt = HyperparameterOptimizer.optimize_gradient_boosting(
                self.X_train, self.y_train, cv=5)
            opt_results['GradientBoosting'] = gb_opt
            
            print(f"    Best n_estimators: {gb_opt['best_params']['model__n_estimators']}")
            print(f"    Best learning_rate: {gb_opt['best_params']['model__learning_rate']}")
            print(f"    Best max_depth: {gb_opt['best_params']['model__max_depth']}")
            print(f"    Best subsample: {gb_opt['best_params']['model__subsample']}")
            print(f"    Best max_features: {gb_opt['best_params']['model__max_features']}")
            print(f"    Best min_samples_split: {gb_opt['best_params']['model__min_samples_split']}")
            print(f"    Best min_samples_leaf: {gb_opt['best_params']['model__min_samples_leaf']}")
            print(f"    CV RMSE: {-gb_opt['best_score']:.4f}")
            print(f"    Combinations tested: {gb_opt['n_combinations']}")

        if 'Ridge' in algorithms:
            print("\n4.  Optimizing Ridge Regression (Multiple Linear Regression)...")
            HyperparametersDisplay.print_hyperparameters('RIDGE_REGRESSION')
            ridge_opt = HyperparameterOptimizer.optimize_ridge_regression(
                self.X_train, self.y_train, cv=5)
            opt_results['Ridge'] = ridge_opt
            
            print(f"    Best alpha: {ridge_opt['best_params']['model__alpha']}")
            print(f"    Best solver: {ridge_opt['best_params']['model__solver']}")
            print(f"    Fit intercept: {ridge_opt['best_params']['model__fit_intercept']}")
            print(f"    CV RMSE: {-ridge_opt['best_score']:.4f}")
            print(f"    Combinations tested: {ridge_opt['n_combinations']}")

        if 'DecisionTree' in algorithms:
            print("\n5.  Optimizing Decision Tree Regressor...")
            HyperparametersDisplay.print_hyperparameters('DECISION_TREE')
            dt_opt = HyperparameterOptimizer.optimize_decision_tree(
                self.X_train, self.y_train, cv=5)
            opt_results['DecisionTree'] = dt_opt
            
            print(f"    Best max_depth: {dt_opt['best_params']['model__max_depth']}")
            print(f"    Best min_samples_split: {dt_opt['best_params']['model__min_samples_split']}")
            print(f"    Best criterion: {dt_opt['best_params']['model__criterion']}")
            print(f"    Best splitter: {dt_opt['best_params']['model__splitter']}")
            print(f"    Best max_features: {dt_opt['best_params']['model__max_features']}")
            print(f"    CV RMSE: {-dt_opt['best_score']:.4f}")
            print(f"    Combinations tested: {dt_opt['n_combinations']}")

        if 'KNN' in algorithms:
            print("\n6.  Optimizing K-Nearest Neighbors Regressor...")
            HyperparametersDisplay.print_hyperparameters('KNN')
            knn_opt = HyperparameterOptimizer.optimize_knn(
                self.X_train, self.y_train, cv=5)
            opt_results['KNN'] = knn_opt
            
            print(f"    Best n_neighbors: {knn_opt['best_params']['model__n_neighbors']}")
            print(f"    Best weights: {knn_opt['best_params']['model__weights']}")
            print(f"    Best algorithm: {knn_opt['best_params']['model__algorithm']}")
            print(f"    Best metric: {knn_opt['best_params']['model__metric']}")
            print(f"    CV RMSE: {-knn_opt['best_score']:.4f}")
            print(f"    Combinations tested: {knn_opt['n_combinations']}")

        if 'SVR' in algorithms:
            print("\n7.  Optimizing Support Vector Regressor...")
            HyperparametersDisplay.print_hyperparameters('SVM')
            svr_opt = HyperparameterOptimizer.optimize_svr(
                self.X_train, self.y_train, cv=5)
            opt_results['SVR'] = svr_opt
            
            print(f"    Best C: {svr_opt['best_params']['model__C']}")
            print(f"    Best kernel: {svr_opt['best_params']['model__kernel']}")
            print(f"    Best gamma: {svr_opt['best_params']['model__gamma']}")
            print(f"    Best epsilon: {svr_opt['best_params']['model__epsilon']}")
            print(f"    CV RMSE: {-svr_opt['best_score']:.4f}")
            print(f"    Combinations tested: {svr_opt['n_combinations']}")

        self.hyperopt_results = opt_results
        return opt_results
    
    def step5_train_models(self, algorithms=['RandomForestRegressor', 'XGBoost', 'GradientBoosting', 'Ridge', 'DecisionTree', 'KNN', 'SVR']):
        """Step 5: Train optimized models + comprehensive validation."""
        print("\n" + "█"*75)
        print(" STEP 5: TRAINING ML ALGORITHMS ".center(73))
        print("█"*75)
        
        # =====================================================================
        # OPTIMIZATION: PRE-CALCULATE AOA (Feature space is constant)
        # =====================================================================
        print("\n[Pre-computation] Calculating Area of Applicability (AOA)...")
        
        # We need a validator instance, but the model doesn't matter for AOA.
        # Passing None as model is safe for AOA calculation.
        aoa_validator = RigorousValidator(
            self.X_train_selected, self.X_test_selected,
            self.y_train, self.y_test,
            model=None, 
            method_name=f"{self.current_method}_COMMON", # Title for the plot
            mode=self.current_mode, 
            backfill=self.backfill
        )
        # This generates ONE PDF file for the feature set
        shared_aoa_results = aoa_validator.area_of_applicability()
        print("OK: AOA calculated and saved (will be reused for all models).")
        # =====================================================================

        model_results = []
        
        for idx, algo in enumerate(algorithms, 1):
            print(f"\n{idx}. Training {algo}...")
            
            # 1. Get Pipeline
            optimized_pipeline = None
            if algo in self.hyperopt_results:
                optimized_pipeline = self.hyperopt_results[algo]['model']
            
            # 2. Create display name (Algorithm + Selection Method)
            combo_name = f"{algo} ({self.current_method})"
            
            # 3. Train & Validate (Injecting precomputed AOA)
            result = prepare_model(
                self.X_train_selected, self.X_test_selected,
                self.y_train, self.y_test,
                cv_folds=5, 
                prefit_model=optimized_pipeline, 
                method_name=combo_name,
                mode=self.current_mode, 
                backfill=self.backfill,
                precomputed_aoa=shared_aoa_results # <--- PASSING AOA HERE
            )
            
            # 4. Save Model File
            model_out_dir = Path(CONFIG.get('paths.models'))
            model_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Unique tag for filename
            tag = f"RS{self.random_state}_{self.current_method}_n{self.current_nfeatures}_{self.current_mode}_{self.backfill}_{algo}"
            model_path = model_out_dir / f"{tag}_model.pkl"

            import joblib
            joblib.dump(result['model'], model_path)
            result['model_path'] = str(model_path)
            print(f"   [Model Saved] -> {model_path}")
            
            # 5. Extract & Log Stats
            ts_stats = result.get('target_shuffling', {})
            
            result['algorithm'] = algo
            result['p_value'] = ts_stats.get('p_value', 1.0)
            result['significant'] = ts_stats.get('significant', False)
            result['truly_significant'] = ts_stats.get('truly_significant', False)

            print(f"  Train RMSE: {result['train_rmse']:.2f} (R²={result['train_r2']:.2f}, Train MAE: {result['train_mae']:.2f})")
            print(f"  Test RMSE:  {result['test_rmse']:.2f} (R²={result['test_r2']:.2f}, Test MAE: {result['test_mae']:.2f})")
            print(f"  CV RMSE:    {result['cv_rmse_mean']:.2f} ± {result['cv_rmse_std']:.2f}")
            print(f"  CV MAE:     {result['cv_mae_mean']:.2f}±{result['cv_mae_std']:.2f}")
            print(f"  RMSE/MAE ratio:     {result['test_rmse']/result['test_mae']:.2f}")

            if ts_stats:
                null_mean = ts_stats.get('null_rmse_mean', 0.0)
                null_std = ts_stats.get('null_rmse_std', 0.0)
                p_val = ts_stats.get('p_value', 1.0)
                null_min = ts_stats.get('null_rmse_min', 100)
                null_max = ts_stats.get('null_rmse_max', 1000)
                z_score = ts_stats.get('z_score', 0.0)
                
                print(f"  Null RMSE:  {null_mean:.2f} ± {null_std:.2f} (Target Shuffling Baseline)")
                threshold = ts_stats.get('threshold_good', 0.0)
                print(f"  Threshold:  < {threshold:.2f} (Required for True Significance)")
                print(f"  Target shuffling minimal RMSE: {null_min:.2f}; maximal RMSE: {null_max:.2f}")
                print(f"  Z-score (How many sigmas is our model away from noise mean?): {z_score:.2f} {'(PASS: >2σ)' if z_score > 2 else '(FAIL)'}")
                
                if ts_stats.get('truly_significant', False):
                    print(f"   Status:      TRULY SIGNIFICANT (p={p_val:.2e})")
                elif ts_stats.get('significant', False):
                    print(f"  Status:      Significant (p={p_val:.2e})")
                else:
                    print(f"  Status:      Not significant (p={p_val:.2e})")
            else:
                print("  Status:     Target Shuffling failed or skipped.")
        
            model_results.append(result)
        
        self.model_results = model_results
        return model_results
    
    def step6_compile_results(self):
        """Step 6: Comprehensive results table compilation.
    
        Creates Excel sheet with:
        | Algorithm | TrainR² | TestR² | CVR²±σ | RMSE±σ | MAE±σ | p-value | Significant |
        
        Ranks models by TestR², highlights top-3, exports to Excel.
        
        Returns:
            pd.DataFrame: Complete results table
        """
        print("\n" + "█"*75)
        print(" STEP 6: RESULTS COMPILATION & ANALYSIS ".center(73))
        print("█"*75)
        
        results_data = []
        for result in self.model_results:
            results_data.append({
                'Algorithm': result['algorithm'],
                'Train_R2': result['train_r2'],
                'Test_R2': result['test_r2'],
                'Train_RMSE': result['train_rmse'],
                'Test_RMSE': result['test_rmse'],
                'Train_MAE': result['train_mae'],
                'Test_MAE': result['test_mae'],
                'CV_R2_Mean': result['cv_r2_mean'],
                'CV_R2_Std': result['cv_r2_std'],
                'CV_RMSE_Mean': result['cv_rmse_mean'],
                'CV_RMSE_Std': result['cv_rmse_std'],
                'CV_MAE_Mean': result['cv_mae_mean'],
                'CV_MAE_Std': result['cv_mae_std'],
                'P_Value': result['p_value'],
                'Significant': result['significant'],
                'Truly_Significant': result.get('truly_significant', False),
            })
        
        self.results_df = pd.DataFrame(results_data)
        
        # Print summary table
        print("\nTOP 3 MODELS (Ranked by Test RMSE):")
        print("-" * 95)
        print(f"{'Algorithm':<25} | {'RMSE':<8} | {'R2':<6} | {'p-value':<10} | {'Status'}")
        print("-" * 95)
        
        for idx, row in self.results_df.nsmallest(3, 'Test_RMSE').iterrows():
            if row['Truly_Significant']:
                status = "High Significance"
            elif row['Significant']:
                status = "Significant"
            else:
                status = "Not Significant"
            
            print(f"{row['Algorithm']:<25} | {row['Test_RMSE']:.3f}    | {row['Test_R2']:.2f}   | {row['P_Value']:.1e}  | {status}")
        
        # Baseline comparison
        print("\nBASELINE COMPARISON:")
        print("-" * 70)
        
        if 'dummy' in self.baselines:
            dummy_rmse = self.baselines['dummy']['RMSE']
            best_rmse = self.results_df['Test_RMSE'].min()
            improvement = 100 * (dummy_rmse - best_rmse) / dummy_rmse
            
            print(f"Dummy Mean RMSE:  {dummy_rmse:.2f}")
            print(f"Best Model RMSE:  {best_rmse:.2f}")
            print(f"Improvement:      {improvement:.1f}%")
        else:
            print("Baseline Dummy Mean not available.")
        
        return self
    
    def step7_uncertainty_and_aoa(self):
        """Step 7: Uncertainty quantification + AOA analysis.
    
        Computes:
        - Bootstrap confidence intervals (2.5-97.5%)
        - Area of Applicability (Delaunay triangulation PCA-2D)
        - Interpolative % (inside training hull)
        - Extrapolative % (outside hull)
        - Statistical significance (t-test vs null)
        
        Generates high-res PDF: applicability_domain.pdf
        """
        print("\n" + "█"*75)
        print(" STEP 7: UNCERTAINTY QUANTIFICATION & AOA ANALYSIS ".center(73))
        print("█"*75)


        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to analyze.")
            return

        # 1. Identify Best Model (by Test RMSE, safer than R2 for small data)
        # Using RMSE because R2 can be negative for poor models
        best_idx = self.results_df['Test_RMSE'].idxmin() 
        
        # We need to map back to the list index (assuming order is preserved)
        # Or search in self.model_results by algorithm name
        best_algo_name = self.results_df.loc[best_idx, 'Algorithm']
        best_result = next((r for r in self.model_results if r['algorithm'] == best_algo_name), None)
        
        if not best_result:
            print(f"Error: Could not find detailed results for {best_algo_name}")
            return

        print(f"\nWinning Algorithm: {best_algo_name}")
        print("-" * 40)
        
        # 2. Bootstrap Reporting
        boot = best_result.get('bootstrap')
        if boot:
            print("\nBootstrap Uncertainty (95% CI on Predictions):")
            print("Sample predictions for first 3 test molecules:")
            
            # Safe loop limit
            n_samples = min(3, len(boot['median']))
            
            for i in range(n_samples):
                # Retrieve actual test value for comparison
                true_val = self.y_test.iloc[i]
                pred_median = boot['median'][i]
                lower = boot['lower_95'][i]
                upper = boot['upper_95'][i]
                width = upper - lower
                
                print(f"  Mol {i+1}: True={true_val:.1f} | Pred={pred_median:.1f} "
                      f"(95% CI: {lower:.1f} - {upper:.1f}, Width: {width:.1f})")
        else:
            print("\nBootstrap results not available.")
        
        # 3. AOA Reporting
        aoa = best_result.get('aoa')
        if aoa:
            print("\nArea of Applicability (AOA) Status:")
            print(f"  Interpolative (Reliable):  {aoa['pct_interpolative']:.1f}% of test set")
            print(f"  Extrapolative (Uncertain): {aoa['pct_extrapolative']:.1f}% of test set")
            print(f"  (See generated PDF file for visual plot)")
        else:
            print("\nAOA results not available.")
            
        return best_result
    
    def run_complete_analysis(self, 
                              method='consensus', 
                              n_features=20, 
                              mode='fast_aggr', 
                              backfill=False, 
                              preselected_features: Optional[List[str]] = None,
                              **kwargs):
        """Execute the full 7-step QSPR regression pipeline.
        
        If preselected_features is provided, Step 2 (feature selection) is completely skipped.
        
        Args:
            method: Feature selection method (ignored when using preselected_features)
            n_features: Target number of features (ignored when using preselected)
            mode: Selection mode ('fast_aggr' or 'cv_safe')
            backfill: Whether to backfill features in consensus selection
            preselected_features: Optional list of feature names to use directly
            **kwargs: Additional arguments passed to feature selection
            
        Returns:
            self
        """
        print("\n" + "═" + "═"*73 + "═")
        print(" ADVANCED QSPR ANALYSIS - COMPLETE PIPELINE (REGRESSION) ".center(73))
        print("═" + "═"*73 + "═")
        
        self.step1_load_and_prepare_data()
        
        # ====================== PRESELECTED FEATURES MODE ======================
        if preselected_features is not None and len(preselected_features) > 0:
            print("\n" + "█"*75)
            print(" PRESELECTED FEATURES MODE - SKIPPING STEP 2 ".center(73))
            print("█"*75)
            
            # Check which features actually exist in the loaded data
            available = set(self.data.columns) - {self.target_col}
            valid_features = [f for f in preselected_features if f in available]
            
            if len(valid_features) < len(preselected_features):
                print(f"WARNING: {len(preselected_features) - len(valid_features)} "
                      f"features not found and were skipped.")
            
            self.selected_features = pd.DataFrame({'feature': valid_features})
            self.method_name = 'preselected'
            self.current_mode = 'preselected'
            self.backfill = False
            
            print(f"→ Using {len(valid_features)} pre-selected features (Step 2 skipped).")
            
            # Show correlations with target (same as normal flow)
            print("\n" + "="*75)
            self.correlation_df = print_feature_correlations(
                self.train_data, self.selected_features, self.target_col
            )
            
        else:
            # Normal flow with feature selection
            self.step2_feature_selection(method=method, n_features=n_features, 
                                       mode=mode, backfill=backfill, **kwargs)
        
        # Continue pipeline (unchanged)
        actual_features = len(self.selected_features)
        if actual_features < 2:
            print("\n" + "!"*75)
            print(f" WARNING: Only {actual_features} feature(s) available.".center(73))
            print(" SKIPPING remaining steps for this configuration.".center(73))
            print("!"*75 + "\n")
            return self 
        
        self.step3_baseline_comparison()
        self.step4_hyperparameter_optimization()
        self.step5_train_models()
        self.step6_compile_results()
        self.step7_uncertainty_and_aoa()
        
        return self
    
    def export_results(self, output_file='QSPR_Results.xlsx'):
        """Export results to Excel"""
        if not hasattr(self, 'results_df') or self.results_df is None or self.results_df.empty:
            print(f"\n SKIPPING EXPORT: No results generated for {output_file} (likely due to insufficient features).")

        with pd.ExcelWriter(output_file) as writer:
            self.results_df.to_excel(writer, sheet_name='All_Results', index=False)

            # Feature selection results
            self.selected_features.to_excel(writer, sheet_name='Selected_Features', index=False)
            
            # Hyperparameter optimization results
            if self.hyperopt_results:
                hp_data = []
                for algo, result in self.hyperopt_results.items():
                    hp_data.append({
                        'Algorithm': algo,
                        'Best_Params': str(result['best_params']),
                        'CV_RMSE_Best': -1 * result['best_score'],
                    })
                pd.DataFrame(hp_data).to_excel(writer, sheet_name='Hyperparameter_Optimization', index=False)
        
        print(f"\nOK: Results exported to {output_file}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Set up console output logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_path = CONFIG.get('paths.logs')
    log_filename = f"{logs_path}/QSPR_Analysis_{timestamp}.txt"
    
    # Create Results directory if it doesn't exist
    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Redirect stdout to both console and file
    sys.stdout = DualWriter(log_filename)
    sys.stderr = sys.stdout
    
    print(f"Console output saved to: {log_filename}\n")

    PRESELECTED_FEATURES = ['IC1', 'PEOE_VSA6', 'CIC1', 'GATS6are', 'SdssC', 'GATS4d'
] # 7 fast aggr
    
    '''
        'IC1',
                                'PEOE_VSA6',
                                'SMR_VSA6',
                                'CIC1',
                                'GATS6are',
                                'GATS4d'
    ] # 6 fast aggr
    '''

    '''
    'SlogP_VSA2',
                                'IC1',
                                'PEOE_VSA6',
                                'CIC1',
                                'GATS6are',
                                'SdssC'
    ] # 6 cv safe

    '''


    print("\n" + "═"*80)
    print(f" SINGLE RUN – PRE-SELECTED FEATURES ({len(PRESELECTED_FEATURES)})".center(78))
    print("═"*80 + "\n")

    try:
        analysis = AdvancedQSPRAnalysis()
        analysis.random_state = 42
        analysis.current_method = "preselected"
        analysis.current_nfeatures = len(PRESELECTED_FEATURES)
        analysis.current_mode = "preselected"
        analysis.backfill = False

        analysis.run_complete_analysis(preselected_features=PRESELECTED_FEATURES)

        if hasattr(analysis, 'results_df') and not analysis.results_df.empty:
            output_file = f"../Results/QSPR_Preselected_{timestamp}.xlsx"
            analysis.export_results(output_file)
            print(f"\nSaved → {output_file}")
        else:
            print("\nRun was skipped (likely < 2 valid features after checking)")

    except Exception as exc:
        print("\n" + "!"*80)
        print("ERROR during pre-selected features run".center(78))
        print("!"*80)
        print(str(exc))
        #traceback.print_exc()
        print("!"*80 + "\n")

    if isinstance(sys.stdout, DualWriter):
        sys.stdout.close()