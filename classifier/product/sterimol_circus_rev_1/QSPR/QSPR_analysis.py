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

TARGET = 'epox_cla'
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

# Review 1
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import xgboost as xgb
# Review 1
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
        1. **Dummy Majority-Class Predictor**: Predicts majority class for all
        2. **Target Shuffling**: 100x null hypothesis (strongest test)
        
        Saves selected train/test data to CSV:
        ../Results/SelectedData/RS{rs}_{method}_{nfeat}_*.csv
        
        Returns:
            dict: Baseline metrics (dummy_Accuracy/dummy_F1_score, null_accuracy/std/distribution)
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
        train_df_to_save['epox_cla'] = self.y_train
        test_df_to_save = self.test_data[selected_feature_list].copy()
        test_df_to_save['epox_cla'] = self.y_test

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
        
        # Baseline 1: Dummy majority-class
        print("1. Dummy Majority-Class Predictor")
        dummy_acc, dummy_f1, dummy_bal_acc = BaselineModels.baseline_dummy_majority(
            self.y_train, self.y_test
        )
        baselines['dummy'] = {'Accuracy': dummy_acc, 'F1': dummy_f1, 'Balanced_Acc': dummy_bal_acc}
        print(f" Accuracy: {dummy_acc:.2f}")
        print(f" F1-score: {dummy_f1:.2f}")
        print(f" Balanced Acc: {dummy_bal_acc:.2f}")
        print(" (Predicts majority class of training target for all samples)")

        self.baselines = baselines
 
        return baselines

    # Review 1:
    def set_fixed_hyperparameters(self, fixed_params):
        """
        Build fixed classification pipelines for reviewer reruns without GridSearchCV.

        Parameters
        ----------
        fixed_params : dict
            Dictionary keyed by algorithm labels used in self.step5trainmodels().
            Parameter names must include the pipeline prefix, e.g.:
            {"model__n_estimators": 500}.
        """

        estimators = {
            "RandomForestClassifier": RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            ),
            "GradientBoosting": GradientBoostingClassifier(
                random_state=42
            ),
            "LogisticRegression": LogisticRegression(
                random_state=42,
                max_iter=5000
            ),
            "DecisionTree": DecisionTreeClassifier(
                random_state=42
            ),
            "KNN": KNeighborsClassifier(),
            "SVC": SVC(
                probability=True,
                random_state=42
            ),
        }

        self.hyperopt_results = {}

        for algorithm, params in fixed_params.items():
            if algorithm not in estimators:
                raise ValueError(
                    f"Unknown classification algorithm: {algorithm}. "
                    f"Available algorithms: {list(estimators.keys())}"
                )

            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimators[algorithm]),
            ])

            pipeline.set_params(**params)

            self.hyperopt_results[algorithm] = {
                "best_params": params,
                "best_score": np.nan,
                "n_combinations": 1,
                "model": pipeline,
            }

        print(
            "\nUsing fixed historical classification hyperparameters; "
            "GridSearchCV was skipped."
        )


    def load_fixed_models(self, model_paths):
        """
        Load historical final classification pipelines and skip GridSearchCV.

        Parameters
        ----------
        model_paths : dict
            Dictionary in the form:
            {"RandomForestClassifier": "path/to/model.joblib"}.
        """
        import joblib

        self.hyperopt_results = {}

        for algorithm, path in model_paths.items():
            pipeline = joblib.load(path)

            if not isinstance(pipeline, Pipeline):
                raise TypeError(
                    f"Loaded object for '{algorithm}' is not a sklearn Pipeline."
                )

            self.hyperopt_results[algorithm] = {
                "best_params": pipeline.get_params(),
                "best_score": np.nan,
                "n_combinations": 1,
                "model": pipeline,
            }

        print(
            "\nLoaded historical frozen classification pipelines; "
            "GridSearchCV was skipped."
        )
    # Review 1   


    def step4_hyperparameter_optimization(self, algorithms=['RandomForest', 'XGBoost', 'GradientBoosting',
                'LogisticRegression', 'DecisionTree', 'KNN', 'SVC']):
        """Step 4: GridSearchCV hyperparameter optimization (7 models).
    
        **Models**:
        | Algorithm |
        |-----------|
        | RandomForest |
        | XGBoost |
        | GradientBoosting |
        | LogisticRegression |
        | DecisionTree |
        | KNN |
        | SVC |
        
        Args:
            algorithms (list): ML models to optimize
        
        Returns:
            dict: {model_name: (best_params, best_score, n_combinations)}
        """ 
        print("\n" + "█"*75)
        print("█" + " STEP 4: HYPERPARAMETER OPTIMIZATION (GRIDSEARCHCV, CLASSIFICATION) ".center(73) + "█")
        print("█"*75)
        
        opt_results = {}

        get_p = lambda res, key: res['best_params'].get(f'model__{key}', 'N/A')

        if 'RandomForest' in algorithms:
            print("\n1.  Optimizing Random Forest classifier hyperparameters...")
            HyperparametersDisplay.print_hyperparameters('RANDOM_FOREST')
            rf_opt = HyperparameterOptimizer.optimize_random_forest(
                self.X_train, self.y_train, cv=5)
            opt_results['RandomForest'] = rf_opt
            
            print(f"    Best n_estimators: {get_p(rf_opt, 'n_estimators')}")
            print(f"    Best max_features: {get_p(rf_opt, 'max_features')}")
            print(f"    Best max_depth:    {get_p(rf_opt, 'max_depth')}")
            print(f"    Best bootstrap:    {get_p(rf_opt, 'bootstrap')}")
            print(f"    Best min_sample_spl: {get_p(rf_opt, 'min_samples_split')}")
            print(f"    Best min_sample_lf:  {get_p(rf_opt, 'min_samples_leaf')}")
            print(f"    Best class_weight:   {get_p(rf_opt, 'class_weight')}")
            print(f"    CV Balanced Acc.:  {rf_opt['best_score']:.4f}")
            print(f"    Combinations tested: {rf_opt['n_combinations']}")

        if 'XGBoost' in algorithms:
            print("\n2.  Optimizing XGBoost (XGBClassifier) hyperparameters...")
            HyperparametersDisplay.print_hyperparameters('XGBOOST')
            xgb_opt = HyperparameterOptimizer.optimize_xgboost(
                self.X_train, self.y_train, cv=5)
            opt_results['XGBoost'] = xgb_opt
            
            print(f"    Best n_estimators: {get_p(xgb_opt, 'n_estimators')}")
            print(f"    Best learning_rate: {get_p(xgb_opt, 'learning_rate')}")
            print(f"    Best max_depth:    {get_p(xgb_opt, 'max_depth')}")
            print(f"    Best colsample_bytree: {get_p(xgb_opt, 'colsample_bytree')}")
            print(f"    Best subsample:    {get_p(xgb_opt, 'subsample')}")
            print(f"    Best reg_alpha:    {get_p(xgb_opt, 'reg_alpha')}")
            print(f"    Best reg_lambda:   {get_p(xgb_opt, 'reg_lambda')}")
            print(f"    Best min_child_wt: {get_p(xgb_opt, 'min_child_weight')}")
            print(f"    CV Balanced Acc.:  {xgb_opt['best_score']:.4f}")
            print(f"    Combinations tested: {xgb_opt['n_combinations']}")

        if 'GradientBoosting' in algorithms:
            print("\n3  Optimizing Gradient Boosting classifier hyperparameters...")
            HyperparametersDisplay.print_hyperparameters('GRADIENT_BOOSTING')
            gb_opt = HyperparameterOptimizer.optimize_gradient_boosting(
                self.X_train, self.y_train, cv=5)
            opt_results['GradientBoosting'] = gb_opt
            
            print(f"    Best n_estimators: {get_p(gb_opt, 'n_estimators')}")
            print(f"    Best learning_rate: {get_p(gb_opt, 'learning_rate')}")
            print(f"    Best max_depth:    {get_p(gb_opt, 'max_depth')}")
            print(f"    Best subsample:    {get_p(gb_opt, 'subsample')}")
            print(f"    Best max_features: {get_p(gb_opt, 'max_features')}")
            print(f"    Best min_sample_spl: {get_p(gb_opt, 'min_samples_split')}")
            print(f"    Best min_sample_lf:  {get_p(gb_opt, 'min_samples_leaf')}")
            print(f"    CV Balanced Acc.:  {gb_opt['best_score']:.4f}")
            print(f"    Combinations tested: {gb_opt['n_combinations']}")

        if 'LogisticRegression' in algorithms:
            print("\n4.  Optimizing Logistic Regression (Linear Classifier)...")
            HyperparametersDisplay.print_hyperparameters('LOGISTIC_REGRESSION')
            lr_opt = HyperparameterOptimizer.optimize_logistic_regression(
                self.X_train, self.y_train, cv=5)
            opt_results['LogisticRegression'] = lr_opt
            
            print(f"    Best C:            {get_p(lr_opt, 'C')}")
            print(f"    Best penalty:      {get_p(lr_opt, 'penalty')}")
            print(f"    Best solver:       {get_p(lr_opt, 'solver')}")
            print(f"    Best class_weight: {get_p(lr_opt, 'class_weight')}")
            print(f"    CV Balanced Acc.:  {lr_opt['best_score']:.4f}")
            print(f"    Combinations tested: {lr_opt['n_combinations']}")

        if 'DecisionTree' in algorithms:
            print("\n5.  Optimizing Decision Tree classifier...")
            HyperparametersDisplay.print_hyperparameters('DECISION_TREE')
            dt_opt = HyperparameterOptimizer.optimize_decision_tree(
                self.X_train, self.y_train, cv=5)
            opt_results['DecisionTree'] = dt_opt
            
            print(f"    Best max_depth:    {get_p(dt_opt, 'max_depth')}")
            print(f"    Best min_sample_spl: {get_p(dt_opt, 'min_samples_split')}")
            print(f"    Best min_sample_lf:  {get_p(dt_opt, 'min_samples_leaf')}")
            print(f"    Best criterion:    {get_p(dt_opt, 'criterion')}")
            print(f"    Best ccp_alpha:    {get_p(dt_opt, 'ccp_alpha')}")
            print(f"    Best class_weight: {get_p(dt_opt, 'class_weight')}")
            print(f"    CV Balanced Acc.:  {dt_opt['best_score']:.4f}")
            print(f"    Combinations tested: {dt_opt['n_combinations']}")

        if 'KNN' in algorithms:
            print("\n6.  Optimizing K-Nearest Neighbors classifier...")
            HyperparametersDisplay.print_hyperparameters('KNN')
            knn_opt = HyperparameterOptimizer.optimize_knn(
                self.X_train, self.y_train, cv=5)
            opt_results['KNN'] = knn_opt
            
            print(f"    Best n_neighbors:  {get_p(knn_opt, 'n_neighbors')}")
            print(f"    Best weights:      {get_p(knn_opt, 'weights')}")
            print(f"    Best metric:       {get_p(knn_opt, 'metric')}")
            print(f"    Best p:            {get_p(knn_opt, 'p')}")
            print(f"    CV Balanced Acc.:  {knn_opt['best_score']:.4f}")
            print(f"    Combinations tested: {knn_opt['n_combinations']}")

        if 'SVC' in algorithms:
            print("\n7. Optimizing Support Vector Classifier (SVC)...")
            HyperparametersDisplay.print_hyperparameters('SVM')
            svc_opt = HyperparameterOptimizer.optimize_svc(
                self.X_train, self.y_train, cv=5)
            opt_results['SVC'] = svc_opt
            
            print(f"    Best C:            {get_p(svc_opt, 'C')}")
            print(f"    Best kernel:       {get_p(svc_opt, 'kernel')}")
            print(f"    Best gamma:        {get_p(svc_opt, 'gamma')}")
            print(f"    Best class_weight: {get_p(svc_opt, 'class_weight')}")
            print(f"    CV Balanced Acc.:  {svc_opt['best_score']:.4f}")
            print(f"    Combinations tested: {svc_opt['n_combinations']}")

        self.hyperopt_results = opt_results
        return opt_results

    # Review round 1

    
    def export_reviewer_validation_outputs(self, result, tag):
        """
        Export reviewer-facing validation files for binary classification.

        Expected `result` keys
        ----------------------
        cv_fold_metrics:
            Output from RigorousValidator.cross_validation_analysis(), containing:
            fold, cv_acc, cv_bal_acc, cv_logloss and aggregate CV metrics.

        target_shuffling_bal_acc:
            One-dimensional null distribution of balanced accuracy.

        target_shuffling:
            Dictionary from RigorousValidator.statistical_test().

        prediction_uncertainty_bootstrap:
            Dictionary from bootstrap_confidence_intervals(), containing raw
            bootstrap probabilities and per-test-compound summary fields.

        X_test:
            External test-set descriptor DataFrame.
        """
        outdir = Path(
            CONFIG.get("paths.results", "Results")
        ) / "ReviewerValidation"
        outdir.mkdir(parents=True, exist_ok=True)

        def get_metric(metrics, *keys, required=True):
            """Return the first available matching metric from a dictionary."""
            for key in keys:
                if key in metrics:
                    return metrics[key]

            if required:
                raise KeyError(
                    f"Missing metric. Expected one of {list(keys)}. "
                    f"Available keys: {list(metrics.keys())}"
                )

            return None

        def validate_fold_vector(values, name, n_folds):
            """Ensure that a metric contains one numerical value per CV fold."""
            values = np.asarray(values, dtype=float)

            if values.ndim != 1:
                raise ValueError(
                    f"CV metric '{name}' must be one-dimensional; "
                    f"received shape {values.shape}."
                )

            if len(values) != n_folds:
                raise ValueError(
                    f"CV metric '{name}' contains {len(values)} values, "
                    f"but {n_folds} folds were found."
                )

            return values

        # ==============================================================
        # 1. Fold-level cross-validation metrics
        # ==============================================================
        cv_results = result["cv_fold_metrics"]

        folds = np.asarray(
            get_metric(cv_results, "fold", "folds"),
            dtype=int,
        )

        if folds.ndim != 1 or len(folds) == 0:
            raise ValueError(
                "'fold' must be a non-empty one-dimensional array."
            )

        n_folds = len(folds)

        accuracy = validate_fold_vector(
            get_metric(cv_results, "cv_acc", "accuracy", "acc"),
            "accuracy",
            n_folds,
        )

        balanced_accuracy = validate_fold_vector(
            get_metric(
                cv_results,
                "cv_bal_acc",
                "balanced_accuracy",
                "bal_acc",
            ),
            "balanced_accuracy",
            n_folds,
        )

        logloss = validate_fold_vector(
            get_metric(
                cv_results,
                "cv_logloss",
                "logloss",
                "log_loss",
            ),
            "log_loss",
            n_folds,
        )

        cv_df = pd.DataFrame({
            "fold": folds,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "log_loss": logloss,
        })

        # Future optional fold-wise metrics; add only when a valid vector exists.
        optional_cv_metrics = {
            "f1_score": ("cv_f1", "f1", "f1_score"),
            "mcc": ("cv_mcc", "mcc"),
            "roc_auc": ("cv_roc_auc", "roc_auc", "auc_roc"),
            "pr_auc": ("cv_pr_auc", "pr_auc", "average_precision"),
            "precision": ("cv_precision", "precision"),
            "sensitivity": ("cv_sensitivity", "sensitivity", "recall"),
            "specificity": ("cv_specificity", "specificity"),
        }

        for output_name, candidate_keys in optional_cv_metrics.items():
            metric_values = get_metric(
                cv_results,
                *candidate_keys,
                required=False,
            )

            if metric_values is None:
                continue

            cv_df[output_name] = validate_fold_vector(
                metric_values,
                output_name,
                n_folds,
            )

        cv_df.to_csv(
            outdir / f"{tag}_cv_fold_metrics.csv",
            index=False,
        )

        # ==============================================================
        # 2. Target shuffling / permutation-test distribution
        # ==============================================================
        shuffle_scores = np.asarray(
            result["target_shuffling_bal_acc"],
            dtype=float,
        )

        if shuffle_scores.ndim != 1 or shuffle_scores.size == 0:
            raise ValueError(
                "target_shuffling_bal_acc must be a non-empty "
                "one-dimensional array."
            )

        shuffle_metric = "balanced_accuracy"

        shuffle_df = pd.DataFrame({
            "shuffle_id": np.arange(1, len(shuffle_scores) + 1),
            shuffle_metric: shuffle_scores,
        })

        shuffle_df.to_csv(
            outdir / f"{tag}_target_shuffling_distribution.csv",
            index=False,
        )

        ts = result.get("target_shuffling", {})

        summary_df = pd.DataFrame([{
            "target_shuffling_metric": shuffle_metric,
            "observed_balanced_accuracy": result.get("test_bal_acc"),
            "null_balanced_accuracy_mean": float(np.mean(shuffle_scores)),
            "null_balanced_accuracy_std": float(
                np.std(shuffle_scores, ddof=1)
            ) if len(shuffle_scores) > 1 else 0.0,
            "null_balanced_accuracy_min": float(np.min(shuffle_scores)),
            "null_balanced_accuracy_max": float(np.max(shuffle_scores)),
            "empirical_one_sided_permutation_p": ts.get(
                "p_value",
                ts.get("empirical_p_value"),
            ),
            "z_score": ts.get("z_score"),
            "significantly_better_than_noise": ts.get(
                "significantly_better",
                ts.get("significant"),
            ),
            "n_shuffles": len(shuffle_scores),

            # Directly returned by prepare_model()
            "train_accuracy": result.get("train_acc"),
            "test_accuracy": result.get("test_acc"),
            "train_balanced_accuracy": result.get("train_bal_acc"),
            "test_balanced_accuracy": result.get("test_bal_acc"),
            "train_f1_macro": result.get("train_f1"),
            "test_f1_macro": result.get("test_f1"),
            "cv_accuracy_mean": result.get("cv_acc_mean"),
            "cv_accuracy_std": result.get("cv_acc_std"),
            "cv_balanced_accuracy_mean": result.get("cv_bal_acc_mean"),
            "cv_balanced_accuracy_std": result.get("cv_bal_acc_std"),
            "cv_logloss_mean": result.get("cv_logloss_mean"),
            "cv_logloss_std": result.get("cv_logloss_std"),

            # Values exported only if you later add them in prepare_model()
            "test_precision": result.get("test_precision"),
            "test_recall_sensitivity": result.get("test_recall"),
            "test_specificity": result.get("test_specificity"),
            "test_mcc": result.get("test_mcc"),
            "test_roc_auc": result.get("test_roc_auc"),
            "test_pr_auc": result.get("test_pr_auc"),
        }])

        summary_df.to_csv(
            outdir / f"{tag}_target_shuffling_summary.csv",
            index=False,
        )

        # ==============================================================
        # 3. Raw bootstrap P(positive class) for external test compounds
        # ==============================================================
        boot = result["prediction_uncertainty_bootstrap"]

        bootstrap_probabilities = boot.get("bootstrap_probabilities")

        if bootstrap_probabilities is None:
            raise KeyError(
                "Missing 'bootstrap_probabilities' in "
                "'prediction_uncertainty_bootstrap'."
            )

        bootstrap_probabilities = np.asarray(
            bootstrap_probabilities,
            dtype=float,
        )

        if bootstrap_probabilities.ndim != 2:
            raise ValueError(
                "bootstrap_probabilities must be a 2D array with shape "
                "(n_bootstrap_valid, n_test_compounds); "
                f"received {bootstrap_probabilities.shape}."
            )

        test_indices = np.asarray(
            boot.get("test_index", result["X_test"].index.to_numpy())
        )

        if bootstrap_probabilities.shape[1] != len(test_indices):
            raise ValueError(
                "Bootstrap probability matrix does not match the number of "
                "test observations: "
                f"{bootstrap_probabilities.shape[1]} columns vs "
                f"{len(test_indices)} test indices."
            )

        raw_bootstrap_df = pd.DataFrame(
            bootstrap_probabilities,
            columns=[f"test_{idx}" for idx in test_indices],
        )

        raw_bootstrap_df.insert(
            0,
            "bootstrap_iteration",
            np.arange(1, len(raw_bootstrap_df) + 1),
        )

        raw_bootstrap_df.to_csv(
            outdir / f"{tag}_test_prediction_uncertainty_raw.csv",
            index=False,
        )

        # ==============================================================
        # 4. Molecule-level bootstrap uncertainty summary
        # ==============================================================
        required_bootstrap_summary_keys = [
            "observed_target",
            "final_model_prediction",
            "final_model_probability",
            "lower_95",
            "median",
            "upper_95",
            "std_dev",
        ]

        missing_bootstrap_keys = [
            key for key in required_bootstrap_summary_keys
            if key not in boot
        ]

        if missing_bootstrap_keys:
            raise KeyError(
                "Bootstrap output is missing required summary fields: "
                f"{missing_bootstrap_keys}"
            )

        bootstrap_summary_df = pd.DataFrame({
            "test_index": test_indices,
            "observed_class": np.asarray(boot["observed_target"]),
            "final_model_prediction": np.asarray(
                boot["final_model_prediction"]
            ),
            "final_model_probability": np.asarray(
                boot["final_model_probability"],
                dtype=float,
            ),
            "bootstrap_probability_lower_95": np.asarray(
                boot["lower_95"],
                dtype=float,
            ),
            "bootstrap_probability_median": np.asarray(
                boot["median"],
                dtype=float,
            ),
            "bootstrap_probability_upper_95": np.asarray(
                boot["upper_95"],
                dtype=float,
            ),
            "bootstrap_probability_std_dev": np.asarray(
                boot["std_dev"],
                dtype=float,
            ),
        })

        bootstrap_summary_df.to_csv(
            outdir / f"{tag}_test_prediction_uncertainty_summary.csv",
            index=False,
        )

        print(
            "[Reviewer validation] Saved CV fold metrics, target-shuffling "
            f"outputs, and bootstrap probability outputs for {tag}."
        )
    # Review round 1


    def step5_train_models(self, algorithms=['RandomForest', 'XGBoost', 'GradientBoosting',
                'LogisticRegression', 'DecisionTree', 'KNN', 'SVC']):
        """Step 5: Train optimized models + comprehensive validation.
    
            For each model:
            1. Train with best hyperparameters
            2. 5-fold CV (Accuracy±σ, LogLoss±σ)
            3. Bootstrap CIs (100x resampling, positive class probability)
            4. AOA analysis (Delaunay triangulation PCA-space)
            5. t-test vs null hypothesis (p-value, significance)
        
        Saves models: ../Results/TrainedModels/RS{rs}_{algo}.pkl
        
        Returns:
            list: Model results dictionary
        """
        print("\n" + "█"*75)
        print("█" + " STEP 5: TRAINING ML CLASSIFIERS ".center(73) + "█")
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
            tag = f"RS{self.random_state}_{self.current_method}_n{self.current_nfeatures}_{self.current_mode}_{self.backfill}_{algo}_fine-tuned"
            #Review round 1
            self.export_reviewer_validation_outputs(result, tag)
            #Review round 1
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

            print(f" Train Accuracy: {result['train_acc']:.3f}")
            print(f" Train Balanced Acc:  {result['train_bal_acc']:.3f}")
            print(f" Train F1-score:   {result['train_f1']:.3f}")
            
            print("-" * 30)

            print(f" Test Accuracy: {result['test_acc']:.3f}")
            print(f" Test Balanced Acc:   {result['test_bal_acc']:.3f}")
            print(f" Test F1-score:  {result['test_f1']:.3f}")

            print(f" CV Accuracy:    {result['cv_acc_mean']:.3f} ± {result['cv_acc_std']:.3f}")
            print(f" CV Balanced Acc:     {result['cv_bal_acc_mean']:.3f} ± {result['cv_bal_acc_std']:.3f}")

            if ts_stats:
                null_mean = ts_stats.get('null_acc_mean', 0.0)
                null_std = ts_stats.get('null_acc_std', 0.0)
                p_val = ts_stats.get('p_value', 1.0)
                status_msg = ts_stats.get('status_msg', "Unknown") # statistical_test
                threshold = ts_stats.get('threshold_good', 0.0)    # 95. percentile
                null_min = ts_stats.get('null_acc_min', 100)
                null_max = ts_stats.get('null_acc_max', 1000)
                z_score = ts_stats.get('z_score', 0.0)
                
                print(f"   Null Balanced Accuracy:  {null_mean:.2f} ± {null_std:.2f} (Target Shuffling Baseline)")
                
                # (95th percentile)
                print(f"   Threshold (95th %tile):  > {threshold:.2f} (To beat 95% of random models)")
                
                # status_msg, more descriptive
                print(f"   Status:      {status_msg} (p={p_val:.2e})")

                print(f"  Target shuffling minimal Balanced accuracy: {null_min:.2f}; maximal balanced accuracy: {null_max:.2f}")
                print(f"  Z-score (How many sigmas is our model away from noise mean?): {z_score:.2f} {'(PASS: >2σ)' if z_score > 2 else '(FAIL)'}")
                
                
            else:
                print("   Status:      Target Shuffling failed or skipped.")
        
            model_results.append(result)
        
        self.model_results = model_results
        return model_results
    
    def step6_compile_results(self):
        """Step 6: Comprehensive results table compilation.
    
        Creates Excel sheet with:
        | Algorithm | TrainAcc | TestAcc | TrainF1 | TestF1 | CVAcc±σ | LogLoss±σ | p-value | Significant |
            
        Ranks models by TestAcc, highlights top-3, exports to Excel.
            
        Returns:
            pd.DataFrame: Complete results table
        """
        print("\n" + "█"*75)
        print("█" + " STEP 6: RESULTS COMPILATION & ANALYSIS (CLASSIFICATION) ".center(73) + "█")
        print("█"*75)
        
        results_data = []
        for result in self.model_results:
            results_data.append({
                'Algorithm': result['algorithm'],
                'Train_Acc': result['train_acc'],
                'Test_Acc': result['test_acc'],
                'Train_F1': result['train_f1'],
                'Test_F1': result['test_f1'],
                'Train_Bal_Acc': result['train_bal_acc'],
                'Test_Bal_Acc': result['test_bal_acc'],
                'CV_Acc_Mean': result['cv_acc_mean'],
                'CV_Acc_Std': result['cv_acc_std'],
                'CV_Bal_Acc': result['cv_bal_acc_mean'],
                'CV_Bal_Std': result['cv_bal_acc_std'],
                'CV_LogLoss_Mean': result['cv_logloss_mean'],
                'CV_LogLoss_Std': result['cv_logloss_std'],
                'P_Value': result['p_value'],
                'Significant': result['significant'],
                'Truly_Significant': result.get('truly_significant', False),
            })
        
        self.results_df = pd.DataFrame(results_data)
        
        # Print summary table
        print("\nTOP 3 MODELS (Ranked by Test Balanced Accuracy):")
        print("-" * 95)
        print(f"{'Algorithm':<25} | {'Balanced Accuracy':<8} | {'Accuracy':<6} | {'p-value':<10} | {'Status'}")
        print("-" * 95)
        
        for idx, row in self.results_df.nlargest(3, 'Test_Bal_Acc').iterrows():
            if row['Truly_Significant']:
                status = "High Significance"
            elif row['Significant']:
                status = "Significant"
            else:
                status = "Not Significant"
            
            print(f"{row['Algorithm']:<25} | {row['Test_Bal_Acc']:.3f}    | {row['Test_Acc']:.2f}   | {row['P_Value']:.1e}  | {status}")
        
        # Baseline comparison
        print("\nBASELINE COMPARISON:")
        print("-" * 70)
        
        if 'dummy' in self.baselines:
            dummy_acc = self.baselines['dummy']['Accuracy']
            best_model_row = self.results_df.loc[self.results_df['Test_Bal_Acc'].idxmax()]
            best_acc = best_model_row['Test_Acc'] 
            improvement = 100 * (best_acc - dummy_acc) / dummy_acc

            print(f"Dummy Majority Acc: {dummy_acc:.2f}")
            print(f"Best Model Acc:     {best_acc:.2f}")
            print(f"Improvement:        {improvement:.1f}%")
        else:
            print("Dummy Majority Acc not available.")
        
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

        # 1. Identify Best Model (by Test balanced accuracy)
        best_idx = self.results_df['Test_Bal_Acc'].idxmax() 
        
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
            print("\nBootstrap Uncertainty (95% CI on predicted probability of positive class):")
            print("Sample probabilities for first 3 test samples:")

            # Safe loop limit (prefer length of bootstrap arrays)
            n_samples = min(3, len(boot.get('median', [])))

            for i in range(n_samples):
                # True label for comparison (if available)
                true_val = self.y_test.iloc[i] if hasattr(self, "y_test") else None

                pred_median = boot['median'][i]
                lower = boot['lower_95'][i]
                upper = boot['upper_95'][i]
                width = upper - lower

                if true_val is not None:
                    print(f"  Sample {i+1}: True={int(true_val)} | "
                        f"PredProb={pred_median:.3f} (95% CI: {lower:.3f}-{upper:.3f}, Width: {width:.3f})")
                else:
                    print(f"  Sample {i+1}: PredProb={pred_median:.3f} "
                        f"(95% CI: {lower:.3f}-{upper:.3f}, Width: {width:.3f})")
        else:
            print("\nBootstrap results not available.")

        # 3. AOA Reporting
        aoa = best_result.get('aoa')
        if aoa:
            print("\nArea of Applicability (AOA) Status:")
            print(f"  Interpolative (Reliable):  {aoa['pct_interpolative']:.1f}% of test set")
            print(f"  Extrapolative (Uncertain): {aoa['pct_extrapolative']:.1f}% of test set")
            print("  (See generated PDF file for visual plot)")
        else:
            print("\nAOA results not available.")

        return best_result
    
    def run_complete_analysis(self, 
                                method='consensus', 
                                n_features=20, 
                                mode='fast_aggr', 
                                backfill=False, 
                                preselected_features: Optional[List[str]] = None,   # ← NEW
                                skip_hyperparameter_optimization=False, # Review 1
                                **kwargs):
            """Execute full 7-step QSPR pipeline.
            
            NEW: If preselected_features is supplied, Step 2 is completely skipped.
            """
            
            print("\n" + "═" + "═"*73 + "═")
            print(" ADVANCED QSPR ANALYSIS - COMPLETE PIPELINE ".center(73))
            print("═" + "═"*73 + "═")
            
            self.step1_load_and_prepare_data()
            
            # ====================== PRESELECTED FEATURES MODE ======================
            if preselected_features is not None and len(preselected_features) > 0:
                print("\n" + "█"*75)
                print(" PRESELECTED FEATURES MODE - SKIPPING STEP 2 ".center(73))
                print("█"*75)
                
                # Verify features exist in the loaded data
                available = set(self.data.columns) - {self.target_col}
                valid = [f for f in preselected_features if f in available]
                
                if len(valid) < len(preselected_features):
                    print(f"WARNING: {len(preselected_features)-len(valid)} feature(s) not found in dataset "
                        f"and were dropped.")
                
                self.selected_features = pd.DataFrame({'feature': valid})
                self.method_name = 'preselected'
                self.current_mode = 'preselected'
                self.backfill = False
                
                print(f"→ Using {len(valid)} pre-selected features directly (bypassing consensus).")
                
                # Still show nice correlation table (exactly like normal flow)
                print("\n" + "="*75)
                self.correlation_df = print_feature_correlations(
                    self.train_data, self.selected_features, self.target_col
                )
                
            else:
                # Normal flow (what you had before)
                self.step2_feature_selection(method=method, n_features=n_features, 
                                            mode=mode, backfill=backfill, **kwargs)
            
            # Rest of the method stays 100% unchanged
            actual_features = len(self.selected_features)
            if actual_features < 2:
                ...
                return self 
            
            self.step3_baseline_comparison()
            #Review 1 #self.step4_hyperparameter_optimization()
            if not skip_hyperparameter_optimization:
                self.step4_hyperparameter_optimization()
            elif not getattr(self, "hyperopt_results", None):
                raise ValueError(
                    "skip_hyperparameter_optimization=True requires "
                    "self.hyperopt_results to be populated first."
                )
            # Review 1
            
            self.step5_train_models()
            self.step6_compile_results()
            self.step7_uncertainty_and_aoa()
            
            return self
    
    def export_results(self, output_file='QSPR_Classification_Results.xlsx'):
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
                        'CV_Bal_Acc_Score': result['best_score'],
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

    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)

    sys.stdout = DualWriter(log_filename)
    sys.stderr = sys.stdout

    print(f"Console output saved to: {log_filename}\n")

    # ─── Pre-selected features ───────────────────────────────────────────────
    PRESELECTED_FEATURES = [
        'Sterimol_Beta_R2_B5',
        'Sterimol_Alpha_R1_L',
        'CircuS_r1-3_O1[C@@H](C)C1',
        'CircuS_r1-3_ccc',
        'CircuS_r1-3_C[C@@H]1O[C@H]1C(=O)C',
        'CircuS_r1-3_[C@@H]1([C@H](C)O1)C'
    ]

    MODEL_PATHS = {
    "RandomForest": (
        "../Results/Trained_Models/"
        "RS42_consensus_n6_fast_aggr_False_RandomForest_model.pkl"
    ),
    "XGBoost": (
        "../Results/Trained_Models/"
        "RS42_consensus_n6_fast_aggr_False_XGBoost_model.pkl"
    ),
    "GradientBoosting": (
        "../Results/Trained_Models/"
        "RS42_consensus_n6_fast_aggr_False_GradientBoosting_model.pkl"
    ),
    "LogisticRegression": (
        "../Results/Trained_Models/"
        "RS42_consensus_n6_fast_aggr_False_LogisticRegression_model.pkl"
    ),
    "DecisionTree": (
        "../Results/Trained_Models/"
        "RS42_consensus_n6_fast_aggr_False_DecisionTree_model.pkl"
    ),
    "KNN": (
        "../Results/Trained_Models/"
        "RS42_consensus_n6_fast_aggr_False_KNN_model.pkl"
    ),
    "SVC": (
        "../Results/Trained_Models/"
        "RS42_consensus_n6_fast_aggr_False_SVC_model.pkl"
    ),
    }

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
        # Review 1
        analysis.load_fixed_models(MODEL_PATHS)


        analysis.run_complete_analysis(preselected_features=PRESELECTED_FEATURES, skip_hyperparameter_optimization=True)

        if hasattr(analysis, 'results_df') and not analysis.results_df.empty:
            output_file = f"../Results/QSPR_Classification_Preselected_{timestamp}.xlsx"
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