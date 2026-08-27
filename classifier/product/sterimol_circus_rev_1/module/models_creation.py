"""
Module for QSPR analysis with:
>> CircuS/ChyLine descriptors (DOPtools integration)
>> Advanced descriptors (Sterimol, Electronic, ChemBERTa)
>> Multiple feature selection methods with consensus voting
>> Hyperparameter optimization (GridSearchCV)
>> Rigorous validation (CV, bootstrap, AOA, statistics)
>> Uncertainty quantification and baseline models

Import this module in QSPR analysis scripts.
Location: module/models_creation.py
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import os
os.environ["PYTHONWARNINGS"] = "ignore"

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

import pandas as pd
import numpy as np

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score, KFold, GridSearchCV, train_test_split, StratifiedKFold
)
from sklearn.impute import SimpleImputer

from sklearn.base import clone

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE, RFECV, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import xgboost as xgb
from scipy import stats
from scipy.spatial import ConvexHull
import math

from collections import Counter
from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt

from hyperparameters import HyperparametersConfig

import sys
_banner_key = 'MODELS_CREATION_BANNER_'
if os.environ.get(_banner_key) != 'TRUE':
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                models_creation.py MODULE READY                             ║
║                                                                            ║
║         Import this module in your QSPR analysis scripts                   ║
║         Location: module/models_creation.py                                ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    os.environ[_banner_key] = 'TRUE'
    sys.modules['models_creation_banner'] = type('Banner', (), {'printed': True})()


# =============================================================================
# MACHINE LEARNING
# =============================================================================
TARGET = 'epox_cla'
# =============================================================================
# ADVANCED FEATURE SELECTION
# =============================================================================

class AdvancedFeatureSelector:
    """
    4-method consensus feature selection (TRAINING DATA ONLY).
    
    Two modes:
    - "fast_aggr": Aggressive single-split approach
      * StandardScaler
      * Permutation on single train/test split
      * Classic RFE (no CV)
      * Mutual Information
      * Strict consensus (no backfill)
    
    - "cv_safe": Conservative CV-based approach (more robust for generalization)
      * StandardScaler
      * Permutation with 5-Fold CV
      * RFECV (Recursive Feature Elimination with CV)
      * Mutual Information
      * Consensus with optional backfill
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Training target
        mode (str): "fast_aggr" or "cv_safe" (default: "fast_aggr")
    """
    
    def __init__(self, X, y, mode="fast_aggr", backfill=False):
        self.X = X
        self.y = y
        self.mode = mode
        self.backfill = backfill
        
        if mode not in ["fast_aggr", "cv_safe"]:
            raise ValueError(f"Mode must be 'fast_aggr' or 'cv_safe', got '{mode}'")
        
        self.scaler = StandardScaler()
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=X.columns)
        
        print(f"  AdvancedFeatureSelector initialized in MODE: {mode}")

    def _get_stable_rf(self):
        """Stable RF model for feature selection (RFE / RFECV)."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=2,
            max_features='sqrt',
            min_samples_leaf=4,
            bootstrap=True,
            random_state=42,
            class_weight='balanced',
            n_jobs=1
        )
    def _get_stable_gbr(self):
        """GBR configured to avoid overfitting on small N."""
        return GradientBoostingClassifier(
            n_estimators=30,
            learning_rate=0.05,
            max_depth=1,
            max_features='sqrt',
            min_samples_leaf=4, 
            subsample=0.8,
            random_state=42
        )
    
    # ========================================================================
    # MODE 1: FAST_AGGR (Single split)
    # ========================================================================

    def _permutation_importance_fast_aggr(self, n_features=15, model=None):
        """
        Method 1 (fast_aggr): Permutation importance - single random split.

        Uses a single 80/20 train/test split to approximate out-of-sample 
        importance. High n_repeats helps stabilize results given the small 
        sample size and high feature count.
        """
        if model is None:
            model = self._get_stable_gbr()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, 
            self.y, 
            stratify=self.y,
            test_size=0.20, 
            random_state=42
        )
        print("--- Test Set Preview ---")
        print(f"Number of samples in test set: {len(y_test)}")
        print("\nTarget values (y_test) in test set:")
        print(y_test)
        model_fit = clone(model)
        model_fit.fit(X_train, y_train)

        perm_imp = permutation_importance(
            model_fit, X_test, y_test,
            scoring='balanced_accuracy',
            n_repeats=30,
            random_state=42,
            n_jobs=-1
        )

        importances = pd.Series(perm_imp.importances_mean, index=self.X.columns)
        return importances.nlargest(n_features).index.tolist()

    def _rfe_selection_fast_aggr(self, n_features=15):
        """
        Method 2 (fast_aggr): Standard RFE (no CV), direct on all features.

        No initial RFE / pre-filter inside this method. External pre-filtering
        (NaN/variance/correlation) should be done on TRAIN ONLY before
        constructing this selector.
        """
        model = self._get_stable_rf()

        rfe = RFE(
            estimator=model,
            n_features_to_select=n_features,
            step=1,
        )
        rfe.fit(self.X_scaled, self.y)

        return list(self.X.columns[rfe.support_])

    def _mutual_information_fast_aggr(self, n_features=15):
        """Method 3 (fast_aggr): Mutual Information (non-linear, single pass)."""
        mi_scores = mutual_info_classif(
            self.X_scaled, self.y,
            n_neighbors=5,
            random_state=42
        )
        mi_series = pd.Series(mi_scores, index=self.X.columns)
        return mi_series.nlargest(n_features).index.tolist()

    # ========================================================================
    # MODE 2: CV_SAFE (5-Fold CV - Conservative)
    # ========================================================================

    def _permutation_importance_cv_safe(self, n_features=15, model=None):
        """
        Method 1 (cv_safe): Permutation importance with 5-Fold Stratified CV.

        Pure CV-based permutation: for each (optional: stratified) fold, fit on training
        part and compute permutation importance on validation fold, then
        average across folds. No internal model-based pre-filtering.
        """
        if model is None:
            model = self._get_stable_gbr()

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        total_importances = pd.Series(0.0, index=self.X_scaled.columns)
        counts = pd.Series(0, index=self.X_scaled.columns)

        print(f"\n=== Starting CV Permutation Importance (5 Folds) ===")

        for i, (train_idx, val_idx) in enumerate(kf.split(self.X_scaled, self.y), 1):
                X_train_fold = self.X_scaled.iloc[train_idx]
                y_train_fold = self.y.iloc[train_idx]
                X_val_fold = self.X_scaled.iloc[val_idx]
                y_val_fold = self.y.iloc[val_idx]

                print(f"\n--- Fold {i} Preview ---")
                print(f"Validation samples: {len(y_val_fold)}")
                print(f"Validation Target Mean: {y_val_fold.mean():.4f}")
                print(f"Target values in Fold {i}:")
                print(y_val_fold.values)
                # ---------------------

                model_fold = clone(model)
                model_fold.fit(X_train_fold, y_train_fold)

                perm_imp = permutation_importance(
                    model_fold, X_val_fold, y_val_fold,
                    scoring='balanced_accuracy',
                    n_repeats=20,
                    random_state=42,
                    n_jobs=-1
                )

                fold_series = pd.Series(
                    perm_imp.importances_mean,
                    index=self.X_scaled.columns
                )
                total_importances = total_importances.add(fold_series, fill_value=0.0)
                counts = counts.add(pd.Series(1, index=self.X_scaled.columns), fill_value=0)

        avg_importances = total_importances / counts.replace(0, 1)
        return avg_importances.nlargest(n_features).index.tolist()

    def _rfe_selection_cv_safe(self, n_features=15):
        """
        Method 2 (cv_safe): RFECV (Recursive Feature Elimination with CV).

        Uses a (optional: stratified) 5-fold CV for ranking features, without an internal
        initial RFE. External pre-filtering (NaN/variance/correlation) should
        be done before constructing this selector.
        """
        model = self._get_stable_rf()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(self.X_scaled, self.y))

        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=splits,
            scoring='balanced_accuracy',
            min_features_to_select=1,
            n_jobs=-1
        )
        rfecv.fit(self.X_scaled, self.y)

        feature_ranks = pd.DataFrame({
            'feature': self.X.columns,
            'rank': rfecv.ranking_
        }).sort_values('rank')

        return feature_ranks.head(n_features)['feature'].tolist()

    def _mutual_information_cv_safe(self, n_features=15):
        """
        Method 3 (cv_safe): Robust Mutual Information via 5-Fold Stratified CV.

        Calculates MI scores across 5 folds to ensure that the relationship
        between features and the target is consistent across different data
        subsets. Reduces the risk of selecting features correlated with noise.
        """
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_mi_scores = []

        for train_idx, _ in kf.split(self.X_scaled, self.y):
            mi = mutual_info_classif(
                self.X_scaled.iloc[train_idx],
                self.y.iloc[train_idx],
                n_neighbors=5,
                random_state=42
            )
            fold_mi_scores.append(mi)

        avg_mi = np.mean(fold_mi_scores, axis=0)
        mi_series = pd.Series(avg_mi, index=self.X.columns)

        return mi_series.nlargest(n_features).index.tolist()

    # ========================================================================
    # UNIFIED PUBLIC INTERFACE (Dispatch to Mode)
    # ========================================================================

    def permutation_importance_selection(self, n_features=15, model=None):
        """Permutation importance (dispatches to mode)."""
        if self.mode == "fast_aggr":
            return self._permutation_importance_fast_aggr(n_features, model)
        else:  # cv_safe
            return self._permutation_importance_cv_safe(n_features, model)

    def rfe_selection(self, n_features=15):
        """RFE / RFECV (dispatches to mode)."""
        if self.mode == "fast_aggr":
            return self._rfe_selection_fast_aggr(n_features)
        else:  # cv_safe
            return self._rfe_selection_cv_safe(n_features)

    def mutual_information_selection(self, n_features=15):
        """Mutual Information (dispatches to mode)."""
        if self.mode == "fast_aggr":
            return self._mutual_information_fast_aggr(n_features)
        else:  # cv_safe
            return self._mutual_information_cv_safe(n_features)
    
    def consensus_selection(self, n_features=15, min_methods=2, search_multiplier=1.5):
        """
        Method 4: Consensus voting - PRIORITY: voting=3 → >=2, smart ranking.
        
        Two-stage selection:
        1. Take ALL features with voting==3 (strict consensus)
        2. Fill remaining with top voting>=2, ranked by: votes > RFE_pos > perm_pos > MI_pos
        
        Prints:
        - List of voting=3 features
        - Total voting>=2 count  
        - Stage progress
        
        Args:
            n_features (int): Target number of features to return.
            min_methods (int): Minimum methods that must agree (votes).
            search_multiplier (float): Pool expansion factor for sub-methods.
            backfill (bool): Whether to fill missing features (if None, mode decides)
                - fast_aggr defaults to False (strict consensus)
                - cv_safe defaults to False (but can override to True)
        """
        backfill = self.backfill
        if backfill is None:
            backfill = (self.mode == "cv_safe")
        
        n_search = int(max(n_features * search_multiplier, n_features + 2))
        n_search = min(n_search, self.X.shape[1])
        
        print(f"  Running consensus selection ({self.mode} mode)...")
        print(f"  Target: {n_features} | Pool: {n_search} | backfill={backfill}")
        
        # Get expanded pools from sub-methods (TRAIN ONLY, direct selection)
        perm_feats = self.permutation_importance_selection(n_features=n_search)
        rfe_feats = self.rfe_selection(n_features=n_search)
        mi_feats = self.mutual_information_selection(n_features=n_search)
        
        all_feats = perm_feats + rfe_feats + mi_feats
        feat_counts = Counter(all_feats)
        
        # PRIORITY SELECTION + PRINTS
        voting3 = [f for f, c in feat_counts.items() if c == 3]
        voting_ge2 = [f for f, c in feat_counts.items() if c >= 2]
        
        print(f"  Voting = 3: {sorted(voting3)} ({len(voting3)})")
        print(f"  Voting >= 2: {len(voting_ge2)} total (skip count = 1)")

        # STAGE 1: ALL voting=3
        candidates = list(voting3)
        print(f"  Stage 1: +{len(voting3)} from voting = 3")
        
        # STAGE 2: Fill with top voting>=2 ("SMART" RANKING)
        remaining = n_features - len(candidates)
        if remaining > 0:
            extras = [f for f in voting_ge2 if f not in voting3]
            
            def rank_extra(feat):
                rfe_pos = rfe_feats.index(feat) if feat in rfe_feats else 999
                perm_pos = perm_feats.index(feat) if feat in perm_feats else 999
                mi_pos = mi_feats.index(feat) if feat in mi_feats else 999
                return (feat_counts[feat], -rfe_pos, -perm_pos, -mi_pos)  # RFE > perm > MI
            
            extras.sort(key=rank_extra, reverse=True)
            take = min(remaining, len(extras))
            candidates.extend(extras[:take])
            print(f"  Stage 2: +{take} from voting >= 2 (ranked RFE > perm > MI)")
        
        # BACKFILL (optional)
        if backfill and len(candidates) < n_features:
            missing = n_features - len(candidates)
            fillers = [f for f in rfe_feats if f not in candidates][:missing]
            fillers.extend([f for f in perm_feats if f not in candidates][:missing-len(fillers)])
            candidates.extend(fillers)
            print(f"  Backfill: +{len(fillers)}")
        
        # FINAL SORT (if overshot)
        if len(candidates) > n_features:
            def final_rank(feat):
                rfe_pos = rfe_feats.index(feat) if feat in rfe_feats else 999
                perm_pos = perm_feats.index(feat) if feat in perm_feats else 999
                mi_pos = mi_feats.index(feat) if feat in mi_feats else 999
                return (feat_counts[feat], -rfe_pos, -perm_pos, -mi_pos)  # RFE > perm > MI
            
            candidates.sort(key=final_rank, reverse=True)
            candidates = candidates[:n_features]
        
        print(f"  Final consensus: {len(candidates)} features")
        
        return candidates

    

def select_features_advanced(data, target_col, method='consensus', n_features=20, 
                            mode='fast_aggr', backfill=False, **kwargs):
    """
    Advanced feature selection wrapper.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset with features and target column
    target_col : str
        Target property column name
    method : str
        'consensus' (default), 'permutation', 'rfe', 'mutual_info'
    n_features : int
        Number of features to select
    mode : str
        'fast_aggr' (aggressive single-split) or 'cv_safe' (5-fold CV)
    backfill : bool
        Add non-consensus features if needed (default: False)
    **kwargs
        Passed to selection methods:
        - consensus: search_multiplier, min_methods
        - others: ignored
        
    Returns
    -------
    pd.DataFrame
        Selected features (column: 'feature')
    """
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    X = X.fillna(X.median())
    
    selector = AdvancedFeatureSelector(X, y, mode=mode, backfill=backfill)
    
    if method == 'consensus':
        selected = selector.consensus_selection(n_features=n_features, **kwargs)
    elif method == 'permutation':
        selected = selector.permutation_importance_selection(n_features=n_features)
    elif method == 'rfe':
        selected = selector.rfe_selection(n_features=n_features)
    elif method == 'mutual_info':
        selected = selector.mutual_information_selection(n_features=n_features)
    else:
        selected = selector.consensus_selection(n_features=n_features, **kwargs)
    
    return pd.DataFrame({'feature': selected})

    
class BaselineModels:
    """Statistical validation baselines.
    
    - baseline_dummy_majority(): Absolute performance floor
    - baseline_target_shuffling(): Null hypothesis (100 permutations) → strong randomization test
      using the provided optimized estimator structure.
    """
    
    @staticmethod
    def baseline_dummy_majority(y_train, y_test):
        from collections import Counter

        majority_class = Counter(y_train).most_common(1)[0][0]
        y_pred = np.full_like(y_test, majority_class)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        
        return acc, f1, bal_acc
    
    @staticmethod
    def baseline_target_shuffling(X_train, X_test, y_train, y_test, estimator=None, n_shuffles=100):
        """
        Target shuffling with OPTIMIZED estimator (Post-Hoc Validation).
        
        Tests whether the model structure (with optimized hyperparameters) 
        performs better than random chance (shuffled target).
        
        Args:
            X_train, X_test: Feature matrices
            y_train, y_test: Target vectors
            estimator: Fitted estimator (Pipeline or model) to clone. 
                       If None, defaults to RandomForestClassifier (fallback).
            n_shuffles: Number of permutations (default: 100)
        
        Returns:
            tuple: (mean_null_acc, std_null_acc, list_of_null_acc)
                   Metric used is balanced_accuracy_score.
        """
        null_acc_list = []
        # Suppress warnings during loop
        os.environ['PYTHONWARNINGS'] = 'ignore'

        # Fallback if no estimator provided (backward compatibility)
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
            print('No estimator was passed from prepare_model function, using default random forest classifier...')

        for shuffle_iter in range(n_shuffles):
            # Shuffle target (destroy true relationship)
            y_shuffled = np.random.permutation(y_train)
            
            # Clone the estimator (preserves hyperparameters!)
            # This ensures we test the MODEL STRUCTURE, not just default params.
            model_shuffled = clone(estimator)
            
            # Set random_state for reproducibility (if supported)
            # Handle both Pipeline and regular estimators
            if hasattr(model_shuffled, 'random_state'):
                model_shuffled.random_state = 42 + shuffle_iter
            elif hasattr(model_shuffled, 'steps'):  # Pipeline
                step_name = model_shuffled.steps[-1][0]
                if hasattr(model_shuffled.named_steps[step_name], 'random_state'):
                    model_shuffled.named_steps[step_name].random_state = 42 + shuffle_iter

            # Fit & Predict on noise
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model_shuffled.fit(X_train, y_shuffled)
                except Exception:
                    # Fallback for edge cases (e.g. single class in shuffle)
                    continue
            
            y_pred = model_shuffled.predict(X_test)
            
            # Metric: Balanced Accuracy (robust to imbalance)
            acc = balanced_accuracy_score(y_test, y_pred)
            null_acc_list.append(acc)
        
        # Safety check
        if not null_acc_list:
            return 0.0, 0.0, []
        
        return np.mean(null_acc_list), np.std(null_acc_list), null_acc_list

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

class HyperparameterOptimizer:
    """GridSearchCV optimization for 7 ML algorithms using PIPELINES.
    
    Comprehensive parameter grids with nested 5-fold CV (scoring='accuracy' or 'balanced_accuracy).
    Returns best_params, best_score, and total combinations tested.
    
    Static methods for each algorithm with wide grids.

    Ensures scaling happens inside CV folds (preventing data leakage).
    """
    @staticmethod
    def _optimize_pipeline(X_train, y_train, estimator, param_grid_raw, cv=5):
        """Helper to run GridSearchCV on a Pipeline(Imputer->Scaler->Model)"""
        
        # Build Pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', estimator)
        ])
        
        # Rename params to model__param
        param_grid = {f"model__{k}": v for k, v in param_grid_raw.items()}
        
        # GridSearch
        gs = GridSearchCV(
            pipeline, 
            param_grid, 
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            n_jobs=-1, 
            verbose=0,
            refit=True
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_train, y_train)
            
        n_combinations = len(gs.cv_results_['params'])

        return {
            'best_params': gs.best_params_, 
            'best_score': gs.best_score_,
            'n_combinations': n_combinations,
            'model': gs.best_estimator_
        }
    
    @staticmethod
    def optimize_random_forest(X_train, y_train, cv=5):
        """GridSearchCV optimization for Random Forest"""
        os.environ['PYTHONWARNINGS'] = 'ignore'
        return HyperparameterOptimizer._optimize_pipeline(
            X_train, y_train, 
            RandomForestClassifier(random_state=42),
            HyperparametersConfig.RANDOM_FOREST, cv=cv
        )

    @staticmethod
    def optimize_xgboost(X_train, y_train, cv=5):
        """GridSearchCV optimization for XGBoost"""

        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

        return HyperparameterOptimizer._optimize_pipeline(
            X_train, y_train, 
            xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_weight),
            HyperparametersConfig.XGBOOST, cv=cv
        )
        

    @staticmethod
    def optimize_gradient_boosting(X_train, y_train, cv=5):
        """GridSearchCV optimization for Gradient Boosting"""
        
        return HyperparameterOptimizer._optimize_pipeline(
            X_train, y_train, 
            GradientBoostingClassifier(random_state=42),
            HyperparametersConfig.GRADIENT_BOOSTING, cv=cv
        )

    
    @staticmethod
    def optimize_logistic_regression(X_train, y_train, cv=5):
        """GridSearchCV optimization for LogisticRegression"""
        
        return HyperparameterOptimizer._optimize_pipeline(
            X_train, y_train, 
            LogisticRegression(max_iter=1000),
            HyperparametersConfig.LOGISTIC_REGRESSION, cv=cv
        )

    @staticmethod
    def optimize_decision_tree(X_train, y_train, cv=5):
        """GridSearchCV optimization for Decision Tree"""
        
        return HyperparameterOptimizer._optimize_pipeline(
            X_train, y_train, 
            DecisionTreeClassifier(random_state=42),
            HyperparametersConfig.DECISION_TREE, cv=cv
        )

    @staticmethod
    def optimize_knn(X_train, y_train, cv=5):
        """GridSearchCV optimization for K-Nearest Neighbors"""
        
        return HyperparameterOptimizer._optimize_pipeline(
            X_train, y_train, 
            KNeighborsClassifier(),
            HyperparametersConfig.KNN, cv=cv
        )

    @staticmethod
    def optimize_svc(X_train, y_train, cv=5):
        """GridSearchCV optimization for Support Vector Classifier (SVC)"""
        
        return HyperparameterOptimizer._optimize_pipeline(
            X_train, y_train, 
            SVC(max_iter=5000, probability=True, cache_size=2000),
            HyperparametersConfig.SVM, cv=cv
        )
    
# =============================================================================
# PREPARE FINAL MODEL
# =============================================================================

def prepare_model(X_train, X_test, y_train, y_test,
                 cv_folds=5, prefit_model=None, method_name='unknown', 
                 mode='fast_agg', backfill=False,
                 precomputed_aoa=None):
    """ 
    Train model with hyperparameter optimization and rigorous validation.
    
    Args:
        prefit_model: Optimized Pipeline from GridSearchCV
        precomputed_aoa: (Optional) Result of AOA to avoid recalculating it
    """

    if prefit_model is not None:
        best_model = prefit_model
    else:
        raise ValueError("prefit_model cannot be None in this workflow.")

    # Predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train, average='macro')
    test_f1 = f1_score(y_test, y_pred_test, average='macro')

    # Validation Framework
    validator = RigorousValidator(X_train, X_test, y_train, y_test, best_model, method_name, mode, backfill)
    
    cv_results = validator.cross_validation_analysis(n_folds=cv_folds)
    bootstrap_results = validator.bootstrap_confidence_intervals(n_bootstrap=100)
    
    # AOA Handling
    if precomputed_aoa is not None:
        aoa_results = precomputed_aoa
    else:
        # Calculate only if not provided
        aoa_results = validator.area_of_applicability()

    # Target Shuffling (Validation of Model Structure)
    print(f"   Running Target Shuffling (Validation) for {method_name}...")
    
    null_mean, null_std, null_dist = BaselineModels.baseline_target_shuffling(
        X_train, X_test, y_train, y_test, 
        estimator=best_model, 
        n_shuffles=100
    )
    #Review 1
    null_dist = np.asarray(null_dist, dtype=float)
    #Review 1

    # Statistical test
    ts_stats = RigorousValidator.statistical_test(test_bal_acc, null_dist)

    if ts_stats['significantly_better']:
        print("      PASS: Model significantly better than noise.")
    else:
        print("      WARNING: Model indistinguishable from noise.")

    return {
        'model': best_model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_bal_acc': train_bal_acc,
        'test_bal_acc': test_bal_acc,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'cv_acc_mean': cv_results['cv_acc_mean'],
        'cv_acc_std': cv_results['cv_acc_std'],
        'cv_bal_acc_mean': cv_results['cv_bal_acc_mean'],
        'cv_bal_acc_std': cv_results['cv_bal_acc_std'],
        'cv_logloss_mean': cv_results['cv_logloss_mean'],
        'cv_logloss_std': cv_results['cv_logloss_std'],
        # Review 1: all five fold-wise CV values
        'cv_fold_metrics': cv_results,
        # Review 1: target shuffling
        'target_shuffling_bal_acc': null_dist,

        'bootstrap': bootstrap_results,
        'prediction_uncertainty_bootstrap': bootstrap_results,
        'aoa': aoa_results,
        'target_shuffling': ts_stats,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }

# =============================================================================
# RIGOROUS VALIDATION
# =============================================================================

class RigorousValidator:
    """7-layer model validation framework.
    
    1. Cross-validation (5-fold or LOO for small data)
    2. Bootstrap CIs (100x resampling, 95% intervals)
    3. AOA analysis (Delaunay triangulation in PCA-2D)
    4. Statistical tests (t-test vs null hypothesis)
    5. Baseline comparisons (dummy + target shuffling)
    
    Generates AOA plots (PDF, 300 DPI).
    """
    
    
    def __init__(self, X_train, X_test, y_train, y_test, model, method_name='unknown', mode='fast_agg', backfill=False):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.method_name = method_name
        self.mode = mode
        self.backfill = backfill
    
    def cross_validation_analysis(self, n_folds=5):
        """Stratified k-fold CV with fold-wise accuracy, balanced accuracy and log-loss."""
        kf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42,
        )

        cv_acc = []
        cv_bal_acc = []
        cv_logloss = []

        for fold_number, (train_idx, val_idx) in enumerate(
            kf.split(self.X_train, self.y_train),
            start=1,
        ):
            X_fold_train = self.X_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            fold_model = clone(self.model)
            fold_model.fit(X_fold_train, y_fold_train)

            y_pred = fold_model.predict(X_fold_val)

            cv_acc.append(
                accuracy_score(y_fold_val, y_pred)
            )

            cv_bal_acc.append(
                balanced_accuracy_score(y_fold_val, y_pred)
            )

            if not hasattr(fold_model, "predict_proba"):
                raise TypeError(
                    f"{type(fold_model).__name__} does not implement "
                    "predict_proba(). Use probability-enabled classifiers, "
                    "e.g. SVC(probability=True)."
                )

            y_proba = fold_model.predict_proba(X_fold_val)

            # Explicit labels make log-loss robust if a validation fold
            # happens to contain only one represented class.
            cv_logloss.append(
                log_loss(
                    y_fold_val,
                    y_proba,
                    labels=fold_model.classes_,
                )
            )

        cv_acc = np.asarray(cv_acc, dtype=float)
        cv_bal_acc = np.asarray(cv_bal_acc, dtype=float)
        cv_logloss = np.asarray(cv_logloss, dtype=float)

        return {
            # Fold-wise values required by export_reviewer_validation_outputs()
            "fold": np.arange(1, n_folds + 1),
            "cv_acc": cv_acc,
            "cv_bal_acc": cv_bal_acc,
            "cv_logloss": cv_logloss,

            # Existing aggregate values retained for prepare_model() and reports
            "cv_acc_mean": np.mean(cv_acc),
            "cv_acc_std": np.std(cv_acc),
            "cv_bal_acc_mean": np.mean(cv_bal_acc),
            "cv_bal_acc_std": np.std(cv_bal_acc),
            "cv_logloss_mean": np.mean(cv_logloss),
            "cv_logloss_std": np.std(cv_logloss),
        }
    
    #Review 1
    def bootstrap_confidence_intervals(self, n_bootstrap=100, positive_class=1):
        """
        Bootstrap uncertainty quantification for binary classification.

        Bootstrap samples are drawn from the training set. Each cloned model is
        fitted on one bootstrap sample and predicts the probability of
        `positive_class` for every observation in the external test set.

        Parameters
        ----------
        n_bootstrap : int, default=100
            Number of bootstrap resamples.
        positive_class : int or str, default=1
            Label treated as the positive class.

        Returns
        -------
        dict
            Bootstrap probability intervals and raw per-test-observation
            probabilities, with structure analogous to the regression version.
        """
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1.")

        if positive_class not in np.unique(self.y_train):
            raise ValueError(
                f"positive_class={positive_class!r} is not present in y_train. "
                f"Available training classes: {np.unique(self.y_train).tolist()}"
            )

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "The fitted classifier does not implement predict_proba(). "
                "Use a probabilistic classifier or enable probability estimates "
                "(e.g. SVC(probability=True))."
            )

        bootstrap_probs = []

        for boot in range(n_bootstrap):
            indices = np.random.choice(
                len(self.X_train),
                size=len(self.X_train),
                replace=True,
            )

            X_boot = self.X_train.iloc[indices]
            y_boot = self.y_train.iloc[indices]

            # A bootstrap sample can exceptionally contain only one class,
            # particularly for small or highly imbalanced data.
            if y_boot.nunique() < 2:
                continue

            boot_model = clone(self.model)
            boot_model.fit(X_boot, y_boot)

            if not hasattr(boot_model, "predict_proba"):
                raise TypeError(
                    "A bootstrap-fitted classifier does not implement "
                    "predict_proba()."
                )

            boot_classes = boot_model.classes_

            if positive_class not in boot_classes:
                continue

            positive_class_index = np.where(
                boot_classes == positive_class
            )[0][0]

            y_prob_boot = boot_model.predict_proba(
                self.X_test
            )[:, positive_class_index]

            bootstrap_probs.append(y_prob_boot)

        if len(bootstrap_probs) == 0:
            raise RuntimeError(
                "No valid bootstrap models were fitted. "
                "Check class imbalance, training-set size, and positive_class."
            )

        bootstrap_probs = np.asarray(bootstrap_probs, dtype=float)

        final_classes = self.model.classes_

        if positive_class not in final_classes:
            raise ValueError(
                f"positive_class={positive_class!r} is absent from the final "
                f"model classes: {final_classes.tolist()}"
            )

        final_positive_class_index = np.where(
            final_classes == positive_class
        )[0][0]

        final_model_probabilities = self.model.predict_proba(
            self.X_test
        )[:, final_positive_class_index]

        final_model_prediction = self.model.predict(self.X_test)

        return {
            # Interval summary — probability of the positive class
            "lower_95": np.percentile(bootstrap_probs, 2.5, axis=0),
            "median": np.percentile(bootstrap_probs, 50.0, axis=0),
            "upper_95": np.percentile(bootstrap_probs, 97.5, axis=0),
            "std_dev": np.std(bootstrap_probs, axis=0),

            # Full per-test-observation metadata
            "test_index": self.X_test.index.to_numpy(),
            "observed_target": np.asarray(self.y_test),
            "positive_class": positive_class,
            "final_model_prediction": np.asarray(final_model_prediction),
            "final_model_probability": np.asarray(
                final_model_probabilities,
                dtype=float,
            ),
            "n_bootstrap_requested": n_bootstrap,
            "n_bootstrap_valid": len(bootstrap_probs),

            # Shape: (n_bootstrap_valid, n_test_observations)
            "bootstrap_probabilities": bootstrap_probs,
        } #Review 1
    
    def area_of_applicability(self):
        """Convex hull analysis for applicability domain"""
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test) 

        number_of_features = self.X_train.shape[1]

        method = self.method_name

        mode = self.mode

        backfill = self.backfill
        
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        from scipy.spatial import Delaunay
        tri = Delaunay(X_train_pca)
        inside_mask = tri.find_simplex(X_test_pca) >= 0


        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Plot training points (Delaunay triangulation)
        plt.triplot(X_train_pca[:, 0], X_train_pca[:, 1], tri.simplices, 
                    'b-', alpha=0.5, linewidth=1, label='Training Convex Hull')


        # Plot test points: inside (interpolative) in green, outside (extrapolative) in red
        plt.scatter(X_test_pca[inside_mask, 0], X_test_pca[inside_mask, 1], 
                    c='green', s=50, alpha=0.7, label='Interpolative (Inside)', edgecolors='black')
        plt.scatter(X_test_pca[~inside_mask, 0], X_test_pca[~inside_mask, 1], 
                    c='red', s=50, alpha=0.7, label='Extrapolative (Outside)', edgecolors='black')
        
        # Plot test points: inside (interpolative) in green WITH NUMBERS
        inside_idx = np.where(inside_mask)[0]
        for i, idx in enumerate(inside_idx):
            plt.scatter(X_test_pca[idx, 0], X_test_pca[idx, 1], 
                        c='green', s=50, alpha=0.7, edgecolors='black')
            plt.annotate(f'T{idx}', (X_test_pca[idx, 0], X_test_pca[idx, 1]), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8,
                        color='darkgreen', fontweight='bold')
        
        # Plot test points: outside (extrapolative) in red WITH NUMBERS
        outside_idx = np.where(~inside_mask)[0]
        for i, idx in enumerate(outside_idx):
            plt.scatter(X_test_pca[idx, 0], X_test_pca[idx, 1], 
                        c='red', s=50, alpha=0.7, edgecolors='black')
            plt.annotate(f'T{idx}', (X_test_pca[idx, 0], X_test_pca[idx, 1]), 
                        xytext=(3, 3), textcoords='offset points', fontsize=8,
                        color='darkred', fontweight='bold')
        
        # Plot training points WITH NUMBERS
        for i in range(len(X_train_pca)):
            plt.scatter(X_train_pca[i, 0], X_train_pca[i, 1], 
                        c='blue', s=30, alpha=0.8, marker='^')
            plt.annotate(f'{i}', (X_train_pca[i, 0], X_train_pca[i, 1]), 
                        xytext=(3, 3), textcoords='offset points', fontsize=7,
                        color='darkblue', fontweight='bold')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Applicability Domain: Delaunay Triangulation in PCA Space\n(Train #, Test T#)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('applicability_domain_highres_'+str(method)+'_'+str(number_of_features)+'_'+str(mode)+'_'+str(backfill)+'_.pdf', dpi=300, bbox_inches='tight',
            format='pdf', facecolor='white')
        print(" High-res PDF saved as 'applicability_domain_highres_'"+str(method)+'_'+str(number_of_features)+'_'+str(mode)+'_'+str(backfill)+"'_.pdf'")
        
        return {
            'inside_mask': inside_mask,
            'n_interpolative': np.sum(inside_mask),
            'n_extrapolative': np.sum(~inside_mask),
            'pct_interpolative': 100 * np.sum(inside_mask) / len(inside_mask),
            'pct_extrapolative': 100 * np.sum(~inside_mask) / len(inside_mask),
        }
    
    
    @staticmethod
    def statistical_test(our_acc, null_acc_dist):
        """ t-test: our model vs null hypothesis
            Statistical significance test using empirical p-value and Z-score.
            Standard for Permutation Tests / Target Shuffling.
            
            Args:
                our_acc: Accuracy (or Balanced Acc) of our model (Higher is better)
                null_acc_dist: List/Array of null accuracies from shuffling
        """
        null_mean = np.mean(null_acc_dist)
        null_std = np.std(null_acc_dist)
        n_null = len(null_acc_dist)
        
        # 1. Z-score (Distance in standard deviations)
        # How many sigmas is our model BETTER than noise mean?
        z_score = (our_acc - null_mean) / null_std if null_std > 0 else 0
        
        # 2. Empirical p-value (Definition of Permutation Test)
        # Proportion of random models that were BETTER (higher Acc) than ours.
        # We add 1 to num and den to avoid p=0 (Laplace smoothing)
        n_better_random = np.sum(np.array(null_acc_dist) >= our_acc)
        p_value = (n_better_random + 1) / (n_null + 1)
        
        # 3. Decision Rules
        better_than_null = our_acc > null_mean
        
        # Criteria:
        # - Significant: p < 0.05
        # - Truly Significant: Better by > 2 sigmas (Z > 2) AND p < 0.05
        
        significant = p_value < 0.05
        
        # Rule: our_acc > null_mean + 2*std
        is_2_sigma_better = z_score > 2.0 
        
        truly_significant = is_2_sigma_better and (p_value < 0.05)

        threshold_good = np.percentile(null_acc_dist, 95)

        status = "Not Significant"
        if significant:
            status = "Significant (Better than random)"
        elif z_score > 1.0:
            status = "Borderline (Good signal, but high variance)"
        
        return {
            'z_score': z_score,
            'p_value': p_value,
            'better_than_null': better_than_null,
            'significantly_better': is_2_sigma_better,
            'significant': significant,
            'truly_significant': truly_significant,
            'null_acc_mean': null_mean,
            'null_acc_std': null_std,
            'null_acc_min': np.min(null_acc_dist),
            'null_acc_max': np.max(null_acc_dist),
            'threshold_good': threshold_good, #null_mean + 2 * null_std,
            'our_acc': our_acc,
            'status_msg': status
        }
# =============================================================================
# HELPERS
# =============================================================================

# =============================================================================
# FEATURE CORRELATION ANALYSIS
# =============================================================================

def print_feature_correlations(data, selected_features, target_col):
    """
    Print selected features with their correlation to target variable
    USING TRAINING DATASET (same as feature selection)
    """
    correlations = []
    
    for feature in selected_features['feature']:
        # Calculate Pearson correlation coefficient
        corr_coef, p_value = stats.pearsonr(data[feature], data[target_col])
        
        correlations.append({
            'Feature': feature,
            'Correlation': corr_coef,
            'p-value': p_value,
            '|Correlation|': abs(corr_coef),  # For sorting
        })
    
    # Create DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations).sort_values('|Correlation|', ascending=False)
    
    print("\n" + "█"*75)
    print(f" FEATURE CORRELATIONS WITH TARGET {target_col} ".center(73))
    print("█"*75)
    print(f"\nAnalysis on: TRAINING DATASET ({len(data)} structures)")
    print(f"Method: Pearson correlation coefficient\n")
    
    print(f"{'Feature':<40} {'Correlation':>12} {'p-value':>15}")
    print("-" * 70)
    
    for idx, row in corr_df.iterrows():

        print(f"{row['Feature']:<40} {row['Correlation']:>12.4f} {row['p-value']:>15.2e}")

    print("\n" + "="*70)
    
    abs_r = corr_df['Correlation'].abs()

    n_strong   = (abs_r > 0.5).sum()
    n_moderate = ((abs_r > 0.3) & (abs_r <= 0.5)).sum()
    n_weak     = ((abs_r > 0.1) & (abs_r <= 0.3)).sum()
    n_neglig   = (abs_r <= 0.1).sum()

    print(f"Strong correlations (|r| > 0.5): {n_strong}")
    print(f"Moderate correlations (0.3 < |r| ≤ 0.5): {n_moderate}")
    print(f"Weak correlations (0.1 < |r| ≤ 0.3): {n_weak}")
    print(f"Negligible correlations (|r| ≤ 0.1): {n_neglig}")
    print("="*70)
    
    return corr_df

# =============================================================================
# FEATURE PRE-FILTERING UTILITIES
# =============================================================================

def drop_high_nan_features(X: pd.DataFrame, target_col: str = TARGET, 
                          max_nan_ratio: float = 0.2) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with excessive missing values (> max_nan_ratio).

    Parameters
    ----------
    X : pd.DataFrame
        Input features + optional target column
    target_col : str, optional
        Target column name to preserve (not removed)
    max_nan_ratio : float
        Maximum NaN ratio (0.0-1.0) to retain feature

    Returns
    -------
    tuple: (filtered_df, kept_columns)
        Cleaned data + list of retained column names

    Notes
    -----
    - Protects target column from removal
    - Handles edge cases gracefully
    - Prints summary statistics
    """

    if target_col is not None and target_col in X.columns:
        features = X.drop(columns=[target_col])
        target = X[[target_col]]
    else:
        features = X
        target = None

    nan_ratio = features.isna().mean()
    kept_cols = nan_ratio[nan_ratio <= max_nan_ratio].index.tolist()

    print(f"\n[NaN Filter] Max NaN ratio: {max_nan_ratio}")
    print(f" Start features: {features.shape[1]}")
    print(f" Kept features: {len(kept_cols)}")
    print(f" Removed: {features.shape[1] - len(kept_cols)}")

    X_reduced = features[kept_cols]

    if target is not None:
        X_reduced = pd.concat([X_reduced, target], axis=1)

    return X_reduced, kept_cols


def remove_low_variance_features(X: pd.DataFrame, target_col: str = TARGET,
                                 threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove near-constant features using VarianceThreshold.

    Parameters
    ----------
    X : pd.DataFrame
        Input features + optional target column
    target_col : str, optional
        Target column name to preserve
    threshold : float
        Minimum variance threshold to retain features

    Returns
    -------
    tuple: (filtered_df, kept_columns)
        Variance-filtered data + list of retained column names

    Notes
    -----
    - Uses sklearn VarianceThreshold on numeric columns only
    - Target column is protected from removal
    - Non-numeric columns are also protected
    """

    if target_col is not None and target_col in X.columns:
        features = X.drop(columns=[target_col])
        target = X[[target_col]]
    else:
        features = X
        target = None

    numeric_cols = features.select_dtypes(include=['number']).columns
    X_numeric = features[numeric_cols]

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_numeric)

    kept_cols = [col for col, keep in zip(numeric_cols, selector.get_support()) if keep]

    X_filtered = selector.transform(X_numeric)

    print(f"\n[Variance Filter] Variance threshold: {threshold}")
    print(f" Start features: {len(numeric_cols)}")
    print(f" Kept features: {len(kept_cols)}")
    print(f" Removed: {len(numeric_cols) - len(kept_cols)}")

    X_reduced = pd.DataFrame(X_filtered, columns=kept_cols, index=X_numeric.index)

    if target is not None:
        X_reduced = pd.concat([X_reduced, target], axis=1)

    return X_reduced, kept_cols


def drop_highly_correlated_features(train_data: pd.DataFrame, target_col: str = TARGET,
                                    threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly collinear feature pairs (|correlation| > threshold).

    For each pair of correlated features, removes the one with 
    LOWER correlation to the target variable.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training data (TRAINING ONLY - prevents data leakage)
    target_col : str
        Target column name (used to determine which feature to keep)
    threshold : float
        Maximum absolute correlation (0.0-1.0) to retain both features

    Returns
    -------
    tuple: (reduced_df, kept_columns)
        Collinearity-filtered data + list of retained column names
    """
    
    # Separate features and target
    features = train_data.drop(columns=[target_col]).select_dtypes(include=['number'])
    target = train_data[target_col]

    # EDGE CASE #1: Less than 2 features
    if features.shape[1] < 2:
        print(f"  Only {features.shape[1]} feature(s), skipping collinearity check")
        return pd.concat([features, target], axis=1), features.columns.tolist()

    # EDGE CASE #2: NaN in features (fill with 0 for calculation)
    if features.isna().any().any():
        features = features.fillna(0)

    # Calculate Feature-Feature Correlation Matrix
    corr_matrix = features.corr().abs()

    # Calculate Feature-Target Correlation Vector
    # Calculates abs correlation of each feature with the target
    target_corrs = features.corrwith(target).abs().fillna(0)

    # Upper triangle (avoid double-counting)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Select feature with LOWER target correlation for removal
    to_drop = set()

    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > threshold:

                if target_corrs[row] < target_corrs[col]:
                    to_drop.add(row)  # Drop row because it predicts target worse
                else:
                    to_drop.add(col)  # Drop col because it predicts target worse

    to_drop = list(to_drop)
    kept = [c for c in features.columns if c not in to_drop]

    print(f"\n[Collinearity Filter] Collinearity threshold: {threshold}")
    print(f" Strategy: Keep feature with higher correlation to '{target_col}'")
    print(f" Start features: {features.shape[1]}")
    print(f" Kept features: {len(kept)}")
    print(f" Removed: {len(to_drop)}")

    reduced = pd.concat([features[kept], target], axis=1)

    return reduced, kept