"""
Central configuration of hyperparameters for all ML algorithms in QSPR
================================================================================
Single Source of Truth - all hyperparameters in one place
Automatically passed to optimization in module_creation.py

Usage:
    from hyperparameters import HyperparametersConfig, HyperparametersDisplay
    
    # Retrieve parameters for GridSearchCV
    param_grid = HyperparametersConfig.RANDOM_FOREST
    
    # Display parameters in the console
    HyperparametersDisplay.print_hyperparameters(‘RANDOM_FOREST’)
"""



from typing import Dict, List, Any, Union
import warnings
import numpy as np


class HyperparametersConfig:
    """
    Central definition of hyperparameters for GridSearchCV/RandomizedSearchCV.
    
    Each algorithm has:
    - Full parameter configuration
    - Explanatory comments
    - Ranges of tested values"""

    
    # ==================== TREE-BASED ALGORITHMS ====================
    
    RANDOM_FOREST = {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 0.5],
            'max_samples': [None, 0.8, 0.9],
            'bootstrap': [True],
            'max_depth': [2, 3],
            'min_samples_split': [5, 6, 7],
            'min_samples_leaf': [3, 4, 5],
            'class_weight': ['balanced', 'balanced_subsample'],
        }
   
    XGBOOST = {
            'n_estimators': [30, 40, 50, 100],
            'colsample_bytree': [0.5, 0.7],
            'learning_rate': [0.05, 0.1],
            'max_depth': [1],
            'subsample': [0.6, 0.7, 0.8],
            'reg_alpha': [0.1, 1, 10],
            'reg_lambda': [1, 5, 10],
            'min_child_weight': [3, 5],
        }
   
    GRADIENT_BOOSTING = {
            'n_estimators': [30, 40, 50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [1],
            'subsample': [0.7, 0.8],
            'max_features': ['sqrt', 0.5],
            'min_samples_split': [5, 8],
            'min_samples_leaf': [3, 4, 5],
        }
        
    # ==================== SVM ALGORITHMS ====================
   
    SVM = {
            'C': [0.1, 0.5, 1, 2, 5, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'class_weight': ['balanced'],
    }
   
    LOGISTIC_REGRESSION = {
        'C': [0.01, 0.1, 0.5, 1, 2, 5, 10], 
        'penalty': ['l2', 'l1'],            
        'solver': ['liblinear'],           
        'class_weight': ['balanced'],
    }
    # ==================== DISTANCE-BASED ALGORITHMS ====================
   
    KNN = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform'],
            'algorithm': ['auto'],
            'p': [1, 1.5, 2, 3, np.inf],
            'metric': ['minkowski'],
    }
   
   
    # ==================== REGRESSION TREE ALGORITHMS ====================
   
    DECISION_TREE = {
            'max_depth': [2, 3],
            'min_samples_split': [5, 8],
            'min_samples_leaf': [4, 6],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', None],
            'class_weight': ['balanced'],
            'ccp_alpha': [0.0, 0.01, 0.02],
    } 
    
    @classmethod
    def get_all_algorithms(cls) -> Dict[str, Dict[str, List[Any]]]:
        """
        Returns a dictionary of all available algorithms and their parameters.
        
        Returns:
            Dict with keys = algorithm names, values = parameters"""
        algorithms = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name.isupper():
                attr = getattr(cls, attr_name)
                if isinstance(attr, dict) and not callable(attr):
                    algorithms[attr_name] = attr
        return algorithms
    
    @classmethod
    def get_algorithm(cls, algorithm_name: str) -> Dict[str, List[Any]]:
        """
        Returns the parameters for the selected algorithm.
        
        Args:
            algorithm_name: The name of the algorithm (e.g., ‘RANDOM_FOREST’)
            
        Returns:
            A dictionary of parameters
            
        Raises:
            AttributeError: If the algorithm does not exist
        """
        algorithm_name = algorithm_name.upper()
        if not hasattr(cls, algorithm_name):
            available = ', '.join(cls.get_all_algorithms().keys())
            raise AttributeError(
                f"Algorithm '{algorithm_name}' does not exist.\n"
                f"Available algorithms: {available}"
            )
        return getattr(cls, algorithm_name)


class HyperparametersDisplay:
    """
    Formatting and displaying hyperparameters.
    Supports automatic printing of parameters to the console."""
    
    @staticmethod
    def print_hyperparameters(algo_name: str, verbose: bool = True) -> Dict[str, List[Any]]:
        """
        Prints the hyperparameters of the algorithm to the console and returns them.
        
        Args:
            algo_name: Name of the algorithm (e.g., ‘RANDOM_FOREST’)
            verbose: Whether to print to the console (default: True)
            
        Returns:
            Dictionary of parameters
            
        Example:
            >>> HyperparametersDisplay.print_hyperparameters(‘RANDOM_FOREST’)
            Testing n_estimators: [2, 4, 8, 16, 32, 50, 75, 100]
            Testing max_depth: [5, 10, 15, 20, None]
            ...
        """
        params = HyperparametersConfig.get_algorithm(algo_name)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"  Algorithm hyperparameters: {algo_name.upper()}")
            print(f"{'='*70}")
            for param_name, values in params.items():
                print(f"   Testing {param_name}: {values}")
            print(f"{'='*70}\n")
        
        return params
    
    @staticmethod
    def format_parameter_table(algo_name: str) -> str:
        """
        Formats parameters as a text table.
        
        Args:
            algo_name: Algorithm name
            
        Returns:
            String with formatted table
        """
        params = HyperparametersConfig.get_algorithm(algo_name)
        
        header = f"\n{'Parameter':<30} | {'Values':<50}\n"
        separator = "-" * 85
        
        table = header + separator + "\n"
        for param_name, values in params.items():
            values_str = str(values)[:47] + "..." if len(str(values)) > 50 else str(values)
            table += f"{param_name:<30} | {values_str:<50}\n"
        
        return table
    
    @staticmethod
    def get_param_count(algo_name: str) -> int:
        """
        Returns the number of hyperparameter combinations (approximate).
        
        Args:
            algo_name: Algorithm name
            
        Returns:
            Number of combinations
        """
        params = HyperparametersConfig.get_algorithm(algo_name)
        combinations = 1
        for values in params.values():
            combinations *= len(values)
        return combinations
    
    @staticmethod
    def print_algorithm_summary() -> None:
        """
        Prints a summary of all available algorithms
        with the number of parameter combinations.
        """
        all_algorithms = HyperparametersConfig.get_all_algorithms()
        
        print(f"\n{'='*80}")
        print(f"  Available algorithms-summary")
        print(f"{'='*80}")
        print(f"{'Algorithm':<30} | {'Param Count':<15} | {'Est. Combinations':<20}")
        print(f"{'-'*80}")
        
        for algo_name in sorted(all_algorithms.keys()):
            param_count = len(all_algorithms[algo_name])
            combinations = HyperparametersDisplay.get_param_count(algo_name)
            print(
                f"{algo_name:<30} | {param_count:<15} | "
                f"{combinations:>18,}"
            )
        
        print(f"{'='*80}\n")


class HyperparametersValidator:
    """
    Validation and verification of hyperparameter configuration.
    """
    
    @staticmethod
    def validate_config() -> bool:
        """
        Checks if all configurations are correct.
        
        Returns:
            True if everything is OK, False if there are problems"""
        all_algorithms = HyperparametersConfig.get_all_algorithms()
        issues = []
        
        for algo_name, params in all_algorithms.items():
            # Checking if parameters are not empty
            if not params:
                issues.append(f"{algo_name}: no params")
                continue
            
            # Checking if all values are inside list
            for param_name, values in params.items():
                if not isinstance(values, (list, tuple)):
                    issues.append(
                        f"{algo_name}.{param_name}: value must be a list / tuple, "
                        f"got {type(values)}"
                    )
        
        if issues:
            print("CONFIGURATION ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("The hyperparameter configuration is correct!")
            return True
    
    @staticmethod
    def get_warnings() -> List[str]:
        """
        Returns configuration warnings.
        
        Returns:
            List of warnings
        """
        warnings_list = []
        all_algorithms = HyperparametersConfig.get_all_algorithms()
        
        for algo_name, params in all_algorithms.items():
            total_combinations = HyperparametersDisplay.get_param_count(algo_name)
            
            # Warning about too many combinations
            if total_combinations > 10000:
                warnings_list.append(
                    f"{algo_name}: ~{total_combinations:,} combinations - "
                    f"GridSearchCV will be VERY SLOW. Consider RandomizedSearchCV instead."
                )
            
            # Warning about too few combinations
            elif total_combinations < 10:
                warnings_list.append(
                    f"{algo_name}: merely {total_combinations} combinations - "
                    f"may not be sufficient for a thorough search."
                )
        
        return warnings_list