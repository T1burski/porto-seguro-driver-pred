import optuna
import os
import yaml
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple

os.environ['LIGHTGBM_VERBOSITY'] = '-1'
os.environ['OPTUNA_VERBOSITY'] = 'WARNING'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pr_auc_score(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    root_dir = Path(__file__).parent.parent
    config_path = root_dir / "config" / "model_configs.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_base_train_params(config: Dict[str, Any]):
    """Extract base LightGBM parameters, excluding search_space and training-specific params."""
    base_params = config["lightgbm_params"].copy()

    training_params = {}

    training_params["eval_metric"] = base_params["eval_metric"]

    params_to_remove = ["search_space", "eval_metric"] 
    for param in params_to_remove:
        base_params.pop(param, None)
    
    return base_params, training_params

def suggest_hyperparameters(trial: optuna.Trial, config: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest hyperparameters based on search space configuration."""
    search_space = config["lightgbm_params"]["search_space"]
    suggested_params = {}
    
    for param_name, param_config in search_space.items():
        if "choices" in param_config:
            # Categorical parameter
            suggested_params[param_name] = trial.suggest_categorical(
                param_name, param_config["choices"]
            )
        elif "low" in param_config and "high" in param_config:
            # Numeric parameter
            if param_name in ["num_leaves", "max_depth", "min_child_samples", "n_estimators"]:
                # Integer parameters
                suggested_params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
            else:
                # Float parameters
                suggested_params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
    
    return suggested_params

def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[
            ('onehot', 
             OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), 
             lambda X: [c for c in X.columns if (len(X[c].unique()) < 10) and 
                       (c.endswith('_cat') or c.endswith('_bin'))]),
            ('target_encoder', 
             TargetEncoder(smooth="auto"), 
             lambda X: [c for c in X.columns if (len(X[c].unique()) >= 10) and 
                       (c.endswith('_cat') or c.endswith('_bin'))])
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> float:
    """Objective function for Optuna optimization."""
    try:
        # Get base parameters (valid for LGBMClassifier constructor)
        base_params = get_base_train_params(config)[0]
        
        # Get suggested hyperparameters
        suggested_params = suggest_hyperparameters(trial, config)
        
        # Get training parameters (for fit method)
        training_params = get_base_train_params(config)[1]
        
        # Combine base and suggested parameters for model constructor
        model_params = {**base_params, **suggested_params}
        
        # Handle special case for scale_pos_weight
        if model_params.get("scale_pos_weight") == "balanced":
            pos_count = sum(y)
            neg_count = len(y) - pos_count
            model_params["scale_pos_weight"] = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=config["optuna_settings"]["cv_folds"],
            shuffle=True,
            random_state=config["lightgbm_params"]["random_state"]
        )
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create and fit preprocessor
            preprocessor = create_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train, y_train)
            X_val_transformed = preprocessor.transform(X_val)
            
            # Create and train model
            model = LGBMClassifier(**model_params)
            
            model.fit(
                X_train_transformed, y_train,
                eval_set=[(X_val_transformed, y_val)],
                callbacks=[
                    early_stopping(stopping_rounds=config["optuna_settings"]["early_stopping_rounds"]),
                    log_evaluation(0)
                ],
                **training_params 
            )
            
            # Predict and calculate score
            y_pred_proba = model.predict_proba(X_val_transformed)[:, 1]
            fold_score = pr_auc_score(y_val, y_pred_proba)
            scores.append(fold_score)
        
        mean_score = np.mean(scores)
        
        return mean_score
        
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        # Return a high score to indicate failure
        return float('inf')

def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Optimize hyperparameters using Optuna.
    
    Returns:
        Tuple of (best_params, study_object)
    """
    config = load_config()
    
    # Create study
    study = optuna.create_study(
        direction=config["optuna_settings"]["direction"],
        sampler=optuna.samplers.TPESampler(
            seed=config["lightgbm_params"]["random_state"],
            n_startup_trials=10,
            n_ei_candidates=24
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
    )
    
    logger.info("Starting hyperparameter optimization...")
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, config),
        n_trials=config["optuna_settings"]["n_trials"],
        timeout=config["optuna_settings"]["timeout"],
        show_progress_bar=False,
        gc_after_trial=True
    )
    
    logger.info("Optimization completed!")
    logger.info(f"Best score: {study.best_value:.6f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    return study.best_params, study

