"""
Hyperparameter tuning with Optuna.
100 trials per model per target.
SQLite persistence for crash-safe resume.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

from .utils import atomic_json_write

logger = logging.getLogger("bp_pipeline")

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
N_TRIALS = 100
CV_FOLDS = 5


def get_search_space(model_name: str, trial: optuna.Trial) -> Dict:
    """Define Optuna search space for each model."""
    if model_name == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True),
        }
    elif model_name == "dt":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    elif model_name == "rf":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    elif model_name == "xgb":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    elif model_name == "lgbm":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500]),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_model(model_name: str, params: Dict):
    """Create a model instance with given parameters."""
    if model_name == "ridge":
        return Ridge(**params)
    elif model_name == "dt":
        return DecisionTreeRegressor(**params, random_state=RANDOM_SEED)
    elif model_name == "rf":
        return RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
    elif model_name == "xgb":
        return xgb.XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
    elif model_name == "lgbm":
        return lgb.LGBMRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def tune_model(
    model_name: str,
    target: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    checkpoint_dir: Path,
    n_trials: int = N_TRIALS,
) -> Dict:
    """
    Run Optuna hyperparameter tuning for one model + target.
    SQLite persistence: survives wall-time kills.
    Resume: completed trials preserved automatically.
    """
    storage_path = checkpoint_dir / f"optuna_{model_name}_{target}.db"
    storage_url = f"sqlite:///{storage_path}"
    study_name = f"{model_name}_{target}"

    logger.info(f"Tuning {model_name} ({target}): {n_trials} trials")
    logger.info(f"  Storage: {storage_path}")

    def objective(trial):
        params = get_search_space(model_name, trial)
        model = create_model(model_name, params)

        scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            n_jobs=1,  # avoid nested parallelism
        )
        return -scores.mean()  # minimize MAE

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Calculate remaining trials
    completed = len(study.trials)
    remaining = n_trials - completed
    if remaining <= 0:
        logger.info(f"  Already completed {completed} trials, skipping")
    else:
        logger.info(f"  {completed} trials already done, running {remaining} more")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    # Save best params
    best_params = study.best_params
    best_value = study.best_value

    params_path = checkpoint_dir / "best_params" / f"{model_name}_{target}.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(params_path, {
        "model": model_name,
        "target": target,
        "best_params": best_params,
        "best_mae": best_value,
        "n_trials": len(study.trials),
    })

    logger.info(f"  Best MAE: {best_value:.4f}")
    logger.info(f"  Best params: {best_params}")

    return best_params


def tune_all(
    X_train: np.ndarray,
    y_train_sbp: np.ndarray,
    y_train_dbp: np.ndarray,
    checkpoint_dir: Path,
    models_to_tune: Optional[list] = None,
    n_trials: int = N_TRIALS,
) -> Dict:
    """Tune all models for both targets. Returns best_params dict."""
    if models_to_tune is None:
        models_to_tune = ["ridge", "dt", "rf", "xgb", "lgbm"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    all_params = {}

    for model_name in models_to_tune:
        for target in ["sbp", "dbp"]:
            key = f"{model_name}_{target}"
            y = y_train_sbp if target == "sbp" else y_train_dbp

            # Check if already tuned
            params_path = checkpoint_dir / "best_params" / f"{model_name}_{target}.json"
            if params_path.exists():
                import json
                with open(params_path) as f:
                    saved = json.load(f)
                if saved.get("n_trials", 0) >= n_trials:
                    logger.info(f"SKIP tuning {key} (already have {saved['n_trials']} trials)")
                    all_params[key] = saved["best_params"]
                    continue

            params = tune_model(model_name, target, X_train, y, checkpoint_dir, n_trials)
            all_params[key] = params

    return all_params
