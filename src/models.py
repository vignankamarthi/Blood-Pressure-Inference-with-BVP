"""
Model training with exhaustive checkpointing.
5 models x 2 targets = 10 models total.
Every model checkpointed at ~30-second granularity.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

from .utils import atomic_json_write, load_json, clear_memory

logger = logging.getLogger("bp_pipeline")


def load_status(checkpoint_dir: Path) -> Dict:
    """Load training status checkpoint."""
    status_path = checkpoint_dir / "status.json"
    return load_json(status_path) or {"completed": [], "in_progress": None}


def save_status(checkpoint_dir: Path, status: Dict) -> None:
    """Save training status checkpoint."""
    atomic_json_write(checkpoint_dir / "status.json", status)


def is_model_done(checkpoint_dir: Path, model_name: str, target: str) -> bool:
    """Check if a model has already been trained and saved."""
    model_path = checkpoint_dir / "models" / f"{model_name}_{target}.pkl"
    status = load_status(checkpoint_dir)
    key = f"{model_name}_{target}"
    return model_path.exists() and key in status.get("completed", [])


def train_ridge(X_train, y_train, params: Dict, checkpoint_dir: Path, target: str):
    """Train Ridge regression. Fast -- no mid-training checkpoint needed."""
    logger.info(f"Training Ridge ({target})...")
    model = Ridge(**params)
    model.fit(X_train, y_train)
    model_path = checkpoint_dir / "models" / f"ridge_{target}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"  Saved to {model_path}")
    return model


def train_decision_tree(X_train, y_train, params: Dict, checkpoint_dir: Path, target: str):
    """Train Decision Tree. Fast -- no mid-training checkpoint needed."""
    logger.info(f"Training DecisionTree ({target})...")
    model = DecisionTreeRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    model_path = checkpoint_dir / "models" / f"dt_{target}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"  Saved to {model_path}")
    return model


def train_random_forest(
    X_train, y_train, params: Dict, checkpoint_dir: Path, target: str
):
    """
    Train Random Forest with warm_start checkpointing.
    Saves every 50 trees. Resume loads latest checkpoint.
    """
    logger.info(f"Training RandomForest ({target})...")
    models_dir = checkpoint_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    target_trees = params.get("n_estimators", 500)
    step = 50  # checkpoint every 50 trees

    # Check for existing progress
    progress_path = checkpoint_dir / f"rf_progress_{target}.json"
    progress = load_json(progress_path) or {"current_trees": 0}
    start_trees = progress["current_trees"]

    if start_trees > 0:
        # Load existing model
        latest_path = models_dir / f"rf_{target}_n{start_trees}.pkl"
        if latest_path.exists():
            logger.info(f"  Resuming from {start_trees} trees")
            model = joblib.load(latest_path)
        else:
            logger.info(f"  Progress file says {start_trees} but model not found, starting fresh")
            start_trees = 0
            model = None
    else:
        model = None

    current = start_trees
    while current < target_trees:
        next_target = min(current + step, target_trees)

        if model is None:
            model = RandomForestRegressor(
                n_estimators=next_target,
                warm_start=True,
                random_state=42,
                n_jobs=-1,
                **{k: v for k, v in params.items() if k != "n_estimators"}
            )
        else:
            model.n_estimators = next_target

        model.fit(X_train, y_train)
        current = next_target

        # Checkpoint
        ckpt_path = models_dir / f"rf_{target}_n{current}.pkl"
        joblib.dump(model, ckpt_path)
        atomic_json_write(progress_path, {"current_trees": current, "target_trees": target_trees})
        logger.info(f"  RF checkpoint: {current}/{target_trees} trees saved")

    # Save final model
    final_path = models_dir / f"rf_{target}.pkl"
    joblib.dump(model, final_path)
    logger.info(f"  Final RF saved to {final_path}")
    return model


def train_xgboost(X_train, y_train, params: Dict, checkpoint_dir: Path, target: str):
    """
    Train XGBoost with periodic checkpointing.
    Saves model after training completes. For mid-training resume,
    uses incremental training with warm start.
    """
    logger.info(f"Training XGBoost ({target})...")
    models_dir = checkpoint_dir / "models"
    xgb_ckpt_dir = checkpoint_dir / f"xgb_{target}"
    xgb_ckpt_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    n_rounds = params.pop("n_estimators", 500)

    # Check for existing checkpoint
    progress_path = checkpoint_dir / f"xgb_progress_{target}.json"
    progress = load_json(progress_path) or {"current_round": 0}
    start_round = progress["current_round"]

    if start_round >= n_rounds:
        final_path = models_dir / f"xgb_{target}.pkl"
        if final_path.exists():
            return joblib.load(final_path)

    # Incremental training with checkpointing every 'step' rounds
    step = 25
    current = start_round
    model = None

    # Load existing model if resuming
    if current > 0:
        ckpt_path = xgb_ckpt_dir / f"xgb_round_{current}.json"
        if ckpt_path.exists():
            logger.info(f"  Resuming XGBoost from round {current}")
            model = xgb.XGBRegressor(
                n_estimators=step, random_state=42, n_jobs=-1, **params)
            model.fit(X_train[:1], y_train[:1])  # dummy fit to init
            model.get_booster().load_model(str(ckpt_path))

    while current < n_rounds:
        next_target = min(current + step, n_rounds)
        batch_size = next_target - current

        model = xgb.XGBRegressor(
            n_estimators=batch_size, random_state=42, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        current = next_target

        # Save checkpoint
        ckpt_path = xgb_ckpt_dir / f"xgb_round_{current}.json"
        model.get_booster().save_model(str(ckpt_path))
        atomic_json_write(progress_path, {"current_round": current, "target_rounds": n_rounds})
        logger.info(f"  XGBoost checkpoint: {current}/{n_rounds} rounds")

    # Save final sklearn-compatible model
    final_path = models_dir / f"xgb_{target}.pkl"
    joblib.dump(model, final_path)
    logger.info(f"  Final XGBoost saved to {final_path}")
    return model


class LGBMCheckpointCallback:
    """Custom LightGBM callback for checkpointing every N iterations."""

    def __init__(self, checkpoint_dir: Path, target: str, interval: int = 25):
        self.checkpoint_dir = checkpoint_dir
        self.target = target
        self.interval = interval
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, env):
        iteration = env.iteration + 1
        if iteration % self.interval == 0:
            path = self.checkpoint_dir / f"lgbm_iter_{iteration}.txt"
            env.model.save_model(str(path))
            progress_path = self.checkpoint_dir.parent / f"lgbm_progress_{self.target}.json"
            atomic_json_write(progress_path, {
                "current_iter": iteration,
                "target_iter": env.end_iteration,
            })


def train_lightgbm(X_train, y_train, params: Dict, checkpoint_dir: Path, target: str):
    """
    Train LightGBM with custom checkpoint callback.
    Saves every 25 iterations. Resume via init_model.
    """
    logger.info(f"Training LightGBM ({target})...")
    models_dir = checkpoint_dir / "models"
    lgbm_ckpt_dir = checkpoint_dir / f"lgbm_{target}"
    lgbm_ckpt_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    n_estimators = params.pop("n_estimators", 500)

    # Check for existing progress
    progress_path = checkpoint_dir / f"lgbm_progress_{target}.json"
    progress = load_json(progress_path) or {"current_iter": 0}
    start_iter = progress["current_iter"]

    init_model = None
    remaining = n_estimators - start_iter
    if start_iter > 0 and remaining > 0:
        # Find latest checkpoint
        ckpt_files = sorted(lgbm_ckpt_dir.glob("lgbm_iter_*.txt"))
        if ckpt_files:
            init_model = str(ckpt_files[-1])
            logger.info(f"  Resuming LightGBM from iteration {start_iter}")

    if remaining <= 0:
        final_path = models_dir / f"lgbm_{target}.pkl"
        if final_path.exists():
            return joblib.load(final_path)

    callback = LGBMCheckpointCallback(lgbm_ckpt_dir, target, interval=25)

    model = lgb.LGBMRegressor(
        n_estimators=remaining,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params,
    )

    model.fit(
        X_train, y_train,
        init_model=init_model,
        callbacks=[callback],
    )

    # Save final
    final_path = models_dir / f"lgbm_{target}.pkl"
    joblib.dump(model, final_path)
    atomic_json_write(progress_path, {"current_iter": n_estimators, "target_iter": n_estimators})
    logger.info(f"  Final LightGBM saved to {final_path}")
    return model


# Model registry
MODELS = {
    "ridge": train_ridge,
    "dt": train_decision_tree,
    "rf": train_random_forest,
    "xgb": train_xgboost,
    "lgbm": train_lightgbm,
}

TARGETS = ["sbp", "dbp"]


def train_all(
    X_train, y_train_sbp, y_train_dbp,
    best_params: Dict,
    checkpoint_dir: Path,
    models_to_train: Optional[list] = None,
    resume: bool = False,
) -> Dict:
    """
    Train all models for both SBP and DBP targets.
    Returns dict of {model_target: trained_model}.
    """
    if models_to_train is None:
        models_to_train = list(MODELS.keys())

    trained = {}
    status = load_status(checkpoint_dir)

    for model_name in models_to_train:
        if model_name not in MODELS:
            logger.warning(f"Unknown model: {model_name}")
            continue

        train_fn = MODELS[model_name]

        for target in TARGETS:
            key = f"{model_name}_{target}"

            # Check if already done (resume logic)
            if resume and is_model_done(checkpoint_dir, model_name, target):
                logger.info(f"SKIP {key} (already completed)")
                model_path = checkpoint_dir / "models" / f"{model_name}_{target}.pkl"
                trained[key] = joblib.load(model_path)
                continue

            # Update status
            status["in_progress"] = key
            save_status(checkpoint_dir, status)

            # Get target array
            y = y_train_sbp if target == "sbp" else y_train_dbp

            # Get params for this model
            params = best_params.get(key, best_params.get(model_name, {})).copy()

            # Train
            try:
                model = train_fn(X_train, y, params, checkpoint_dir, target)
                trained[key] = model

                # Mark completed
                if key not in status["completed"]:
                    status["completed"].append(key)
                status["in_progress"] = None
                save_status(checkpoint_dir, status)

            except Exception as e:
                logger.error(f"FAILED {key}: {e}")
                status["in_progress"] = None
                save_status(checkpoint_dir, status)
                raise

            clear_memory()

    return trained
