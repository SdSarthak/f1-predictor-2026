"""
F1 Race Predictor - Machine Learning Model Trainer
Supports XGBoost, Random Forest, and Ensemble models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import joblib
import yaml

from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1ModelTrainer:
    """
    Machine Learning model trainer for F1 race position prediction.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_column: str = 'finish_position'
        self.model_type: str = 'xgboost'
        self.feature_importance: Dict[str, float] = {}
        # Race key per prepared row, used to keep a race's cars on one side of
        # the train/test split. Set by `prepare_data`.
        self.groups: Optional[np.ndarray] = None
        # Fixed generator for the input-perturbation uncertainty estimate.
        self.uncertainty_seed: int = 42

    # Columns that identify a single race. Every car in a race shares its
    # circuit, weather and safety-car history, and finishing positions are a
    # permutation within it - so splitting a race across train and test leaks.
    GROUP_COLUMNS = ('year', 'round')

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration, falling back to the built-in hyperparameters."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(f"Could not load {config_path}: {exc}. Using defaults.")
            return {}
    
    def prepare_data(self, df: pd.DataFrame, 
                    feature_columns: List[str],
                    target_column: str = 'finish_position') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Filter to available columns
        available_features = [c for c in feature_columns if c in df.columns]
        missing_features = set(feature_columns) - set(available_features)
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        if not available_features:
            raise ValueError(
                "None of the requested feature columns are present in the dataset"
            )
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is not in the dataset")

        self.feature_columns = available_features

        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_column])
        if df_clean.empty:
            raise ValueError(
                f"No rows left after dropping missing '{target_column}' values"
            )

        # Fill missing features with appropriate values
        X = df_clean[available_features].copy()
        y = df_clean[target_column].copy()

        # Fill NaN with column means
        X = X.fillna(X.mean())

        # Replace any remaining NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        # Record which race each row belongs to, aligned with the filtered rows.
        if all(column in df_clean.columns for column in self.GROUP_COLUMNS):
            self.groups = (
                df_clean[list(self.GROUP_COLUMNS)]
                .astype(str)
                .agg('_'.join, axis=1)
                .to_numpy()
            )
        else:
            logger.warning(
                f"Columns {self.GROUP_COLUMNS} unavailable; falling back to a "
                "row-level split, which lets a race straddle train and test"
            )
            self.groups = None

        return X.values, y.values

    def _split(self,
               X: np.ndarray,
               y: np.ndarray,
               groups: Optional[np.ndarray],
               test_size: float,
               random_state: int):
        """
        Hold out whole races where possible.

        A random row split puts 19 of a race's 20 cars in training and the 20th
        in test. Finishing position is a ranking within the race, so the model
        is then scored on a race it has almost entirely memorised.
        """
        if groups is not None and len(np.unique(groups)) > 1:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(self, X: np.ndarray, y: np.ndarray,
             model_type: str = None,
             test_size: float = 0.2,
             cross_validate: bool = True,
             groups: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Target vector
            model_type: 'xgboost', 'random_forest', or 'ensemble'
            test_size: Proportion for test set
            cross_validate: Whether to perform cross-validation
            groups: Race key per row; defaults to what `prepare_data` recorded.
                Whole races are held out together so a race's other 19 cars are
                never visible while scoring the 20th.

        Returns:
            Dictionary of evaluation metrics
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"X {X.shape} and y {y.shape} are not aligned")
        if len(X) < 5:
            raise ValueError(f"Need at least 5 samples to train, got {len(X)}")

        model_config = self.config.get('model', {})
        self.model_type = model_type or model_config.get('type', 'xgboost')
        random_state = model_config.get('random_state', 42)

        if groups is None and self.groups is not None and len(self.groups) == len(y):
            groups = self.groups

        # Split first, then scale. Fitting the scaler on the full matrix leaks
        # the test set's mean and variance into training.
        X_train_raw, X_test_raw, y_train, y_test = self._split(
            X, y, groups, test_size, random_state
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Build model
        if self.model_type == 'xgboost':
            self.model = self._build_xgboost(model_config.get('xgboost', {}))
        elif self.model_type == 'random_forest':
            self.model = self._build_random_forest(model_config.get('random_forest', {}))
        elif self.model_type == 'ensemble':
            self.model = self._build_ensemble(model_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)

        # Cross-validation
        if cross_validate:
            cv_metrics = self._cross_validate(
                X, y, groups, model_config.get('cv_folds', 5)
            )
            metrics.update(cv_metrics)

        # Feature importance
        self._extract_feature_importance()

        return metrics

    def _cross_validate(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        groups: Optional[np.ndarray],
                        cv_folds: int) -> Dict[str, float]:
        """
        Cross-validate with the scaler inside the fold.

        The scaler has to be refit per fold, otherwise every fold's held-out
        rows have already contributed their mean and variance to the transform.
        """
        # Each fold needs an untouched estimator, and the scaler must be refit
        # inside the fold rather than shared across folds.
        fold_model = self._clone_model()
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', fold_model)])

        if groups is not None:
            n_groups = len(np.unique(groups))
            if n_groups < 2:
                logger.warning("Only one race in the dataset; skipping cross-validation")
                return {}
            splitter = GroupKFold(n_splits=min(cv_folds, n_groups))
        else:
            splitter = KFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=42)
            groups = None

        cv_scores = cross_val_score(
            pipeline, X, y,
            groups=groups,
            cv=splitter,
            scoring='neg_mean_absolute_error',
        )

        logger.info(
            f"Cross-validation MAE: {-cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})"
        )
        return {
            'cv_mae_mean': float(-cv_scores.mean()),
            'cv_mae_std': float(cv_scores.std()),
        }

    def _clone_model(self):
        """A fresh, unfitted estimator of the configured type."""
        model_config = self.config.get('model', {})
        if self.model_type == 'random_forest':
            return self._build_random_forest(model_config.get('random_forest', {}))
        if self.model_type == 'ensemble':
            return self._build_ensemble(model_config)
        return self._build_xgboost(model_config.get('xgboost', {}))

    def _build_xgboost(self, params: Dict) -> xgb.XGBRegressor:
        """Build XGBoost model."""
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror',
        }
        default_params.update(params)
        
        return xgb.XGBRegressor(**default_params)
    
    def _build_random_forest(self, params: Dict) -> RandomForestRegressor:
        """Build Random Forest model."""
        default_params = {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
        }
        default_params.update(params)
        
        return RandomForestRegressor(**default_params)
    
    def _build_ensemble(self, config: Dict) -> VotingRegressor:
        """Build ensemble of XGBoost and Random Forest."""
        xgb_model = self._build_xgboost(config.get('xgboost', {}))
        rf_model = self._build_random_forest(config.get('random_forest', {}))
        
        return VotingRegressor([
            ('xgboost', xgb_model),
            ('random_forest', rf_model),
        ])
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Position accuracy (within N positions)
        within_1 = np.mean(np.abs(y_true - y_pred) <= 1)
        within_2 = np.mean(np.abs(y_true - y_pred) <= 2)
        within_3 = np.mean(np.abs(y_true - y_pred) <= 3)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'within_1_position': within_1,
            'within_2_positions': within_2,
            'within_3_positions': within_3,
        }
        
        logger.info(f"Model Performance:")
        logger.info(f"  MAE: {mae:.3f} positions")
        logger.info(f"  RMSE: {rmse:.3f}")
        logger.info(f"  R²: {r2:.3f}")
        logger.info(f"  Within ±1 position: {within_1*100:.1f}%")
        logger.info(f"  Within ±2 positions: {within_2*100:.1f}%")
        logger.info(f"  Within ±3 positions: {within_3*100:.1f}%")
        
        return metrics
    
    def _extract_feature_importance(self):
        """Extract and store feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # For VotingRegressor, average importance
            importance = np.zeros(len(self.feature_columns))
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importance += estimator.feature_importances_
            importance /= len(self.model.estimators_)
        else:
            return
        
        self.feature_importance = dict(zip(self.feature_columns, importance))
        
        # Sort and log
        sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("\nTop 10 Feature Importance:")
        for feature, imp in sorted_importance[:10]:
            logger.info(f"  {feature}: {imp:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid position range
        predictions = np.clip(predictions, 1, 20)
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        Uses bootstrap resampling for XGBoost/RF.
        
        Returns:
            predictions: Mean predicted positions
            std: Standard deviation of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)

        if isinstance(self.model, VotingRegressor):
            # Spread across the ensemble members is the uncertainty signal.
            all_preds = np.array([
                estimator.predict(X_scaled)
                for estimator in self.model.named_estimators_.values()
            ])
            predictions = all_preds.mean(axis=0)
            std = all_preds.std(axis=0)
        elif isinstance(self.model, RandomForestRegressor):
            # Spread across the individual trees.
            all_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            predictions = all_preds.mean(axis=0)
            std = all_preds.std(axis=0)
        elif isinstance(self.model, xgb.XGBRegressor):
            # Boosted trees are deterministic, so perturb the inputs instead.
            predictions = self.model.predict(X_scaled)
            std = self._estimate_uncertainty_perturbation(X_scaled, n_samples)
        else:
            predictions = self.model.predict(X_scaled)
            std = np.ones(len(predictions)) * 1.5  # Default uncertainty

        # A degenerate (zero) spread makes the Monte Carlo stage deterministic,
        # so keep a small floor on the reported uncertainty.
        std = np.maximum(np.asarray(std, dtype=float), 0.25)

        return np.clip(predictions, 1, 20), std
    
    def _estimate_uncertainty_perturbation(self, X: np.ndarray, n_samples: int = 50) -> np.ndarray:
        """
        Estimate uncertainty by perturbing inputs.

        Uses a fixed generator so two `--predict` runs over the same model and
        grid report the same uncertainty; the global numpy state used to make
        every run differ.
        """
        if n_samples < 2:
            raise ValueError(f"Need at least 2 perturbation samples, got {n_samples}")

        noise_level = 0.05
        rng = np.random.default_rng(self.uncertainty_seed)
        all_preds = []

        for _ in range(n_samples):
            noise = rng.normal(0, noise_level, X.shape)
            X_noisy = X + noise
            preds = self.model.predict(X_noisy)
            all_preds.append(preds)

        return np.std(all_preds, axis=0)
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        """
        logger.info("Starting hyperparameter tuning...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [300, 500, 700],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
            }
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        else:
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            base_model, 
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV MAE: {-grid_search.best_score_:.3f}")
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def save_model(self, path: str = "models/f1_predictor.joblib"):
        """Save the trained model."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'uncertainty_seed': self.uncertainty_seed,
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = "models/f1_predictor.joblib"):
        """Load a trained model."""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']
        self.uncertainty_seed = model_data.get('uncertainty_seed', 42)
        
        logger.info(f"Model loaded from {path}")


def train_model(df: pd.DataFrame, 
               feature_columns: List[str],
               target_column: str = 'finish_position',
               model_type: str = 'xgboost',
               save_path: str = "models/f1_predictor.joblib") -> Tuple[F1ModelTrainer, Dict]:
    """
    Main function to train the F1 prediction model.
    
    Returns:
        trainer: Trained F1ModelTrainer instance
        metrics: Dictionary of evaluation metrics
    """
    trainer = F1ModelTrainer()
    
    # Prepare data
    X, y = trainer.prepare_data(df, feature_columns, target_column)
    
    # Train
    metrics = trainer.train(X, y, model_type=model_type)
    
    # Save
    trainer.save_model(save_path)
    
    return trainer, metrics


