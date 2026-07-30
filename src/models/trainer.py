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

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
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
        
        self.feature_columns = available_features
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_column])
        
        # Fill missing features with appropriate values
        X = df_clean[available_features].copy()
        y = df_clean[target_column].copy()
        
        # Fill NaN with column means
        X = X.fillna(X.mean())
        
        # Replace any remaining NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X.values, y.values
    
    def train(self, X: np.ndarray, y: np.ndarray,
             model_type: str = None,
             test_size: float = 0.2,
             cross_validate: bool = True) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: 'xgboost', 'random_forest', or 'ensemble'
            test_size: Proportion for test set
            cross_validate: Whether to perform cross-validation
            
        Returns:
            Dictionary of evaluation metrics
        """
        model_config = self.config.get('model', {})
        self.model_type = model_type or model_config.get('type', 'xgboost')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=model_config.get('random_state', 42)
        )
        
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
            cv_folds = model_config.get('cv_folds', 5)
            cv_scores = cross_val_score(
                self.model, X_scaled, y, 
                cv=cv_folds, 
                scoring='neg_mean_absolute_error'
            )
            metrics['cv_mae_mean'] = -cv_scores.mean()
            metrics['cv_mae_std'] = cv_scores.std()
            logger.info(f"Cross-validation MAE: {metrics['cv_mae_mean']:.3f} (+/- {metrics['cv_mae_std']:.3f})")
        
        # Feature importance
        self._extract_feature_importance()
        
        return metrics
    
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
        
        if hasattr(self.model, 'estimators_'):
            # For ensemble models, use individual estimator predictions
            all_preds = []
            for name, estimator in self.model.named_estimators_.items():
                preds = estimator.predict(X_scaled)
                all_preds.append(preds)
            
            all_preds = np.array(all_preds)
            predictions = all_preds.mean(axis=0)
            std = all_preds.std(axis=0)
        elif isinstance(self.model, xgb.XGBRegressor):
            # For XGBoost, use iterations
            predictions = self.model.predict(X_scaled)
            
            # Approximate uncertainty using feature perturbation
            std = self._estimate_uncertainty_perturbation(X_scaled, n_samples)
        else:
            predictions = self.model.predict(X_scaled)
            std = np.ones(len(predictions)) * 1.5  # Default uncertainty
        
        return np.clip(predictions, 1, 20), std
    
    def _estimate_uncertainty_perturbation(self, X: np.ndarray, n_samples: int = 50) -> np.ndarray:
        """Estimate uncertainty by perturbing inputs."""
        noise_level = 0.05
        all_preds = []
        
        for _ in range(n_samples):
            noise = np.random.normal(0, noise_level, X.shape)
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


