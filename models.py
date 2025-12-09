"""
ML Models for trend following.
Uses LightGBM for gradient boosting classification and regression.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import json

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

from config import ModelConfig


class TrendClassifier:
    """
    Predicts whether a tradeable trend is starting.
    
    Output: probability of uptrend, downtrend, or no trend
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the trend classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels (-1, 0, 1)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
            
        Returns:
            Dictionary with training metrics
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")
        
        self.feature_names = list(X_train.columns)
        
        # Convert labels: -1, 0, 1 -> 0, 1, 2 for LightGBM
        y_train_adj = y_train + 1
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'min_child_samples': self.config.min_child_samples,
            'verbose': -1,
            'force_row_wise': True,
            'random_state': 42,  # For reproducibility
            'deterministic': True,  # For reproducibility
        }
        
        callbacks = []
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))
        
        eval_set = [(X_train, y_train_adj)]
        eval_names = ['train']
        
        if X_val is not None and y_val is not None:
            y_val_adj = y_val + 1
            eval_set.append((X_val, y_val_adj))
            eval_names.append('valid')
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=verbose))
        
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_train, y_train_adj,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train) - 1  # Convert back
        metrics = {
            'train_accuracy': (train_pred == y_train).mean(),
        }
        
        if X_val is not None:
            val_pred = self.model.predict(X_val) - 1
            metrics['val_accuracy'] = (val_pred == y_val).mean()
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict trend probabilities.
        
        Args:
            X: Features
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.model.predict_proba(X)
        pred = self.model.predict(X) - 1  # Convert back to -1, 0, 1
        
        return {
            'prediction': pred,
            'prob_down': proba[:, 0],    # Class 0 = -1 (downtrend)
            'prob_neutral': proba[:, 1],  # Class 1 = 0 (no trend)
            'prob_up': proba[:, 2],       # Class 2 = 1 (uptrend)
        }
    
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'config': self.config
            }, f)
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.feature_importance = data['feature_importance']
        self.config = data['config']


class EntryQualityModel:
    """
    Predicts the quality of a pullback entry.
    
    Output: probability of successful bounce, expected R:R
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.classifier = None  # Binary: will it bounce?
        self.regressor = None   # Continuous: expected R:R
        self.feature_names: List[str] = []
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_success: pd.Series,
        y_rr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_success_val: Optional[pd.Series] = None,
        y_rr_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train both classifier and regressor.
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")
        
        self.feature_names = list(X_train.columns)
        
        # Train classifier
        clf_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'num_leaves': self.config.num_leaves,
            'feature_fraction': self.config.feature_fraction,
            'verbose': -1,
        }
        
        self.classifier = lgb.LGBMClassifier(**clf_params)
        
        eval_set_clf = [(X_train, y_success)]
        if X_val is not None:
            eval_set_clf.append((X_val, y_success_val))
        
        callbacks = [lgb.log_evaluation(period=50)] if verbose else []
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=verbose))
        
        self.classifier.fit(
            X_train, y_success,
            eval_set=eval_set_clf,
            callbacks=callbacks
        )
        
        # Train regressor (on successful trades only for better R:R prediction)
        success_mask = y_success == 1
        if success_mask.sum() > 100:
            reg_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'n_estimators': self.config.n_estimators // 2,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'verbose': -1,
            }
            
            self.regressor = lgb.LGBMRegressor(**reg_params)
            self.regressor.fit(
                X_train[success_mask], 
                y_rr[success_mask]
            )
        
        # Metrics
        metrics = {
            'train_accuracy': (self.classifier.predict(X_train) == y_success).mean(),
            'train_precision': self._precision(self.classifier.predict(X_train), y_success),
        }
        
        if X_val is not None:
            val_pred = self.classifier.predict(X_val)
            metrics['val_accuracy'] = (val_pred == y_success_val).mean()
            metrics['val_precision'] = self._precision(val_pred, y_success_val)
        
        return metrics
    
    def _precision(self, pred, true):
        """Calculate precision"""
        pred_positive = pred == 1
        if pred_positive.sum() == 0:
            return 0.0
        return (pred[pred_positive] == true[pred_positive]).mean()
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict entry quality.
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet")
        
        result = {
            'bounce_prob': self.classifier.predict_proba(X)[:, 1],
            'bounce_pred': self.classifier.predict(X),
        }
        
        if self.regressor is not None:
            result['expected_rr'] = self.regressor.predict(X)
        
        return result
    
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'regressor': self.regressor,
                'feature_names': self.feature_names,
                'config': self.config
            }, f)
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.classifier = data['classifier']
        self.regressor = data['regressor']
        self.feature_names = data['feature_names']
        self.config = data['config']


class RegimeClassifier:
    """
    Classifies current market regime.
    
    Output: regime type (ranging, trending up, trending down, volatile)
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names: List[str] = []
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> Dict:
        """Train regime classifier"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for training")
        
        self.feature_names = list(X_train.columns)
        
        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'verbose': -1,
        }
        
        self.model = lgb.LGBMClassifier(**params)
        
        eval_set = [(X_train, y_train)]
        callbacks = [lgb.log_evaluation(period=50)] if verbose else []
        
        if X_val is not None:
            eval_set.append((X_val, y_val))
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=verbose))
        
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        
        metrics = {
            'train_accuracy': (self.model.predict(X_train) == y_train).mean()
        }
        
        if X_val is not None:
            metrics['val_accuracy'] = (self.model.predict(X_val) == y_val).mean()
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict regime"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        # Get the actual classes the model knows about
        classes = self.model.classes_
        n_samples = len(X)
        
        # Initialize probabilities for all possible regimes (0-3)
        prob_ranging = np.zeros(n_samples)
        prob_trend_up = np.zeros(n_samples)
        prob_trend_down = np.zeros(n_samples)
        prob_volatile = np.zeros(n_samples)
        
        # Map probabilities based on which classes the model learned
        for i, cls in enumerate(classes):
            if cls == 0:
                prob_ranging = proba[:, i]
            elif cls == 1:
                prob_trend_up = proba[:, i]
            elif cls == 2:
                prob_trend_down = proba[:, i]
            elif cls == 3:
                prob_volatile = proba[:, i]
        
        return {
            'regime': predictions,
            'prob_ranging': prob_ranging,
            'prob_trend_up': prob_trend_up,
            'prob_trend_down': prob_trend_down,
            'prob_volatile': prob_volatile,
        }
    
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'config': self.config
            }, f)
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.config = data['config']


class TrendFollowerModels:
    """
    Container for all trend follower models.
    Provides unified interface for training and prediction.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.trend_classifier = TrendClassifier(config)
        self.entry_model = EntryQualityModel(config)
        self.regime_classifier = RegimeClassifier(config)
        
    def save_all(self, model_dir: Path):
        """Save all models"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.trend_classifier.save(model_dir / 'trend_classifier.pkl')
        self.entry_model.save(model_dir / 'entry_model.pkl')
        self.regime_classifier.save(model_dir / 'regime_classifier.pkl')
        
        print(f"Models saved to {model_dir}")
    
    def load_all(self, model_dir: Path):
        """Load all models"""
        model_dir = Path(model_dir)
        
        self.trend_classifier.load(model_dir / 'trend_classifier.pkl')
        self.entry_model.load(model_dir / 'entry_model.pkl')
        self.regime_classifier.load(model_dir / 'regime_classifier.pkl')
        
        print(f"Models loaded from {model_dir}")


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    
    print("Models module loaded successfully")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    
    if LIGHTGBM_AVAILABLE:
        models = TrendFollowerModels(DEFAULT_CONFIG.model)
        print("Model container initialized")
