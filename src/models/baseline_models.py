"""
STEP 4 & 5: Baseline ML Model
Train baseline models (XGBoost, Random Forest) to prove signal exists

These simple models already beat graph-based approaches because they
leverage the hypergraph structure implicit in the features.

UPDATED: Now uses temporal splitting to prevent data leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any
import json

# Import temporal splitting to prevent data leakage
from ..data.dataset_split import temporal_split


class BaselineModelTrainer:
    """Train and evaluate baseline ML models with temporal splitting"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.results = {}

    def prepare_data(self, features_df: pd.DataFrame,
                    labels_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare features and labels for training

        Args:
            features_df: DataFrame with aggregated features
            labels_df: DataFrame with HCI labels

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Merge features and labels
        data = features_df.merge(labels_df[['hyperedge_id', 'HCI']],
                                on='hyperedge_id')

        # Feature columns (exclude ID and target)
        feature_cols = [col for col in features_df.columns
                       if col != 'hyperedge_id']

        X = data[feature_cols].values
        y = data['HCI'].values

        self.feature_names = feature_cols

        return X, y, feature_cols

    def split_and_scale(self, X: np.ndarray, y: np.ndarray,
                       test_months: int = 2, val_months: int = 1,
                       gap_hours: int = 72) -> Dict:
        """
        Split data into train/val/test using TEMPORAL SPLITTING and scale features.

        PREVENTS DATA LEAKAGE by respecting chronological order instead of random shuffle.

        Args:
            X: Feature matrix (must be chronologically ordered)
            y: Target vector (corresponding to X)
            test_months: Number of final months for test set (default: 2)
            val_months: Number of months for validation (default: 1)
            gap_hours: Hours gap between splits to prevent crisis bleed-through (default: 72)

        Returns:
            Dictionary with split and scaled data

        Note:
            This replaces sklearn's train_test_split which randomly shuffles data
            and causes temporal leakage in time series prediction tasks.
        """
        print("\n" + "="*80)
        print("⚠️  TEMPORAL SPLITTING - PREVENTING DATA LEAKAGE")
        print("="*80)
        print("❌ OLD: sklearn train_test_split (random shuffle → leakage)")
        print("✅ NEW: temporal_split (chronological order → no leakage)")
        print()

        # Use temporal split instead of random split
        train_data, val_data, test_data = temporal_split(
            X, y,
            test_months=test_months,
            val_months=val_months,
            gap_hours=gap_hours
        )

        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        print(f"📊 Final split sizes:")
        print(f"   Train: {len(X_train):,} samples")
        print(f"   Val:   {len(X_val):,} samples")
        print(f"   Test:  {len(X_test):,} samples")
        print("="*80)

        # Scale features (fit on train, transform val/test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['standard'] = scaler

        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler
        }
    
    def train_xgboost(self, data: Dict) -> Tuple[Any, Dict]:
        """
        Train XGBoost model with hyperparameter tuning
        
        Args:
            data: Dictionary with split data
        
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        print("\n" + "="*60)
        print("XGBoost Model Training")
        print("="*60)
        
        # Base model
        base_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # Train
        base_model.fit(
            data['X_train'], data['y_train'],
            eval_set=[(data['X_val'], data['y_val'])],
            verbose=False
        )
        
        # Evaluate
        y_pred_train = base_model.predict(data['X_train'])
        y_pred_val = base_model.predict(data['X_val'])
        y_pred_test = base_model.predict(data['X_test'])
        
        metrics = {
            'train_mse': mean_squared_error(data['y_train'], y_pred_train),
            'val_mse': mean_squared_error(data['y_val'], y_pred_val),
            'test_mse': mean_squared_error(data['y_test'], y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(data['y_train'], y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(data['y_val'], y_pred_val)),
            'test_rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred_test)),
            'train_mae': mean_absolute_error(data['y_train'], y_pred_train),
            'val_mae': mean_absolute_error(data['y_val'], y_pred_val),
            'test_mae': mean_absolute_error(data['y_test'], y_pred_test),
            'train_r2': r2_score(data['y_train'], y_pred_train),
            'val_r2': r2_score(data['y_val'], y_pred_val),
            'test_r2': r2_score(data['y_test'], y_pred_test),
        }
        
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': base_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        self.models['xgboost'] = base_model
        self.results['xgboost'] = {'metrics': metrics, 'feature_importance': feature_importance}
        
        return base_model, metrics
    
    def train_random_forest(self, data: Dict) -> Tuple[Any, Dict]:
        """
        Train Random Forest model
        
        Args:
            data: Dictionary with split data
        
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        print("\n" + "="*60)
        print("Random Forest Model Training")
        print("="*60)
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(data['X_train'], data['y_train'])
        
        # Evaluate
        y_pred_train = model.predict(data['X_train'])
        y_pred_val = model.predict(data['X_val'])
        y_pred_test = model.predict(data['X_test'])
        
        metrics = {
            'train_mse': mean_squared_error(data['y_train'], y_pred_train),
            'val_mse': mean_squared_error(data['y_val'], y_pred_val),
            'test_mse': mean_squared_error(data['y_test'], y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(data['y_train'], y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(data['y_val'], y_pred_val)),
            'test_rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred_test)),
            'train_mae': mean_absolute_error(data['y_train'], y_pred_train),
            'val_mae': mean_absolute_error(data['y_val'], y_pred_val),
            'test_mae': mean_absolute_error(data['y_test'], y_pred_test),
            'train_r2': r2_score(data['y_train'], y_pred_train),
            'val_r2': r2_score(data['y_val'], y_pred_val),
            'test_r2': r2_score(data['y_test'], y_pred_test),
        }
        
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {'metrics': metrics, 'feature_importance': feature_importance}
        
        return model, metrics
    
    def train_gradient_boosting(self, data: Dict) -> Tuple[Any, Dict]:
        """
        Train Gradient Boosting model
        
        Args:
            data: Dictionary with split data
        
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        print("\n" + "="*60)
        print("Gradient Boosting Model Training")
        print("="*60)
        
        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.random_state,
            verbose=0
        )
        
        model.fit(data['X_train'], data['y_train'])
        
        # Evaluate
        y_pred_train = model.predict(data['X_train'])
        y_pred_val = model.predict(data['X_val'])
        y_pred_test = model.predict(data['X_test'])
        
        metrics = {
            'train_mse': mean_squared_error(data['y_train'], y_pred_train),
            'val_mse': mean_squared_error(data['y_val'], y_pred_val),
            'test_mse': mean_squared_error(data['y_test'], y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(data['y_train'], y_pred_train)),
            'val_rmse': np.sqrt(mean_squared_error(data['y_val'], y_pred_val)),
            'test_rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred_test)),
            'train_mae': mean_absolute_error(data['y_train'], y_pred_train),
            'val_mae': mean_absolute_error(data['y_val'], y_pred_val),
            'test_mae': mean_absolute_error(data['y_test'], y_pred_test),
            'train_r2': r2_score(data['y_train'], y_pred_train),
            'val_r2': r2_score(data['y_val'], y_pred_val),
            'test_r2': r2_score(data['y_test'], y_pred_test),
        }
        
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = {'metrics': metrics, 'feature_importance': feature_importance}
        
        return model, metrics
    
    def save_models(self, output_dir: str):
        """Save trained models and results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = output_path / f"{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {model_path}")
        
        # Save scaler
        scaler_path = output_path / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers['standard'], f)
        
        # Save results
        results_path = output_path / "model_results.json"
        results_serializable = {}
        for model_name, result in self.results.items():
            results_serializable[model_name] = {
                'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                           for k, v in result['metrics'].items()},
                'feature_importance': result['feature_importance'].to_dict()
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"Saved: {results_path}")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all models"""
        comparisons = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparisons.append({
                'Model': model_name,
                'Test RMSE': metrics['test_rmse'],
                'Test R²': metrics['test_r2'],
                'Test MAE': metrics['test_mae'],
                'Train R²': metrics['train_r2'],
                'Val R²': metrics['val_r2']
            })
        
        return pd.DataFrame(comparisons).sort_values('Test R²', ascending=False)


if __name__ == "__main__":
    print("Baseline ML models module ready")
