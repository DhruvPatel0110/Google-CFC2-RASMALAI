import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List, Optional
from src import config

class MedicineDemandForecaster:
    """Multi-step XGBoost time-series regression forecaster for medicine consumption."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(config.MODELS_DIR / "medicine_demand_xgb.joblib")
        self.meta_path = str(config.MODELS_DIR / "medicine_demand_meta.joblib")
        self.model: Optional[MultiOutputRegressor] = None
        self.feature_cols: List[str] = []
        self.target_cols: List[str] = [f"target_day_{i}" for i in range(1, 8)] + ["target_7d_total"]
        self.metrics: Dict[str, float] = {}

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identifies feature columns excluding identifiers and future target columns."""
        exclude = [
            'date', 'phc_id', 'drug_id', 'district', 'qty_dispensed', 'patients_served_estimate',
            'opening_stock', 'closing_stock', 'received_qty', 'reorder_point', 'safety_stock'
        ] + [f"target_day_{i}" for i in range(1, 8)] + ["target_7d_total"]
        
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        return feature_cols

    def train(self, feature_df: pd.DataFrame, test_days: int = 30) -> Dict[str, float]:
        """Trains the multi-output XGBoost model using a temporal train/test split."""
        self.feature_cols = self.get_feature_columns(feature_df)
        
        # Temporal Split: last N days reserved for out-of-time evaluation
        cutoff_date = feature_df['date'].max() - pd.Timedelta(days=test_days)
        train_mask = feature_df['date'] <= cutoff_date
        test_mask = feature_df['date'] > cutoff_date
        
        X_train = feature_df.loc[train_mask, self.feature_cols]
        y_train = feature_df.loc[train_mask, self.target_cols]
        
        X_test = feature_df.loc[test_mask, self.feature_cols]
        y_test = feature_df.loc[test_mask, self.target_cols]
        
        base_xgb = xgb.XGBRegressor(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        
        self.model = MultiOutputRegressor(base_xgb)
        self.model.fit(X_train, y_train)
        
        # Evaluation on test set
        y_pred = self.model.predict(X_test)
        y_test_np = y_test.values
        
        rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
        mae = mean_absolute_error(y_test_np, y_pred)
        r2 = r2_score(y_test_np, y_pred)
        
        # Total 7-day consumption metrics
        tot_idx = self.target_cols.index("target_7d_total")
        tot_rmse = np.sqrt(mean_squared_error(y_test_np[:, tot_idx], y_pred[:, tot_idx]))
        tot_r2 = r2_score(y_test_np[:, tot_idx], y_pred[:, tot_idx])
        
        self.metrics = {
            "overall_rmse": float(rmse),
            "overall_mae": float(mae),
            "overall_r2": float(r2),
            "7d_total_rmse": float(tot_rmse),
            "7d_total_r2": float(tot_r2),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test))
        }
        
        # Save model and metadata
        self.save()
        return self.metrics

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Generates 7-day demand predictions for input feature rows."""
        if self.model is None:
            self.load()
            
        X = feature_df.reindex(columns=self.feature_cols, fill_value=0)
        preds = self.model.predict(X)
        
        # Ensure non-negative consumption forecasts
        preds = np.clip(preds, a_min=0.0, a_max=None)
        
        pred_df = pd.DataFrame(preds, columns=self.target_cols, index=feature_df.index)
        
        # Combine with identification columns
        result = pd.concat([
            feature_df[['date', 'phc_id', 'drug_id']],
            pred_df
        ], axis=1)
        
        return result

    def get_feature_importances(self) -> pd.DataFrame:
        """Extracts averaged feature importances across all multi-target estimators."""
        if self.model is None:
            self.load()
            
        importances = np.mean([
            est.feature_importances_ for est in self.model.estimators_
        ], axis=0)
        
        fi_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)
        
        return fi_df

    def save(self):
        """Persists trained model and metadata."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump({
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols,
            "metrics": self.metrics
        }, self.meta_path)

    def load(self):
        """Loads model and metadata from disk."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train first.")
        self.model = joblib.load(self.model_path)
        meta = joblib.load(self.meta_path)
        self.feature_cols = meta["feature_cols"]
        self.target_cols = meta["target_cols"]
        self.metrics = meta.get("metrics", {})
