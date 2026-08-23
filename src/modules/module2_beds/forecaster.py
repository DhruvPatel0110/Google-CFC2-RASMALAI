import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List, Optional
from src import config

class BedOccupancyForecaster:
    """Multi-step XGBoost forecaster for daily bed occupancy and patient admissions."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or str(config.MODELS_DIR / "bed_occupancy_xgb.joblib")
        self.meta_path = str(config.MODELS_DIR / "bed_occupancy_meta.joblib")
        self.model: Optional[MultiOutputRegressor] = None
        self.feature_cols: List[str] = []
        self.target_cols: List[str] = [f"target_occ_day_{i}" for i in range(1, 8)] + [f"target_adm_day_{i}" for i in range(1, 8)]
        self.metrics: Dict[str, float] = {}

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identifies feature columns excluding identifiers and future target columns."""
        exclude = ['date', 'phc_id', 'district'] + self.target_cols
        
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        return feature_cols

    def train(self, feature_df: pd.DataFrame, test_days: int = 30) -> Dict[str, float]:
        """Trains the multi-target XGBoost model using a temporal train/test split."""
        self.feature_cols = self.get_feature_columns(feature_df)
        
        # Temporal Split
        cutoff_date = feature_df['date'].max() - pd.Timedelta(days=test_days)
        train_mask = feature_df['date'] <= cutoff_date
        test_mask = feature_df['date'] > cutoff_date
        
        X_train = feature_df.loc[train_mask, self.feature_cols]
        y_train = feature_df.loc[train_mask, self.target_cols]
        
        X_test = feature_df.loc[test_mask, self.feature_cols]
        y_test = feature_df.loc[test_mask, self.target_cols]
        
        base_xgb = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        
        self.model = MultiOutputRegressor(base_xgb)
        self.model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = self.model.predict(X_test)
        y_test_np = y_test.values
        
        occ_indices = [i for i, c in enumerate(self.target_cols) if 'occ' in c]
        adm_indices = [i for i, c in enumerate(self.target_cols) if 'adm' in c]
        
        occ_rmse = np.sqrt(mean_squared_error(y_test_np[:, occ_indices], y_pred[:, occ_indices]))
        occ_mae = mean_absolute_error(y_test_np[:, occ_indices], y_pred[:, occ_indices])
        occ_r2 = r2_score(y_test_np[:, occ_indices], y_pred[:, occ_indices])
        
        adm_rmse = np.sqrt(mean_squared_error(y_test_np[:, adm_indices], y_pred[:, adm_indices]))
        adm_r2 = r2_score(y_test_np[:, adm_indices], y_pred[:, adm_indices])
        
        self.metrics = {
            "occupancy_r2": float(occ_r2),
            "occupancy_rmse": float(occ_rmse),
            "occupancy_mae": float(occ_mae),
            "admissions_r2": float(adm_r2),
            "admissions_rmse": float(adm_rmse),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test))
        }
        
        self.save()
        return self.metrics

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Generates 7-day occupancy and admission projections."""
        if self.model is None:
            self.load()
            
        X = feature_df[self.feature_cols]
        preds = self.model.predict(X)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        
        pred_df = pd.DataFrame(preds, columns=self.target_cols, index=feature_df.index)
        
        result = pd.concat([
            feature_df[['date', 'phc_id', 'total_beds']],
            pred_df
        ], axis=1)
        
        return result

    def get_feature_importances(self) -> pd.DataFrame:
        """Extracts averaged feature importances."""
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
        """Saves model and metadata to disk."""
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
