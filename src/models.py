"""
Model definitions and training utilities for peptide absorbance prediction.

Provides model creation functions, training pipelines, evaluation metrics,
and a high-level predictor class for production use.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    StackingRegressor,
)
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
    StratifiedKFold,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    f1_score,
)
from typing import Dict, List, Optional, Tuple, Union

from .feature_extraction import extract_all_features, extract_features_dataframe


# =============================================================================
# Regression Model Definitions
# =============================================================================

def create_rf_regressor(**kwargs) -> RandomForestRegressor:
    """Create a Random Forest regressor with sensible defaults."""
    defaults = dict(n_estimators=200, random_state=42)
    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)


def create_gbr_regressor(**kwargs) -> GradientBoostingRegressor:
    """Create a Gradient Boosting regressor with sensible defaults."""
    defaults = dict(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        min_samples_split=8,
        min_samples_leaf=5,
        random_state=42,
    )
    defaults.update(kwargs)
    return GradientBoostingRegressor(**defaults)


def create_stacking_regressor(
    best_rf: Optional[RandomForestRegressor] = None,
) -> StackingRegressor:
    """Create a stacking ensemble of Ridge, RF, and GBR with Ridge meta-learner."""
    rf = best_rf if best_rf is not None else create_rf_regressor()
    base_models = [
        ('ridge', Ridge()),
        ('rf', rf),
        ('gbm', create_gbr_regressor()),
    ]
    return StackingRegressor(
        estimators=base_models,
        final_estimator=RidgeCV(),
        cv=5,
    )


# =============================================================================
# Hyperparameter Tuning
# =============================================================================

RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
}


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = 'r2',
) -> GridSearchCV:
    """Run GridSearchCV to tune a Random Forest regressor.

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.
        param_grid: Parameter grid dict. Defaults to RF_PARAM_GRID.
        cv: Number of cross-validation folds.
        scoring: Scoring metric.

    Returns:
        Fitted GridSearchCV object.
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID

    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary with R², RMSE, and MAE.
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
    }


def compare_models(
    models: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Train and evaluate multiple models, returning a comparison table.

    Args:
        models: Dict mapping model name to sklearn estimator.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.

    Returns:
        DataFrame with columns Model, R2, RMSE, MAE.
    """
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
        metrics['Model'] = name
        results.append(metrics)
    return pd.DataFrame(results)[['Model', 'r2', 'rmse', 'mae']]


# =============================================================================
# High-Level Predictor Class
# =============================================================================

class PeptideAbsorbancePredictor:
    """End-to-end predictor for peptide absorbance from amino acid sequences.

    Wraps feature extraction, preprocessing, and a Gradient Boosting
    regression model into a single interface for training, prediction,
    and model persistence.

    Example:
        predictor = PeptideAbsorbancePredictor()
        metrics = predictor.train('data/peptide_csv1_last20.csv')
        results = predictor.predict(['SYENSHSQAINVDRT', 'DEHRNNQSSSTAIVY'])
        predictor.save('model.joblib')
    """

    def __init__(self, model_path: Optional[str] = None):
        self.feature_columns = None

        if model_path:
            state = joblib.load(model_path)
            self.model = state['model']
            self.feature_columns = state.get('feature_columns')
        else:
            self.model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler()),
                ('regressor', create_gbr_regressor()),
            ])

    def train(
        self,
        data_path: str,
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """Train the model on a CSV dataset.

        Expects a CSV with 'Sequence' and 'Absorbance' columns
        (or absorbance time-point columns).

        Args:
            data_path: Path to the training CSV file.
            test_size: Fraction of data to hold out for validation.

        Returns:
            Dictionary of training metrics.
        """
        data = pd.read_csv(data_path)

        features_list = []
        valid_indices = []
        for idx, row in data.iterrows():
            extra = {col: row.get(col) for col in data.columns
                     if col not in ['Number', 'Sequence', 'Notes', 'Absorbance']}
            feats = extract_all_features(row['Sequence'], additional_data=extra)
            if feats is not None:
                features_list.append(feats)
                valid_indices.append(idx)

        X = pd.DataFrame(features_list)
        y = data.loc[valid_indices, 'Absorbance'].values if 'Absorbance' in data.columns else None

        if y is None:
            raise ValueError("CSV must contain an 'Absorbance' column for training")

        self.feature_columns = X.columns

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42,
        )
        self.model.fit(X_train, y_train)

        y_pred_val = self.model.predict(X_val)
        metrics = evaluate_regression(y_val, y_pred_val)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()

        if hasattr(self.model[-1], 'feature_importances_'):
            imp = dict(zip(self.feature_columns, self.model[-1].feature_importances_))
            metrics['top_features'] = dict(
                sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
            )

        return metrics

    def predict(
        self,
        sequences: Union[str, List[str]],
    ) -> Dict[str, float]:
        """Predict absorbance for one or more peptide sequences.

        Args:
            sequences: A single sequence string or list of sequences.

        Returns:
            Dictionary mapping each sequence to its predicted absorbance.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        results = {}
        features_list = []
        valid_seqs = []

        for seq in sequences:
            feats = extract_all_features(seq)
            if feats is not None:
                features_list.append(feats)
                valid_seqs.append(seq)
            else:
                results[seq] = None

        if features_list:
            X = pd.DataFrame(features_list)
            if self.feature_columns is not None:
                X = X.reindex(columns=self.feature_columns, fill_value=0)
            preds = self.model.predict(X)
            for seq, pred in zip(valid_seqs, preds):
                results[seq] = float(pred)

        return results

    def save(self, path: str) -> None:
        """Save the trained model and metadata to disk.

        Args:
            path: File path for the saved model (.joblib).
        """
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
        }, path)

    @classmethod
    def load(cls, path: str) -> 'PeptideAbsorbancePredictor':
        """Load a saved predictor from disk.

        Args:
            path: Path to the .joblib model file.

        Returns:
            Loaded PeptideAbsorbancePredictor instance.
        """
        return cls(model_path=path)
