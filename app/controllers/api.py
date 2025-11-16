from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib
import os
from pathlib import Path
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    xgb = None

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False
    cb = None

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.lightgbm
    import mlflow.xgboost
    import mlflow.catboost
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False
    mlflow = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False
    optuna = None

app = FastAPI(title="DS Control Panel API", version="1.0.0")

# Enable CORS for notebook access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MLFLOW_DIR = BASE_DIR / "mlruns"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
MLFLOW_DIR.mkdir(exist_ok=True)

# Setup MLflow
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(str(MLFLOW_DIR))
    mlflow.set_experiment("ds-control-panel-v2")

# Request/Response models
class PredictionRequest(BaseModel):
    features: list

class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float

class TrainingResponse(BaseModel):
    message: str
    metrics: list[ModelMetrics]
    best_model: str

class OptimizationResponse(BaseModel):
    message: str
    model_name: str
    best_params: dict
    best_metrics: ModelMetrics
    n_trials: int

# Global variables
trained_models = {}
best_model_name = None
model_metrics = {}
X_test_global = None
y_test_global = None
feature_names = None

def load_or_generate_data():
    """Load existing dataset or generate a synthetic financial dataset"""
    data_path = DATA_DIR / "financial_data.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded existing dataset: {len(df)} rows")
        return df
    
    # Generate synthetic financial classification dataset
    print("Generating synthetic financial dataset...")
    np.random.seed(42)
    n_samples = 10000
    
    # Features: credit score, income, debt, age, loan amount, etc.
    data = {
        'credit_score': np.random.normal(650, 100, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'debt': np.random.normal(20000, 10000, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'loan_amount': np.random.normal(50000, 30000, n_samples),
        'employment_years': np.random.normal(5, 3, n_samples),
        'savings': np.random.normal(10000, 15000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target: 1 if good credit risk, 0 if bad
    # Based on a combination of features
    risk_score = (
        (df['credit_score'] - 500) / 200 +
        (df['income'] - 30000) / 50000 -
        (df['debt'] - 15000) / 30000 +
        (df['savings'] - 5000) / 20000
    )
    df['target'] = (risk_score > 0.5).astype(int)
    
    # Add some noise
    noise = np.random.random(n_samples) < 0.1
    df.loc[noise, 'target'] = 1 - df.loc[noise, 'target']
    
    df.to_csv(data_path, index=False)
    print(f"Generated and saved dataset: {len(df)} rows")
    return df

def prepare_data(df):
    """Prepare data for training"""
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_lightgbm(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=5):
    """Train LightGBM model"""
    if not LIGHTGBM_AVAILABLE:
        return None, None
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return model, metrics

def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=5):
    """Train XGBoost model"""
    if not XGBOOST_AVAILABLE:
        return None, None
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return model, metrics

def train_extra_trees(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10):
    """Train Extra Trees model"""
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return model, metrics

def train_catboost(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=5):
    """Train CatBoost model"""
    if not CATBOOST_AVAILABLE:
        return None, None
    
    model = cb.CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=learning_rate,
        depth=max_depth,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return model, metrics

def optimize_lightgbm(X_train, y_train, X_test, y_test, n_trials=20):
    """Optimize LightGBM hyperparameters using Optuna"""
    if not LIGHTGBM_AVAILABLE or not OPTUNA_AVAILABLE:
        return None, None, None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_model, best_metrics = train_lightgbm(
        X_train, y_train, X_test, y_test,
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth']
    )
    
    return best_model, best_metrics, best_params

def optimize_xgboost(X_train, y_train, X_test, y_test, n_trials=20):
    """Optimize XGBoost hyperparameters using Optuna"""
    if not XGBOOST_AVAILABLE or not OPTUNA_AVAILABLE:
        return None, None, None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_model, best_metrics = train_xgboost(
        X_train, y_train, X_test, y_test,
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth']
    )
    
    return best_model, best_metrics, best_params

def optimize_random_forest(X_train, y_train, X_test, y_test, n_trials=20):
    """Optimize Random Forest hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE:
        return None, None, None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_model, best_metrics = train_random_forest(
        X_train, y_train, X_test, y_test,
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth']
    )
    
    return best_model, best_metrics, best_params

def optimize_extra_trees(X_train, y_train, X_test, y_test, n_trials=20):
    """Optimize Extra Trees hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE:
        return None, None, None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = ExtraTreesClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_model, best_metrics = train_extra_trees(
        X_train, y_train, X_test, y_test,
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth']
    )
    
    return best_model, best_metrics, best_params

def optimize_catboost(X_train, y_train, X_test, y_test, n_trials=20):
    """Optimize CatBoost hyperparameters using Optuna"""
    if not CATBOOST_AVAILABLE or not OPTUNA_AVAILABLE:
        return None, None, None
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0, 1),
            'random_state': 42,
            'verbose': False
        }
        
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_model, best_metrics = train_catboost(
        X_train, y_train, X_test, y_test,
        n_estimators=best_params['iterations'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['depth']
    )
    
    return best_model, best_metrics, best_params

@app.get("/")
def root():
    endpoints = {
        "train": "/train",
        "optimize": "/optimize",
        "predict": "/predict",
        "metrics": "/metrics",
        "models": "/models"
    }
    if MLFLOW_AVAILABLE:
        endpoints.update({
            "mlflow_experiments": "/mlflow/experiments",
            "mlflow_runs": "/mlflow/runs",
            "mlflow_ui": "http://localhost:5000 (run: mlflow ui --backend-store-uri ./mlruns)"
        })
    return {
        "message": "DS Control Panel API",
        "endpoints": endpoints,
        "mlflow_available": MLFLOW_AVAILABLE,
        "optuna_available": OPTUNA_AVAILABLE
    }

@app.post("/train", response_model=TrainingResponse)
def train_models(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    include_catboost: bool = False
):
    """Train all models with specified parameters"""
    global trained_models, best_model_name, model_metrics, X_test_global, y_test_global, feature_names
    
    # Load or generate data
    df = load_or_generate_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Store test data globally for visualizations
    X_test_global = X_test
    y_test_global = y_test
    feature_names = list(X_test.columns)
    
    trained_models = {}
    model_metrics = {}
    all_metrics = []
    
    # Start MLflow run if available
    if MLFLOW_AVAILABLE:
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("include_catboost", include_catboost)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Train LightGBM
            if LIGHTGBM_AVAILABLE:
                print("Training LightGBM...")
                lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test, n_estimators, learning_rate, max_depth)
                if lgb_model is not None:
                    trained_models['lightgbm'] = lgb_model
                    model_metrics['lightgbm'] = lgb_metrics
                    all_metrics.append(ModelMetrics(model_name='lightgbm', **lgb_metrics))
                    # Log to MLflow
                    mlflow.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
                    mlflow.lightgbm.log_model(lgb_model, "lightgbm_model")
            else:
                print("LightGBM not available (missing libomp dependency)")
            
            # Train XGBoost
            if XGBOOST_AVAILABLE:
                print("Training XGBoost...")
                xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, n_estimators, learning_rate, max_depth)
                if xgb_model is not None:
                    trained_models['xgboost'] = xgb_model
                    model_metrics['xgboost'] = xgb_metrics
                    all_metrics.append(ModelMetrics(model_name='xgboost', **xgb_metrics))
                    # Log to MLflow
                    mlflow.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
                    mlflow.xgboost.log_model(xgb_model, "xgboost_model")
            else:
                print("XGBoost not available")
            
            # Train Random Forest
            print("Training Random Forest...")
            rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, n_estimators, max_depth)
            trained_models['random_forest'] = rf_model
            model_metrics['random_forest'] = rf_metrics
            all_metrics.append(ModelMetrics(model_name='random_forest', **rf_metrics))
            # Log to MLflow
            mlflow.log_metrics({f"random_forest_{k}": v for k, v in rf_metrics.items()})
            mlflow.sklearn.log_model(rf_model, "random_forest_model")
            
            # Train Extra Trees
            print("Training Extra Trees...")
            et_model, et_metrics = train_extra_trees(X_train, y_train, X_test, y_test, n_estimators, max_depth)
            trained_models['extra_trees'] = et_model
            model_metrics['extra_trees'] = et_metrics
            all_metrics.append(ModelMetrics(model_name='extra_trees', **et_metrics))
            # Log to MLflow
            mlflow.log_metrics({f"extra_trees_{k}": v for k, v in et_metrics.items()})
            mlflow.sklearn.log_model(et_model, "extra_trees_model")
            
            # Train CatBoost (optional)
            if include_catboost:
                print("Training CatBoost...")
                cb_model, cb_metrics = train_catboost(X_train, y_train, X_test, y_test, n_estimators, learning_rate, max_depth)
                if cb_model is not None:
                    trained_models['catboost'] = cb_model
                    model_metrics['catboost'] = cb_metrics
                    all_metrics.append(ModelMetrics(model_name='catboost', **cb_metrics))
                    # Log to MLflow
                    mlflow.log_metrics({f"catboost_{k}": v for k, v in cb_metrics.items()})
                    mlflow.catboost.log_model(cb_model, "catboost_model")
            
            # Find best model based on AUC
            best_model_name = max(model_metrics.items(), key=lambda x: x[1]['auc'])[0]
            best_auc = model_metrics[best_model_name]['auc']
            
            # Log best model info
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_auc", best_auc)
            
            # Save best model
            best_model_path = MODELS_DIR / f"{best_model_name}_best.pkl"
            joblib.dump(trained_models[best_model_name], best_model_path)
            
            # Log best model to MLflow
            if best_model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
                mlflow.lightgbm.log_model(trained_models[best_model_name], "best_model")
            elif best_model_name == 'xgboost' and XGBOOST_AVAILABLE:
                mlflow.xgboost.log_model(trained_models[best_model_name], "best_model")
            elif best_model_name == 'catboost' and include_catboost and CATBOOST_AVAILABLE:
                mlflow.catboost.log_model(trained_models[best_model_name], "best_model")
            else:
                mlflow.sklearn.log_model(trained_models[best_model_name], "best_model")
    else:
        # Fallback to original behavior if MLflow not available
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            print("Training LightGBM...")
            lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test, n_estimators, learning_rate, max_depth)
            if lgb_model is not None:
                trained_models['lightgbm'] = lgb_model
                model_metrics['lightgbm'] = lgb_metrics
                all_metrics.append(ModelMetrics(model_name='lightgbm', **lgb_metrics))
        else:
            print("LightGBM not available (missing libomp dependency)")
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, n_estimators, learning_rate, max_depth)
            if xgb_model is not None:
                trained_models['xgboost'] = xgb_model
                model_metrics['xgboost'] = xgb_metrics
                all_metrics.append(ModelMetrics(model_name='xgboost', **xgb_metrics))
        else:
            print("XGBoost not available")
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, n_estimators, max_depth)
        trained_models['random_forest'] = rf_model
        model_metrics['random_forest'] = rf_metrics
        all_metrics.append(ModelMetrics(model_name='random_forest', **rf_metrics))
        
        # Train Extra Trees
        print("Training Extra Trees...")
        et_model, et_metrics = train_extra_trees(X_train, y_train, X_test, y_test, n_estimators, max_depth)
        trained_models['extra_trees'] = et_model
        model_metrics['extra_trees'] = et_metrics
        all_metrics.append(ModelMetrics(model_name='extra_trees', **et_metrics))
        
        # Train CatBoost (optional)
        if include_catboost:
            print("Training CatBoost...")
            cb_model, cb_metrics = train_catboost(X_train, y_train, X_test, y_test, n_estimators, learning_rate, max_depth)
            if cb_model is not None:
                trained_models['catboost'] = cb_model
                model_metrics['catboost'] = cb_metrics
                all_metrics.append(ModelMetrics(model_name='catboost', **cb_metrics))
        
        # Find best model based on AUC
        best_model_name = max(model_metrics.items(), key=lambda x: x[1]['auc'])[0]
        
        # Save best model
        best_model_path = MODELS_DIR / f"{best_model_name}_best.pkl"
        joblib.dump(trained_models[best_model_name], best_model_path)
    
    return TrainingResponse(
        message="Models trained successfully",
        metrics=all_metrics,
        best_model=best_model_name
    )

@app.post("/optimize", response_model=OptimizationResponse)
def optimize_hyperparameters(
    model_name: str,
    n_trials: int = 20
):
    """Optimize hyperparameters for a specific model using Optuna"""
    global trained_models, model_metrics, X_test_global, y_test_global, feature_names
    
    if not OPTUNA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Optuna is not available. Please install optuna.")
    
    # Load or generate data
    df = load_or_generate_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Store test data globally for visualizations
    X_test_global = X_test
    y_test_global = y_test
    feature_names = list(X_test.columns)
    
    # Map model names
    model_name_lower = model_name.lower().replace('_', '').replace('-', '')
    optimization_functions = {
        'lightgbm': optimize_lightgbm,
        'xgboost': optimize_xgboost,
        'randomforest': optimize_random_forest,
        'extratrees': optimize_extra_trees,
        'catboost': optimize_catboost
    }
    
    if model_name_lower not in optimization_functions:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not supported. Available models: {list(optimization_functions.keys())}"
        )
    
    optimize_func = optimization_functions[model_name_lower]
    
    # Start MLflow run if available
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"optuna_{model_name}"):
            # Log optimization parameters
            mlflow.log_param("optimization_type", "optuna")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Run optimization
            print(f"Optimizing {model_name} with {n_trials} trials...")
            best_model, best_metrics, best_params = optimize_func(X_train, y_train, X_test, y_test, n_trials)
            
            if best_model is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Optimization failed for {model_name}. Model may not be available."
                )
            
            # Store optimized model
            trained_models[model_name_lower] = best_model
            model_metrics[model_name_lower] = best_metrics
            
            # Log all best parameters
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            # Log metrics
            mlflow.log_metrics({f"{model_name_lower}_{k}": v for k, v in best_metrics.items()})
            
            # Log model
            if model_name_lower == 'lightgbm' and LIGHTGBM_AVAILABLE:
                mlflow.lightgbm.log_model(best_model, "optimized_model")
            elif model_name_lower == 'xgboost' and XGBOOST_AVAILABLE:
                mlflow.xgboost.log_model(best_model, "optimized_model")
            elif model_name_lower == 'catboost' and CATBOOST_AVAILABLE:
                mlflow.catboost.log_model(best_model, "optimized_model")
            else:
                mlflow.sklearn.log_model(best_model, "optimized_model")
            
            # Save model
            model_path = MODELS_DIR / f"{model_name_lower}_optimized.pkl"
            joblib.dump(best_model, model_path)
            
            return OptimizationResponse(
                message=f"Hyperparameter optimization completed for {model_name}",
                model_name=model_name,
                best_params=best_params,
                best_metrics=ModelMetrics(model_name=model_name, **best_metrics),
                n_trials=n_trials
            )
    else:
        # Fallback without MLflow
        print(f"Optimizing {model_name} with {n_trials} trials...")
        best_model, best_metrics, best_params = optimize_func(X_train, y_train, X_test, y_test, n_trials)
        
        if best_model is None:
            raise HTTPException(
                status_code=500,
                detail=f"Optimization failed for {model_name}. Model may not be available."
            )
        
        # Store optimized model
        trained_models[model_name_lower] = best_model
        model_metrics[model_name_lower] = best_metrics
        
        # Save model
        model_path = MODELS_DIR / f"{model_name_lower}_optimized.pkl"
        joblib.dump(best_model, model_path)
        
        return OptimizationResponse(
            message=f"Hyperparameter optimization completed for {model_name}",
            model_name=model_name,
            best_params=best_params,
            best_metrics=ModelMetrics(model_name=model_name, **best_metrics),
            n_trials=n_trials
        )

@app.get("/metrics")
def get_metrics():
    """Get metrics for all trained models"""
    if not model_metrics:
        raise HTTPException(status_code=404, detail="No models trained yet. Call /train first.")
    return {
        "metrics": model_metrics,
        "best_model": best_model_name
    }

@app.get("/models")
def list_models():
    """List available models"""
    return {
        "trained_models": list(trained_models.keys()),
        "best_model": best_model_name,
        "saved_models": [f.name for f in MODELS_DIR.glob("*.pkl")]
    }

@app.get("/mlflow/experiments")
def get_mlflow_experiments():
    """Get list of MLflow experiments"""
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow is not available. Please install mlflow.")
    
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = client.search_experiments()
        
        experiments_list = []
        for exp in experiments:
            experiments_list.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "tags": exp.tags
            })
        
        return {
            "experiments": experiments_list,
            "count": len(experiments_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching experiments: {str(e)}")

@app.get("/mlflow/runs")
def get_mlflow_runs(experiment_id: str = None, limit: int = 10):
    """Get list of MLflow runs"""
    if not MLFLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="MLflow is not available. Please install mlflow.")
    
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        if experiment_id:
            runs = client.search_runs(experiment_ids=[experiment_id], max_results=limit)
        else:
            # Get default experiment
            exp = mlflow.get_experiment_by_name("ds-control-panel")
            if exp:
                runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=limit)
            else:
                return {"runs": [], "count": 0}
        
        runs_list = []
        for run in runs:
            runs_list.append({
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            })
        
        return {
            "runs": runs_list,
            "count": len(runs_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching runs: {str(e)}")

@app.get("/predict")
def predict_get():
    """Get endpoint showing usage instructions for /predict"""
    return {
        "message": "Use POST method to make predictions",
        "usage": {
            "method": "POST",
            "url": "/predict",
            "body": {
                "features": "[credit_score, income, debt, age, loan_amount, employment_years, savings]"
            },
            "optional_query_param": "model_name (e.g., ?model_name=random_forest)",
            "example": {
                "curl": 'curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d \'{"features": [750, 80000, 15000, 35, 60000, 8, 25000]}\''
            }
        }
    }

@app.post("/predict")
def predict(request: PredictionRequest, model_name: str = None):
    """Make prediction using specified model or best model"""
    if not trained_models:
        raise HTTPException(status_code=404, detail="No models trained yet. Call /train first.")
    
    # Use specified model or best model
    model_key = model_name if model_name and model_name in trained_models else best_model_name
    model = trained_models[model_key]
    
    # Convert features to numpy array
    features = np.array(request.features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    
    return {
        "model_used": model_key,
        "prediction": int(prediction),
        "probability": {
            "class_0": float(prediction_proba[0]),
            "class_1": float(prediction_proba[1])
        }
    }

@app.get("/visualizations/confusion-matrix/{model_name}")
def get_confusion_matrix(model_name: str):
    """Get confusion matrix data for a specific model"""
    if not trained_models:
        raise HTTPException(status_code=404, detail="No models trained yet. Call /train first.")
    
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    
    if X_test_global is None or y_test_global is None:
        raise HTTPException(status_code=404, detail="Test data not available. Please train models first.")
    
    model = trained_models[model_name]
    y_pred = model.predict(X_test_global)
    cm = confusion_matrix(y_test_global, y_pred)
    
    return {
        "model_name": model_name,
        "confusion_matrix": cm.tolist(),
        "labels": ["Bad Credit", "Good Credit"]
    }

@app.get("/visualizations/roc-curve/{model_name}")
def get_roc_curve(model_name: str):
    """Get ROC curve data for a specific model"""
    if not trained_models:
        raise HTTPException(status_code=404, detail="No models trained yet. Call /train first.")
    
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    
    if X_test_global is None or y_test_global is None:
        raise HTTPException(status_code=404, detail="Test data not available. Please train models first.")
    
    model = trained_models[model_name]
    y_pred_proba = model.predict_proba(X_test_global)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_global, y_pred_proba)
    auc = roc_auc_score(y_test_global, y_pred_proba)
    
    return {
        "model_name": model_name,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc)
    }

@app.get("/visualizations/feature-importance/{model_name}")
def get_feature_importance(model_name: str):
    """Get feature importance for a specific model"""
    if not trained_models:
        raise HTTPException(status_code=404, detail="No models trained yet. Call /train first.")
    
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    
    if feature_names is None:
        raise HTTPException(status_code=404, detail="Feature names not available. Please train models first.")
    
    model = trained_models[model_name]
    
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    else:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' does not support feature importance.")
    
    # Create sorted list of features and importances
    feature_importance = list(zip(feature_names, importances.tolist()))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "model_name": model_name,
        "features": [f[0] for f in feature_importance],
        "importances": [f[1] for f in feature_importance]
    }

@app.get("/visualizations/model-comparison")
def get_model_comparison():
    """Get comparison data for all trained models"""
    if not model_metrics:
        raise HTTPException(status_code=404, detail="No models trained yet. Call /train first.")
    
    comparison_data = []
    for model_name, metrics in model_metrics.items():
        comparison_data.append({
            "model_name": model_name,
            "accuracy": metrics.get('accuracy', 0),
            "precision": metrics.get('precision', 0),
            "recall": metrics.get('recall', 0),
            "f1_score": metrics.get('f1_score', 0),
            "auc": metrics.get('auc', 0)
        })
    
    return {
        "models": comparison_data,
        "best_model": best_model_name
    }

@app.get("/visualizations/all/{model_name}")
def get_all_visualizations(model_name: str):
    """Get all visualization data for a specific model in one call"""
    if not trained_models:
        raise HTTPException(status_code=404, detail="No models trained yet. Call /train first.")
    
    if model_name not in trained_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    
    if X_test_global is None or y_test_global is None:
        raise HTTPException(status_code=404, detail="Test data not available. Please train models first.")
    
    model = trained_models[model_name]
    
    # Get predictions
    y_pred = model.predict(X_test_global)
    y_pred_proba = model.predict_proba(X_test_global)[:, 1]
    
    # Confusion matrix
    cm = confusion_matrix(y_test_global, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_global, y_pred_proba)
    auc = roc_auc_score(y_test_global, y_pred_proba)
    
    # Clean infinity values from thresholds for JSON serialization
    import numpy as np
    thresholds_clean = [float(t) if np.isfinite(t) else None for t in thresholds]
    fpr_clean = [float(f) if np.isfinite(f) else 0.0 for f in fpr]
    tpr_clean = [float(t) if np.isfinite(t) else 1.0 for t in tpr]
    
    # Feature importance
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_.tolist()
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance().tolist()
    
    # Get metrics
    metrics = model_metrics.get(model_name, {})
    
    return {
        "model_name": model_name,
        "metrics": metrics,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": ["Bad Credit", "Good Credit"]
        },
        "roc_curve": {
            "fpr": fpr_clean,
            "tpr": tpr_clean,
            "thresholds": thresholds_clean,
            "auc": float(auc)
        },
        "feature_importance": {
            "features": feature_names if feature_names else [],
            "importances": importances if importances else []
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

