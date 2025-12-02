from __future__ import annotations

import pathlib
from typing import Tuple, List, Optional
import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef


# In-house LDA implementation
class InHouseLDA:
    def __init__(self, reg_param: float = 1e-6):
        self.reg_param = float(reg_param)
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.cov_ = None
        self.inv_cov_ = None
        self.w_ = None
        self.b_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Weighted priors with boosting for minority class
        weighted_counts = np.array([sample_weight[y == cls].sum() for cls in self.classes_])
        # Apply extra boost to minority class (class 1): multiply by 2.5
        if len(weighted_counts) == 2:
            weighted_counts[1] *= 2.5
        self.priors_ = weighted_counts / weighted_counts.sum()
        
        self.means_ = np.zeros((n_classes, n_features), dtype=float)

        # Weighted pooled covariance
        S = np.zeros((n_features, n_features), dtype=float)
        for idx, cls in enumerate(self.classes_):
            mask = (y == cls)
            Xc = X[mask]
            wc = sample_weight[mask]
            self.means_[idx] = np.average(Xc, axis=0, weights=wc)
            Xc_centered = Xc - self.means_[idx]
            S += (Xc_centered.T * wc) @ Xc_centered
        
        total_weight = sample_weight.sum()
        self.cov_ = S / max(total_weight - n_classes, 1)
        self.cov_ += self.reg_param * np.eye(n_features)
        self.inv_cov_ = np.linalg.inv(self.cov_)

        # Linear discriminants
        self.w_ = (self.inv_cov_ @ self.means_.T).T
        quad = np.einsum('ij,ij->i', self.means_ @ self.inv_cov_, self.means_)
        self.b_ = -0.5 * quad + np.log(self.priors_)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.w_.T + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        m = np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores - m)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


# Глобальні змінні для моделі
MODEL: Optional[InHouseLDA] = None
POLY_FEATURES: Optional[PolynomialFeatures] = None
THRESHOLD: float = 0.5
FEATURE_COLUMNS: List[str] = []
MEAN_TRAIN: Optional[np.ndarray] = None
STD_TRAIN: Optional[np.ndarray] = None
VOLATILITY_THRESHOLD: float = 0.0


class PredictionRequest(BaseModel):
    """Запит для передбачення з конкретними значеннями цін"""
    price: float = Field(..., gt=0, description="Поточна ціна USD/UAH")
    open_price: float = Field(..., gt=0, description="Ціна відкриття")
    high_price: float = Field(..., gt=0, description="Максимальна ціна за день")
    low_price: float = Field(..., gt=0, description="Мінімальна ціна за день")
    price_yesterday: float = Field(None, gt=0, description="Ціна вчора (опціонально)")


class PredictionResponse(BaseModel):
    """Відповідь з результатом передбачення"""
    prediction: str = Field(..., description="Передбачення: 'high_volatility' або 'low_volatility'")
    probability_high_vol: float = Field(..., description="Ймовірність високої волатильності")
    confidence: str = Field(..., description="Рівень впевненості: 'high', 'medium', 'low'")
    features_used: dict = Field(..., description="Використані ознаки для передбачення")


def load_and_preprocess_data(csv_content: str) -> Tuple[pd.DataFrame, List[str], float]:
    """Завантажує та обробляє CSV дані для навчання моделі"""
    # Читаємо CSV з рядка
    df = pd.read_csv(io.StringIO(csv_content))
    
    # Очищаємо назви колонок
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    
    # Парсимо дату та сортуємо
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Конвертуємо числові колонки
    for col in ["Price", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                      .str.replace(",", "", regex=False)
                      .astype(float))
    
    # Feature engineering
    df["return_1d"] = df["Price"].pct_change()
    df["range_pct"] = (df["High"] - df["Low"]) / df["Open"].replace(0, np.nan)
    df["gap_open"] = df["Open"].pct_change()
    
    # Additional volatility features
    df["return_2d"] = df["Price"].pct_change(2)
    df["return_3d"] = df["Price"].pct_change(3)
    df["rolling_std_5"] = df["return_1d"].rolling(5, min_periods=3).std()
    df["rolling_std_10"] = df["return_1d"].rolling(10, min_periods=5).std()
    
    # ATR (Average True Range)
    df["tr"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            np.abs(df["High"] - df["Price"].shift(1)),
            np.abs(df["Low"] - df["Price"].shift(1))
        )
    )
    df["atr_5"] = df["tr"].rolling(5, min_periods=3).mean() / df["Price"]
    df["atr_10"] = df["tr"].rolling(10, min_periods=5).mean() / df["Price"]
    
    # Absolute gap and return
    df["abs_gap"] = np.abs(df["gap_open"])
    df["abs_return"] = np.abs(df["return_1d"])
    
    # Target: tomorrow's return for volatility label
    df["ret_tomorrow"] = df["Price"].shift(-1) / df["Price"] - 1.0
    
    # Time-aware split (80/20)
    n = len(df)
    split_idx = int(n * 0.8)
    
    # Compute volatility threshold on TRAIN only (no leakage)
    train_abs_ret = df.loc[:split_idx-1, "ret_tomorrow"].abs().dropna()
    pct = 0.75
    vol_threshold = float(np.quantile(train_abs_ret, pct)) if len(train_abs_ret) > 0 else float(train_abs_ret.mean())
    
    # Large-move label: 1 if |ret_{t+1}| >= threshold, else 0
    df["y_vol"] = (df["ret_tomorrow"].abs() >= vol_threshold).astype(int)
    
    # Feature columns
    feature_cols = [
        "return_1d", "range_pct", "gap_open",
        "return_2d", "return_3d",
        "rolling_std_5", "rolling_std_10",
        "atr_5", "atr_10",
        "abs_gap", "abs_return"
    ]
    
    # Drop NA
    use_cols = feature_cols + ["ret_tomorrow", "y_vol"]
    df = df.dropna(subset=use_cols).reset_index(drop=True)
    
    return df, feature_cols, vol_threshold


def train_model(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[InHouseLDA, PolynomialFeatures, float, np.ndarray, np.ndarray]:
    """Навчає модель на даних"""
    X = df[feature_cols].values
    y = df["y_vol"].values
    
    # Time-aware split (80/20)
    n = len(df)
    split_idx = int(n * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Standardization
    mu_tr = X_train.mean(axis=0)
    sig_tr = X_train.std(axis=0) + 1e-9
    X_train_z = (X_train - mu_tr) / sig_tr
    X_test_z = (X_test - mu_tr) / sig_tr
    
    # Polynomial features (interactions)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_z)
    X_test_poly = poly.transform(X_test_z)
    
    # Compute sample weights for class balancing
    sample_weights = compute_sample_weight('balanced', y_train)
    
    # Train In-house LDA with regularization
    model = InHouseLDA(reg_param=1e-3)
    model.fit(X_train_poly, y_train, sample_weight=sample_weights)
    
    # Find optimal threshold (maximize F1)
    y_proba = model.predict_proba(X_test_poly)[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_threshold = 0.5
    
    from sklearn.metrics import f1_score
    for t in thresholds:
        pred_t = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
    
    return model, poly, best_threshold, mu_tr, sig_tr


def prepare_features(price: float, open_price: float, high_price: float, low_price: float, 
                     price_yesterday: Optional[float] = None) -> np.ndarray:
    """Підготовлює ознаки для передбачення з поточних цін"""
    # Базові розрахунки
    if price_yesterday is not None and price_yesterday > 0:
        return_1d = (price - price_yesterday) / price_yesterday
        gap_open = (open_price - price_yesterday) / price_yesterday
    else:
        # Якщо немає вчорашньої ціни, використовуємо середні значення
        return_1d = 0.001
        gap_open = 0.0
    
    range_pct = (high_price - low_price) / open_price if open_price > 0 else 0.01
    
    # Approximate volatility features (using current day's range as proxy)
    return_2d = return_1d * 0.8  # approximation
    return_3d = return_1d * 0.9  # approximation
    rolling_std_5 = range_pct * 0.3  # approximation based on range
    rolling_std_10 = range_pct * 0.25
    
    # ATR approximation
    tr = high_price - low_price
    atr_5 = (tr / price) * 0.8
    atr_10 = (tr / price) * 0.7
    
    abs_gap = abs(gap_open)
    abs_return = abs(return_1d)
    
    return np.array([[return_1d, range_pct, gap_open,
                     return_2d, return_3d,
                     rolling_std_5, rolling_std_10,
                     atr_5, atr_10,
                     abs_gap, abs_return]])


app = FastAPI(
    title="USD/UAH Volatility Prediction API (Gaussian LDA)",
    description="API для передбачення високої волатильності курсу USD/UAH з використанням власної реалізації Linear Discriminant Analysis",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Ініціалізація при запуску - навчання моделі на даних"""
    global MODEL, POLY_FEATURES, THRESHOLD, FEATURE_COLUMNS, MEAN_TRAIN, STD_TRAIN, VOLATILITY_THRESHOLD
    
    try:
        # Шукаємо файл з даними
        possible_paths = [
            pathlib.Path(__file__).resolve().parent / "datasets" / "USD_UAH Historical Data.csv",
            pathlib.Path(__file__).resolve().parent.parent.parent / "datasets" / "USD_UAH Historical Data.csv",
        ]
        
        data_path = None
        for p in possible_paths:
            if p.exists():
                data_path = p
                break
        
        if data_path and data_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()
            
            df, feature_cols, vol_threshold = load_and_preprocess_data(csv_content)
            MODEL, POLY_FEATURES, THRESHOLD, MEAN_TRAIN, STD_TRAIN = train_model(df, feature_cols)
            FEATURE_COLUMNS = feature_cols
            VOLATILITY_THRESHOLD = vol_threshold
            
            print(f"✅ Модель навчена на {len(df)} записах")
            print(f"   Поріг волатильності: {vol_threshold:.5f}")
            print(f"   Оптимальний поріг класифікації: {THRESHOLD:.3f}")
        else:
            print("⚠️ Файл з даними не знайдено, модель буде навчена при завантаженні CSV")
            
    except Exception as e:
        print(f"❌ Помилка при ініціалізації: {e}")


@app.get("/health")
def health_check():
    """Перевірка стану API"""
    return {
        "status": "ok", 
        "model_loaded": MODEL is not None,
        "model_type": "InHouse Linear Discriminant Analysis",
        "volatility_threshold": VOLATILITY_THRESHOLD if MODEL is not None else None
    }


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Завантажує CSV файл та навчає модель"""
    global MODEL, POLY_FEATURES, THRESHOLD, FEATURE_COLUMNS, MEAN_TRAIN, STD_TRAIN, VOLATILITY_THRESHOLD
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл повинен бути CSV формату")
    
    try:
        # Читаємо вміст файлу
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Обробляємо дані та навчаємо модель
        df, feature_cols, vol_threshold = load_and_preprocess_data(csv_content)
        MODEL, POLY_FEATURES, THRESHOLD, MEAN_TRAIN, STD_TRAIN = train_model(df, feature_cols)
        FEATURE_COLUMNS = feature_cols
        VOLATILITY_THRESHOLD = vol_threshold
        
        return {
            "message": "CSV файл успішно завантажено та модель навчена",
            "records_count": len(df),
            "features": feature_cols,
            "volatility_threshold": vol_threshold,
            "classification_threshold": THRESHOLD,
            "model_type": "InHouse Linear Discriminant Analysis"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Помилка при обробці файлу: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict_volatility(request: PredictionRequest):
    """Передбачає чи буде завтра день з високою волатильністю"""
    if MODEL is None or POLY_FEATURES is None or MEAN_TRAIN is None or STD_TRAIN is None:
        raise HTTPException(status_code=400, detail="Модель не навчена. Спочатку завантажте CSV файл.")
    
    try:
        # Підготовлюємо ознаки
        features = prepare_features(
            request.price, 
            request.open_price, 
            request.high_price, 
            request.low_price,
            request.price_yesterday
        )
        
        # Нормалізуємо ознаки
        features_scaled = (features - MEAN_TRAIN) / STD_TRAIN
        
        # Застосовуємо polynomial features
        features_poly = POLY_FEATURES.transform(features_scaled)
        
        # Отримуємо ймовірності
        probabilities = MODEL.predict_proba(features_poly)[0]
        prob_high_vol = float(probabilities[1])
        
        # Визначаємо передбачення
        prediction = "high_volatility" if prob_high_vol >= THRESHOLD else "low_volatility"
        
        # Визначаємо рівень впевненості
        if prob_high_vol > 0.75 or prob_high_vol < 0.25:
            confidence = "high"
        elif prob_high_vol > 0.6 or prob_high_vol < 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            prediction=prediction,
            probability_high_vol=prob_high_vol,
            confidence=confidence,
            features_used={
                "price": request.price,
                "open": request.open_price,
                "high": request.high_price,
                "low": request.low_price,
                "price_yesterday": request.price_yesterday,
                "range": request.high_price - request.low_price,
                "range_pct": (request.high_price - request.low_price) / request.open_price
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка при передбаченні: {str(e)}")


@app.get("/")
def root():
    """Головна сторінка API"""
    return {
        "message": "USD/UAH Volatility Prediction API (Gaussian LDA)",
        "description": "API для передбачення високої волатильності курсу USD/UAH з використанням власної реалізації Linear Discriminant Analysis",
        "model": "In-house Linear Discriminant Analysis (LDA) з polynomial features",
        "endpoints": {
            "upload_csv": "POST /upload-csv - завантажити CSV файл та навчити модель",
            "predict": "POST /predict - передбачити волатильність на наступний день",
            "health": "GET /health - перевірити стан API",
            "docs": "GET /docs - Swagger документація"
        },
        "model_loaded": MODEL is not None,
        "volatility_threshold": VOLATILITY_THRESHOLD if MODEL is not None else None
    }

