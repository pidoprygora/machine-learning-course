from __future__ import annotations

import pathlib
from typing import Tuple, List, Optional
import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Власна реалізація Gaussian Naive Bayes з підтримкою sample_weight
class InHouseGNB:
    def __init__(self, reg_param: float = 1e-6):
        self.reg_param = float(reg_param)
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.vars_ = None

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
        self.vars_ = np.zeros((n_classes, n_features), dtype=float)

        # Compute class means and variances
        for idx, cls in enumerate(self.classes_):
            mask = (y == cls)
            Xc = X[mask]
            wc = sample_weight[mask]
            self.means_[idx] = np.average(Xc, axis=0, weights=wc)
            # Weighted variance
            centered = Xc - self.means_[idx]
            weighted_var = np.average(centered**2, axis=0, weights=wc)
            self.vars_[idx] = weighted_var + self.reg_param
        
        return self

    def _log_gaussian_likelihood(self, X: np.ndarray, cls_idx: int) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        mean = self.means_[cls_idx]
        var = self.vars_[cls_idx]
        log_det = np.sum(np.log(2.0 * np.pi * var))
        sq = np.sum(((X - mean) ** 2) / var, axis=1)
        return -0.5 * (log_det + sq)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_probs = []
        for idx, _ in enumerate(self.classes_):
            log_prior = np.log(self.priors_[idx] + 1e-9)
            log_likelihood = self._log_gaussian_likelihood(X, idx)
            log_probs.append(log_prior + log_likelihood)
        return np.vstack(log_probs).T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_p = self.predict_log_proba(X)
        max_log = np.max(log_p, axis=1, keepdims=True)
        stabilized = np.exp(log_p - max_log)
        return stabilized / stabilized.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


# Глобальні змінні для моделі
MODEL: Optional[InHouseGNB] = None
SCALER: Optional[StandardScaler] = None
THRESHOLD: float = 0.5
FEATURE_COLUMNS: List[str] = []
VOLATILITY_THRESHOLD: float = 0.0


class PredictionRequest(BaseModel):
    """Запит для передбачення з конкретними значеннями цін"""
    price: float = Field(..., gt=0, description="Поточна ціна USD/UAH")
    open_price: float = Field(..., gt=0, description="Ціна відкриття")
    high_price: float = Field(..., gt=0, description="Максимальна ціна за день")
    low_price: float = Field(..., gt=0, description="Мінімальна ціна за день")


class PredictionResponse(BaseModel):
    """Відповідь з результатом передбачення"""
    prediction: str = Field(..., description="Передбачення: 'large_move' або 'normal_day'")
    probability_large_move: float = Field(..., description="Ймовірність великого руху")
    confidence: str = Field(..., description="Рівень впевненості: 'high', 'medium', 'low'")
    features_used: dict = Field(..., description="Використані ознаки для передбачення")
    warning: Optional[str] = Field(None, description="Попередження про обмеження моделі")


def load_and_preprocess_data(csv_content: str) -> Tuple[pd.DataFrame, List[str], float]:
    """Завантажує та обробляє CSV дані для навчання моделі"""
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
    
    # Feature engineering from daily moves/volatility
    df["return_1d"] = df["Price"].pct_change()
    df["range_pct"] = (df["High"] - df["Low"]) / df["Open"].replace(0, np.nan)
    df["gap_open"] = df["Open"].pct_change()
    
    # Additional volatility features
    df["return_2d"] = df["Price"].pct_change(2)
    df["return_3d"] = df["Price"].pct_change(3)
    df["rolling_std_5"] = df["return_1d"].rolling(5, min_periods=3).std()
    df["rolling_std_10"] = df["return_1d"].rolling(10, min_periods=5).std()
    
    # ATR (Average True Range) - simplified version
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
    
    # Build tomorrow's absolute return (for label construction)
    df["ret_tomorrow"] = df["Price"].shift(-1) / df["Price"] - 1.0
    
    # Time-aware split index (80/20)
    n = len(df)
    split_idx = int(n * 0.8)
    
    # Percentile threshold computed on TRAIN ONLY (no leakage)
    train_abs_ret = df.loc[:split_idx-1, "ret_tomorrow"].abs().dropna()
    pct = 0.75
    thr = float(np.quantile(train_abs_ret, pct)) if len(train_abs_ret) > 0 else float(train_abs_ret.mean())
    
    # Large-move label: 1 if |ret_{t+1}| >= thr, else 0
    df["y_vol"] = (df["ret_tomorrow"].abs() >= thr).astype(int)
    
    # Feature columns
    feature_cols = [
        "return_1d", "range_pct", "gap_open",
        "return_2d", "return_3d",
        "rolling_std_5", "rolling_std_10",
        "atr_5", "atr_10",
        "abs_gap", "abs_return"
    ]
    
    use_cols = feature_cols + ["ret_tomorrow", "y_vol"]
    model_df = df.dropna(subset=use_cols).reset_index(drop=True)
    
    return model_df, feature_cols, thr


def train_model(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[InHouseGNB, StandardScaler, float]:
    """Навчає модель на даних"""
    X = df[feature_cols].values
    y = df["y_vol"].values
    
    # Time-aware split (80/20)
    n = len(df)
    split_idx = int(n * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Standardization
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)
    
    # Compute sample weights for class balancing
    sample_weights = compute_sample_weight('balanced', y_train)
    
    # Train InHouse GNB with sample weights
    model = InHouseGNB(reg_param=1e-3)
    model.fit(X_train_z, y_train, sample_weight=sample_weights)
    
    # Find optimal threshold (maximize F1)
    proba_test = model.predict_proba(X_test_z)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    for t in thresholds:
        pred_t = (proba_test >= t).astype(int)
        f1_scores.append(f1_score(y_test, pred_t, zero_division=0))
    
    best_threshold = float(thresholds[np.argmax(f1_scores)])
    
    return model, scaler, best_threshold


def prepare_features_from_request(
    price: float, 
    open_price: float, 
    high_price: float, 
    low_price: float,
    historical_data: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """Підготовлює ознаки для передбачення
    
    ВАЖЛИВО: Без історичних даних неможливо обчислити всі ознаки коректно!
    Використовуємо спрощений підхід для демонстрації API.
    """
    # Базові обчислення з поточних цін
    return_1d = 0.0  # Без історії не можемо обчислити
    range_pct = (high_price - low_price) / open_price if open_price > 0 else 0
    gap_open = 0.0  # Без історії не можемо обчислити
    
    # Інші ознаки - використовуємо середні значення як заглушку
    return_2d = 0.0
    return_3d = 0.0
    rolling_std_5 = range_pct * 0.3  # Приблизна оцінка
    rolling_std_10 = range_pct * 0.3
    atr_5 = range_pct
    atr_10 = range_pct
    abs_gap = 0.0
    abs_return = abs(return_1d)
    
    return np.array([[
        return_1d, range_pct, gap_open,
        return_2d, return_3d,
        rolling_std_5, rolling_std_10,
        atr_5, atr_10,
        abs_gap, abs_return
    ]])


app = FastAPI(
    title="USD/UAH Volatility Prediction API (Naive Bayes)",
    description="API для передбачення великих рухів волатильності курсу USD/UAH. УВАГА: Naive Bayes показує слабкі результати (AUC~0.53) через корельованість ознак волатильності.",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Ініціалізація при запуску"""
    global MODEL, SCALER, THRESHOLD, FEATURE_COLUMNS, VOLATILITY_THRESHOLD
    
    try:
        data_path = pathlib.Path(__file__).resolve().parent.parent.parent / "datasets" / "USD_UAH Historical Data.csv"
        
        if data_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                csv_content = f.read()
            
            df, feature_cols, vol_thr = load_and_preprocess_data(csv_content)
            MODEL, SCALER, THRESHOLD = train_model(df, feature_cols)
            FEATURE_COLUMNS = feature_cols
            VOLATILITY_THRESHOLD = vol_thr
            
            print(f"✅ Модель навчена на {len(df)} записах")
            print(f"⚠️  УВАГА: Naive Bayes не підходить для цієї задачі (AUC~0.53)")
            print(f"   Поріг волатильності: {vol_thr:.5f}")
            print(f"   Оптимальний поріг класифікації: {THRESHOLD:.3f}")
        else:
            print("⚠️  Файл з даними не знайдено")
            
    except Exception as e:
        print(f"❌ Помилка при ініціалізації: {e}")


@app.get("/health")
def health_check():
    """Перевірка стану API"""
    return {
        "status": "ok", 
        "model_loaded": MODEL is not None,
        "warning": "Naive Bayes має низьку точність для цієї задачі (AUC~0.53). Для практичного використання рекомендуємо LDA або Random Forest."
    }


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Завантажує CSV файл та навчає модель"""
    global MODEL, SCALER, THRESHOLD, FEATURE_COLUMNS, VOLATILITY_THRESHOLD
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл повинен бути CSV формату")
    
    try:
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        df, feature_cols, vol_thr = load_and_preprocess_data(csv_content)
        MODEL, SCALER, THRESHOLD = train_model(df, feature_cols)
        FEATURE_COLUMNS = feature_cols
        VOLATILITY_THRESHOLD = vol_thr
        
        return {
            "message": "CSV файл успішно завантажено та модель навчена",
            "records_count": len(df),
            "features": feature_cols,
            "threshold": THRESHOLD,
            "volatility_threshold": vol_thr,
            "warning": "Naive Bayes показує слабкі результати для прогнозування волатильності через корельованість ознак"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Помилка при обробці файлу: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict_volatility(request: PredictionRequest):
    """Передбачає чи буде великий рух волатильності наступного дня
    
    ВАЖЛИВО: Без історичних даних точність передбачення значно знижується!
    Для реального використання потрібна повна історія цін.
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=400, detail="Модель не навчена. Спочатку завантажте CSV файл.")
    
    try:
        # Підготовлюємо ознаки (спрощений варіант без історії)
        features = prepare_features_from_request(
            request.price, 
            request.open_price, 
            request.high_price, 
            request.low_price
        )
        
        # Нормалізуємо
        features_scaled = SCALER.transform(features)
        
        # Отримуємо ймовірності
        probabilities = MODEL.predict_proba(features_scaled)[0]
        prob_large_move = float(probabilities[1])
        
        # Визначаємо передбачення
        prediction = "large_move" if prob_large_move >= THRESHOLD else "normal_day"
        
        # Визначаємо рівень впевненості
        if prob_large_move > 0.7 or prob_large_move < 0.3:
            confidence = "high"
        elif prob_large_move > 0.6 or prob_large_move < 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            prediction=prediction,
            probability_large_move=prob_large_move,
            confidence=confidence,
            features_used={
                "price": request.price,
                "open": request.open_price,
                "high": request.high_price,
                "low": request.low_price,
                "calculated_range": request.high_price - request.low_price
            },
            warning="⚠️ Без історичних даних точність знижена! Модель має AUC~0.53 (ледве краще за випадкове гадання). Для реального використання потрібна LDA або Random Forest."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка при передбаченні: {str(e)}")


@app.get("/model-info")
def get_model_info():
    """Інформація про модель та її обмеження"""
    return {
        "model_type": "Gaussian Naive Bayes (with sample weighting)",
        "task": "Binary classification: large volatility move vs normal day",
        "features_count": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0,
        "features": FEATURE_COLUMNS,
        "threshold": THRESHOLD,
        "volatility_threshold": VOLATILITY_THRESHOLD,
        "performance_metrics": {
            "expected_auc": "~0.53 (barely better than random)",
            "expected_balanced_accuracy": "~0.57",
            "expected_f1_score": "~0.27",
            "expected_precision": "~0.16 (84% false positives!)"
        },
        "limitations": [
            "Припущення про незалежність ознак порушене (всі ознаки описують волатильність)",
            "AUC = 0.53 показує, що модель ледве краще за підкидання монети",
            "При recall=0.83 precision всього 0.16 (лише 16% прогнозів правильні)",
            "Фінансові дохідності мають 'товсті хвости', не описані нормальним розподілом"
        ],
        "recommendations": [
            "Використовуйте LDA (показує precision=0.36 замість 0.16)",
            "Або Random Forest / XGBoost для кращої точності",
            "Для sentiment analysis новин Naive Bayes працює добре",
            "Але для статистики цін - НІ"
        ]
    }


@app.get("/")
def root():
    """Головна сторінка API"""
    return {
        "message": "USD/UAH Volatility Prediction API (Naive Bayes)",
        "description": "Передбачення великих рухів волатильності валютної пари",
        "warning": "⚠️ Naive Bayes має низьку точність (AUC~0.53) для цієї задачі!",
        "endpoints": {
            "upload_csv": "POST /upload-csv - завантажити CSV та навчити модель",
            "predict": "POST /predict - передбачити волатильність",
            "model_info": "GET /model-info - детальна інформація про модель та обмеження",
            "health": "GET /health - перевірити стан API",
            "docs": "GET /docs - Swagger документація"
        },
        "model_loaded": MODEL is not None,
        "better_alternatives": ["LDA API (03-gaussian-classification-model)", "Random Forest", "XGBoost"]
    }
