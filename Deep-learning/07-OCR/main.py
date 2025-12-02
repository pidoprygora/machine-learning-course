"""
Optical Character Recognition (OCR) - Розпізнавання тексту
Фокус: Confusion Matrix + Accuracy Trend + Типові помилки

Що реалізовано:
- Навчання CNN моделі для розпізнавання рукописних цифр (MNIST)
- Побудова Confusion Matrix для всіх класів (60%)
- Графік тренду точності (Accuracy Trend) по епохах (20%)
- Візуалізація та аналіз типових помилок розпізнавання (20%)
- Live-режим: розпізнавання цифр з веб-камери
"""

import time
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import cv2

# Налаштування
tf.random.set_seed(42)
np.random.seed(42)

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Директорії
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def print_section(title: str) -> None:
    """Красивий заголовок секції в консолі."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# 1. Завантаження та підготовка даних MNIST
# ---------------------------------------------------------------------------

IMG_SIZE = (28, 28)
NUM_CLASSES = 10
BATCH_SIZE = 64


def load_mnist_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Завантаження датасету MNIST.
    Повертає: (x_train, y_train, x_test, y_test)
    """
    print_section("ЗАВАНТАЖЕННЯ ДАТАСЕТУ MNIST")
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Нормалізація [0, 255] -> [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Додаємо канал (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    print(f"\n✓ Train set: {x_train.shape[0]} зображень")
    print(f"✓ Test set:  {x_test.shape[0]} зображень")
    print(f"✓ Розмір зображення: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"✓ Кількість класів: {NUM_CLASSES} (цифри 0-9)")
    
    # Розподіл класів
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n📊 Розподіл класів у train set:")
    for u, c in zip(unique, counts):
        print(f"   Цифра {u}: {c} прикладів ({100*c/len(y_train):.1f}%)")
    
    return x_train, y_train, x_test, y_test


def visualize_dataset_samples(x_data: np.ndarray, y_data: np.ndarray, num_samples: int = 20) -> None:
    """Візуалізація прикладів з датасету."""
    print_section("ВІЗУАЛІЗАЦІЯ ПРИКЛАДІВ З ДАТАСЕТУ")
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    
    indices = np.random.choice(len(x_data), num_samples, replace=False)
    
    for idx, ax in enumerate(axes.flatten()):
        i = indices[idx]
        ax.imshow(x_data[i, :, :, 0], cmap='gray')
        ax.set_title(f"{y_data[i]}", fontsize=12)
        ax.axis('off')
    
    plt.suptitle("Приклади з датасету MNIST", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "dataset_samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено приклади датасету: {out_path}")
    plt.close(fig)


def plot_class_distribution(y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Візуалізація розподілу класів."""
    print_section("РОЗПОДІЛ КЛАСІВ")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    axes[0].bar(unique_train, counts_train, color=colors, edgecolor='black')
    axes[0].set_xlabel("Цифра", fontsize=12)
    axes[0].set_ylabel("Кількість", fontsize=12)
    axes[0].set_title("Розподіл класів (Train)", fontsize=14)
    axes[0].set_xticks(range(10))
    
    # Test distribution  
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    axes[1].bar(unique_test, counts_test, color=colors, edgecolor='black')
    axes[1].set_xlabel("Цифра", fontsize=12)
    axes[1].set_ylabel("Кількість", fontsize=12)
    axes[1].set_title("Розподіл класів (Test)", fontsize=14)
    axes[1].set_xticks(range(10))
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено розподіл класів: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Побудова CNN моделі для OCR
# ---------------------------------------------------------------------------


def build_ocr_model(input_shape: Tuple[int, int, int] = (28, 28, 1)) -> keras.Model:
    """
    Побудова CNN моделі для розпізнавання цифр.
    Архітектура: Conv -> Conv -> MaxPool -> Conv -> Conv -> MaxPool -> Dense -> Output
    """
    model = keras.Sequential([
        # Перший блок
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Другий блок
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Третій блок
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Fully connected
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name="OCR_CNN")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_ocr_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64
) -> Tuple[keras.Model, pd.DataFrame]:
    """
    Навчання OCR моделі.
    Повертає: (model, history_df)
    """
    print_section("НАВЧАННЯ OCR МОДЕЛІ")
    
    model_path = MODELS_DIR / "ocr_cnn.h5"
    
    print("\n  Створення нової моделі...")
    model = build_ocr_model(input_shape=(*IMG_SIZE, 1))
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n  Навчання моделі...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Зберігаємо історію
    hist_df = pd.DataFrame(history.history)
    hist_df['epoch'] = range(1, len(hist_df) + 1)
    hist_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"\n✓ Історія навчання збережена: results/training_history.csv")
    
    # Зберігаємо модель
    model.save(model_path)
    print(f"✓ Модель збережена: {model_path}")
    
    return model, hist_df


# ---------------------------------------------------------------------------
# 3. Accuracy Trend - Графік тренду точності (20%)
# ---------------------------------------------------------------------------


def plot_accuracy_trend(hist_df: pd.DataFrame) -> None:
    """
    Побудова графіку тренду точності по епохах.
    Це основна вимога лабораторної (20%).
    """
    print_section("ACCURACY TREND - ТРЕНД ТОЧНОСТІ (20%)")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = hist_df['epoch'] if 'epoch' in hist_df.columns else range(1, len(hist_df) + 1)
    
    # Графік точності
    axes[0].plot(epochs, hist_df['accuracy'], 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    axes[0].plot(epochs, hist_df['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[0].set_xlabel('Епоха', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Тренд точності (Accuracy Trend)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0.9, 1.01])
    
    # Додаємо анотації для кінцевих значень
    final_train_acc = hist_df['accuracy'].iloc[-1]
    final_val_acc = hist_df['val_accuracy'].iloc[-1]
    axes[0].annotate(f'{final_train_acc:.4f}', 
                     xy=(len(epochs), final_train_acc), 
                     xytext=(5, 0), textcoords='offset points',
                     fontsize=10, color='blue')
    axes[0].annotate(f'{final_val_acc:.4f}', 
                     xy=(len(epochs), final_val_acc), 
                     xytext=(5, 0), textcoords='offset points',
                     fontsize=10, color='red')
    
    # Графік втрат
    axes[1].plot(epochs, hist_df['loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[1].plot(epochs, hist_df['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[1].set_xlabel('Епоха', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Тренд функції втрат (Loss Trend)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "accuracy_trend.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено графік тренду точності: {out_path}")
    plt.close(fig)
    
    # Додатковий детальний графік
    plot_detailed_accuracy_analysis(hist_df)


def plot_detailed_accuracy_analysis(hist_df: pd.DataFrame) -> None:
    """Детальний аналіз тренду точності."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(hist_df) + 1)
    
    # 1. Accuracy з заповненням області
    axes[0, 0].fill_between(epochs, hist_df['accuracy'], alpha=0.3, color='blue')
    axes[0, 0].fill_between(epochs, hist_df['val_accuracy'], alpha=0.3, color='red')
    axes[0, 0].plot(epochs, hist_df['accuracy'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, hist_df['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Епоха', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Динаміка точності', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Різниця між train і val accuracy (overfitting indicator)
    acc_diff = np.array(hist_df['accuracy']) - np.array(hist_df['val_accuracy'])
    colors = ['green' if d < 0.01 else 'orange' if d < 0.03 else 'red' for d in acc_diff]
    axes[0, 1].bar(epochs, acc_diff, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 1].axhline(y=0.01, color='orange', linestyle='--', label='Поріг уваги (1%)')
    axes[0, 1].axhline(y=0.03, color='red', linestyle='--', label='Поріг overfitting (3%)')
    axes[0, 1].set_xlabel('Епоха', fontsize=11)
    axes[0, 1].set_ylabel('Train Acc - Val Acc', fontsize=11)
    axes[0, 1].set_title('Індикатор перенавчання', fontsize=12)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Покращення accuracy по епохах
    val_acc_improvement = np.diff(hist_df['val_accuracy'], prepend=hist_df['val_accuracy'].iloc[0])
    colors = ['green' if i > 0 else 'red' for i in val_acc_improvement]
    axes[1, 0].bar(epochs, val_acc_improvement, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Епоха', fontsize=11)
    axes[1, 0].set_ylabel('Δ Validation Accuracy', fontsize=11)
    axes[1, 0].set_title('Покращення точності по епохах', fontsize=12)
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Learning rate (якщо є) або кумулятивне покращення
    cumulative_improvement = np.cumsum(val_acc_improvement)
    axes[1, 1].plot(epochs, cumulative_improvement, 'g-o', linewidth=2, markersize=6)
    axes[1, 1].fill_between(epochs, cumulative_improvement, alpha=0.3, color='green')
    axes[1, 1].set_xlabel('Епоха', fontsize=11)
    axes[1, 1].set_ylabel('Кумулятивне покращення', fontsize=11)
    axes[1, 1].set_title('Кумулятивний прогрес навчання', fontsize=12)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "accuracy_trend_detailed.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Збережено детальний аналіз точності: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Confusion Matrix - Матриця помилок (60%)
# ---------------------------------------------------------------------------


def build_confusion_matrices(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Побудова Confusion Matrix для всіх класів.
    Це головна вимога лабораторної (60%).
    
    Повертає: (y_true, y_pred, y_probs)
    """
    print_section("CONFUSION MATRIX - МАТРИЦЯ ПОМИЛОК (60%)")
    
    # Отримуємо передбачення
    print("\n  Отримання передбачень на тест-сеті...")
    y_probs = model.predict(x_test, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    
    # Обчислюємо confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Зберігаємо числові дані
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{i}" for i in range(10)],
        columns=[f"Pred_{i}" for i in range(10)]
    )
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    print(f"\n✓ Confusion matrix збережена: results/confusion_matrix.csv")
    
    # Візуалізація основної confusion matrix
    plot_confusion_matrix_main(cm, y_test, y_pred)
    
    # Нормалізована confusion matrix
    plot_confusion_matrix_normalized(cm)
    
    # Per-class analysis
    plot_per_class_metrics(y_test, y_pred, y_probs)
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    
    # Зберігаємо classification report
    report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")
    print(f"✓ Classification report збережено: results/classification_report.csv")
    
    return y_test, y_pred, y_probs


def plot_confusion_matrix_main(cm: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Основна візуалізація confusion matrix."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
        cbar_kws={'label': 'Кількість'}
    )
    
    ax.set_xlabel('Передбачена цифра', fontsize=14)
    ax.set_ylabel('Справжня цифра', fontsize=14)
    ax.set_title('Confusion Matrix (Матриця помилок)\nOCR модель на MNIST', fontsize=16, fontweight='bold')
    
    # Додаємо статистику
    accuracy = accuracy_score(y_test, y_pred)
    total_correct = np.trace(cm)
    total_samples = np.sum(cm)
    
    stats_text = f"Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})"
    ax.text(0.5, -0.1, stats_text, transform=ax.transAxes, fontsize=12,
            ha='center', va='top', fontweight='bold')
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено confusion matrix: {out_path}")
    plt.close(fig)


def plot_confusion_matrix_normalized(cm: np.ndarray) -> None:
    """Нормалізована confusion matrix (у відсотках)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Нормалізація по рядках (precision-oriented)
    cm_row_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_row_norm,
        annot=True,
        fmt='.2%',
        cmap='Greens',
        xticklabels=range(10),
        yticklabels=range(10),
        ax=axes[0],
        vmin=0, vmax=1
    )
    axes[0].set_xlabel('Передбачена цифра', fontsize=12)
    axes[0].set_ylabel('Справжня цифра', fontsize=12)
    axes[0].set_title('Нормалізована по рядках (Recall)', fontsize=14, fontweight='bold')
    
    # Нормалізація по стовпцях (recall-oriented)
    cm_col_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    
    sns.heatmap(
        cm_col_norm,
        annot=True,
        fmt='.2%',
        cmap='Oranges',
        xticklabels=range(10),
        yticklabels=range(10),
        ax=axes[1],
        vmin=0, vmax=1
    )
    axes[1].set_xlabel('Передбачена цифра', fontsize=12)
    axes[1].set_ylabel('Справжня цифра', fontsize=12)
    axes[1].set_title('Нормалізована по стовпцях (Precision)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "confusion_matrix_normalized.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Збережено нормалізовану confusion matrix: {out_path}")
    plt.close(fig)


def plot_per_class_metrics(y_test: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> None:
    """Метрики для кожного класу окремо."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Обчислюємо метрики для кожного класу
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(10):
        y_true_binary = (y_test == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        precisions.append(precision_score(y_true_binary, y_pred_binary, zero_division=0))
        recalls.append(recall_score(y_true_binary, y_pred_binary, zero_division=0))
        f1_scores.append(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    
    x = np.arange(10)
    width = 0.25
    
    # 1. Precision, Recall, F1 по класах
    axes[0, 0].bar(x - width, precisions, width, label='Precision', color='#3498db')
    axes[0, 0].bar(x, recalls, width, label='Recall', color='#2ecc71')
    axes[0, 0].bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')
    axes[0, 0].set_xlabel('Цифра', fontsize=11)
    axes[0, 0].set_ylabel('Score', fontsize=11)
    axes[0, 0].set_title('Метрики по класах', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0.95, 1.01])
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # 2. Кількість помилок по класах
    errors_per_class = []
    for i in range(10):
        mask = y_test == i
        errors = np.sum(y_pred[mask] != i)
        errors_per_class.append(errors)
    
    colors = plt.cm.Reds(np.array(errors_per_class) / max(errors_per_class))
    axes[0, 1].bar(x, errors_per_class, color=colors, edgecolor='black')
    axes[0, 1].set_xlabel('Цифра', fontsize=11)
    axes[0, 1].set_ylabel('Кількість помилок', fontsize=11)
    axes[0, 1].set_title('Кількість помилок по класах', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Додаємо значення над стовпцями
    for i, v in enumerate(errors_per_class):
        axes[0, 1].text(i, v + 0.5, str(v), ha='center', fontsize=10)
    
    # 3. Найчастіші помилки (confusion pairs)
    cm = confusion_matrix(y_test, y_pred)
    np.fill_diagonal(cm, 0)  # Прибираємо діагональ
    
    # Знаходимо топ-10 пар помилок
    flat_indices = np.argsort(cm.ravel())[::-1][:10]
    top_pairs = []
    top_counts = []
    
    for idx in flat_indices:
        true_digit = idx // 10
        pred_digit = idx % 10
        count = cm[true_digit, pred_digit]
        if count > 0:
            top_pairs.append(f"{true_digit}→{pred_digit}")
            top_counts.append(count)
    
    axes[1, 0].barh(range(len(top_pairs)), top_counts, color='#e74c3c', edgecolor='black')
    axes[1, 0].set_yticks(range(len(top_pairs)))
    axes[1, 0].set_yticklabels(top_pairs)
    axes[1, 0].set_xlabel('Кількість помилок', fontsize=11)
    axes[1, 0].set_title('Топ-10 найчастіших помилок', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # Додаємо значення
    for i, v in enumerate(top_counts):
        axes[1, 0].text(v + 0.5, i, str(v), va='center', fontsize=10)
    
    # 4. Розподіл confidence для правильних і неправильних передбачень
    max_probs = np.max(y_probs, axis=1)
    correct_mask = y_test == y_pred
    
    axes[1, 1].hist(max_probs[correct_mask], bins=50, alpha=0.7, label=f'Правильні (n={np.sum(correct_mask)})', color='green', density=True)
    axes[1, 1].hist(max_probs[~correct_mask], bins=50, alpha=0.7, label=f'Помилкові (n={np.sum(~correct_mask)})', color='red', density=True)
    axes[1, 1].set_xlabel('Confidence (max probability)', fontsize=11)
    axes[1, 1].set_ylabel('Щільність', fontsize=11)
    axes[1, 1].set_title('Розподіл впевненості моделі', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "per_class_metrics.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Збережено метрики по класах: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Типові помилки - Ілюстрація (20%)
# ---------------------------------------------------------------------------


def illustrate_typical_errors(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    num_examples: int = 25
) -> None:
    """
    Візуалізація та аналіз типових помилок розпізнавання.
    Це важлива вимога лабораторної (20%).
    """
    print_section("ТИПОВІ ПОМИЛКИ - ІЛЮСТРАЦІЯ (20%)")
    
    # Знаходимо всі помилки
    error_mask = y_test != y_pred
    error_indices = np.where(error_mask)[0]
    
    print(f"\n📊 Статистика помилок:")
    print(f"   Всього тестових прикладів: {len(y_test)}")
    print(f"   Кількість помилок: {len(error_indices)}")
    print(f"   Accuracy: {(1 - len(error_indices)/len(y_test))*100:.2f}%")
    
    if len(error_indices) == 0:
        print("\n✓ Помилок немає! Модель ідеальна.")
        return
    
    # Сортуємо помилки за confidence (найвпевненіші помилки цікавіші)
    error_confidences = np.max(y_probs[error_indices], axis=1)
    sorted_indices = error_indices[np.argsort(error_confidences)[::-1]]
    
    # Візуалізація найвпевненіших помилок
    plot_confident_errors(x_test, y_test, y_pred, y_probs, sorted_indices, num_examples)
    
    # Аналіз типів помилок
    analyze_error_patterns(y_test, y_pred, x_test, error_indices)
    
    # Візуалізація помилок по парах цифр
    plot_error_pairs(x_test, y_test, y_pred, y_probs, error_indices)


def plot_confident_errors(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    sorted_error_indices: np.ndarray,
    num_examples: int = 25
) -> None:
    """Візуалізація найвпевненіших помилок моделі."""
    
    n = min(num_examples, len(sorted_error_indices))
    cols = 5
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
    
    for idx, ax in enumerate(axes[:n]):
        i = sorted_error_indices[idx]
        
        img = x_test[i, :, :, 0]
        true_label = y_test[i]
        pred_label = y_pred[i]
        confidence = y_probs[i, pred_label]
        true_prob = y_probs[i, true_label]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true_label} ({true_prob:.1%})\nPred: {pred_label} ({confidence:.1%})", 
                     fontsize=10, color='red')
        ax.axis('off')
    
    # Приховуємо зайві осі
    for ax in axes[n:]:
        ax.axis('off')
    
    plt.suptitle('Топ помилок з найвищою впевненістю моделі', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "typical_errors_confident.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено найвпевненіші помилки: {out_path}")
    plt.close(fig)


def analyze_error_patterns(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    x_test: np.ndarray,
    error_indices: np.ndarray
) -> None:
    """Аналіз патернів помилок."""
    
    # Знаходимо найчастіші пари помилок
    error_pairs = {}
    for i in error_indices:
        pair = (y_test[i], y_pred[i])
        if pair not in error_pairs:
            error_pairs[pair] = []
        error_pairs[pair].append(i)
    
    # Сортуємо за частотою
    sorted_pairs = sorted(error_pairs.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("\n📊 Найчастіші пари помилок:")
    for pair, indices in sorted_pairs[:10]:
        print(f"   {pair[0]} → {pair[1]}: {len(indices)} помилок")
    
    # Зберігаємо статистику
    error_stats = []
    for pair, indices in sorted_pairs:
        error_stats.append({
            'true_digit': pair[0],
            'predicted_digit': pair[1],
            'count': len(indices),
            'percentage': 100 * len(indices) / len(error_indices)
        })
    
    error_df = pd.DataFrame(error_stats)
    error_df.to_csv(OUTPUT_DIR / "error_patterns.csv", index=False)
    print(f"\n✓ Патерни помилок збережено: results/error_patterns.csv")


def plot_error_pairs(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    error_indices: np.ndarray
) -> None:
    """Візуалізація помилок для найпроблемніших пар цифр."""
    
    # Знаходимо топ-6 пар помилок
    error_pairs = {}
    for i in error_indices:
        pair = (y_test[i], y_pred[i])
        if pair not in error_pairs:
            error_pairs[pair] = []
        error_pairs[pair].append(i)
    
    sorted_pairs = sorted(error_pairs.items(), key=lambda x: len(x[1]), reverse=True)[:6]
    
    fig, axes = plt.subplots(6, 5, figsize=(15, 18))
    
    for row, (pair, indices) in enumerate(sorted_pairs):
        true_digit, pred_digit = pair
        
        # Беремо до 5 прикладів
        examples = indices[:5]
        
        for col, idx in enumerate(examples):
            img = x_test[idx, :, :, 0]
            confidence = y_probs[idx, pred_digit]
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f"Conf: {confidence:.1%}", fontsize=9)
            axes[row, col].axis('off')
            
            if col == 0:
                axes[row, col].set_ylabel(f"{true_digit}→{pred_digit}\n({len(indices)} пом.)", 
                                          fontsize=11, rotation=0, labelpad=50)
        
        # Приховуємо зайві колонки
        for col in range(len(examples), 5):
            axes[row, col].axis('off')
    
    plt.suptitle('Приклади помилок для найпроблемніших пар цифр', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "error_pairs_examples.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✓ Збережено приклади пар помилок: {out_path}")
    plt.close(fig)


def create_error_analysis_summary(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> None:
    """Створення підсумкового звіту про помилки."""
    
    print_section("ПІДСУМОК АНАЛІЗУ ПОМИЛОК")
    
    error_mask = y_test != y_pred
    correct_mask = ~error_mask
    
    # Статистика впевненості
    correct_conf = np.max(y_probs[correct_mask], axis=1)
    error_conf = np.max(y_probs[error_mask], axis=1)
    
    print("\n📊 Порівняння впевненості моделі:")
    print(f"   Правильні передбачення:")
    print(f"      Середня впевненість: {np.mean(correct_conf):.4f}")
    print(f"      Мін/Макс: {np.min(correct_conf):.4f} / {np.max(correct_conf):.4f}")
    
    if len(error_conf) > 0:
        print(f"   Помилкові передбачення:")
        print(f"      Середня впевненість: {np.mean(error_conf):.4f}")
        print(f"      Мін/Макс: {np.min(error_conf):.4f} / {np.max(error_conf):.4f}")
    
    # Threshold analysis
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    print("\n📊 Аналіз порогу впевненості:")
    print(f"   {'Поріг':<10} {'Відхилено':<15} {'Accuracy залишку':<20}")
    
    for thr in thresholds:
        max_probs = np.max(y_probs, axis=1)
        accepted_mask = max_probs >= thr
        
        if np.sum(accepted_mask) > 0:
            acc = accuracy_score(y_test[accepted_mask], y_pred[accepted_mask])
            rejected_pct = 100 * (1 - np.mean(accepted_mask))
            print(f"   {thr:<10.2f} {rejected_pct:<15.1f}% {acc:<20.4f}")


# ---------------------------------------------------------------------------
# 6. Live-режим: OCR з веб-камери
# ---------------------------------------------------------------------------


def preprocess_for_ocr(frame: np.ndarray) -> np.ndarray:
    """Попередня обробка кадру для OCR."""
    # Конвертуємо в grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Гаусове розмиття для зменшення шуму
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Адаптивне порогування
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh


def find_digit_contours(thresh: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Знаходить контури цифр на зображенні."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Фільтруємо за розміром
        area = w * h
        aspect_ratio = h / w if w > 0 else 0
        
        if area > 100 and 0.5 < aspect_ratio < 3:
            digit_boxes.append((x, y, w, h))
    
    # Сортуємо зліва направо
    digit_boxes.sort(key=lambda b: b[0])
    
    return digit_boxes


def extract_digit(thresh: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Витягує та нормалізує зображення цифри."""
    x, y, w, h = box
    
    # Витягуємо регіон з padding
    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(thresh.shape[1], x + w + pad)
    y2 = min(thresh.shape[0], y + h + pad)
    
    digit_img = thresh[y1:y2, x1:x2]
    
    # Ресайз до 28x28 зі збереженням пропорцій
    h, w = digit_img.shape
    
    if h > w:
        new_h = 20
        new_w = int(w * 20 / h)
    else:
        new_w = 20
        new_h = int(h * 20 / w)
    
    if new_w > 0 and new_h > 0:
        digit_img = cv2.resize(digit_img, (new_w, new_h))
    
    # Центруємо на 28x28
    final_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    
    if new_w > 0 and new_h > 0:
        final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_img
    
    return final_img


def live_ocr(model: keras.Model, confidence_threshold: float = 0.7):
    """
    Live-режим: розпізнавання цифр з веб-камери.
    """
    print_section("LIVE OCR З ВЕБ-КАМЕРИ")
    
    print("\n  Відкриття веб-камери...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("  ⚠ Не вдалося відкрити веб-камеру")
        return
    
    print("\n✓ Live OCR запущено!")
    print("\n  Керування:")
    print("    - 'q' або 'ESC' — вихід")
    print("    - '+' — збільшити поріг впевненості")
    print("    - '-' — зменшити поріг впевненості")
    print("    - 's' — зберегти кадр")
    print("    - 'r' — показати ROI (region of interest)")
    
    fps_history = []
    saved_frames = 0
    show_roi = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ⚠ Не вдалося прочитати кадр")
                break
            
            start_time = time.time()
            
            # Визначаємо ROI (центральна частина кадру)
            h, w = frame.shape[:2]
            roi_size = min(h, w) // 2
            roi_x = (w - roi_size) // 2
            roi_y = (h - roi_size) // 2
            
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Попередня обробка
            thresh = preprocess_for_ocr(roi)
            
            # Знаходимо цифри
            boxes = find_digit_contours(thresh)
            
            recognized_text = ""
            
            for box in boxes:
                x, y, bw, bh = box
                
                # Витягуємо цифру
                digit_img = extract_digit(thresh, box)
                
                # Нормалізуємо та передбачаємо
                digit_normalized = digit_img.astype(np.float32) / 255.0
                digit_input = np.expand_dims(np.expand_dims(digit_normalized, axis=-1), axis=0)
                
                probs = model.predict(digit_input, verbose=0)[0]
                pred_digit = np.argmax(probs)
                confidence = probs[pred_digit]
                
                if confidence >= confidence_threshold:
                    recognized_text += str(pred_digit)
                    
                    # Малюємо box на ROI
                    color = (0, 255, 0)
                    cv2.rectangle(roi, (x, y), (x+bw, y+bh), color, 2)
                    cv2.putText(roi, f"{pred_digit} ({confidence:.0%})", 
                               (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Копіюємо ROI назад у frame
            frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size] = roi
            
            # Малюємо рамку ROI
            if show_roi:
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (255, 0, 0), 2)
            
            # FPS
            dt = time.time() - start_time
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Info overlay
            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"Threshold: {confidence_threshold:.0%}",
                f"Recognized: {recognized_text if recognized_text else 'None'}",
            ]
            
            y_text = 30
            for line in info_lines:
                cv2.putText(frame, line, (10, y_text),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_text += 30
            
            cv2.imshow("Live OCR - Digit Recognition", frame)
            
            # Показуємо threshold image
            if show_roi:
                cv2.imshow("Threshold", thresh)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n  Зупинка...")
                break
            elif key == ord("+") or key == ord("="):
                confidence_threshold = min(0.99, confidence_threshold + 0.05)
                print(f"  Threshold: {confidence_threshold:.0%}")
            elif key == ord("-") or key == ord("_"):
                confidence_threshold = max(0.1, confidence_threshold - 0.05)
                print(f"  Threshold: {confidence_threshold:.0%}")
            elif key == ord("s"):
                saved_frames += 1
                out_path = OUTPUT_DIR / f"live_ocr_{saved_frames}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"  ✓ Кадр збережено: {out_path}")
            elif key == ord("r"):
                show_roi = not show_roi
                if not show_roi:
                    cv2.destroyWindow("Threshold")
                print(f"  Show ROI: {show_roi}")
    
    except KeyboardInterrupt:
        print("\n  Перервано користувачем")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✓ Live OCR завершено")
        if fps_history:
            print(f"  Середній FPS: {np.mean(fps_history):.1f}")
        print(f"  Збережено кадрів: {saved_frames}")


# ---------------------------------------------------------------------------
# 7. Головне меню
# ---------------------------------------------------------------------------


def main():
    print("\n" + "=" * 80)
    print("  OPTICAL CHARACTER RECOGNITION (OCR)")
    print("  Confusion Matrix (60%) + Accuracy Trend (20%) + Typical Errors (20%)")
    print("=" * 80)
    
    print("\n📋 Меню:")
    print("  1. Навчити OCR модель та побудувати Accuracy Trend (20%)")
    print("  2. Побудувати Confusion Matrix (60%)")
    print("  3. Ілюструвати типові помилки (20%)")
    print("  4. Live OCR з веб-камери")
    print("  5. Виконати все послідовно (1→2→3)")
    
    choice = input("\n  Виберіть опцію (1-5): ").strip()
    
    # Завантажуємо дані
    x_train, y_train, x_test, y_test = load_mnist_dataset()
    
    model = None
    hist_df = None
    y_pred = None
    y_probs = None
    
    if choice in {"1", "5"}:
        # Візуалізація даних
        visualize_dataset_samples(x_train, y_train)
        plot_class_distribution(y_train, y_test)
        
        # Навчання моделі
        model, hist_df = train_ocr_model(x_train, y_train, x_test, y_test, epochs=15)
        
        # Accuracy Trend (20%)
        plot_accuracy_trend(hist_df)
    
    if choice in {"2", "5"}:
        # Завантажуємо модель якщо не навчена
        if model is None:
            model_path = MODELS_DIR / "ocr_cnn.h5"
            if model_path.exists():
                print(f"\n  Завантаження моделі з {model_path}...")
                model = keras.models.load_model(model_path)
            else:
                print("\n⚠ Спочатку виконайте пункт 1 для навчання моделі.")
                return
        
        # Confusion Matrix (60%)
        y_test_cm, y_pred, y_probs = build_confusion_matrices(model, x_test, y_test)
    
    if choice in {"3", "5"}:
        # Завантажуємо модель якщо не навчена
        if model is None:
            model_path = MODELS_DIR / "ocr_cnn.h5"
            if model_path.exists():
                print(f"\n  Завантаження моделі з {model_path}...")
                model = keras.models.load_model(model_path)
            else:
                print("\n⚠ Спочатку виконайте пункт 1 для навчання моделі.")
                return
        
        # Отримуємо передбачення якщо ще не отримані
        if y_pred is None:
            y_probs = model.predict(x_test, verbose=1)
            y_pred = np.argmax(y_probs, axis=1)
        
        # Typical Errors (20%)
        illustrate_typical_errors(model, x_test, y_test, y_pred, y_probs)
        create_error_analysis_summary(y_test, y_pred, y_probs)
    
    if choice == "4":
        # Завантажуємо модель
        model_path = MODELS_DIR / "ocr_cnn.h5"
        if model_path.exists():
            print(f"\n  Завантаження моделі з {model_path}...")
            model = keras.models.load_model(model_path)
            live_ocr(model, confidence_threshold=0.7)
        else:
            print("\n⚠ Спочатку виконайте пункт 1 для навчання моделі.")
            return
    
    print_section("ПІДСУМОК")
    print("\n✅ Лабораторна «Optical Character Recognition (OCR)» виконана.")
    print("\n📁 Результати збережені в каталозі 'results':")
    print("  - dataset_samples.png — приклади з датасету")
    print("  - class_distribution.png — розподіл класів")
    print("  - training_history.csv — числові дані навчання")
    print("  - accuracy_trend.png — тренд точності (20%)")
    print("  - accuracy_trend_detailed.png — детальний аналіз тренду")
    print("  - confusion_matrix.png — матриця помилок (60%)")
    print("  - confusion_matrix_normalized.png — нормалізована матриця")
    print("  - per_class_metrics.png — метрики по класах")
    print("  - classification_report.csv — звіт класифікації")
    print("  - typical_errors_confident.png — типові помилки (20%)")
    print("  - error_pairs_examples.png — приклади пар помилок")
    print("  - error_patterns.csv — патерни помилок")
    print("  - live_ocr_*.jpg — збережені кадри з live-режиму")
    
    print("\n📊 Відповідність вимогам:")
    print("  ✅ Confusion Matrix — 60%")
    print("  ✅ Accuracy Trend — 20%")
    print("  ✅ Typical Errors — 20%")


if __name__ == "__main__":
    main()

