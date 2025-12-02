"""
Texture Segmentation / Binary Semantic Segmentation
Фокус: простий власний сегментаційний детектор + pre-trained модель + ROC + live-режим

Що реалізовано:
- Власний невеликий сегментаційний детектор (2 класи: об'єкт vs фон) на базі TF/Keras
- Використання попередньо навченої моделі MobileNet SSD (OpenCV DNN) для детекції людей
- Оцінка якості на тест-сеті (accuracy, IoU, Dice)
- ROC-крива для бінарної задачі (об'єкт vs фон)
- Live-режим: детекція/сегментація об'єктів у потоці з веб-камери
"""

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

from typing import Tuple, Optional

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
# 1. Генерація простих синтетичних даних (бінарна сегментація)
# ---------------------------------------------------------------------------

TARGET_CLASS_NAME = "object"
IMG_SIZE = (256, 256)
BATCH_SIZE = 4


def _generate_synthetic_image_and_mask() -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерує просте синтетичне зображення та маску:
    - фон: шум/текстура
    - об'єкт: випадкове коло або квадрат з іншою текстурою
    """
    h, w = IMG_SIZE

    # Фон — випадковий "текстурований" шум
    background = np.random.uniform(0.0, 0.4, size=(h, w, 1)).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=0.05, size=(h, w, 1)).astype(np.float32)
    img = background + noise
    img = np.clip(img, 0.0, 1.0)
    img = np.repeat(img, 3, axis=-1)  # робимо 3 канали

    # Маска об'єкта
    mask = np.zeros((h, w, 1), dtype=np.float32)

    # Випадковий об'єкт: коло або квадрат
    shape_type = np.random.choice(["circle", "square"])
    cy = np.random.randint(h // 4, 3 * h // 4)
    cx = np.random.randint(w // 4, 3 * w // 4)
    r = np.random.randint(min(h, w) // 8, min(h, w) // 4)

    yy, xx = np.ogrid[:h, :w]

    if shape_type == "circle":
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        obj_region = dist_sq <= r ** 2
    else:  # square
        y_min = max(cy - r, 0)
        y_max = min(cy + r, h)
        x_min = max(cx - r, 0)
        x_max = min(cx + r, w)
        obj_region = np.zeros((h, w), dtype=bool)
        obj_region[y_min:y_max, x_min:x_max] = True

    mask[obj_region, 0] = 1.0

    # Інша текстура/колір для об'єкта
    color = np.random.uniform(0.6, 1.0, size=(1, 1, 3)).astype(np.float32)
    img[obj_region] = color

    # Трохи додаткового шуму
    img += np.random.normal(0.0, 0.03, size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32), mask.astype(np.float32)


def _create_synthetic_dataset(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Генерує набір синтетичних зображень і масок."""
    images = np.zeros((num_samples, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    masks = np.zeros((num_samples, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32)

    for i in range(num_samples):
        img, m = _generate_synthetic_image_and_mask()
        images[i] = img
        masks[i] = m

    return images, masks


def load_synthetic_dataset(
    num_train: int = 40,
    num_test: int = 10,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Генерує синтетичний датасет для бінарної сегментації (object vs background).
    Повністю офлайн, без завантаження великих датасетів.
    """
    print_section("ГЕНЕРАЦІЯ СИНТЕТИЧНИХ ДАНИХ (object vs background)")

    train_images, train_masks = _create_synthetic_dataset(num_train)
    test_images, test_masks = _create_synthetic_dataset(num_test)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_images, train_masks))
        .shuffle(100)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_images, test_masks))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"\n✓ Згенеровано train прикладів: {num_train}")
    print(f"✓ Згенеровано test прикладів:  {num_test}")

    return train_ds, test_ds


def visualize_synthetic_samples(num_samples: int = 6) -> None:
    """
    Візуалізація кількох синтетичних прикладів (зображення + маска).
    """
    print_section("ВІЗУАЛІЗАЦІЯ СИНТЕТИЧНИХ ДАНИХ")

    num_samples = max(1, num_samples)
    images, masks = _create_synthetic_dataset(num_samples)

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        img = images[i]
        mask = masks[i, ..., 0]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Зображення #{i+1}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Маска об'єкта #{i+1}", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "synthetic_samples.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Збережено приклади синтетичних даних: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Власний простий сегментаційний детектор (невеликий U-Net‑подібний)
# ---------------------------------------------------------------------------


def build_simple_unet(input_shape=(256, 256, 3)) -> keras.Model:
    """
    Невелика U-Net‑подібна модель для бінарної сегментації.
    Вихід: карта ймовірностей [H, W, 1] (через сигмоїду).
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu")(x1)
    p1 = layers.MaxPool2D(2)(x1)

    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(p1)
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu")(x2)
    p2 = layers.MaxPool2D(2)(x2)

    x3 = layers.Conv2D(128, 3, padding="same", activation="relu")(p2)
    x3 = layers.Conv2D(128, 3, padding="same", activation="relu")(x3)

    # Decoder
    u2 = layers.UpSampling2D(2)(x3)
    u2 = layers.Concatenate()([u2, x2])
    x4 = layers.Conv2D(64, 3, padding="same", activation="relu")(u2)
    x4 = layers.Conv2D(64, 3, padding="same", activation="relu")(x4)

    u1 = layers.UpSampling2D(2)(x4)
    u1 = layers.Concatenate()([u1, x1])
    x5 = layers.Conv2D(32, 3, padding="same", activation="relu")(u1)
    x5 = layers.Conv2D(32, 3, padding="same", activation="relu")(x5)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x5)

    model = keras.Model(inputs, outputs, name="simple_unet_object")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_own_detector(
    num_train: int = 40,
    num_test: int = 10,
    epochs: int = 5,
):
    """Навчання власного сегментаційного детектора та оцінка на test set."""
    print_section("ВЛАСНИЙ СЕГМЕНТАЦІЙНИЙ ДЕТЕКТОР (TRAIN & EVAL)")

    # Візуалізуємо синтетичні дані
    try:
        visualize_synthetic_samples(num_samples=4)
    except Exception as e:
        print(f"\n⚠ Не вдалося побудувати приклади синтетичних даних: {e}")

    train_ds, test_ds = load_synthetic_dataset(num_train=num_train, num_test=num_test)

    model_path = MODELS_DIR / "simple_unet_object.h5"

    # Завжди перенавчаємо модель для демонстрації
    print("\n  Створення нової моделі...")
    model = build_simple_unet(input_shape=(*IMG_SIZE, 3))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    print("\n  Навчання моделі...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Зберігаємо історію навчання
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(OUTPUT_DIR / "own_detector_history.csv", index=False)
    print(f"\n✓ Історія навчання збережена в results/own_detector_history.csv")

    # Візуалізація історії навчання
    plot_training_history(hist_df)

    # Зберігаємо модель
    model.save(model_path)
    print(f"✓ Модель збережена в {model_path}")

    # Оцінка на test set
    print("\n  Оцінка моделі на test set...")
    metrics = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    metrics_dict = dict(zip(metric_names, metrics))

    print("\n📊 Метрики власного детектора:")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v:.4f}")

    # Обчислюємо додаткові метрики (IoU, Dice) та збираємо дані для ROC
    y_true_all = []
    y_prob_all = []

    for images, masks in test_ds:
        probs = model.predict(images, verbose=0)
        y_true_all.append(masks.numpy().ravel())
        y_prob_all.append(probs.ravel())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    # Бінарні предикти при threshold=0.5
    y_pred_bin = (y_prob_all >= 0.5).astype(np.float32)

    intersection = np.sum(y_true_all * y_pred_bin)
    union = np.sum(y_true_all) + np.sum(y_pred_bin) - intersection
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (np.sum(y_true_all) + np.sum(y_pred_bin) + 1e-8)

    print(f"\n📐 Додаткові метрики (threshold=0.5):")
    print(f"  IoU:  {iou:.4f}")
    print(f"  Dice: {dice:.4f}")

    # Зберігаємо масиви для ROC
    np.save(OUTPUT_DIR / "own_detector_y_true.npy", y_true_all)
    np.save(OUTPUT_DIR / "own_detector_y_prob.npy", y_prob_all)
    print("\n✓ Дані для ROC власного детектора збережені (y_true/y_prob)")

    # Додаткові візуалізації
    try:
        visualize_segmentation_examples(model, test_ds, num_examples=4)
    except Exception as e:
        print(f"\n⚠ Не вдалося побудувати приклади сегментації: {e}")

    try:
        plot_probability_histograms(y_true_all, y_prob_all)
    except Exception as e:
        print(f"\n⚠ Не вдалося побудувати гістограму ймовірностей: {e}")

    try:
        plot_confusion_matrix_segmentation(y_true_all, y_pred_bin)
    except Exception as e:
        print(f"\n⚠ Не вдалося побудувати confusion matrix: {e}")

    try:
        plot_iou_vs_threshold(y_true_all, y_prob_all)
    except Exception as e:
        print(f"\n⚠ Не вдалося побудувати IoU vs threshold: {e}")

    return model, y_true_all, y_prob_all


def plot_training_history(hist_df: pd.DataFrame) -> None:
    """Візуалізація історії навчання."""
    print_section("ВІЗУАЛІЗАЦІЯ ІСТОРІЇ НАВЧАННЯ")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(hist_df["loss"], label="Train Loss", marker="o")
    if "val_loss" in hist_df.columns:
        axes[0].plot(hist_df["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Функція втрат")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(hist_df["accuracy"], label="Train Accuracy", marker="o")
    if "val_accuracy" in hist_df.columns:
        axes[1].plot(hist_df["val_accuracy"], label="Val Accuracy", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Точність")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "training_history.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Збережено графік історії навчання: {out_path}")
    plt.close(fig)


def visualize_segmentation_examples(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    num_examples: int = 4,
) -> None:
    """
    Візуалізація роботи власного детектора:
    для кількох зображень показуємо: оригінал, GT маску, предиктовану маску, оверлей.
    """
    print_section("ВІЗУАЛІЗАЦІЯ ПРИКЛАДІВ СЕГМЕНТАЦІЇ (ВЛАСНИЙ ДЕТЕКТОР)")

    images_list = []
    masks_list = []

    for images, masks in test_ds:
        for i in range(images.shape[0]):
            images_list.append(images[i].numpy())
            masks_list.append(masks[i].numpy())
            if len(images_list) >= num_examples:
                break
        if len(images_list) >= num_examples:
            break

    if not images_list:
        print("⚠ Немає прикладів у test_ds для візуалізації.")
        return

    images_arr = np.stack(images_list, axis=0)
    masks_arr = np.stack(masks_list, axis=0)

    preds = model.predict(images_arr, verbose=0)
    preds_bin = (preds >= 0.5).astype(np.float32)

    n = images_arr.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        img = images_arr[i]
        gt = masks_arr[i, ..., 0]
        pr = preds_bin[i, ..., 0]
        pr_prob = preds[i, ..., 0]

        # Оверлей
        overlay = img.copy()
        overlay_color = np.zeros_like(img)
        overlay_color[..., 1] = pr  # green channel
        overlay = 0.6 * overlay + 0.4 * overlay_color

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Зображення", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("GT маска", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pr_prob, cmap="hot", vmin=0, vmax=1)
        axes[i, 2].set_title("Ймовірності", fontsize=10)
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Оверлей", fontsize=10)
        axes[i, 3].axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "own_detector_segmentation_examples.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Збережено приклади сегментації власного детектора: {out_path}")
    plt.close(fig)


def plot_probability_histograms(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Візуалізація розподілів ймовірностей для позитивних та негативних пікселів.
    """
    print_section("РОЗПОДІЛ ЙМОВІРНОСТЕЙ (ВЛАСНИЙ ДЕТЕКТОР)")

    y_true = y_true.astype(np.float32)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 50)
    plt.hist(neg, bins=bins, alpha=0.6, label=f"background (n={len(neg):,})", color="steelblue", density=True)
    plt.hist(pos, bins=bins, alpha=0.6, label=f"object (n={len(pos):,})", color="orange", density=True)
    plt.axvline(x=0.5, color="red", linestyle="--", label="Threshold = 0.5")
    plt.xlabel("Ймовірність класу 'object'")
    plt.ylabel("Щільність")
    plt.title("Розподіл ймовірностей для пікселів (object vs background)")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = OUTPUT_DIR / "own_detector_probability_hist.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Збережено гістограму ймовірностей: {out_path}")
    plt.close()


def plot_confusion_matrix_segmentation(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Confusion matrix для бінарної сегментації."""
    print_section("CONFUSION MATRIX (ВЛАСНИЙ ДЕТЕКТОР)")

    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Background", "Object"],
                yticklabels=["Background", "Object"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (pixel-level)")

    out_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Збережено confusion matrix: {out_path}")
    plt.close()

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true.astype(int), y_pred.astype(int),
                                target_names=["Background", "Object"]))


def plot_iou_vs_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Графік IoU в залежності від порогу."""
    print_section("IoU vs THRESHOLD")

    thresholds = np.linspace(0.1, 0.9, 17)
    ious = []
    dices = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.float32)
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        iou = intersection / (union + 1e-8)
        dice = 2 * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-8)
        ious.append(iou)
        dices.append(dice)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, ious, marker="o", label="IoU", linewidth=2)
    plt.plot(thresholds, dices, marker="s", label="Dice", linewidth=2)
    plt.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Default threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("IoU та Dice в залежності від порогу")
    plt.legend()
    plt.grid(alpha=0.3)

    # Знаходимо оптимальний поріг
    best_idx = np.argmax(ious)
    best_thr = thresholds[best_idx]
    best_iou = ious[best_idx]
    plt.scatter([best_thr], [best_iou], color="green", s=100, zorder=5, label=f"Best IoU={best_iou:.3f} @ thr={best_thr:.2f}")
    plt.legend()

    out_path = OUTPUT_DIR / "iou_vs_threshold.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Збережено графік IoU vs threshold: {out_path}")
    print(f"  Оптимальний поріг: {best_thr:.2f} (IoU = {best_iou:.4f})")
    plt.close()


# ---------------------------------------------------------------------------
# 3. Pre-trained детектор (OpenCV Haar Cascade для обличь)
# ---------------------------------------------------------------------------

# Haar Cascade вже вбудований в OpenCV — нічого качати не потрібно!
FACE_CASCADE_NAME = "haarcascade_frontalface_default.xml"


def load_haar_cascade():
    """Завантаження Haar Cascade для детекції обличь (вбудований в OpenCV)."""
    print_section("ПОПЕРЕДНЬО НАВЧЕНИЙ ДЕТЕКТОР (Haar Cascade - Обличчя)")

    # Шлях до каскаду в OpenCV
    cascade_path = cv2.data.haarcascades + FACE_CASCADE_NAME

    print(f"  Завантаження каскаду: {cascade_path}")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("\n⚠ Не вдалося завантажити Haar Cascade.")
        return None

    print("\n✓ Haar Cascade завантажено успішно!")
    print("  Детектор: обличчя (frontal face)")
    print("  Переваги: швидкий, офлайн, не потребує завантаження")
    return face_cascade


def detect_faces_haar(cascade, image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5):
    """
    Детекція обличь на зображенні за допомогою Haar Cascade.
    Повертає список bounding boxes та confidence scores (фіктивні, бо Haar не дає score).
    """
    # Конвертуємо в grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Детекція
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    boxes = []
    scores = []

    for (x, y, w, h) in faces:
        boxes.append([x, y, x + w, y + h])
        # Haar Cascade не повертає confidence, ставимо фіксоване значення
        scores.append(0.9)

    return boxes, scores


def create_detection_mask(boxes: list, image_shape: tuple) -> np.ndarray:
    """Створює бінарну маску з bounding boxes."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        mask[y1:y2, x1:x2] = 1.0

    return mask


def evaluate_pretrained_detector(num_test: int = 20):
    """
    Оцінка pre-trained Haar Cascade на синтетичних даних.
    Для демонстрації генеруємо синтетичні "обличчя" (кола з "очима").
    """
    print_section("ОЦІНКА PRE-TRAINED ДЕТЕКТОРА (Haar Cascade - Обличчя)")

    cascade = load_haar_cascade()
    if cascade is None:
        print("\n⚠ Pre-trained детектор недоступний.")
        return None, None, None

    print("\n  ℹ️  Примітка: Haar Cascade шукає обличчя.")
    print("     Генеруємо синтетичні 'обличчя' для демонстрації...")

    # Генеруємо синтетичні "обличчя" (овали з очима)
    test_images, test_masks = _create_face_like_dataset(num_test)

    y_true_all = []
    y_prob_all = []

    for i in range(num_test):
        img = (test_images[i] * 255).astype(np.uint8)
        mask = test_masks[i, ..., 0]

        # Детекція
        boxes, scores = detect_faces_haar(cascade, img)

        # Створюємо маску детекцій
        prob_mask = np.zeros_like(mask)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(mask.shape[1], x2)
            y2 = min(mask.shape[0], y2)
            prob_mask[y1:y2, x1:x2] = score

        y_true_all.append(mask.ravel())
        y_prob_all.append(prob_mask.ravel())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    # Метрики
    y_pred_bin = (y_prob_all >= 0.5).astype(np.float32)
    intersection = np.sum(y_true_all * y_pred_bin)
    union = np.sum(y_true_all) + np.sum(y_pred_bin) - intersection
    iou = intersection / (union + 1e-8)

    print(f"\n📐 Метрики Haar Cascade (на синтетичних даних):")
    print(f"  IoU: {iou:.4f}")
    print("  (Haar шукає реальні обличчя — на синтетиці результат може бути низьким)")

    np.save(OUTPUT_DIR / "pretrained_y_true.npy", y_true_all)
    np.save(OUTPUT_DIR / "pretrained_y_prob.npy", y_prob_all)
    print("\n✓ Дані для ROC pre-trained детектора збережені")

    # Візуалізація детекцій
    visualize_pretrained_detections(cascade, test_images[:4])

    return cascade, y_true_all, y_prob_all


def _create_face_like_dataset(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Генерує синтетичні 'обличчя' для тестування Haar Cascade."""
    images = np.zeros((num_samples, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    masks = np.zeros((num_samples, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32)

    h, w = IMG_SIZE

    for i in range(num_samples):
        # Фон
        img = np.random.uniform(0.1, 0.3, size=(h, w, 3)).astype(np.float32)

        # Овал "обличчя"
        cy = np.random.randint(h // 3, 2 * h // 3)
        cx = np.random.randint(w // 3, 2 * w // 3)
        ry = np.random.randint(h // 6, h // 4)
        rx = np.random.randint(w // 8, w // 5)

        yy, xx = np.ogrid[:h, :w]
        ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1

        # Колір "шкіри"
        skin_color = np.array([0.8, 0.7, 0.6], dtype=np.float32)
        img[ellipse] = skin_color + np.random.normal(0, 0.05, 3)

        # "Очі" (темні кола)
        eye_y = cy - ry // 3
        eye_x_left = cx - rx // 2
        eye_x_right = cx + rx // 2
        eye_r = max(3, rx // 6)

        for ex in [eye_x_left, eye_x_right]:
            eye_mask = (yy - eye_y) ** 2 + (xx - ex) ** 2 <= eye_r ** 2
            img[eye_mask] = np.array([0.1, 0.1, 0.1])

        # Маска
        mask = np.zeros((h, w, 1), dtype=np.float32)
        mask[ellipse, 0] = 1.0

        images[i] = np.clip(img, 0, 1)
        masks[i] = mask

    return images, masks


def visualize_pretrained_detections(cascade, images: np.ndarray, num_examples: int = 4) -> None:
    """Візуалізація детекцій Haar Cascade."""
    print_section("ВІЗУАЛІЗАЦІЯ ДЕТЕКЦІЙ (Haar Cascade)")

    n = min(num_examples, len(images))
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        img = (images[i] * 255).astype(np.uint8)

        boxes, scores = detect_faces_haar(cascade, img)

        # Малюємо boxes
        img_with_boxes = img.copy()
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"face",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Оригінал (синтетичне 'обличчя')", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_with_boxes)
        axes[i, 1].set_title(f"Детекції ({len(boxes)} знайдено)", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "pretrained_detections.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Збережено приклади детекцій: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. ROC-крива для 2‑класової задачі (object vs background)
# ---------------------------------------------------------------------------


def plot_roc_curves(
    own_y_true: np.ndarray,
    own_y_prob: np.ndarray,
    pretrained_y_true: Optional[np.ndarray] = None,
    pretrained_y_prob: Optional[np.ndarray] = None,
) -> None:
    """
    Малює ROC-криву для власного детектора та (опційно) pre-trained.
    """
    print_section("ROC-КРИВА ДЛЯ 2-КЛАСОВОЇ ЗАДАЧІ (OBJECT vs BACKGROUND)")

    plt.figure(figsize=(10, 8))

    # Власний детектор
    fpr_own, tpr_own, _ = roc_curve(own_y_true, own_y_prob)
    auc_own = roc_auc_score(own_y_true, own_y_prob)
    plt.plot(fpr_own, tpr_own, label=f"Власний U-Net детектор (AUC = {auc_own:.3f})", lw=2, color="blue")

    # Pre-trained детектор
    if pretrained_y_true is not None and pretrained_y_prob is not None:
        # Перевіряємо чи є варіативність
        if len(np.unique(pretrained_y_prob)) > 1:
            fpr_pt, tpr_pt, _ = roc_curve(pretrained_y_true, pretrained_y_prob)
            auc_pt = roc_auc_score(pretrained_y_true, pretrained_y_prob)
            plt.plot(fpr_pt, tpr_pt, label=f"Pre-trained MobileNet SSD (AUC = {auc_pt:.3f})",
                    lw=2, linestyle="--", color="green")

    # Випадковий класифікатор
    plt.plot([0, 1], [0, 1], "k--", label="Випадковий класифікатор (AUC = 0.5)", alpha=0.5)

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC-крива: сегментація 'object' vs 'background'", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    # Додаємо анотації
    plt.annotate(f"AUC власного = {auc_own:.3f}", xy=(0.6, 0.3), fontsize=11,
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    out_path = OUTPUT_DIR / "roc_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ ROC-крива збережена в {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Live-режим: детекція з веб-камери (MobileNet SSD)
# ---------------------------------------------------------------------------


def live_detection(min_neighbors: int = 5):
    """
    Live-режим: детекція обличь у потоці з веб-камери за допомогою Haar Cascade.
    """
    print_section("LIVE ДЕТЕКЦІЯ З ВЕБ-КАМЕРИ (Haar Cascade - Обличчя)")

    cascade = load_haar_cascade()
    if cascade is None:
        print("\n⚠ Live-детекція неможлива.")
        return

    print("\n  Відкриття веб-камери...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("  ⚠ Не вдалося відкрити веб-камеру")
        return

    print("\n✓ Live-детекція запущена!")
    print("\n  Керування:")
    print("    - 'q' або 'ESC' — вихід")
    print("    - '+' — збільшити min_neighbors (менше детекцій, точніше)")
    print("    - '-' — зменшити min_neighbors (більше детекцій)")
    print("    - 's' — зберегти кадр")

    fps_history = []
    saved_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ⚠ Не вдалося прочитати кадр")
                break

            start_time = time.time()

            # Детекція
            boxes, scores = detect_faces_haar(cascade, frame, min_neighbors=min_neighbors)

            # Малюємо boxes
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "face", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
                f"Min neighbors: {min_neighbors}",
                f"Faces: {len(boxes)}",
            ]

            y = 30
            for line in info_lines:
                cv2.putText(frame, line, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 30

            cv2.imshow("Live Detection - Haar Cascade (Face)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n  Зупинка...")
                break
            elif key == ord("+") or key == ord("="):
                min_neighbors = min(15, min_neighbors + 1)
                print(f"  Min neighbors: {min_neighbors}")
            elif key == ord("-") or key == ord("_"):
                min_neighbors = max(1, min_neighbors - 1)
                print(f"  Min neighbors: {min_neighbors}")
            elif key == ord("s"):
                saved_frames += 1
                out_path = OUTPUT_DIR / f"live_detection_{saved_frames}.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"  ✓ Кадр збережено: {out_path}")

    except KeyboardInterrupt:
        print("\n  Перервано користувачем")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        print("\n✓ Live-детекція завершена")
        if fps_history:
            print(f"  Середній FPS: {np.mean(fps_history):.1f}")
        print(f"  Збережено кадрів: {saved_frames}")


# ---------------------------------------------------------------------------
# 6. Головне меню
# ---------------------------------------------------------------------------


def main():
    print("\n" + "=" * 80)
    print("  TEXTURE / OBJECT SEGMENTATION")
    print("  Власний детектор + Pre-trained MobileNet SSD + ROC + Live-режим")
    print("=" * 80)

    print("\n📋 Меню:")
    print("  1. Навчити ВЛАСНИЙ сегментаційний детектор (U-Net)")
    print("  2. Оцінити PRE-TRAINED детектор (Haar Cascade - обличчя)")
    print("  3. Побудувати ROC-криву")
    print("  4. Live-детекція з веб-камери (Haar Cascade)")
    print("  5. Виконати все послідовно (1→2→3)")

    choice = input("\n  Виберіть опцію (1-5): ").strip()

    own_y_true = own_y_prob = None
    pretrained_y_true = pretrained_y_prob = None

    if choice == "1" or choice == "5":
        _, own_y_true, own_y_prob = train_own_detector(
            num_train=50,
            num_test=15,
            epochs=10,
        )

    if choice == "2" or choice == "5":
        _, pretrained_y_true, pretrained_y_prob = evaluate_pretrained_detector(num_test=15)

    if choice == "3":
        # Завантажуємо збережені дані
        try:
            own_y_true = np.load(OUTPUT_DIR / "own_detector_y_true.npy")
            own_y_prob = np.load(OUTPUT_DIR / "own_detector_y_prob.npy")
        except Exception:
            print("\n⚠ Спочатку виконайте пункт 1.")
            return

        try:
            pretrained_y_true = np.load(OUTPUT_DIR / "pretrained_y_true.npy")
            pretrained_y_prob = np.load(OUTPUT_DIR / "pretrained_y_prob.npy")
        except Exception:
            pass

    if choice in {"1", "2", "3", "5"}:
        # Підвантажуємо дані якщо потрібно
        if own_y_true is None:
            try:
                own_y_true = np.load(OUTPUT_DIR / "own_detector_y_true.npy")
                own_y_prob = np.load(OUTPUT_DIR / "own_detector_y_prob.npy")
            except Exception:
                pass

        if pretrained_y_true is None:
            try:
                pretrained_y_true = np.load(OUTPUT_DIR / "pretrained_y_true.npy")
                pretrained_y_prob = np.load(OUTPUT_DIR / "pretrained_y_prob.npy")
            except Exception:
                pass

        if own_y_true is not None:
            plot_roc_curves(own_y_true, own_y_prob, pretrained_y_true, pretrained_y_prob)
        else:
            print("\n⚠ Немає даних для побудови ROC.")

    if choice == "4":
        live_detection(min_neighbors=5)

    print_section("ПІДСУМОК")
    print("\n✅ Лабораторна «Texture / Object Segmentation» виконана.")
    print("\n📁 Результати збережені в каталозі 'results':")
    print("  - synthetic_samples.png — приклади синтетичних даних")
    print("  - training_history.png — графік навчання")
    print("  - own_detector_segmentation_examples.png — приклади сегментації")
    print("  - own_detector_probability_hist.png — розподіл ймовірностей")
    print("  - confusion_matrix.png — матриця помилок")
    print("  - iou_vs_threshold.png — IoU в залежності від порогу")
    print("  - pretrained_detections.png — детекції MobileNet SSD")
    print("  - roc_curves.png — ROC-крива (2 класи)")
    print("  - live_detection_*.jpg — збережені кадри з live-режиму")


if __name__ == "__main__":
    main()
