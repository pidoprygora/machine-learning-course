"""
Feature Visualization - Візуалізація фільтрів згортки та шарів нейронної мережі
Лабораторна робота на основі 05-Texture-Segmentation

Що реалізовано:
1. Візуалізація фільтрів згортки (Convolution Filters) - 50%
   - Відображення ваг фільтрів кожного згорткового шару
   - Аналіз патернів, які шукають фільтри на різних рівнях мережі
   
2. Візуалізація шарів (Layer Activations) - 50%
   - Відображення активацій (feature maps) на кожному шарі
   - Аналіз того, як мережа "бачить" зображення на різних глибинах
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

# Налаштування
tf.random.set_seed(42)
np.random.seed(42)

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10

# Директорії
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def print_section(title: str) -> None:
    """Красивий заголовок секції в консолі."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# 1. Побудова та завантаження моделі (U-Net з 05-Texture-Segmentation)
# ---------------------------------------------------------------------------

IMG_SIZE = (256, 256)


def build_simple_unet(input_shape=(256, 256, 3)) -> keras.Model:
    """
    Невелика U-Net‑подібна модель для бінарної сегментації.
    Така сама архітектура як у 05-Texture-Segmentation.
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu", name="enc_conv1_1")(inputs)
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu", name="enc_conv1_2")(x1)
    p1 = layers.MaxPool2D(2, name="enc_pool1")(x1)

    x2 = layers.Conv2D(64, 3, padding="same", activation="relu", name="enc_conv2_1")(p1)
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu", name="enc_conv2_2")(x2)
    p2 = layers.MaxPool2D(2, name="enc_pool2")(x2)

    x3 = layers.Conv2D(128, 3, padding="same", activation="relu", name="bottleneck_conv1")(p2)
    x3 = layers.Conv2D(128, 3, padding="same", activation="relu", name="bottleneck_conv2")(x3)

    # Decoder
    u2 = layers.UpSampling2D(2, name="dec_upsample1")(x3)
    u2 = layers.Concatenate(name="dec_concat1")([u2, x2])
    x4 = layers.Conv2D(64, 3, padding="same", activation="relu", name="dec_conv1_1")(u2)
    x4 = layers.Conv2D(64, 3, padding="same", activation="relu", name="dec_conv1_2")(x4)

    u1 = layers.UpSampling2D(2, name="dec_upsample2")(x4)
    u1 = layers.Concatenate(name="dec_concat2")([u1, x1])
    x5 = layers.Conv2D(32, 3, padding="same", activation="relu", name="dec_conv2_1")(u1)
    x5 = layers.Conv2D(32, 3, padding="same", activation="relu", name="dec_conv2_2")(x5)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output_conv")(x5)

    model = keras.Model(inputs, outputs, name="simple_unet_object")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _generate_synthetic_image_and_mask() -> Tuple[np.ndarray, np.ndarray]:
    """Генерує просте синтетичне зображення та маску."""
    h, w = IMG_SIZE

    background = np.random.uniform(0.0, 0.4, size=(h, w, 1)).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=0.05, size=(h, w, 1)).astype(np.float32)
    img = background + noise
    img = np.clip(img, 0.0, 1.0)
    img = np.repeat(img, 3, axis=-1)

    mask = np.zeros((h, w, 1), dtype=np.float32)

    shape_type = np.random.choice(["circle", "square"])
    cy = np.random.randint(h // 4, 3 * h // 4)
    cx = np.random.randint(w // 4, 3 * w // 4)
    r = np.random.randint(min(h, w) // 8, min(h, w) // 4)

    yy, xx = np.ogrid[:h, :w]

    if shape_type == "circle":
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        obj_region = dist_sq <= r ** 2
    else:
        y_min = max(cy - r, 0)
        y_max = min(cy + r, h)
        x_min = max(cx - r, 0)
        x_max = min(cx + r, w)
        obj_region = np.zeros((h, w), dtype=bool)
        obj_region[y_min:y_max, x_min:x_max] = True

    mask[obj_region, 0] = 1.0

    color = np.random.uniform(0.6, 1.0, size=(1, 1, 3)).astype(np.float32)
    img[obj_region] = color

    img += np.random.normal(0.0, 0.03, size=img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32), mask.astype(np.float32)


def train_model_for_visualization(epochs: int = 5) -> keras.Model:
    """Навчає модель для подальшої візуалізації."""
    print_section("НАВЧАННЯ МОДЕЛІ ДЛЯ ВІЗУАЛІЗАЦІЇ")
    
    # Генеруємо дані
    num_train = 40
    num_val = 10
    
    train_images = np.zeros((num_train, *IMG_SIZE, 3), dtype=np.float32)
    train_masks = np.zeros((num_train, *IMG_SIZE, 1), dtype=np.float32)
    
    for i in range(num_train):
        train_images[i], train_masks[i] = _generate_synthetic_image_and_mask()
    
    val_images = np.zeros((num_val, *IMG_SIZE, 3), dtype=np.float32)
    val_masks = np.zeros((num_val, *IMG_SIZE, 1), dtype=np.float32)
    
    for i in range(num_val):
        val_images[i], val_masks[i] = _generate_synthetic_image_and_mask()
    
    print(f"  Згенеровано train: {num_train}, val: {num_val}")
    
    # Створюємо та навчаємо модель
    model = build_simple_unet()
    model.summary()
    
    print("\n  Навчання моделі...")
    history = model.fit(
        train_images, train_masks,
        validation_data=(val_images, val_masks),
        epochs=epochs,
        batch_size=4,
        verbose=1
    )
    
    # Зберігаємо модель
    model_path = MODELS_DIR / "unet_for_visualization.h5"
    model.save(model_path)
    print(f"\n✓ Модель збережена: {model_path}")
    
    return model


# ---------------------------------------------------------------------------
# 2. ВІЗУАЛІЗАЦІЯ ФІЛЬТРІВ ЗГОРТКИ (50%)
# ---------------------------------------------------------------------------

def get_conv_layers(model: keras.Model) -> List[Tuple[str, layers.Conv2D]]:
    """Повертає список згорткових шарів моделі."""
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            conv_layers.append((layer.name, layer))
    return conv_layers


def visualize_filters(model: keras.Model, max_filters: int = 64) -> None:
    """
    Візуалізує фільтри (ваги) згорткових шарів.
    
    Фільтри показують, які патерни шукає кожен шар:
    - Перші шари: прості краї, градієнти, кольори
    - Глибші шари: складніші текстури та форми
    """
    print_section("ВІЗУАЛІЗАЦІЯ ФІЛЬТРІВ ЗГОРТКИ")
    
    conv_layers = get_conv_layers(model)
    print(f"\n  Знайдено {len(conv_layers)} згорткових шарів:")
    
    for name, layer in conv_layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            filters = weights[0]
            print(f"    {name}: форма фільтрів {filters.shape}")
    
    # Візуалізація фільтрів кожного згорткового шару
    for layer_name, layer in conv_layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
            
        filters = weights[0]  # Ваги фільтрів: (height, width, in_channels, out_channels)
        
        # Нормалізуємо фільтри для відображення
        f_min, f_max = filters.min(), filters.max()
        filters_normalized = (filters - f_min) / (f_max - f_min + 1e-8)
        
        n_filters = min(filters.shape[-1], max_filters)
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2 * n_rows))
        fig.suptitle(f"Фільтри шару: {layer_name}\n"
                     f"Форма: {filters.shape} (H×W×In×Out)", 
                     fontsize=14, fontweight='bold')
        
        # Правильна обробка axes для різних випадків
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i in range(n_filters):
            ax = axes[i]
            
            # Для кольорових фільтрів (3 вхідних канали) показуємо як RGB
            if filters.shape[2] == 3:
                f = filters_normalized[:, :, :, i]
                ax.imshow(f)
            else:
                # Для інших - показуємо середнє по вхідних каналах
                f = filters_normalized[:, :, :, i].mean(axis=2)
                ax.imshow(f, cmap='viridis')
            
            ax.set_title(f"F{i}", fontsize=8)
            ax.axis('off')
        
        # Приховуємо пусті subplot'и
        for i in range(n_filters, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"filters_{layer_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Збережено: {out_path}")


def visualize_filters_comparison(model: keras.Model) -> None:
    """
    Порівняння фільтрів різних шарів на одному графіку.
    Показує еволюцію складності патернів від простих до складних.
    """
    print_section("ПОРІВНЯННЯ ФІЛЬТРІВ НА РІЗНИХ ГЛИБИНАХ")
    
    conv_layers = get_conv_layers(model)
    
    # Вибираємо репрезентативні шари
    selected_layers = conv_layers[:min(6, len(conv_layers))]
    
    fig, axes = plt.subplots(len(selected_layers), 8, figsize=(18, 3 * len(selected_layers)))
    fig.suptitle("Еволюція фільтрів: від простих до складних патернів\n"
                 "(перші 8 фільтрів кожного шару)", fontsize=14, fontweight='bold')
    
    for row_idx, (layer_name, layer) in enumerate(selected_layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
            
        filters = weights[0]
        f_min, f_max = filters.min(), filters.max()
        filters_normalized = (filters - f_min) / (f_max - f_min + 1e-8)
        
        n_show = min(8, filters.shape[-1])
        
        for col_idx in range(8):
            ax = axes[row_idx, col_idx] if len(selected_layers) > 1 else axes[col_idx]
            
            if col_idx < n_show:
                if filters.shape[2] == 3:
                    f = filters_normalized[:, :, :, col_idx]
                    ax.imshow(f)
                else:
                    f = filters_normalized[:, :, :, col_idx].mean(axis=2)
                    ax.imshow(f, cmap='viridis')
            
            ax.axis('off')
            
            if col_idx == 0:
                ax.set_ylabel(f"{layer_name}\n({filters.shape[-1]} фільтрів)", 
                            fontsize=9, rotation=0, ha='right', va='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "filters_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Порівняння фільтрів збережено: {out_path}")


def visualize_filter_statistics(model: keras.Model) -> None:
    """
    Статистичний аналіз фільтрів: розподіл ваг, норми фільтрів.
    """
    print_section("СТАТИСТИКА ФІЛЬТРІВ ЗГОРТКИ")
    
    conv_layers = get_conv_layers(model)
    
    layer_names = []
    weight_means = []
    weight_stds = []
    filter_norms = []
    
    for layer_name, layer in conv_layers:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
            
        filters = weights[0]
        
        layer_names.append(layer_name)
        weight_means.append(filters.mean())
        weight_stds.append(filters.std())
        
        # Норма Фробеніуса для кожного фільтра
        norms = []
        for i in range(filters.shape[-1]):
            norm = np.linalg.norm(filters[:, :, :, i])
            norms.append(norm)
        filter_norms.append(np.mean(norms))
    
    # Графіки
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Статистичний аналіз фільтрів згортки", fontsize=14, fontweight='bold')
    
    # 1. Середнє значення ваг
    axes[0, 0].bar(range(len(layer_names)), weight_means, color='steelblue', alpha=0.8)
    axes[0, 0].set_xticks(range(len(layer_names)))
    axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylabel("Середнє ваг")
    axes[0, 0].set_title("Середнє значення ваг по шарах")
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Стандартне відхилення ваг
    axes[0, 1].bar(range(len(layer_names)), weight_stds, color='orange', alpha=0.8)
    axes[0, 1].set_xticks(range(len(layer_names)))
    axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_ylabel("Std ваг")
    axes[0, 1].set_title("Стандартне відхилення ваг")
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Середня норма фільтрів
    axes[1, 0].bar(range(len(layer_names)), filter_norms, color='green', alpha=0.8)
    axes[1, 0].set_xticks(range(len(layer_names)))
    axes[1, 0].set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel("Норма Фробеніуса")
    axes[1, 0].set_title("Середня норма фільтрів")
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Гістограма ваг всіх фільтрів
    all_weights = []
    for layer_name, layer in conv_layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            all_weights.extend(weights[0].flatten())
    
    axes[1, 1].hist(all_weights, bins=100, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel("Значення ваги")
    axes[1, 1].set_ylabel("Частота")
    axes[1, 1].set_title("Розподіл всіх ваг фільтрів")
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Нуль')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "filter_statistics.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Статистика фільтрів збережена: {out_path}")


# ---------------------------------------------------------------------------
# 3. ВІЗУАЛІЗАЦІЯ ШАРІВ / АКТИВАЦІЙ (50%)
# ---------------------------------------------------------------------------

def create_activation_model(model: keras.Model) -> keras.Model:
    """
    Створює модель для отримання активацій всіх проміжних шарів.
    """
    layer_outputs = [layer.output for layer in model.layers if 'input' not in layer.name.lower()]
    layer_names = [layer.name for layer in model.layers if 'input' not in layer.name.lower()]
    
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    return activation_model, layer_names


def get_sample_image() -> np.ndarray:
    """Отримує приклад зображення для візуалізації активацій."""
    img, _ = _generate_synthetic_image_and_mask()
    return img


def visualize_layer_activations(model: keras.Model, image: Optional[np.ndarray] = None) -> None:
    """
    Візуалізує активації (feature maps) на кожному шарі мережі.
    
    Показує, як мережа "бачить" зображення на різних глибинах:
    - Перші шари: низькорівневі ознаки (краї, кольори)
    - Глибші шари: високорівневі абстракції
    """
    print_section("ВІЗУАЛІЗАЦІЯ АКТИВАЦІЙ ШАРІВ")
    
    if image is None:
        image = get_sample_image()
    
    # Створюємо модель активацій
    activation_model, layer_names = create_activation_model(model)
    
    # Отримуємо активації
    input_image = np.expand_dims(image, axis=0)
    activations = activation_model.predict(input_image, verbose=0)
    
    print(f"\n  Аналіз {len(activations)} шарів:")
    
    # Зберігаємо вхідне зображення
    fig_input, ax_input = plt.subplots(1, 1, figsize=(8, 8))
    ax_input.imshow(image)
    ax_input.set_title("Вхідне зображення", fontsize=14, fontweight='bold')
    ax_input.axis('off')
    plt.tight_layout()
    out_path = OUTPUT_DIR / "input_image_for_activations.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig_input)
    print(f"  ✓ Вхідне зображення: {out_path}")
    
    # Візуалізуємо активації кожного шару
    for idx, (activation, layer_name) in enumerate(zip(activations, layer_names)):
        if len(activation.shape) != 4:  # Пропускаємо не 4D тензори
            continue
            
        n_features = activation.shape[-1]
        
        # Показуємо максимум 64 feature maps
        n_show = min(64, n_features)
        n_cols = 8
        n_rows = (n_show + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2 * n_rows))
        fig.suptitle(f"Активації шару: {layer_name}\n"
                     f"Форма виходу: {activation.shape[1:]} (H×W×Channels)", 
                     fontsize=12, fontweight='bold')
        
        # Правильна обробка axes для різних випадків
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i in range(n_show):
            ax = axes[i]
            feature_map = activation[0, :, :, i]
            
            # Нормалізуємо для відображення
            if feature_map.max() != feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f"Ch{i}", fontsize=7)
            ax.axis('off')
        
        for i in range(n_show, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"activations_{idx:02d}_{layer_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    {layer_name}: {activation.shape[1:]} -> {out_path.name}")


def visualize_activation_heatmaps(model: keras.Model, image: Optional[np.ndarray] = None) -> None:
    """
    Створює теплові карти середньої активації по каналах для кожного шару.
    Показує, які області зображення найбільше активують кожен шар.
    """
    print_section("ТЕПЛОВІ КАРТИ АКТИВАЦІЙ")
    
    if image is None:
        image = get_sample_image()
    
    activation_model, layer_names = create_activation_model(model)
    input_image = np.expand_dims(image, axis=0)
    activations = activation_model.predict(input_image, verbose=0)
    
    # Вибираємо тільки згорткові шари
    conv_activations = []
    conv_names = []
    
    for activation, name in zip(activations, layer_names):
        if len(activation.shape) == 4 and 'conv' in name.lower():
            conv_activations.append(activation)
            conv_names.append(name)
    
    n_layers = len(conv_activations)
    if n_layers == 0:
        print("  Немає згорткових шарів для візуалізації")
        return
    
    # Створюємо сітку
    n_cols = 4
    n_rows = (n_layers + 1 + n_cols - 1) // n_cols  # +1 для оригінального зображення
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle("Теплові карти: середня активація по каналах\n"
                 "(яскравіші області = сильніша активація)", fontsize=14, fontweight='bold')
    
    # Правильна обробка axes
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Оригінальне зображення
    axes[0].imshow(image)
    axes[0].set_title("Оригінал", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Теплові карти для кожного шару
    for idx, (activation, name) in enumerate(zip(conv_activations, conv_names)):
        ax = axes[idx + 1]
        
        # Середня активація по всіх каналах
        heatmap = activation[0].mean(axis=-1)
        
        # Нормалізація
        if heatmap.max() != heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Масштабуємо до розміру оригінального зображення
        import cv2
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Накладаємо на оригінальне зображення
        ax.imshow(image, alpha=0.6)
        im = ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
        ax.set_title(f"{name}\n{activation.shape[1:3]}", fontsize=9)
        ax.axis('off')
    
    # Приховуємо зайві axes
    for i in range(len(conv_activations) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "activation_heatmaps.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Теплові карти збережені: {out_path}")


def visualize_activation_statistics(model: keras.Model, image: Optional[np.ndarray] = None) -> None:
    """
    Статистичний аналіз активацій: розподіл значень, спарсність.
    """
    print_section("СТАТИСТИКА АКТИВАЦІЙ ШАРІВ")
    
    if image is None:
        image = get_sample_image()
    
    activation_model, layer_names = create_activation_model(model)
    input_image = np.expand_dims(image, axis=0)
    activations = activation_model.predict(input_image, verbose=0)
    
    # Збираємо статистику
    layer_stats = []
    
    for activation, name in zip(activations, layer_names):
        if len(activation.shape) == 4:
            stats = {
                'name': name,
                'shape': str(activation.shape[1:]),
                'mean': activation.mean(),
                'std': activation.std(),
                'min': activation.min(),
                'max': activation.max(),
                'sparsity': (activation == 0).sum() / activation.size * 100,  # % нулів
                'dead_channels': (activation.mean(axis=(0, 1, 2)) == 0).sum()
            }
            layer_stats.append(stats)
    
    # Візуалізація
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Статистичний аналіз активацій шарів", fontsize=14, fontweight='bold')
    
    names = [s['name'] for s in layer_stats]
    means = [s['mean'] for s in layer_stats]
    stds = [s['std'] for s in layer_stats]
    sparsities = [s['sparsity'] for s in layer_stats]
    maxs = [s['max'] for s in layer_stats]
    
    x = range(len(names))
    
    # 1. Середнє значення активацій
    axes[0, 0].bar(x, means, color='steelblue', alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0, 0].set_ylabel("Середнє")
    axes[0, 0].set_title("Середнє значення активацій")
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Максимальне значення
    axes[0, 1].bar(x, maxs, color='orange', alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[0, 1].set_ylabel("Максимум")
    axes[0, 1].set_title("Максимальне значення активацій")
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Стандартне відхилення
    axes[1, 0].bar(x, stds, color='green', alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[1, 0].set_ylabel("Std")
    axes[1, 0].set_title("Стандартне відхилення активацій")
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Спарсність (% нульових активацій)
    axes[1, 1].bar(x, sparsities, color='red', alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    axes[1, 1].set_ylabel("% нульових")
    axes[1, 1].set_title("Спарсність активацій (ReLU ефект)")
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "activation_statistics.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Статистика активацій збережена: {out_path}")
    
    # Виводимо таблицю
    print("\n📊 Детальна статистика по шарах:")
    print("-" * 90)
    print(f"{'Шар':<25} {'Форма':<18} {'Mean':>8} {'Std':>8} {'Max':>8} {'Sparsity':>10}")
    print("-" * 90)
    for s in layer_stats:
        print(f"{s['name']:<25} {s['shape']:<18} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{s['max']:>8.4f} {s['sparsity']:>9.2f}%")


def visualize_layer_progression(model: keras.Model, image: Optional[np.ndarray] = None) -> None:
    """
    Показує прогресію обробки зображення через всі шари мережі.
    """
    print_section("ПРОГРЕСІЯ ОБРОБКИ ЧЕРЕЗ ШАРИ")
    
    if image is None:
        image = get_sample_image()
    
    activation_model, layer_names = create_activation_model(model)
    input_image = np.expand_dims(image, axis=0)
    activations = activation_model.predict(input_image, verbose=0)
    
    # Вибираємо ключові шари для відображення
    key_activations = []
    key_names = []
    
    for activation, name in zip(activations, layer_names):
        if len(activation.shape) == 4:
            key_activations.append(activation)
            key_names.append(name)
    
    # Показуємо до 12 шарів
    n_show = min(12, len(key_activations))
    step = max(1, len(key_activations) // n_show)
    
    selected_idx = list(range(0, len(key_activations), step))[:n_show]
    
    n_cols = 4
    n_rows = (len(selected_idx) + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle("Прогресія обробки: від вхідного зображення до виходу\n"
                 "(перший feature map кожного шару)", fontsize=14, fontweight='bold')
    
    # Правильна обробка axes
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Вхідне зображення
    axes[0].imshow(image)
    axes[0].set_title("Вхід\n256×256×3", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Активації
    for plot_idx, act_idx in enumerate(selected_idx):
        ax = axes[plot_idx + 1]
        activation = key_activations[act_idx]
        name = key_names[act_idx]
        
        # Показуємо перший feature map
        feature_map = activation[0, :, :, 0]
        
        # Нормалізація
        if feature_map.max() != feature_map.min():
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f"{name}\n{activation.shape[1:-1]}", fontsize=9)
        ax.axis('off')
    
    for i in range(len(selected_idx) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "layer_progression.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Прогресія через шари збережена: {out_path}")


# ---------------------------------------------------------------------------
# 4. ВІЗУАЛІЗАЦІЯ PRE-TRAINED МОДЕЛІ (VGG16)
# ---------------------------------------------------------------------------

def visualize_pretrained_filters(max_filters: int = 64) -> None:
    """
    Візуалізує фільтри попередньо навченої моделі VGG16.
    Показує, яких патернів навчилася модель на ImageNet.
    """
    print_section("ФІЛЬТРИ PRE-TRAINED МОДЕЛІ (VGG16)")
    
    print("  Завантаження VGG16...")
    vgg = VGG16(weights='imagenet', include_top=False)
    
    # Отримуємо згорткові шари
    conv_layers = [(layer.name, layer) for layer in vgg.layers 
                   if isinstance(layer, layers.Conv2D)]
    
    print(f"  Знайдено {len(conv_layers)} згорткових шарів")
    
    # Візуалізуємо перші та останні шари
    layers_to_show = [conv_layers[0], conv_layers[len(conv_layers)//2], conv_layers[-1]]
    
    for layer_name, layer in layers_to_show:
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
            
        filters = weights[0]
        f_min, f_max = filters.min(), filters.max()
        filters_normalized = (filters - f_min) / (f_max - f_min + 1e-8)
        
        n_filters = min(filters.shape[-1], max_filters)
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2 * n_rows))
        fig.suptitle(f"VGG16: Фільтри шару {layer_name}\n"
                     f"Форма: {filters.shape}", fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
        for i in range(n_filters):
            ax = axes[i]
            
            if filters.shape[2] == 3:
                f = filters_normalized[:, :, :, i]
                ax.imshow(f)
            else:
                f = filters_normalized[:, :, :, i].mean(axis=2)
                ax.imshow(f, cmap='viridis')
            
            ax.axis('off')
        
        for i in range(n_filters, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"vgg16_filters_{layer_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Збережено: {out_path}")


def visualize_pretrained_activations(image: Optional[np.ndarray] = None) -> None:
    """
    Візуалізує активації VGG16 на прикладі зображення.
    """
    print_section("АКТИВАЦІЇ PRE-TRAINED МОДЕЛІ (VGG16)")
    
    if image is None:
        image = get_sample_image()
    
    # Масштабуємо зображення до 224x224 (вхід VGG16)
    import cv2
    image_resized = cv2.resize(image, (224, 224))
    
    print("  Завантаження VGG16...")
    vgg = VGG16(weights='imagenet', include_top=False)
    
    # Створюємо модель активацій
    layer_outputs = [layer.output for layer in vgg.layers if 'conv' in layer.name]
    layer_names = [layer.name for layer in vgg.layers if 'conv' in layer.name]
    
    activation_model = Model(inputs=vgg.input, outputs=layer_outputs)
    
    # Препроцесинг для VGG
    from tensorflow.keras.applications.vgg16 import preprocess_input
    input_image = np.expand_dims(image_resized * 255, axis=0)
    input_image = preprocess_input(input_image)
    
    activations = activation_model.predict(input_image, verbose=0)
    
    # Вибираємо ключові шари
    key_indices = [0, len(activations)//3, 2*len(activations)//3, len(activations)-1]
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("VGG16: Активації на різних глибинах мережі\n"
                 "(ImageNet pre-trained)", fontsize=14, fontweight='bold')
    
    # Перший ряд - оригінал та теплові карти
    axes[0, 0].imshow(image_resized)
    axes[0, 0].set_title("Вхід (224×224)", fontsize=10)
    axes[0, 0].axis('off')
    
    for i, idx in enumerate(key_indices[:-1]):
        activation = activations[idx]
        heatmap = activation[0].mean(axis=-1)
        
        if heatmap.max() != heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        axes[0, i+1].imshow(image_resized, alpha=0.6)
        axes[0, i+1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        axes[0, i+1].set_title(f"{layer_names[idx]}\n{activation.shape[1:3]}", fontsize=9)
        axes[0, i+1].axis('off')
    
    # Другий ряд - окремі feature maps
    for i, idx in enumerate(key_indices):
        activation = activations[idx]
        # Показуємо мозаїку з перших 9 feature maps
        n_show = min(9, activation.shape[-1])
        
        # Створюємо мозаїку 3x3
        grid_size = 3
        mosaic = np.zeros((activation.shape[1] * grid_size, activation.shape[2] * grid_size))
        
        for j in range(n_show):
            row = j // grid_size
            col = j % grid_size
            fm = activation[0, :, :, j]
            if fm.max() != fm.min():
                fm = (fm - fm.min()) / (fm.max() - fm.min())
            mosaic[row*activation.shape[1]:(row+1)*activation.shape[1],
                   col*activation.shape[2]:(col+1)*activation.shape[2]] = fm
        
        axes[1, i].imshow(mosaic, cmap='viridis')
        axes[1, i].set_title(f"{layer_names[idx]}\n{activation.shape[-1]} каналів", fontsize=9)
        axes[1, i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / "vgg16_activations.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Активації VGG16 збережені: {out_path}")


# ---------------------------------------------------------------------------
# 5. ГОЛОВНЕ МЕНЮ
# ---------------------------------------------------------------------------

def create_summary_report(model: keras.Model) -> None:
    """Створює підсумковий звіт про візуалізацію."""
    print_section("ПІДСУМКОВИЙ ЗВІТ")
    
    conv_layers = get_conv_layers(model)
    
    total_params = model.count_params()
    conv_params = sum(np.prod(layer.get_weights()[0].shape) + 
                      (layer.get_weights()[1].shape[0] if len(layer.get_weights()) > 1 else 0)
                      for _, layer in conv_layers if len(layer.get_weights()) > 0)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FEATURE VISUALIZATION - ПІДСУМОК                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  МОДЕЛЬ: U-Net для бінарної сегментації                                      ║
║  Загальна кількість параметрів: {total_params:,}                             
║  Параметри згорткових шарів: {conv_params:,}                                 
║  Кількість згорткових шарів: {len(conv_layers)}                              
╠══════════════════════════════════════════════════════════════════════════════╣
║  ВІЗУАЛІЗАЦІЯ ФІЛЬТРІВ (50%):                                                ║
║  ✓ Відображення ваг фільтрів кожного шару                                    ║
║  ✓ Порівняння патернів на різних глибинах                                    ║
║  ✓ Статистичний аналіз ваг                                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ВІЗУАЛІЗАЦІЯ ШАРІВ (50%):                                                   ║
║  ✓ Feature maps для кожного шару                                             ║
║  ✓ Теплові карти активацій                                                   ║
║  ✓ Статистика активацій (спарсність, розподіл)                               ║
║  ✓ Прогресія обробки через мережу                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ДОДАТКОВО:                                                                  ║
║  ✓ Візуалізація pre-trained VGG16                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📁 Збережені результати в 'results/':")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    - {f.name}")


def main():
    print("\n" + "=" * 80)
    print("  FEATURE VISUALIZATION")
    print("  Візуалізація фільтрів згортки та шарів нейронної мережі")
    print("=" * 80)
    
    print("\n📋 Меню:")
    print("  1. Візуалізація ФІЛЬТРІВ згортки (власна U-Net)")
    print("  2. Візуалізація ШАРІВ / активацій (власна U-Net)")
    print("  3. Візуалізація pre-trained VGG16")
    print("  4. Виконати все послідовно (повна лабораторна)")
    
    choice = input("\n  Виберіть опцію (1-4): ").strip()
    
    model = None
    
    # Для варіантів 1, 2, 4 потрібна навчена модель
    if choice in {"1", "2", "4"}:
        model_path = MODELS_DIR / "unet_for_visualization.h5"
        
        if model_path.exists():
            print(f"\n  Завантаження моделі з {model_path}...")
            model = keras.models.load_model(model_path)
        else:
            model = train_model_for_visualization(epochs=5)
    
    # Генеруємо тестове зображення
    test_image = get_sample_image()
    
    if choice == "1" or choice == "4":
        # === ВІЗУАЛІЗАЦІЯ ФІЛЬТРІВ (50%) ===
        visualize_filters(model)
        visualize_filters_comparison(model)
        visualize_filter_statistics(model)
    
    if choice == "2" or choice == "4":
        # === ВІЗУАЛІЗАЦІЯ ШАРІВ (50%) ===
        visualize_layer_activations(model, test_image)
        visualize_activation_heatmaps(model, test_image)
        visualize_activation_statistics(model, test_image)
        visualize_layer_progression(model, test_image)
    
    if choice == "3" or choice == "4":
        # === PRE-TRAINED VGG16 ===
        visualize_pretrained_filters()
        visualize_pretrained_activations(test_image)
    
    if choice in {"1", "2", "4"} and model is not None:
        create_summary_report(model)
    
    print("\n" + "=" * 80)
    print("  ✅ Лабораторна робота завершена!")
    print("=" * 80)


if __name__ == "__main__":
    main()

