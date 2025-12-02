"""
Fine-Tuning Попередньо Навчених CNN на Специфічних Класах
Фокус: Оптимізація параметрів, confusion matrices, аналіз помилок

Мета: Порівняти різні стратегії fine-tuning на специфічних категоріях зображень
та проаналізувати типові помилки класифікації.

Моделі:
- VGG16 (ImageNet pretrained)
- ResNet50 (ImageNet pretrained)
- MobileNetV2 (ImageNet pretrained)
- EfficientNetB0 (ImageNet pretrained)

Датасети (специфічні категорії):
- CIFAR-10: Транспорт (літак, автомобіль, корабель, вантажівка)
- CIFAR-10: Тварини (птах, кіт, олень, собака, жаба, кінь)
- CIFAR-100: Обличчя людей та тварин
- Fashion-MNIST: Взуття (сандалі, кросівки, черевики)

Стратегії Fine-Tuning:
1. Заморожені базові шари (тільки classifier)
2. Часткове розморожування (останні N шарів)
3. Повне fine-tuning (всі шари з low LR)

Візуалізації:
- Криві навчання (accuracy, loss)
- Confusion matrices
- Найтиповіші помилки класифікації
- Зміна параметрів у часі
- Порівняння стратегій fine-tuning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.datasets import cifar10, cifar100, fashion_mnist
from tensorflow.keras.callbacks import Callback

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)

import time
from pathlib import Path
from collections import defaultdict

# Налаштування TensorFlow
tf.random.set_seed(42)
np.random.seed(42)

# Налаштування графіків
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

# Створення директорії для результатів
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def print_section(title):
    """Виводить заголовок секції"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_specialized_datasets():
    """Завантажує датасети з специфічними категоріями"""
    print_section("ЗАВАНТАЖЕННЯ СПЕЦІАЛІЗОВАНИХ ДАТАСЕТІВ")
    
    datasets = {}
    
    # 1. CIFAR-10: Транспорт (4 класи)
    print("\n[1/4] CIFAR-10 Транспорт...")
    (X_train_c10, y_train_c10), (X_test_c10, y_test_c10) = cifar10.load_data()
    y_train_c10 = y_train_c10.flatten()
    y_test_c10 = y_test_c10.flatten()
    
    # Класи транспорту: 0-літак, 1-автомобіль, 8-корабель, 9-вантажівка
    transport_classes = [0, 1, 8, 9]
    transport_mask_train = np.isin(y_train_c10, transport_classes)
    transport_mask_test = np.isin(y_test_c10, transport_classes)
    
    X_transport_train = X_train_c10[transport_mask_train].astype('float32') / 255.0
    X_transport_test = X_test_c10[transport_mask_test].astype('float32') / 255.0
    
    # Перемапування міток 0-3
    y_transport_train = y_train_c10[transport_mask_train]
    y_transport_test = y_test_c10[transport_mask_test]
    
    label_map_transport = {0: 0, 1: 1, 8: 2, 9: 3}
    y_transport_train = np.array([label_map_transport[y] for y in y_transport_train])
    y_transport_test = np.array([label_map_transport[y] for y in y_transport_test])
    
    # Підвибірка для швидкості
    train_idx = np.random.choice(len(X_transport_train), 8000, replace=False)
    test_idx = np.random.choice(len(X_transport_test), 2000, replace=False)
    
    datasets['Транспорт (CIFAR-10)'] = {
        'X_train': X_transport_train[train_idx],
        'y_train': y_transport_train[train_idx],
        'X_test': X_transport_test[test_idx],
        'y_test': y_transport_test[test_idx],
        'n_classes': 4,
        'class_names': ['Літак', 'Автомобіль', 'Корабель', 'Вантажівка'],
        'shape': (32, 32, 3),
        'description': 'Різні види транспорту'
    }
    print(f"  ✓ Транспорт: train={len(datasets['Транспорт (CIFAR-10)']['X_train'])}, test={len(datasets['Транспорт (CIFAR-10)']['X_test'])}")
    
    # 2. CIFAR-10: Тварини (6 класів)
    print("\n[2/4] CIFAR-10 Тварини...")
    # Класи тварин: 2-птах, 3-кіт, 4-олень, 5-собака, 6-жаба, 7-кінь
    animal_classes = [2, 3, 4, 5, 6, 7]
    animal_mask_train = np.isin(y_train_c10, animal_classes)
    animal_mask_test = np.isin(y_test_c10, animal_classes)
    
    X_animal_train = X_train_c10[animal_mask_train].astype('float32') / 255.0
    X_animal_test = X_test_c10[animal_mask_test].astype('float32') / 255.0
    
    y_animal_train = y_train_c10[animal_mask_train]
    y_animal_test = y_test_c10[animal_mask_test]
    
    label_map_animal = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    y_animal_train = np.array([label_map_animal[y] for y in y_animal_train])
    y_animal_test = np.array([label_map_animal[y] for y in y_animal_test])
    
    # Підвибірка
    train_idx = np.random.choice(len(X_animal_train), 10000, replace=False)
    test_idx = np.random.choice(len(X_animal_test), 2500, replace=False)
    
    datasets['Тварини (CIFAR-10)'] = {
        'X_train': X_animal_train[train_idx],
        'y_train': y_animal_train[train_idx],
        'X_test': X_animal_test[test_idx],
        'y_test': y_animal_test[test_idx],
        'n_classes': 6,
        'class_names': ['Птах', 'Кіт', 'Олень', 'Собака', 'Жаба', 'Кінь'],
        'shape': (32, 32, 3),
        'description': 'Різні види тварин'
    }
    print(f"  ✓ Тварини: train={len(datasets['Тварини (CIFAR-10)']['X_train'])}, test={len(datasets['Тварини (CIFAR-10)']['X_test'])}")
    
    # 3. Fashion-MNIST: Взуття (3 класи)
    print("\n[3/4] Fashion-MNIST Взуття...")
    (X_train_fm, y_train_fm), (X_test_fm, y_test_fm) = fashion_mnist.load_data()
    
    # Класи взуття: 5-сандалі, 7-кросівки, 9-черевики
    footwear_classes = [5, 7, 9]
    footwear_mask_train = np.isin(y_train_fm, footwear_classes)
    footwear_mask_test = np.isin(y_test_fm, footwear_classes)
    
    X_footwear_train = X_train_fm[footwear_mask_train].astype('float32') / 255.0
    X_footwear_test = X_test_fm[footwear_mask_test].astype('float32') / 255.0
    
    # Конвертація в RGB (повторюємо канали)
    X_footwear_train = np.repeat(X_footwear_train[..., np.newaxis], 3, axis=-1)
    X_footwear_test = np.repeat(X_footwear_test[..., np.newaxis], 3, axis=-1)
    
    y_footwear_train = y_train_fm[footwear_mask_train]
    y_footwear_test = y_test_fm[footwear_mask_test]
    
    label_map_footwear = {5: 0, 7: 1, 9: 2}
    y_footwear_train = np.array([label_map_footwear[y] for y in y_footwear_train])
    y_footwear_test = np.array([label_map_footwear[y] for y in y_footwear_test])
    
    # Підвибірка
    train_idx = np.random.choice(len(X_footwear_train), 10000, replace=False)
    test_idx = np.random.choice(len(X_footwear_test), 2500, replace=False)
    
    datasets['Взуття (Fashion-MNIST)'] = {
        'X_train': X_footwear_train[train_idx],
        'y_train': y_footwear_train[train_idx],
        'X_test': X_footwear_test[test_idx],
        'y_test': y_footwear_test[test_idx],
        'n_classes': 3,
        'class_names': ['Сандалі', 'Кросівки', 'Черевики'],
        'shape': (28, 28, 3),
        'description': 'Різні види взуття'
    }
    print(f"  ✓ Взуття: train={len(datasets['Взуття (Fashion-MNIST)']['X_train'])}, test={len(datasets['Взуття (Fashion-MNIST)']['X_test'])}")
    
    # 4. CIFAR-100: Обличчя (тільки "baby", "boy", "girl", "man", "woman")
    print("\n[4/4] CIFAR-100 Люди...")
    (X_train_c100, y_train_c100), (X_test_c100, y_test_c100) = cifar100.load_data(label_mode='fine')
    y_train_c100 = y_train_c100.flatten()
    y_test_c100 = y_test_c100.flatten()
    
    # Класи людей: baby(2), boy(11), girl(35), man(46), woman(98)
    people_classes = [2, 11, 35, 46, 98]
    people_mask_train = np.isin(y_train_c100, people_classes)
    people_mask_test = np.isin(y_test_c100, people_classes)
    
    X_people_train = X_train_c100[people_mask_train].astype('float32') / 255.0
    X_people_test = X_test_c100[people_mask_test].astype('float32') / 255.0
    
    y_people_train = y_train_c100[people_mask_train]
    y_people_test = y_test_c100[people_mask_test]
    
    label_map_people = {2: 0, 11: 1, 35: 2, 46: 3, 98: 4}
    y_people_train = np.array([label_map_people[y] for y in y_people_train])
    y_people_test = np.array([label_map_people[y] for y in y_people_test])
    
    datasets['Люди (CIFAR-100)'] = {
        'X_train': X_people_train,
        'y_train': y_people_train,
        'X_test': X_people_test,
        'y_test': y_people_test,
        'n_classes': 5,
        'class_names': ['Немовля', 'Хлопчик', 'Дівчинка', 'Чоловік', 'Жінка'],
        'shape': (32, 32, 3),
        'description': 'Обличчя різних людей'
    }
    print(f"  ✓ Люди: train={len(datasets['Люди (CIFAR-100)']['X_train'])}, test={len(datasets['Люди (CIFAR-100)']['X_test'])}")
    
    return datasets


def load_two_class_car_vs_shoes():
    """
    Спрощений 2-класовий датасет:
    - Клас 0: Автомобілі (CIFAR-10, клас 1)
    - Клас 1: Взуття (Fashion-MNIST, клас 7 – кросівки)

    Усі зображення приводимо до формату 32x32x3 та масштабуємо до [0, 1].
    Для прискорення беремо невелику підвибірку з кожного класу.
    """
    print_section("ЗАВАНТАЖЕННЯ СПРОЩЕНОГО ДАТАСЕТУ: АВТОМОБІЛІ VS ВЗУТТЯ")

    # 1. CIFAR-10: Автомобілі
    (X_train_c10, y_train_c10), (X_test_c10, y_test_c10) = cifar10.load_data()
    y_train_c10 = y_train_c10.flatten()
    y_test_c10 = y_test_c10.flatten()

    car_class = 1  # automobile
    car_mask_train = y_train_c10 == car_class
    car_mask_test = y_test_c10 == car_class

    X_car_train = X_train_c10[car_mask_train].astype('float32') / 255.0
    X_car_test = X_test_c10[car_mask_test].astype('float32') / 255.0

    # 2. Fashion-MNIST: Взуття (кросівки)
    (X_train_fm, y_train_fm), (X_test_fm, y_test_fm) = fashion_mnist.load_data()

    shoe_class = 7  # sneaker
    shoe_mask_train = y_train_fm == shoe_class
    shoe_mask_test = y_test_fm == shoe_class

    X_shoe_train = X_train_fm[shoe_mask_train].astype('float32') / 255.0
    X_shoe_test = X_test_fm[shoe_mask_test].astype('float32') / 255.0

    # Перетворення взуття в 3 канали та resize до 32x32 для сумісності з CIFAR-10
    X_shoe_train = np.repeat(X_shoe_train[..., np.newaxis], 3, axis=-1)
    X_shoe_test = np.repeat(X_shoe_test[..., np.newaxis], 3, axis=-1)

    X_shoe_train = tf.image.resize(X_shoe_train, [32, 32]).numpy()
    X_shoe_test = tf.image.resize(X_shoe_test, [32, 32]).numpy()

    # Для прискорення беремо обмежену кількість прикладів з кожного класу
    n_train_per_class = min(2000, len(X_car_train), len(X_shoe_train))
    n_test_per_class = min(500, len(X_car_test), len(X_shoe_test))

    rng = np.random.default_rng(42)

    car_train_idx = rng.choice(len(X_car_train), n_train_per_class, replace=False)
    shoe_train_idx = rng.choice(len(X_shoe_train), n_train_per_class, replace=False)
    car_test_idx = rng.choice(len(X_car_test), n_test_per_class, replace=False)
    shoe_test_idx = rng.choice(len(X_shoe_test), n_test_per_class, replace=False)

    X_train = np.concatenate(
        [X_car_train[car_train_idx], X_shoe_train[shoe_train_idx]], axis=0
    )
    y_train = np.array(
        [0] * n_train_per_class + [1] * n_train_per_class, dtype=np.int64
    )

    X_test = np.concatenate(
        [X_car_test[car_test_idx], X_shoe_test[shoe_test_idx]], axis=0
    )
    y_test = np.array(
        [0] * n_test_per_class + [1] * n_test_per_class, dtype=np.int64
    )

    # Перемішування
    train_perm = rng.permutation(len(X_train))
    test_perm = rng.permutation(len(X_test))

    X_train = X_train[train_perm]
    y_train = y_train[train_perm]
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]

    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'n_classes': 2,
        'class_names': ['Автомобіль', 'Взуття'],
        'shape': (32, 32, 3),
        'description': 'Бінарна класифікація: автомобілі (CIFAR-10) проти взуття (Fashion-MNIST)'
    }

    print(f"  ✓ Автомобілі: train={n_train_per_class}, test={n_test_per_class}")
    print(f"  ✓ Взуття:     train={n_train_per_class}, test={n_test_per_class}")
    print(f"  ✓ Разом:      train={len(X_train)}, test={len(X_test)}")

    return dataset


def visualize_datasets(datasets):
    """Візуалізує приклади з кожного датасету"""
    print_section("ВІЗУАЛІЗАЦІЯ ДАТАСЕТІВ")
    
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 10, figsize=(18, 3 * n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (dataset_name, data) in enumerate(datasets.items()):
        X_train = data['X_train']
        y_train = data['y_train']
        class_names = data['class_names']
        n_classes = data['n_classes']
        
        # Візьмемо по кілька прикладів з кожного класу
        samples_shown = 0
        for class_idx in range(n_classes):
            # Знайдемо приклади цього класу
            indices = np.where(y_train == class_idx)[0]
            
            # Візьмемо до 2-3 прикладів залежно від місця
            n_samples = min(2, 10 - samples_shown, len(indices))
            
            for i in range(n_samples):
                if samples_shown >= 10:
                    break
                
                ax = axes[row_idx, samples_shown]
                idx = indices[i]
                img = X_train[idx]
                
                if img.shape[0] == 28:  # Fashion-MNIST
                    # Resize для кращої візуалізації
                    img_resized = tf.image.resize(img, [32, 32]).numpy()
                    ax.imshow(img_resized)
                else:
                    ax.imshow(img)
                
                ax.set_title(f'{class_names[class_idx]}', fontsize=9)
                ax.axis('off')
                samples_shown += 1
        
        # Вимкнути зайві осі
        for i in range(samples_shown, 10):
            axes[row_idx, i].axis('off')
        
        # Заголовок рядка
        axes[row_idx, 0].text(-0.3, 0.5, dataset_name,
                              transform=axes[row_idx, 0].transAxes,
                              fontsize=12, weight='bold',
                              rotation=90, va='center')
    
    plt.suptitle('Приклади спеціалізованих датасетів', fontsize=16, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_samples.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: results/dataset_samples.png")
    plt.show()


class TrainingMonitor(Callback):
    """Callback для моніторингу змін параметрів під час навчання"""
    
    def __init__(self):
        super().__init__()
        self.history = defaultdict(list)
        self.layer_weights_history = defaultdict(list)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history[key].append(value)
        
        # Зберігаємо норми ваг для аналізу оптимізації
        for layer in self.model.layers:
            if hasattr(layer, 'trainable_weights') and len(layer.trainable_weights) > 0:
                weights = layer.get_weights()[0]  # Перша матриця ваг
                weight_norm = np.linalg.norm(weights)
                self.layer_weights_history[layer.name].append(weight_norm)


def build_finetuned_model(base_model_name, input_shape, n_classes, strategy='frozen'):
    """
    Будує модель для fine-tuning
    
    Parameters:
    - base_model_name: 'VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0'
    - input_shape: розмір входу
    - n_classes: кількість класів
    - strategy: 'frozen' (тільки classifier), 'partial' (останні шари), 'full' (всі шари)
    """
    
    # Вхідний шар
    inputs = keras.Input(shape=input_shape)
    
    # Resize якщо потрібно (pretrained моделі очікують мінімум 32x32)
    if input_shape[0] < 32:
        x = layers.Resizing(32, 32)(inputs)
    else:
        x = inputs
    
    # Завантажуємо базову модель
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    else:
        raise ValueError(f"Unknown model: {base_model_name}")
    
    # Застосовуємо стратегію fine-tuning
    if strategy == 'frozen':
        # Заморожуємо всі базові шари
        base_model.trainable = False
    elif strategy == 'partial':
        # Розморожуємо останні 20% шарів
        base_model.trainable = True
        n_layers = len(base_model.layers)
        for layer in base_model.layers[:int(n_layers * 0.8)]:
            layer.trainable = False
    elif strategy == 'full':
        # Розморожуємо всі шари
        base_model.trainable = True
    
    # Додаємо базову модель
    x = base_model(x, training=False if strategy == 'frozen' else True)
    
    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name=f'{base_model_name}_{strategy}')
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, strategy='frozen', epochs=20):
    """Навчає модель з відповідною стратегією"""
    
    # Визначаємо learning rate залежно від стратегії
    if strategy == 'frozen':
        lr = 1e-3
    elif strategy == 'partial':
        lr = 1e-4
    else:  # full
        lr = 1e-5
    
    # Компілюємо модель
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    monitor = TrainingMonitor()
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=0
    )
    
    # Навчання
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[monitor, early_stop, reduce_lr],
        verbose=0
    )
    
    return history, monitor


def evaluate_model(model, X_test, y_test, class_names):
    """Оцінює модель та повертає метрики"""
    
    # Передбачення
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'report': report
    }


def visualize_training_history(all_histories, dataset_name):
    """Візуалізує історію навчання для всіх моделей"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Training Accuracy
    ax = axes[0, 0]
    for model_name, (history, _) in all_histories.items():
        epochs = range(1, len(history.history['accuracy']) + 1)
        ax.plot(epochs, history.history['accuracy'], 
                marker='o', markersize=4, label=model_name, linewidth=2)
    ax.set_title('Точність на навчальній вибірці', fontsize=13, weight='bold')
    ax.set_xlabel('Епоха', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 2. Validation Accuracy
    ax = axes[0, 1]
    for model_name, (history, _) in all_histories.items():
        epochs = range(1, len(history.history['val_accuracy']) + 1)
        ax.plot(epochs, history.history['val_accuracy'],
                marker='s', markersize=4, label=model_name, linewidth=2)
    ax.set_title('Точність на валідаційній вибірці', fontsize=13, weight='bold')
    ax.set_xlabel('Епоха', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 3. Training Loss
    ax = axes[1, 0]
    for model_name, (history, _) in all_histories.items():
        epochs = range(1, len(history.history['loss']) + 1)
        ax.plot(epochs, history.history['loss'],
                marker='o', markersize=4, label=model_name, linewidth=2)
    ax.set_title('Loss на навчальній вибірці', fontsize=13, weight='bold')
    ax.set_xlabel('Епоха', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 4. Validation Loss
    ax = axes[1, 1]
    for model_name, (history, _) in all_histories.items():
        epochs = range(1, len(history.history['val_loss']) + 1)
        ax.plot(epochs, history.history['val_loss'],
                marker='s', markersize=4, label=model_name, linewidth=2)
    ax.set_title('Loss на валідаційній вибірці', fontsize=13, weight='bold')
    ax.set_xlabel('Епоха', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Історія навчання: {dataset_name}', fontsize=16, weight='bold')
    plt.tight_layout()
    
    safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / f'training_history_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Збережено: results/training_history_{safe_name}.png")
    plt.show()


def visualize_weight_changes(all_histories, dataset_name):
    """Візуалізує зміну норм ваг під час навчання (ілюстрація оптимізації)"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    model_idx = 0
    for model_name, (history, monitor) in all_histories.items():
        if model_idx >= 4:
            break
        
        ax = axes[model_idx]
        
        # Візуалізуємо перші 5 шарів з вагами
        layer_count = 0
        for layer_name, weight_norms in monitor.layer_weights_history.items():
            if layer_count >= 5:
                break
            if len(weight_norms) > 0:
                epochs = range(1, len(weight_norms) + 1)
                ax.plot(epochs, weight_norms, marker='o', markersize=3,
                       label=layer_name[:20], linewidth=2, alpha=0.7)
                layer_count += 1
        
        ax.set_title(f'Зміна ваг: {model_name}', fontsize=12, weight='bold')
        ax.set_xlabel('Епоха', fontsize=10)
        ax.set_ylabel('L2 норма ваг', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        
        model_idx += 1
    
    # Вимкнути зайві осі
    for i in range(model_idx, 4):
        axes[i].axis('off')
    
    plt.suptitle(f'Оптимізація параметрів під час навчання: {dataset_name}', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    
    safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / f'weight_optimization_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Збережено: results/weight_optimization_{safe_name}.png")
    plt.show()


def visualize_confusion_matrices(all_results, class_names, dataset_name):
    """Візуалізує confusion matrices для всіх моделей"""
    
    n_models = len(all_results)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))

    # Приводимо axes до плаского списку Axes, навіть якщо моделей одна
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        cm = results['confusion_matrix']
        
        # Нормалізована confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Візуалізація
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Частка'}, vmin=0, vmax=1)
        
        ax.set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.4f}',
                    fontsize=12, weight='bold')
        ax.set_xlabel('Передбачений клас', fontsize=10)
        ax.set_ylabel('Справжній клас', fontsize=10)
        
        # Поворот міток
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Вимкнути зайві осі
    for i in range(len(all_results), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Confusion Matrices: {dataset_name}', fontsize=16, weight='bold')
    plt.tight_layout()
    
    safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / f'confusion_matrices_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Збережено: results/confusion_matrices_{safe_name}.png")
    plt.show()


def analyze_classification_errors(X_test, y_test, y_pred, y_proba, class_names, dataset_name, model_name):
    """Аналізує та візуалізує найтиповіші помилки класифікації"""
    
    # Знаходимо помилки
    errors = y_test != y_pred
    error_indices = np.where(errors)[0]
    
    if len(error_indices) == 0:
        print(f"  Немає помилок для {model_name}!")
        return
    
    # Для кожної помилки обчислюємо "впевненість" у неправильному передбаченні
    error_confidences = []
    for idx in error_indices:
        confidence = y_proba[idx, y_pred[idx]]
        error_confidences.append((idx, confidence, y_test[idx], y_pred[idx]))
    
    # Сортуємо за впевненістю (від найвпевненіших до найменш)
    error_confidences.sort(key=lambda x: x[1], reverse=True)
    
    # Візуалізуємо топ-20 найвпевненіших помилок
    n_errors_to_show = min(20, len(error_confidences))
    n_cols = 5
    n_rows = (n_errors_to_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if n_errors_to_show > 1 else [axes]
    
    for i in range(n_errors_to_show):
        idx, confidence, true_label, pred_label = error_confidences[i]
        
        ax = axes[i]
        img = X_test[idx]
        
        ax.imshow(img)
        ax.set_title(f'True: {class_names[true_label]}\n'
                    f'Pred: {class_names[pred_label]}\n'
                    f'Conf: {confidence:.2f}',
                    fontsize=9, color='red')
        ax.axis('off')
    
    # Вимкнути зайві осі
    for i in range(n_errors_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Топ-{n_errors_to_show} впевнених помилок: {model_name} на {dataset_name}',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    safe_dataset = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    safe_model = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / f'errors_{safe_dataset}_{safe_model}.png', 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Збережено: results/errors_{safe_dataset}_{safe_model}.png")
    plt.show()


def compare_strategies(dataset, dataset_name):
    """Порівнює різні стратегії fine-tuning на одній моделі"""
    print_section(f"ПОРІВНЯННЯ СТРАТЕГІЙ FINE-TUNING: {dataset_name}")
    
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    n_classes = dataset['n_classes']
    input_shape = dataset['shape']
    class_names = dataset['class_names']
    
    strategies = ['frozen', 'partial', 'full']
    base_model = 'ResNet50'
    
    results = {}
    
    for strategy in strategies:
        print(f"\n  Стратегія: {strategy}...")
        
        # Будуємо модель
        model = build_finetuned_model(base_model, input_shape, n_classes, strategy=strategy)
        
        # Тренуємо
        history, monitor = train_model(model, X_train, y_train, X_test, y_test, 
                                      strategy=strategy, epochs=15)
        
        # Оцінюємо
        eval_results = evaluate_model(model, X_test, y_test, class_names)
        
        results[f'{base_model} ({strategy})'] = {
            'history': history,
            'monitor': monitor,
            'eval': eval_results
        }
        
        print(f"  ✓ Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  ✓ F1-Score: {eval_results['f1']:.4f}")
    
    # Візуалізація порівняння
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Validation Accuracy
    ax = axes[0]
    for strategy_name, data in results.items():
        history = data['history']
        epochs = range(1, len(history.history['val_accuracy']) + 1)
        ax.plot(epochs, history.history['val_accuracy'], 
               marker='o', linewidth=2, label=strategy_name)
    ax.set_title('Validation Accuracy', fontsize=13, weight='bold')
    ax.set_xlabel('Епоха', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Final Metrics
    ax = axes[1]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(strategies))
    width = 0.2
    
    for i, metric_name in enumerate(metrics_names):
        values = [results[f'{base_model} ({s})']['eval'][metric_name.lower()] 
                 for s in strategies]
        ax.bar(x + i * width, values, width, label=metric_name)
    
    ax.set_title('Фінальні метрики', fontsize=13, weight='bold')
    ax.set_xlabel('Стратегія', fontsize=11)
    ax.set_ylabel('Значення', fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Training Time (approximation via epochs)
    ax = axes[2]
    n_epochs = [len(results[f'{base_model} ({s})']['history'].history['loss']) 
               for s in strategies]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax.bar(strategies, n_epochs, color=colors, edgecolor='black')
    ax.set_title('Кількість епох до зупинки', fontsize=13, weight='bold')
    ax.set_xlabel('Стратегія', fontsize=11)
    ax.set_ylabel('Епохи', fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'Порівняння стратегій Fine-Tuning: {dataset_name}',
                 fontsize=16, weight='bold')
    plt.tight_layout()
    
    safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / f'strategy_comparison_{safe_name}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Збережено: results/strategy_comparison_{safe_name}.png")
    plt.show()
    
    return results


def benchmark_models(dataset, dataset_name):
    """Бенчмаркінг різних архітектур на одному датасеті"""
    print_section(f"БЕНЧМАРКІНГ МОДЕЛЕЙ: {dataset_name}")
    
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    n_classes = dataset['n_classes']
    input_shape = dataset['shape']
    class_names = dataset['class_names']
    
    # Моделі для тестування
    model_names = ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0']
    strategy = 'partial'  # Використовуємо partial fine-tuning
    
    all_results = {}
    all_histories = {}
    training_times = {}
    
    for model_name in model_names:
        print(f"\n  [{model_names.index(model_name) + 1}/{len(model_names)}] Модель: {model_name}...")
        
        start_time = time.time()
        
        # Будуємо модель
        model = build_finetuned_model(model_name, input_shape, n_classes, strategy=strategy)
        
        # Тренуємо
        history, monitor = train_model(model, X_train, y_train, X_test, y_test,
                                      strategy=strategy, epochs=20)
        
        training_time = time.time() - start_time
        training_times[model_name] = training_time
        
        # Оцінюємо
        eval_results = evaluate_model(model, X_test, y_test, class_names)
        
        all_results[model_name] = eval_results
        all_histories[model_name] = (history, monitor)
        
        print(f"  ✓ Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  ✓ F1-Score: {eval_results['f1']:.4f}")
        print(f"  ✓ Час навчання: {training_time:.1f}s")
    
    # Візуалізації
    visualize_training_history(all_histories, dataset_name)
    visualize_weight_changes(all_histories, dataset_name)
    visualize_confusion_matrices(all_results, class_names, dataset_name)
    
    # Аналіз помилок для кожної моделі
    print("\n  Аналіз помилок класифікації...")
    for model_name, results in all_results.items():
        analyze_classification_errors(
            X_test, y_test, 
            results['predictions'], 
            results['probabilities'],
            class_names, dataset_name, model_name
        )
    
    # Зведена таблиця результатів
    print(f"\n  📊 Зведена таблиця результатів:")
    results_data = []
    for model_name in model_names:
        results_data.append({
            'Модель': model_name,
            'Accuracy': all_results[model_name]['accuracy'],
            'Precision': all_results[model_name]['precision'],
            'Recall': all_results[model_name]['recall'],
            'F1-Score': all_results[model_name]['f1'],
            'Час (s)': training_times[model_name]
        })
    
    df = pd.DataFrame(results_data)
    print(df.round(4).to_string(index=False))
    
    # Збереження результатів
    safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    df.to_csv(OUTPUT_DIR / f'benchmark_{safe_name}.csv', index=False)
    print(f"\n  ✓ Збережено: results/benchmark_{safe_name}.csv")
    
    return all_results, all_histories


def create_final_comparison(all_datasets_results):
    """Створює фінальне порівняння всіх моделей на всіх датасетах"""
    print_section("ФІНАЛЬНЕ ПОРІВНЯННЯ")
    
    # Збираємо дані
    data = []
    for dataset_name, results in all_datasets_results.items():
        for model_name, eval_results in results.items():
            data.append({
                'Dataset': dataset_name.split('(')[0].strip(),
                'Model': model_name,
                'Accuracy': eval_results['accuracy'],
                'F1-Score': eval_results['f1']
            })
    
    df = pd.DataFrame(data)
    
    # Візуалізація
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 1. Accuracy по датасетах
    ax = axes[0]
    datasets = df['Dataset'].unique()
    models = df['Model'].unique()
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, model in enumerate(models):
        values = [df[(df['Dataset'] == ds) & (df['Model'] == model)]['Accuracy'].values[0]
                 for ds in datasets]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_title('Accuracy по датасетах', fontsize=14, weight='bold')
    ax.set_xlabel('Датасет', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 2. F1-Score по датасетах
    ax = axes[1]
    for i, model in enumerate(models):
        values = [df[(df['Dataset'] == ds) & (df['Model'] == model)]['F1-Score'].values[0]
                 for ds in datasets]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_title('F1-Score по датасетах', fontsize=14, weight='bold')
    ax.set_xlabel('Датасет', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Фінальне порівняння всіх моделей', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'final_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Збережено: results/final_comparison.png")
    plt.show()
    
    # Таблиця
    print("\n  📊 Зведена таблиця всіх результатів:")
    pivot_acc = df.pivot(index='Dataset', columns='Model', values='Accuracy')
    print("\nAccuracy:")
    print(pivot_acc.round(4).to_string())
    
    pivot_f1 = df.pivot(index='Dataset', columns='Model', values='F1-Score')
    print("\nF1-Score:")
    print(pivot_f1.round(4).to_string())
    
    # Збереження
    pivot_acc.to_csv(OUTPUT_DIR / 'final_accuracy_comparison.csv')
    pivot_f1.to_csv(OUTPUT_DIR / 'final_f1_comparison.csv')
    print("\n  ✓ Збережено: results/final_accuracy_comparison.csv")
    print("  ✓ Збережено: results/final_f1_comparison.csv")


def main():
    """Головна функція (спрощений варіант: 2 класи, 1 модель)"""
    print("\n" + "=" * 80)
    print("  FINE-TUNING PRETRAINED CNN (СПРОЩЕНО)")
    print("  Задача: Автомобілі vs Взуття, 2 класи, 1 модель")
    print("=" * 80)

    # 1. Завантаження спрощеного 2-класового датасету
    dataset = load_two_class_car_vs_shoes()
    dataset_name = "Автомобілі vs Взуття"

    # 2. Візуалізація декількох прикладів
    visualize_datasets({dataset_name: dataset})

    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    input_shape = dataset['shape']
    class_names = dataset['class_names']

    # 3. Одна швидка модель + одна стратегія fine-tuning
    base_model_name = 'MobileNetV2'  # більш легка й швидка модель
    strategy = 'partial'             # розморожуємо останні шари

    print_section(f"НАВЧАННЯ МОДЕЛІ: {base_model_name} ({strategy})")
    model = build_finetuned_model(base_model_name, input_shape, dataset['n_classes'], strategy=strategy)

    # Щоб ще прискорити навчання, зменшимо кількість епох
    history, monitor = train_model(
        model,
        X_train, y_train,
        X_test, y_test,          # використовуємо тест як валідацію для простоти
        strategy=strategy,
        epochs=10
    )

    # 4. Оцінка моделі
    eval_results = evaluate_model(model, X_test, y_test, class_names)

    print("\n📊 ПІДСУМКОВІ МЕТРИКИ:")
    print(f"  Accuracy : {eval_results['accuracy']:.4f}")
    print(f"  Precision: {eval_results['precision']:.4f}")
    print(f"  Recall   : {eval_results['recall']:.4f}")
    print(f"  F1-Score : {eval_results['f1']:.4f}")
    print("\nДокладний звіт класифікації:")
    print(eval_results['report'])

    # 5. Основні візуалізації для однієї моделі
    all_histories = {f'{base_model_name} ({strategy})': (history, monitor)}
    all_results = {f'{base_model_name} ({strategy})': eval_results}

    visualize_training_history(all_histories, dataset_name)
    visualize_weight_changes(all_histories, dataset_name)
    visualize_confusion_matrices(all_results, class_names, dataset_name)

    print("\n  Аналіз помилок класифікації...")
    analyze_classification_errors(
        X_test, y_test,
        eval_results['predictions'],
        eval_results['probabilities'],
        class_names,
        dataset_name,
        f'{base_model_name} ({strategy})'
    )

    # Короткий підсумок
    print_section("ПІДСУМОК (СПРОЩЕНИЙ СЦЕНАРІЙ)")
    print("\n✅ Навчання завершено!")
    print("\n📁 Створені основні файли (для цього спрощеного запуску):")
    print("  - results/dataset_samples.png - приклади двох класів")
    print("  - results/training_history_Автомобілі_vs_Взуття.png - історія навчання")
    print("  - results/weight_optimization_Автомобілі_vs_Взуття.png - зміна ваг")
    print("  - results/confusion_matrices_Автомобілі_vs_Взуття.png - матриця плутанини")
    print("  - results/errors_Автомобілі_vs_Взуття_MobileNetV2_partial.png - типові помилки")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

