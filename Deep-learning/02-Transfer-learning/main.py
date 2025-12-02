"""
Transfer Learning та Бенчмаркінг Класифікаторів
Фокус: Порівняння класифікаторів поверх різних feature extractors

Мета: Дослідити як різні комбінації pretrained моделей та класифікаторів
впливають на точність, а також вивчити залежність від глибини заморожування.

Feature Extractors:
- VGG16 (pretrained ImageNet)
- VGG19 (pretrained ImageNet)
- ResNet50 (pretrained ImageNet)
- MobileNetV2 (pretrained ImageNet)
- InceptionV3 (pretrained ImageNet)

Класифікатори:
- Logistic Regression
- SVM (лінійний та RBF)
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Naive Bayes
- MLP (Neural Network)

Дослідження:
- Точність для різних комбінацій
- Вплив глибини заморожування шарів
- Confusion matrices
- Типові помилки класифікації
- Час навчання та інференції
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
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, MobileNetV2, InceptionV3
)
from tensorflow.keras.datasets import cifar10

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

import time
from pathlib import Path
from collections import defaultdict

# Налаштування
tf.random.set_seed(42)
np.random.seed(42)

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Назви класів CIFAR-10
CIFAR10_CLASSES = [
    'літак', 'автомобіль', 'птах', 'кіт', 'олень',
    'собака', 'жаба', 'кінь', 'корабель', 'вантажівка'
]


def print_section(title):
    """Виводить заголовок секції"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_cifar10_data(n_train=5000, n_test=2000):
    """Завантажує CIFAR-10 датасет"""
    print_section("ЗАВАНТАЖЕННЯ CIFAR-10")
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Нормалізація
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Підмножина для швидкості
    indices_train = np.random.choice(len(X_train), n_train, replace=False)
    indices_test = np.random.choice(len(X_test), n_test, replace=False)
    
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]
    X_test = X_test[indices_test]
    y_test = y_test[indices_test]
    
    print(f"✓ Train set: {X_train.shape} | {len(np.unique(y_train))} класів")
    print(f"✓ Test set: {X_test.shape} | {len(np.unique(y_test))} класів")
    
    return X_train, y_train, X_test, y_test


def visualize_dataset_samples(X_train, y_train):
    """Візуалізує приклади з датасету"""
    print_section("ВІЗУАЛІЗАЦІЯ ДАТАСЕТУ")
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
    axes = axes.flatten()
    
    for i in range(10):
        # По 2 приклади кожного класу
        indices = np.where(y_train == i)[0][:2]
        
        for j, idx in enumerate(indices):
            ax = axes[i + j * 10]
            ax.imshow(X_train[idx])
            if j == 0:
                ax.set_title(f'{CIFAR10_CLASSES[i]}', fontsize=10, weight='bold')
            ax.axis('off')
    
    plt.suptitle('CIFAR-10: Приклади зображень', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_samples.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: results/dataset_samples.png")
    plt.show()


def build_feature_extractor(model_name, freeze_layers=None):
    """
    Створює feature extractor з pretrained моделі
    
    Args:
        model_name: назва моделі ('VGG16', 'VGG19', 'ResNet50', 'MobileNetV2', 'InceptionV3')
        freeze_layers: кількість шарів для заморожування (None = всі)
    """
    input_shape = (32, 32, 3)
    
    # Створюємо input для ресайзу
    inputs = keras.Input(shape=input_shape)
    
    # Ресайз до мінімального розміру для pretrained моделей
    if model_name == 'InceptionV3':
        target_size = 75  # мінімум для InceptionV3
    else:
        target_size = 32
    
    x = inputs
    if target_size != 32:
        x = layers.Resizing(target_size, target_size)(x)
    
    # Завантаження base моделі
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False,
                          input_shape=(target_size, target_size, 3))
    elif model_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False,
                          input_shape=(target_size, target_size, 3))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
                             input_shape=(target_size, target_size, 3))
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                input_shape=(target_size, target_size, 3))
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                input_shape=(target_size, target_size, 3))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Заморожування шарів
    if freeze_layers is not None:
        # Розморожуємо всі шари
        for layer in base_model.layers:
            layer.trainable = True
        
        # Заморожуємо перші freeze_layers
        for layer in base_model.layers[:freeze_layers]:
            layer.trainable = False
    else:
        # Заморожуємо всі шари
        base_model.trainable = False
    
    # Додаємо pooling
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Feature extractor
    feature_extractor = models.Model(inputs, x, name=f'{model_name}_extractor')
    
    return feature_extractor, base_model


def extract_features(model, X_data, batch_size=64):
    """Витягує ознаки з даних"""
    features = model.predict(X_data, batch_size=batch_size, verbose=0)
    return features


def get_classifiers():
    """Повертає словник з класифікаторами"""
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=3000, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    }
    return classifiers


def benchmark_classifiers(features_train, y_train, features_test, y_test, scaler=None):
    """
    Бенчмарк різних класифікаторів
    
    Returns:
        DataFrame з результатами
    """
    classifiers = get_classifiers()
    results = {}
    predictions = {}
    
    # Стандартизація (важливо для деяких класифікаторів)
    if scaler is None:
        scaler = StandardScaler()
        features_train_scaled = scaler.fit_transform(features_train)
    else:
        features_train_scaled = scaler.transform(features_train)
    
    features_test_scaled = scaler.transform(features_test)
    
    for clf_name, clf in classifiers.items():
        print(f"\n  [{list(classifiers.keys()).index(clf_name) + 1}/{len(classifiers)}] {clf_name}...")
        
        start_time = time.time()
        
        try:
            # Навчання
            clf.fit(features_train_scaled, y_train)
            train_time = time.time() - start_time
            
            # Передбачення
            start_inference = time.time()
            y_pred = clf.predict(features_test_scaled)
            inference_time = time.time() - start_inference
            
            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted')
            
            results[clf_name] = {
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Precision': precision,
                'Recall': recall,
                'Train Time (s)': train_time,
                'Inference Time (s)': inference_time
            }
            
            predictions[clf_name] = y_pred
            
            print(f"    ✓ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Train: {train_time:.2f}s")
            
        except Exception as e:
            print(f"    ⚠ Помилка: {e}")
            results[clf_name] = {
                'Accuracy': np.nan,
                'F1-Score': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'Train Time (s)': np.nan,
                'Inference Time (s)': np.nan
            }
            predictions[clf_name] = None
    
    results_df = pd.DataFrame(results).T
    return results_df, predictions, scaler


def benchmark_extractors_with_classifiers(X_train, y_train, X_test, y_test):
    """Бенчмарк різних комбінацій екстракторів та класифікаторів"""
    print_section("БЕНЧМАРК: ЕКСТРАКТОРИ × КЛАСИФІКАТОРИ")
    
    extractors = ['VGG16', 'VGG19', 'ResNet50', 'MobileNetV2', 'InceptionV3']
    all_results = {}
    all_predictions = {}
    all_features = {}
    scalers = {}
    
    for extractor_name in extractors:
        print(f"\n{'='*80}")
        print(f"  Feature Extractor: {extractor_name}")
        print('='*80)
        
        # Створення екстрактора
        print(f"\n  Завантаження {extractor_name}...")
        try:
            feature_extractor, _ = build_feature_extractor(extractor_name)
            
            # Витягування ознак
            print(f"  Екстракція ознак...")
            start_time = time.time()
            features_train = extract_features(feature_extractor, X_train)
            features_test = extract_features(feature_extractor, X_test)
            extraction_time = time.time() - start_time
            
            print(f"  ✓ Розмірність ознак: {features_train.shape[1]}")
            print(f"  ✓ Час екстракції: {extraction_time:.2f}s")
            
            all_features[extractor_name] = (features_train, features_test)
            
            # Бенчмарк класифікаторів
            results_df, predictions, scaler = benchmark_classifiers(
                features_train, y_train, features_test, y_test
            )
            
            all_results[extractor_name] = results_df
            all_predictions[extractor_name] = predictions
            scalers[extractor_name] = scaler
            
            # Виведення таблиці
            print(f"\n  📊 Результати для {extractor_name}:")
            print(results_df[['Accuracy', 'F1-Score', 'Train Time (s)']].round(4).to_string())
            
            # Збереження
            safe_name = extractor_name.replace('/', '_')
            results_df.to_csv(OUTPUT_DIR / f'benchmark_{safe_name}.csv')
            print(f"\n  ✓ Збережено: results/benchmark_{safe_name}.csv")
            
        except Exception as e:
            print(f"  ⚠ Помилка при роботі з {extractor_name}: {e}")
            continue
    
    return all_results, all_predictions, all_features, scalers


def study_freezing_depth(X_train, y_train, X_test, y_test, model_name='VGG16'):
    """
    Досліджує вплив глибини заморожування шарів на точність
    
    Args:
        model_name: модель для експерименту
    """
    print_section(f"ДОСЛІДЖЕННЯ ГЛИБИНИ ЗАМОРОЖУВАННЯ: {model_name}")
    
    # Створюємо базову модель щоб дізнатися кількість шарів
    _, base_model = build_feature_extractor(model_name)
    total_layers = len(base_model.layers)
    
    print(f"\n  Загальна кількість шарів: {total_layers}")
    
    # Тестуємо різні глибини заморожування
    freeze_configs = [
        0,  # Всі шари trainable
        total_layers // 4,
        total_layers // 2,
        3 * total_layers // 4,
        total_layers  # Всі шари frozen
    ]
    
    results = {}
    
    # Використаємо один класифікатор (Logistic Regression) для швидкості
    print(f"\n  Тестуємо {len(freeze_configs)} конфігурацій заморожування...")
    print(f"  Класифікатор: Logistic Regression\n")
    
    for freeze_layers in freeze_configs:
        percent_frozen = (freeze_layers / total_layers) * 100
        print(f"  [{freeze_configs.index(freeze_layers) + 1}/{len(freeze_configs)}] " +
              f"Заморожено: {freeze_layers}/{total_layers} шарів ({percent_frozen:.0f}%)...")
        
        try:
            # Створення екстрактора
            feature_extractor, _ = build_feature_extractor(model_name, freeze_layers)
            
            # Якщо є незаморожені шари, потрібно дотренувати
            if freeze_layers < total_layers:
                print(f"      Дотренування незаморожених шарів...")
                
                # Додаємо класифікаційний шар
                inputs = feature_extractor.input
                x = feature_extractor.output
                outputs = layers.Dense(10, activation='softmax')(x)
                full_model = models.Model(inputs, outputs)
                
                full_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Fine-tuning (кілька epochs)
                full_model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=3,
                    batch_size=64,
                    verbose=0
                )
                
                # Оновлюємо feature extractor
                feature_extractor = models.Model(
                    inputs=full_model.input,
                    outputs=full_model.layers[-2].output  # До останнього Dense
                )
            
            # Екстракція ознак
            features_train = extract_features(feature_extractor, X_train)
            features_test = extract_features(feature_extractor, X_test)
            
            # Класифікація
            scaler = StandardScaler()
            features_train_scaled = scaler.fit_transform(features_train)
            features_test_scaled = scaler.transform(features_test)
            
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(features_train_scaled, y_train)
            
            y_pred = clf.predict(features_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[freeze_layers] = {
                'frozen_layers': freeze_layers,
                'percent_frozen': percent_frozen,
                'accuracy': accuracy
            }
            
            print(f"      ✓ Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"      ⚠ Помилка: {e}")
            results[freeze_layers] = {
                'frozen_layers': freeze_layers,
                'percent_frozen': percent_frozen,
                'accuracy': np.nan
            }
    
    results_df = pd.DataFrame(results).T
    
    # Візуалізація
    visualize_freezing_depth(results_df, model_name)
    
    return results_df


def visualize_freezing_depth(results_df, model_name):
    """Візуалізує залежність accuracy від глибини заморожування"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Графік 1: Accuracy vs. кількість заморожених шарів
    ax = axes[0]
    ax.plot(results_df['frozen_layers'], results_df['accuracy'], 
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.fill_between(results_df['frozen_layers'], results_df['accuracy'], 
                     alpha=0.3, color='#2E86AB')
    ax.set_xlabel('Кількість заморожених шарів', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
    ax.set_title(f'{model_name}: Вплив заморожування шарів', fontsize=13, weight='bold')
    ax.grid(alpha=0.3)
    
    # Додаємо значення на графік
    for idx, row in results_df.iterrows():
        ax.annotate(f'{row["accuracy"]:.3f}', 
                   xy=(row['frozen_layers'], row['accuracy']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)
    
    # Графік 2: Accuracy vs. відсоток заморожених шарів
    ax = axes[1]
    ax.plot(results_df['percent_frozen'], results_df['accuracy'],
            marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax.fill_between(results_df['percent_frozen'], results_df['accuracy'],
                     alpha=0.3, color='#A23B72')
    ax.set_xlabel('Відсоток заморожених шарів (%)', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
    ax.set_title(f'{model_name}: Accuracy vs. Freeze Ratio', fontsize=13, weight='bold')
    ax.grid(alpha=0.3)
    
    # Додаємо значення
    for idx, row in results_df.iterrows():
        ax.annotate(f'{row["accuracy"]:.3f}',
                   xy=(row['percent_frozen'], row['accuracy']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9)
    
    plt.suptitle(f'Дослідження глибини заморожування шарів: {model_name}',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    safe_name = model_name.replace('/', '_')
    plt.savefig(OUTPUT_DIR / f'freezing_depth_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено: results/freezing_depth_{safe_name}.png")
    plt.show()


def create_comparison_heatmap(all_results):
    """Створює heatmap порівняння всіх комбінацій"""
    print_section("HEATMAP: ПОРІВНЯННЯ ВСІХ КОМБІНАЦІЙ")
    
    # Створюємо матрицю accuracy
    extractors = list(all_results.keys())
    classifiers = list(all_results[extractors[0]].index)
    
    matrix = np.zeros((len(classifiers), len(extractors)))
    
    for i, clf in enumerate(classifiers):
        for j, ext in enumerate(extractors):
            matrix[i, j] = all_results[ext].loc[clf, 'Accuracy']
    
    # Візуалізація
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    
    # Налаштування осей
    ax.set_xticks(np.arange(len(extractors)))
    ax.set_yticks(np.arange(len(classifiers)))
    ax.set_xticklabels(extractors, rotation=45, ha='right')
    ax.set_yticklabels(classifiers)
    
    # Додаємо значення в клітинки
    for i in range(len(classifiers)):
        for j in range(len(extractors)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black" if matrix[i, j] > 0.5 else "white",
                          fontsize=9, weight='bold')
    
    ax.set_xlabel('Feature Extractor', fontsize=12, weight='bold')
    ax.set_ylabel('Classifier', fontsize=12, weight='bold')
    ax.set_title('Accuracy: Екстрактори × Класифікатори', fontsize=14, weight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', fontsize=11, weight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'heatmap_all_combinations.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: results/heatmap_all_combinations.png")
    plt.show()


def plot_confusion_matrices(all_predictions, y_test, extractor_name, top_n=4):
    """
    Візуалізує confusion matrices для найкращих класифікаторів
    
    Args:
        all_predictions: словник з передбаченнями
        y_test: справжні мітки
        extractor_name: назва екстрактора
        top_n: кількість топ класифікаторів для візуалізації
    """
    predictions = all_predictions[extractor_name]
    
    # Відфільтруємо None значення
    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
    
    if len(valid_predictions) == 0:
        print(f"⚠ Немає валідних передбачень для {extractor_name}")
        return
    
    # Беремо top_n класифікаторів (за алфавітом, можна змінити логіку)
    classifiers = list(valid_predictions.keys())[:top_n]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, clf_name in enumerate(classifiers):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        y_pred = valid_predictions[clf_name]
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Нормалізація
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Візуалізація
        im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Додаємо значення
        for i in range(len(CIFAR10_CLASSES)):
            for j in range(len(CIFAR10_CLASSES)):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                             ha="center", va="center",
                             color="white" if cm_normalized[i, j] > 0.5 else "black",
                             fontsize=7)
        
        ax.set_xticks(np.arange(len(CIFAR10_CLASSES)))
        ax.set_yticks(np.arange(len(CIFAR10_CLASSES)))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
        ax.set_xlabel('Передбачений клас', fontsize=10)
        ax.set_ylabel('Справжній клас', fontsize=10)
        
        accuracy = accuracy_score(y_test, y_pred)
        ax.set_title(f'{clf_name}\nAccuracy: {accuracy:.3f}', fontsize=11, weight='bold')
    
    plt.suptitle(f'Confusion Matrices: {extractor_name}', fontsize=14, weight='bold')
    plt.tight_layout()
    
    safe_name = extractor_name.replace('/', '_')
    plt.savefig(OUTPUT_DIR / f'confusion_matrices_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Збережено: results/confusion_matrices_{safe_name}.png")
    plt.show()


def find_typical_errors(X_test, y_test, y_pred, extractor_name, clf_name, n_examples=10):
    """
    Знаходить та візуалізує найтиповіші помилки класифікації
    
    Args:
        X_test: тестові зображення
        y_test: справжні мітки
        y_pred: передбачені мітки
        n_examples: кількість прикладів помилок
    """
    # Знаходимо помилки
    errors_indices = np.where(y_test != y_pred)[0]
    
    if len(errors_indices) == 0:
        print("  ✓ Немає помилок!")
        return
    
    # Аналізуємо найчастіші помилки (confused pairs)
    confusion_pairs = defaultdict(int)
    for idx in errors_indices:
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        pair = (true_label, pred_label)
        confusion_pairs[pair] += 1
    
    # Топ помилкові пари
    top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\n  📊 Топ-5 найчастіших помилок:")
    for (true_cls, pred_cls), count in top_confusions:
        print(f"    {CIFAR10_CLASSES[true_cls]:>12} → {CIFAR10_CLASSES[pred_cls]:<12} : {count} разів")
    
    # Візуалізація помилок
    n_to_show = min(n_examples, len(errors_indices))
    selected_errors = np.random.choice(errors_indices, n_to_show, replace=False)
    
    n_cols = 5
    n_rows = (n_to_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_errors):
        ax = axes[i]
        
        img = X_test[idx]
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        
        ax.imshow(img)
        ax.set_title(f'True: {CIFAR10_CLASSES[true_label]}\nPred: {CIFAR10_CLASSES[pred_label]}',
                    fontsize=9, color='red', weight='bold')
        ax.axis('off')
    
    # Вимкнути зайві осі
    for i in range(n_to_show, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Типові помилки: {extractor_name} + {clf_name}',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    safe_ext = extractor_name.replace('/', '_')
    safe_clf = clf_name.replace(' ', '_')
    plt.savefig(OUTPUT_DIR / f'typical_errors_{safe_ext}_{safe_clf}.png',
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Збережено: results/typical_errors_{safe_ext}_{safe_clf}.png")
    plt.show()


def plot_accuracy_comparison(all_results):
    """Порівняльний графік accuracy для всіх комбінацій"""
    print_section("ПОРІВНЯННЯ ACCURACY")
    
    extractors = list(all_results.keys())
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(all_results[extractors[0]]))
    width = 0.15
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for i, extractor in enumerate(extractors):
        accuracies = all_results[extractor]['Accuracy'].values
        offset = width * (i - len(extractors) / 2)
        ax.bar(x + offset, accuracies, width, label=extractor, color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Classifier', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
    ax.set_title('Порівняння Accuracy: Різні комбінації екстракторів та класифікаторів',
                 fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_results[extractors[0]].index, rotation=45, ha='right')
    ax.legend(title='Feature Extractor', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: results/accuracy_comparison.png")
    plt.show()


def plot_time_comparison(all_results):
    """Порівняння часу навчання"""
    print_section("ПОРІВНЯННЯ ЧАСУ НАВЧАННЯ")
    
    extractors = list(all_results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train time
    ax = axes[0]
    for extractor in extractors:
        train_times = all_results[extractor]['Train Time (s)'].values
        ax.plot(train_times, marker='o', label=extractor, linewidth=2)
    
    ax.set_xlabel('Classifier Index', fontsize=11, weight='bold')
    ax.set_ylabel('Train Time (s)', fontsize=11, weight='bold')
    ax.set_title('Час навчання класифікаторів', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Inference time
    ax = axes[1]
    for extractor in extractors:
        inference_times = all_results[extractor]['Inference Time (s)'].values
        ax.plot(inference_times, marker='s', label=extractor, linewidth=2)
    
    ax.set_xlabel('Classifier Index', fontsize=11, weight='bold')
    ax.set_ylabel('Inference Time (s)', fontsize=11, weight='bold')
    ax.set_title('Час інференції класифікаторів', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Порівняння швидкості', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'time_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: results/time_comparison.png")
    plt.show()


def create_summary_report(all_results):
    """Створює підсумковий звіт"""
    print_section("ПІДСУМКОВИЙ ЗВІТ")
    
    # Знаходимо найкращу комбінацію
    best_accuracy = 0
    best_combo = ("", "")
    
    for extractor, results_df in all_results.items():
        max_acc_idx = results_df['Accuracy'].idxmax()
        max_acc = results_df.loc[max_acc_idx, 'Accuracy']
        
        if max_acc > best_accuracy:
            best_accuracy = max_acc
            best_combo = (extractor, max_acc_idx)
    
    print(f"\n🏆 НАЙКРАЩА КОМБІНАЦІЯ:")
    print(f"  Feature Extractor: {best_combo[0]}")
    print(f"  Classifier: {best_combo[1]}")
    print(f"  Accuracy: {best_accuracy:.4f}")
    
    # Середня accuracy для кожного екстрактора
    print(f"\n📊 СЕРЕДНЯ ACCURACY ПО ЕКСТРАКТОРАХ:")
    for extractor, results_df in all_results.items():
        mean_acc = results_df['Accuracy'].mean()
        print(f"  {extractor:>15}: {mean_acc:.4f}")
    
    # Середня accuracy для кожного класифікатора
    print(f"\n📊 СЕРЕДНЯ ACCURACY ПО КЛАСИФІКАТОРАХ:")
    classifiers = list(all_results[list(all_results.keys())[0]].index)
    for clf in classifiers:
        accuracies = [all_results[ext].loc[clf, 'Accuracy'] for ext in all_results.keys()]
        mean_acc = np.mean(accuracies)
        print(f"  {clf:>20}: {mean_acc:.4f}")
    
    # Створюємо зведену таблицю
    summary_data = []
    for extractor, results_df in all_results.items():
        for clf_name in results_df.index:
            row = {
                'Feature Extractor': extractor,
                'Classifier': clf_name,
                'Accuracy': results_df.loc[clf_name, 'Accuracy'],
                'F1-Score': results_df.loc[clf_name, 'F1-Score'],
                'Train Time (s)': results_df.loc[clf_name, 'Train Time (s)']
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    
    # Збереження
    summary_df.to_csv(OUTPUT_DIR / 'summary_all_results.csv', index=False)
    print(f"\n✓ Збережено: results/summary_all_results.csv")
    
    print(f"\n📋 ТОП-10 КОМБІНАЦІЙ:")
    print(summary_df.head(10)[['Feature Extractor', 'Classifier', 'Accuracy']].to_string(index=False))


def main():
    """Головна функція"""
    print("\n" + "="*80)
    print("  TRANSFER LEARNING ТА БЕНЧМАРКІНГ КЛАСИФІКАТОРІВ")
    print("  Дослідження комбінацій Feature Extractors × Classifiers")
    print("="*80)
    
    # 1. Завантаження даних
    X_train, y_train, X_test, y_test = load_cifar10_data(n_train=5000, n_test=2000)
    
    # 2. Візуалізація датасету
    visualize_dataset_samples(X_train, y_train)
    
    # 3. Бенчмарк екстракторів та класифікаторів
    all_results, all_predictions, all_features, scalers = benchmark_extractors_with_classifiers(
        X_train, y_train, X_test, y_test
    )
    
    # 4. Дослідження глибини заморожування (VGG16 як приклад)
    print("\n")
    freezing_results = study_freezing_depth(X_train, y_train, X_test, y_test, model_name='VGG16')
    freezing_results.to_csv(OUTPUT_DIR / 'freezing_depth_VGG16.csv')
    
    # Також для ResNet50
    print("\n")
    freezing_results_resnet = study_freezing_depth(X_train, y_train, X_test, y_test, model_name='ResNet50')
    freezing_results_resnet.to_csv(OUTPUT_DIR / 'freezing_depth_ResNet50.csv')
    
    # 5. Порівняльні візуалізації
    create_comparison_heatmap(all_results)
    plot_accuracy_comparison(all_results)
    plot_time_comparison(all_results)
    
    # 6. Confusion matrices для кожного екстрактора
    print_section("СТВОРЕННЯ CONFUSION MATRICES")
    for extractor_name in all_results.keys():
        print(f"\n  Confusion matrices для {extractor_name}...")
        plot_confusion_matrices(all_predictions, y_test, extractor_name, top_n=4)
    
    # 7. Аналіз типових помилок для найкращих комбінацій
    print_section("АНАЛІЗ ТИПОВИХ ПОМИЛОК")
    
    # Знайдемо топ-3 комбінації
    top_combos = []
    for extractor, results_df in all_results.items():
        for clf_name in results_df.index:
            acc = results_df.loc[clf_name, 'Accuracy']
            top_combos.append((acc, extractor, clf_name))
    
    top_combos.sort(reverse=True)
    
    for i, (acc, extractor, clf_name) in enumerate(top_combos[:3]):
        print(f"\n[{i+1}/3] {extractor} + {clf_name} (Accuracy: {acc:.4f})")
        
        y_pred = all_predictions[extractor][clf_name]
        if y_pred is not None:
            find_typical_errors(X_test, y_test, y_pred, extractor, clf_name, n_examples=10)
    
    # 8. Підсумковий звіт
    create_summary_report(all_results)
    
    # Підсумок
    print_section("ЗАВЕРШЕННЯ")
    print("\n✅ Аналіз завершено!")
    print("\n📁 Створені файли:")
    print("  - results/dataset_samples.png - приклади датасету")
    print("  - results/benchmark_*.csv - результати для кожного екстрактора")
    print("  - results/freezing_depth_*.png - залежність від глибини заморожування")
    print("  - results/heatmap_all_combinations.png - heatmap всіх комбінацій")
    print("  - results/accuracy_comparison.png - порівняння accuracy")
    print("  - results/time_comparison.png - порівняння швидкості")
    print("  - results/confusion_matrices_*.png - confusion matrices")
    print("  - results/typical_errors_*.png - типові помилки класифікації")
    print("  - results/summary_all_results.csv - зведена таблиця")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

