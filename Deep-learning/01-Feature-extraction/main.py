"""
Бенчмаркінг екстракторів ознак (Feature Extractors)
Фокус: Візуалізація CNN фільтрів, активацій, порівняння якості

Мета: Порівняти різні feature extractors на якість кластеризації
та візуалізувати внутрішні представлення нейронних мереж.

Екстрактори:
- Власний CNN (простий)
- VGG16 (pretrained)
- ResNet50 (pretrained)
- MobileNetV2 (pretrained)
- Autoencoder (власний)

Візуалізації:
- CNN фільтри (перші шари)
- Активації проміжних шарів
- PCA (2D/3D проекції)
- t-SNE (2D/3D проекції)

Метрики:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Adjusted Rand Index
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
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

import time
from pathlib import Path

# Налаштування TensorFlow
tf.random.set_seed(42)
np.random.seed(42)

# Налаштування графіків
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 8)
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


def load_datasets():
    """Завантажує датасети з різним рівнем складності"""
    print_section("ЗАВАНТАЖЕННЯ ДАТАСЕТІВ")
    
    datasets = {}
    
    # 1. MNIST - легко розділити (10 класів цифр)
    print("\n[1/3] Завантаження MNIST...")
    (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
    
    # Нормалізація та решейп
    X_train_mnist = X_train_mnist.astype('float32') / 255.0
    X_test_mnist = X_test_mnist.astype('float32') / 255.0
    X_train_mnist = np.expand_dims(X_train_mnist, -1)
    X_test_mnist = np.expand_dims(X_test_mnist, -1)
    
    # Візьмемо підмножину для швидкості
    indices_train = np.random.choice(len(X_train_mnist), 6000, replace=False)
    indices_test = np.random.choice(len(X_test_mnist), 2000, replace=False)
    
    datasets['MNIST (легко)'] = {
        'X_train': X_train_mnist[indices_train],
        'y_train': y_train_mnist[indices_train],
        'X_test': X_test_mnist[indices_test],
        'y_test': y_test_mnist[indices_test],
        'n_classes': 10,
        'shape': (28, 28, 1),
        'description': 'Цифри 0-9, чіткі границі між класами'
    }
    print(f"  ✓ MNIST: train={len(datasets['MNIST (легко)']['X_train'])}, test={len(datasets['MNIST (легко)']['X_test'])}")
    
    # 2. Fashion-MNIST - важче розділити (10 класів одягу)
    print("\n[2/3] Завантаження Fashion-MNIST...")
    (X_train_fmnist, y_train_fmnist), (X_test_fmnist, y_test_fmnist) = fashion_mnist.load_data()
    
    X_train_fmnist = X_train_fmnist.astype('float32') / 255.0
    X_test_fmnist = X_test_fmnist.astype('float32') / 255.0
    X_train_fmnist = np.expand_dims(X_train_fmnist, -1)
    X_test_fmnist = np.expand_dims(X_test_fmnist, -1)
    
    datasets['Fashion-MNIST (важко)'] = {
        'X_train': X_train_fmnist[indices_train],
        'y_train': y_train_fmnist[indices_train],
        'X_test': X_test_fmnist[indices_test],
        'y_test': y_test_fmnist[indices_test],
        'n_classes': 10,
        'shape': (28, 28, 1),
        'description': 'Одяг, складніша текстура'
    }
    print(f"  ✓ Fashion-MNIST: train={len(datasets['Fashion-MNIST (важко)']['X_train'])}, test={len(datasets['Fashion-MNIST (важко)']['X_test'])}")
    
    # 3. CIFAR-10 - середня складність, колірні зображення
    print("\n[3/3] Завантаження CIFAR-10...")
    (X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
    
    X_train_cifar = X_train_cifar.astype('float32') / 255.0
    X_test_cifar = X_test_cifar.astype('float32') / 255.0
    y_train_cifar = y_train_cifar.flatten()
    y_test_cifar = y_test_cifar.flatten()
    
    # Підмножина
    indices_train_cifar = np.random.choice(len(X_train_cifar), 5000, replace=False)
    indices_test_cifar = np.random.choice(len(X_test_cifar), 1500, replace=False)
    
    datasets['CIFAR-10 (колірні)'] = {
        'X_train': X_train_cifar[indices_train_cifar],
        'y_train': y_train_cifar[indices_train_cifar],
        'X_test': X_test_cifar[indices_test_cifar],
        'y_test': y_test_cifar[indices_test_cifar],
        'n_classes': 10,
        'shape': (32, 32, 3),
        'description': 'Об\'єкти в природному середовищі'
    }
    print(f"  ✓ CIFAR-10: train={len(datasets['CIFAR-10 (колірні)']['X_train'])}, test={len(datasets['CIFAR-10 (колірні)']['X_test'])}")
    
    return datasets


def visualize_datasets(datasets):
    """Візуалізує приклади з кожного датасету"""
    print_section("ВІЗУАЛІЗАЦІЯ ДАТАСЕТІВ")
    
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 10, figsize=(15, 3 * n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (dataset_name, data) in enumerate(datasets.items()):
        X_train = data['X_train']
        y_train = data['y_train']
        n_classes = data['n_classes']
        
        # Візьмемо по одному прикладу кожного класу
        for class_idx in range(min(10, n_classes)):
            ax = axes[row_idx, class_idx]
            
            # Знайдемо перший приклад цього класу
            idx = np.where(y_train == class_idx)[0][0]
            img = X_train[idx]
            
            if img.shape[-1] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
            
            ax.set_title(f'Клас {class_idx}', fontsize=9)
            ax.axis('off')
        
        # Заголовок рядка
        axes[row_idx, 0].text(-0.5, 0.5, dataset_name, 
                              transform=axes[row_idx, 0].transAxes,
                              fontsize=12, weight='bold', 
                              rotation=90, va='center')
    
    plt.suptitle('Приклади зображень з датасетів', fontsize=14, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_samples.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: results/dataset_samples.png")
    plt.show()


class CustomCNN:
    """Власний простий CNN екстрактор"""
    
    def __init__(self, input_shape, n_classes=10, name="CustomCNN"):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.name = name
        self.model = None
        self.feature_extractor = None
    
    def build(self):
        """Будує архітектуру CNN"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Блок 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        
        # Блок 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        
        # Блок 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = layers.GlobalAveragePooling2D(name='gap')(x)
        
        # Feature vector
        features = layers.Dense(256, activation='relu', name='features')(x)
        features = layers.Dropout(0.5)(features)
        
        # Класифікатор
        outputs = layers.Dense(self.n_classes, activation='softmax', name='output')(features)
        
        self.model = models.Model(inputs, outputs, name=self.name)
        
        # Feature extractor (до останнього Dense шару)
        self.feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('features').output
        )
        
        return self
    
    def compile_and_train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
        """Компілює та навчає модель"""
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n  Навчання {self.name}...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        print(f"  ✓ Train accuracy: {train_acc:.4f}")
        print(f"  ✓ Val accuracy: {val_acc:.4f}")
        
        return history
    
    def extract_features(self, X):
        """Витягує ознаки з даних"""
        return self.feature_extractor.predict(X, verbose=0)


class AutoencoderExtractor:
    """Автоенкодер для екстракції ознак"""
    
    def __init__(self, input_shape, latent_dim=128, name="Autoencoder"):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.name = name
        self.autoencoder = None
        self.encoder = None
    
    def build(self):
        """Будує автоенкодер"""
        # Encoder
        encoder_input = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        self.encoder = models.Model(encoder_input, latent, name='encoder')
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        
        # Обчислюємо розмір після pooling
        h = self.input_shape[0] // 4
        w = self.input_shape[1] // 4
        
        x = layers.Dense(h * w * 64, activation='relu')(latent_inputs)
        x = layers.Reshape((h, w, 64))(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        
        # Фінальний шар має відповідати input_shape
        decoder_output = layers.Conv2D(self.input_shape[-1], (3, 3), 
                                       activation='sigmoid', padding='same')(x)
        
        decoder = models.Model(latent_inputs, decoder_output, name='decoder')
        
        # Повний автоенкодер
        autoencoder_output = decoder(latent)
        self.autoencoder = models.Model(encoder_input, autoencoder_output, name=self.name)
        
        return self
    
    def compile_and_train(self, X_train, X_val, epochs=20, batch_size=128):
        """Компілює та навчає автоенкодер"""
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse'
        )
        
        print(f"\n  Навчання {self.name}...")
        history = self.autoencoder.fit(
            X_train, X_train,  # Вхід = вихід
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        print(f"  ✓ Train loss: {train_loss:.6f}")
        print(f"  ✓ Val loss: {val_loss:.6f}")
        
        return history
    
    def extract_features(self, X):
        """Витягує ознаки (latent representation)"""
        return self.encoder.predict(X, verbose=0)


def build_pretrained_extractor(model_name, input_shape):
    """Будує feature extractor на основі pretrained моделі"""
    
    # Для grayscale зображень треба конвертувати в RGB
    needs_rgb = (input_shape[-1] == 1)
    
    # Створюємо wrapper для конвертації
    if needs_rgb:
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)  # Перетворення в RGB
    else:
        inputs = keras.Input(shape=input_shape)
        x = inputs
    
    # Ресайз для pretrained моделей (мінімум 32x32)
    target_size = max(32, input_shape[0])
    if input_shape[0] != target_size or input_shape[1] != target_size:
        x = layers.Resizing(target_size, target_size)(x)
    
    # Завантажуємо pretrained модель
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=(target_size, target_size, 3))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False,
                             input_shape=(target_size, target_size, 3))
    elif model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                input_shape=(target_size, target_size, 3))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    base_model.trainable = False  # Заморожуємо ваги
    
    # Додаємо base_model
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Feature extractor
    feature_extractor = models.Model(inputs, x, name=f'{model_name}_extractor')
    
    return feature_extractor


def visualize_conv_filters(model, layer_name, n_filters=32):
    """Візуалізує фільтри конволюційного шару"""
    
    # Отримуємо ваги шару
    layer = model.get_layer(layer_name)
    filters, biases = layer.get_weights()
    
    # Нормалізуємо фільтри
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min + 1e-8)
    
    # Візуалізуємо перші n_filters
    n_filters = min(n_filters, filters.shape[-1])
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
    axes = axes.flatten()
    
    for i in range(n_filters):
        ax = axes[i]
        f = filters[:, :, :, i]
        
        # Якщо є кілька каналів, візьмемо середнє
        if f.shape[-1] > 1:
            f = f.mean(axis=-1)
        else:
            f = f.squeeze()
        
        ax.imshow(f, cmap='viridis')
        ax.set_title(f'F{i+1}', fontsize=8)
        ax.axis('off')
    
    # Вимикаємо зайві осі
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Фільтри шару: {layer_name}', fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig


def visualize_feature_maps(model, layer_name, image, n_maps=32):
    """Візуалізує feature maps (активації) для конкретного зображення"""
    
    # Створюємо модель для отримання активацій
    activation_model = models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Отримуємо активації
    activations = activation_model.predict(np.expand_dims(image, 0), verbose=0)
    activations = activations[0]  # Перше зображення
    
    # Візуалізуємо перші n_maps
    n_maps = min(n_maps, activations.shape[-1])
    n_cols = 8
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
    axes = axes.flatten()
    
    for i in range(n_maps):
        ax = axes[i]
        feature_map = activations[:, :, i]
        
        ax.imshow(feature_map, cmap='viridis')
        ax.set_title(f'Map {i+1}', fontsize=8)
        ax.axis('off')
    
    # Вимикаємо зайві осі
    for i in range(n_maps, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Активації шару: {layer_name}', fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig


def visualize_cnn_internals(custom_cnn, X_test):
    """Візуалізує внутрішні представлення CNN"""
    print_section("ВІЗУАЛІЗАЦІЯ CNN: ФІЛЬТРИ ТА АКТИВАЦІЇ")
    
    model = custom_cnn.model
    
    # 1. Візуалізуємо фільтри першого conv шару
    print("\n  Візуалізація фільтрів conv1...")
    fig1 = visualize_conv_filters(model, 'conv1', n_filters=32)
    plt.savefig(OUTPUT_DIR / 'cnn_filters_conv1.png', dpi=300, bbox_inches='tight')
    print("  ✓ Збережено: results/cnn_filters_conv1.png")
    plt.close()
    
    # 2. Візуалізуємо активації для прикладу
    print("\n  Візуалізація активацій...")
    test_image = X_test[0]
    
    # Показуємо оригінальне зображення
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Оригінал
    if test_image.shape[-1] == 1:
        axes[0].imshow(test_image.squeeze(), cmap='gray')
    else:
        axes[0].imshow(test_image)
    axes[0].set_title('Оригінальне зображення', fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # Активації різних шарів
    layer_names = ['conv1', 'conv2', 'conv3']
    for idx, layer_name in enumerate(layer_names):
        activation_model = models.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        activations = activation_model.predict(np.expand_dims(test_image, 0), verbose=0)[0]
        
        # Середнє по всіх feature maps
        mean_activation = activations.mean(axis=-1)
        
        axes[idx + 1].imshow(mean_activation, cmap='viridis')
        axes[idx + 1].set_title(f'Активації: {layer_name}', fontsize=12, weight='bold')
        axes[idx + 1].axis('off')
    
    plt.suptitle('Прогресивна екстракція ознак у CNN', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cnn_activations_progression.png', dpi=300, bbox_inches='tight')
    print("  ✓ Збережено: results/cnn_activations_progression.png")
    plt.close()
    
    # 3. Детальні feature maps для conv1
    fig2 = visualize_feature_maps(model, 'conv1', test_image, n_maps=32)
    plt.savefig(OUTPUT_DIR / 'cnn_feature_maps_conv1.png', dpi=300, bbox_inches='tight')
    print("  ✓ Збережено: results/cnn_feature_maps_conv1.png")
    plt.close()


def apply_pca(features, n_components=2):
    """Застосовує PCA для зменшення розмірності"""
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    
    return features_pca, explained_variance


def apply_tsne(features, n_components=2, perplexity=30):
    """Застосовує t-SNE для зменшення розмірності"""
    tsne = TSNE(n_components=n_components, random_state=42, 
                perplexity=min(perplexity, len(features) - 1))
    features_tsne = tsne.fit_transform(features)
    
    return features_tsne


def visualize_dimensionality_reduction(features, labels, method_name, extractor_name):
    """Візуалізує результати PCA/t-SNE"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 2D проекція
    ax = axes[0]
    scatter = ax.scatter(features[:, 0], features[:, 1], 
                        c=labels, cmap='tab10', 
                        s=20, alpha=0.6, edgecolors='black', linewidth=0.3)
    ax.set_title(f'{method_name} 2D проекція', fontsize=13, weight='bold')
    ax.set_xlabel(f'{method_name}1', fontsize=11)
    ax.set_ylabel(f'{method_name}2', fontsize=11)
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Клас')
    
    # 3D проекція (якщо є)
    if features.shape[1] >= 3:
        ax = fig.add_subplot(122, projection='3d')
        scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2],
                           c=labels, cmap='tab10',
                           s=20, alpha=0.6, edgecolors='black', linewidth=0.3)
        ax.set_title(f'{method_name} 3D проекція', fontsize=13, weight='bold')
        ax.set_xlabel(f'{method_name}1', fontsize=11)
        ax.set_ylabel(f'{method_name}2', fontsize=11)
        ax.set_zlabel(f'{method_name}3', fontsize=11)
    else:
        # Якщо немає 3D, робимо гістограму класів
        ax = axes[1]
        unique, counts = np.unique(labels, return_counts=True)
        ax.bar(unique, counts, color='steelblue', edgecolor='black')
        ax.set_title('Розподіл класів', fontsize=13, weight='bold')
        ax.set_xlabel('Клас', fontsize=11)
        ax.set_ylabel('Кількість зразків', fontsize=11)
        ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle(f'{method_name}: {extractor_name}', fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig


def benchmark_extractors(dataset_name, dataset, extractors):
    """Бенчмаркінг різних feature extractors"""
    print(f"\n{'='*80}")
    print(f"  БЕНЧМАРКІНГ ЕКСТРАКТОРІВ: {dataset_name}")
    print('='*80)
    
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    n_classes = dataset['n_classes']
    
    results = {}
    all_features = {}
    
    for extractor_name, extractor in extractors.items():
        print(f"\n[{list(extractors.keys()).index(extractor_name) + 1}/{len(extractors)}] Екстрактор: {extractor_name}")
        
        start_time = time.time()
        
        # Витягуємо ознаки
        features_train = extractor.predict(X_train, verbose=0)
        features_test = extractor.predict(X_test, verbose=0)
        
        extraction_time = time.time() - start_time
        
        # Flatten якщо потрібно
        if len(features_train.shape) > 2:
            features_train = features_train.reshape(len(features_train), -1)
            features_test = features_test.reshape(len(features_test), -1)
        
        print(f"  ✓ Розмірність ознак: {features_train.shape[1]}")
        print(f"  ✓ Час екстракції: {extraction_time:.2f}s")
        
        all_features[extractor_name] = (features_test, y_test)
        
        # Кластеризація з K-Means
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_test)
        
        # Метрики
        try:
            silhouette = silhouette_score(features_test, clusters)
            davies_bouldin = davies_bouldin_score(features_test, clusters)
            calinski = calinski_harabasz_score(features_test, clusters)
            ari = adjusted_rand_score(y_test, clusters)
            nmi = normalized_mutual_info_score(y_test, clusters)
            
            results[extractor_name] = {
                'Розмірність': features_train.shape[1],
                'Час (s)': extraction_time,
                'Silhouette': silhouette,
                'Davies-Bouldin': davies_bouldin,
                'Calinski-Harabasz': calinski,
                'ARI': ari,
                'NMI': nmi
            }
            
            print(f"  ✓ Silhouette: {silhouette:.4f}")
            print(f"  ✓ ARI: {ari:.4f}")
            print(f"  ✓ NMI: {nmi:.4f}")
            
        except Exception as e:
            print(f"  ⚠ Помилка при обчисленні метрик: {e}")
            results[extractor_name] = {
                'Розмірність': features_train.shape[1],
                'Час (s)': extraction_time,
                'Silhouette': np.nan,
                'Davies-Bouldin': np.nan,
                'Calinski-Harabasz': np.nan,
                'ARI': np.nan,
                'NMI': np.nan
            }
    
    # Таблиця результатів
    print(f"\n📊 Порівняльна таблиця:")
    results_df = pd.DataFrame(results).T
    print(results_df.round(4).to_string())
    
    return results_df, all_features


def visualize_all_projections(all_features, dataset_name):
    """Візуалізує PCA та t-SNE для всіх екстракторів"""
    print_section(f"ВІЗУАЛІЗАЦІЯ PCA ТА t-SNE: {dataset_name}")
    
    n_extractors = len(all_features)
    
    # PCA візуалізації
    print("\n  Обчислення PCA проекцій...")
    fig_pca, axes_pca = plt.subplots(2, (n_extractors + 1) // 2, 
                                     figsize=(8 * ((n_extractors + 1) // 2), 14))
    axes_pca = axes_pca.flatten()
    
    for idx, (extractor_name, (features, labels)) in enumerate(all_features.items()):
        ax = axes_pca[idx]
        
        # PCA
        features_pca, var_explained = apply_pca(features, n_components=2)
        
        scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1],
                           c=labels, cmap='tab10',
                           s=15, alpha=0.6, edgecolors='black', linewidth=0.2)
        ax.set_title(f'{extractor_name}\nVar explained: {var_explained:.2%}', 
                    fontsize=11, weight='bold')
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.grid(alpha=0.3)
        
        if idx == 0:
            plt.colorbar(scatter, ax=ax, label='Клас')
    
    # Вимикаємо зайві осі
    for idx in range(n_extractors, len(axes_pca)):
        axes_pca[idx].axis('off')
    
    plt.suptitle(f'PCA Проекції: {dataset_name}', fontsize=16, weight='bold')
    plt.tight_layout()
    
    safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / f'pca_projections_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Збережено: results/pca_projections_{safe_name}.png")
    plt.show()
    
    # t-SNE візуалізації
    print("\n  Обчислення t-SNE проекцій (це може зайняти час)...")
    fig_tsne, axes_tsne = plt.subplots(2, (n_extractors + 1) // 2,
                                       figsize=(8 * ((n_extractors + 1) // 2), 14))
    axes_tsne = axes_tsne.flatten()
    
    for idx, (extractor_name, (features, labels)) in enumerate(all_features.items()):
        ax = axes_tsne[idx]
        
        # t-SNE (беремо підвибірку для швидкості)
        if len(features) > 1000:
            indices = np.random.choice(len(features), 1000, replace=False)
            features_subset = features[indices]
            labels_subset = labels[indices]
        else:
            features_subset = features
            labels_subset = labels
        
        features_tsne = apply_tsne(features_subset, n_components=2, perplexity=30)
        
        scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1],
                           c=labels_subset, cmap='tab10',
                           s=15, alpha=0.6, edgecolors='black', linewidth=0.2)
        ax.set_title(f'{extractor_name}', fontsize=11, weight='bold')
        ax.set_xlabel('t-SNE1', fontsize=10)
        ax.set_ylabel('t-SNE2', fontsize=10)
        ax.grid(alpha=0.3)
        
        if idx == 0:
            plt.colorbar(scatter, ax=ax, label='Клас')
    
    # Вимикаємо зайві осі
    for idx in range(n_extractors, len(axes_tsne)):
        axes_tsne[idx].axis('off')
    
    plt.suptitle(f't-SNE Проекції: {dataset_name}', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'tsne_projections_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Збережено: results/tsne_projections_{safe_name}.png")
    plt.show()


def create_comparison_chart(all_results):
    """Створює порівняльні діаграми для всіх датасетів"""
    print_section("ЗВЕДЕНЕ ПОРІВНЯННЯ ЕКСТРАКТОРІВ")
    
    metrics = ['Silhouette', 'ARI', 'NMI']
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Збираємо дані для всіх датасетів
        data_to_plot = []
        labels = []
        
        for dataset_name, results_df in all_results.items():
            if metric in results_df.columns:
                data_to_plot.append(results_df[metric].values)
                labels.append(dataset_name)
        
        if data_to_plot:
            # Box plot
            positions = np.arange(len(data_to_plot))
            bp = ax.boxplot(data_to_plot, positions=positions, 
                          patch_artist=True, widths=0.6)
            
            # Кольори
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.set_ylabel(metric, fontsize=12, weight='bold')
            ax.set_title(f'Порівняння: {metric}', fontsize=13, weight='bold')
            ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Бенчмарк якості екстракторів ознак', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'extractors_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Збережено: results/extractors_comparison.png")
    plt.show()


def main():
    """Головна функція"""
    print("\n" + "="*80)
    print("  БЕНЧМАРКІНГ ЕКСТРАКТОРІВ ОЗНАК (FEATURE EXTRACTORS)")
    print("  Візуалізація CNN, PCA, t-SNE та порівняння якості")
    print("="*80)
    
    # 1. Завантаження датасетів
    datasets = load_datasets()
    
    # 2. Візуалізація датасетів
    visualize_datasets(datasets)
    
    # 3. Створення та навчання екстракторів
    all_results = {}
    
    for dataset_name, dataset in datasets.items():
        print_section(f"РОБОТА З ДАТАСЕТОМ: {dataset_name}")
        
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        input_shape = dataset['shape']
        n_classes = dataset['n_classes']
        
        extractors = {}
        
        # 3.1. Власний CNN
        print("\n[1/5] Створення власного CNN...")
        custom_cnn = CustomCNN(input_shape, n_classes, "CustomCNN")
        custom_cnn.build()
        custom_cnn.compile_and_train(X_train, y_train, X_test, y_test, epochs=5)
        extractors['Custom CNN'] = custom_cnn.feature_extractor
        
        # Візуалізація внутрішніх представлень CNN (тільки для першого датасету)
        if dataset_name == list(datasets.keys())[0]:
            visualize_cnn_internals(custom_cnn, X_test)
        
        # 3.2. Autoencoder
        print("\n[2/5] Створення Autoencoder...")
        autoencoder = AutoencoderExtractor(input_shape, latent_dim=128, name="Autoencoder")
        autoencoder.build()
        autoencoder.compile_and_train(X_train, X_test, epochs=10)
        extractors['Autoencoder'] = autoencoder.encoder
        
        # 3.3. Pretrained моделі (тільки для CIFAR-10 або якщо розмір достатній)
        if input_shape[0] >= 28:
            print("\n[3/5] Завантаження VGG16...")
            try:
                vgg16_extractor = build_pretrained_extractor('VGG16', input_shape)
                extractors['VGG16'] = vgg16_extractor
                print("  ✓ VGG16 завантажено")
            except Exception as e:
                print(f"  ⚠ Помилка при завантаженні VGG16: {e}")
            
            print("\n[4/5] Завантаження ResNet50...")
            try:
                resnet_extractor = build_pretrained_extractor('ResNet50', input_shape)
                extractors['ResNet50'] = resnet_extractor
                print("  ✓ ResNet50 завантажено")
            except Exception as e:
                print(f"  ⚠ Помилка при завантаженні ResNet50: {e}")
            
            print("\n[5/5] Завантаження MobileNetV2...")
            try:
                mobilenet_extractor = build_pretrained_extractor('MobileNetV2', input_shape)
                extractors['MobileNetV2'] = mobilenet_extractor
                print("  ✓ MobileNetV2 завантажено")
            except Exception as e:
                print(f"  ⚠ Помилка при завантаженні MobileNetV2: {e}")
        
        # 4. Бенчмаркінг екстракторів
        results_df, all_features = benchmark_extractors(dataset_name, dataset, extractors)
        all_results[dataset_name] = results_df
        
        # Збереження результатів
        safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
        results_df.to_csv(OUTPUT_DIR / f'benchmark_{safe_name}.csv')
        print(f"\n✓ Збережено: results/benchmark_{safe_name}.csv")
        
        # 5. Візуалізація PCA та t-SNE
        visualize_all_projections(all_features, dataset_name)
    
    # 6. Зведене порівняння
    create_comparison_chart(all_results)
    
    # Підсумок
    print_section("ПІДСУМОК")
    print("\n✅ Аналіз завершено!")
    print("\n📁 Створені файли:")
    print("  - results/dataset_samples.png - приклади датасетів")
    print("  - results/cnn_filters_conv1.png - фільтри першого шару")
    print("  - results/cnn_activations_progression.png - прогресія активацій")
    print("  - results/cnn_feature_maps_conv1.png - детальні feature maps")
    print("  - results/pca_projections_*.png - PCA проекції")
    print("  - results/tsne_projections_*.png - t-SNE проекції")
    print("  - results/extractors_comparison.png - порівняння екстракторів")
    print("  - results/benchmark_*.csv - детальні метрики")
    
    print("\n📊 Зведена таблиця результатів:")
    for dataset_name, results_df in all_results.items():
        print(f"\n{dataset_name}:")
        print(results_df[['Silhouette', 'ARI', 'NMI']].round(4).to_string())
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

