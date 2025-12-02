"""
Бенчмаркінг алгоритмів неконтрольованого навчання (Unsupervised Learning)
Фокус: Візуалізація ітерацій K-Means та EM (Gaussian Mixture Models)

Мета: Порівняти різні алгоритми кластеризації на 2D даних
та візуалізувати процес навчання покроково.

Алгоритми:
- K-Means Clustering
- Gaussian Mixture Models (EM Algorithm)
- DBSCAN (для порівняння)
- Hierarchical Clustering (для порівняння)

Метрики:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Adjusted Rand Index (якщо є справжні мітки)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
)
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches

# Налаштування графіків
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

def print_section(title):
    """Виводить заголовок секції"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def generate_datasets():
    """Генерує різні типи 2D датасетів для кластеризації"""
    print_section("ГЕНЕРАЦІЯ ДАНИХ")
    
    np.random.seed(42)
    
    datasets = {}
    
    # 1. Blobs - чіткі гаусівські кластери (ідеально для K-Means і GMM)
    X_blobs, y_blobs = make_blobs(
        n_samples=600, 
        centers=4, 
        n_features=2,
        cluster_std=0.8,
        random_state=42
    )
    datasets['Blobs (4 кластери)'] = (X_blobs, y_blobs)
    print("✓ Згенеровано датасет 'Blobs': 600 точок, 4 кластери")
    
    # 2. Blobs з різною дисперсією
    X_varied, y_varied = make_blobs(
        n_samples=600,
        centers=5,
        n_features=2,
        cluster_std=[0.5, 1.0, 1.5, 0.7, 1.2],
        random_state=42
    )
    datasets['Blobs (різна дисперсія)'] = (X_varied, y_varied)
    print("✓ Згенеровано датасет 'Varied Blobs': 600 точок, 5 кластерів")
    
    # 3. Anisotropic - витягнуті кластери
    random_state = np.random.RandomState(42)
    X_aniso = random_state.randn(600, 2)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    y_aniso = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_aniso)
    datasets['Anisotropic'] = (X_aniso, y_aniso)
    print("✓ Згенеровано датасет 'Anisotropic': 600 точок, 3 кластери")
    
    return datasets


def visualize_original_data(datasets):
    """Візуалізує оригінальні дані"""
    print_section("ВІЗУАЛІЗАЦІЯ ВИХІДНИХ ДАНИХ")
    
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for idx, (name, (X, y)) in enumerate(datasets.items()):
        ax = axes[idx]
        
        # Малюємо точки
        for i in range(len(np.unique(y))):
            mask = y == i
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=colors[i % len(colors)], 
                      s=50, alpha=0.6, 
                      edgecolors='black', linewidth=0.5,
                      label=f'Клас {i}')
        
        ax.set_title(f'{name}\n({len(X)} точок, {len(np.unique(y))} класів)', 
                    fontsize=12, weight='bold')
        ax.set_xlabel('X₁', fontsize=11)
        ax.set_ylabel('X₂', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Розподіл згенерованих датасетів', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('original_data_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: original_data_distribution.png")
    plt.show()


class KMeansVisualizer:
    """Клас для візуалізації ітерацій K-Means"""
    
    def __init__(self, n_clusters=4, max_iter=20):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids_history = []
        self.labels_history = []
        self.inertia_history = []
    
    def fit(self, X):
        """Навчання K-Means зі збереженням історії"""
        # Ініціалізація центроїдів (k-means++)
        np.random.seed(42)
        n_samples = X.shape[0]
        
        # Перший центроїд випадковий
        centroids = [X[np.random.randint(n_samples)]]
        
        # Решта центроїдів за алгоритмом k-means++
        for _ in range(1, self.n_clusters):
            distances = cdist(X, centroids, 'euclidean')
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances ** 2
            probabilities /= probabilities.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.random()
            for idx, cum_prob in enumerate(cumulative_probs):
                if r < cum_prob:
                    centroids.append(X[idx])
                    break
        
        centroids = np.array(centroids)
        
        # Ітеративний процес
        for iteration in range(self.max_iter):
            # E-step: призначення точок до найближчих центроїдів
            distances = cdist(X, centroids, 'euclidean')
            labels = np.argmin(distances, axis=1)
            
            # Обчислення inertia
            inertia = np.sum([distances[i, labels[i]]**2 for i in range(len(X))])
            
            # Збереження історії
            self.centroids_history.append(centroids.copy())
            self.labels_history.append(labels.copy())
            self.inertia_history.append(inertia)
            
            # M-step: оновлення центроїдів
            new_centroids = np.array([X[labels == k].mean(axis=0) 
                                      for k in range(self.n_clusters)])
            
            # Перевірка на збіжність
            if np.allclose(centroids, new_centroids, rtol=1e-6):
                print(f"  Збіжність досягнута на ітерації {iteration + 1}")
                break
            
            centroids = new_centroids
        
        self.final_centroids = centroids
        self.final_labels = labels
        
        return self
    
    def visualize_iterations(self, X, save_path='kmeans_iterations.png'):
        """Візуалізація ключових ітерацій"""
        # Вибираємо ключові ітерації для відображення
        iterations_to_show = [0, 1, 2, 5, len(self.centroids_history)-1]
        iterations_to_show = [i for i in iterations_to_show if i < len(self.centroids_history)]
        
        n_plots = len(iterations_to_show)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for plot_idx, iter_idx in enumerate(iterations_to_show):
            ax = axes[plot_idx]
            centroids = self.centroids_history[iter_idx]
            labels = self.labels_history[iter_idx]
            inertia = self.inertia_history[iter_idx]
            
            # Малюємо точки
            for k in range(self.n_clusters):
                mask = labels == k
                ax.scatter(X[mask, 0], X[mask, 1],
                          c=colors[k % len(colors)],
                          s=50, alpha=0.5,
                          edgecolors='black', linewidth=0.3)
            
            # Малюємо центроїди
            ax.scatter(centroids[:, 0], centroids[:, 1],
                      c='red', s=300, alpha=0.9,
                      marker='*', edgecolors='black', linewidth=2,
                      label='Центроїди', zorder=10)
            
            # Малюємо лінії від попередніх центроїдів (якщо є)
            if iter_idx > 0:
                prev_centroids = self.centroids_history[iter_idx - 1]
                for k in range(self.n_clusters):
                    ax.plot([prev_centroids[k, 0], centroids[k, 0]],
                           [prev_centroids[k, 1], centroids[k, 1]],
                           'k--', alpha=0.3, linewidth=1)
            
            ax.set_title(f'Ітерація {iter_idx + 1}\nInertia: {inertia:.2f}',
                        fontsize=11, weight='bold')
            ax.set_xlabel('X₁', fontsize=10)
            ax.set_ylabel('X₂', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'K-Means: Візуалізація ітерацій (k={self.n_clusters})',
                    fontsize=14, weight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Збережено: {save_path}")
        plt.show()
    
    def plot_inertia_curve(self):
        """Графік зміни inertia"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.inertia_history) + 1), 
                self.inertia_history, 
                marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        plt.xlabel('Ітерація', fontsize=12, weight='bold')
        plt.ylabel('Inertia (сума квадратів відстаней)', fontsize=12, weight='bold')
        plt.title('Збіжність K-Means: зміна Inertia', fontsize=14, weight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('kmeans_convergence.png', dpi=300, bbox_inches='tight')
        print("✓ Збережено: kmeans_convergence.png")
        plt.show()


class GMMVisualizer:
    """Клас для візуалізації ітерацій Gaussian Mixture Models (EM)"""
    
    def __init__(self, n_components=4, max_iter=20):
        self.n_components = n_components
        self.max_iter = max_iter
        self.means_history = []
        self.covariances_history = []
        self.weights_history = []
        self.log_likelihood_history = []
    
    def fit(self, X):
        """Навчання GMM зі збереженням історії"""
        # Використовуємо sklearn GMM з модифікацією
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=1,  # По одній ітерації за раз
            random_state=42,
            warm_start=True
        )
        
        # Ініціалізація
        gmm.fit(X)
        
        for iteration in range(self.max_iter):
            # Зберігаємо параметри
            self.means_history.append(gmm.means_.copy())
            self.covariances_history.append(gmm.covariances_.copy())
            self.weights_history.append(gmm.weights_.copy())
            
            # Обчислюємо log-likelihood
            log_likelihood = gmm.score(X) * len(X)
            self.log_likelihood_history.append(log_likelihood)
            
            # Робимо ще одну ітерацію EM
            old_means = gmm.means_.copy()
            gmm.max_iter = iteration + 2
            gmm.fit(X)
            
            # Перевірка на збіжність
            if np.allclose(old_means, gmm.means_, rtol=1e-4):
                print(f"  Збіжність досягнута на ітерації {iteration + 1}")
                break
        
        self.gmm = gmm
        self.final_labels = gmm.predict(X)
        
        return self
    
    def visualize_iterations(self, X, save_path='gmm_iterations.png'):
        """Візуалізація ключових ітерацій GMM"""
        iterations_to_show = [0, 1, 2, 5, len(self.means_history)-1]
        iterations_to_show = [i for i in iterations_to_show if i < len(self.means_history)]
        
        n_plots = len(iterations_to_show)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for plot_idx, iter_idx in enumerate(iterations_to_show):
            ax = axes[plot_idx]
            means = self.means_history[iter_idx]
            covariances = self.covariances_history[iter_idx]
            weights = self.weights_history[iter_idx]
            log_like = self.log_likelihood_history[iter_idx]
            
            # Малюємо всі точки сірим
            ax.scatter(X[:, 0], X[:, 1], c='lightgray', s=20, alpha=0.4)
            
            # Малюємо гаусіани
            for k in range(self.n_components):
                mean = means[k]
                cov = covariances[k]
                
                # Малюємо еліпси (1, 2, 3 sigma)
                for n_std in [1, 2, 3]:
                    self._plot_gaussian_ellipse(ax, mean, cov, n_std, 
                                                colors[k % len(colors)])
                
                # Центр гаусіана
                ax.scatter(mean[0], mean[1], 
                          c=colors[k % len(colors)], 
                          s=200, marker='*',
                          edgecolors='black', linewidth=2,
                          zorder=10, alpha=0.9)
                
                # Підпис з вагою
                ax.text(mean[0], mean[1] + 0.3, 
                       f'w={weights[k]:.2f}',
                       ha='center', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.set_title(f'Ітерація {iter_idx + 1}\nLog-Likelihood: {log_like:.1f}',
                        fontsize=11, weight='bold')
            ax.set_xlabel('X₁', fontsize=10)
            ax.set_ylabel('X₂', fontsize=10)
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'GMM (EM): Візуалізація ітерацій (k={self.n_components})',
                    fontsize=14, weight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Збережено: {save_path}")
        plt.show()
    
    def _plot_gaussian_ellipse(self, ax, mean, cov, n_std, color):
        """Малює еліпс для гаусіана"""
        # Власні значення та власні вектори
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # Кут повороту
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Ширина та висота еліпса
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        # Малюємо еліпс
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor=color, alpha=0.1,
                         edgecolor=color, linewidth=1.5 if n_std == 2 else 0.5)
        ax.add_patch(ellipse)
    
    def plot_log_likelihood_curve(self):
        """Графік зміни log-likelihood"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.log_likelihood_history) + 1), 
                self.log_likelihood_history, 
                marker='o', linewidth=2, markersize=8, color='#4ECDC4')
        plt.xlabel('Ітерація', fontsize=12, weight='bold')
        plt.ylabel('Log-Likelihood', fontsize=12, weight='bold')
        plt.title('Збіжність GMM (EM): зміна Log-Likelihood', fontsize=14, weight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('gmm_convergence.png', dpi=300, bbox_inches='tight')
        print("✓ Збережено: gmm_convergence.png")
        plt.show()


def benchmark_clustering_algorithms(X, y_true, dataset_name):
    """Бенчмаркінг різних алгоритмів кластеризації"""
    print(f"\n{'='*70}")
    print(f"  БЕНЧМАРКІНГ: {dataset_name}")
    print('='*70)
    
    n_clusters = len(np.unique(y_true))
    
    # Алгоритми для тестування
    algorithms = {
        'K-Means': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'GMM (EM)': GaussianMixture(n_components=n_clusters, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Hierarchical': AgglomerativeClustering(n_clusters=n_clusters)
    }
    
    results = {}
    predictions = {}
    
    for name, algorithm in algorithms.items():
        print(f"\n[{list(algorithms.keys()).index(name) + 1}/{len(algorithms)}] Навчання: {name}...")
        
        # Навчання
        if name == 'GMM (EM)':
            algorithm.fit(X)
            labels = algorithm.predict(X)
        else:
            labels = algorithm.fit_predict(X)
        
        predictions[name] = labels
        
        # Обчислення метрик (тільки якщо немає шумових точок -1)
        if len(np.unique(labels)) > 1 and -1 not in labels:
            results[name] = {
                'Silhouette Score': silhouette_score(X, labels),
                'Davies-Bouldin Index': davies_bouldin_score(X, labels),
                'Calinski-Harabasz Index': calinski_harabasz_score(X, labels),
                'Adjusted Rand Index': adjusted_rand_score(y_true, labels),
                'Normalized Mutual Info': normalized_mutual_info_score(y_true, labels),
                'N Clusters': len(np.unique(labels))
            }
            print(f"  ✓ Silhouette: {results[name]['Silhouette Score']:.4f}")
            print(f"  ✓ ARI: {results[name]['Adjusted Rand Index']:.4f}")
        else:
            print(f"  ⚠ Алгоритм знайшов {len(np.unique(labels))} кластерів (є шумові точки)")
            results[name] = {
                'Silhouette Score': np.nan,
                'Davies-Bouldin Index': np.nan,
                'Calinski-Harabasz Index': np.nan,
                'Adjusted Rand Index': adjusted_rand_score(y_true, labels),
                'Normalized Mutual Info': normalized_mutual_info_score(y_true, labels),
                'N Clusters': len(np.unique(labels))
            }
    
    # Таблиця порівняння
    print(f"\n📊 Порівняння алгоритмів:")
    results_df = pd.DataFrame(results).T
    print(results_df.round(4).to_string())
    
    return results, predictions, results_df


def visualize_clustering_results(X, predictions, dataset_name):
    """Візуалізація результатів кластеризації"""
    n_algorithms = len(predictions)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(5*n_algorithms, 4))
    
    if n_algorithms == 1:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#DDA0DD']
    
    for idx, (name, labels) in enumerate(predictions.items()):
        ax = axes[idx]
        
        unique_labels = np.unique(labels)
        
        for k in unique_labels:
            if k == -1:
                # Шумові точки (для DBSCAN)
                mask = labels == k
                ax.scatter(X[mask, 0], X[mask, 1],
                          c='black', s=20, alpha=0.3,
                          marker='x', label='Шум')
            else:
                mask = labels == k
                ax.scatter(X[mask, 0], X[mask, 1],
                          c=colors[k % len(colors)],
                          s=50, alpha=0.6,
                          edgecolors='black', linewidth=0.3,
                          label=f'Кластер {k}')
        
        ax.set_title(f'{name}\n({len(unique_labels)} кластерів)',
                    fontsize=11, weight='bold')
        ax.set_xlabel('X₁', fontsize=10)
        ax.set_ylabel('X₂', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Результати кластеризації: {dataset_name}',
                fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    
    safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'clustering_results_{safe_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Збережено: clustering_results_{safe_name}.png")
    plt.show()


def elbow_method_analysis(X, max_k=10):
    """Метод ліктя для визначення оптимальної кількості кластерів"""
    print_section("МЕТОД ЛІКТЯ (ELBOW METHOD)")
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_score(X, labels):.4f}")
    
    # Візуалізація
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve
    ax1.plot(K_range, inertias, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    ax1.set_xlabel('Кількість кластерів (k)', fontsize=12, weight='bold')
    ax1.set_ylabel('Inertia', fontsize=12, weight='bold')
    ax1.set_title('Метод ліктя: Inertia vs k', fontsize=13, weight='bold')
    ax1.grid(alpha=0.3)
    
    # Silhouette scores
    ax2.plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
    ax2.set_xlabel('Кількість кластерів (k)', fontsize=12, weight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, weight='bold')
    ax2.set_title('Silhouette Score vs k', fontsize=13, weight='bold')
    ax2.grid(alpha=0.3)
    
    # Оптимальне k
    optimal_k = K_range[np.argmax(silhouette_scores)]
    ax2.axvline(optimal_k, color='red', linestyle='--', linewidth=2, 
                label=f'Оптимальне k={optimal_k}')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    print("\n✓ Збережено: elbow_method.png")
    plt.show()
    
    print(f"\n✅ Оптимальна кількість кластерів (за Silhouette): k={optimal_k}")


def main():
    """Головна функція"""
    print("\n" + "="*70)
    print("  БЕНЧМАРКІНГ АЛГОРИТМІВ НЕКОНТРОЛЬОВАНОГО НАВЧАННЯ")
    print("  Фокус: K-Means та EM (Gaussian Mixture Models)")
    print("="*70)
    
    # 1. Генерація даних
    datasets = generate_datasets()
    
    # 2. Візуалізація оригінальних даних
    visualize_original_data(datasets)
    
    # Працюємо з першим датасетом для детальної візуалізації
    main_dataset_name = 'Blobs (4 кластери)'
    X, y_true = datasets[main_dataset_name]
    
    # Розділення на train/test
    print_section("РОЗДІЛЕННЯ ДАНИХ")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.3, random_state=42, stratify=y_true
    )
    print(f"✓ Train set: {len(X_train)} точок ({len(X_train)/len(X)*100:.1f}%)")
    print(f"✓ Test set:  {len(X_test)} точок ({len(X_test)/len(X)*100:.1f}%)")
    
    # Стандартизація (опціонально, але рекомендується)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Метод ліктя
    elbow_method_analysis(X_train_scaled, max_k=10)
    
    # 4. K-Means з візуалізацією ітерацій
    print_section("K-MEANS: ВІЗУАЛІЗАЦІЯ ІТЕРАЦІЙ")
    kmeans_viz = KMeansVisualizer(n_clusters=4, max_iter=20)
    kmeans_viz.fit(X_train_scaled)
    kmeans_viz.visualize_iterations(X_train_scaled, 'kmeans_iterations.png')
    kmeans_viz.plot_inertia_curve()
    
    # 5. GMM з візуалізацією ітерацій
    print_section("GMM (EM): ВІЗУАЛІЗАЦІЯ ІТЕРАЦІЙ")
    gmm_viz = GMMVisualizer(n_components=4, max_iter=20)
    gmm_viz.fit(X_train_scaled)
    gmm_viz.visualize_iterations(X_train_scaled, 'gmm_iterations.png')
    gmm_viz.plot_log_likelihood_curve()
    
    # 6. Бенчмаркінг на всіх датасетах
    all_results = {}
    for dataset_name, (X_data, y_data) in datasets.items():
        # Стандартизація
        X_scaled = scaler.fit_transform(X_data)
        
        # Бенчмаркінг
        results, predictions, results_df = benchmark_clustering_algorithms(
            X_scaled, y_data, dataset_name
        )
        all_results[dataset_name] = results_df
        
        # Візуалізація результатів
        visualize_clustering_results(X_scaled, predictions, dataset_name)
    
    # 7. Зведена таблиця результатів
    print_section("ЗВЕДЕНІ РЕЗУЛЬТАТИ")
    
    for dataset_name, results_df in all_results.items():
        print(f"\n{'='*70}")
        print(f"  {dataset_name}")
        print('='*70)
        print(results_df.to_string())
        
        # Збереження CSV
        safe_name = dataset_name.replace(' ', '_').replace('(', '').replace(')', '')
        results_df.to_csv(f'results_{safe_name}.csv')
        print(f"✓ Збережено: results_{safe_name}.csv")
    
    # Підсумок
    print_section("ПІДСУМОК")
    print("\n✅ Аналіз завершено!")
    print("\n📁 Створені файли:")
    print("  - original_data_distribution.png - вихідні дані")
    print("  - elbow_method.png - метод ліктя")
    print("  - kmeans_iterations.png - ітерації K-Means")
    print("  - kmeans_convergence.png - збіжність K-Means")
    print("  - gmm_iterations.png - ітерації GMM (EM)")
    print("  - gmm_convergence.png - збіжність GMM")
    print("  - clustering_results_*.png - результати кластеризації")
    print("  - results_*.csv - метрики для кожного датасету")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

