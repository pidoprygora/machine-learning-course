"""
Бенчмаркінг лінійної дискримінації та гаусівської моделі
Датасет: Wine dataset (sklearn.datasets.load_wine)

Мета: Порівняти різні класифікатори для розпізнавання типів вина
за хімічними властивостями.

Класифікатори:
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Gaussian Naive Bayes (GaussianNB)

Метрики:
- Accuracy, Balanced Accuracy, Precision, Recall, F1-score
- ROC-AUC (One-vs-Rest для мультикласової класифікації)
- Confusion Matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc, confusion_matrix, 
    classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from scipy import stats
from itertools import cycle

# Налаштування графіків
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams['font.size'] = 10

def print_section(title):
    """Виводить заголовок секції"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_and_explore_data():
    """Завантаження та первинний аналіз датасету Wine"""
    print_section("ЗАВАНТАЖЕННЯ ДАНИХ")
    
    # Завантаження датасету Wine
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    print(f"✓ Датасет Wine завантажено")
    print(f"  - Кількість зразків: {X.shape[0]}")
    print(f"  - Кількість ознак: {X.shape[1]}")
    print(f"  - Класи: {target_names}")
    print(f"\n📊 Розподіл класів:")
    for i, name in enumerate(target_names):
        count = np.sum(y == i)
        print(f"  - Клас {i} ({name}): {count} зразків ({count/len(y)*100:.1f}%)")
    
    # Створення DataFrame для зручності
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = df['target'].map({i: name for i, name in enumerate(target_names)})
    
    print(f"\n📝 Опис ознак:")
    print(df[feature_names].describe().T[['mean', 'std', 'min', 'max']].round(2))
    
    return df, X, y, feature_names, target_names


def engineer_features(df, feature_names):
    """Створення нових ознак для покращення предиктивної сили"""
    print_section("ІНЖЕНЕРІЯ ОЗНАК")
    
    X_original = df[feature_names].copy()
    
    # 1. Поліноміальні ознаки (квадрати важливих змінних)
    important_features = ['alcohol', 'flavanoids', 'color_intensity', 'proline', 'od280/od315_of_diluted_wines']
    for feat in important_features:
        if feat in feature_names:
            X_original[f'{feat}_squared'] = X_original[feat] ** 2
    
    # 2. Взаємодії між важливими ознаками
    X_original['alcohol_x_flavanoids'] = X_original['alcohol'] * X_original['flavanoids']
    X_original['color_x_hue'] = X_original['color_intensity'] * X_original['hue']
    X_original['phenols_x_flavanoids'] = X_original['total_phenols'] * X_original['flavanoids']
    
    # 3. Відношення
    X_original['alcohol_proline_ratio'] = X_original['alcohol'] / (X_original['proline'] + 1)
    X_original['phenols_ratio'] = X_original['total_phenols'] / (X_original['nonflavanoid_phenols'] + 0.01)
    
    # 4. Логарифмічні перетворення для асиметричних розподілів
    X_original['log_proline'] = np.log1p(X_original['proline'])
    X_original['log_od280'] = np.log1p(X_original['od280/od315_of_diluted_wines'])
    
    # 5. Стандартизовані комбінації (z-scores)
    for col in ['alcohol', 'malic_acid', 'ash']:
        if col in X_original.columns:
            X_original[f'{col}_zscore'] = stats.zscore(X_original[col])
    
    new_features = [col for col in X_original.columns if col not in feature_names]
    
    print(f"✓ Створено {len(new_features)} нових ознак")
    print(f"  - Поліноміальні ознаки: {len([f for f in new_features if 'squared' in f])}")
    print(f"  - Взаємодії: {len([f for f in new_features if '_x_' in f])}")
    print(f"  - Відношення: {len([f for f in new_features if 'ratio' in f])}")
    print(f"  - Логарифмічні: {len([f for f in new_features if 'log_' in f])}")
    print(f"  - Z-scores: {len([f for f in new_features if 'zscore' in f])}")
    print(f"\n✓ Загальна кількість ознак: {len(X_original.columns)}")
    
    return X_original.values, list(X_original.columns)


def visualize_distributions(df, feature_names, target_names):
    """Візуалізація розподілу ознак по класах"""
    print_section("ВІЗУАЛІЗАЦІЯ РОЗПОДІЛУ ДАНИХ")
    
    # Вибираємо топ-6 найважливіших ознак (за варіацією між класами)
    top_features = ['alcohol', 'flavanoids', 'color_intensity', 'proline', 
                   'od280/od315_of_diluted_wines', 'hue']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, feat in enumerate(top_features):
        ax = axes[i]
        
        for j, target_name in enumerate(target_names):
            data = df[df['target'] == j][feat]
            ax.hist(data, bins=15, alpha=0.5, label=target_name, color=colors[j], density=True)
            
            # Додаємо KDE
            kde_data = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data)
            ax.plot(kde_data, kde(kde_data), color=colors[j], linewidth=2)
        
        ax.set_title(f'{feat}', fontsize=11, weight='bold')
        ax.set_xlabel('Значення', fontsize=9)
        ax.set_ylabel('Щільність', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Розподіл ключових ознак за класами вина', 
                 fontsize=14, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('wine_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: wine_distributions.png")
    plt.show()
    
    # Кореляційна матриця
    print("\n📊 Побудова кореляційної матриці...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    corr_matrix = df[top_features + ['target']].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Кореляційна матриця ознак', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('wine_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: wine_correlation.png")
    plt.show()


def visualize_2d_projection(X_scaled, y, target_names):
    """Візуалізація 2D проєкції даних через LDA"""
    print("\n📊 Побудова 2D проєкції через LDA...")
    
    lda_viz = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda_viz.fit_transform(X_scaled, y)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', '^']
    
    for i, (color, marker, target_name) in enumerate(zip(colors, markers, target_names)):
        mask = y == i
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                  c=color, marker=marker, s=100, alpha=0.6,
                  edgecolors='black', linewidth=0.5, label=target_name)
    
    ax.set_xlabel('LD1 (перша дискримінантна компонента)', fontsize=11, weight='bold')
    ax.set_ylabel('LD2 (друга дискримінантна компонента)', fontsize=11, weight='bold')
    ax.set_title('2D проєкція датасету Wine через Linear Discriminant Analysis', 
                fontsize=13, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Додаємо explained variance
    explained_var = lda_viz.explained_variance_ratio_
    textstr = f'LD1: {explained_var[0]:.1%}\nLD2: {explained_var[1]:.1%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('wine_lda_projection.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: wine_lda_projection.png")
    plt.show()


def train_and_evaluate_models(X_train, X_test, y_train, y_test, target_names):
    """Навчання та оцінка різних класифікаторів"""
    print_section("НАВЧАННЯ ТА ОЦІНКА МОДЕЛЕЙ")
    
    # Ініціалізація моделей
    models = {
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        'Gaussian Naive Bayes': GaussianNB()
    }
    
    results = {}
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        print(f"\n[{list(models.keys()).index(name) + 1}/{len(models)}] Навчання: {name}...")
        
        # Навчання
        model.fit(X_train, y_train)
        
        # Прогнози
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        predictions[name] = y_pred
        probabilities[name] = y_proba
        
        # Обчислення метрик
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
            'Precision (macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall (macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1-score (macro)': f1_score(y_test, y_pred, average='macro', zero_division=0)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name]['CV Accuracy (mean)'] = cv_scores.mean()
        results[name]['CV Accuracy (std)'] = cv_scores.std()
        
        print(f"  ✓ Accuracy: {results[name]['Accuracy']:.4f}")
        print(f"  ✓ CV Accuracy: {results[name]['CV Accuracy (mean)']:.4f} ± {results[name]['CV Accuracy (std)']:.4f}")
    
    # Таблиця порівняння
    print(f"\n📊 Порівняння моделей:")
    results_df = pd.DataFrame(results).T
    print(results_df.round(4).to_string())
    
    # Збереження результатів
    results_df.to_csv('model_comparison.csv')
    print("\n✓ Збережено: model_comparison.csv")
    
    return models, results, predictions, probabilities


def plot_confusion_matrices(predictions, y_test, target_names):
    """Побудова матриць помилок для всіх моделей"""
    print_section("МАТРИЦІ ПОМИЛОК")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    for idx, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        
        # Нормалізована матриця помилок
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        ax = axes[idx]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names,
                   cbar_kws={'format': '%.0f%%'}, ax=ax)
        
        ax.set_title(f'{name}\n(Accuracy: {accuracy_score(y_test, y_pred):.2%})', 
                    fontsize=11, weight='bold')
        ax.set_ylabel('Справжній клас', fontsize=10)
        ax.set_xlabel('Прогнозований клас', fontsize=10)
    
    plt.suptitle('Матриці помилок (нормалізовані)', fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: confusion_matrices.png")
    plt.show()


def plot_roc_curves(probabilities, y_test, target_names):
    """Побудова ROC кривих (One-vs-Rest) для всіх моделей"""
    print_section("ROC КРИВІ ТА AUC")
    
    # Бінаризація міток для OvR підходу
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = cycle(['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    all_auc_scores = {}
    
    for idx, (model_name, y_proba) in enumerate(probabilities.items()):
        ax = axes[idx]
        
        # Обчислення ROC для кожного класу
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        all_auc_scores[model_name] = roc_auc
        
        # Малювання кривих для кожного класу
        color_iter = cycle(['#FF6B6B', '#4ECDC4', '#45B7D1'])
        for i, color, target_name in zip(range(n_classes), color_iter, target_names):
            ax.plot(fpr[i], tpr[i], color=color, lw=2, alpha=0.8,
                   label=f'{target_name} (AUC = {roc_auc[i]:.3f})')
        
        # Micro та Macro average
        ax.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=2, linestyle='--',
               label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')
        ax.plot(fpr["macro"], tpr["macro"], color='navy', lw=2, linestyle='--',
               label=f'Macro-avg (AUC = {roc_auc["macro"]:.3f})')
        
        # Діагональна лінія (випадковий класифікатор)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3, label='Випадковий (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10, weight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=10, weight='bold')
        ax.set_title(f'{model_name}', fontsize=11, weight='bold')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('ROC криві (One-vs-Rest мультикласова класифікація)', 
                fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Збережено: roc_curves.png")
    plt.show()
    
    # Таблиця AUC scores
    print("\n📊 AUC Scores:")
    auc_df = pd.DataFrame({
        model_name: {
            f'{target_names[i]}': scores[i] for i in range(n_classes)
        } | {
            'Micro-average': scores['micro'],
            'Macro-average': scores['macro']
        }
        for model_name, scores in all_auc_scores.items()
    }).T
    print(auc_df.round(4).to_string())
    
    auc_df.to_csv('auc_scores.csv')
    print("\n✓ Збережено: auc_scores.csv")


def analyze_feature_importance(models, feature_names_all, target_names):
    """Аналіз важливості ознак"""
    print_section("АНАЛІЗ ПРЕДИКТИВНОЇ СИЛИ ОЗНАК")
    
    # Для LDA можна отримати коефіцієнти
    lda_model = models['Linear Discriminant Analysis']
    
    if hasattr(lda_model, 'coef_'):
        # Коефіцієнти для кожного класу
        coefs = lda_model.coef_
        
        # Середня абсолютна важливість по всіх класах
        mean_importance = np.abs(coefs).mean(axis=0)
        
        # Топ-15 найважливіших ознак
        top_indices = np.argsort(mean_importance)[-15:][::-1]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        y_pos = np.arange(len(top_indices))
        ax.barh(y_pos, mean_importance[top_indices], color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names_all[i] for i in top_indices], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Середня абсолютна важливість', fontsize=11, weight='bold')
        ax.set_title('Топ-15 найважливіших ознак (Linear Discriminant Analysis)', 
                    fontsize=13, weight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Збережено: feature_importance.png")
        plt.show()
        
        # Детальна таблиця коефіцієнтів
        print("\n📊 Коефіцієнти LDA для кожного класу:")
        coef_df = pd.DataFrame(
            coefs.T,
            columns=[f'Клас {i} ({name})' for i, name in enumerate(target_names)],
            index=feature_names_all
        )
        
        # Виводимо топ-10
        coef_df['Mean_Abs'] = mean_importance
        coef_df_sorted = coef_df.sort_values('Mean_Abs', ascending=False)
        print(coef_df_sorted.head(10).round(4).to_string())
        
        coef_df_sorted.to_csv('feature_coefficients.csv')
        print("\n✓ Збережено: feature_coefficients.csv")


def detailed_classification_reports(predictions, y_test, target_names):
    """Детальні звіти класифікації для всіх моделей"""
    print_section("ДЕТАЛЬНІ ЗВІТИ КЛАСИФІКАЦІЇ")
    
    for name, y_pred in predictions.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print('='*70)
        print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


def main():
    """Головна функція"""
    print("\n" + "="*70)
    print("  БЕНЧМАРКІНГ ЛІНІЙНОЇ ДИСКРИМІНАЦІЇ ТА ГАУСІВСЬКОЇ МОДЕЛІ")
    print("  Датасет: Wine (sklearn)")
    print("="*70)
    
    # 1. Завантаження даних
    df, X, y, feature_names, target_names = load_and_explore_data()
    
    # 2. Візуалізація розподілів
    visualize_distributions(df, feature_names, target_names)
    
    # 3. Інженерія ознак
    X_engineered, feature_names_all = engineer_features(df, feature_names)
    
    # 4. Розділення на train/test (стратифіковане)
    print_section("РОЗДІЛЕННЯ ДАНИХ")
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"✓ Train set: {len(X_train)} зразків ({len(X_train)/len(X_engineered)*100:.1f}%)")
    print(f"✓ Test set:  {len(X_test)} зразків ({len(X_test)/len(X_engineered)*100:.1f}%)")
    
    # Розподіл класів у train та test
    print("\n📊 Розподіл класів у train set:")
    for i, name in enumerate(target_names):
        count = np.sum(y_train == i)
        print(f"  - {name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    print("\n📊 Розподіл класів у test set:")
    for i, name in enumerate(target_names):
        count = np.sum(y_test == i)
        print(f"  - {name}: {count} ({count/len(y_test)*100:.1f}%)")
    
    # 5. Стандартизація
    print_section("СТАНДАРТИЗАЦІЯ ОЗНАК")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Ознаки стандартизовані (mean=0, std=1)")
    print(f"  - Train mean: {X_train_scaled.mean():.6f}")
    print(f"  - Train std:  {X_train_scaled.std():.6f}")
    
    # 6. Візуалізація 2D проєкції
    visualize_2d_projection(X_train_scaled, y_train, target_names)
    
    # 7. Навчання та оцінка моделей
    models, results, predictions, probabilities = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, target_names
    )
    
    # 8. Матриці помилок
    plot_confusion_matrices(predictions, y_test, target_names)
    
    # 9. ROC криві
    plot_roc_curves(probabilities, y_test, target_names)
    
    # 10. Аналіз важливості ознак
    analyze_feature_importance(models, feature_names_all, target_names)
    
    # 11. Детальні звіти
    detailed_classification_reports(predictions, y_test, target_names)
    
    # Підсумок
    print_section("ПІДСУМОК")
    print("\n✅ Аналіз завершено!")
    print("\n📁 Створені файли:")
    print("  - wine_distributions.png - розподіл ознак")
    print("  - wine_correlation.png - кореляційна матриця")
    print("  - wine_lda_projection.png - 2D проєкція через LDA")
    print("  - confusion_matrices.png - матриці помилок")
    print("  - roc_curves.png - ROC криві")
    print("  - feature_importance.png - важливість ознак")
    print("  - model_comparison.csv - порівняння моделей")
    print("  - auc_scores.csv - AUC scores")
    print("  - feature_coefficients.csv - коефіцієнти ознак")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

