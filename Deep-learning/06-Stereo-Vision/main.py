"""
Stereo Vision - Обчислення карти глибини (Disparity Map)
Фокус: Порівняння різних методів стерео-зіставлення

Реалізовано:
1. Block Matching (BM) - класичний метод блочного зіставлення (25%)
2. Semi-Global Block Matching (SGBM) - напівглобальне зіставлення (25%)
3. SIFT Feature Matching - зіставлення за ключовими точками (25%)
4. Template Matching - зіставлення шаблонів (25%)

Метрики та візуалізація:
- Disparity maps для кожного методу
- Порівняльний аналіз методів
- 3D реконструкція точок
- Час виконання та якість
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import urllib.request
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Налаштування
np.random.seed(42)
sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams['font.size'] = 11

# Директорії
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def print_section(title: str) -> None:
    """Виводить заголовок секції."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# 1. Завантаження стерео-пар зображень
# ---------------------------------------------------------------------------

def download_stereo_pair() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Завантажує стерео-пару зображень з Middlebury Stereo Dataset.
    Якщо не вдається завантажити - генерує синтетичну пару.
    """
    print_section("ЗАВАНТАЖЕННЯ СТЕРЕО-ПАРИ ЗОБРАЖЕНЬ")
    
    # URL для Middlebury stereo dataset (Tsukuba)
    left_url = "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/tsukuba/im3.ppm"
    right_url = "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/tsukuba/im4.ppm"
    
    left_path = DATA_DIR / "left.ppm"
    right_path = DATA_DIR / "right.ppm"
    
    try:
        # Спробуємо завантажити
        if not left_path.exists():
            print("  Завантаження лівого зображення...")
            urllib.request.urlretrieve(left_url, left_path)
            print(f"  ✓ Збережено: {left_path}")
        else:
            print(f"  ✓ Використано кеш: {left_path}")
        
        if not right_path.exists():
            print("  Завантаження правого зображення...")
            urllib.request.urlretrieve(right_url, right_path)
            print(f"  ✓ Збережено: {right_path}")
        else:
            print(f"  ✓ Використано кеш: {right_path}")
        
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        
        if left_img is not None and right_img is not None:
            print(f"\n✓ Стерео-пара завантажена успішно!")
            print(f"  Розмір: {left_img.shape[1]}x{left_img.shape[0]}")
            return left_img, right_img
            
    except Exception as e:
        print(f"  ⚠ Не вдалося завантажити з Middlebury: {e}")
    
    # Якщо не вдалося - генеруємо синтетичну стерео-пару
    print("\n  Генерація синтетичної стерео-пари...")
    left_img, right_img = generate_synthetic_stereo_pair()
    
    return left_img, right_img


def generate_synthetic_stereo_pair(
    width: int = 640,
    height: int = 480,
    max_disparity: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерує синтетичну стерео-пару з відомою диспаратністю.
    Створює сцену з кількома об'єктами на різних глибинах.
    """
    # Фонове зображення з текстурою
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Додаємо текстуру фону (шахова дошка)
    for y in range(0, height, 32):
        for x in range(0, width, 32):
            color = np.random.randint(100, 200, 3).tolist()
            cv2.rectangle(left_img, (x, y), (x+32, y+32), color, -1)
    
    # Додаємо випадковий шум для текстури
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    left_img = cv2.add(left_img, noise)
    
    # Додаємо об'єкти на різних глибинах
    objects = [
        # (центр_x, центр_y, радіус, глибина/disparity, колір)
        (320, 240, 80, 48, (255, 100, 100)),   # Близько (великий disparity)
        (480, 180, 50, 32, (100, 255, 100)),   # Середньо
        (160, 320, 60, 20, (100, 100, 255)),   # Далеко
        (400, 350, 40, 40, (255, 255, 100)),   # Близько
        (200, 150, 45, 16, (255, 100, 255)),   # Далеко
    ]
    
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    for cx, cy, r, disp, color in objects:
        # Малюємо круглий об'єкт з текстурою
        cv2.circle(left_img, (cx, cy), r, color, -1)
        
        # Додаємо текстуру на об'єкті
        for i in range(5):
            tx = cx + np.random.randint(-r//2, r//2)
            ty = cy + np.random.randint(-r//2, r//2)
            tr = np.random.randint(5, 15)
            tc = tuple(max(0, min(255, c + np.random.randint(-50, 50))) for c in color)
            cv2.circle(left_img, (tx, ty), tr, tc, -1)
        
        # Заповнюємо карту диспаратності
        y_coords, x_coords = np.ogrid[:height, :width]
        mask = (x_coords - cx)**2 + (y_coords - cy)**2 <= r**2
        disparity_map[mask] = disp
    
    # Фонова диспаратність (далека площина)
    disparity_map[disparity_map == 0] = 8
    
    # Створюємо праве зображення зміщенням пікселів
    right_img = np.zeros_like(left_img)
    
    for y in range(height):
        for x in range(width):
            disp = int(disparity_map[y, x])
            new_x = x - disp
            if 0 <= new_x < width:
                right_img[y, new_x] = left_img[y, x]
    
    # Заповнюємо прогалини
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    mask = (right_gray == 0).astype(np.uint8) * 255
    right_img = cv2.inpaint(right_img, mask, 3, cv2.INPAINT_TELEA)
    
    # Зберігаємо ground truth disparity
    np.save(DATA_DIR / "ground_truth_disparity.npy", disparity_map)
    
    # Зберігаємо зображення
    cv2.imwrite(str(DATA_DIR / "left_synthetic.png"), left_img)
    cv2.imwrite(str(DATA_DIR / "right_synthetic.png"), right_img)
    
    print(f"  ✓ Синтетична стерео-пара згенерована!")
    print(f"  ✓ Розмір: {width}x{height}")
    print(f"  ✓ Максимальний disparity: {max_disparity}")
    
    return left_img, right_img


def visualize_stereo_pair(left_img: np.ndarray, right_img: np.ndarray) -> None:
    """Візуалізує стерео-пару."""
    print_section("ВІЗУАЛІЗАЦІЯ СТЕРЕО-ПАРИ")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ліве зображення
    axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ліве зображення (Left)", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Праве зображення
    axes[1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Праве зображення (Right)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle("Стерео-пара зображень", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "stereo_pair.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Block Matching (BM) - 25%
# ---------------------------------------------------------------------------

def compute_disparity_bm(
    left_img: np.ndarray,
    right_img: np.ndarray,
    num_disparities: int = 64,
    block_size: int = 15
) -> Tuple[np.ndarray, float]:
    """
    Обчислює карту диспаратності методом Block Matching.
    
    Block Matching - це базовий алгоритм для обчислення disparity:
    1. Для кожного пікселя в лівому зображенні шукаємо відповідність у правому
    2. Порівняння відбувається по блоках (вікнах) фіксованого розміру
    3. Критерій - Sum of Absolute Differences (SAD)
    
    Параметри:
    - num_disparities: максимальний діапазон пошуку (має бути кратним 16)
    - block_size: розмір блоку (непарне число)
    """
    print_section("BLOCK MATCHING (BM) APPROACH")
    
    print(f"\n  Параметри:")
    print(f"    - num_disparities: {num_disparities}")
    print(f"    - block_size: {block_size}")
    
    # Конвертуємо в grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Створюємо StereoBM matcher
    stereo_bm = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )
    
    # Додаткові налаштування для кращої якості
    stereo_bm.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
    stereo_bm.setPreFilterSize(9)
    stereo_bm.setPreFilterCap(31)
    stereo_bm.setTextureThreshold(10)
    stereo_bm.setUniquenessRatio(15)
    stereo_bm.setSpeckleWindowSize(100)
    stereo_bm.setSpeckleRange(32)
    
    # Обчислюємо disparity map
    print("\n  Обчислення disparity map...")
    start_time = time.time()
    
    disparity = stereo_bm.compute(left_gray, right_gray)
    
    elapsed_time = time.time() - start_time
    
    # Нормалізуємо disparity
    disparity_normalized = cv2.normalize(
        disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    # Статистика
    valid_mask = disparity > 0
    if np.any(valid_mask):
        disparity_float = disparity.astype(np.float32) / 16.0
        valid_disparity = disparity_float[valid_mask]
        print(f"\n  Результати:")
        print(f"    - Час обчислення: {elapsed_time*1000:.1f} ms")
        print(f"    - Min disparity: {valid_disparity.min():.2f}")
        print(f"    - Max disparity: {valid_disparity.max():.2f}")
        print(f"    - Mean disparity: {valid_disparity.mean():.2f}")
        print(f"    - Valid pixels: {np.sum(valid_mask) / disparity.size * 100:.1f}%")
    
    return disparity_normalized, elapsed_time


def test_bm_parameters(
    left_img: np.ndarray,
    right_img: np.ndarray
) -> pd.DataFrame:
    """Тестує різні параметри Block Matching."""
    print_section("ДОСЛІДЖЕННЯ ПАРАМЕТРІВ BM")
    
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    results = []
    
    # Різні комбінації параметрів
    disparities_list = [16, 32, 64, 128]
    block_sizes = [5, 11, 15, 21]
    
    for num_disp in disparities_list:
        for block_size in block_sizes:
            try:
                stereo_bm = cv2.StereoBM_create(
                    numDisparities=num_disp,
                    blockSize=block_size
                )
                
                start_time = time.time()
                disparity = stereo_bm.compute(left_gray, right_gray)
                elapsed = time.time() - start_time
                
                valid_mask = disparity > 0
                coverage = np.sum(valid_mask) / disparity.size * 100
                
                results.append({
                    'num_disparities': num_disp,
                    'block_size': block_size,
                    'time_ms': elapsed * 1000,
                    'coverage_%': coverage
                })
                
                print(f"  disp={num_disp:3d}, block={block_size:2d}: "
                      f"time={elapsed*1000:.1f}ms, coverage={coverage:.1f}%")
                
            except Exception as e:
                print(f"  ⚠ Помилка з disp={num_disp}, block={block_size}: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "bm_parameters.csv", index=False)
    print(f"\n✓ Результати збережено: results/bm_parameters.csv")
    
    return df


# ---------------------------------------------------------------------------
# 3. Semi-Global Block Matching (SGBM) - 25%
# ---------------------------------------------------------------------------

def compute_disparity_sgbm(
    left_img: np.ndarray,
    right_img: np.ndarray,
    num_disparities: int = 64,
    block_size: int = 5,
    mode: int = cv2.STEREO_SGBM_MODE_SGBM
) -> Tuple[np.ndarray, float]:
    """
    Обчислює карту диспаратності методом Semi-Global Block Matching.
    
    SGBM - покращений алгоритм, що враховує не лише локальні блоки,
    а й глобальну консистентність вздовж кількох напрямків:
    1. Використовує динамічне програмування вздовж 8 або 16 напрямків
    2. Мінімізує глобальну енергетичну функцію
    3. Краща якість на слабо-текстурованих ділянках
    
    Параметри:
    - num_disparities: діапазон пошуку (кратний 16)
    - block_size: розмір блоку (1-11, непарне)
    - mode: SGBM_MODE_SGBM (8 напрямків) або SGBM_MODE_HH (повний)
    """
    print_section("SEMI-GLOBAL BLOCK MATCHING (SGBM) APPROACH")
    
    mode_name = "8 напрямків" if mode == cv2.STEREO_SGBM_MODE_SGBM else "повний (16 напрямків)"
    print(f"\n  Параметри:")
    print(f"    - num_disparities: {num_disparities}")
    print(f"    - block_size: {block_size}")
    print(f"    - mode: {mode_name}")
    
    # Конвертуємо в grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Параметри SGBM
    P1 = 8 * 3 * block_size ** 2   # Штраф за малі зміни disparity
    P2 = 32 * 3 * block_size ** 2  # Штраф за великі зміни disparity
    
    # Створюємо StereoSGBM matcher
    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=mode
    )
    
    # Обчислюємо disparity map
    print("\n  Обчислення disparity map...")
    start_time = time.time()
    
    disparity = stereo_sgbm.compute(left_gray, right_gray)
    
    elapsed_time = time.time() - start_time
    
    # Нормалізуємо disparity
    disparity_normalized = cv2.normalize(
        disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    # Статистика
    valid_mask = disparity > 0
    if np.any(valid_mask):
        disparity_float = disparity.astype(np.float32) / 16.0
        valid_disparity = disparity_float[valid_mask]
        print(f"\n  Результати:")
        print(f"    - Час обчислення: {elapsed_time*1000:.1f} ms")
        print(f"    - Min disparity: {valid_disparity.min():.2f}")
        print(f"    - Max disparity: {valid_disparity.max():.2f}")
        print(f"    - Mean disparity: {valid_disparity.mean():.2f}")
        print(f"    - Valid pixels: {np.sum(valid_mask) / disparity.size * 100:.1f}%")
    
    return disparity_normalized, elapsed_time


def compare_bm_sgbm(
    left_img: np.ndarray,
    right_img: np.ndarray
) -> None:
    """Порівнює BM та SGBM методи."""
    print_section("ПОРІВНЯННЯ BM vs SGBM")
    
    # BM
    disparity_bm, time_bm = compute_disparity_bm(left_img, right_img)
    
    # SGBM
    disparity_sgbm, time_sgbm = compute_disparity_sgbm(left_img, right_img)
    
    # Візуалізація
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Оригінал
    axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Ліве зображення", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Праве зображення", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Disparity maps
    im1 = axes[1, 0].imshow(disparity_bm, cmap='plasma')
    axes[1, 0].set_title(f"Block Matching (BM)\nЧас: {time_bm*1000:.1f} ms", 
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[1, 1].imshow(disparity_sgbm, cmap='plasma')
    axes[1, 1].set_title(f"Semi-Global Block Matching (SGBM)\nЧас: {time_sgbm*1000:.1f} ms",
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle("Порівняння BM та SGBM методів", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "bm_vs_sgbm.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 4. SIFT Feature Matching Approach - 25%
# ---------------------------------------------------------------------------

def compute_disparity_sift(
    left_img: np.ndarray,
    right_img: np.ndarray,
    n_features: int = 5000
) -> Tuple[np.ndarray, float, Dict]:
    """
    Обчислює карту диспаратності методом SIFT feature matching.
    
    SIFT (Scale-Invariant Feature Transform) - алгоритм детекції ключових точок:
    1. Знаходить ключові точки в обох зображеннях
    2. Обчислює дескриптори для кожної точки
    3. Зіставляє точки між лівим та правим зображеннями
    4. Disparity = різниця x-координат зіставлених точок
    
    Переваги:
    - Інваріантний до масштабу та повороту
    - Добре працює з великими зміщеннями
    
    Недоліки:
    - Розріджена карта (тільки в ключових точках)
    - Може бути повільним
    """
    print_section("SIFT FEATURE MATCHING APPROACH")
    
    print(f"\n  Параметри:")
    print(f"    - n_features: {n_features}")
    
    # Конвертуємо в grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Створюємо SIFT детектор
    sift = cv2.SIFT_create(nfeatures=n_features)
    
    start_time = time.time()
    
    # Знаходимо ключові точки та дескриптори
    print("\n  Детекція ключових точок...")
    kp1, desc1 = sift.detectAndCompute(left_gray, None)
    kp2, desc2 = sift.detectAndCompute(right_gray, None)
    
    print(f"    - Ліве зображення: {len(kp1)} точок")
    print(f"    - Праве зображення: {len(kp2)} точок")
    
    # Matcher - використовуємо BFMatcher з cross-check
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    print("  Зіставлення точок...")
    matches = bf.match(desc1, desc2)
    
    # Сортуємо по якості
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"    - Знайдено відповідностей: {len(matches)}")
    
    # Фільтруємо matches за epipolar constraint (y-координати мають бути близькі)
    good_matches = []
    disparities_list = []
    match_points = []
    
    for m in matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        
        # Перевіряємо epipolar constraint (різниця y не більше 2 пікселів)
        if abs(pt1[1] - pt2[1]) < 2:
            disparity = pt1[0] - pt2[0]
            
            # Disparity має бути позитивним (ліва точка правіше за праву)
            if 0 < disparity < left_gray.shape[1] // 2:
                good_matches.append(m)
                disparities_list.append(disparity)
                match_points.append((pt1, pt2, disparity))
    
    elapsed_time = time.time() - start_time
    
    print(f"    - Після фільтрації: {len(good_matches)} відповідностей")
    
    # Створюємо розріджену карту диспаратності
    height, width = left_gray.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    
    for pt1, pt2, disp in match_points:
        x, y = int(pt1[0]), int(pt1[1])
        if 0 <= x < width and 0 <= y < height:
            disparity_map[y, x] = disp
    
    # Інтерполяція для заповнення прогалин (опційно)
    # Використовуємо Gaussian blur для "розмазування" точок
    disparity_dense = cv2.GaussianBlur(disparity_map, (15, 15), 0)
    mask = disparity_map > 0
    disparity_dense[~mask] = 0
    
    # Нормалізуємо для візуалізації
    disparity_normalized = cv2.normalize(
        disparity_dense, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    # Статистика
    if disparities_list:
        disparities_arr = np.array(disparities_list)
        print(f"\n  Результати:")
        print(f"    - Час обчислення: {elapsed_time*1000:.1f} ms")
        print(f"    - Min disparity: {disparities_arr.min():.2f}")
        print(f"    - Max disparity: {disparities_arr.max():.2f}")
        print(f"    - Mean disparity: {disparities_arr.mean():.2f}")
        print(f"    - Std disparity: {disparities_arr.std():.2f}")
    
    # Додаткова інформація для візуалізації
    info = {
        'keypoints_left': kp1,
        'keypoints_right': kp2,
        'matches': good_matches,
        'match_points': match_points,
        'disparities': disparities_list
    }
    
    return disparity_normalized, elapsed_time, info


def visualize_sift_matches(
    left_img: np.ndarray,
    right_img: np.ndarray,
    info: Dict,
    max_matches: int = 100
) -> None:
    """Візуалізує SIFT зіставлення."""
    print_section("ВІЗУАЛІЗАЦІЯ SIFT ЗІСТАВЛЕНЬ")
    
    kp1 = info['keypoints_left']
    kp2 = info['keypoints_right']
    matches = info['matches'][:max_matches]
    
    # Малюємо зіставлення
    matches_img = cv2.drawMatches(
        left_img, kp1,
        right_img, kp2,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Зіставлення
    axes[0].imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"SIFT Feature Matches (показано {len(matches)} з {len(info['matches'])})",
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Розподіл диспаратностей
    if info['disparities']:
        axes[1].hist(info['disparities'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(info['disparities']), color='red', linestyle='--', 
                        label=f"Середнє: {np.mean(info['disparities']):.1f}")
        axes[1].set_xlabel("Disparity (пікселі)", fontsize=12)
        axes[1].set_ylabel("Кількість точок", fontsize=12)
        axes[1].set_title("Розподіл диспаратності по SIFT точках", fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "sift_matches.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Template Matching Approach - 25%
# ---------------------------------------------------------------------------

def compute_disparity_template_matching(
    left_img: np.ndarray,
    right_img: np.ndarray,
    template_size: int = 15,
    search_range: int = 64,
    step: int = 4
) -> Tuple[np.ndarray, float]:
    """
    Обчислює карту диспаратності методом Template Matching.
    
    Template Matching - пряме порівняння шаблонів:
    1. Для кожного пікселя (з кроком step) в лівому зображенні
    2. Вирізаємо шаблон (template) навколо цього пікселя
    3. Шукаємо найкращу відповідність в правому зображенні
    4. Disparity = різниця x-координат
    
    Використовуємо Normalized Cross-Correlation (NCC) як метрику.
    
    Параметри:
    - template_size: розмір шаблону (template_size x template_size)
    - search_range: максимальний діапазон пошуку
    - step: крок сканування (для швидкості)
    """
    print_section("TEMPLATE MATCHING APPROACH")
    
    print(f"\n  Параметри:")
    print(f"    - template_size: {template_size}x{template_size}")
    print(f"    - search_range: {search_range}")
    print(f"    - step: {step}")
    
    # Конвертуємо в grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    height, width = left_gray.shape
    half_template = template_size // 2
    
    # Ініціалізуємо карту диспаратності
    disparity_map = np.zeros((height, width), dtype=np.float32)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    
    start_time = time.time()
    
    print("\n  Обчислення disparity map...")
    total_points = ((height - template_size) // step) * ((width - template_size - search_range) // step)
    processed = 0
    
    # Сканування з кроком step
    for y in range(half_template, height - half_template, step):
        for x in range(half_template + search_range, width - half_template, step):
            # Вирізаємо шаблон з лівого зображення
            template = left_gray[
                y - half_template : y + half_template + 1,
                x - half_template : x + half_template + 1
            ]
            
            # Область пошуку в правому зображенні
            search_region = right_gray[
                y - half_template : y + half_template + 1,
                x - search_range - half_template : x + half_template + 1
            ]
            
            # Template matching з NCC
            result = cv2.matchTemplate(
                search_region, 
                template, 
                cv2.TM_CCOEFF_NORMED
            )
            
            # Знаходимо найкращу позицію
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Disparity = зміщення від початку пошукової області
            best_x = max_loc[0]
            disparity = search_range - best_x
            
            # Зберігаємо якщо confidence достатньо висока
            if max_val > 0.5:
                disparity_map[y, x] = disparity
                confidence_map[y, x] = max_val
            
            processed += 1
            
            # Прогрес
            if processed % 5000 == 0:
                progress = processed / total_points * 100
                print(f"    Прогрес: {progress:.1f}%", end='\r')
    
    elapsed_time = time.time() - start_time
    print(f"    Прогрес: 100.0%")
    
    # Інтерполяція для заповнення прогалин
    # Використовуємо resize для upsampling
    disparity_dense = cv2.resize(
        disparity_map, 
        (width, height), 
        interpolation=cv2.INTER_LINEAR
    )
    
    # Медіанний фільтр для згладжування
    disparity_dense = cv2.medianBlur(disparity_dense.astype(np.uint8), 5)
    
    # Нормалізуємо
    disparity_normalized = cv2.normalize(
        disparity_dense, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    
    # Статистика
    valid_mask = disparity_map > 0
    if np.any(valid_mask):
        valid_disparity = disparity_map[valid_mask]
        print(f"\n  Результати:")
        print(f"    - Час обчислення: {elapsed_time:.1f} s")
        print(f"    - Min disparity: {valid_disparity.min():.2f}")
        print(f"    - Max disparity: {valid_disparity.max():.2f}")
        print(f"    - Mean disparity: {valid_disparity.mean():.2f}")
        print(f"    - Mean confidence: {confidence_map[valid_mask].mean():.3f}")
        print(f"    - Valid pixels: {np.sum(valid_mask) / (height * width / step / step) * 100:.1f}%")
    
    return disparity_normalized, elapsed_time


# ---------------------------------------------------------------------------
# 6. Порівняння всіх методів
# ---------------------------------------------------------------------------

def compare_all_methods(
    left_img: np.ndarray,
    right_img: np.ndarray
) -> pd.DataFrame:
    """Порівнює всі 4 методи обчислення disparity."""
    print_section("ПОРІВНЯННЯ ВСІХ МЕТОДІВ")
    
    results = {}
    disparity_maps = {}
    
    # 1. Block Matching
    print("\n" + "-" * 40)
    print("  1/4: Block Matching")
    disparity_bm, time_bm = compute_disparity_bm(left_img, right_img)
    results['BM'] = {'time_s': time_bm, 'method': 'Block Matching'}
    disparity_maps['BM'] = disparity_bm
    
    # 2. SGBM
    print("\n" + "-" * 40)
    print("  2/4: Semi-Global Block Matching")
    disparity_sgbm, time_sgbm = compute_disparity_sgbm(left_img, right_img)
    results['SGBM'] = {'time_s': time_sgbm, 'method': 'Semi-Global Block Matching'}
    disparity_maps['SGBM'] = disparity_sgbm
    
    # 3. SIFT
    print("\n" + "-" * 40)
    print("  3/4: SIFT Feature Matching")
    disparity_sift, time_sift, sift_info = compute_disparity_sift(left_img, right_img)
    results['SIFT'] = {'time_s': time_sift, 'method': 'SIFT Feature Matching'}
    disparity_maps['SIFT'] = disparity_sift
    
    # 4. Template Matching (з більшим кроком для швидкості)
    print("\n" + "-" * 40)
    print("  4/4: Template Matching")
    disparity_tm, time_tm = compute_disparity_template_matching(
        left_img, right_img, step=8
    )
    results['Template'] = {'time_s': time_tm, 'method': 'Template Matching'}
    disparity_maps['Template'] = disparity_tm
    
    # Візуалізація порівняння
    visualize_all_methods(left_img, disparity_maps, results)
    
    # Додаткові візуалізації
    visualize_sift_matches(left_img, right_img, sift_info)
    
    # Створюємо DataFrame
    df = pd.DataFrame(results).T
    df['time_ms'] = df['time_s'] * 1000
    df = df[['method', 'time_ms']]
    
    print_section("ПІДСУМКОВА ТАБЛИЦЯ")
    print("\n" + df.to_string())
    
    df.to_csv(OUTPUT_DIR / "methods_comparison.csv")
    print(f"\n✓ Збережено: results/methods_comparison.csv")
    
    return df


def visualize_all_methods(
    left_img: np.ndarray,
    disparity_maps: Dict[str, np.ndarray],
    results: Dict
) -> None:
    """Візуалізує результати всіх методів."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Оригінальне зображення
    axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Оригінал (ліве зображення)", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # BM
    im1 = axes[0, 1].imshow(disparity_maps['BM'], cmap='plasma')
    time_bm = results['BM']['time_s'] * 1000
    axes[0, 1].set_title(f"Block Matching (BM)\n{time_bm:.1f} ms", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # SGBM
    im2 = axes[0, 2].imshow(disparity_maps['SGBM'], cmap='plasma')
    time_sgbm = results['SGBM']['time_s'] * 1000
    axes[0, 2].set_title(f"SGBM\n{time_sgbm:.1f} ms", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # SIFT
    im3 = axes[1, 0].imshow(disparity_maps['SIFT'], cmap='plasma')
    time_sift = results['SIFT']['time_s'] * 1000
    axes[1, 0].set_title(f"SIFT Feature Matching\n{time_sift:.1f} ms", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Template Matching
    im4 = axes[1, 1].imshow(disparity_maps['Template'], cmap='plasma')
    time_tm = results['Template']['time_s'] * 1000
    axes[1, 1].set_title(f"Template Matching\n{time_tm:.1f} ms", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Графік порівняння часу
    methods = list(results.keys())
    times = [results[m]['time_s'] * 1000 for m in methods]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    bars = axes[1, 2].bar(methods, times, color=colors, edgecolor='black')
    axes[1, 2].set_ylabel("Час (мс)", fontsize=11)
    axes[1, 2].set_title("Порівняння швидкості", fontsize=12, fontweight='bold')
    axes[1, 2].grid(alpha=0.3, axis='y')
    
    # Додаємо значення над стовпчиками
    for bar, t in zip(bars, times):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                        f'{t:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle("Порівняння методів обчислення Disparity Map", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "all_methods_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 7. 3D Реконструкція
# ---------------------------------------------------------------------------

def reconstruct_3d(
    left_img: np.ndarray,
    disparity: np.ndarray,
    focal_length: float = 500.0,
    baseline: float = 60.0  # мм
) -> None:
    """
    Виконує 3D реконструкцію на основі карти диспаратності.
    
    Формула: Z = (f * B) / d
    де:
    - Z: глибина
    - f: фокусна відстань (пікселі)
    - B: базова лінія (відстань між камерами)
    - d: диспаратність
    """
    print_section("3D РЕКОНСТРУКЦІЯ")
    
    height, width = disparity.shape
    
    # Створюємо матрицю Q для reprojection
    Q = np.float32([
        [1, 0, 0, -width / 2],
        [0, -1, 0, height / 2],
        [0, 0, 0, -focal_length],
        [0, 0, 1 / baseline, 0]
    ])
    
    # Reprojectимо в 3D
    disparity_float = disparity.astype(np.float32) / 16.0
    points_3d = cv2.reprojectImageTo3D(disparity_float, Q)
    
    # Маска валідних точок
    mask = disparity > 0
    
    # Витягуємо координати та кольори
    valid_points = points_3d[mask]
    colors = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)[mask] / 255.0
    
    # Фільтруємо точки за глибиною
    z_values = valid_points[:, 2]
    valid_z = (z_values > -1000) & (z_values < 1000)
    
    valid_points = valid_points[valid_z]
    colors = colors[valid_z]
    
    # Субсемплінг для візуалізації
    if len(valid_points) > 10000:
        indices = np.random.choice(len(valid_points), 10000, replace=False)
        valid_points = valid_points[indices]
        colors = colors[indices]
    
    print(f"\n  Кількість 3D точок: {len(valid_points)}")
    
    # 3D візуалізація
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(
        valid_points[:, 0],
        valid_points[:, 1],
        valid_points[:, 2],
        c=colors,
        s=1,
        alpha=0.5
    )
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z (глибина)', fontsize=11)
    ax.set_title('3D Реконструкція сцени', fontsize=14, fontweight='bold')
    
    # Налаштовуємо вигляд
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "3d_reconstruction.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 8. Головне меню
# ---------------------------------------------------------------------------

def main():
    """Головна функція."""
    print("\n" + "=" * 80)
    print("  STEREO VISION - ОБЧИСЛЕННЯ КАРТИ ГЛИБИНИ")
    print("  Block Matching | SGBM | SIFT | Template Matching")
    print("=" * 80)
    
    print("\n📋 Меню:")
    print("  1. Завантажити стерео-пару та візуалізувати")
    print("  2. Block Matching (BM)")
    print("  3. Semi-Global Block Matching (SGBM)")
    print("  4. SIFT Feature Matching")
    print("  5. Template Matching")
    print("  6. Порівняння всіх методів")
    print("  7. 3D Реконструкція")
    print("  8. Виконати все (повний аналіз)")
    
    choice = input("\n  Виберіть опцію (1-8): ").strip()
    
    # Завантажуємо стерео-пару
    left_img, right_img = download_stereo_pair()
    
    if left_img is None or right_img is None:
        print("\n⚠ Не вдалося завантажити стерео-пару!")
        return
    
    if choice == "1" or choice == "8":
        visualize_stereo_pair(left_img, right_img)
    
    if choice == "2" or choice == "8":
        disparity_bm, _ = compute_disparity_bm(left_img, right_img)
        test_bm_parameters(left_img, right_img)
    
    if choice == "3" or choice == "8":
        disparity_sgbm, _ = compute_disparity_sgbm(left_img, right_img)
        compare_bm_sgbm(left_img, right_img)
    
    if choice == "4" or choice == "8":
        disparity_sift, _, sift_info = compute_disparity_sift(left_img, right_img)
        visualize_sift_matches(left_img, right_img, sift_info)
    
    if choice == "5" or choice == "8":
        disparity_tm, _ = compute_disparity_template_matching(
            left_img, right_img, step=8
        )
    
    if choice == "6" or choice == "8":
        comparison_df = compare_all_methods(left_img, right_img)
    
    if choice == "7" or choice == "8":
        # Використовуємо SGBM для 3D реконструкції (найкраща якість)
        disparity_sgbm, _ = compute_disparity_sgbm(left_img, right_img)
        reconstruct_3d(left_img, disparity_sgbm)
    
    # Підсумок
    print_section("ПІДСУМОК")
    print("\n✅ Лабораторна «Stereo Vision» виконана!")
    print("\n📁 Результати збережені в каталозі 'results':")
    print("  - stereo_pair.png — візуалізація стерео-пари")
    print("  - bm_vs_sgbm.png — порівняння BM та SGBM")
    print("  - bm_parameters.csv — дослідження параметрів BM")
    print("  - sift_matches.png — візуалізація SIFT зіставлень")
    print("  - all_methods_comparison.png — порівняння всіх методів")
    print("  - methods_comparison.csv — таблиця порівняння")
    print("  - 3d_reconstruction.png — 3D реконструкція сцени")
    
    print("\n💡 Методи:")
    print("  - BM: швидкий, базовий метод")
    print("  - SGBM: найкраща якість, оптимальний вибір")
    print("  - SIFT: добре для великих зміщень, розріджена карта")
    print("  - Template Matching: простий, але повільний")


if __name__ == "__main__":
    main()

