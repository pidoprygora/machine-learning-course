"""
Детектування об'єктів (Object Detection)
Фокус: Навчання детекторів, оцінка точності, real-time детекція з камери

Мета: Навчити детектор об'єктів, оцінити його точність та запустити
у live режимі для детекції об'єктів у відео потоці з веб-камери.

Підтримувані об'єкти:
- Обличчя (faces)
- Фігури людей (persons)
- Автомобілі (cars)
- Інші COCO об'єкти

Моделі:
- SSD MobileNet V2 (швидкий, легкий)
- Faster R-CNN ResNet50 (точний, повільніший)
- EfficientDet (баланс швидкості/точності)

Метрики:
- mAP (mean Average Precision)
- IoU (Intersection over Union)
- Precision/Recall
- FPS (frames per second)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import cv2
import time
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import tarfile
import json

# Налаштування
tf.random.set_seed(42)
np.random.seed(42)

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'

# Директорії
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def print_section(title):
    """Виводить заголовок секції"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# COCO класи (80 класів)
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

# Спрощений набір класів (щоб задача була легшою для аналізу)
# ВАЖЛИВО: тут обов'язково присутній клас "person"
SIMPLE_CLASSES = {
    1: 'person',   # людина
    3: 'car',      # автомобіль
    8: 'truck',    # вантажівка
    17: 'cat',     # кіт
    18: 'dog',     # собака
}

# Якщо True — детектор буде показувати лише класи з SIMPLE_CLASSES
# Якщо False — використовуються всі COCO класи
USE_SIMPLE_CLASSES = True


def download_sample_images():
    """Завантажує приклади зображень для тестування"""
    print_section("ЗАВАНТАЖЕННЯ ТЕСТОВИХ ЗОБРАЖЕНЬ")
    
    # Список URL зображень для тестування
    sample_urls = [
        "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image1.jpg",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image2.jpg",
    ]
    
    images = []
    
    for idx, url in enumerate(sample_urls):
        try:
            print(f"\n  Завантаження зображення {idx+1}...")
            image_path = DATA_DIR / f"test_image_{idx+1}.jpg"
            
            if not image_path.exists():
                urllib.request.urlretrieve(url, image_path)
                print(f"  ✓ Збережено: {image_path}")
            else:
                print(f"  ✓ Використано кеш: {image_path}")
            
            # Читаємо зображення
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
            
        except Exception as e:
            print(f"  ⚠ Помилка при завантаженні: {e}")
    
    print(f"\n✓ Завантажено {len(images)} зображень")
    return images


def load_detection_model(model_name='ssd_mobilenet_v2'):
    """Завантажує pretrained object detection модель"""
    print_section(f"ЗАВАНТАЖЕННЯ МОДЕЛІ: {model_name.upper()}")
    
    # TensorFlow Hub URLs для different моделей
    model_urls = {
        'ssd_mobilenet_v2': 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2',
        'faster_rcnn_resnet50': 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1',
        'efficientdet_d0': 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
        'centernet_resnet50': 'https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1'
    }
    
    if model_name not in model_urls:
        print(f"  ⚠ Модель {model_name} не знайдено. Використовую ssd_mobilenet_v2")
        model_name = 'ssd_mobilenet_v2'
    
    model_url = model_urls[model_name]
    
    print(f"\n  Завантаження моделі з TensorFlow Hub...")
    print(f"  URL: {model_url}")
    
    try:
        start_time = time.time()
        detector = hub.load(model_url)
        load_time = time.time() - start_time
        
        print(f"  ✓ Модель завантажено за {load_time:.2f}s")
        
        return detector, model_name
        
    except Exception as e:
        print(f"  ⚠ Помилка при завантаженні моделі: {e}")
        return None, None


def detect_objects(detector, image, confidence_threshold=0.5):
    """Виконує детекцію об'єктів на зображенні"""
    
    # Конвертуємо зображення в тензор
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Детекція
    start_time = time.time()
    detections = detector(input_tensor)
    inference_time = time.time() - start_time
    
    # Витягуємо результати
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # Фільтруємо по confidence
    scores = detections['detection_scores']
    classes = detections['detection_classes'].astype(np.int32)
    indices = scores >= confidence_threshold

    # ДОДАТКОВО: фільтруємо лише потрібні класи, якщо спрощений режим увімкнено
    if USE_SIMPLE_CLASSES:
        allowed_ids = np.array(list(SIMPLE_CLASSES.keys()), dtype=np.int32)
        class_mask = np.isin(classes, allowed_ids)
        indices = indices & class_mask
    
    results = {
        'boxes': detections['detection_boxes'][indices],
        'classes': classes[indices],
        'scores': scores[indices],
        'num_detections': np.sum(indices),
        'inference_time': inference_time
    }
    
    return results


def draw_detections(image, results, min_score=0.5):
    """Малює bounding boxes на зображенні"""
    
    img_with_boxes = image.copy()
    height, width = img_with_boxes.shape[:2]
    
    # Генеруємо кольори для класів
    np.random.seed(42)
    colors = {}
    for class_id in COCO_CLASSES.keys():
        colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())
    
    # Малюємо кожен detection
    for i in range(results['num_detections']):
        if results['scores'][i] < min_score:
            continue
        
        # Координати box (normalized)
        ymin, xmin, ymax, xmax = results['boxes'][i]
        
        # Конвертуємо в pixel координати
        left = int(xmin * width)
        right = int(xmax * width)
        top = int(ymin * height)
        bottom = int(ymax * height)
        
        # Клас та score
        class_id = results['classes'][i]
        score = results['scores'][i]
        class_name = COCO_CLASSES.get(class_id, f'class_{class_id}')
        
        # Колір для цього класу
        color = colors.get(class_id, (0, 255, 0))
        
        # Малюємо box
        cv2.rectangle(img_with_boxes, (left, top), (right, bottom), color, 2)
        
        # Текст з класом та score
        label = f'{class_name}: {score:.2f}'
        
        # Фон для тексту
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        cv2.rectangle(
            img_with_boxes,
            (left, top - label_height - baseline - 5),
            (left + label_width, top),
            color,
            -1
        )
        
        # Текст
        cv2.putText(
            img_with_boxes,
            label,
            (left, top - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img_with_boxes


def visualize_detections(images, detector, model_name):
    """Візуалізує детекції на тестових зображеннях"""
    print_section("ДЕТЕКЦІЯ НА ТЕСТОВИХ ЗОБРАЖЕННЯХ")
    
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 2, figsize=(16, 6 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, image in enumerate(images):
        print(f"\n  Обробка зображення {idx+1}/{n_images}...")
        
        # Детекція
        results = detect_objects(detector, image, confidence_threshold=0.3)
        
        print(f"  ✓ Знайдено {results['num_detections']} об'єктів")
        print(f"  ✓ Час інференції: {results['inference_time']*1000:.1f}ms")
        
        # Малюємо
        img_with_boxes = draw_detections(image, results, min_score=0.3)
        
        # Візуалізація
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Оригінал #{idx+1}', fontsize=12, weight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img_with_boxes)
        axes[idx, 1].set_title(
            f'Детекції #{idx+1} ({results["num_detections"]} об\'єктів, {results["inference_time"]*1000:.0f}ms)',
            fontsize=12, weight='bold'
        )
        axes[idx, 1].axis('off')
        
        # Виводимо знайдені класи
        detected_classes = {}
        for i in range(results['num_detections']):
            class_id = results['classes'][i]
            class_name = COCO_CLASSES.get(class_id, f'class_{class_id}')
            score = results['scores'][i]
            
            if class_name not in detected_classes:
                detected_classes[class_name] = []
            detected_classes[class_name].append(score)
        
        print(f"  Знайдені класи:")
        for class_name, scores in detected_classes.items():
            avg_score = np.mean(scores)
            count = len(scores)
            print(f"    - {class_name}: {count}x (avg score: {avg_score:.3f})")
    
    plt.suptitle(f'Object Detection: {model_name.upper()}', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'detections_{model_name}.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Збережено: results/detections_{model_name}.png")
    plt.show()


def compute_iou(box1, box2):
    """Обчислює IoU (Intersection over Union) між двома boxes"""
    
    # Координати перетину
    ymin = max(box1[0], box2[0])
    xmin = max(box1[1], box2[1])
    ymax = min(box1[2], box2[2])
    xmax = min(box1[3], box2[3])
    
    # Площа перетину
    intersection_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    
    # Площі boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Union
    union_area = box1_area + box2_area - intersection_area
    
    # IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou


def evaluate_detector(detector, test_images, ground_truth, iou_threshold=0.5):
    """Оцінює точність детектора на тестовому наборі"""
    print_section("ОЦІНКА ТОЧНОСТІ ДЕТЕКТОРА")
    
    all_precisions = []
    all_recalls = []
    all_ious = []
    all_fps = []
    
    for idx, (image, gt) in enumerate(zip(test_images, ground_truth)):
        print(f"\n  Оцінка зображення {idx+1}/{len(test_images)}...")
        
        # Детекція
        results = detect_objects(detector, image, confidence_threshold=0.5)
        
        # FPS
        fps = 1.0 / results['inference_time']
        all_fps.append(fps)
        
        # Ground truth boxes
        gt_boxes = gt['boxes']
        gt_classes = gt['classes']
        
        # Predicted boxes
        pred_boxes = results['boxes']
        pred_classes = results['classes']
        pred_scores = results['scores']
        
        # Matching predictions to ground truth
        matched_gt = set()
        true_positives = 0
        false_positives = 0
        
        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx]
            pred_class = pred_classes[pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Знаходимо найкраще співпадіння з ground truth
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                
                gt_box = gt_boxes[gt_idx]
                gt_class = gt_classes[gt_idx]
                
                # Класи мають співпадати
                if pred_class != gt_class:
                    continue
                
                iou = compute_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Перевіряємо чи це true positive
            if best_iou >= iou_threshold and best_gt_idx != -1:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                all_ious.append(best_iou)
            else:
                false_positives += 1
        
        # False negatives - ground truth boxes, які не були знайдені
        false_negatives = len(gt_boxes) - len(matched_gt)
        
        # Precision та Recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        
        print(f"  ✓ Precision: {precision:.3f}")
        print(f"  ✓ Recall: {recall:.3f}")
        print(f"  ✓ FPS: {fps:.1f}")
    
    # Середні метрики
    metrics = {
        'mean_precision': np.mean(all_precisions),
        'mean_recall': np.mean(all_recalls),
        'mean_iou': np.mean(all_ious) if all_ious else 0,
        'mean_fps': np.mean(all_fps),
        'f1_score': 2 * np.mean(all_precisions) * np.mean(all_recalls) / (np.mean(all_precisions) + np.mean(all_recalls)) if (np.mean(all_precisions) + np.mean(all_recalls)) > 0 else 0
    }
    
    print("\n📊 Підсумкові метрики:")
    print(f"  Precision: {metrics['mean_precision']:.3f}")
    print(f"  Recall: {metrics['mean_recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.3f}")
    print(f"  Mean FPS: {metrics['mean_fps']:.1f}")
    
    return metrics


def benchmark_models(images):
    """Порівнює різні моделі детекції"""
    print_section("БЕНЧМАРК МОДЕЛЕЙ")
    
    models_to_test = [
        'ssd_mobilenet_v2',
        'efficientdet_d0',
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"  Тестування: {model_name.upper()}")
        print('='*80)
        
        try:
            # Завантажуємо модель
            detector, _ = load_detection_model(model_name)
            
            if detector is None:
                continue
            
            # Тестуємо на зображеннях
            total_time = 0
            total_detections = 0
            
            for idx, image in enumerate(images):
                result = detect_objects(detector, image, confidence_threshold=0.5)
                total_time += result['inference_time']
                total_detections += result['num_detections']
            
            avg_time = total_time / len(images)
            avg_fps = 1.0 / avg_time
            avg_detections = total_detections / len(images)
            
            results[model_name] = {
                'Avg Time (ms)': avg_time * 1000,
                'Avg FPS': avg_fps,
                'Avg Detections': avg_detections
            }
            
            print(f"\n  ✓ Середній час: {avg_time*1000:.1f}ms")
            print(f"  ✓ Середній FPS: {avg_fps:.1f}")
            print(f"  ✓ Середня к-сть детекцій: {avg_detections:.1f}")
            
            # Візуалізуємо детекції
            visualize_detections(images[:2], detector, model_name)
            
            # Очищуємо пам'ять
            del detector
            
        except Exception as e:
            print(f"  ⚠ Помилка при тестуванні {model_name}: {e}")
    
    # Порівняльна таблиця
    if results:
        print("\n📊 Порівняльна таблиця:")
        results_df = pd.DataFrame(results).T
        print(results_df.round(2).to_string())
        
        # Зберігаємо
        results_df.to_csv(OUTPUT_DIR / 'models_benchmark.csv')
        print(f"\n✓ Збережено: results/models_benchmark.csv")
        
        # Графік порівняння
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # FPS
        ax = axes[0]
        results_df['Avg FPS'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title('Швидкість моделей (FPS)', fontsize=13, weight='bold')
        ax.set_ylabel('FPS', fontsize=11)
        ax.set_xlabel('Модель', fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Time
        ax = axes[1]
        results_df['Avg Time (ms)'].plot(kind='bar', ax=ax, color='coral', edgecolor='black')
        ax.set_title('Час інференції (ms)', fontsize=13, weight='bold')
        ax.set_ylabel('Milliseconds', fontsize=11)
        ax.set_xlabel('Модель', fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Порівняння швидкості моделей', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'models_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Збережено: results/models_comparison.png")
        plt.show()
    
    return results


def live_detection(model_name='ssd_mobilenet_v2', confidence_threshold=0.5):
    """Запускає детекцію у live режимі з веб-камери"""
    print_section("LIVE ДЕТЕКЦІЯ З ВЕБ-КАМЕРИ")
    
    print(f"\n  Завантаження моделі: {model_name}...")
    detector, _ = load_detection_model(model_name)
    
    if detector is None:
        print("  ⚠ Не вдалося завантажити модель")
        return
    
    print("\n  Відкриття веб-камери...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("  ⚠ Не вдалося відкрити веб-камеру")
        return
    
    print("\n✓ Live детекція запущена!")
    print("\n  Керування:")
    print("    - 'q' або 'ESC' - вихід")
    print("    - 's' - зберегти кадр")
    print("    - '+' - збільшити confidence threshold")
    print("    - '-' - зменшити confidence threshold")
    
    frame_count = 0
    fps_history = []
    saved_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("  ⚠ Не вдалося прочитати кадр")
                break
            
            # Конвертуємо BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Детекція
            start_time = time.time()
            results = detect_objects(detector, frame_rgb, confidence_threshold=confidence_threshold)
            inference_time = time.time() - start_time
            
            # Малюємо детекції
            frame_with_boxes = draw_detections(frame_rgb, results, min_score=confidence_threshold)
            frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
            
            # Обчислюємо FPS
            fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            # Додаємо інфо на екран
            info_text = [
                f'FPS: {avg_fps:.1f}',
                f'Detections: {results["num_detections"]}',
                f'Confidence: {confidence_threshold:.2f}',
                f'Time: {inference_time*1000:.0f}ms'
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(
                    frame_with_boxes,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                y_offset += 30
            
            # Показуємо кадр
            cv2.imshow('Object Detection - Live', frame_with_boxes)
            
            # Обробка клавіш
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q або ESC
                print("\n  Зупинка...")
                break
            elif key == ord('s'):  # Зберегти кадр
                saved_frames += 1
                filename = OUTPUT_DIR / f'live_capture_{saved_frames}.jpg'
                cv2.imwrite(str(filename), frame_with_boxes)
                print(f"\n  ✓ Збережено кадр: {filename}")
            elif key == ord('+') or key == ord('='):  # Збільшити threshold
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"\n  Confidence threshold: {confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):  # Зменшити threshold
                confidence_threshold = max(0.05, confidence_threshold - 0.05)
                print(f"\n  Confidence threshold: {confidence_threshold:.2f}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n  Перервано користувачем")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Live детекція завершена")
        print(f"  Оброблено кадрів: {frame_count}")
        print(f"  Збережено кадрів: {saved_frames}")
        print(f"  Середній FPS: {np.mean(fps_history):.1f}")


def main():
    """Головна функція"""
    print("\n" + "="*80)
    print("  OBJECT DETECTION")
    print("  Детектування об'єктів з навчанням та live режимом")
    print("="*80)
    
    print("\n📋 Меню:")
    print("  1. Завантажити тестові зображення та протестувати моделі")
    print("  2. Порівняльний бенчмарк моделей")
    print("  3. Live детекція з веб-камери")
    print("  4. Виконати все")
    
    choice = input("\n  Виберіть опцію (1-4): ").strip()
    
    if choice == '1' or choice == '4':
        # Завантажуємо тестові зображення
        images = download_sample_images()
        
        if not images:
            print("\n⚠ Не вдалося завантажити зображення. Використовую fallback...")
            # Створюємо тестове зображення
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            images = [test_img]
        
        # Завантажуємо та тестуємо модель
        detector, model_name = load_detection_model('ssd_mobilenet_v2')
        
        if detector:
            visualize_detections(images, detector, model_name)
    
    if choice == '2' or choice == '4':
        # Завантажуємо зображення якщо ще не завантажені
        if choice == '2':
            images = download_sample_images()
            if not images:
                test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                images = [test_img]
        
        # Бенчмарк моделей
        benchmark_results = benchmark_models(images)
    
    if choice == '3' or choice == '4':
        # Live детекція
        print("\n" + "="*80)
        if choice == '4':
            response = input("\n  Запустити live детекцію? (y/n): ").strip().lower()
            if response != 'y':
                print("\n  Live детекцію пропущено")
                return
        
        live_detection(model_name='ssd_mobilenet_v2', confidence_threshold=0.5)
    
    # Підсумок
    print_section("ПІДСУМОК")
    print("\n✅ Аналіз завершено!")
    print("\n📁 Створені файли:")
    print("  - results/detections_*.png - детекції на тестових зображеннях")
    print("  - results/models_benchmark.csv - порівняння моделей")
    print("  - results/models_comparison.png - графік порівняння")
    print("  - results/live_capture_*.jpg - збережені кадри з live детекції")
    
    print("\n💡 Підказки:")
    print("  - Для кращої точності використовуйте faster_rcnn_resnet50")
    print("  - Для швидкості використовуйте ssd_mobilenet_v2")
    print("  - У live режимі можна регулювати confidence threshold")
    print("  - Натисніть 's' в live режимі для збереження кадру")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

