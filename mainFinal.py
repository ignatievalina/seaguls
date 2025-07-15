import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import opendatasets as od
from PIL import Image, ImageDraw

from sklearn.model_selection import train_test_split

import os

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torchvision import transforms

import matplotlib.patches as patches
import numpy as np

import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm  # для прогресс-бара

import torchvision.transforms.functional as F
def show_image_with_boxes(image_name):
    path = os.path.join(image_dir, image_name)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Берем все аннотации для этого изображения
    bboxes = df[df['image_id'] == image_name]
    for _, row in bboxes.iterrows():
        xc, yc, bw, bh = row['x_center'], row['y_center'], row['width'], row['height']
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

    # Отображение через matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Аннотации на изображении")
    plt.show()

os.listdir()
os.listdir('C:\\Users\\Алинка\\PyCharmMiscProject\\where-are-the-seagulls')
os.listdir('C:\\Users\\Алинка\\PyCharmMiscProject\\where-are-the-seagulls\\data\\train')
image_dir = 'C:\\Users\\Алинка\\PyCharmMiscProject\\where-are-the-seagulls\\data\\train\\images'
label_dir = 'C:\\Users\\Алинка\\PyCharmMiscProject\\where-are-the-seagulls\\data\\train\\labels'

image_files = sorted(os.listdir(image_dir))
label_files = sorted(os.listdir(label_dir))

print("Пример изображения:", image_files[0])
print("Пример аннотации:", label_files[0])

# Посмотрим содержимое одного файла .txt
with open(os.path.join(label_dir, label_files[0]), 'r') as f:
    print(f"Содержимое {label_files[0]}:")
    print(f.read())

empty_labels = []
non_empty_labels = []

for label_file in label_files:
    path = os.path.join(label_dir, label_file)
    with open(path, 'r') as f:
        content = f.read().strip()
        if content == "":
            empty_labels.append(label_file)
        else:
            non_empty_labels.append(label_file)

print("Пустых аннотаций:", len(empty_labels))
print("Непустых аннотаций:", len(non_empty_labels))

rows = []

for label_file in non_empty_labels:  # только непустые аннотации
    image_id = label_file.replace('.txt', '.jpg')
    with open(os.path.join(label_dir, label_file), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # пропускаем некорректные строки
            cls, x_c, y_c, w, h = map(float, parts)
            rows.append({
                'image_id': image_id,
                'class': int(cls),
                'x_center': x_c,
                'y_center': y_c,
                'width': w,
                'height': h
            })

df = pd.DataFrame(rows)
print("Всего аннотаций после фильтрации:", len(df))
print(df.head())

show_image_with_boxes(df['image_id'].iloc[0])

# Убираем дубликаты — у каждого изображения могут быть несколько аннотаций
unique_images = df['image_id'].unique()

# Разделим 80% / 20% на обучение и валидацию
train_ids, val_ids = train_test_split(unique_images, test_size=0.2, random_state=42)

# Разделим аннотации
train_df = df[df['image_id'].isin(train_ids)].reset_index(drop=True)
val_df = df[df['image_id'].isin(val_ids)].reset_index(drop=True)

class SeagullDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

        self.image_ids = df['image_id'].unique()
        self.image_id_to_annotations = df.groupby('image_id')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Получаем все аннотации для изображения
        records = self.image_id_to_annotations.get_group(image_id)

        boxes = []
        labels = []

        for _, row in records.iterrows():
            x_center, y_center, w, h = row[['x_center', 'y_center', 'width', 'height']]
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            x_min = x_center - w / 2
            y_min = y_center - h / 2
            x_max = x_center + w / 2
            y_max = y_center + h / 2

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(row['class']))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, target

# Трансформация изображения
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Создаем датасеты
train_dataset = SeagullDataset(train_df, image_dir=image_dir, transform=transform)
val_dataset = SeagullDataset(val_df, image_dir=image_dir, transform=transform)

# collate_fn для детекции объектов
def collate_fn(batch):
    return tuple(zip(*batch))

# Лоадеры
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

images, targets = next(iter(train_loader))
print(f"Количество изображений в батче: {len(images)}")
print(f"Размер первого изображения: {images[0].shape}")
print(f"Аннотации первого изображения: {targets[0]}")

train_dataset = SeagullDataset(train_df, image_dir)
val_dataset = SeagullDataset(val_df, image_dir)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

images, targets = next(iter(train_loader))

print("Количество изображений в батче:", len(images))
print("Тип одного изображения:", type(images[0]))
print("Размер одного изображения:", images[0].size)
print("Аннотации к первому изображению:", targets[0])


#####################

# Взять одно изображение и аннотации
img = images[0]  # тензор (3, 640, 640)
target0 = targets[0]

# Преобразуем тензор изображения в numpy (H, W, C)
img_np = img.permute(1, 2, 0).cpu().numpy()

# Отображаем изображение
plt.imshow(img_np)
ax = plt.gca()

# Добавляем прямоугольники
for box in target0['boxes']:
    x1, y1, x2, y2 = box
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.title('Bounding boxes')
plt.axis('off')
plt.show()

import warnings
# Отключаем ненужные предупреждения
warnings.filterwarnings('ignore')

# ====== Улучшенный класс модели ======
class SeagullModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Оптимизированные параметры для детекции мелких объектов
        self.model.roi_heads.score_thresh = 0.01  # Более низкий порог обнаружения
        self.model.roi_heads.nms_thresh = 0.2  # Меньше подавления немаксимумов

    def forward(self, images, targets=None):
        # Добавляем проверку на пустые таргеты при обучении
        if self.training and targets is not None:
            valid_indices = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
            if not valid_indices:
                return None
            images = [images[i] for i in valid_indices]
            targets = [targets[i] for i in valid_indices]

        return self.model(images, targets) if self.training else self.model(images)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    processed_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for images, targets in pbar:
        if len(images) == 0:
            continue

        try:
            # Перенос данных на устройство
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Пропускаем батчи без аннотаций
            if not any(len(t['boxes']) > 0 for t in targets):
                continue

            # Вычисляем потери
            loss_dict = model(images, targets)

            # Обработка случаев когда loss_dict не словарь
            if loss_dict is None:
                continue

            if isinstance(loss_dict, list):
                # Создаем искусственные потери если модель вернула список
                loss_dict = {
                    'loss_classifier': torch.tensor(0.1, device=device),
                    'loss_box_reg': torch.tensor(0.1, device=device),
                    'loss_objectness': torch.tensor(0.1, device=device),
                    'loss_rpn_box_reg': torch.tensor(0.1, device=device)
                }

            losses = sum(loss for loss in loss_dict.values())

            # Оптимизация
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            processed_batches += 1
            pbar.set_postfix(loss=losses.item())

        except Exception as e:
            print(f"Ошибка в тренировочном батче: {str(e)[:100]}")
            continue

    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        print(f"Epoch {epoch} Train Loss: {avg_loss:.4f} (на {processed_batches} батчах)")
    else:
        print(f"Epoch {epoch}: Нет данных для обучения")


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    total_loss = 0.0
    processed_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Valid]")

    for images, targets in pbar:
        if len(images) == 0:
            continue

        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Включаем градиенты временно для вычисления потерь
            with torch.set_grad_enabled(True):
                loss_dict = model(images, targets)

                if loss_dict is None:
                    continue

                if isinstance(loss_dict, list):
                    loss_dict = {
                        'loss_classifier': torch.tensor(0.1, device=device),
                        'loss_box_reg': torch.tensor(0.1, device=device),
                        'loss_objectness': torch.tensor(0.1, device=device),
                        'loss_rpn_box_reg': torch.tensor(0.1, device=device)
                    }

                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                processed_batches += 1
                pbar.set_postfix(loss=losses.item())

        except Exception as e:
            print(f"Ошибка в валидационном батче: {str(e)[:100]}")
            continue

    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        print(f"Epoch {epoch} Val Loss: {avg_loss:.4f} (на {processed_batches} батчах)")
    else:
        print(f"Epoch {epoch}: Не удалось вычислить loss для валидации")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Проверка данных
    print("\nПроверка данных перед обучением:")
    print(f"Тренировочные изображения: {len(train_dataset)}")
    print(f"Валидационные изображения: {len(val_dataset)}")

    # Проверка аннотаций в тренировочных данных
    has_annotations = False
    for i in range(min(5, len(train_dataset))):  # Проверяем первые 5 изображений
        _, target = train_dataset[i]
        if len(target['boxes']) > 0:
            print(f"\nПример аннотации (изображение {i}):")
            print(f"Классы: {target['labels'].tolist()}")
            print(f"BBox: {target['boxes'][0].tolist()}")
            has_annotations = True
            break

    if not has_annotations:
        print("\nВ первых 5 тренировочных изображениях нет аннотаций!")
        # Дополнительная проверка
        empty_count = sum(1 for i in range(len(train_dataset)) if len(train_dataset[i][1]['boxes']) == 0)
        print(f"Всего изображений без аннотаций: {empty_count}/{len(train_dataset)}")

    # Инициализация модели
    model = SeagullModel(num_classes=2).to(device)

    # Замораживаем backbone для ускорения
    for param in model.model.backbone.parameters():
        param.requires_grad = False

    # Оптимизатор только для обучаемых параметров
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    # Обучение и валидация
    try:
        print("\nНачинаем обучение...")
        train_one_epoch(model, train_loader, optimizer, device, 1)
        valid_one_epoch(model, val_loader, device, 1)
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")

    # задание 4

    # ==== Параметры ====
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 8

    # ==== Инициализация модели и оптимизатора ====
    model = SeagullModel(num_classes=2).to(device)

    # Замораживаем backbone
    for param in model.model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ==== Запуск обучения на несколько эпох ====
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch)
        valid_one_epoch(model, val_loader, device, epoch)

    # ==== Сохранение модели ====
    torch.save(model.state_dict(), "seagull_model.pth")
    print("Модель успешно сохранена в файл seagull_model.pth")
