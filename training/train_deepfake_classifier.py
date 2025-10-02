"""
EfficientNet 딥페이크 분류 모델 학습 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
from pathlib import Path

# 상위 디렉토리에서 모델 import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deepfake_classifier import DeepfakeClassifier

class DeepfakeDataset(Dataset):
    """딥페이크 분류용 데이터셋 클래스"""
    
    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
            transform: 이미지 변환
            split: 'train', 'val', 'test'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        
        # 이미지 경로와 라벨 수집
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """이미지 경로와 라벨 로드"""
        split_dir = self.data_dir / self.split

        # real 이미지 (라벨 0) - 대소문자 구분 없이 처리
        for class_name in ['real', 'REAL', 'Real']:
            real_dir = split_dir / class_name
            if real_dir.exists():
                for img_path in real_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((str(img_path), 0))
                break  # 하나라도 찾으면 종료

        # fake 이미지 (라벨 1) - 대소문자 구분 없이 처리
        for class_name in ['fake', 'FAKE', 'Fake']:
            fake_dir = split_dir / class_name
            if fake_dir.exists():
                for img_path in fake_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((str(img_path), 1))
                break  # 하나라도 찾으면 종료
        
        print(f"{self.split} 데이터셋: {len(self.samples)}개 샘플")
        
        # 클래스별 개수 출력
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        print(f"  - Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {img_path}, {e}")
            # 더미 이미지 생성
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(augment: bool = True):
    """데이터 변환 파이프라인 생성"""
    
    if augment:
        # 학습용 (데이터 증강 포함)
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 검증/테스트용 (증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 예측값 저장
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(targets, predictions)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(targets, predictions)
    
    # 정밀도, 재현율, F1 스코어 계산
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='binary'
    )
    
    return epoch_loss, epoch_acc, precision, recall, f1, predictions, targets

def train_deepfake_classifier(
    data_dir: str,
    model_name: str = 'efficientnet-b0',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    save_dir: str = 'runs/deepfake_classifier',
    augment: bool = True,
    device: str = 'auto',
    resume: str = None
):
    """
    딥페이크 분류 모델 학습
    
    Args:
        data_dir: 데이터 디렉토리 경로
        model_name: EfficientNet 모델 이름
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        weight_decay: 가중치 감쇠
        save_dir: 모델 저장 디렉토리
        augment: 데이터 증강 사용 여부
        device: 디바이스
        resume: 재개할 체크포인트 경로
    """
    
    # 디바이스 설정
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"사용 디바이스: {device}")
    
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 데이터 변환
    train_transform, val_transform = get_transforms(augment)
    
    # 데이터셋 생성
    print("\n데이터셋 로드 중...")
    train_dataset = DeepfakeDataset(data_dir, train_transform, 'train')
    val_dataset = DeepfakeDataset(data_dir, val_transform, 'val')

    # 데이터셋 검증
    if len(train_dataset) == 0:
        raise ValueError(f"학습 데이터셋이 비어있습니다: {data_dir}/train")
    if len(val_dataset) == 0:
        raise ValueError(f"검증 데이터셋이 비어있습니다: {data_dir}/val")

    # 최소 데이터 크기 경고
    if len(train_dataset) < 100:
        print(f"⚠️  경고: 학습 데이터가 너무 적습니다 ({len(train_dataset)}개). 최소 100개 이상 권장합니다.")
    if len(val_dataset) < 20:
        print(f"⚠️  경고: 검증 데이터가 너무 적습니다 ({len(val_dataset)}개). 최소 20개 이상 권장합니다.")

    # num_workers 설정 (Windows에서는 0, Linux/Mac에서는 4)
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4

    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )

    print(f"✅ 데이터로더 생성 완료 (num_workers={num_workers})")
    
    # 모델 생성
    model = DeepfakeClassifier(model_name=model_name, num_classes=2, pretrained=True)
    model.to(device)
    
    # 손실 함수와 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 체크포인트에서 재개
    start_epoch = 0
    if resume and os.path.exists(resume):
        try:
            print(f"\n체크포인트에서 재개: {resume}")
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"✅ 에폭 {start_epoch}부터 재개")
        except Exception as e:
            print(f"⚠️  체크포인트 로드 실패: {e}")
            print("새로운 학습을 시작합니다.")
            start_epoch = 0
    else:
        print("\n새로운 학습 시작")
    
    # 학습 이력 저장용
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    print("학습 시작...")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n에폭 [{epoch+1}/{epochs}]")
        
        # 학습
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 검증
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 이력 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        
        # 체크포인트 저장 (매 10 에폭마다)
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, checkpoint_path)
            print(f"체크포인트 저장: {checkpoint_path}")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 최고 성능 모델을 위한 확장된 체크포인트 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            }, best_model_path)
            print(f"새로운 최고 성능 모델 저장! Val Acc: {val_acc:.4f}")
    
    # 학습 완료
    print(f"\n학습 완료! 최고 검증 정확도: {best_val_acc:.4f}")
    
    # 학습 이력 저장
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 학습 곡선 플롯
    plot_training_curves(history, save_dir)
    
    return model, best_model_path, history

def plot_training_curves(history, save_dir):
    """학습 곡선 플롯"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision, Recall, F1
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].plot(history['val_f1'], label='F1-Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 학습률
    axes[1, 1].text(0.1, 0.5, 'Training Completed\nCheck training_history.json\nfor detailed metrics', 
                   transform=axes[1, 1].transAxes, fontsize=12,
                   verticalalignment='center')
    axes[1, 1].set_title('Training Info')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='딥페이크 분류 모델 학습')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='데이터 디렉토리 경로')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0',
                       help='EfficientNet 모델 이름')
    parser.add_argument('--epochs', type=int, default=50,
                       help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='가중치 감쇠')
    parser.add_argument('--save_dir', type=str, default='runs/deepfake_classifier',
                       help='모델 저장 디렉토리')
    parser.add_argument('--no_augment', action='store_true',
                       help='데이터 증강 비활성화')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto, cpu, cuda, cuda:0, ...)')
    parser.add_argument('--resume', type=str,
                       help='재개할 체크포인트 경로')
    
    args = parser.parse_args()
    
    # 학습 실행
    model, best_model_path, history = train_deepfake_classifier(
        data_dir=args.data_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        augment=not args.no_augment,
        device=args.device,
        resume=args.resume
    )
    
    print(f"최고 성능 모델: {best_model_path}")

if __name__ == "__main__":
    main()

"""
사용 예제:

1. 기본 학습:
python train_deepfake_classifier.py --data_dir /path/to/deepfake_dataset

2. 고급 설정:
python train_deepfake_classifier.py \
    --data_dir /path/to/deepfake_dataset \
    --model_name efficientnet-b3 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --device cuda:0

데이터셋 구조:
deepfake_dataset/
├── train/
│   ├── real/
│   │   ├── real_001.jpg
│   │   └── real_002.jpg
│   └── fake/
│       ├── fake_001.jpg
│       └── fake_002.jpg
├── val/
│   ├── real/
│   └── fake/
└── test/ (선택사항)
    ├── real/
    └── fake/
"""