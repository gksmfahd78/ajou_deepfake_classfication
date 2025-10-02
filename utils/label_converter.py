"""
커스텀 라벨 형식을 YOLO와 EfficientNet 형식으로 변환하는 유틸리티
"""

import os
import cv2
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

def parse_custom_label(label_path: str) -> List[Dict]:
    """
    커스텀 라벨 파일 파싱
    
    Args:
        label_path: 라벨 파일 경로
        
    Returns:
        얼굴 정보 리스트 [{'bbox': (x1, y1, x2, y2), 'label': 'fake/real'}, ...]
    """
    faces = []
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('RECT'):
                    continue
                
                # RECT,x1,y1,x2,y2,label 형식 파싱
                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        label = parts[5].strip().lower()

                        # 좌표 검증
                        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                            print(f"⚠️  잘못된 좌표 무시: {label_path} - ({x1},{y1},{x2},{y2})")
                            continue

                        # 'none'을 'real'로 처리 (기본값)
                        if label == 'none' or label == '':
                            label = 'real'

                        # 라벨 검증
                        if label not in ['real', 'fake']:
                            print(f"⚠️  알 수 없는 라벨 무시: {label_path} - {label}")
                            continue

                        faces.append({
                            'bbox': (x1, y1, x2, y2),
                            'label': label
                        })
                    except ValueError as e:
                        print(f"⚠️  좌표 파싱 실패: {label_path} - {e}")
                        continue
    
    except Exception as e:
        print(f"라벨 파일 파싱 실패: {label_path}, {e}")
    
    return faces

def convert_to_yolo_format(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
):
    """
    커스텀 라벨을 YOLO 형식으로 변환
    
    Args:
        image_dir: 원본 이미지 디렉토리
        label_dir: 커스텀 라벨 디렉토리  
        output_dir: YOLO 형식 출력 디렉토리
    """
    
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    images_out = output_path / 'images'
    labels_out = output_path / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 수집
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.glob(f'*{ext}'))
        image_files.extend(image_path.glob(f'*{ext.upper()}'))
    
    print(f"총 {len(image_files)}개 이미지 처리 중...")
    print(f"이미지 디렉토리: {os.path.normpath(image_dir)}")
    print(f"라벨 디렉토리: {os.path.normpath(label_dir)}")
    print(f"출력 디렉토리: {os.path.normpath(output_dir)}")
    
    converted_count = 0
    
    for img_file in tqdm(image_files):
        # 대응하는 라벨 파일 찾기
        label_file = label_path / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            print(f"라벨 파일 없음: {label_file}")
            continue
        
        # 이미지 크기 가져오기
        try:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            h, w = image.shape[:2]
        except:
            continue
        
        # 커스텀 라벨 파싱
        faces = parse_custom_label(str(label_file))
        
        if not faces:
            continue
        
        # YOLO 형식으로 변환
        yolo_labels = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']

            # 좌표가 이미지 범위 내인지 검증
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                print(f"⚠️  이미지 범위 벗어남: {img_file.name} - ({x1},{y1},{x2},{y2}), 이미지 크기: ({w},{h})")
                # 좌표를 이미지 범위 내로 제한
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

            # YOLO 형식: (x_center, y_center, width, height) - 정규화됨
            x_center = (x1 + x2) / 2.0 / w
            y_center = (y1 + y2) / 2.0 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            # 정규화된 좌표 검증 (0~1 범위)
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                print(f"⚠️  비정상적인 정규화 좌표: {img_file.name} - center:({x_center:.3f},{y_center:.3f}), size:({width:.3f},{height:.3f})")
                continue

            # 클래스 ID: 0 (얼굴)
            yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # YOLO 라벨 파일 저장
        yolo_label_file = labels_out / f"{img_file.stem}.txt"
        with open(yolo_label_file, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        # 이미지 복사
        shutil.copy2(img_file, images_out / img_file.name)
        
        converted_count += 1
    
    print(f"YOLO 형식 변환 완료: {converted_count}개 파일")
    
    # dataset.yaml 파일 생성
    dataset_yaml = output_path / 'dataset.yaml'
    # 경로를 정규화하여 슬래시로 통일
    normalized_path = os.path.normpath(str(output_path.absolute())).replace('\\', '/')
    yaml_content = f"""path: {normalized_path}
train: images
val: images  # 실제 사용시 별도 분할 필요
test: images  # 선택사항

nc: 1  # 클래스 수 (얼굴)
names: ['face']  # 클래스 이름
"""
    
    with open(dataset_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"YOLO 데이터셋 설정 파일 생성: {os.path.normpath(str(dataset_yaml))}")

def extract_faces_for_classification(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    face_size: Tuple[int, int] = (224, 224),
    padding: float = 0.1,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
):
    """
    커스텀 라벨을 사용하여 얼굴을 추출하고 분류용 데이터셋 생성
    
    Args:
        image_dir: 원본 이미지 디렉토리
        label_dir: 커스텀 라벨 디렉토리
        output_dir: 분류용 데이터셋 출력 디렉토리
        face_size: 출력 얼굴 크기
        padding: 얼굴 주위 여백 비율
    """
    
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성 (REAL/FAKE 폴더 - 대문자)
    real_dir = output_path / 'REAL'
    fake_dir = output_path / 'FAKE'
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 수집
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.glob(f'*{ext}'))
        image_files.extend(image_path.glob(f'*{ext.upper()}'))
    
    print(f"총 {len(image_files)}개 이미지에서 얼굴 추출 중...")
    print(f"이미지 디렉토리: {os.path.normpath(image_dir)}")
    print(f"라벨 디렉토리: {os.path.normpath(label_dir)}")
    print(f"출력 디렉토리: {os.path.normpath(output_dir)}")
    
    real_count = 0
    fake_count = 0
    
    for img_file in tqdm(image_files):
        # 대응하는 라벨 파일 찾기
        label_file = label_path / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            continue
        
        # 이미지 로드
        try:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            h, w = image.shape[:2]
        except:
            continue
        
        # 커스텀 라벨 파싱
        faces = parse_custom_label(str(label_file))
        
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face['bbox']
            label = face['label']
            
            # 패딩 추가
            face_w = x2 - x1
            face_h = y2 - y1
            pad_w = int(face_w * padding)
            pad_h = int(face_h * padding)
            
            # 패딩 적용된 좌표
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(w, x2 + pad_w)
            y2_pad = min(h, y2 + pad_h)
            
            # 얼굴 영역 추출
            face_img = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_img.size == 0:
                continue
            
            # 크기 조정
            face_resized = cv2.resize(face_img, face_size)
            
            # 저장 경로 결정
            if label == 'fake':
                save_dir = fake_dir
                fake_count += 1
            else:  # 'real' 또는 기타
                save_dir = real_dir
                real_count += 1
            
            # 파일명 생성 (중복 방지)
            if len(faces) == 1:
                filename = f"{img_file.stem}.jpg"
            else:
                filename = f"{img_file.stem}_face_{i}.jpg"
            
            # 저장
            save_path = save_dir / filename
            cv2.imwrite(str(save_path), face_resized)
    
    print(f"얼굴 추출 완료:")
    print(f"  - Real: {real_count}개")
    print(f"  - Fake: {fake_count}개")
    print(f"  - 총합: {real_count + fake_count}개")

def create_train_val_split(
    source_dir: str,
    target_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    분류용 데이터셋을 train/val로 분할
    
    Args:
        source_dir: 원본 데이터 디렉토리 (real/fake 폴더 포함)
        target_dir: 분할된 데이터 저장 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
    """
    
    import random
    random.seed(seed)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 출력 디렉토리 생성
    for split in ['train', 'val']:
        for class_name in ['real', 'fake']:
            (target_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # 각 클래스별로 분할
    for class_name in ['real', 'fake']:
        class_dir = source_path / class_name
        if not class_dir.exists():
            continue
        
        # 이미지 파일 수집
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        random.shuffle(image_files)
        
        total_files = len(image_files)
        train_count = int(total_files * train_ratio)
        
        print(f"{class_name} 클래스: 총 {total_files}개")
        print(f"  - Train: {train_count}개")
        print(f"  - Val: {total_files - train_count}개")
        
        # 분할 및 복사
        for i, img_file in enumerate(image_files):
            if i < train_count:
                split = 'train'
            else:
                split = 'val'
            
            target_file = target_path / split / class_name / img_file.name
            shutil.copy2(img_file, target_file)

def main():
    parser = argparse.ArgumentParser(description='커스텀 라벨 형식 변환 도구')
    parser.add_argument('command', choices=['yolo', 'classification', 'split'],
                       help='변환 타입')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='이미지 디렉토리')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='라벨 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='출력 디렉토리')
    parser.add_argument('--face_size', type=int, nargs=2, default=[224, 224],
                       help='얼굴 이미지 크기 (분류용)')
    parser.add_argument('--padding', type=float, default=0.1,
                       help='얼굴 주위 여백 비율')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='학습 데이터 비율')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드')
    
    args = parser.parse_args()
    
    if args.command == 'yolo':
        convert_to_yolo_format(
            args.image_dir,
            args.label_dir,
            args.output_dir
        )
    
    elif args.command == 'classification':
        extract_faces_for_classification(
            args.image_dir,
            args.label_dir,
            args.output_dir,
            tuple(args.face_size),
            args.padding
        )
    
    elif args.command == 'split':
        create_train_val_split(
            args.image_dir,  # source_dir로 사용
            args.output_dir,
            args.train_ratio,
            1.0 - args.train_ratio,
            args.seed
        )

if __name__ == "__main__":
    main()

"""
사용 예제:

1. YOLO 형식으로 변환:
python label_converter.py yolo --image_dir images --label_dir labels --output_dir data/yolo_format

2. 분류용 얼굴 추출:
python label_converter.py classification --image_dir images --label_dir labels --output_dir data/faces

3. 분류 데이터를 train/val로 분할:
python label_converter.py split --image_dir data/faces --output_dir data/classification_split --train_ratio 0.8

전체 파이프라인:
1. 원본 이미지/라벨 → YOLO 형식 변환 (얼굴 탐지용)
2. 원본 이미지/라벨 → 얼굴 추출 (분류용)
3. 추출된 얼굴 → train/val 분할
"""