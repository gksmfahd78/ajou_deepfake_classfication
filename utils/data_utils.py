"""
데이터 전처리 및 관리 유틸리티
"""

import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
from typing import List, Tuple, Dict
from tqdm import tqdm
import argparse

def split_dataset(
    source_dir: str, 
    target_dir: str, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.2, 
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    데이터셋을 train/val/test로 분할
    
    Args:
        source_dir: 원본 데이터 디렉토리
        target_dir: 분할된 데이터 저장 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        seed: 랜덤 시드
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합이 1이 아닙니다."
    
    random.seed(seed)
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 클래스별로 처리 (real, fake)
    for class_name in ['real', 'fake']:
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"경고: {class_dir} 디렉토리를 찾을 수 없습니다.")
            continue
        
        # 이미지 파일 수집
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(class_dir.glob(ext))
        
        print(f"{class_name} 클래스: {len(image_files)}개 파일")
        
        # 랜덤 셔플
        random.shuffle(image_files)
        
        # 분할 계산
        total_files = len(image_files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = total_files - train_count - val_count
        
        print(f"  - Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        # 분할 및 복사
        splits = [
            ('train', image_files[:train_count]),
            ('val', image_files[train_count:train_count + val_count]),
            ('test', image_files[train_count + val_count:])
        ]
        
        for split_name, files in splits:
            if len(files) == 0:
                continue
                
            split_dir = target_path / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in tqdm(files, desc=f"복사 중 {split_name}/{class_name}"):
                target_file = split_dir / file_path.name
                shutil.copy2(file_path, target_file)

def create_yolo_labels_from_faces(
    image_dir: str, 
    output_dir: str, 
    face_detector_model: str = None
):
    """
    얼굴 이미지에서 YOLO 라벨 자동 생성
    
    Args:
        image_dir: 얼굴 이미지가 있는 디렉토리
        output_dir: 라벨 파일 저장 디렉토리
        face_detector_model: 얼굴 탐지 모델 경로
    """
    
    # 얼굴 탐지기 초기화 (OpenCV 사용)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 수집
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(image_path.glob(ext))
    
    print(f"{len(image_files)}개 이미지에서 얼굴 라벨 생성 중...")
    
    success_count = 0
    
    for img_file in tqdm(image_files):
        try:
            # 이미지 로드
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]
            
            # 얼굴 탐지
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # 라벨 파일 생성
                label_file = output_path / (img_file.stem + '.txt')
                
                with open(label_file, 'w') as f:
                    for (x, y, face_w, face_h) in faces:
                        # YOLO 형식으로 변환 (normalized)
                        x_center = (x + face_w / 2) / w
                        y_center = (y + face_h / 2) / h
                        width = face_w / w
                        height = face_h / h
                        
                        # 클래스 0 (얼굴)
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                success_count += 1
        
        except Exception as e:
            print(f"처리 실패: {img_file}, 에러: {e}")
    
    print(f"라벨 생성 완료: {success_count}/{len(image_files)}")

def extract_faces_from_images(
    image_dir: str, 
    output_dir: str, 
    face_size: Tuple[int, int] = (224, 224),
    padding: float = 0.2
):
    """
    이미지에서 얼굴 영역만 추출하여 저장
    
    Args:
        image_dir: 원본 이미지 디렉토리
        output_dir: 추출된 얼굴 이미지 저장 디렉토리
        face_size: 출력 얼굴 이미지 크기
        padding: 얼굴 주위 여백 비율
    """
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 수집
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(image_path.glob(ext))
    
    print(f"{len(image_files)}개 이미지에서 얼굴 추출 중...")
    
    extracted_count = 0
    
    for img_file in tqdm(image_files):
        try:
            # 이미지 로드
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]
            
            # 얼굴 탐지
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for i, (x, y, face_w, face_h) in enumerate(faces):
                # 패딩 추가
                pad_w = int(face_w * padding)
                pad_h = int(face_h * padding)
                
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w, x + face_w + pad_w)
                y2 = min(h, y + face_h + pad_h)
                
                # 얼굴 영역 추출
                face_img = image[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    # 크기 조정
                    face_resized = cv2.resize(face_img, face_size)
                    
                    # 저장
                    if len(faces) == 1:
                        output_file = output_path / f"{img_file.stem}.jpg"
                    else:
                        output_file = output_path / f"{img_file.stem}_face_{i}.jpg"
                    
                    cv2.imwrite(str(output_file), face_resized)
                    extracted_count += 1
        
        except Exception as e:
            print(f"처리 실패: {img_file}, 에러: {e}")
    
    print(f"얼굴 추출 완료: {extracted_count}개")

def validate_dataset(data_dir: str) -> Dict:
    """
    데이터셋 검증 및 통계 생성
    
    Args:
        data_dir: 데이터셋 디렉토리
        
    Returns:
        데이터셋 통계 정보
    """
    
    data_path = Path(data_dir)
    stats = {
        'total_images': 0,
        'splits': {},
        'classes': {},
        'image_sizes': [],
        'file_formats': {}
    }
    
    # 분할별 통계
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
        
        split_stats = {'total': 0, 'classes': {}}
        
        # 클래스별 통계
        for class_name in ['real', 'fake']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            # 이미지 파일 수집
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(class_dir.glob(ext))
            
            split_stats['classes'][class_name] = len(image_files)
            split_stats['total'] += len(image_files)
            
            # 파일 형식 통계
            for img_file in image_files:
                ext = img_file.suffix.lower()
                stats['file_formats'][ext] = stats['file_formats'].get(ext, 0) + 1
                
                # 이미지 크기 확인 (일부만)
                if len(stats['image_sizes']) < 100:
                    try:
                        with Image.open(img_file) as img:
                            stats['image_sizes'].append(img.size)
                    except:
                        pass
        
        stats['splits'][split] = split_stats
        stats['total_images'] += split_stats['total']
    
    # 클래스 전체 통계
    for split_data in stats['splits'].values():
        for class_name, count in split_data['classes'].items():
            stats['classes'][class_name] = stats['classes'].get(class_name, 0) + count
    
    return stats

def print_dataset_stats(stats: Dict):
    """데이터셋 통계 출력"""
    
    print("=== 데이터셋 통계 ===")
    print(f"전체 이미지 수: {stats['total_images']}")
    
    print("\n분할별 통계:")
    for split_name, split_data in stats['splits'].items():
        print(f"  {split_name}: {split_data['total']}개")
        for class_name, count in split_data['classes'].items():
            percentage = (count / split_data['total'] * 100) if split_data['total'] > 0 else 0
            print(f"    - {class_name}: {count}개 ({percentage:.1f}%)")
    
    print("\n클래스별 전체 통계:")
    for class_name, count in stats['classes'].items():
        percentage = (count / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
        print(f"  {class_name}: {count}개 ({percentage:.1f}%)")
    
    print("\n파일 형식:")
    for ext, count in stats['file_formats'].items():
        percentage = (count / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
        print(f"  {ext}: {count}개 ({percentage:.1f}%)")
    
    if stats['image_sizes']:
        print(f"\n이미지 크기 (샘플 {len(stats['image_sizes'])}개):")
        unique_sizes = list(set(stats['image_sizes']))[:10]
        for size in unique_sizes:
            count = stats['image_sizes'].count(size)
            print(f"  {size[0]}x{size[1]}: {count}개")

def main():
    parser = argparse.ArgumentParser(description='데이터 전처리 유틸리티')
    parser.add_argument('command', choices=['split', 'extract_faces', 'create_labels', 'validate'],
                       help='실행할 명령')
    parser.add_argument('--source_dir', type=str, help='원본 데이터 디렉토리')
    parser.add_argument('--target_dir', type=str, help='대상 디렉토리')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='학습 데이터 비율')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='검증 데이터 비율')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='테스트 데이터 비율')
    parser.add_argument('--face_size', type=int, nargs=2, default=[224, 224], help='얼굴 이미지 크기')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        split_dataset(
            args.source_dir, args.target_dir,
            args.train_ratio, args.val_ratio, args.test_ratio,
            args.seed
        )
    
    elif args.command == 'extract_faces':
        extract_faces_from_images(
            args.source_dir, args.target_dir,
            tuple(args.face_size)
        )
    
    elif args.command == 'create_labels':
        create_yolo_labels_from_faces(
            args.source_dir, args.target_dir
        )
    
    elif args.command == 'validate':
        stats = validate_dataset(args.source_dir)
        print_dataset_stats(stats)
        
        # JSON으로 저장
        if args.target_dir:
            output_file = Path(args.target_dir) / 'dataset_stats.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n통계 파일 저장: {output_file}")

if __name__ == "__main__":
    main()

"""
사용 예제:

1. 데이터셋 분할:
python data_utils.py split --source_dir /path/to/original --target_dir /path/to/split

2. 얼굴 추출:
python data_utils.py extract_faces --source_dir /path/to/images --target_dir /path/to/faces

3. YOLO 라벨 생성:
python data_utils.py create_labels --source_dir /path/to/face_images --target_dir /path/to/labels

4. 데이터셋 검증:
python data_utils.py validate --source_dir /path/to/dataset --target_dir /path/to/output
"""