"""
커스텀 라벨 형식에서 YOLO와 EfficientNet 데이터셋을 준비하는 통합 스크립트
"""

import os
import argparse
from pathlib import Path
import sys

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from utils.label_converter import (
    convert_to_yolo_format, 
    extract_faces_for_classification,
    create_train_val_split
)
from utils.data_utils import validate_dataset, print_dataset_stats

def prepare_full_pipeline(
    image_dir: str,
    label_dir: str,
    output_base_dir: str = 'data',
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05
):
    """
    전체 데이터 준비 파이프라인
    
    Args:
        image_dir: 원본 이미지 디렉토리
        label_dir: 커스텀 라벨 디렉토리 (RECT,x1,y1,x2,y2,label 형식)
        output_base_dir: 출력 베이스 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
    """
    
    output_path = Path(output_base_dir)
    output_path.mkdir(exist_ok=True)
    
    # 경로를 정규화하여 출력
    image_dir_norm = os.path.normpath(image_dir)
    label_dir_norm = os.path.normpath(label_dir)
    output_dir_norm = os.path.normpath(output_base_dir)
    
    print("=== 딥페이크 탐지 데이터셋 준비 파이프라인 ===")
    print(f"입력 이미지: {image_dir_norm}")
    print(f"입력 라벨: {label_dir_norm}")
    print(f"출력 디렉토리: {output_dir_norm}")
    
    # 1. YOLO 얼굴 탐지용 데이터 준비
    print("\n1. YOLO 얼굴 탐지 데이터셋 생성 중...")
    yolo_dir = output_path / 'face_detection'
    convert_to_yolo_format(image_dir, label_dir, os.path.normpath(str(yolo_dir)))
    
    # 2. 딥페이크 분류용 얼굴 추출
    print("\n2. 딥페이크 분류용 얼굴 추출 중...")
    faces_dir = output_path / 'extracted_faces'
    extract_faces_for_classification(
        image_dir, label_dir, os.path.normpath(str(faces_dir)),
        face_size=(224, 224),
        padding=0.15
    )
    
    # 3. 분류 데이터를 train/val/test로 분할
    print("\n3. 분류 데이터 분할 중...")
    classification_dir = output_path / 'deepfake_classification'
    
    # 3단계 분할을 위한 함수
    import random
    import shutil
    
    random.seed(42)
    
    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        for class_name in ['real', 'fake']:
            (classification_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # 각 클래스별로 분할
    for class_name in ['real', 'fake']:
        class_dir = faces_dir / class_name
        if not class_dir.exists():
            continue
        
        # 이미지 파일 수집
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        random.shuffle(image_files)
        
        total_files = len(image_files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = total_files - train_count - val_count
        
        print(f"  {class_name} 클래스: 총 {total_files}개")
        print(f"    - Train: {train_count}개 ({train_ratio*100:.1f}%)")
        print(f"    - Val: {val_count}개 ({val_ratio*100:.1f}%)")
        print(f"    - Test: {test_count}개 ({test_ratio*100:.1f}%)")
        
        # 분할 및 복사
        splits = [
            ('train', image_files[:train_count]),
            ('val', image_files[train_count:train_count + val_count]),
            ('test', image_files[train_count + val_count:])
        ]
        
        for split_name, files in splits:
            for img_file in files:
                target_file = classification_dir / split_name / class_name / img_file.name
                shutil.copy2(img_file, target_file)
    
    # 4. 데이터셋 검증 및 통계
    print("\n4. 데이터셋 검증 중...")
    
    # YOLO 데이터셋 통계
    print("\n=== YOLO 얼굴 탐지 데이터셋 ===")
    yolo_images = list((yolo_dir / 'images').glob('*'))
    yolo_labels = list((yolo_dir / 'labels').glob('*'))
    print(f"이미지: {len(yolo_images)}개")
    print(f"라벨: {len(yolo_labels)}개")
    
    # 분류 데이터셋 통계
    print("\n=== 딥페이크 분류 데이터셋 ===")
    stats = validate_dataset(str(classification_dir))
    print_dataset_stats(stats)
    
    # 5. 설정 파일 업데이트
    print("\n5. 설정 파일 업데이트 중...")
    
    # config/train_config.yaml 업데이트
    config_file = Path('config/train_config.yaml')
    if config_file.exists():
        import yaml
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 경로 업데이트 (정규화된 경로 사용)
        config['yolo']['data_path'] = os.path.normpath(str(yolo_dir.absolute())).replace('\\', '/')
        config['deepfake_classifier']['data_path'] = os.path.normpath(str(classification_dir.absolute())).replace('\\', '/')
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"설정 파일 업데이트: {os.path.normpath(str(config_file))}")
    
    print("\n=== 데이터 준비 완료 ===")
    print(f"YOLO 데이터: {os.path.normpath(str(yolo_dir))}")
    print(f"분류 데이터: {os.path.normpath(str(classification_dir))}")
    print(f"추출된 얼굴: {os.path.normpath(str(faces_dir))}")
    
    print("\n다음 단계:")
    print("1. YOLO 학습: python resume_training.py yolo --data_path", os.path.normpath(str(yolo_dir)))
    print("2. 분류기 학습: python resume_training.py classifier --data_dir", os.path.normpath(str(classification_dir)))
    print("3. 통합 학습: python resume_training.py config")

def main():
    parser = argparse.ArgumentParser(description='딥페이크 탐지 데이터셋 준비')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='원본 이미지 디렉토리')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='커스텀 라벨 디렉토리')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='출력 디렉토리')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='학습 데이터 비율')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='검증 데이터 비율')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                       help='테스트 데이터 비율')
    
    args = parser.parse_args()
    
    # 비율 검증
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"오류: 데이터 분할 비율의 합이 1이 아닙니다 ({total_ratio})")
        return
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.image_dir):
        print(f"오류: 이미지 디렉토리를 찾을 수 없습니다: {args.image_dir}")
        return
    
    if not os.path.exists(args.label_dir):
        print(f"오류: 라벨 디렉토리를 찾을 수 없습니다: {args.label_dir}")
        return
    
    # 파이프라인 실행
    prepare_full_pipeline(
        args.image_dir,
        args.label_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

if __name__ == "__main__":
    main()

"""
사용 예제:

1. 기본 사용법:
python prepare_data.py --image_dir path/to/images --label_dir path/to/labels

2. 커스텀 분할 비율:
python prepare_data.py \
    --image_dir path/to/images \
    --label_dir path/to/labels \
    --output_dir my_data \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1

3. 실행 후 바로 학습:
python prepare_data.py --image_dir images --label_dir labels
python resume_training.py config

라벨 형식:
각 이미지에 대응하는 .txt 파일에서:
RECT,x1,y1,x2,y2,label
RECT,138,167,187,219,fake
RECT,348,249,376,286,real
...

여기서:
- x1,y1: 박스 왼쪽 위 좌표
- x2,y2: 박스 오른쪽 아래 좌표  
- label: fake, real, none (none은 real로 처리)
"""