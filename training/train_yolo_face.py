"""
YOLO 얼굴 탐지 모델 학습 스크립트
"""

import os
import yaml
from ultralytics import YOLO
import argparse
from pathlib import Path

def create_dataset_yaml(data_path: str, output_path: str):
    """
    YOLO 학습용 데이터셋 YAML 파일 생성
    
    Args:
        data_path: 데이터셋 루트 경로
        output_path: YAML 파일 저장 경로
    """
    dataset_config = {
        'path': data_path,  # 데이터셋 루트 경로
        'train': 'train/images',  # 학습 이미지 경로
        'val': 'val/images',      # 검증 이미지 경로
        'test': 'test/images',    # 테스트 이미지 경로 (선택사항)
        
        # 클래스 정의
        'nc': 1,  # 클래스 수 (얼굴 1개)
        'names': ['face']  # 클래스 이름
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"데이터셋 설정 파일 생성: {output_path}")

def train_yolo_face_detector(
    data_yaml: str,
    model_size: str = 'n',
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = 'auto',
    save_dir: str = 'runs/face_detection',
    pretrained: bool = True,
    resume: str = None
):
    """
    YOLO 얼굴 탐지 모델 학습
    
    Args:
        data_yaml: 데이터셋 YAML 파일 경로
        model_size: 모델 크기 (n, s, m, l, x)
        epochs: 학습 에폭 수
        imgsz: 입력 이미지 크기
        batch_size: 배치 크기
        device: 디바이스 (auto, cpu, 0, 1,...)
        save_dir: 모델 저장 디렉토리
        pretrained: 사전 훈련된 모델 사용 여부
        resume: 재개할 체크포인트 경로
    """
    
    # 모델 초기화
    if resume and os.path.exists(resume):
        print(f"체크포인트에서 재개: {resume}")
        model = YOLO(resume)
    else:
        model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
        model = YOLO(model_name)
    
    print(f"모델 초기화 완료: {model_name}")
    print(f"학습 데이터: {data_yaml}")
    print(f"에폭 수: {epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"이미지 크기: {imgsz}")
    
    # 학습 시작
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=save_dir,
        name='face_detector',
        save=True,
        save_period=10,  # 10 에폭마다 체크포인트 저장
        val=True,
        plots=True,
        verbose=True,
        resume=bool(resume)  # 재개 모드
    )
    
    print("학습 완료!")
    print(f"최고 mAP50: {results.box.map50}")
    print(f"최고 mAP50-95: {results.box.map}")
    
    # 최종 모델 저장
    best_model_path = os.path.join(save_dir, 'face_detector', 'weights', 'best.pt')
    print(f"최고 성능 모델: {best_model_path}")
    
    return results, best_model_path

def validate_model(model_path: str, data_yaml: str):
    """모델 검증"""
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    print("검증 결과:")
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")
    print(f"Precision: {results.box.mp}")
    print(f"Recall: {results.box.mr}")

def main():
    parser = argparse.ArgumentParser(description='YOLO 얼굴 탐지 모델 학습')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='데이터셋 루트 경로')
    parser.add_argument('--model_size', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='모델 크기')
    parser.add_argument('--epochs', type=int, default=100,
                       help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='입력 이미지 크기')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto, cpu, 0, 1,...)')
    parser.add_argument('--save_dir', type=str, default='runs/face_detection',
                       help='모델 저장 디렉토리')
    parser.add_argument('--validate', action='store_true',
                       help='학습 후 검증 수행')
    parser.add_argument('--resume', type=str, 
                       help='재개할 체크포인트 경로')
    
    args = parser.parse_args()
    
    # 데이터셋 YAML 파일 생성
    data_yaml = os.path.join(args.data_path, 'dataset.yaml')
    if not os.path.exists(data_yaml):
        create_dataset_yaml(args.data_path, data_yaml)
    
    # 학습 실행
    results, best_model_path = train_yolo_face_detector(
        data_yaml=data_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=args.save_dir,
        resume=args.resume
    )
    
    # 검증 (선택사항)
    if args.validate:
        validate_model(best_model_path, data_yaml)

if __name__ == "__main__":
    main()

"""
사용 예제:

1. 기본 학습:
python train_yolo_face.py --data_path /path/to/face_dataset

2. 고급 설정:
python train_yolo_face.py \
    --data_path /path/to/face_dataset \
    --model_size m \
    --epochs 200 \
    --batch_size 32 \
    --device 0 \
    --validate

데이터셋 구조:
face_dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml (자동 생성)

라벨 형식 (YOLO format):
class_id x_center y_center width height
예: 0 0.5 0.5 0.3 0.4
"""