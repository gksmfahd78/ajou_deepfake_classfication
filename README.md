\# 딥페이크 탐지 시스템 (Deepfake Detection System)

아주대 융시공 공모전 준비용입니다.

YOLO와 EfficientNet을 결합한 고성능 딥페이크 탐지 시스템입니다.

## 🎯 시스템 개요

이 시스템은 두 단계로 딥페이크를 탐지합니다:
1. **YOLO v8**: 얼굴 영역 탐지
2. **EfficientNet**: 탐지된 얼굴의 딥페이크 여부 분류

## 🚀 주요 특징

- **단일 데이터셋으로 두 모델 학습**: 한 번의 라벨링으로 얼굴 탐지와 딥페이크 분류 모델 모두 학습
- **커스텀 라벨 형식 지원**: `RECT,x1,y1,x2,y2,label` 형식 자동 변환
- **체크포인트 기반 학습 재개**: 언제든 중단하고 이어서 학습 가능
- **실시간 추론**: 이미지, 비디오, 웹캠 지원
- **ONNX 변환 지원**: 추론 속도 2-5배 향상
- **완전 자동화**: 데이터 준비부터 학습까지 스크립트로 처리

## 📁 프로젝트 구조

```
deepfake_detection/
├── models/                   # 모델 정의
│   ├── face_detector.py      # YOLO 얼굴 탐지기
│   ├── deepfake_classifier.py # EfficientNet 분류기
│   └── __init__.py
├── training/                 # 학습 스크립트
│   ├── train_yolo_face.py    # YOLO 독립 학습
│   ├── train_deepfake_classifier.py # EfficientNet 독립 학습
│   └── __init__.py
├── utils/                    # 유틸리티
│   ├── label_converter.py    # 라벨 형식 변환
│   ├── checkpoint_manager.py # 체크포인트 관리
│   ├── data_utils.py         # 데이터 전처리
│   ├── model_converter.py    # ONNX 변환
│   └── __init__.py
├── config/
│   └── train_config.yaml     # 학습 설정
├── input/                   # 📥 입력 데이터 (사용자가 넣음)
│   ├── images/              # 원본 이미지들
│   └── labels/              # 라벨 파일들 (RECT 형식)
├── pretrained_models/       # 📦 사전 훈련된 모델들 (선택사항)
│   ├── yolo/               # YOLO 모델들 (.pt)
│   └── efficientnet/       # EfficientNet 모델들 (.pth)
├── onnx_models/            # 🚀 ONNX 변환된 모델들 (배포용)
│   ├── yolo_face_detector.onnx
│   └── deepfake_classifier.onnx
├── deepfake_detector.py      # 통합 파이프라인
├── resume_training.py        # 학습/재개 메인 스크립트
├── prepare_data.py          # 데이터 준비 스크립트
├── demo.py                  # 종합 데모 스크립트
├── example_usage.py         # 사용 예제
├── requirements.txt         # 패키지 의존성
└── README.md
```

## 🛠️ 설치

```bash
# 의존성 설치
pip install -r requirements.txt
```

## 📊 데이터 준비

### 1. 라벨 형식
각 이미지에 대응하는 `.txt` 파일:
```
RECT,138,167,187,219,fake
RECT,348,249,376,286,real
RECT,100,100,200,200,none   # none은 real로 처리
```
- `x1,y1`: 박스 왼쪽 위 좌표
- `x2,y2`: 박스 오른쪽 아래 좌표
- `label`: `fake`, `real`, `none`

### 2. 데이터셋 자동 준비
**데이터를 `input/` 폴더에 넣은 후:**
```bash
python prepare_data.py --image_dir input/images --label_dir input/labels
```

**자동 생성되는 구조:**
```
data/
├── face_detection/          # YOLO용
│   ├── images/
│   ├── labels/
│   └── dataset.yaml
├── deepfake_classification/ # EfficientNet용
│   ├── train/real/, train/fake/
│   ├── val/real/, val/fake/
│   └── test/real/, test/fake/
└── extracted_faces/         # 원본 추출 얼굴
```

## 🎯 모델 학습

### 빠른 시작
```bash
# 1. 데이터 준비 (input/ 폴더에 이미지와 라벨 넣은 후)
python prepare_data.py --image_dir input/images --label_dir input/labels

# 2. 통합 학습
python resume_training.py config --mode both
```

### 개별 모델 학습
```bash
# YOLO 얼굴 탐지기만
python resume_training.py yolo --data_path data/face_detection

# EfficientNet 분류기만  
python resume_training.py classifier --data_dir data/deepfake_classification

# 설정 파일로 커스텀 학습
python resume_training.py config --config config/train_config.yaml
```

### 학습 재개 (YOLO & EfficientNet 모두 지원)
```bash
# 🔄 YOLO 모델 재개 (자동으로 최신 체크포인트 찾기)
python resume_training.py yolo --data_path data/face_detection

# 🔄 EfficientNet 모델 재개 (자동으로 최신 체크포인트 찾기)  
python resume_training.py classifier --data_dir data/deepfake_classification

# 🔄 특정 체크포인트에서 YOLO 재개
python resume_training.py yolo \
    --data_path data/face_detection \
    --yolo_checkpoint runs/face_detection/face_detector/weights/last.pt

# 🔄 특정 체크포인트에서 EfficientNet 재개
python resume_training.py classifier \
    --data_dir data/deepfake_classification \
    --classifier_checkpoint runs/deepfake_classifier/checkpoint_epoch_30.pth

# 🔄 설정 파일로 통합 재개 (둘 다 자동 재개)
python resume_training.py config --mode both

# 📋 사용 가능한 체크포인트 보기
python resume_training.py list
```

**체크포인트 자동 저장:**
- **YOLO**: 10 에폭마다 + 최고 성능 모델 자동 저장
- **EfficientNet**: 10 에폭마다 + 최고 성능 모델 자동 저장  
- **재개 시**: 옵티마이저, 스케줄러 상태도 함께 복원

## 🔍 모델 사용

### 기본 사용법
```python
from deepfake_detector import DeepfakeDetectionPipeline

# 파이프라인 초기화 (기본 모델 사용)
detector = DeepfakeDetectionPipeline()

# 이미지 분석
results = detector.detect_deepfake_from_image("test_image.jpg")
print(f"결과: {results['overall_result']}")
print(f"신뢰도: {results['confidence']:.3f}")

# 비디오 분석
results = detector.detect_deepfake_from_video("test_video.mp4")
```

### 학습된 모델 사용
```python
# 커스텀 학습된 모델로 초기화
detector = DeepfakeDetectionPipeline(
    yolo_model_path="runs/face_detection/face_detector/weights/best.pt",
    efficientnet_model='efficientnet-b0',
    classifier_weights_path="runs/deepfake_classifier/best_model.pth",
    confidence_threshold=0.7
)

# 또는 사전 훈련된 모델 사용 (YOLO만 해당, EfficientNet은 직접 학습 필요)
detector = DeepfakeDetectionPipeline(
    yolo_model_path="pretrained_models/yolo/yolov8n-face.pt",
    classifier_weights_path="pretrained_models/efficientnet/my-efficientnet-b0-deepfake.pth"  # 직접 학습한 모델
)

# 실시간 웹캠 (example_usage.py 참조)
python example_usage.py
```

### 결과 형식
```python
{
    "faces_detected": 2,
    "faces": [
        {
            "face_id": 0,
            "bbox": [100, 150, 200, 250],
            "prediction": "fake", 
            "confidence": 0.89
        },
        {
            "face_id": 1,
            "bbox": [300, 100, 400, 200],
            "prediction": "real",
            "confidence": 0.95
        }
    ],
    "overall_result": "fake",  # 하나라도 fake면 전체 fake
    "confidence": 0.92
}
```

## 🛠️ 체크포인트 관리

```bash
# 체크포인트 목록 보기
python utils/checkpoint_manager.py list --save_dir runs

# 최고 성능 모델 찾기  
python utils/checkpoint_manager.py find \
    --save_dir runs --model_type classifier --find_type best

# 오래된 체크포인트 정리 (최신 5개만 보존)
python utils/checkpoint_manager.py clean \
    --model_type classifier --keep_count 5
```

## 📋 설정 파일

`config/train_config.yaml`에서 모든 학습 파라미터 조정 가능:

```yaml
# YOLO 설정
yolo:
  model_size: 'n'  # n, s, m, l, x
  epochs: 100
  batch_size: 16
  
# EfficientNet 설정  
deepfake_classifier:
  model_name: 'efficientnet-b0'  # b0~b7
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

## 📈 데이터 전처리 도구

```bash
# 개별 도구 사용
python utils/label_converter.py yolo \
    --image_dir images --label_dir labels --output_dir yolo_data

python utils/label_converter.py classification \
    --image_dir images --label_dir labels --output_dir faces

# 데이터셋 통계
python utils/data_utils.py validate --source_dir data/deepfake_classification
```

## 🎥 실제 사용 예제

```python
# 1. 이미지 배치 처리
import glob
from deepfake_detector import DeepfakeDetectionPipeline

detector = DeepfakeDetectionPipeline(
    classifier_weights_path="runs/deepfake_classifier/best_model.pth"
)

for img_path in glob.glob("test_images/*.jpg"):
    results = detector.detect_deepfake_from_image(img_path)
    print(f"{img_path}: {results['overall_result']} ({results['confidence']:.3f})")

# 2. 실시간 웹캠 (간단 버전)
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector.detect_deepfake_from_array(frame)
    vis_frame = detector.visualize_results(frame, results)
    
    cv2.imshow('Deepfake Detection', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🔧 고급 사용법

### 모델 성능 튜닝
```bash
# 더 큰 모델로 높은 정확도
python resume_training.py config --config config/train_config.yaml
# train_config.yaml에서 model_size: 'm', efficientnet-b3 등으로 설정

# 더 많은 에폭으로 학습
python resume_training.py classifier \
    --data_dir data/deepfake_classification --epochs 100
```

### 배치 추론
```python
# 여러 이미지 한번에 처리
face_images = [...]  # 추출된 얼굴 이미지 리스트
predictions, confidences = detector.deepfake_classifier.predict_batch(face_images)
```

## 📝 주의사항

- **GPU 권장**: CUDA 사용 시 학습 속도 대폭 향상
- **데이터 품질**: 고품질 라벨링이 모델 성능에 직결
- **모델 크기**: EfficientNet-B0~B7, YOLO n~x 중 컴퓨팅 자원에 맞게 선택
- **체크포인트**: 정기적으로 체크포인트 정리 권장

## 🆘 문제 해결

```bash
# 1. 체크포인트 확인
python resume_training.py list

# 2. 데이터셋 검증
python utils/data_utils.py validate --source_dir data

# 3. GPU 메모리 부족시 배치 크기 줄이기
# config/train_config.yaml에서 batch_size 값 감소
```

## ⚡ ONNX 변환 (성능 최적화)

학습된 모델을 ONNX로 변환하여 추론 속도를 2-5배 향상시킬 수 있습니다.

### 자동 변환
```bash
# 최고 성능 모델들 자동 변환
python utils/model_converter.py best --output_dir onnx_models --dynamic_batch
```

### 개별 변환
```bash
# YOLO 모델
python utils/model_converter.py yolo \
    --input runs/face_detection/best.pt \
    --output onnx_models/yolo_face.onnx

# 분류기 모델
python utils/model_converter.py classifier \
    --input runs/classifier/best_model.pth \
    --output onnx_models/classifier.onnx
```

### 성능 최적화 옵션
```bash
# FP16 정밀도 (메모리 절약, 속도 향상)
python utils/model_converter.py best --half_precision --output_dir onnx_models_fp16

# 동적 배치 크기 지원
python utils/model_converter.py best --dynamic_batch --output_dir onnx_models
```

### 성능 벤치마크
```bash
python utils/model_converter.py benchmark \
    --pytorch_model runs/classifier/best_model.pth \
    --onnx_model onnx_models/classifier.onnx
```

**ONNX 모델 장점:**
- 추론 속도 2-5배 향상
- 메모리 사용량 20-40% 감소
- CPU 환경에서 특히 큰 성능 향상
- 크로스플랫폼 호환성

## 🎮 종합 데모

### 다양한 데모 모드
```bash
# 단일 이미지 분석
python demo.py image --input test_image.jpg

# 배치 이미지 처리
python demo.py batch --input test_images/ --output results/

# 비디오 분석
python demo.py video --input test_video.mp4 --frame_interval 15

# 실시간 웹캠
python demo.py webcam

# 커스텀 모델 사용
python demo.py image --input test.jpg \
    --yolo_model runs/face_detection/best.pt \
    --classifier_model runs/classifier/best_model.pth \
    --confidence 0.7
```

## 📊 성능 지표

### YOLO 얼굴 탐지
- **mAP50**: 얼굴 탐지 정확도
- **Precision**: 정밀도
- **Recall**: 재현율
- **FPS**: 초당 처리 프레임

### EfficientNet 분류
- **Accuracy**: 전체 정확도
- **Precision/Recall**: 클래스별 성능
- **F1-Score**: 균형 지표
- **AUC**: ROC 곡선 아래 면적

## 🚨 주의사항

1. **모델 성능**: 학습 데이터의 품질과 다양성이 중요합니다
2. **처리 속도**: GPU 사용 시 최적 성능을 발휘합니다
3. **메모리 사용**: 대용량 비디오 처리 시 배치 크기 조정이 필요합니다
4. **업데이트**: 새로운 딥페이크 기법에 대응하여 주기적 재학습이 필요합니다

## 📋 시스템 요구사항

### 최소 요구사항
- Python 3.8+
- 8GB RAM
- CPU: 4코어 이상

### 권장 요구사항
- Python 3.9+
- 16GB+ RAM
- GPU: NVIDIA RTX 시리즈 (8GB+ VRAM)
- SSD 저장소

## 🔗 참고 자료

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [ONNX Runtime](https://onnxruntime.ai/)
- [딥페이크 탐지 연구](https://arxiv.org/abs/1901.08971)

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능
