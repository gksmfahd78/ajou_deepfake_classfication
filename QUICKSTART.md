# 🚀 빠른 시작 가이드

## 📋 사전 준비

### 1. 환경 설정
```bash
# 1. 프로젝트 클론/다운로드
cd deepfake_detection

# 2. 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비 (필수)
```
input/
├── images/           # 🖼️ 이미지 파일들을 여기에!
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── labels/           # 📝 라벨 파일들을 여기에!
    ├── img001.txt    # RECT,x1,y1,x2,y2,label
    ├── img002.txt
    └── ...
```

**라벨 파일 예시 (`input/labels/img001.txt`):**
```
RECT,138,167,187,219,fake
RECT,348,249,376,286,real
```

### 3. 사전 훈련된 모델 (선택사항)
**더 좋은 성능을 원한다면 `pretrained_models/` 폴더에 넣어주세요:**
```
pretrained_models/
├── yolo/                    # 📦 YOLO 얼굴 탐지 모델들
│   └── yolov8n-face.pt     # 얼굴 전용 YOLO 모델 (다운로드 가능)
└── efficientnet/           # 📦 딥페이크 분류 모델들  
    └── my-efficientnet-b0-deepfake.pth  # 직접 학습한 모델 (학습 후 복사)
```

**⚠️ 없어도 됩니다!** 기본 모델이 자동으로 다운로드됩니다.

**📝 중요 사항:**
- **YOLO**: 얼굴 전용 모델 다운로드 가능 (성능 향상)
- **EfficientNet**: 딥페이크 전용 모델은 직접 학습해야 함 (필수)

## ⚡ 3단계 실행

### 1단계: 데이터 준비 (2분)
```bash
python prepare_data.py --image_dir input/images --label_dir input/labels
```

**결과:** 
- `data/face_detection/` - YOLO용 데이터
- `data/deepfake_classification/` - EfficientNet용 데이터

### 2단계: 모델 학습 (30-60분)
```bash
# 빠른 테스트용 (작은 에폭)
python resume_training.py config --mode both

# 또는 개별 학습
python resume_training.py yolo --data_path data/face_detection
python resume_training.py classifier --data_dir data/deepfake_classification
```

### 3단계: 추론 테스트 (즉시)
```bash
# 단일 이미지 테스트
python demo.py image --input test_image.jpg

# 실시간 웹캠 데모
python demo.py webcam
```

## 🎯 즉시 사용 가능한 데모

### ⚠️ 중요: 기본 모델의 한계
```python
from deepfake_detector import DeepfakeDetectionPipeline

# ❌ 기본 모델 (정확도 매우 낮음 - 권장하지 않음)
detector = DeepfakeDetectionPipeline()

# 이미지 분석
results = detector.detect_deepfake_from_image("your_image.jpg")
print(f"결과: {results['overall_result']}")
print(f"신뢰도: {results['confidence']:.3f}")
```

**⚠️ 경고:**
- 기본 YOLOv8n은 일반 객체 탐지 모델이므로 얼굴 탐지 정확도가 낮습니다
- EfficientNet은 학습 없이 무작위 예측만 수행합니다
- **실제 사용을 위해서는 반드시 모델을 학습하세요!**

### ✅ 직접 학습한 모델 사용 (권장)
```python
# 🎯 커스텀 학습된 모델 사용 (권장!)
detector = DeepfakeDetectionPipeline(
    yolo_model_path="runs/face_detection/face_detector/weights/best.pt",
    classifier_weights_path="runs/deepfake_classifier/best_model.pth",
    confidence_threshold=0.7
)

results = detector.detect_deepfake_from_image("test.jpg")
print(f"✅ 결과: {results['overall_result']}")
print(f"📊 신뢰도: {results['confidence']:.3f}")
```

## 📊 다양한 데모 모드

```bash
# 1. 단일 이미지 분석
python demo.py image --input test.jpg

# 2. 폴더 내 모든 이미지 배치 처리
python demo.py batch --input test_images/ --output results/

# 3. 비디오 분석
python demo.py video --input test_video.mp4

# 4. 실시간 웹캠 (종료: q키)
python demo.py webcam

# 5. 고급 설정
python demo.py image --input test.jpg \
    --classifier_model runs/classifier/best_model.pth \
    --confidence 0.8
```

## ⚙️ 설정 커스터마이징

### 모델 크기 조정 (`config/train_config.yaml`)
```yaml
# 빠른 학습 (낮은 정확도)
yolo:
  model_size: 'n'        # nano
  epochs: 50
deepfake_classifier:
  model_name: 'efficientnet-b0'
  epochs: 30

# 높은 정확도 (느린 학습)  
yolo:
  model_size: 'l'        # large
  epochs: 200
deepfake_classifier:
  model_name: 'efficientnet-b4'
  epochs: 100
```

### 데이터 분할 비율 조정
```bash
python prepare_data.py \
    --image_dir images --label_dir labels \
    --train_ratio 0.8 --val_ratio 0.15 --test_ratio 0.05
```

## 🔧 문제 해결

### GPU 메모리 부족
```yaml
# config/train_config.yaml에서 배치 크기 줄이기
yolo:
  batch_size: 8    # 기본값: 16
deepfake_classifier:
  batch_size: 16   # 기본값: 32
```

### 학습 중단 후 재개
```bash
# 자동으로 최신 체크포인트 찾아서 재개
python resume_training.py classifier --data_dir data/deepfake_classification

# 사용 가능한 체크포인트 확인
python resume_training.py list
```

### 데이터셋 검증
```bash
# 데이터셋 통계 확인
python utils/data_utils.py validate --source_dir data/deepfake_classification

# 체크포인트 관리
python utils/checkpoint_manager.py list --save_dir runs
```

## 📈 성능 최적화 팁

### 1. 하드웨어별 권장 설정

**CPU만 사용:**
```yaml
yolo:
  model_size: 'n'
  batch_size: 4
deepfake_classifier:
  model_name: 'efficientnet-b0'  
  batch_size: 8
```

**GPU 8GB:**
```yaml
yolo:
  model_size: 's'
  batch_size: 16
deepfake_classifier:
  model_name: 'efficientnet-b2'
  batch_size: 32
```

**GPU 16GB+:**
```yaml
yolo:
  model_size: 'm'
  batch_size: 32
deepfake_classifier:
  model_name: 'efficientnet-b4'
  batch_size: 64
```

### 2. 데이터 품질 개선
- 해상도: 최소 224x224 이상 권장
- 얼굴 크기: 이미지의 최소 10% 이상
- 라벨 정확도: 잘못된 라벨은 성능 저하 원인

### 3. 학습 전략
```bash
# 1단계: 작은 모델로 빠른 검증
python resume_training.py config # 기본 설정

# 2단계: 큰 모델로 정확도 향상  
# config/train_config.yaml 수정 후
python resume_training.py config

# 3단계: 미세 조정
python resume_training.py classifier --epochs 200
```

## 🎉 완료!

이제 딥페이크 탐지 시스템이 준비되었습니다!

- 📁 **학습된 모델**: `runs/` 폴더
- 📊 **결과 시각화**: `demo.py`로 확인
- ⚙️ **설정 조정**: `config/train_config.yaml`
- 🔄 **학습 재개**: `resume_training.py`

더 자세한 내용은 [README.md](README.md)를 참조하세요.