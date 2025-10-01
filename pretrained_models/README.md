# 📦 사전 훈련된 모델 폴더

여기에 사전 훈련된 모델들을 넣어주세요.

## 📁 폴더 구조

```
pretrained_models/
├── yolo/                    # YOLO 얼굴 탐지 모델들
│   ├── yolov8n-face.pt     # YOLOv8 nano 얼굴 탐지 모델
│   ├── yolov8s-face.pt     # YOLOv8 small 얼굴 탐지 모델
│   └── ...
├── efficientnet/           # EfficientNet 딥페이크 분류 모델들 (직접 학습 필요)
│   ├── my-efficientnet-b0-deepfake.pth  # 내가 학습한 모델
│   ├── my-efficientnet-b3-deepfake.pth  # 내가 학습한 모델
│   └── ...
└── README.md               # 이 파일
```

## 🎯 모델 타입별 설명

### YOLO 얼굴 탐지 모델 (.pt 파일)
- **기본 모델**: YOLOv8n/s/m/l/x.pt (일반 객체 탐지)
- **얼굴 전용**: 얼굴 탐지에 특화된 YOLO 모델
- **위치**: `pretrained_models/yolo/`

### EfficientNet 딥페이크 분류 모델 (.pth 파일)  
- **시작점**: ImageNet 사전훈련 모델 (자동 사용, 마지막 층만 교체)
- **학습 과정**: ImageNet 특징 추출 + 딥페이크 분류층 학습
- **딥페이크 전용**: 학습 완료 후 여기에 저장하여 재사용
- **위치**: `pretrained_models/efficientnet/` (학습된 모델 백업용)

## 🚀 사용 방법

### 1. 파이프라인에서 사용
```python
from deepfake_detector import DeepfakeDetectionPipeline

# 사전 훈련된 모델 사용
detector = DeepfakeDetectionPipeline(
    yolo_model_path="pretrained_models/yolo/yolov8n-face.pt",
    classifier_weights_path="pretrained_models/efficientnet/efficientnet-b0-deepfake.pth"
)
```

### 2. 개별 학습 시 사용
```bash
# YOLO 사전훈련 모델에서 시작
python resume_training.py yolo \
    --data_path data/face_detection \
    --yolo_checkpoint pretrained_models/yolo/yolov8n-face.pt

# EfficientNet 처음 학습 (ImageNet → 딥페이크 전이학습)
python resume_training.py classifier \
    --data_dir data/deepfake_classification
# 내부적으로: ImageNet 사전훈련 + 마지막 층 교체 + 딥페이크 학습

# 학습된 딥페이크 모델에서 이어서 학습 (성능 향상)
python resume_training.py classifier \
    --data_dir data/deepfake_classification \
    --classifier_checkpoint pretrained_models/efficientnet/my-efficientnet-b0-deepfake.pth
```

### 3. 데모에서 사용
```bash
# 사전훈련 모델로 데모 실행
python demo.py webcam \
    --yolo_model pretrained_models/yolo/yolov8n-face.pt \
    --classifier_model pretrained_models/efficientnet/my-efficientnet-b0-deepfake.pth
```

## 📥 모델 다운로드 소스

### YOLO 얼굴 탐지 모델
- **Ultralytics**: https://github.com/ultralytics/ultralytics
- **YOLO-Face**: https://github.com/akanazawa/yolo-face  
- **WiderFace 훈련**: 얼굴 탐지 전용 모델들

### EfficientNet 딥페이크 분류 모델
- **완전 자동**: ImageNet 사전훈련 모델이 첫 실행 시 자동 다운로드 (사용자 다운로드 불필요!)
- **전이학습**: 마지막 분류층만 교체하여 딥페이크 분류 학습
- **백업/재사용**: 학습 완료 후 `runs/deepfake_classifier/best_model.pth`를 여기로 복사
- **연구 모델**: 딥페이크 탐지 논문의 모델들 (희귀)
- **공개 모델**: Kaggle, GitHub 등의 딥페이크 경진대회 모델들

## ⚠️ 주의사항

1. **라이선스 확인**: 모델 사용 전 라이선스 조건 확인
2. **모델 호환성**: PyTorch 버전과 모델 호환성 확인  
3. **메모리 사용량**: 큰 모델(B4~B7)은 더 많은 GPU 메모리 필요
4. **파일 크기**: 대용량 모델 파일 관리 주의

## 🔄 없어도 되는 경우

**기본 모델 자동 사용:**
- **YOLO**: `yolov8n.pt` 자동 다운로드 (첫 실행 시) - 일반 객체 탐지용
- **EfficientNet**: ImageNet 사전훈련 모델 자동 사용 - 딥페이크 학습 필요

따라서 사전훈련 모델이 없어도 시스템이 정상 작동합니다!

## 📝 권장 워크플로우

1. **처음 시작**: 사전훈련 모델 없이 시작
   ```bash
   python prepare_data.py --image_dir input/images --label_dir input/labels
   python resume_training.py config --mode both
   ```

2. **모델 저장**: 학습 완료 후 좋은 모델을 여기에 백업
   ```bash
   # 학습된 모델을 pretrained_models로 복사
   cp runs/face_detection/face_detector/weights/best.pt pretrained_models/yolo/my-yolo-face.pt
   cp runs/deepfake_classifier/best_model.pth pretrained_models/efficientnet/my-efficientnet-b0.pth
   ```

3. **재사용**: 다음 학습 시 백업된 모델에서 시작
   ```bash
   python resume_training.py yolo --yolo_checkpoint pretrained_models/yolo/my-yolo-face.pt
   ```