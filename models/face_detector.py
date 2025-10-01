import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional
import warnings

class YOLOFaceDetector:
    def __init__(self, model_path: Optional[str] = None, face_class_id: Optional[int] = None):
        """
        YOLO 기반 얼굴 탐지 모델

        Args:
            model_path: 사전 훈련된 YOLO 모델 경로 (None이면 기본 YOLOv8n 사용)
            face_class_id: 얼굴 클래스 ID (기본 YOLOv8n의 경우 0=person)
        """
        try:
            if model_path:
                self.model = YOLO(model_path)
            else:
                # YOLOv8n을 사용하여 얼굴 탐지
                self.model = YOLO('yolov8n.pt')
                warnings.warn(
                    "기본 YOLOv8n 모델은 얼굴 전용 모델이 아닙니다. "
                    "더 나은 성능을 위해 얼굴 탐지 전용 모델을 학습하거나 사용하세요.",
                    UserWarning
                )
        except Exception as e:
            raise RuntimeError(f"YOLO 모델 로드 실패: {e}")

        # 얼굴 클래스 ID (커스텀 모델은 자동 감지, 기본 모델은 person 클래스 사용)
        self.face_class_id = face_class_id if face_class_id is not None else 0
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        이미지에서 얼굴을 탐지

        Args:
            image: 입력 이미지 (numpy array)
            confidence_threshold: 신뢰도 임계값

        Returns:
            얼굴 영역 좌표 리스트 [(x1, y1, x2, y2), ...]
        """
        try:
            results = self.model(image, verbose=False)
            faces = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 클래스 ID 확인 (얼굴 클래스만 필터링)
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()

                        # 지정된 얼굴 클래스 ID와 신뢰도 조건 확인
                        if class_id == self.face_class_id and confidence >= confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                            # 유효한 좌표인지 확인
                            if x2 > x1 and y2 > y1:
                                faces.append((x1, y1, x2, y2))

            return faces
        except Exception as e:
            warnings.warn(f"얼굴 탐지 중 오류 발생: {e}")
            return []
    
    def extract_face_regions(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        탐지된 얼굴 영역을 추출

        Args:
            image: 원본 이미지
            faces: 얼굴 좌표 리스트

        Returns:
            추출된 얼굴 이미지 리스트
        """
        face_images = []
        h, w = image.shape[:2]

        for x1, y1, x2, y2 in faces:
            # 좌표를 이미지 범위 내로 제한
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # 유효한 영역인지 확인
            if x2 > x1 and y2 > y1:
                face_img = image[y1:y2, x1:x2]
                if face_img.size > 0:
                    face_images.append(face_img)
                else:
                    warnings.warn(f"빈 얼굴 영역이 감지되었습니다: ({x1},{y1},{x2},{y2})")

        return face_images