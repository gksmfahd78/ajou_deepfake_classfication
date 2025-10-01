import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional

class YOLOFaceDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        YOLO 기반 얼굴 탐지 모델
        
        Args:
            model_path: 사전 훈련된 YOLO 모델 경로 (None이면 기본 YOLOv8n 사용)
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            # YOLOv8n을 사용하여 얼굴 탐지
            self.model = YOLO('yolov8n.pt')
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        이미지에서 얼굴을 탐지
        
        Args:
            image: 입력 이미지 (numpy array)
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            얼굴 영역 좌표 리스트 [(x1, y1, x2, y2), ...]
        """
        results = self.model(image)
        faces = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf.item()
                    if confidence >= confidence_threshold:
                        # YOLO는 일반적으로 person 클래스를 탐지하므로,
                        # 얼굴 전용 모델이나 커스텀 훈련이 필요할 수 있음
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        faces.append((x1, y1, x2, y2))
        
        return faces
    
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
        for x1, y1, x2, y2 in faces:
            face_img = image[y1:y2, x1:x2]
            if face_img.size > 0:
                face_images.append(face_img)
        return face_images