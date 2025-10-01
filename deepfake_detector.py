import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from models.face_detector import YOLOFaceDetector
from models.deepfake_classifier import DeepfakeClassifier

class DeepfakeDetectionPipeline:
    def __init__(self, 
                 yolo_model_path: Optional[str] = None,
                 efficientnet_model: str = 'efficientnet-b0',
                 classifier_weights_path: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        """
        딥페이크 탐지 파이프라인
        
        Args:
            yolo_model_path: YOLO 모델 경로
            efficientnet_model: EfficientNet 모델 이름
            classifier_weights_path: 훈련된 분류기 가중치 경로
            confidence_threshold: 얼굴 탐지 신뢰도 임계값
        """
        # 얼굴 탐지 모델 초기화
        try:
            self.face_detector = YOLOFaceDetector(yolo_model_path)
        except Exception as e:
            raise RuntimeError(f"얼굴 탐지 모델 초기화 실패: {e}")

        # 딥페이크 분류 모델 초기화
        try:
            self.deepfake_classifier = DeepfakeClassifier(
                model_name=efficientnet_model,
                num_classes=2,
                pretrained=True
            )
        except Exception as e:
            raise RuntimeError(f"딥페이크 분류 모델 초기화 실패: {e}")

        # 훈련된 가중치가 있다면 로드
        if classifier_weights_path:
            if os.path.exists(classifier_weights_path):
                try:
                    self.deepfake_classifier.load_weights(classifier_weights_path)
                except Exception as e:
                    raise RuntimeError(f"분류기 가중치 로드 실패: {e}")
            else:
                raise FileNotFoundError(f"분류기 가중치 파일을 찾을 수 없습니다: {classifier_weights_path}")

        self.confidence_threshold = confidence_threshold
    
    def detect_deepfake_from_image(self, image_path: str) -> Dict:
        """
        이미지에서 딥페이크 탐지

        Args:
            image_path: 입력 이미지 경로

        Returns:
            탐지 결과 딕셔너리
        """
        # 파일 존재 여부 확인
        if not os.path.exists(image_path):
            return {"error": f"이미지 파일을 찾을 수 없습니다: {image_path}"}

        # 이미지 로드
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"이미지를 로드할 수 없습니다: {image_path}"}
        except Exception as e:
            return {"error": f"이미지 로드 중 오류 발생: {e}"}

        return self.detect_deepfake_from_array(image)
    
    def detect_deepfake_from_array(self, image: np.ndarray) -> Dict:
        """
        numpy 배열 이미지에서 딥페이크 탐지
        
        Args:
            image: 입력 이미지 (numpy array)
            
        Returns:
            탐지 결과 딕셔너리
        """
        results = {
            "faces_detected": 0,
            "faces": [],
            "overall_result": "real",
            "confidence": 0.0
        }
        
        # 1단계: 얼굴 탐지
        face_boxes = self.face_detector.detect_faces(image, self.confidence_threshold)
        results["faces_detected"] = len(face_boxes)
        
        if len(face_boxes) == 0:
            results["overall_result"] = "no_face_detected"
            return results
        
        # 2단계: 얼굴 영역 추출
        face_images = self.face_detector.extract_face_regions(image, face_boxes)
        
        # 3단계: 각 얼굴에 대해 딥페이크 분류
        fake_count = 0
        total_confidence = 0.0
        
        for i, (face_img, face_box) in enumerate(zip(face_images, face_boxes)):
            if face_img.size > 0:
                # 딥페이크 예측
                prediction, confidence = self.deepfake_classifier.predict(face_img)
                
                face_result = {
                    "face_id": i,
                    "bbox": face_box,  # (x1, y1, x2, y2)
                    "prediction": "fake" if prediction == 1 else "real",
                    "confidence": confidence
                }
                
                results["faces"].append(face_result)
                
                if prediction == 1:  # fake
                    fake_count += 1
                
                total_confidence += confidence
        
        # 전체 결과 결정
        if len(results["faces"]) > 0:
            avg_confidence = total_confidence / len(results["faces"])
            results["confidence"] = avg_confidence
            
            # 하나라도 fake면 전체를 fake로 판단
            if fake_count > 0:
                results["overall_result"] = "fake"
            else:
                results["overall_result"] = "real"
        
        return results
    
    def detect_deepfake_from_video(self, video_path: str,
                                  frame_interval: int = 30,
                                  max_frames: int = 100) -> Dict:
        """
        비디오에서 딥페이크 탐지

        Args:
            video_path: 비디오 파일 경로
            frame_interval: 프레임 샘플링 간격
            max_frames: 최대 처리 프레임 수

        Returns:
            비디오 분석 결과
        """
        # 파일 존재 여부 확인
        if not os.path.exists(video_path):
            return {"error": f"비디오 파일을 찾을 수 없습니다: {video_path}"}

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"비디오를 열 수 없습니다: {video_path}"}
        except Exception as e:
            return {"error": f"비디오 로드 중 오류 발생: {e}"}
        
        results = {
            "total_frames_analyzed": 0,
            "frames_with_faces": 0,
            "fake_frames": 0,
            "real_frames": 0,
            "overall_result": "real",
            "confidence": 0.0,
            "frame_results": []
        }
        
        frame_count = 0
        analyzed_frames = 0
        total_confidence = 0.0
        fake_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or analyzed_frames >= max_frames:
                break
            
            # 프레임 간격에 따라 샘플링
            if frame_count % frame_interval == 0:
                frame_result = self.detect_deepfake_from_array(frame)
                
                if frame_result["faces_detected"] > 0:
                    results["frames_with_faces"] += 1
                    
                    if frame_result["overall_result"] == "fake":
                        fake_frame_count += 1
                        results["fake_frames"] += 1
                    elif frame_result["overall_result"] == "real":
                        results["real_frames"] += 1
                    
                    total_confidence += frame_result["confidence"]
                
                frame_result["frame_number"] = frame_count
                results["frame_results"].append(frame_result)
                analyzed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        results["total_frames_analyzed"] = analyzed_frames
        
        # 전체 비디오 결과 결정
        if results["frames_with_faces"] > 0:
            results["confidence"] = total_confidence / results["frames_with_faces"]
            
            # fake 프레임 비율로 판단
            fake_ratio = fake_frame_count / results["frames_with_faces"]
            if fake_ratio > 0.3:  # 30% 이상 fake면 전체를 fake로 판단
                results["overall_result"] = "fake"
            else:
                results["overall_result"] = "real"
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        탐지 결과를 이미지에 시각화
        
        Args:
            image: 원본 이미지
            results: 탐지 결과
            
        Returns:
            시각화된 이미지
        """
        vis_image = image.copy()
        
        for face in results.get("faces", []):
            x1, y1, x2, y2 = face["bbox"]
            prediction = face["prediction"]
            confidence = face["confidence"]
            
            # 박스 색상 (real: 초록, fake: 빨강)
            color = (0, 255, 0) if prediction == "real" else (0, 0, 255)
            
            # 얼굴 박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트
            label = f"{prediction}: {confidence:.2f}"
            
            # 텍스트 배경
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(vis_image, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # 텍스트
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_image