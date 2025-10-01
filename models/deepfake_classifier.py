import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms

class DeepfakeClassifier(nn.Module):
    def __init__(self, model_name: str = 'efficientnet-b0', num_classes: int = 2, pretrained: bool = True):
        """
        EfficientNet 기반 딥페이크 분류 모델
        
        Args:
            model_name: EfficientNet 모델 이름 (b0~b7)
            num_classes: 클래스 수 (2: real/fake)
            pretrained: 사전 훈련된 가중치 사용 여부
        """
        super(DeepfakeClassifier, self).__init__()
        
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
        
        # 마지막 분류층 교체
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)
        
        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        return self.backbone(x)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        이미지 전처리

        Args:
            image: OpenCV 이미지 (BGR)

        Returns:
            전처리된 텐서
        """
        try:
            # 이미지가 비어있는지 확인
            if image is None or image.size == 0:
                raise ValueError("입력 이미지가 비어있습니다.")

            # BGR to RGB 변환
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                # 그레이스케일을 RGB로 변환
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"지원하지 않는 이미지 형식입니다: {image.shape}")

            # PIL Image로 변환
            pil_image = Image.fromarray(image)

            # 전처리 적용
            tensor = self.transform(pil_image)
            tensor = tensor.unsqueeze(0)  # 배치 차원 추가

            return tensor.to(self.device)
        except Exception as e:
            raise RuntimeError(f"이미지 전처리 중 오류 발생: {e}")
    
    def predict(self, image: np.ndarray, return_probability: bool = True) -> Tuple[int, float]:
        """
        딥페이크 예측
        
        Args:
            image: 입력 얼굴 이미지
            return_probability: 확률값 반환 여부
            
        Returns:
            (예측 클래스, 확률) - 0: real, 1: fake
        """
        self.eval()
        
        with torch.no_grad():
            # 이미지 전처리
            input_tensor = self.preprocess_image(image)
            
            # 예측
            outputs = self.forward(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            if return_probability:
                return predicted_class, confidence
            else:
                return predicted_class, None
    
    def predict_batch(self, images: list) -> Tuple[list, list]:
        """
        배치 단위 예측
        
        Args:
            images: 이미지 리스트
            
        Returns:
            (예측 클래스 리스트, 확률 리스트)
        """
        self.eval()
        
        # 배치 텐서 생성
        batch_tensors = []
        for image in images:
            tensor = self.preprocess_image(image)
            batch_tensors.append(tensor.squeeze(0))
        
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy().tolist()
            confidences = torch.max(probabilities, dim=1)[0].cpu().numpy().tolist()
            
            return predicted_classes, confidences
    
    def load_weights(self, checkpoint_path: str):
        """모델 가중치 로드"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 체크포인트가 딕셔너리인지 확인
            if isinstance(checkpoint, dict):
                # 'model_state_dict' 키가 있는 경우
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'])
                # 'state_dict' 키가 있는 경우
                elif 'state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['state_dict'])
                # 딕셔너리지만 키가 없는 경우 (직접 state_dict인 경우)
                else:
                    self.load_state_dict(checkpoint)
            else:
                # 체크포인트가 직접 state_dict인 경우
                self.load_state_dict(checkpoint)

            print(f"모델 가중치 로드 완료: {checkpoint_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        except KeyError as e:
            raise KeyError(f"체크포인트 형식이 올바르지 않습니다. 누락된 키: {e}")
        except Exception as e:
            raise RuntimeError(f"모델 가중치 로드 중 오류 발생: {e}")
    
    def save_weights(self, checkpoint_path: str, epoch: int = 0, loss: float = 0.0):
        """모델 가중치 저장"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"모델 가중치 저장 완료: {checkpoint_path}")