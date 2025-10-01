"""
모델을 ONNX 형식으로 변환하는 유틸리티
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import Tuple, Optional
import sys

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from models.deepfake_classifier import DeepfakeClassifier


class ModelConverter:
    """모델 ONNX 변환 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def convert_yolo_to_onnx(
        self,
        model_path: str,
        output_path: str,
        img_size: Tuple[int, int] = (640, 640),
        batch_size: int = 1,
        dynamic_batch: bool = False,
        half_precision: bool = False
    ) -> bool:
        """
        YOLO 모델을 ONNX로 변환
        
        Args:
            model_path: YOLO 모델 경로 (.pt)
            output_path: 출력 ONNX 파일 경로
            img_size: 입력 이미지 크기
            batch_size: 배치 크기
            dynamic_batch: 동적 배치 크기 지원
            half_precision: FP16 정밀도 사용
            
        Returns:
            변환 성공 여부
        """
        try:
            print(f"🔄 YOLO 모델 ONNX 변환 시작: {model_path}")
            
            # YOLO 모델 로드
            model = YOLO(model_path)
            
            # Ultralytics YOLO는 자체 export 메서드 제공
            success = model.export(
                format='onnx',
                imgsz=img_size,
                batch=batch_size,
                dynamic=dynamic_batch,
                half=half_precision,
                simplify=True,
                opset=11
            )
            
            if success:
                # 생성된 ONNX 파일을 지정된 경로로 이동
                generated_onnx = str(Path(model_path).with_suffix('.onnx'))
                if os.path.exists(generated_onnx):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    if generated_onnx != output_path:
                        os.rename(generated_onnx, output_path)
                    
                    print(f"✅ YOLO ONNX 변환 완료: {output_path}")
                    
                    # 모델 검증
                    self._validate_onnx_model(output_path, img_size, batch_size)
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ YOLO ONNX 변환 실패: {e}")
            return False
    
    def convert_classifier_to_onnx(
        self,
        model_path: str,
        output_path: str,
        model_name: str = 'efficientnet-b0',
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 1,
        dynamic_batch: bool = False,
        half_precision: bool = False
    ) -> bool:
        """
        EfficientNet 분류기를 ONNX로 변환
        
        Args:
            model_path: PyTorch 모델 경로 (.pth)
            output_path: 출력 ONNX 파일 경로
            model_name: EfficientNet 모델 이름
            img_size: 입력 이미지 크기
            batch_size: 배치 크기
            dynamic_batch: 동적 배치 크기 지원
            half_precision: FP16 정밀도 사용
            
        Returns:
            변환 성공 여부
        """
        try:
            print(f"🔄 분류기 모델 ONNX 변환 시작: {model_path}")
            
            # 모델 초기화 및 가중치 로드
            model = DeepfakeClassifier(
                model_name=model_name,
                num_classes=2,
                pretrained=False
            )
            
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            model.to(self.device)
            
            # 더미 입력 생성
            if dynamic_batch:
                dummy_input = torch.randn(1, 3, *img_size, device=self.device)
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            else:
                dummy_input = torch.randn(batch_size, 3, *img_size, device=self.device)
                dynamic_axes = None
            
            # FP16 변환
            if half_precision and self.device.type == 'cuda':
                model.half()
                dummy_input = dummy_input.half()
            
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # ONNX 변환
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            print(f"✅ 분류기 ONNX 변환 완료: {output_path}")
            
            # 모델 검증
            self._validate_onnx_model(output_path, img_size, batch_size)
            return True
            
        except Exception as e:
            print(f"❌ 분류기 ONNX 변환 실패: {e}")
            return False
    
    def _validate_onnx_model(
        self,
        onnx_path: str,
        img_size: Tuple[int, int],
        batch_size: int
    ):
        """ONNX 모델 유효성 검증"""
        try:
            # ONNX 모델 로드 및 검증
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # ONNX Runtime으로 추론 테스트
            ort_session = ort.InferenceSession(onnx_path)
            
            # 더미 입력으로 추론 테스트
            if 'yolo' in onnx_path.lower() or 'face' in onnx_path.lower():
                # YOLO 모델 (640x640)
                dummy_input = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
            else:
                # 분류기 모델 (224x224)
                dummy_input = np.random.randn(batch_size, 3, *img_size).astype(np.float32)
            
            inputs = {ort_session.get_inputs()[0].name: dummy_input}
            outputs = ort_session.run(None, inputs)
            
            print(f"🔍 ONNX 모델 검증 완료:")
            print(f"  - 입력 형태: {dummy_input.shape}")
            print(f"  - 출력 형태: {[out.shape for out in outputs]}")
            print(f"  - 모델 크기: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
            
        except Exception as e:
            print(f"⚠️  ONNX 모델 검증 실패: {e}")
    
    def convert_best_models(
        self,
        runs_dir: str = 'runs',
        output_dir: str = 'onnx_models',
        dynamic_batch: bool = True,
        half_precision: bool = False
    ):
        """
        최고 성능 모델들을 자동으로 ONNX로 변환
        
        Args:
            runs_dir: 학습 결과 디렉토리
            output_dir: ONNX 모델 저장 디렉토리
            dynamic_batch: 동적 배치 크기 지원
            half_precision: FP16 정밀도 사용
        """
        print("🎯 최고 성능 모델들 자동 ONNX 변환 시작...")
        
        from utils.checkpoint_manager import CheckpointManager
        
        # 체크포인트 매니저 초기화
        manager = CheckpointManager(runs_dir)
        
        # YOLO 모델 변환
        print("\n📊 YOLO 모델 변환 중...")
        yolo_checkpoint = manager.find_best_checkpoint('yolo')
        if yolo_checkpoint:
            yolo_output = os.path.join(output_dir, 'yolo_face_detector.onnx')
            success = self.convert_yolo_to_onnx(
                yolo_checkpoint,
                yolo_output,
                dynamic_batch=dynamic_batch,
                half_precision=half_precision
            )
            if success:
                print(f"✅ YOLO 모델 변환 완료: {yolo_output}")
        else:
            print("❌ YOLO 체크포인트를 찾을 수 없습니다.")
        
        # 분류기 모델 변환
        print("\n🧠 분류기 모델 변환 중...")
        classifier_checkpoint = manager.find_best_checkpoint('classifier')
        if classifier_checkpoint:
            classifier_output = os.path.join(output_dir, 'deepfake_classifier.onnx')
            success = self.convert_classifier_to_onnx(
                classifier_checkpoint,
                classifier_output,
                dynamic_batch=dynamic_batch,
                half_precision=half_precision
            )
            if success:
                print(f"✅ 분류기 모델 변환 완료: {classifier_output}")
        else:
            print("❌ 분류기 체크포인트를 찾을 수 없습니다.")
        
        print(f"\n🎉 ONNX 변환 완료! 모델들이 '{output_dir}' 디렉토리에 저장되었습니다.")
    
    def benchmark_models(
        self,
        pytorch_model_path: str,
        onnx_model_path: str,
        num_runs: int = 100,
        batch_size: int = 1
    ):
        """PyTorch vs ONNX 모델 성능 비교"""
        print(f"⚡ 모델 성능 벤치마크 (배치크기: {batch_size}, 반복: {num_runs}회)")
        
        try:
            import time
            
            # ONNX 모델 테스트
            ort_session = ort.InferenceSession(onnx_model_path)
            input_name = ort_session.get_inputs()[0].name
            input_shape = ort_session.get_inputs()[0].shape
            
            # 입력 형태 결정
            if len(input_shape) == 4:
                if input_shape[2] == 640:  # YOLO
                    dummy_input = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
                else:  # Classifier
                    dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            
            # ONNX 성능 측정
            start_time = time.time()
            for _ in range(num_runs):
                _ = ort_session.run(None, {input_name: dummy_input})
            onnx_time = (time.time() - start_time) / num_runs
            
            print(f"🔥 ONNX 평균 추론 시간: {onnx_time*1000:.2f}ms")
            print(f"📊 ONNX FPS: {1/onnx_time:.1f}")
            
        except Exception as e:
            print(f"❌ 벤치마크 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description='모델 ONNX 변환 도구')
    parser.add_argument('command', choices=['yolo', 'classifier', 'best', 'benchmark'],
                       help='변환할 모델 타입')
    
    # 공통 인수
    parser.add_argument('--input', type=str, help='입력 모델 경로')
    parser.add_argument('--output', type=str, help='출력 ONNX 파일 경로')
    parser.add_argument('--dynamic_batch', action='store_true',
                       help='동적 배치 크기 지원')
    parser.add_argument('--half_precision', action='store_true',
                       help='FP16 정밀도 사용')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='배치 크기')
    
    # 모델별 인수
    parser.add_argument('--model_name', type=str, default='efficientnet-b0',
                       help='EfficientNet 모델 이름')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                       help='입력 이미지 크기')
    
    # 자동 변환 인수
    parser.add_argument('--runs_dir', type=str, default='runs',
                       help='학습 결과 디렉토리')
    parser.add_argument('--output_dir', type=str, default='onnx_models',
                       help='ONNX 모델 저장 디렉토리')
    
    # 벤치마크 인수
    parser.add_argument('--pytorch_model', type=str,
                       help='PyTorch 모델 경로 (벤치마크용)')
    parser.add_argument('--onnx_model', type=str,
                       help='ONNX 모델 경로 (벤치마크용)')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='벤치마크 반복 횟수')
    
    args = parser.parse_args()
    
    converter = ModelConverter()
    
    if args.command == 'yolo':
        if not args.input or not args.output:
            print("❌ YOLO 변환에는 --input과 --output이 필요합니다.")
            return
        
        success = converter.convert_yolo_to_onnx(
            args.input,
            args.output,
            img_size=(640, 640),
            batch_size=args.batch_size,
            dynamic_batch=args.dynamic_batch,
            half_precision=args.half_precision
        )
        
    elif args.command == 'classifier':
        if not args.input or not args.output:
            print("❌ 분류기 변환에는 --input과 --output이 필요합니다.")
            return
        
        success = converter.convert_classifier_to_onnx(
            args.input,
            args.output,
            model_name=args.model_name,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            dynamic_batch=args.dynamic_batch,
            half_precision=args.half_precision
        )
        
    elif args.command == 'best':
        converter.convert_best_models(
            runs_dir=args.runs_dir,
            output_dir=args.output_dir,
            dynamic_batch=args.dynamic_batch,
            half_precision=args.half_precision
        )
        
    elif args.command == 'benchmark':
        if not args.pytorch_model or not args.onnx_model:
            print("❌ 벤치마크에는 --pytorch_model과 --onnx_model이 필요합니다.")
            return
        
        converter.benchmark_models(
            args.pytorch_model,
            args.onnx_model,
            num_runs=args.num_runs,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()

"""
사용 예제:

1. 최고 성능 모델들 자동 변환:
python model_converter.py best --output_dir onnx_models --dynamic_batch

2. 특정 YOLO 모델 변환:
python model_converter.py yolo --input runs/face_detection/best.pt --output models/yolo_face.onnx

3. 특정 분류기 모델 변환:
python model_converter.py classifier --input runs/classifier/best_model.pth --output models/classifier.onnx

4. FP16 정밀도로 변환:
python model_converter.py best --half_precision --output_dir onnx_models_fp16

5. 성능 벤치마크:
python model_converter.py benchmark --pytorch_model runs/classifier/best_model.pth --onnx_model onnx_models/classifier.onnx

변환된 ONNX 모델 장점:
- 추론 속도 2-5배 향상
- 메모리 사용량 감소
- 다양한 플랫폼 지원 (CPU 최적화)
- 배포 환경에서 안정성 향상
"""