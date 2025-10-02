"""
학습 재개를 위한 간편 스크립트
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from utils.checkpoint_manager import CheckpointManager
from training.train_yolo_face import train_yolo_face_detector, create_dataset_yaml
from training.train_deepfake_classifier import train_deepfake_classifier
import yaml

def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 필수 키 검증
        if 'yolo' not in config or 'deepfake_classifier' not in config:
            raise ValueError("설정 파일에 'yolo' 또는 'deepfake_classifier' 섹션이 없습니다.")

        return config
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        raise

def train_yolo_with_config(config: dict, resume_path: str = None):
    """설정 파일을 사용한 YOLO 학습"""
    yolo_config = config['yolo']
    
    # 데이터셋 YAML 파일 생성
    data_path = yolo_config['data_path']
    data_yaml = os.path.join(data_path, 'dataset.yaml')
    
    if not os.path.exists(data_yaml):
        create_dataset_yaml(data_path, data_yaml)
    
    # 학습 실행
    results, best_model_path = train_yolo_face_detector(
        data_yaml=data_yaml,
        model_size=yolo_config['model_size'],
        epochs=yolo_config['epochs'],
        imgsz=yolo_config['imgsz'],
        batch_size=yolo_config['batch_size'],
        device=yolo_config['device'],
        save_dir=yolo_config['save_dir'],
        pretrained=yolo_config['pretrained'],
        resume=resume_path
    )
    
    return best_model_path

def train_classifier_with_config(config: dict, resume_path: str = None):
    """설정 파일을 사용한 딥페이크 분류기 학습"""
    classifier_config = config['deepfake_classifier']
    
    # 학습 실행
    model, best_model_path, history = train_deepfake_classifier(
        data_dir=classifier_config['data_path'],
        model_name=classifier_config['model_name'],
        epochs=classifier_config['epochs'],
        batch_size=classifier_config['batch_size'],
        learning_rate=classifier_config['learning_rate'],
        weight_decay=classifier_config['weight_decay'],
        save_dir=classifier_config['save_dir'],
        augment=classifier_config['augment'],
        device=classifier_config['device'],
        resume=resume_path
    )
    
    return best_model_path

def resume_yolo_training(
    data_path: str,
    checkpoint_path: str = None,
    save_dir: str = 'runs/face_detection',
    epochs: int = None
):
    """YOLO 학습 재개 또는 처음부터 학습"""

    print("=== YOLO 얼굴 탐지 모델 학습 ===")

    # 체크포인트 자동 찾기
    if not checkpoint_path:
        manager = CheckpointManager(save_dir)
        checkpoint_path = manager.find_latest_checkpoint('yolo')

        if checkpoint_path:
            print(f"✅ 체크포인트 발견: {checkpoint_path}")
            print("   이어서 학습합니다.")
        else:
            print("⚠️  체크포인트가 없습니다.")
            print("   처음부터 학습을 시작합니다.")

    # 데이터셋 YAML 확인
    data_yaml = os.path.join(data_path, 'dataset.yaml')
    if not os.path.exists(data_yaml):
        create_dataset_yaml(data_path, data_yaml)

    # 에폭 수 설정
    if epochs is None:
        epochs = 100  # 기본값

    # 학습 시작 (재개 또는 새로 시작)
    results, best_model_path = train_yolo_face_detector(
        data_yaml=data_yaml,
        resume=checkpoint_path,
        epochs=epochs,
        save_dir=save_dir
    )

    print(f"✅ YOLO 학습 완료! 최고 성능 모델: {best_model_path}")
    return best_model_path

def resume_classifier_training(
    data_dir: str,
    checkpoint_path: str = None,
    save_dir: str = 'runs/deepfake_classifier',
    epochs: int = None
):
    """분류기 학습 재개 또는 처음부터 학습"""

    print("=== EfficientNet 딥페이크 분류기 학습 ===")

    # 체크포인트 자동 찾기
    if not checkpoint_path:
        manager = CheckpointManager(save_dir)
        checkpoint_path = manager.find_latest_checkpoint('classifier')

        if checkpoint_path:
            print(f"✅ 체크포인트 발견: {checkpoint_path}")
            print("   이어서 학습합니다.")
        else:
            print("⚠️  체크포인트가 없습니다.")
            print("   처음부터 학습을 시작합니다.")

    # 에폭 수 설정
    if epochs is None:
        epochs = 50  # 기본값

    # 학습 시작 (재개 또는 새로 시작)
    model, best_model_path, history = train_deepfake_classifier(
        data_dir=data_dir,
        resume=checkpoint_path,
        epochs=epochs,
        save_dir=save_dir
    )

    print(f"✅ 분류기 학습 완료! 최고 성능 모델: {best_model_path}")
    return best_model_path

def resume_with_config(
    config_path: str,
    mode: str = 'both',
    checkpoint_yolo: str = None,
    checkpoint_classifier: str = None
):
    """설정 파일을 사용한 학습 재개"""
    
    print("=== 설정 파일을 사용한 학습 재개 ===")
    
    # 설정 로드
    config = load_config(config_path)
    
    # 체크포인트 자동 찾기
    if mode in ['yolo', 'both'] and not checkpoint_yolo:
        yolo_save_dir = config['yolo']['save_dir']
        manager = CheckpointManager(yolo_save_dir)
        checkpoint_yolo = manager.find_latest_checkpoint('yolo')
        
        if checkpoint_yolo:
            print(f"YOLO 체크포인트 자동 발견: {checkpoint_yolo}")
        else:
            print("YOLO 체크포인트를 찾을 수 없습니다.")
    
    if mode in ['classifier', 'both'] and not checkpoint_classifier:
        classifier_save_dir = config['deepfake_classifier']['save_dir']
        manager = CheckpointManager(classifier_save_dir)
        checkpoint_classifier = manager.find_latest_checkpoint('classifier')
        
        if checkpoint_classifier:
            print(f"분류기 체크포인트 자동 발견: {checkpoint_classifier}")
        else:
            print("분류기 체크포인트를 찾을 수 없습니다.")
    
    # 학습 재개
    results = {}
    errors = {}

    if mode in ['yolo', 'both']:
        try:
            if checkpoint_yolo:
                print("\n▶ YOLO 학습 재개 시작...")
                print(f"   체크포인트: {checkpoint_yolo}")
            else:
                print("\n▶ YOLO 처음부터 학습 시작...")
                print("   체크포인트가 없어 새로 학습합니다.")

            yolo_model_path = train_yolo_with_config(config, checkpoint_yolo)
            results['yolo'] = yolo_model_path
            print(f"✅ YOLO 학습 완료: {yolo_model_path}")
        except Exception as e:
            error_msg = f"YOLO 학습 실패: {e}"
            print(f"❌ {error_msg}")
            errors['yolo'] = str(e)

    if mode in ['classifier', 'both']:
        try:
            if checkpoint_classifier:
                print("\n▶ 분류기 학습 재개 시작...")
                print(f"   체크포인트: {checkpoint_classifier}")
            else:
                print("\n▶ 분류기 처음부터 학습 시작...")
                print("   체크포인트가 없어 새로 학습합니다.")

            classifier_model_path = train_classifier_with_config(config, checkpoint_classifier)
            results['classifier'] = classifier_model_path
            print(f"✅ 분류기 학습 완료: {classifier_model_path}")
        except Exception as e:
            error_msg = f"분류기 학습 실패: {e}"
            print(f"❌ {error_msg}")
            errors['classifier'] = str(e)

    # 결과 요약
    print("\n=== 학습 재개 결과 ===")
    if results:
        print("\n✅ 성공:")
        for model_type, model_path in results.items():
            print(f"  {model_type}: {model_path}")

    if errors:
        print("\n❌ 실패:")
        for model_type, error in errors.items():
            print(f"  {model_type}: {error}")
        # 하나라도 실패하면 경고 메시지
        print("\n⚠️  일부 모델 학습이 실패했습니다. 위 에러 메시지를 확인하세요.")

    if not results and not errors:
        print("⚠️  재개할 체크포인트가 없습니다.")

    return results

def list_available_checkpoints(save_dir: str = 'runs'):
    """사용 가능한 체크포인트 목록 표시"""
    
    print("=== 사용 가능한 체크포인트 ===")
    
    manager = CheckpointManager(save_dir)
    
    # YOLO 체크포인트
    yolo_checkpoints = manager.list_checkpoints('yolo')
    if yolo_checkpoints:
        print(f"\nYOLO 체크포인트 ({len(yolo_checkpoints)}개):")
        for i, checkpoint in enumerate(yolo_checkpoints[:5], 1):  # 최신 5개만 표시
            marker = " ⭐" if checkpoint.get('is_best', False) else ""
            print(f"  {i}. {checkpoint['name']} ({checkpoint['size_mb']}MB) - {checkpoint['modified_time'].strftime('%Y-%m-%d %H:%M')}{marker}")
    else:
        print("\nYOLO 체크포인트가 없습니다.")
    
    # 분류기 체크포인트
    classifier_checkpoints = manager.list_checkpoints('classifier')
    if classifier_checkpoints:
        print(f"\n분류기 체크포인트 ({len(classifier_checkpoints)}개):")
        for i, checkpoint in enumerate(classifier_checkpoints[:5], 1):  # 최신 5개만 표시
            marker = " ⭐" if checkpoint.get('is_best', False) else ""
            epoch_info = f"에폭 {checkpoint.get('epoch', '?')}" if checkpoint.get('epoch') else ""
            acc_info = f"정확도 {checkpoint.get('val_acc', '?'):.4f}" if isinstance(checkpoint.get('val_acc'), (int, float)) else ""
            info = f" ({epoch_info}, {acc_info})" if epoch_info or acc_info else ""
            print(f"  {i}. {checkpoint['name']} ({checkpoint['size_mb']}MB){info} - {checkpoint['modified_time'].strftime('%Y-%m-%d %H:%M')}{marker}")
    else:
        print("\n분류기 체크포인트가 없습니다.")

def main():
    parser = argparse.ArgumentParser(description='딥페이크 탐지 모델 학습 재개')
    parser.add_argument('command', choices=['yolo', 'classifier', 'config', 'list'],
                       help='재개할 학습 타입')
    
    # 공통 인수
    parser.add_argument('--save_dir', type=str, default='runs',
                       help='체크포인트 저장 디렉토리')
    parser.add_argument('--epochs', type=int,
                       help='추가 학습할 에폭 수')
    
    # YOLO 관련
    parser.add_argument('--data_path', type=str,
                       help='YOLO 데이터셋 경로')
    parser.add_argument('--yolo_checkpoint', type=str,
                       help='YOLO 체크포인트 경로')
    
    # 분류기 관련
    parser.add_argument('--data_dir', type=str,
                       help='분류기 데이터셋 경로')
    parser.add_argument('--classifier_checkpoint', type=str,
                       help='분류기 체크포인트 경로')
    
    # 설정 파일 관련
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--mode', type=str, choices=['yolo', 'classifier', 'both'],
                       default='both', help='학습 모드')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_available_checkpoints(args.save_dir)
    
    elif args.command == 'yolo':
        if not args.data_path:
            print("YOLO 데이터셋 경로를 지정해주세요 (--data_path)")
            return
        
        resume_yolo_training(
            data_path=args.data_path,
            checkpoint_path=args.yolo_checkpoint,
            save_dir=args.save_dir,
            epochs=args.epochs
        )
    
    elif args.command == 'classifier':
        if not args.data_dir:
            print("분류기 데이터셋 경로를 지정해주세요 (--data_dir)")
            return
        
        resume_classifier_training(
            data_dir=args.data_dir,
            checkpoint_path=args.classifier_checkpoint,
            save_dir=args.save_dir,
            epochs=args.epochs
        )
    
    elif args.command == 'config':
        resume_with_config(
            config_path=args.config,
            mode=args.mode,
            checkpoint_yolo=args.yolo_checkpoint,
            checkpoint_classifier=args.classifier_checkpoint
        )

if __name__ == "__main__":
    main()

"""
사용 예제:

1. 사용 가능한 체크포인트 목록 보기:
python resume_training.py list

2. YOLO 학습 재개 (자동으로 최신 체크포인트 찾기):
python resume_training.py yolo --data_path data/face_detection

3. 특정 YOLO 체크포인트에서 재개:
python resume_training.py yolo --data_path data/face_detection --yolo_checkpoint runs/face_detection/face_detector/weights/last.pt

4. 분류기 학습 재개:
python resume_training.py classifier --data_dir data/deepfake_classification

5. 설정 파일을 사용한 통합 재개:
python resume_training.py config --config config/train_config.yaml --mode both

6. 더 많은 에폭으로 학습 연장:
python resume_training.py classifier --data_dir data/deepfake_classification --epochs 100
"""