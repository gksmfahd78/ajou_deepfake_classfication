"""
체크포인트 관리 유틸리티
"""

import os
import torch
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import argparse

class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: 체크포인트 저장 디렉토리
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def list_checkpoints(self, model_type: str = None) -> List[Dict]:
        """
        체크포인트 목록 조회
        
        Args:
            model_type: 모델 타입 ('yolo', 'classifier', None for all)
            
        Returns:
            체크포인트 정보 리스트
        """
        checkpoints = []
        
        if model_type == 'yolo' or model_type is None:
            # YOLO 체크포인트 검색
            yolo_patterns = [
                str(self.save_dir / "**" / "*.pt"),
                str(self.save_dir / "**" / "weights" / "*.pt")
            ]
            
            for pattern in yolo_patterns:
                for checkpoint_path in glob.glob(pattern, recursive=True):
                    if self._is_yolo_checkpoint(checkpoint_path):
                        info = self._get_yolo_checkpoint_info(checkpoint_path)
                        if info:
                            checkpoints.append(info)
        
        if model_type == 'classifier' or model_type is None:
            # 분류기 체크포인트 검색
            classifier_pattern = str(self.save_dir / "**" / "*.pth")
            
            for checkpoint_path in glob.glob(classifier_pattern, recursive=True):
                if self._is_classifier_checkpoint(checkpoint_path):
                    info = self._get_classifier_checkpoint_info(checkpoint_path)
                    if info:
                        checkpoints.append(info)
        
        # 수정 시간 기준으로 정렬
        checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return checkpoints
    
    def _is_yolo_checkpoint(self, checkpoint_path: str) -> bool:
        """YOLO 체크포인트인지 확인"""
        path = Path(checkpoint_path)
        
        # 파일 확장자 확인
        if path.suffix != '.pt':
            return False
        
        # YOLO 관련 키워드 확인
        yolo_keywords = ['yolo', 'face_detector', 'detection']
        path_str = str(path).lower()
        
        return any(keyword in path_str for keyword in yolo_keywords)
    
    def _is_classifier_checkpoint(self, checkpoint_path: str) -> bool:
        """분류기 체크포인트인지 확인"""
        path = Path(checkpoint_path)
        
        # 파일 확장자 확인
        if path.suffix != '.pth':
            return False
        
        # 분류기 관련 키워드 확인
        classifier_keywords = ['classifier', 'deepfake', 'efficientnet']
        path_str = str(path).lower()
        
        return any(keyword in path_str for keyword in classifier_keywords)
    
    def _get_yolo_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict]:
        """YOLO 체크포인트 정보 추출"""
        try:
            path = Path(checkpoint_path)
            stat = path.stat()
            
            info = {
                'type': 'yolo',
                'path': str(path),
                'name': path.name,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'is_best': 'best' in path.name.lower(),
                'is_last': 'last' in path.name.lower(),
            }
            
            # YOLO 모델 정보 추출 시도
            try:
                from ultralytics import YOLO
                model = YOLO(checkpoint_path)
                # 모델 메타데이터가 있다면 추가
                if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
                    info['model_info'] = model.model.yaml
            except:
                pass
            
            return info
            
        except Exception as e:
            print(f"YOLO 체크포인트 정보 추출 실패: {checkpoint_path}, {e}")
            return None
    
    def _get_classifier_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict]:
        """분류기 체크포인트 정보 추출"""
        try:
            path = Path(checkpoint_path)
            stat = path.stat()
            
            info = {
                'type': 'classifier',
                'path': str(path),
                'name': path.name,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'is_best': 'best' in path.name.lower(),
            }
            
            # PyTorch 체크포인트 정보 추출
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                if isinstance(checkpoint, dict):
                    info.update({
                        'epoch': checkpoint.get('epoch', 'Unknown'),
                        'train_loss': checkpoint.get('train_loss', 'Unknown'),
                        'val_loss': checkpoint.get('val_loss', 'Unknown'),
                        'val_acc': checkpoint.get('val_acc', 'Unknown'),
                    })
                    
                    # 히스토리 정보가 있다면
                    if 'history' in checkpoint:
                        history = checkpoint['history']
                        if isinstance(history, dict):
                            info['total_epochs'] = len(history.get('train_loss', []))
                            if history.get('val_acc'):
                                info['best_val_acc'] = max(history['val_acc'])
            except:
                pass
            
            return info
            
        except Exception as e:
            print(f"분류기 체크포인트 정보 추출 실패: {checkpoint_path}, {e}")
            return None
    
    def find_latest_checkpoint(self, model_type: str) -> Optional[str]:
        """최신 체크포인트 경로 반환"""
        checkpoints = self.list_checkpoints(model_type)
        
        if not checkpoints:
            return None
        
        # 최신 체크포인트 반환
        return checkpoints[0]['path']
    
    def find_best_checkpoint(self, model_type: str) -> Optional[str]:
        """최고 성능 체크포인트 경로 반환"""
        checkpoints = self.list_checkpoints(model_type)
        
        # best 표시가 있는 체크포인트 우선
        for checkpoint in checkpoints:
            if checkpoint.get('is_best', False):
                return checkpoint['path']
        
        # 분류기의 경우 가장 높은 검증 정확도
        if model_type == 'classifier':
            best_acc = -1
            best_path = None
            
            for checkpoint in checkpoints:
                val_acc = checkpoint.get('val_acc')
                if isinstance(val_acc, (int, float)) and val_acc > best_acc:
                    best_acc = val_acc
                    best_path = checkpoint['path']
            
            if best_path:
                return best_path
        
        # 그 외의 경우 최신 체크포인트
        return checkpoints[0]['path'] if checkpoints else None
    
    def clean_old_checkpoints(self, model_type: str, keep_count: int = 5):
        """오래된 체크포인트 정리"""
        checkpoints = self.list_checkpoints(model_type)
        
        # best 모델과 최신 모델들은 보존
        to_keep = set()
        to_delete = []
        
        # best 모델 보존
        for checkpoint in checkpoints:
            if checkpoint.get('is_best', False):
                to_keep.add(checkpoint['path'])
        
        # 최신 keep_count개 보존
        recent_checkpoints = sorted(checkpoints, key=lambda x: x['modified_time'], reverse=True)
        for checkpoint in recent_checkpoints[:keep_count]:
            to_keep.add(checkpoint['path'])
        
        # 삭제할 체크포인트 선정
        for checkpoint in checkpoints:
            if checkpoint['path'] not in to_keep:
                to_delete.append(checkpoint)
        
        # 삭제 실행
        deleted_count = 0
        for checkpoint in to_delete:
            try:
                os.remove(checkpoint['path'])
                print(f"삭제: {checkpoint['path']}")
                deleted_count += 1
            except Exception as e:
                print(f"삭제 실패: {checkpoint['path']}, {e}")
        
        print(f"총 {deleted_count}개 체크포인트 삭제됨")
    
    def export_checkpoint_info(self, output_path: str):
        """체크포인트 정보를 JSON으로 내보내기"""
        all_checkpoints = self.list_checkpoints()
        
        # datetime 객체를 문자열로 변환
        for checkpoint in all_checkpoints:
            if 'modified_time' in checkpoint:
                checkpoint['modified_time'] = checkpoint['modified_time'].isoformat()
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'total_checkpoints': len(all_checkpoints),
            'checkpoints': all_checkpoints
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"체크포인트 정보 내보내기 완료: {output_path}")

def print_checkpoints_table(checkpoints: List[Dict]):
    """체크포인트 정보를 표 형태로 출력"""
    if not checkpoints:
        print("체크포인트가 없습니다.")
        return
    
    print(f"{'타입':<12} {'이름':<30} {'에폭':<8} {'검증정확도':<12} {'크기(MB)':<10} {'수정시간':<20}")
    print("-" * 95)
    
    for checkpoint in checkpoints:
        type_str = checkpoint['type']
        name_str = checkpoint['name'][:28] + ".." if len(checkpoint['name']) > 30 else checkpoint['name']
        epoch_str = str(checkpoint.get('epoch', '-'))
        acc_str = f"{checkpoint.get('val_acc', '-'):.4f}" if isinstance(checkpoint.get('val_acc'), (int, float)) else str(checkpoint.get('val_acc', '-'))
        size_str = str(checkpoint['size_mb'])
        time_str = checkpoint['modified_time'].strftime('%Y-%m-%d %H:%M')
        
        marker = " ⭐" if checkpoint.get('is_best', False) else ""
        
        print(f"{type_str:<12} {name_str:<30} {epoch_str:<8} {acc_str:<12} {size_str:<10} {time_str:<20}{marker}")

def main():
    parser = argparse.ArgumentParser(description='체크포인트 관리 도구')
    parser.add_argument('command', choices=['list', 'clean', 'find', 'export'],
                       help='실행할 명령')
    parser.add_argument('--save_dir', type=str, default='runs',
                       help='체크포인트 저장 디렉토리')
    parser.add_argument('--model_type', type=str, choices=['yolo', 'classifier'],
                       help='모델 타입')
    parser.add_argument('--keep_count', type=int, default=5,
                       help='정리 시 보존할 체크포인트 수')
    parser.add_argument('--output', type=str,
                       help='출력 파일 경로')
    parser.add_argument('--find_type', type=str, choices=['latest', 'best'], default='best',
                       help='찾을 체크포인트 타입')
    
    args = parser.parse_args()
    
    manager = CheckpointManager(args.save_dir)
    
    if args.command == 'list':
        checkpoints = manager.list_checkpoints(args.model_type)
        print_checkpoints_table(checkpoints)
    
    elif args.command == 'clean':
        if not args.model_type:
            print("정리할 모델 타입을 지정해주세요 (--model_type)")
            return
        
        manager.clean_old_checkpoints(args.model_type, args.keep_count)
    
    elif args.command == 'find':
        if not args.model_type:
            print("찾을 모델 타입을 지정해주세요 (--model_type)")
            return
        
        if args.find_type == 'latest':
            checkpoint_path = manager.find_latest_checkpoint(args.model_type)
        else:
            checkpoint_path = manager.find_best_checkpoint(args.model_type)
        
        if checkpoint_path:
            print(f"{args.find_type} {args.model_type} 체크포인트: {checkpoint_path}")
        else:
            print(f"{args.model_type} 체크포인트를 찾을 수 없습니다.")
    
    elif args.command == 'export':
        output_path = args.output or 'checkpoints_info.json'
        manager.export_checkpoint_info(output_path)

if __name__ == "__main__":
    main()

"""
사용 예제:

1. 모든 체크포인트 목록:
python checkpoint_manager.py list --save_dir runs

2. YOLO 체크포인트만 목록:
python checkpoint_manager.py list --save_dir runs --model_type yolo

3. 오래된 분류기 체크포인트 정리 (최신 3개만 보존):
python checkpoint_manager.py clean --save_dir runs --model_type classifier --keep_count 3

4. 최고 성능 YOLO 모델 찾기:
python checkpoint_manager.py find --save_dir runs --model_type yolo --find_type best

5. 체크포인트 정보 내보내기:
python checkpoint_manager.py export --save_dir runs --output my_checkpoints.json
"""