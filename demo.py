"""
딥페이크 탐지 시스템 데모 스크립트

이 스크립트는 다양한 사용 시나리오를 보여줍니다:
1. 단일 이미지 분석
2. 배치 이미지 처리  
3. 비디오 분석
4. 실시간 웹캠 처리
5. 결과 시각화 및 저장
"""

import cv2
import os
import glob
import json
import argparse
from pathlib import Path
import time
from deepfake_detector import DeepfakeDetectionPipeline

class DeepfakeDemo:
    def __init__(self,
                 yolo_model_path=None,
                 classifier_weights_path=None,
                 confidence_threshold=0.5):
        """
        데모 클래스 초기화

        Args:
            yolo_model_path: 커스텀 YOLO 모델 경로
            classifier_weights_path: 훈련된 분류기 경로
            confidence_threshold: 탐지 신뢰도 임계값
        """
        print("딥페이크 탐지 파이프라인 초기화 중...")

        try:
            self.detector = DeepfakeDetectionPipeline(
                yolo_model_path=yolo_model_path,
                efficientnet_model='efficientnet-b0',
                classifier_weights_path=classifier_weights_path,
                confidence_threshold=confidence_threshold
            )
            print("✅ 파이프라인 초기화 완료!")
        except FileNotFoundError as e:
            print(f"❌ 파일을 찾을 수 없습니다: {e}")
            raise
        except RuntimeError as e:
            print(f"❌ 모델 초기화 실패: {e}")
            raise
        except Exception as e:
            print(f"❌ 예상치 못한 오류 발생: {e}")
            raise
    
    def analyze_single_image(self, image_path, save_result=True):
        """단일 이미지 분석"""
        print(f"\n📷 이미지 분석: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ 이미지를 찾을 수 없습니다: {image_path}")
            return None
        
        start_time = time.time()
        results = self.detector.detect_deepfake_from_image(image_path)
        process_time = time.time() - start_time
        
        # 결과 출력
        print(f"⏱️  처리 시간: {process_time:.2f}초")
        print(f"👥 탐지된 얼굴 수: {results['faces_detected']}")
        print(f"🎯 전체 결과: {results['overall_result']}")
        print(f"📊 평균 신뢰도: {results['confidence']:.3f}")
        
        if results['faces_detected'] > 0:
            print("\n얼굴별 상세 결과:")
            for face in results['faces']:
                status_emoji = "🚨" if face['prediction'] == 'fake' else "✅"
                print(f"  {status_emoji} 얼굴 {face['face_id']}: {face['prediction']} "
                      f"(신뢰도: {face['confidence']:.3f})")
        
        # 결과 시각화 저장
        if save_result and results['faces_detected'] > 0:
            image = cv2.imread(image_path)
            vis_image = self.detector.visualize_results(image, results)
            
            output_path = f"result_{Path(image_path).stem}.jpg"
            cv2.imwrite(output_path, vis_image)
            print(f"💾 결과 이미지 저장: {output_path}")
        
        return results
    
    def analyze_batch_images(self, image_dir, output_dir="batch_results"):
        """배치 이미지 처리"""
        print(f"\n📁 배치 이미지 분석: {image_dir}")
        
        # 이미지 파일 수집
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        if not image_files:
            print(f"❌ {image_dir}에서 이미지를 찾을 수 없습니다.")
            return
        
        print(f"📊 총 {len(image_files)}개 이미지 처리 중...")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 배치 처리 결과
        batch_results = []
        fake_count = 0
        real_count = 0
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n진행률: {i}/{len(image_files)} - {os.path.basename(img_path)}")
            
            start_time = time.time()
            results = self.detector.detect_deepfake_from_image(img_path)
            process_time = time.time() - start_time
            
            # 통계 업데이트
            if results['overall_result'] == 'fake':
                fake_count += 1
                status_emoji = "🚨"
            elif results['overall_result'] == 'real':
                real_count += 1
                status_emoji = "✅"
            else:
                status_emoji = "❓"
            
            print(f"  {status_emoji} {results['overall_result']} "
                  f"(신뢰도: {results['confidence']:.3f}, {process_time:.2f}초)")
            
            # 결과 저장
            result_data = {
                'filename': os.path.basename(img_path),
                'filepath': img_path,
                'result': results['overall_result'],
                'confidence': results['confidence'],
                'faces_detected': results['faces_detected'],
                'process_time': process_time,
                'faces': results['faces']
            }
            batch_results.append(result_data)
            
            # 시각화 저장
            if results['faces_detected'] > 0:
                image = cv2.imread(img_path)
                vis_image = self.detector.visualize_results(image, results)
                
                output_filename = f"result_{os.path.splitext(os.path.basename(img_path))[0]}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, vis_image)
        
        # 배치 결과 요약
        print(f"\n📈 배치 처리 완료!")
        print(f"  총 이미지: {len(image_files)}개")
        print(f"  🚨 딥페이크: {fake_count}개 ({fake_count/len(image_files)*100:.1f}%)")
        print(f"  ✅ 진짜: {real_count}개 ({real_count/len(image_files)*100:.1f}%)")
        print(f"  ❓ 얼굴 없음: {len(image_files)-fake_count-real_count}개")
        
        # JSON 결과 저장
        json_path = os.path.join(output_dir, 'batch_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        print(f"💾 상세 결과 저장: {json_path}")
        
        return batch_results
    
    def analyze_video(self, video_path, frame_interval=30, max_frames=100, save_result=True):
        """비디오 분석"""
        print(f"\n🎬 비디오 분석: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"❌ 비디오를 찾을 수 없습니다: {video_path}")
            return None
        
        start_time = time.time()
        results = self.detector.detect_deepfake_from_video(
            video_path,
            frame_interval=frame_interval,
            max_frames=max_frames
        )
        process_time = time.time() - start_time
        
        # 결과 출력
        print(f"⏱️  처리 시간: {process_time:.2f}초")
        print(f"📊 분석된 프레임: {results['total_frames_analyzed']}개")
        print(f"👥 얼굴 탐지된 프레임: {results['frames_with_faces']}개")
        print(f"🚨 딥페이크 프레임: {results['fake_frames']}개")
        print(f"✅ 진짜 프레임: {results['real_frames']}개")
        print(f"🎯 전체 결과: {results['overall_result']}")
        print(f"📈 평균 신뢰도: {results['confidence']:.3f}")
        
        if results['frames_with_faces'] > 0:
            fake_ratio = results['fake_frames'] / results['frames_with_faces'] * 100
            print(f"📊 딥페이크 비율: {fake_ratio:.1f}%")
        
        # 결과 저장
        if save_result:
            video_name = Path(video_path).stem
            json_path = f"video_result_{video_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 상세 결과 저장: {json_path}")
        
        return results
    
    def run_webcam_demo(self, show_fps=True):
        """실시간 웹캠 데모"""
        print("\n📹 실시간 웹캠 딥페이크 탐지 시작!")
        print("  - 종료: 'q' 키")
        print("  - 스크린샷: 's' 키")
        print("  - 통계 초기화: 'r' 키")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다.")
            return
        
        # 성능 측정 변수
        frame_count = 0
        fps_start_time = time.time()
        fake_count = 0
        real_count = 0
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 딥페이크 탐지
                start_time = time.time()
                results = self.detector.detect_deepfake_from_array(frame)
                process_time = time.time() - start_time
                
                # 결과 시각화
                vis_frame = self.detector.visualize_results(frame, results)
                
                # 통계 업데이트
                if results['overall_result'] == 'fake':
                    fake_count += 1
                elif results['overall_result'] == 'real':
                    real_count += 1
                
                # FPS 계산
                frame_count += 1
                if show_fps and frame_count % 30 == 0:
                    fps_time = time.time() - fps_start_time
                    fps = 30 / fps_time if fps_time > 0 else 0
                    fps_start_time = time.time()
                
                # 정보 오버레이
                info_text = f"Result: {results['overall_result'].upper()}"
                confidence_text = f"Confidence: {results['confidence']:.3f}"
                stats_text = f"Fake: {fake_count} | Real: {real_count}"
                process_text = f"Process: {process_time*1000:.1f}ms"
                
                if show_fps and 'fps' in locals():
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(vis_frame, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(vis_frame, info_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if results['overall_result'] == 'real' else (0, 0, 255), 2)
                cv2.putText(vis_frame, confidence_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_frame, stats_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_frame, process_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 화면 표시
                cv2.imshow('Deepfake Detection Demo', vis_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    screenshot_path = f"webcam_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(screenshot_path, vis_frame)
                    print(f"📷 스크린샷 저장: {screenshot_path}")
                elif key == ord('r'):
                    fake_count = 0
                    real_count = 0
                    frame_count = 0
                    fps_start_time = time.time()
                    print("📊 통계 초기화됨")
        
        except KeyboardInterrupt:
            print("\n⚠️  사용자가 중단했습니다.")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n📈 웹캠 데모 종료")
            print(f"  🚨 딥페이크: {fake_count}회")
            print(f"  ✅ 진짜: {real_count}회")
            print(f"  📷 스크린샷: {screenshot_count}개")

def main():
    parser = argparse.ArgumentParser(description='딥페이크 탐지 시스템 데모')
    parser.add_argument('mode', choices=['image', 'batch', 'video', 'webcam'],
                       help='데모 모드')
    parser.add_argument('--input', type=str,
                       help='입력 파일 또는 디렉토리 경로')
    parser.add_argument('--output', type=str, default='demo_results',
                       help='출력 디렉토리')
    parser.add_argument('--yolo_model', type=str,
                       help='커스텀 YOLO 모델 경로')
    parser.add_argument('--classifier_model', type=str,
                       help='훈련된 분류기 모델 경로')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='탐지 신뢰도 임계값')
    parser.add_argument('--frame_interval', type=int, default=30,
                       help='비디오 프레임 샘플링 간격')
    parser.add_argument('--max_frames', type=int, default=100,
                       help='비디오 최대 처리 프레임 수')

    args = parser.parse_args()

    # 데모 인스턴스 생성
    try:
        demo = DeepfakeDemo(
            yolo_model_path=args.yolo_model,
            classifier_weights_path=args.classifier_model,
            confidence_threshold=args.confidence
        )
    except Exception as e:
        print(f"\n❌ 데모 초기화 실패: {e}")
        print("\n💡 해결 방법:")
        print("  1. 먼저 모델을 학습하세요: python prepare_data.py --image_dir input/images --label_dir input/labels")
        print("  2. 학습된 모델 경로를 지정하세요:")
        print("     --yolo_model runs/face_detection/face_detector/weights/best.pt")
        print("     --classifier_model runs/deepfake_classifier/best_model.pth")
        return 1

    # 모드별 실행
    try:
        if args.mode == 'image':
            if not args.input:
                print("❌ 이미지 모드에는 --input 경로가 필요합니다.")
                return 1
            demo.analyze_single_image(args.input)

        elif args.mode == 'batch':
            if not args.input:
                print("❌ 배치 모드에는 --input 디렉토리가 필요합니다.")
                return 1
            demo.analyze_batch_images(args.input, args.output)

        elif args.mode == 'video':
            if not args.input:
                print("❌ 비디오 모드에는 --input 파일이 필요합니다.")
                return 1
            demo.analyze_video(args.input, args.frame_interval, args.max_frames)

        elif args.mode == 'webcam':
            demo.run_webcam_demo()

        return 0
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    main()

"""
사용 예제:

1. 단일 이미지 분석:
python demo.py image --input test_image.jpg

2. 배치 이미지 처리:
python demo.py batch --input test_images/ --output results/

3. 비디오 분석:
python demo.py video --input test_video.mp4 --frame_interval 15

4. 실시간 웹캠:
python demo.py webcam

5. 커스텀 모델 사용:
python demo.py image --input test.jpg \
    --yolo_model runs/face_detection/best.pt \
    --classifier_model runs/classifier/best_model.pth \
    --confidence 0.7
"""