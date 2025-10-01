import cv2
import os
from deepfake_detector import DeepfakeDetectionPipeline

def main():
    """딥페이크 탐지 파이프라인 사용 예제"""
    
    # 파이프라인 초기화
    detector = DeepfakeDetectionPipeline(
        yolo_model_path=None,  # 기본 YOLOv8n 사용
        efficientnet_model='efficientnet-b0',
        classifier_weights_path=None,  # 사전 훈련된 가중치 경로 (선택사항)
        confidence_threshold=0.5
    )
    
    # 이미지 분석 예제
    print("=== 이미지 분석 예제 ===")
    image_path = "sample_image.jpg"  # 테스트 이미지 경로
    
    if os.path.exists(image_path):
        # 이미지에서 딥페이크 탐지
        results = detector.detect_deepfake_from_image(image_path)
        
        print(f"탐지된 얼굴 수: {results['faces_detected']}")
        print(f"전체 결과: {results['overall_result']}")
        print(f"평균 신뢰도: {results['confidence']:.3f}")
        
        # 각 얼굴별 결과
        for face in results.get('faces', []):
            print(f"얼굴 {face['face_id']}: {face['prediction']} (신뢰도: {face['confidence']:.3f})")
        
        # 결과 시각화
        image = cv2.imread(image_path)
        vis_image = detector.visualize_results(image, results)
        
        # 결과 이미지 저장
        cv2.imwrite("result_image.jpg", vis_image)
        print("결과 이미지가 'result_image.jpg'로 저장되었습니다.")
    
    else:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # 비디오 분석 예제
    print("\n=== 비디오 분석 예제 ===")
    video_path = "sample_video.mp4"  # 테스트 비디오 경로
    
    if os.path.exists(video_path):
        # 비디오에서 딥페이크 탐지
        results = detector.detect_deepfake_from_video(
            video_path,
            frame_interval=30,  # 30프레임마다 분석
            max_frames=50      # 최대 50프레임 분석
        )
        
        print(f"분석된 프레임 수: {results['total_frames_analyzed']}")
        print(f"얼굴이 탐지된 프레임 수: {results['frames_with_faces']}")
        print(f"가짜로 판단된 프레임 수: {results['fake_frames']}")
        print(f"진짜로 판단된 프레임 수: {results['real_frames']}")
        print(f"전체 결과: {results['overall_result']}")
        print(f"평균 신뢰도: {results['confidence']:.3f}")
    
    else:
        print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    
    # 웹캠 실시간 분석 예제
    print("\n=== 웹캠 실시간 분석 예제 ===")
    print("웹캠을 시작하려면 'y'를 입력하세요 (종료: 'q'키):")
    
    if input().lower() == 'y':
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 분석
            results = detector.detect_deepfake_from_array(frame)
            
            # 결과 시각화
            vis_frame = detector.visualize_results(frame, results)
            
            # 전체 결과 표시
            overall_text = f"Overall: {results['overall_result']}"
            cv2.putText(vis_frame, overall_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Deepfake Detection', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()