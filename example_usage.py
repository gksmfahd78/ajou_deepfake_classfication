import cv2
import os
from deepfake_detector import DeepfakeDetectionPipeline

def main():
    """딥페이크 탐지 파이프라인 사용 예제"""

    print("⚠️  중요: 이 예제를 실행하려면 먼저 모델을 학습해야 합니다!")
    print("학습 방법: python prepare_data.py --image_dir input/images --label_dir input/labels")
    print("그 후: python resume_training.py config\n")

    # 학습된 모델 경로 설정
    yolo_model_path = "runs/face_detection/face_detector/weights/best.pt"
    classifier_weights_path = "runs/deepfake_classifier/best_model.pth"

    # 학습된 모델이 있는지 확인
    models_exist = os.path.exists(yolo_model_path) and os.path.exists(classifier_weights_path)

    if not models_exist:
        print("❌ 학습된 모델을 찾을 수 없습니다.")
        print(f"   YOLO 모델: {yolo_model_path} {'✓' if os.path.exists(yolo_model_path) else '✗'}")
        print(f"   분류기 모델: {classifier_weights_path} {'✓' if os.path.exists(classifier_weights_path) else '✗'}")
        print("\n💡 기본 모델로 실행하시겠습니까? (정확도가 매우 낮을 수 있습니다)")
        response = input("계속하려면 'y'를 입력하세요: ")
        if response.lower() != 'y':
            print("종료합니다.")
            return
        yolo_model_path = None
        classifier_weights_path = None

    # 파이프라인 초기화
    try:
        print("\n딥페이크 탐지 파이프라인 초기화 중...")
        detector = DeepfakeDetectionPipeline(
            yolo_model_path=yolo_model_path,
            efficientnet_model='efficientnet-b0',
            classifier_weights_path=classifier_weights_path,
            confidence_threshold=0.5
        )
        print("✅ 초기화 완료!\n")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    # 이미지 분석 예제
    print("=== 이미지 분석 예제 ===")
    # 여러 가능한 경로 시도
    possible_image_paths = ["sample_image.jpg", "test_image.jpg", "input/images/*.jpg"]
    image_path = None

    for path_pattern in possible_image_paths:
        if '*' in path_pattern:
            import glob
            matches = glob.glob(path_pattern)
            if matches:
                image_path = matches[0]
                break
        elif os.path.exists(path_pattern):
            image_path = path_pattern
            break

    if image_path and os.path.exists(image_path):
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
        print(f"⚠️  테스트 이미지를 찾을 수 없습니다.")
        print(f"   다음 위치에 'sample_image.jpg' 파일을 배치하세요: {os.getcwd()}")

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
        print(f"⚠️  테스트 비디오를 찾을 수 없습니다: {video_path}")
    
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