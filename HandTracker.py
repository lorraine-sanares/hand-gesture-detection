import argparse
import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# 👇 NEW: import picamera2
from picamera2 import Picamera2
from libcamera import Transform

# Global FPS tracking
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:

    # 👇 NEW: Replace OpenCV camera capture with PiCamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (width, height)},
        transform=Transform(hflip=1, vflip=1)
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # let the camera warm up

    # Display parameters
    row_size = 50
    left_margin = 24
    text_color = (0, 0, 0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    def save_result(result: vision.HandLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        DETECTION_RESULT = result
        COUNTER += 1

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=save_result)
    detector = vision.HandLandmarker.create_from_options(options)

    while True:
        # 👇 NEW: Capture image from PiCamera2
        image = picam2.capture_array()
        

        if image is None:
            sys.exit('ERROR: Unable to read from PiCamera2.')
        
        # Convert to BGR so OpenCV + MediaPipe utils can use it
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        fps_text = f'FPS = {FPS:.1f}'
        cv2.putText(image, fps_text, (left_margin, row_size),
                    cv2.FONT_HERSHEY_DUPLEX, font_size, text_color,
                    font_thickness, cv2.LINE_AA)

        if DETECTION_RESULT:
            for idx in range(len(DETECTION_RESULT.hand_landmarks)):
                hand_landmarks = DETECTION_RESULT.hand_landmarks[idx]
                handedness = DETECTION_RESULT.handedness[idx]

                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                ])

                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

                h, w, _ = image.shape
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]
                text_x = int(min(x_coords) * w)
                text_y = int(min(y_coords) * h) - 10

                cv2.putText(image, handedness[0].category_name,
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            1, (88, 205, 54), 1, cv2.LINE_AA)

        cv2.imshow('hand_landmarker', image)
        if cv2.waitKey(1) == 27:  # ESC
            break

    detector.close()
    picam2.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hand_landmarker.task')
    parser.add_argument('--numHands', type=int, default=1)
    parser.add_argument('--minHandDetectionConfidence', type=float, default=0.5)
    parser.add_argument('--minHandPresenceConfidence', type=float, default=0.5)
    parser.add_argument('--minTrackingConfidence', type=float, default=0.5)
    parser.add_argument('--cameraId', type=int, default=0)  # ignored now
    parser.add_argument('--frameWidth', type=int, default=1280)
    parser.add_argument('--frameHeight', type=int, default=960)
    args = parser.parse_args()

    run(args.model, args.numHands, args.minHandDetectionConfidence,
        args.minHandPresenceConfidence, args.minTrackingConfidence,
        args.cameraId, args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()
