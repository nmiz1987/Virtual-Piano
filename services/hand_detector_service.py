import time

import cv2

from typing import List
from collections import namedtuple

from mediapipe.python.solutions import hands, drawing_utils
from mediapipe.python.solutions.hands import Hands, HandLandmark

from common.constants import MAX_HANDS
from common.enums import Landmarks
from common.utils import FPS

FINGERTIPS_TYPES = [
    HandLandmark.THUMB_TIP,
    HandLandmark.INDEX_FINGER_TIP,
    HandLandmark.MIDDLE_FINGER_TIP,
    HandLandmark.RING_FINGER_TIP,
    HandLandmark.PINKY_TIP
]

SolutionsOutputs = namedtuple("SolutionsOutputs", ["multi_hand_landmarks", "multi_handedness"])


def create_hands_object(static_image_mode=False, max_num_hands=MAX_HANDS, min_detection_confidence=0.8,
                        min_tracking_confidence=0.8) -> Hands:
    return Hands(static_image_mode, max_num_hands, min_detection_confidence, min_tracking_confidence)


def process_hands(hands_object: Hands, frame, draw_landmarks=True) -> SolutionsOutputs:
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    graph_output: SolutionsOutputs = hands_object.process(img_rgb)
    if graph_output.multi_hand_landmarks and draw_landmarks:
        for handLms in graph_output.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, handLms, hands.HAND_CONNECTIONS)
    return graph_output


def get_hands_fingertips_landmarks(graph_output: SolutionsOutputs, frame, draw_fingertips=True) -> List[dict]:
    hands_fingertips_landmarks = list()
    if graph_output.multi_hand_landmarks:
        h, w, c = frame.shape
        for hand in graph_output.multi_hand_landmarks:
            fingertip_type_to_landmarks = dict()
            for fingertip_type in FINGERTIPS_TYPES:
                landmark = hand.landmark[fingertip_type]
                # convert the position from decimal to pixel
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                fingertip_type_to_landmarks[fingertip_type] = {Landmarks.LOC_X.value: cx, Landmarks.LOC_Y.value: cy}
                if draw_fingertips:
                    cv2.circle(frame, (cx, cy), radius=4, color=(0, 0, 255), thickness=cv2.FILLED)
            hands_fingertips_landmarks.append(fingertip_type_to_landmarks)
    return hands_fingertips_landmarks


def calibrate(video_capture, keyboard, hands_object):
    window_name = "Calibrating"
    while True:
        success, frame = video_capture.read()
        graph_output = process_hands(hands_object, frame)
        get_hands_fingertips_landmarks(graph_output, frame)
        frame = cv2.addWeighted(frame, 1, keyboard, 1, 1)
        cv2.putText(frame, "Press 'q' to finish calibration", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # frame = cv2.flip(frame, 1)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)


def record_hands(video_capture, keyboard, record_time_in_seconds, display_fps=False):
    window_name = "Recording"
    fps_object = FPS()
    frames = list()
    start_time = time.time()
    while time.time() - start_time < record_time_in_seconds:
        success, frame = video_capture.read()
        if not success:
            continue
        frames.append(frame)
        if display_fps:
            fps, frame = fps_object.update(frame, (5, 40))
        frame = cv2.addWeighted(frame, 1, keyboard, 1, 1)
        frame = cv2.flip(frame, 1)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)
    return frames


def compute_frames_hands_landmarks(video_capture, hands_object: Hands):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_number = 0
    frame_number_to_hands_landmarks = dict()
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        graph_output = process_hands(hands_object, frame, draw_landmarks=False)
        frame_number_to_hands_landmarks[frame_number] = dict()
        hands_fingertips_landmarks = get_hands_fingertips_landmarks(graph_output, frame, draw_fingertips=False)
        frame_number_to_hands_landmarks[frame_number][Landmarks.TIMESTAMP.value] = time.time()
        frame_number_to_hands_landmarks[frame_number][Landmarks.HANDS.value] = hands_fingertips_landmarks
        frame_number += 1
    return frame_number_to_hands_landmarks
