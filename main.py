import os

import pygame

from common.enums import Action, Variables

from common.utils import create_keyboard, write_data_to_file
from services.data_analysis_service import compute_frames_velocity_from_derivative, process_result_video, \
    compute_polynomial_regression, compute_hands_landmarks_dataset, plot_fingers_graphs, \
    compute_hands_fingertips_key_taps
from services.hand_detector_service import create_hands_object, calibrate, record_hands, compute_frames_hands_landmarks
from services.video_service import create_video_capture, write_video, read_video


if __name__ == '__main__':
    width = 640
    height = 480
    pygame.mixer.init(channels=10)
    keyboard = create_keyboard(width, height)
    hands_object = create_hands_object()
    action = os.environ.get(Variables.ACTION.value)
    if action == Action.RECORD.value:
        video_capture = create_video_capture(1, width, height)
        calibrate(video_capture, keyboard, hands_object)
        frames = record_hands(video_capture, keyboard, record_time_in_seconds=10)
        video_capture.release()
        write_video("resources/out_video.mp4", frames, width, height, fps=30.0)
    elif action == Action.PROCESS.value:
        video_capture = read_video("resources/out_video.mp4")
        frame_number_to_hands_landmarks = compute_frames_hands_landmarks(video_capture, hands_object)
        dataset = compute_hands_landmarks_dataset(frame_number_to_hands_landmarks)
        compute_polynomial_regression(frame_number_to_hands_landmarks, dataset)
        compute_frames_velocity_from_derivative(frame_number_to_hands_landmarks)
        compute_hands_fingertips_key_taps(frame_number_to_hands_landmarks)
        write_data_to_file("resources/hands_landmarks.json", frame_number_to_hands_landmarks)
        frames = process_result_video(video_capture, keyboard, frame_number_to_hands_landmarks)
        write_video("resources/result_video.mp4", frames, width, height, fps=30.0)
        if int(os.environ.get(Variables.PLOT_GRAPHS.value, "0")):
            plot_fingers_graphs(frame_number_to_hands_landmarks)
    else:
        raise Exception(f"No action found, available actions: {Action.values()}")
