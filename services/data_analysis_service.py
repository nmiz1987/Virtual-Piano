import os
import time
import cv2
import numpy as np
import pygame

from matplotlib import pyplot as plt
from mediapipe.python.solutions.hands import HandLandmark
from shapely.geometry import Point
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from common.constants import NUM_OF_POINTS_FOR_REGRESSION, FRAMES_WINDOW_SIZE, DEVIATION_MULTIPLAYER, \
    MIN_ACCELERATION, MAX_ACCELERATION, POLYNOMIAL_DEGREE
from common.enums import Landmarks, Keyboard
from common.utils import WHITE_VIRTUAL_KEYS_DATA, BLACK_VIRTUAL_KEYS_DATA, \
    is_white_key, is_black_key
from services.hand_detector_service import MAX_HANDS


def compute_hands_landmarks_dataset(frame_number_to_hands_landmarks):
    frame_number = NUM_OF_POINTS_FOR_REGRESSION
    dataset = list()
    while frame_number <= len(frame_number_to_hands_landmarks):
        hand_to_frame_numbers = dict()
        hand_to_fingertip_type_to_x_points = dict()
        hand_to_fingertip_type_to_y_points = dict()
        for point_index in range(NUM_OF_POINTS_FOR_REGRESSION, 0, -1):
            frame_index = frame_number - point_index
            hands_landmarks = frame_number_to_hands_landmarks[frame_index]
            for hand_index, hand in enumerate(hands_landmarks[Landmarks.HANDS.value]):
                fingertip_type_to_x_points = hand_to_fingertip_type_to_x_points.get(hand_index, {})
                fingertip_type_to_y_points = hand_to_fingertip_type_to_y_points.get(hand_index, {})
                for finger_type, landmarks in hand.items():
                    loc_x_points = fingertip_type_to_x_points.get(finger_type, [])
                    loc_y_points = fingertip_type_to_y_points.get(finger_type, [])
                    loc_x_points.append(landmarks[Landmarks.LOC_X.value])
                    loc_y_points.append(landmarks[Landmarks.LOC_Y.value])
                    fingertip_type_to_x_points[finger_type] = loc_x_points
                    fingertip_type_to_y_points[finger_type] = loc_y_points
                hand_to_fingertip_type_to_x_points[hand_index] = fingertip_type_to_x_points
                hand_to_fingertip_type_to_y_points[hand_index] = fingertip_type_to_y_points
                frame_numbers = hand_to_frame_numbers.get(hand_index, [])
                frame_numbers.append(frame_index)
                hand_to_frame_numbers[hand_index] = frame_numbers
        data = (hand_to_frame_numbers, hand_to_fingertip_type_to_x_points, hand_to_fingertip_type_to_y_points)
        dataset.append(data)
        frame_number += 1
    return dataset


def compute_linear_regression(polynomial_fit, hand, fingertip_type_to_loc_points, axis_type):
    for finger_type, loc_points in fingertip_type_to_loc_points.items():
        loc_points_array = np.array(loc_points, ndmin=2)
        loc_points_array = loc_points_array.reshape(-1, 1)
        regression = LinearRegression()
        try:
            regression.fit(polynomial_fit, loc_points_array)
        except ValueError:
            continue
        coef = regression.coef_.flatten().tolist()
        landmarks = hand[finger_type]
        axis = landmarks.get(axis_type, {})
        axis[Landmarks.COEFFICIENTS.value] = coef
        axis[Landmarks.COEFFICIENT_A.value] = coef[1]
        axis[Landmarks.COEFFICIENT_B.value] = coef[0]
        axis[Landmarks.COEFFICIENT_C.value] = regression.intercept_[0]
        landmarks[axis_type] = axis


def compute_polynomial_regression(frame_number_to_hands_landmarks, dataset):
    for hand_to_frame_numbers, hand_to_fingertip_type_to_x_points, hand_to_fingertip_type_to_y_points in dataset:
        for hand_index, frame_numbers in hand_to_frame_numbers.items():
            if not frame_numbers:
                continue
            decision_frame_number = frame_numbers[-1]
            hands_landmarks = frame_number_to_hands_landmarks[decision_frame_number]
            hand = hands_landmarks[Landmarks.HANDS.value][hand_index]
            fingertip_type_to_loc_x_points = hand_to_fingertip_type_to_x_points[hand_index]
            fingertip_type_to_loc_y_points = hand_to_fingertip_type_to_y_points[hand_index]
            frame_number_array = np.array(frame_numbers, ndmin=2)
            frame_number_array = frame_number_array.reshape(-1, 1)
            polynomial = PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False)
            polynomial_fit = polynomial.fit_transform(frame_number_array)
            compute_linear_regression(polynomial_fit, hand, fingertip_type_to_loc_x_points, Landmarks.AXIS_X.value)
            compute_linear_regression(polynomial_fit, hand, fingertip_type_to_loc_y_points, Landmarks.AXIS_Y.value)


def compute_frames_velocity_from_derivative(frame_number_to_hands_landmarks):
    def _compute_velocity_by_derivative(axis):
        coefficients = axis[Landmarks.COEFFICIENTS.value]
        derivative = 0
        power = 1
        for coef in coefficients:
            derivative += coef * frame_number**(power - 1) * power
            power += 1
        return derivative

    for frame_number, hands_landmarks in frame_number_to_hands_landmarks.items():
        for hand in hands_landmarks[Landmarks.HANDS.value]:
            for landmarks in hand.values():
                axis_x = landmarks.get(Landmarks.AXIS_X.value)
                axis_y = landmarks.get(Landmarks.AXIS_Y.value)
                if axis_x:
                    landmarks[Landmarks.VELOCITY_X.value] = _compute_velocity_by_derivative(axis_x)
                if axis_y:
                    landmarks[Landmarks.VELOCITY_Y.value] = _compute_velocity_by_derivative(axis_y)


def get_hands_to_fingertip_type_to_velocities(frame_number_to_hands_landmarks) -> tuple:
    hands_to_fingertip_type_to_velocities_x = dict()
    hands_to_fingertip_type_to_velocities_y = dict()
    for hands_landmarks in frame_number_to_hands_landmarks.values():
        for hand_index, hand in enumerate(hands_landmarks[Landmarks.HANDS.value]):
            fingertip_type_to_velocities_x = hands_to_fingertip_type_to_velocities_x.get(hand_index, {})
            fingertip_type_to_velocities_y = hands_to_fingertip_type_to_velocities_y.get(hand_index, {})
            for fingertip_type, landmarks in hand.items():
                velocities_x = fingertip_type_to_velocities_x.get(fingertip_type, [])
                velocities_y = fingertip_type_to_velocities_y.get(fingertip_type, [])
                velocity_x = landmarks.get(Landmarks.VELOCITY_X.value)
                if velocity_x is not None:
                    velocities_x.append(velocity_x)
                velocity_y = landmarks.get(Landmarks.VELOCITY_Y.value)
                if velocity_y is not None:
                    velocities_y.append(velocity_y)
                fingertip_type_to_velocities_x[fingertip_type] = velocities_x
                fingertip_type_to_velocities_y[fingertip_type] = velocities_y
            hands_to_fingertip_type_to_velocities_x[hand_index] = fingertip_type_to_velocities_x
            hands_to_fingertip_type_to_velocities_y[hand_index] = fingertip_type_to_velocities_y
    return hands_to_fingertip_type_to_velocities_x, hands_to_fingertip_type_to_velocities_y


def compute_hands_fingertips_global_deviations(frame_number_to_hands_landmarks) -> tuple:
    hands_to_fingertip_type_to_velocities = get_hands_to_fingertip_type_to_velocities(frame_number_to_hands_landmarks)
    for hands_to_fingertip_type_to_velocities_axis in hands_to_fingertip_type_to_velocities:
        for hand_to_fingertip_type_to_velocities in hands_to_fingertip_type_to_velocities_axis.values():
            for fingertip_type, velocities in hand_to_fingertip_type_to_velocities.items():
                hand_to_fingertip_type_to_velocities[fingertip_type] = np.std(velocities)
    hands_fingertips_global_deviations_x, hands_fingertips_global_deviations_y = hands_to_fingertip_type_to_velocities
    return hands_fingertips_global_deviations_x, hands_fingertips_global_deviations_y


def compute_hand_fingertips_key_taps(hand_index, fingertips_global_deviations_y, frame_number_to_hands_landmarks):
    def _compute_standard_deviation_by_key(key):
        velocities = list(fingertip_type_velocities[key])
        return np.std(velocities)

    frame_number = FRAMES_WINDOW_SIZE + NUM_OF_POINTS_FOR_REGRESSION
    while frame_number <= len(frame_number_to_hands_landmarks):
        fingertip_type_velocities = dict()
        for point_index in range(FRAMES_WINDOW_SIZE, 0, -1):
            frame_index = frame_number - point_index
            hands_landmarks = frame_number_to_hands_landmarks[frame_index]
            hands = hands_landmarks[Landmarks.HANDS.value]
            try:
                hand = hands[hand_index]
            except IndexError:
                continue
            for finger_type, landmarks in hand.items():
                velocity_y = landmarks[Landmarks.VELOCITY_Y.value]
                velocity_to_frame_index = fingertip_type_velocities.get(finger_type, {})
                velocity_to_frame_index[velocity_y] = frame_index
                fingertip_type_velocities[finger_type] = velocity_to_frame_index
        if fingertip_type_velocities:
            finger_type = max(fingertip_type_velocities, key=_compute_standard_deviation_by_key)
            global_deviation_y = fingertips_global_deviations_y[finger_type]
            velocity_to_frame_index = fingertip_type_velocities[finger_type]
            max_velocity = max(velocity_to_frame_index)
            min_velocity = min(velocity_to_frame_index)
            is_pulse = velocity_to_frame_index[max_velocity] < velocity_to_frame_index[min_velocity]
            is_pulse_in_range = max_velocity - min_velocity >= global_deviation_y * DEVIATION_MULTIPLAYER
            is_velocity_in_range = abs(max_velocity) >= global_deviation_y and abs(min_velocity) >= global_deviation_y
            if is_pulse and is_pulse_in_range and is_velocity_in_range:
                tap_index = velocity_to_frame_index[max_velocity]
                hands_landmarks = frame_number_to_hands_landmarks[tap_index]
                hand = hands_landmarks[Landmarks.HANDS.value][hand_index]
                hand[finger_type][Landmarks.TAP.value] = True
                frame_number += FRAMES_WINDOW_SIZE
                continue
        frame_number += 1


def compute_hands_fingertips_key_taps(frame_number_to_hands_landmarks):
    _, hands_fingertips_global_deviations_y = compute_hands_fingertips_global_deviations(
        frame_number_to_hands_landmarks
    )
    for hand_index in range(MAX_HANDS):
        fingertips_global_deviations_y = hands_fingertips_global_deviations_y.get(hand_index)
        if fingertips_global_deviations_y:
            compute_hand_fingertips_key_taps(
                hand_index, fingertips_global_deviations_y, frame_number_to_hands_landmarks
            )


def compute_second_derivative(frame_number, axis):
    coefficients = axis[Landmarks.COEFFICIENTS.value]
    second_derivative = 0
    power = 1
    coef_multiplayer = 2
    for coef in coefficients[1:]:
        second_derivative += coef_multiplayer * coef * frame_number ** (power - 1) * power
        power += 1
        coef_multiplayer += 1
    return second_derivative


def determine_key_tapped(hand_index, fingertip_type, frame, frame_number, landmarks):
    def _get_available_channel():
        for channel_index in range(pygame.mixer.get_num_channels()):
            if not pygame.mixer.Channel(channel_index).get_busy():
                return channel_index

    def _normalize_acceleration():
        axis_y = landmarks[Landmarks.AXIS_Y.value]
        second_derivative = compute_second_derivative(frame_number, axis_y)
        return (second_derivative - MIN_ACCELERATION) / (MAX_ACCELERATION - MIN_ACCELERATION)

    loc_y = landmarks.get(Landmarks.LOC_Y.value)
    if is_white_key(loc_y):
        keys_data = WHITE_VIRTUAL_KEYS_DATA
        key_type = "white"
    elif is_black_key(loc_y):
        keys_data = BLACK_VIRTUAL_KEYS_DATA
        key_type = "black"
    else:
        return
    loc_x = landmarks.get(Landmarks.LOC_X.value)
    finger_point = Point(loc_x, loc_y).buffer(3)
    for key, data in keys_data.items():
        key_polygon = data[Keyboard.POLYGON.value]
        inter = finger_point.intersection(key_polygon)
        sound_file = data[Keyboard.SOUND_FILE.value]
        if not inter.is_empty:
            channel = _get_available_channel()
            if channel is not None:
                volume = _normalize_acceleration()
                if volume > 0.8:
                    volume = 1
                elif 0.6 < volume < 0.8:
                    volume = 0.6
                elif 0.3 < volume < 0.6:
                    volume = 0.3
                else:
                    volume = 0.1
                pygame.mixer.Channel(channel).set_volume(volume)
                pygame.mixer.Channel(channel).play(pygame.mixer.Sound(f"resources/sounds/{sound_file}"))
            else:
                volume = 0
            cv2.putText(frame, "TAP", (loc_x, loc_y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            # for debugging purposes
            # print(f"Frame {frame_number} Hand {hand_index}, Finger {HandLandmark(fingertip_type).name} Key {key_type}, {key} volume {volume} tap")
            break


def process_result_video(video_capture, keyboard, frame_number_to_hands_landmarks):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_number = 0
    frames = list()
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        hands_landmarks = frame_number_to_hands_landmarks[frame_number]
        for hand_index, hand in enumerate(hands_landmarks[Landmarks.HANDS.value]):
            for fingertip_type, landmarks in hand.items():
                cv2.putText(frame, f'FRAME: {frame_number}', (5, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                tap = landmarks.get(Landmarks.TAP.value)
                if tap:
                    determine_key_tapped(hand_index, fingertip_type, frame, frame_number, landmarks)
        frame_number += 1
        frame = cv2.addWeighted(frame, 1, keyboard, 1, 1)
        cv2.imshow("Analyzing", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)
        frames.append(frame)
    return frames


def plot_finger_graphs(hand_index, frame_number_to_hands_landmarks):
    def _get_velocities_dict():
        return {
            "velocities_x_frames": list(),
            "velocities_x": list(),
            "velocities_y_frames": list(),
            "velocities_y": list(),
            "acceleration_x": list(),
            "acceleration_x_frames": list(),
            "acceleration_y": list(),
            "acceleration_y_frames": list()
        }

    def _plot(x, y, x_label, y_label, color):
        plt.figure()
        plt.plot(x, y, color=color)
        plt.axis([0, len(x), min(y), max(y)])
        plt.xticks(np.arange(min(x), max(x) + 1, 20))
        plt.yticks(np.arange(min(y), max(y) + 1, 70))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        directory = f'resources/plots/Hand_{hand_index}/{HandLandmark(fingertip_type).name}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/{y_label}.png')
        plt.close()

    fingertip_type_to_velocities = dict()
    for frame_number, hands_landmarks in frame_number_to_hands_landmarks.items():
        try:
            hand = hands_landmarks[Landmarks.HANDS.value][hand_index]
        except IndexError:
            continue
        for fingertip_type, landmarks in hand.items():
            velocities = fingertip_type_to_velocities.get(fingertip_type, _get_velocities_dict())
            velocity_x = landmarks.get(Landmarks.VELOCITY_X.value)
            if velocity_x is not None:
                velocities["velocities_x"].append(round(velocity_x, 2))
                velocities["velocities_x_frames"].append(frame_number)
            velocity_y = landmarks.get(Landmarks.VELOCITY_Y.value)
            if velocity_y is not None:
                velocities["velocities_y"].append(round(velocity_y, 2))
                velocities["velocities_y_frames"].append(frame_number)
            axis_x = landmarks.get(Landmarks.AXIS_X.value)
            if axis_x is not None:
                second_derivative = compute_second_derivative(frame_number, axis_x)
                velocities["acceleration_x"].append(round(second_derivative, 2))
                velocities["acceleration_x_frames"].append(frame_number)
            axis_y = landmarks.get(Landmarks.AXIS_Y.value)
            if axis_y is not None:
                second_derivative = compute_second_derivative(frame_number, axis_y)
                velocities["acceleration_y"].append(round(second_derivative, 2))
                velocities["acceleration_y_frames"].append(frame_number)
            fingertip_type_to_velocities[fingertip_type] = velocities
    for fingertip_type, velocities in fingertip_type_to_velocities.items():
        fingertip_name = HandLandmark(fingertip_type).name
        _plot(
            velocities["velocities_x_frames"],
            velocities["velocities_x"],
            'Frame number',
            f'{fingertip_name} Velocity X', color='blue'
        )
        _plot(
            velocities["velocities_y_frames"],
            velocities["velocities_y"],
            'Frame number',
            f'{fingertip_name} Velocity Y', color='blue'
        )
        _plot(
            velocities["acceleration_x_frames"],
            velocities["acceleration_x"],
            'Frame number',
            f'{fingertip_name} Acceleration X',
            color='red'
        )
        _plot(
            velocities["acceleration_y_frames"],
            velocities["acceleration_y"],
            'Frame number',
            f'{fingertip_name} Acceleration Y',
            color='red'
        )


def plot_fingers_graphs(frame_number_to_hands_landmarks):
    for hand_index in range(MAX_HANDS):
        plot_finger_graphs(hand_index, frame_number_to_hands_landmarks)
