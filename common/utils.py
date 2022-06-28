import cv2
import numpy as np
import time
import json

from shapely.geometry import Polygon

from common.enums import Keyboard

WHITE_KEYS_POINTS = [
    np.array(
        [[8, 384], [64, 246], [102, 246], [84, 298], [73, 298], [41, 384]], dtype=np.int32
    ),
    np.array(
        [[64, 384], [95, 298], [84, 298], [102, 246], [138, 246], [123, 298], [113, 298], [84, 384]], dtype=np.int32
    ),
    np.array(
        [[110, 384], [135, 298], [123, 298], [138, 246], [175, 246], [142, 384]], dtype=np.int32
    ),
    np.array(
        [[142, 384], [175, 246], [210, 246], [202, 298], [191, 298], [175, 384]], dtype=np.int32
    ),
    np.array(
        [[202, 384], [214, 298], [202, 298], [210, 246], [247, 246], [241, 298], [231, 298], [221, 384]], dtype=np.int32
    ),
    np.array(
        [[244, 384], [252, 298], [241, 298], [247, 246], [283, 246], [282, 298], [270, 298], [265, 384]], dtype=np.int32
    ),
    np.array(
        [[290, 384], [293, 298], [282, 298], [283, 246], [321, 246], [322, 384]], dtype=np.int32
    ),
    np.array(
        [[322, 384], [321, 246], [358, 246], [361, 298], [352, 298], [356, 384]], dtype=np.int32
    ),
    np.array(
        [[381, 384], [373, 298], [361, 298], [358, 246], [395, 246], [401, 298], [391, 298], [401, 384]], dtype=np.int32
    ),
    np.array(
        [[425, 384], [413, 298], [401, 298], [395, 246], [431, 246], [457, 384]], dtype=np.int32
    ),
    np.array(
        [[457, 384], [431, 246], [467, 246], [481, 298], [471, 298], [490, 384]], dtype=np.int32
    ),
    np.array(
        [[514, 384], [492, 298], [481, 298], [467, 246], [503, 246], [520, 298], [509, 298], [536, 384]], dtype=np.int32
    ),
    np.array(
        [[560, 384], [530, 298], [520, 298], [503, 246], [539, 246], [558, 298], [549, 298], [580, 384]], dtype=np.int32
    ),
    np.array(
        [[604, 384], [570, 298], [558, 298], [539, 246], [576, 246], [636, 384]], dtype=np.int32
    )
]

BLACK_KEYS_POINTS = [
    np.array([[41, 384], [73, 298], [95, 298], [64, 384]], dtype=np.int32),
    np.array([[86, 384], [113, 298], [135, 298], [110, 384]], dtype=np.int32),
    np.array([[175, 384], [191, 298], [214, 298], [200, 384]], dtype=np.int32),
    np.array([[221, 384], [231, 298], [252, 298], [244, 384]], dtype=np.int32),
    np.array([[265, 384], [270, 298], [293, 298], [290, 384]], dtype=np.int32),
    np.array([[356, 384], [352, 298], [373, 298], [381, 384]], dtype=np.int32),
    np.array([[401, 384], [391, 298], [413, 298], [425, 384]], dtype=np.int32),
    np.array([[490, 384], [471, 298], [492, 298], [514, 384]], dtype=np.int32),
    np.array([[536, 384], [509, 298], [530, 298], [560, 384]], dtype=np.int32),
    np.array([[580, 384], [549, 298], [570, 298], [604, 384]], dtype=np.int32)
]


WHITE_VIRTUAL_KEYS_DATA = {
    14: {
        Keyboard.POLYGON.value: Polygon(
            [(8, 384), (64, 246), (102, 246), (84, 298), (73, 298), (41, 384)]),
        Keyboard.SOUND_FILE.value: "a0.mp3"
    },
    13: {
        Keyboard.POLYGON.value: Polygon(
            [(64, 384), (95, 298), (84, 298), (102, 246), (138, 246), (123, 298), (113, 298), (84, 384)]),
        Keyboard.SOUND_FILE.value: "a1.mp3"
    },
    12: {
        Keyboard.POLYGON.value: Polygon(
            [(110, 384), (135, 298), (123, 298), (138, 246), (175, 246), (142, 384)]),
        Keyboard.SOUND_FILE.value: "a2.mp3"
    },
    11: {
        Keyboard.POLYGON.value: Polygon(
            [(142, 384), (175, 246), (210, 246), (202, 298), (191, 298), (175, 384)]),
        Keyboard.SOUND_FILE.value: "a3.mp3"
    },
    10: {
        Keyboard.POLYGON.value: Polygon(
            [(202, 384), (214, 298), (202, 298), (210, 246), (247, 246), (241, 298), (231, 298), (221, 384)]),
        Keyboard.SOUND_FILE.value: "a4.mp3"
    },
    9: {
        Keyboard.POLYGON.value: Polygon(
            [(244, 384), (252, 298), (241, 298), (247, 246), (283, 246), (282, 298), (270, 298), (265, 384)]),
        Keyboard.SOUND_FILE.value: "a5.mp3"
    },
    8: {
        Keyboard.POLYGON.value: Polygon(
            [(290, 384), (293, 298), (282, 298), (283, 246), (321, 246), (322, 384)]),
        Keyboard.SOUND_FILE.value: "a6.mp3"
    },
    7: {
        Keyboard.POLYGON.value: Polygon(
            [(322, 384), (321, 246), (358, 246), (361, 298), (352, 298), (356, 384)]),
        Keyboard.SOUND_FILE.value: "b0.mp3"
    },
    6: {
        Keyboard.POLYGON.value: Polygon(
            [(381, 384), (373, 298), (361, 298), (358, 246), (395, 246), (401, 298), (391, 298), (401, 384)]),
        Keyboard.SOUND_FILE.value: "b1.mp3"
    },
    5: {
        Keyboard.POLYGON.value: Polygon(
            [(425, 384), (413, 298), (401, 298), (395, 246), (431, 246), (457, 384)]),
        Keyboard.SOUND_FILE.value: "b2.mp3"
    },
    4: {
        Keyboard.POLYGON.value: Polygon(
            [(457, 384), (431, 246), (467, 246), (481, 298), (471, 298), (490, 384)]),
        Keyboard.SOUND_FILE.value: "b3.mp3"
    },
    3: {
        Keyboard.POLYGON.value: Polygon(
            [(514, 384), (492, 298), (481, 298), (467, 246), (503, 246), (520, 298), (509, 298), (536, 384)]),
        Keyboard.SOUND_FILE.value: "b4.mp3"
    },
    2: {
        Keyboard.POLYGON.value: Polygon(
            [(560, 384), (530, 298), (520, 298), (503, 246), (539, 246), (558, 298), (549, 298), (580, 384)]),
        Keyboard.SOUND_FILE.value: "b5.mp3"
    },
    1: {
        Keyboard.POLYGON.value: Polygon(
            [(604, 384), (570, 298), (558, 298), (539, 246), (576, 246), (636, 384)]),
        Keyboard.SOUND_FILE.value: "b6.mp3"
    }
}


BLACK_VIRTUAL_KEYS_DATA = {
    10: {
        Keyboard.POLYGON.value: Polygon(
            [(41, 384), (73, 298), (95, 298), (64, 384)]),
        Keyboard.SOUND_FILE.value: "ds2.mp3"
    },
    9: {
        Keyboard.POLYGON.value: Polygon(
            [(86, 384), (113, 298), (135, 298), (110, 384)]),
        Keyboard.SOUND_FILE.value: "ds3.mp3"
    },
    8: {
        Keyboard.POLYGON.value: Polygon(
            [(175, 384), (191, 298), (214, 298), (200, 384)]),
        Keyboard.SOUND_FILE.value: "ds4.mp3"
    },
    7: {
        Keyboard.POLYGON.value: Polygon(
            [(221, 384), (231, 298), (252, 298), (244, 384)]),
        Keyboard.SOUND_FILE.value: "ds5.mp3"
    },
    6: {
        Keyboard.POLYGON.value: Polygon(
            [(265, 384), (270, 298), (293, 298), (290, 384)]),
        Keyboard.SOUND_FILE.value: "ds6.mp3"
    },
    5: {
        Keyboard.POLYGON.value: Polygon(
            [(356, 384), (352, 298), (373, 298), (381, 384)]),
        Keyboard.SOUND_FILE.value: "g2.mp3"
    },
    4: {
        Keyboard.POLYGON.value: Polygon(
            [(401, 384), (391, 298), (413, 298), (425, 384)]),
        Keyboard.SOUND_FILE.value: "g3.mp3"
    },
    3: {
        Keyboard.POLYGON.value: Polygon(
            [(490, 384), (471, 298), (492, 298), (514, 384)]),
        Keyboard.SOUND_FILE.value: "g4.mp3"
    },
    2: {
        Keyboard.POLYGON.value: Polygon(
            [(536, 384), (509, 298), (530, 298), (560, 384)]),
        Keyboard.SOUND_FILE.value: "g5.mp3"
    },
    1: {
        Keyboard.POLYGON.value: Polygon(
            [(580, 384), (549, 298), (570, 298), (604, 384)]),
        Keyboard.SOUND_FILE.value: "g6.mp3"
    }
}


class FPS:
    """
    Helps in finding Frames Per Second and display on an OpenCV Image
    """

    def __init__(self):
        self.p_time = time.time()

    def update(self, img=None, pos=(20, 50), color=(255, 0, 0), scale=3, thickness=3):
        """
        Update the frame rate
        :param img: Image to display on, can be left blank if only fps value required
        :param pos: Position on the FPS on the image
        :param color: Color of the FPS Value displayed
        :param scale: Scale of the FPS Value displayed
        :param thickness: Thickness of the FPS Value displayed
        :return:
        """
        c_time = time.time()
        try:
            fps = 1 / (c_time - self.p_time)
        except ZeroDivisionError:
            return 0
        self.p_time = c_time
        if img is None:
            return fps
        else:
            cv2.putText(img, f'FPS: {int(fps)}', pos, cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)
            return fps, img


def create_keyboard(width, height):
    keyboard = np.ones((height, width, 3), np.uint8)
    for white_key_point in WHITE_KEYS_POINTS:
        reshaped_white_key = white_key_point.reshape((-1, 1, 2))
        keyboard = cv2.polylines(keyboard, [reshaped_white_key], isClosed=True, color=(255, 255, 255), thickness=1)
    for black_key in BLACK_KEYS_POINTS:
        reshaped_black_key = black_key.reshape((-1, 1, 2))
        keyboard = cv2.polylines(keyboard, [reshaped_black_key], isClosed=True, color=(0, 255, 0), thickness=1)
    return keyboard


def is_white_key(loc_y):
    return 246 < loc_y < 298


def is_black_key(loc_y):
    return 298 < loc_y < 384


def write_data_to_file(filepath, data):
    out_file = open(filepath, "w")
    json.dump(data, out_file)
    out_file.close()


def read_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
