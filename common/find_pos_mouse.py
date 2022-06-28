import cv2
import numpy as np

# draw keyboard on screen
white_keyboards = [
    np.array([
        [8, 384], [64, 246], [102, 246], [84, 298], [73, 298], [41, 384]], dtype=np.int32
    ),
    np.array([
        [64, 384], [95, 298], [84, 298], [102, 246], [138, 246], [123, 298], [113, 298], [84, 384]], dtype=np.int32
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

black_keyboards = [
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


for key in white_keyboards:
    key = key.reshape((-1, 1, 2))

for key in black_keyboards:
    key = key.reshape((-1, 1, 2))


def find_points(event, x, y, flags, param):
   if event == cv2.EVENT_LBUTTONUP:
       print([x, y])


cv2.namedWindow(winname='img')
cv2.setMouseCallback('img', find_points)


def main():
    in_video = cv2.VideoCapture(1)

    in_video.set(3, 640)
    in_video.set(4, 480)

    while True:
        success, frame = in_video.read()
        frame = cv2.flip(frame, 1)

        for pts in white_keyboards:
            frame = cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

        for pts in black_keyboards:
            frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 0), thickness=1)

        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    in_video.release()


if __name__ == "__main__":
    main()
