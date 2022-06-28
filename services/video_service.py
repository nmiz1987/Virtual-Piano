import cv2

from cv2 import VideoCapture, VideoWriter_fourcc, VideoWriter, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT


def create_video_capture(index, width, height) -> VideoCapture:
    video_capture = VideoCapture(index)
    video_capture.set(CAP_PROP_FRAME_WIDTH, width)
    video_capture.set(CAP_PROP_FRAME_HEIGHT, height)
    return video_capture


def write_video(filepath: str, frames: list, width, height, fps: float):
    """
    Writes frames to an mp4 video file
    :param filepath: Path to output video, must end with .mp4
    :param frames: List of frames
    :param width: video width
    :param height: video height
    :param fps: Desired frame rate
    """
    fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = VideoWriter(filepath, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def read_video(filepath: str):
    return cv2.VideoCapture(filepath)
