"""
Video frame extraction utilities.
"""

import numpy as np


def extract_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    """Extract uniformly-spaced frames from a video file.

    Returns BGR numpy arrays (OpenCV convention).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames
