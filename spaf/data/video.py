import numpy as np
import time
import cv2
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import (Tuple, List, TypedDict)

log = logging.getLogger(__name__)


class OCV_rstats(TypedDict):
    # OCV reachability stats
    height: int
    width: int
    frame_count: int
    fps: float
    max_pos_frames: int  # 1-based
    max_pos_msec: float


@contextmanager
def video_capture_open(video_path, tries=1):
    i = 0
    while i < tries:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                break
            else:
                raise IOError(f'OpenCV cannot open {video_path}')
        except Exception:
            time.sleep(1)
            i += 1

    if not cap.isOpened():
        raise IOError(f'OpenCV cannot open {video_path} after {i} tries')

    yield cap
    cap.release()


def video_getHW(cap) -> Tuple[int, int]:
    return (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))


def compute_ocv_rstats(video_path, n_tries=5) -> OCV_rstats:
    with video_capture_open(video_path, n_tries) as vcap:
        height, width = video_getHW(vcap)
        frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vcap.get(cv2.CAP_PROP_FPS)
        while True:
            max_pos_frames = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
            max_pos_msec = vcap.get(cv2.CAP_PROP_POS_MSEC)
            ret = vcap.grab()
            if ret is False:
                break
    ocv_rstats: OCV_rstats = {
        'height': height,
        'width': width,
        'frame_count': frame_count,
        'fps': fps,
        'max_pos_frames': max_pos_frames,
        'max_pos_msec': max_pos_msec,
        }
    return ocv_rstats


def query_video_ocv_stats(video_path, n_tries=5):
    with video_capture_open(video_path, n_tries) as vcap:
        height, width = video_getHW(vcap)
        reported_framecount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_fps = vcap.get(cv2.CAP_PROP_FPS)
        # Try to iterate
        vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret = vcap.grab()
            if ret is False:
                break
        frames_reached = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
        ms_reached = int(vcap.get(cv2.CAP_PROP_POS_MSEC))
    qstats = {
        'reported_framecount': reported_framecount,
        'reported_fps': reported_fps,
        'frames_reached': frames_reached,
        'ms_reached': ms_reached,
        'height': height,
        'width': width}
    return qstats


class VideoCaptureError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)


def video_sorted_enumerate(cap,
        sorted_framenumbers,
        throw_on_failure=True):
    """
    Fast opencv frame iteration
    - Only operates on sorted frames
    - Throws if failed to read the frame.
    - The failures happen quite often

    Args:
        cap: cv2.VideoCapture class
        sorted_framenumbers: Sorted sequence of framenumbers
        strict: Throw if failed to read frame
    Yields:
        (framenumber, frame_BGR)
    """

    def stop_at_0():
        if ret == 0:
            if throw_on_failure:
                raise VideoCaptureError(
                        f'Failed to read frame {f_current}')
            else:
                log.warning(f'Failed to read frame {f_current}')
                return

    assert (np.diff(sorted_framenumbers) >= 0).all(), \
            'framenumber must be nondecreasing'
    sorted_framenumbers = iter(sorted_framenumbers)
    try:  # Will stop iteration if empty
        f_current = next(sorted_framenumbers)
    except StopIteration:
        return
    f_next = f_current
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_current)
    while True:
        while f_current <= f_next:  # Iterate till f_current, f_next match
            f_current += 1
            ret = cap.grab()
            stop_at_0()
        ret, frame_BGR = cap.retrieve()
        yield (f_current-1, frame_BGR)
        try:  # Will stop iteration if empty
            f_next = next(sorted_framenumbers)
        except StopIteration:
            return
        assert f_current == int(cap.get(cv2.CAP_PROP_POS_FRAMES))


def video_sample(cap, framenumbers) -> List:
    sorted_framenumber = np.unique(framenumbers)
    frames_BGR = {}
    for i, frame_BGR in video_sorted_enumerate(cap, sorted_framenumber):
        frames_BGR[i] = frame_BGR
    sampled_BGR = []
    for i in framenumbers:
        sampled_BGR.append(frames_BGR[i])
    return sampled_BGR


"""
Writing videos

video_path = vt_cv.suffix_helper(video_path, 'XVID')
with vt_cv.video_writer_open(
        video_path, sizeWH, 10, 'XVID') as vout:
    ...
    vout.write(im)
"""
# These values work for me
FOURCC_TO_CONTAINER = {
        'VP90': '.webm',
        'XVID': '.avi',
        'MJPG': '.avi',
        'H264': '.mp4',
        'MP4V': '.mp4'
}


def suffix_helper(video_path: Path, fourcc) -> Path:
    """Change video suffix based on 4cc"""
    return video_path.with_suffix(FOURCC_TO_CONTAINER[fourcc])


@contextmanager
def video_writer_open(
        video_path: Path,
        size_WH: Tuple[int, int],
        framerate: float,
        fourcc):

    cv_fourcc = cv2.VideoWriter_fourcc(*fourcc)
    vout = cv2.VideoWriter(str(video_path), cv_fourcc, framerate, size_WH)
    try:
        if not vout.isOpened():
            raise IOError(f'OpenCV cannot open {video_path} for writing')
        yield vout
    finally:
        vout.release()
