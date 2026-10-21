"""Frame-exact plate reading and high-quality intermediate video for the inference workers.

Sources are either an uploaded movie or an uploaded proxy archive (ZIP of JPEG frames plus
plate.json) built by the add-on from an image-sequence plate. Movies are read sequentially from
frame 0, because container seeking is not frame-exact for every codec.
"""
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np


class Plate:
    def __init__(self, path, fps_override=0.):
        self.path = Path(path)
        self.archive = None
        if self.path.suffix.lower() == '.zip':
            self.archive = zipfile.ZipFile(self.path)
            self.plate = json.loads(self.archive.read('plate.json'))
            self.frames = sorted(name for name in self.archive.namelist() if name.startswith('frames/') and name.endswith('.jpg'))
            if len(self.frames) != self.plate.get('frame_count', len(self.frames)):
                raise ValueError('The plate archive is incomplete')
            self.fps = fps_override or self.plate['fps_num']/self.plate['fps_den']
            self.count = len(self.frames)
        else:
            self.plate = None
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                raise ValueError('Cannot decode the uploaded video')
            self.fps = fps_override or capture.get(cv2.CAP_PROP_FPS)
            self.count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Video has no usable frame rate; set the FPS override')

    def read(self, start=0, count=0):
        """Yield BGR frames start, start+1, ... (count=0: to the end)."""
        if self.archive is not None:
            names = self.frames[start:start+count if count else None]
            for name in names:
                frame = cv2.imdecode(np.frombuffer(self.archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('Cannot decode plate frame '+name)
                yield frame
            return
        capture = cv2.VideoCapture(str(self.path))
        try:
            for _ in range(start):
                if not capture.grab():
                    return
            produced = 0
            while not count or produced < count:
                ok, frame = capture.read()
                if not ok:
                    return
                produced += 1
                yield frame
        finally:
            capture.release()

    def frame(self, index):
        return next(self.read(index, 1), None)


def processing_size(width, height, max_side):
    scale = min(1., max_side/max(width, height))
    return int(width*scale)//2*2, int(height*scale)//2*2


class VideoSink:
    """H.264 writer through PyAV: CRF 0 is lossless, CRF ~10 visually lossless in 4:4:4."""

    def __init__(self, path, fps, width, height, crf=10, lossless=False):
        import av
        from fractions import Fraction
        self.container = av.open(str(path), 'w')
        rate = Fraction(fps).limit_denominator(100000)
        self.stream = self.container.add_stream('libx264', rate=rate)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = 'yuv444p'
        self.stream.time_base = 1/rate
        self.stream.options = {'crf': '0' if lossless else str(crf), 'preset': 'veryfast'}
        self._frame = av.VideoFrame
        self.index = 0

    def write(self, bgr):
        frame = self._frame.from_ndarray(np.ascontiguousarray(bgr), format='bgr24')
        frame.pts = self.index
        self.index += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
