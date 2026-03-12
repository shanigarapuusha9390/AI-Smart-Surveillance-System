from collections import deque
from config import FPS, BUFFER_SECONDS


class RollingBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=FPS * BUFFER_SECONDS)

    def add(self, frame):
        self.buffer.append(frame.copy())

    def get(self):
        return list(self.buffer)