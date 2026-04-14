import queue
import threading
import time
import numpy as np

class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self.t.start()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        # 👉 修复 1：强行加上 True，伪装成统一的返回格式 (ret, frame)
        try:
            frame = self.q.get(timeout=5)
            return True, frame
        except queue.Empty:
            return False, None

    def close(self):
        self.enable = False
        self.t.join()
        # 👉 修复 2：把错误的原版 close() 换成合法的 release()
        if hasattr(self.cap, 'release'):
            self.cap.release()
