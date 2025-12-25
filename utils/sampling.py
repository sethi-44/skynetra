class FrameSampler:
    def __init__(self, detect_every=5):
        self.detect_every = detect_every
        self.counter = 0

    def should_run_detector(self):
        self.counter += 1
        return self.counter % self.detect_every == 0
