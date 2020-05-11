import cv2
import numpy as np
class FrameProcessor:
    def __init__(self, frame_height=84, frame_width=84):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def preprocess(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110, :]
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        return np.reshape(observation, (84, 84, 1))

if __name__ == '__main__':
    print("hello")
    bla = FrameProcessor()
    a = np.array([1,2,3,4])
    bla.preprocess(a)