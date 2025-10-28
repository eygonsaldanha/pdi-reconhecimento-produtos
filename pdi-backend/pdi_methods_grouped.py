import cv2
import numpy as np


def process_image_pdi_concat(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()

    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    hist_rgb = np.concatenate([cv2.normalize(hist_b, hist_b).flatten(), cv2.normalize(hist_g, hist_g).flatten(),
        cv2.normalize(hist_r, hist_r).flatten()])
    return np.concatenate([hist_gray, hist_rgb])
