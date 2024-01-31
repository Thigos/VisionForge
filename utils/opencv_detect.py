import cv2
import numpy as np

CONFIDENCE = 0.5
EUCLIDEAN_DISTANCE = 0.1


def detect(larger_image, template_image):
    larger_image_gray = cv2.cvtColor(larger_image, cv2.COLOR_BGR2GRAY)
    template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    w_larger, h_larger = larger_image_gray.shape[::-1]
    w_template, h_template = template_image_gray.shape[::-1]

    resized_larger_image_gray = cv2.resize(larger_image_gray, (w_larger // 2, h_larger // 2))
    resized_template_image_gray = cv2.resize(template_image_gray, (w_template // 2, h_template // 2))

    res = cv2.matchTemplate(resized_larger_image_gray, resized_template_image_gray, cv2.TM_CCORR_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    cv_x1, cv_y1 = max_loc
    cv_x1 *= 2
    cv_y1 *= 2

    cv_x2, cv_y2 = cv_x1 + w_template, cv_y1 + h_template

    if max_val >= CONFIDENCE:
        hist_intensity1 = calc_gray_histogram(template_image_gray)
        hist_intensity2 = calc_gray_histogram(larger_image_gray[cv_y1:cv_y2, cv_x1:cv_x2])

        euclidean_intensity = calculate_similarity(hist_intensity1, hist_intensity2)

        if euclidean_intensity <= EUCLIDEAN_DISTANCE:
            return (cv_x1, cv_y1, cv_x2, cv_y2), euclidean_intensity
        else:
            return None
    else:
        return None


def calc_gray_histogram(img):
    hist_intensity = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_intensity /= hist_intensity.sum()

    return hist_intensity


def calculate_similarity(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)
