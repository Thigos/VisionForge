import cv2

CONFIDENCE = 0.5


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
        return (cv_x1, cv_y1, cv_x2, cv_y2), max_val
    else:
        return None
