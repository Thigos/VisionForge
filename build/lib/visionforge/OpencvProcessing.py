import cv2
import numpy as np

from VisionForge import Tracking


class OpencvProcessing:
    def __init__(self, vf: 'Tracking'):
        self.__visionForge = vf

    def __call__(self, frame, box):
        """
        Perform Template Matching using OpenCV.
        :param frame: The input image frame.
        :param box: The dictionary containing the bounding box information.
        :return: List of dictionaries containing the results.
        """
        cords = box['cords'].copy()
        template_image = box['template_image']

        ((larger_image, template),
         (x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted)) = self.__preprocess_image(frame, cords,
                                                                                                     template_image)
        box['template_image'] = template

        template_matching_result = self.detection(larger_image, template)

        if template_matching_result is not None:
            (cv_x1, cv_y1, cv_x2, cv_y2), (euclidean_intensity, template_matching_conf) = template_matching_result
            x1, y1, x2, y2 = ((x_min_adjusted + cv_x1),
                              (y_min_adjusted + cv_y1),
                              (x_min_adjusted + cv_x2),
                              (y_min_adjusted + cv_y2))

            box['cords'] = [x1, y1, x2, y2]

            return {
                'label': box['label'],
                'cords': [x1, y1, x2, y2],
                'cords_yolo': box['original_cords_yolo'],
                'cords_larger': [x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted],
                'euclidean_intensity': euclidean_intensity,
                'template_matching_conf': template_matching_conf,
                'yolo_conf': box['confidence']
            }

        return []

    def __preprocess_image(self, frame, cords, template):
        """
        Create a larger image and adjust a Template (Template Matching in OpenCV) by adjusting the coordinates of the bounding box.

        :param frame: The input image frame.
        :param template: The image used in the match template.
        :param cords: The coordinates (x_min, y_min, x_max, y_max) of the bounding box.
        :return: Tuple containing larger_image, adjusted coordinates (x_min_adjusted, y_min_adjusted, x_max_adjusted,
                    y_max_adjusted).
        """

        height, width, _ = frame.shape

        (x_min_adjusted, y_min_adjusted,
         x_max_adjusted, y_max_adjusted) = self.__visionForge.generate_cords_large_template(cords)

        if x_min_adjusted < 0:
            x_max_adjusted += x_min_adjusted
            x_min_adjusted = 0
        if y_min_adjusted < 0:
            y_max_adjusted += y_min_adjusted
            y_min_adjusted = 0

        if x_max_adjusted > width:
            diff = x_max_adjusted - width
            x_min_adjusted += diff
            x_max_adjusted = width
        if y_max_adjusted > height:
            diff = y_max_adjusted - height
            y_min_adjusted += diff
            y_max_adjusted = height

        if cords[0] == 0 and (template.shape[1] - 2) > 10 and self.__visionForge.horizontal_tracking:
            template = template[:, 2:]

        if cords[1] == 0 and (template.shape[0] - 2) > 10 and self.__visionForge.vertical_tracking:
            template = template[2:, :]

        if cords[2] == width and (template.shape[1] - 2) > 10 and self.__visionForge.horizontal_tracking:
            template = template[:, :-2]

        if cords[3] == height and (template.shape[0] - 2) > 10 and self.__visionForge.vertical_tracking:
            template = template[:-2, :]

        larger_image = frame[
                       y_min_adjusted: y_max_adjusted,
                       x_min_adjusted: x_max_adjusted
                       ]

        return (larger_image, template), (x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted)

    def detection(self, larger_image, template_image):
        """
        Perform Template Matching using OpenCV.
        :param larger_image: Template image large scale used in the match template.
        :param template_image: Template image used in the match template.
        :return: Tuple containing the coordinates of the bounding box and the similarity metrics.
        """
        larger_image_gray = cv2.cvtColor(larger_image, cv2.COLOR_BGR2GRAY)
        template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

        w_larger, h_larger = larger_image_gray.shape[::-1]
        w_template, h_template = template_image_gray.shape[::-1]

        resized_larger_image_gray = cv2.resize(larger_image_gray, (w_larger // 2, h_larger // 2))
        resized_template_image_gray = cv2.resize(template_image_gray, (w_template // 2, h_template // 2))

        res = cv2.matchTemplate(resized_larger_image_gray, resized_template_image_gray,
                                self.__visionForge.template_matching_method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        cv_x1, cv_y1 = max_loc
        cv_x1 *= 2
        cv_y1 *= 2

        cv_x2, cv_y2 = cv_x1 + w_template, cv_y1 + h_template

        if max_val >= self.__visionForge.match_template_confidence:
            hist_intensity1 = self.__calc_gray_histogram(template_image_gray)
            hist_intensity2 = self.__calc_gray_histogram(larger_image_gray[cv_y1:cv_y2, cv_x1:cv_x2])

            euclidean_intensity = self.__calculate_similarity(hist_intensity1, hist_intensity2)

            if euclidean_intensity <= self.__visionForge.euclidean_distance:
                return (cv_x1, cv_y1, cv_x2, cv_y2), (euclidean_intensity, max_val)

        return None

    def __calc_gray_histogram(self, img):
        """
        Calculate the histogram of the image.
        :param img: The input image.
        :return: The histogram of the image.
        """
        hist_intensity = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_intensity /= hist_intensity.sum()

        return hist_intensity

    def __calculate_similarity(self, hist1, hist2):
        return np.linalg.norm(hist1 - hist2)
