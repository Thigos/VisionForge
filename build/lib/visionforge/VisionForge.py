import threading
from pathlib import Path
from typing import Union

import cv2
import torch
from ultralytics import YOLO
from .OpencvProcessing import OpencvProcessing
from .YOLOProcessing import YOLOProcessing
from .Exceptions import FrameNotFound


class VisionForge:
    shared_frame = None
    last_yolo_shared_frame = None
    shared_boxes = {}

    def __init__(self, model_path: Union[str, Path] = "yolov8n.pt"):
        """
        Initialize the vf object.
        :param model_path: The path to the YOLO model.
        :type model_path: str or Path
        """
        self.horizontal_tracking = True
        self.vertical_tracking = True
        self.horizontal_expansion = 100
        self.vertical_expansion = 100

        self.max_resolution = (1024, 768)
        self.original_resolution = (0, 0)

        self.yolo_confidence = 0.4
        self.match_template_confidence = 0.1
        self.euclidean_distance = 0.2

        self.thread_yolo_condition = threading.Condition()
        self.thread_yolo = threading.Thread(target=self.start_tracking_thread)
        self.__thread_yolo_stop_flag = False

        self.__model_path = str(model_path).strip()

        self.template_matching_method = cv2.TM_CCORR_NORMED

        self.__opencvProcessing = OpencvProcessing(self)
        self.__yoloProcessing = YOLOProcessing(self)

        self.__init_yolo_model()

    def __init_yolo_model(self):
        """
        Initialize the YOLO model.
        """
        model = YOLO(self.__model_path)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        self.model = model

    def start_tracking_thread(self):
        """
        Start the tracking thread.
        """
        while not self.__thread_yolo_stop_flag:
            boxes_dict = self.__yoloProcessing()
            VisionForge.shared_boxes = boxes_dict

            with self.thread_yolo_condition:
                self.thread_yolo_condition.notify()

    def predict(self, frame=None, conf: float = 0.4):
        """
        Predict the objects in the frame.
        :param frame: The input frame.
        :param conf: Yolo confidence.
        :return: List of dictionaries containing the results.
        """
        if frame is None:
            raise FrameNotFound("Error: Frame Not Found")

        self.yolo_confidence = conf

        original_frame_copy = frame.copy()
        frame_resized = original_frame_copy

        self.original_resolution = (original_frame_copy.shape[1], original_frame_copy.shape[0])

        VisionForge.shared_frame = frame_resized

        if not self.thread_yolo.is_alive() and not self.__thread_yolo_stop_flag:
            self.thread_yolo.start()

            with self.thread_yolo_condition:
                self.thread_yolo_condition.wait()

        results = []

        for box in VisionForge.shared_boxes.values():
            result = self.__opencvProcessing(frame_resized, box)

            if result:
                results.append(result)

        return results

    def __generate_cords_large_template(self, cords):
        """
        Generate the coordinates of a larger template.
        :param cords: The coordinates of the bounding box.
        :return: The adjusted coordinates of the bounding box.
        """
        x_min_adjusted = cords[0] - (self.horizontal_tracking * self.horizontal_expansion)
        y_min_adjusted = cords[1] - (self.vertical_tracking * self.vertical_expansion)

        x_max_adjusted = cords[2] + (self.horizontal_tracking * self.horizontal_expansion)
        y_max_adjusted = cords[3] + (self.vertical_tracking * self.vertical_expansion)

        return x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted
