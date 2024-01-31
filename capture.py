import argparse
import time
import cv2
import threading
import numpy as np
from utils import (yolo_detect, opencv_detect)
from utils.logger import Logger
from utils.shared_vars import (SharedFrame, SharedBoxes)

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-f', type=str, help='Cam or Video Path"', required=True)
parser.add_argument('--period', '-t', type=float, help='YOLO detection period (if not applied this value will be the '
                                                       'average processing time of the YOLO model)', default=0)
parser.add_argument('--htracking', help='Enable tracking in Horizontal', action="store_true")
parser.add_argument('--vtracking', help='Enable a tracking in Vertical', action="store_true")

args = parser.parse_args()

yolo_period = args.period
yolo_processing_times = []

VIDEO_PATH = args.path
HORIZONTAL_ENABLED = args.htracking
VERTICAL_ENABLED = args.vtracking

HORIZONTAL_EXPANSION = 50
VERTICAL_EXPANSION = 50
MAX_WIDTH_RESOLUTION = 1024
MAX_HEIGHT_RESOLUTION = 768

if not HORIZONTAL_ENABLED and not VERTICAL_ENABLED:
    raise Exception("At least 1 tracking must be enabled (--htracking or --vtracking)")

shared_frame = SharedFrame(value=None)
shared_boxes = SharedBoxes(value={})

log = Logger().getLogger("Capture")

yolo_condition = threading.Condition()
yolo_stop_flag = False


def preprocess_image(frame, template, cords):
    """
    Create a larger image and adjust a Template (Template Matching in OpenCV) by adjusting the coordinates of the bounding box.

    :param frame: The input image frame.
    :param template: The image used in the match template.
    :param cords: The coordinates (x_min, y_min, x_max, y_max) of the bounding box.
    :return: Tuple containing larger_image, adjusted coordinates (x_min_adjusted, y_min_adjusted, x_max_adjusted,
                y_max_adjusted).
    """

    height, width, _ = frame.shape

    x_min_adjusted = cords[0] - (HORIZONTAL_ENABLED * HORIZONTAL_EXPANSION)
    y_min_adjusted = cords[1] - (VERTICAL_ENABLED * VERTICAL_EXPANSION)

    x_max_adjusted = cords[2] + (HORIZONTAL_ENABLED * HORIZONTAL_EXPANSION)
    y_max_adjusted = cords[3] + (VERTICAL_ENABLED * VERTICAL_EXPANSION)

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

    if cords[0] == 0 and (template.shape[1] - 2) > 10 and HORIZONTAL_ENABLED:
        template = template[:, 2:]

    if cords[1] == 0 and (template.shape[0] - 2) > 10 and VERTICAL_ENABLED:
        template = template[2:, :]

    if cords[2] == width and (template.shape[1] - 2) > 10 and HORIZONTAL_ENABLED:
        template = template[:, :-2]

    if cords[3] == height and (template.shape[0] - 2) > 10 and VERTICAL_ENABLED:
        template = template[:-2, :]

    larger_image = frame[
                   y_min_adjusted: y_max_adjusted,
                   x_min_adjusted: x_max_adjusted
                   ]

    return (larger_image, template), (x_min_adjusted, y_min_adjusted)


def yolo_detection_thread():
    """
    Thread function for continuous YOLO object detection.

    This function runs in a loop, periodically performing YOLO detection on the shared video frame.
    The detected bounding boxes and processing time are updated in the shared data structures.
    """
    global yolo_period

    while not yolo_stop_flag:
        frame = shared_frame.value

        if frame is not None:
            boxes_dict, processing_time = yolo_detect.detect(frame)
            shared_boxes.value = boxes_dict

            if args.period == 0:
                yolo_processing_times.append(processing_time)

                yolo_period = sum(yolo_processing_times) / len(yolo_processing_times)
                log.info(f"YOLO Period Performed to {yolo_period}")

            with yolo_condition:
                yolo_condition.notify()

            time.sleep(yolo_period)


def opencv_detection(frame_copy, frame):
    """
    Detects objects in a video frame using OpenCV.

    :param frame: The input video frame.
    """

    for box in shared_boxes.value.values():
        cords = box['cords']  # x1, y1, x2, y2

        template_image = box['template_image']

        (larger_image, template_image), (x1, y1) = preprocess_image(frame, template_image, cords)

        box['template_image'] = template_image  # Reload Template Image

        opencv_result = opencv_detect.detect(larger_image, template_image)  # OpenCV Box

        if opencv_result is not None:
            (cv_x1, cv_y1, cv_x2, cv_y2), conf = opencv_result

            video_x1, video_y1, video_x2, video_y2 = (x1 + cv_x1), (y1 + cv_y1), (x1 + cv_x2), (y1 + cv_y2)  # video Box
            box['cords'] = [video_x1, video_y1, video_x2, video_y2]

            meio_x = int((video_x1 + video_x2) / 2)
            meio_y = int((video_y1 + video_y2) / 2)

            cv2.circle(frame_copy, (meio_x, meio_y), 2, (0, 255, 0), 2)

            cv2.rectangle(frame_copy, (video_x1, video_y1), (video_x2, video_y2), 255, 2)
            cv2.putText(frame_copy, f"{box['label']} - {'{:.2f}'.format(conf)}", (video_x1, video_y1),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                        1)


def load_video():
    """
        Loads a video from the specified path and adjusts its resolution if needed.

        :return: VideoCapture object if successful, otherwise None.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)

    width_video, height_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if cap.isOpened():
        log.info(f"Video Loaded: {VIDEO_PATH} {width_video}x{height_video}")

        if width_video > MAX_WIDTH_RESOLUTION or height_video > MAX_HEIGHT_RESOLUTION:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_WIDTH_RESOLUTION)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_HEIGHT_RESOLUTION)

            log.info(f"Resolution Adjusted to: {MAX_WIDTH_RESOLUTION}x{MAX_HEIGHT_RESOLUTION}")

        shared_frame.value = cap.read()[1]

        return cap

    log.error("Could not read frame.", exc_info=True)
    return None


def start_yolo_thread():
    thread = threading.Thread(target=yolo_detection_thread)

    thread.start()


def loading_image():
    image = cv2.imread('utils/loading.png')
    cv2.imshow("preview", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


def main():
    """
    Orchestrates the main execution flow for the video processing application.

    - Initializes YOLO, loads the video, and starts the YOLO thread.
    - Waits for YOLO initialization.
    - Enters the main loop for real-time video processing.
    - Performs object detection using OpenCV on each video frame.
    - Calculates and displays the Frames Per Second (FPS) on the processed frame.
    - Breaks the loop if the 'q' key is pressed, releasing the video and closing the display window.
    - Sets the YOLO stop flag when exiting the loop.
    """
    global yolo_stop_flag

    loading_image()

    cap = load_video()

    start_yolo_thread()

    with yolo_condition:
        yolo_condition.wait()

    if cap is not None:
        while True:
            frame_start_time = time.time()

            ret, frame = cap.read()

            shared_frame.value = frame

            if not ret:
                log.error("Could not read frame.", exc_info=True)
                break

            frame_copy = frame.copy()
            opencv_detection(frame_copy, frame)

            frame_end_time = time.time()
            fps = 1 / np.round(frame_end_time - frame_start_time, 2)
            frame_copy = cv2.putText(frame_copy, "FPS: {:.2f}".format(fps), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2,
                                     (0, 0, 0),
                                     2)

            cv2.imshow("preview", frame_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    yolo_stop_flag = True


if __name__ == "__main__":
    main()
