from time import time
from ultralytics import YOLO
from utils.logger import Logger

CONFIDENCE = 0.8
model = YOLO('../model/yolov8n.pt')
model_in_ram = False


def detect(frame):
    global model_in_ram

    boxes_dict = {}

    start_time = time()

    result = model.predict(frame, conf=CONFIDENCE)[0]
    boxes = result.boxes

    for i, box in enumerate(boxes):
        cords = [int(x) for x in box.xyxy[0].tolist()]
        label = result.names[box.cls[0].item()]
        confidence = box.conf[0].item()

        boxes_dict[i] = {
            'label': label,
            'confidence': confidence,
            'cords': cords,
            'template_image': frame[cords[1]: cords[3], cords[0]: cords[2]]
        }

    end_time = time()
    processing_time = end_time - start_time if model_in_ram else 0

    model_in_ram = True

    Logger().getLogger("YOLO-Detection").info(f"""
        Detection With YOLO
        Processing Time: {processing_time}s 
        Boxes: {len(boxes_dict)}
        """)

    return boxes_dict, processing_time
