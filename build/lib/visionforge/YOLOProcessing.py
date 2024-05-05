from vf import VisionForge


class YOLOProcessing:
    def __init__(self, vf: 'vf'):
        self.__visionForge = vf

        self.__yolo_in_ram = False

    def __call__(self, *args, **kwargs):
        return self.detection()

    def find_object(self, new_cords):
        """
        Find the object in the shared_boxes dictionary.
        :param new_cords: The new coordinates of the bounding box.
        :return: The dictionary containing the bounding box information.
        """
        for box in self.__visionForge.shared_boxes.values():
            larger_cords = self.__visionForge.generate_cords_large_template(box['cords'])

            if new_cords[0] >= larger_cords[0] and new_cords[1] >= larger_cords[1] and \
                    new_cords[2] <= larger_cords[2] and new_cords[3] <= larger_cords[3]:
                return box

        return []

    def detection(self):
        """
        Perform object detection using YOLO.
        :return: Dictionary containing the bounding box information.
        """
        model = self.__visionForge.model
        shared_frames = self.__visionForge.shared_frame
        self.__visionForge.last_yolo_shared_frame = shared_frames

        boxes_dict = {}

        result = model.predict(shared_frames, conf=self.__visionForge.yolo_confidence, verbose=False)[0]
        boxes = result.boxes

        for i, box in enumerate(boxes):
            cords = [int(x) for x in box.xyxy[0].tolist()]

            find_object = self.find_object(cords)

            if find_object:
                key = find_object['key']

                if key >= i:
                    boxes_dict[i] = find_object
                else:
                    box_temp = boxes_dict[key]
                    boxes_dict[key] = find_object
                    boxes_dict[i] = box_temp
                continue

            label = result.names[box.cls[0].item()]
            confidence = box.conf[0].item()

            boxes_dict[i] = {
                'key': i,
                'label': label,
                'confidence': confidence,
                'cords': cords,
                'original_cords_yolo': cords,
                'template_image': shared_frames[cords[1]: cords[3], cords[0]: cords[2]],
            }

        if not self.__yolo_in_ram:
            self.__yolo_in_ram = True

        return boxes_dict
