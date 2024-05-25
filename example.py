import cv2
from VisionForge import Tracking

cap = cv2.VideoCapture('navio.mp4')

vf = Tracking("yolov8n.pt")
vf.yolo_confidence = 0.4
vf.match_template_confidence = 0.8
vf.euclidean_distance = 0.1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = vf.predict(frame)

    for result in results:
        video_x1, video_y1, video_x2, video_y2 = result['cords']

        cv2.rectangle(frame, (video_x1, video_y1), (video_x2, video_y2), 255, 2)
        cv2.rectangle(frame, (result['cords_yolo'][:2]), (result['cords_yolo'][2:]), (0, 255, 0), 2)
        cv2.rectangle(frame, (result['cords_larger'][:2]), (result['cords_larger'][2:]), (0, 0, 255), 2)

        cv2.putText(frame, f"{result['label']} - {'VF{:.2f}'.format(result['euclidean_intensity'])}",
                    (video_x1, video_y1),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                    1)

        cv2.putText(frame, f"{result['label']} - {'Y {:.2f}'.format(result['yolo_conf'])}",
                    result['cords_yolo'][:2],
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                    1)

        cv2.putText(frame, f"{result['label']} - {'L {:.2f}'.format(result['template_matching_conf'])}",
                    result['cords_larger'][:2],
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                    1)

    cv2.imshow("preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
