import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("corriendo.mp4")
while True:
    ret, frame= cap.read()
    frame= cv2.resize(frame, (640, 480))
    resultados= model(frame)[0]
    frame= resultados.plot()
    cv2.imshow('', frame)
    if cv2.waitKey(50)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    