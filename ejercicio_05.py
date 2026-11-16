import cv2

cap = cv2.VideoCapture('corriendo.mp4')
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    cv2.line(frame, (500,10), (500,400), (0, 0, 255), 2)
    cv2.imshow('', frame)
    if cv2.waitKey(50) == ord('q'): break
cap.release()
cv2.destroyAllWindows()