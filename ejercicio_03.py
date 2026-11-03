import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("caminando.mp4")
linea=500
contador=0
posiciones={}
while True:
    ret, frame= cap.read()
    if not ret:
        break
    frame= cv2.resize(frame, (640, 480))
    resultados= model.track(frame, persist=True, classes=[0], verbose=False)[0]
    cv2.line(frame, (linea,0), (linea,480), (0,0,255), 2)
    if resultados.boxes.id is not None:
        for caja, id in zip(resultados.boxes.xyxy, resultados.boxes.id):
            x1,  y1, x2, y2 = map(int, caja) #limites de caja
            id = int(id)#ID de caja
            centros=(x1+x2)//2 #centro horisontal 
            if id in posiciones and posiciones[id] < linea<=centros:
                contador+=1
            posiciones[id]=centros
            #print(contador)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame,(centros, (y1+y2)//2), 4, (255,0,0), -1)
    cv2.putText(frame, 'contador:'+str(contador), (20,40), 0, 1,(255,255,255), 2)        
    cv2.imshow('', frame)
    if cv2.waitKey(50)==ord('q'):
        #print(resultados.boxes.id)
        break
cap.release()
cv2.destroyAllWindows()        