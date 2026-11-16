import cv2
import mediapipe  as mp

mp_face_mesh= mp.solutions.face_mesh# Obtenemos los modelos de reconocimiento facial
model= mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing= mp.solutions.drawing_utils #herramienta para dibujar el rostro en 3D
cap= cv2.VideoCapture(0)#OBTENEMOS LA CAMARA

while True:
    ret, frame= cap.read()#LECTURA DE LOS FRAMES
    #---inicio de flujo mediapipe___
    rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#CAmbiamos A RGB PARA MEDIAPIPE
    results= model.process(rgb_frame)#Ejecutamos el modelo sobre nuestro frame rgb
    if results.multi_face_landmarks:
        h,w,_=frame.shape#Tensor 3D (la forma del fotograma)
        for face_landmarks in results.multi_face_landmarks:# Iteramos sobre los puntos de referencia
            for i, lm in enumerate(face_landmarks.landmark):#optener loa valores de los púntos de referencia
                x, y = int(lm.x * w), int(lm.y * h) #obtener la ubicacion 2D de cada valor
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)
    #---fin de flujo mediapipe___

    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1)==ord('q'): break
cap.release()#Liberamos la lectura de la camara
cv2.destroyAllWindows()