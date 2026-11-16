import cv2
import mediapipe  as mp
mp_face_mesh= mp.solutions.face_mesh# Obtenemos los modelos de reconocimiento facial
model= mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing= mp.solutions.drawing_utils #herramienta para dibujar el rostro en 3D
ojo_izquierdo=[33,133]
ojo_derecho=[263,362]
centro=[9]
alto=[10,152]
ancho=[234,454]
cap= cv2.VideoCapture(0)#OBTENEMOS LA CAMARA
while True:
    ret, frame= cap.read()#LECTURA DE LOS FRAMES
    #---inicio de flujo mediapipe___
    rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#CAmbiamos A RGB PARA MEDIAPIPE
    results= model.process(rgb_frame)#Ejecutamos el modelo sobre nuestro frame rgb
    if results.multi_face_landmarks:
        h,w,_=frame.shape#Tensor 3D (la forma del fotograma)
        #---ojo isquierdo---
        for face_landmarks in results.multi_face_landmarks:# Iteramos sobre los puntos de referencia
                inner_left= (int(face_landmarks.landmark[ojo_izquierdo[0]].x * w),
                             int(face_landmarks.landmark[ojo_izquierdo[0]].y * h))
                cv2.circle(frame, inner_left, 3, (0,0,255), -1)

                inner_left= (int(face_landmarks.landmark[ojo_izquierdo[1]].x * w),
                            int(face_landmarks.landmark[ojo_izquierdo[1]].y * h))
                cv2.circle(frame, inner_left, 3, (0,0,255), -1)
        #---ojo derecho---- 
    
                inner_right= (int(face_landmarks.landmark[ojo_derecho[0]].x * w),
                            int(face_landmarks.landmark[ojo_derecho[0]].y * h))
                cv2.circle(frame, inner_right, 3, (0,0,255), -1)
        
                inner_right= (int(face_landmarks.landmark[ojo_derecho[1]].x * w),
                            int(face_landmarks.landmark[ojo_derecho[1]].y * h))
                cv2.circle(frame, inner_right, 3, (0,0,255), -1)   
        # tratando hacer el medio para realisar un tiro 
                inner=(int(face_landmarks.landmark[centro[0]].x * w ),
                         int(face_landmarks.landmark[centro[0]].y * h))
                cv2.circle(frame, inner, 3, (0,255,0), -1)    
        # ----traso de altura----
                inner_h= (int(face_landmarks.landmark[alto[0]].x * w),
                              int(face_landmarks.landmark[alto[0]].y * h))
                cv2.circle(frame, inner_h, 3, (0,255,0), -1)
                        
                inner_h= (int(face_landmarks.landmark[alto[1]].x * w),
                              int(face_landmarks.landmark[alto[1]].y * h))
                cv2.circle(frame, inner_h, 3, (0,255,0), -1) 
        # ----traso de ancho----
                inner_W= (int(face_landmarks.landmark[ancho[0]].x * w),
                          int(face_landmarks.landmark[ancho[0]].y * h))
                cv2.circle(frame, inner_W, 3, (0,255,0), -1)
                       
                inner_w= (int(face_landmarks.landmark[ancho[1]].x * w),
                          int(face_landmarks.landmark[ancho[1]].y * h))
                cv2.circle(frame, inner_W, 3, (0,255,0), -1) 
    #---fin de flujo mediapipe___

    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1)==ord('q'): break
cap.release()#Liberamos la lectura de la camara
cv2.destroyAllWindows()