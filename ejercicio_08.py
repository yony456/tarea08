import cv2
import mediapipe  as mp
mp_face_mesh= mp.solutions.face_mesh# Obtenemos los modelos de reconocimiento facial
model= mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True)
mp_drawing= mp.solutions.drawing_utils #herramienta para dibujar el rostro en 3D
ojo_izquierdo=[33,133]
ojo_derecho=[263,362]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
def iris_center(landmarks, indices, image_shape):
    h, w = image_shape[:2]
    x = int(sum([landmarks[i].x for i in indices]) / len(indices) * w)
    y = int(sum([landmarks[i].y for i in indices]) / len(indices) * h)
    return (x, y)
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

                left_center = iris_center(face_landmarks.landmark, LEFT_IRIS, frame.shape)
                cv2.circle(frame, left_center, 3, (0,255,0), -1)

                inner_left= (int(face_landmarks.landmark[ojo_izquierdo[1]].x * w),
                            int(face_landmarks.landmark[ojo_izquierdo[1]].y * h))
                cv2.circle(frame, inner_left, 3, (0,0,255), -1)


        #---ojo derecho---- 
    
                inner_right= (int(face_landmarks.landmark[ojo_derecho[0]].x * w),
                            int(face_landmarks.landmark[ojo_derecho[0]].y * h))
                cv2.circle(frame, inner_right, 3, (0,0,255), -1)

                right_center = iris_center(face_landmarks.landmark, RIGHT_IRIS, frame.shape)
                cv2.circle(frame, right_center, 3, (0,255,0), -1)
        
                inner_right= (int(face_landmarks.landmark[ojo_derecho[1]].x * w),
                            int(face_landmarks.landmark[ojo_derecho[1]].y * h))
                cv2.circle(frame, inner_right, 3, (0,0,255), -1)
                cv2.putText(frame, f"L Iris: {left_center}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                cv2.putText(frame, f"R Iris: {right_center}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.imshow('MediaPipe', frame)
    if cv2.waitKey(1)==ord('q'): break
cap.release()#Liberamos la lectura de la camara
cv2.destroyAllWindows()            