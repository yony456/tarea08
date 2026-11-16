#pip uninstall protobuf -y
#pip install protobuf==3.20.3
#pip install mediapipe==0.10.14

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)  # refine_landmarks=True detecta iris
mp_drawing = mp.solutions.drawing_utils

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def iris_center(landmarks, indices, image_shape):
    h, w = image_shape[:2]
    x = int(sum([landmarks[i].x for i in indices]) / len(indices) * w)
    y = int(sum([landmarks[i].y for i in indices]) / len(indices) * h)
    return (x, y)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibujar malla facial completa (opcional)
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )
            
            left_center = iris_center(face_landmarks.landmark, LEFT_IRIS, frame.shape)
            right_center = iris_center(face_landmarks.landmark, RIGHT_IRIS, frame.shape)
            
            cv2.circle(frame, left_center, 3, (0,0,255), -1)
            cv2.circle(frame, right_center, 3, (0,0,255), -1)
            
            cv2.putText(frame, f"L Iris: {left_center}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.putText(frame, f"R Iris: {right_center}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    cv2.imshow("Seguimiento de Iris", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
