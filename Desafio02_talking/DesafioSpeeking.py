#Code adapted from van Gent, P. (2016).
# Emotion Recognition Using Facial Landmarks, Python, DLib and OpenCV. A tech blog about fun things with Python and embedded electronics.
# Retrieved from: http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
#Adapted by @SuzanaMota
#Import required modules
import cv2
import dlib
import imutils

#Dlib positions
#  ("mouth", (49, 68)),
value_removeLandsMakrs = 48

import math 
# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf [1]

def compute_mouth_open_3pontos(shape): #função para pegar os pontos especificos da boca e calcular a distancia euclidiana e o ratio
    mouth_left = [49-value_removeLandsMakrs,59-value_removeLandsMakrs] #posição 49 referente a canto esquerdo superior e 59 canto esquerdo inferior...
    mouth_midle = [51-value_removeLandsMakrs,57-value_removeLandsMakrs]
    mouth_right = [53-value_removeLandsMakrs,55-value_removeLandsMakrs]
    mouth_hor = [60-value_removeLandsMakrs,64-value_removeLandsMakrs] #pontos extremos horizontalmente (60 esquerda, 64 direita)
    distance_mouth_left  = math.sqrt((shape.part(mouth_left[0]).x  - shape.part(mouth_left[1]).x)**2 +  
                                     (shape.part(mouth_left[0]).y  - shape.part(mouth_left[1]).y)**2 )
    distance_mouth_midle = math.sqrt((shape.part(mouth_midle[0]).x - shape.part(mouth_midle[1]).x)**2 +  
                                     (shape.part(mouth_midle[0]).y - shape.part(mouth_midle[1]).y)**2 )
    distance_mouth_right = math.sqrt((shape.part(mouth_right[0]).x - shape.part(mouth_right[1]).x)**2 +  
                                     (shape.part(mouth_right[0]).y - shape.part(mouth_left[1]).y)**2 )
    distance_mouth_hor   = math.sqrt((shape.part(mouth_hor[0]).x   - shape.part(mouth_hor[1]).x)**2 +  
                                     (shape.part(mouth_hor[0]).y   - shape.part(mouth_hor[1]).y)**2 )
    
    aspect_ratio = (distance_mouth_left + distance_mouth_midle + distance_mouth_right) / (3.0 * distance_mouth_hor)
    
    return aspect_ratio

#Set up some required objects
#video_capture = cv2.VideoCapture(0) #Webcam object
video_capture = cv2.VideoCapture("people-talking.mp4") #Webcam object
#Change Frame Rate
detector = dlib.get_frontal_face_detector() #Face detector
#Landmark identifier. Set the filename to whatever you named the downloaded file
predictor = dlib.shape_predictor("shapemouth2.dat") 
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
frame_width = 640
frame_height = 360
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while True:
    ret, frame = video_capture.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 1) #Detectar mais de uma face.
    for k,d in enumerate(detections): #Para cada face 
        shape = predictor(gray, d)  #Get coordinates
        distance_month = compute_mouth_open_3pontos(shape) #Função para calcular o MAR (mouth aspect ratio) como artigo [1]
        cv2.putText(frame, str(distance_month), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        for i in range(48-value_removeLandsMakrs,68-value_removeLandsMakrs): #Dlib positions (48 até 67 são pontos referentes a boca)
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0,255,0), thickness=-1) #desenta 1 circulo em cada ponto da dlib para a boca.
        if distance_month >= 0.35: #se o ratio maior que .35 então esta com a boca aberta. (Valor baseado em testes)
            cv2.putText(frame, "Falando!", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    out.write(frame)
    cv2.imshow("frame", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()