import cv2
import mediapipe as mp
import numpy as np
import time 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Curl counter variables
counterSquat = 0 
counterJump = 0
stageSquat = None
stageJump = None

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

cap = cv2.VideoCapture(0)

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:   
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            right_hip = [landmarks[23].x,landmarks[23].y]
            right_knee = [landmarks[25].x,landmarks[25].y]
            right_ankle = [landmarks[27].x,landmarks[27].y]

            left_hip = [landmarks[24].x,landmarks[24].y]
            left_knee = [landmarks[26].x,landmarks[26].y]
            left_ankle = [landmarks[28].x,landmarks[28].y]

            right_elbow = [landmarks[13].x,landmarks[13].y]
            right_shoulder = [landmarks[11].x,landmarks[11].y]

            
            # Calculate angle
            angle1 = calculate_angle(right_hip, right_knee, right_ankle) #angle to detect squat
            #angle2 = calculate_angle(left_hip, left_knee, left_ankle) #angle to detect squat
            angle3 = calculate_angle(right_elbow, right_shoulder, right_hip) #angle to detect jumping jacks 


            cv2.putText(image, str(angle1), 
                           tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(angle3), 
                           tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            time.sleep(0.1)

            # Logic for squat
            if angle1 < 160: 
                stageSquat = "down"
            if stageSquat == "down" and angle1 > 160:  
                counterSquat += 1
                stageSquat = "up" 

            #Logic for jumping jack
            if angle3 > 145: 
                stageJump = "up"
            if stageJump == "up" and angle3 < 90: 
                counterJump += 1
                stageJump = "down"

        except:
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'SQUAT', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counterSquat), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'JUMP', (105,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        cv2.putText(image, str(counterJump), 
                    (100,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()