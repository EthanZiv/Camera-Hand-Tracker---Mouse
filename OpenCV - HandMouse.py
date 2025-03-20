import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

screen_width = 1920
screen_height = 1080

screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break
    
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            wrist_x = int(wrist.x * screen_width)
            wrist_y = int(wrist.y * screen_height)

            index_finger_x = int(index_finger_tip.x * screen_width)
            index_finger_y = int(index_finger_tip.y * screen_height)

            middle_finger_x = int(middle_finger_tip.x * screen_width)
            middle_finger_y = int(middle_finger_tip.y * screen_height)

            pyautogui.moveTo(wrist_x, wrist_y)

            if middle_finger_y < index_finger_y: 
                pyautogui.click()
                click_state = True

            elif middle_finger_y > index_finger_y:
                click_state = False

            cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 255, 0), -1)  
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 0, 255), -1)  
    
    cv2.imshow('Hand Tracking - Mouse Control', frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()  

