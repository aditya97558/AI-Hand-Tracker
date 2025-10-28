import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    lmlist = []

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((id, cx, cy))
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    if lmlist:
        fingers = []
        # Thumb
        if lmlist[tip_ids[0]][1] > lmlist[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers (index to pinky)
        for id in range(1, 5):
            if lmlist[tip_ids[id]][2] < lmlist[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total_fingers = fingers.count(1)

        cv2.putText(img, f'Fingers: {total_fingers}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
