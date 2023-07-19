import mediapipe as mp 
import cv2
import os
import matplotlib.pyplot as plt
import pickle 


mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

hands  = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

data_address = 'C:\Python programming\HandGestureDetection\CollectedData'


data = []
labels = []

for gesture in os.listdir(data_address):      #listdir takes a path as input and returns a list containing the names of the entries in the directory. 
    for imgp in os.listdir(os.path.join(data_address, gesture)):

        data_aux = []

        imgg = cv2.imread(os.path.join(data_address, gesture, imgp))
        #converting image to RGB
        imgg_rgb = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)

        results = hands.process(imgg_rgb)
       

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawings.draw_landmarks(        #Function for showing Landmarks on hands!
                #     imgg_rgb,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )

                for i in range(len(hand_landmarks.landmark)):   #storing X and y coordinates of the landmarks
                    x = hand_landmarks.landmark[i].x   
                    y = hand_landmarks.landmark[i].y 

                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(gesture)

#         plt.figure()
#         plt.imshow(imgg_rgb)

# plt.show()

f = open('C:\Python programming\HandGestureDetection\pikolo.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()