
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time

GESTURES = ['hola', 'gracias', 'bien', 'comer', 'como estas']
SEQUENCE_LENGTH = 30
modelo = load_model('modelo_lsm.h5')
voz = pyttsx3.init()
voz.setProperty('rate', 150)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=3, min_detection_confidence=0.7)

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        all_hands = []
        for hand in results.multi_hand_landmarks:
            keypoints = []
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            all_hands.append(keypoints)
        if len(all_hands) == 1:
            all_hands.append([0]*63)
        return np.array(all_hands).flatten()
    else:
        return np.zeros(126)

st.title("🖐️ Traductor LSM en Streamlit")
st.markdown("Traducción de señas en vivo usando cámara y modelo LSM.")

frame_window = st.image([])
text_placeholder = st.empty()

run = st.checkbox('Iniciar cámara')

sequence = []
predicciones = []
ultima_salida = ""
tiempo_ultima_salida = 0

cap = cv2.VideoCapture(1)

while run:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:]

    salida = "---"

    if len(sequence) == SEQUENCE_LENGTH:
        res = modelo.predict(np.expand_dims(sequence, axis=0))[0]
        pred = np.argmax(res)
        conf = res[pred]

        if conf > 0.9:
            predicciones.append(pred)
            predicciones = predicciones[-5:]

            if predicciones.count(pred) == 5:
                salida = GESTURES[pred]
                if salida != ultima_salida or (time.time() - tiempo_ultima_salida) > 2:
                    ultima_salida = salida
                    tiempo_ultima_salida = time.time()
                    voz.say(salida)
                    voz.runAndWait()

    text_placeholder.markdown(f"Seña reconocida: `{salida}`")
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(img)

cap.release()
