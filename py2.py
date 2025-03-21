import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    raise Exception("Model file 'model.p' not found. Please ensure the model file exists in the correct directory.")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Define label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.predicted_character = "No hand detected"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        H, W, _ = img.shape
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                if len(data_aux) == 42:  # 21 landmarks * 2 (x, y)
                    prediction = model.predict([np.asarray(data_aux)])
                    self.predicted_character = labels_dict[int(prediction[0])]
                else:
                    self.predicted_character = "Invalid data length"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        else:
            self.predicted_character = "No hand detected"

        return img

def main():
    # Title and description
    st.title("Real-Time Sign Language Recognition")
    st.markdown("Show hand gestures (A, B, L) to the webcam")
    st.markdown("**Please click 'Start' to begin the webcam stream.**")

    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Create video transformer instance
    video_transformer = HandGestureTransformer()

    # Webcam feed with label
    st.markdown("**Live Feed**")
    webrtc_ctx = webrtc_streamer(
        key="hand-gesture-recognition",
        video_transformer_factory=lambda: video_transformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Prediction display with label
    st.markdown("**Predicted Gesture**")
    prediction_placeholder = st.empty()

    # Update prediction in real-time
    while True:
        if webrtc_ctx.state.playing:
            prediction_placeholder.text(video_transformer.predicted_character)
        else:
            prediction_placeholder.text("Waiting for webcam...")

if __name__ == "__main__":
    main()