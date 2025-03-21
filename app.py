import pickle
import cv2
import mediapipe as mp
import numpy as np
import gradio as gr



# Load the model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    raise Exception("Model file 'model.p' not found.")

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Define label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'L',3: 'C', 4: 'D'}

def process_frame(frame):
    frame = cv2.resize(frame, (640, 480))
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    predicted_character = "No hand detected"
    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
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

            prediction_proba = model.predict_proba([np.asarray(data_aux)])[0]
            prediction = model.predict([np.asarray(data_aux)])
        
            predicted_character = labels_dict[int(prediction[0])]
            confidence = np.max(prediction_proba) * 100  # Convert to percentage

            # Display prediction and confidence
            display_text = f"{predicted_character} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, predicted_character

def webcam_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, prediction = process_frame(frame)
        yield processed_frame, prediction
    
    cap.release()

# Define Gradio interface with single webcam feed and prediction below
with gr.Blocks(title="Hand Gesture Recognition") as interface:
    gr.Markdown("# Real-Time sign language Recognition")
    gr.Markdown("Show hand gestures (A, B, L) to the webcam")
    
    webcam_output = gr.Image(label="Live Feed", streaming=True)
    text_output = gr.Textbox(label="Predicted Gesture", interactive=False)
    
    gr.Interface(fn=webcam_feed, 
                inputs=None, 
                outputs=[webcam_output, text_output], 
                live=True)


# Launch with HTTPS and the custom client
<<<<<<< HEAD
# Launch with HTTPS
interface.launch(server_name="127.0.0.1",
    server_port=7860,
    ssl_certfile="cert.pem",
    ssl_keyfile="key.pem",ssl_verify=False,share=True, debug=True)
=======
interface.launch(server_name="0.0.0.0",server_port=7860, share=True)
>>>>>>> 6ceddd83971cb1d0de10de3edb24e33dcd545fdf
