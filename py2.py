import pickle
import cv2
import mediapipe as mp
import numpy as np
import gradio as gr
from pymongo import MongoClient
import datetime
import hashlib

# MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')
db = client['signlanguage']
users_collection = db['users']

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
labels_dict = {0: 'A', 1: 'B', 2: 'L', 3: 'C', 4: 'D'}

# Global variables
output_string = ""
current_user = None

def hash_password(password):
    """Hash the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def process_frame(frame):
    global output_string
    
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
            confidence = np.max(prediction_proba) * 100

            if confidence > 50:
                if not output_string or output_string[-1] != predicted_character:
                    output_string += predicted_character

            display_text = f"{predicted_character} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, predicted_character

def webcam_feed():
    global current_user
    if not current_user:
        raise gr.Error("Please login first!")
    
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
        yield processed_frame, prediction, output_string
    
    cap.release()

def signup(username, password):
    global current_user
    
    if not username or not password:
        return "Please enter both username and password", "home"
    
    # Check if user already exists
    if users_collection.find_one({"username": username}):
        return "Username already exists! Please choose another.", "home"
    
    # Create new user
    hashed_password = hash_password(password)
    new_user = {
        "username": username,
        "password": hashed_password,
        "created_at": datetime.datetime.now()
    }
    result = users_collection.insert_one(new_user)
    current_user = {"id": str(result.inserted_id), "username": username}
    return f"Signup successful! Welcome, {username}", "recognition"

def login(username, password):
    global current_user
    
    if not username or not password:
        return "Please enter both username and password", "home"
    
    # Check if user exists and password matches
    user = users_collection.find_one({"username": username})
    if not user:
        return "User not found! Please sign up first.", "home"
    
    hashed_password = hash_password(password)
    if user["password"] != hashed_password:
        return "Incorrect password!", "home"
    
    current_user = {"id": str(user["_id"]), "username": user["username"]}
    return f"Login successful! Welcome back, {username}", "recognition"

# Gradio Interface
with gr.Blocks(title="Sign Language Recognition System") as interface:
    page_state = gr.State(value="home")
    
    with gr.Column(visible=True) as home_page:
        gr.Markdown("# Welcome to Sign Language Recognition System")
        gr.Markdown("Translate hand gestures to text in real-time")
        
        with gr.Row():
            # Signup Section
            with gr.Column():
                gr.Markdown("## Sign Up")
                signup_username = gr.Textbox(label="Username", placeholder="Enter username")
                signup_password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                signup_btn = gr.Button("Sign Up")
            
            # Login Section
            with gr.Column():
                gr.Markdown("## Login")
                login_username = gr.Textbox(label="Username", placeholder="Enter username")
                login_password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                login_btn = gr.Button("Login")
        
        status_output = gr.Textbox(label="Status", interactive=False)
    
    with gr.Column(visible=False) as recognition_page:
        gr.Markdown("# Real-Time Sign Language Recognition")
        user_display = gr.Markdown(f"Current User: {current_user['username'] if current_user else 'Not logged in'}")
        gr.Markdown("Show hand gestures (A, B, L, C, D) to the webcam")
        
        webcam_output = gr.Image(label="Live Feed", streaming=True)
        text_output = gr.Textbox(label="Predicted Gesture", interactive=False)
        string_output = gr.Textbox(label="Output String", interactive=False)
        
        gr.Interface(
            fn=webcam_feed,
            inputs=None,
            outputs=[webcam_output, text_output, string_output],
            live=True
        )

    def handle_signup(username, password, current_page):
        message, new_page = signup(username, password)
        return (
            message,
            gr.update(visible=False) if new_page == "recognition" else gr.update(visible=True),
            gr.update(visible=True) if new_page == "recognition" else gr.update(visible=False),
            new_page,
            f"Current User: {current_user['username']}" if current_user else "Current User: Not logged in"
        )
    
    def handle_login(username, password, current_page):
        message, new_page = login(username, password)
        return (
            message,
            gr.update(visible=False) if new_page == "recognition" else gr.update(visible=True),
            gr.update(visible=True) if new_page == "recognition" else gr.update(visible=False),
            new_page,
            f"Current User: {current_user['username']}" if current_user else "Current User: Not logged in"
        )

    signup_btn.click(
        fn=handle_signup,
        inputs=[signup_username, signup_password, page_state],
        outputs=[status_output, home_page, recognition_page, page_state, user_display]
    )

    login_btn.click(
        fn=handle_login,
        inputs=[login_username, login_password, page_state],
        outputs=[status_output, home_page, recognition_page, page_state, user_display]
    )

# Launch with HTTPS
interface.launch(
    server_name="127.0.0.1",
    server_port=7860,
    ssl_certfile="cert.pem",
    ssl_keyfile="key.pem",
    ssl_verify=False,
    share=True,
    debug=True
)