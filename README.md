# 🤟 GestuRec: Real-Time Sign Language Interpreter 🧠📷🖐️

> An AI-powered real-time sign language detection system that uses computer vision and ML to bridge communication gaps — deployed with Gradio for an interactive web experience.

🔗 [GitHub Repository](https://github.com/LakhaniChandresh0130/sign_language_detection.git)

---

## 📌 About

**GestuRec** is a deep learning-powered sign language interpreter designed to recognize and translate **50+ hand signs** into human-readable text. It utilizes **MediaPipe**, **OpenCV**, **CNNs**, **LSTM**, and **Random Forest** for gesture recognition and sequence modeling. The model is deployed via **Gradio**, allowing real-time interactive interpretation in a secure environment.

---

## ✨ Features

- ✋ Real-time hand sign detection & interpretation
- 🧠 Multi-model ML pipeline (CNN, LSTM, Random Forest)
- 🧾 NLP-based text translation
- 📺 Gradio UI for interactive access
- 🔐 HTTPS-secured deployment
- ⚙️ Jenkins-powered CI/CD for 60% faster deployment
- 🔄 Future ResNet upgrade planned for 3–4% accuracy gain

---

## 🧠 Resume Highlights

- 🤖 **Developed** a real-time sign language detection system using **OpenCV**, **MediaPipe**, **ML models** achieving **95% accuracy**
- 🚀 **Secured** HTTPS communication & reduced deployment time by **60%** using **Jenkins CI/CD**
- 🌐 **Deployed** the model via **Gradio**, enabling an interactive sign language interpreter for **50+ signs**
- 🔬 **Planned** CNN + ResNet integration pipeline for improved performance (**+3–4% accuracy** boost)

---

## 🛠 Tech Stack

### 🎥 Computer Vision
- **OpenCV**
- **MediaPipe** (Hand landmarks tracking)
- **Gradio** (Web UI for ML)

### 🧠 Machine Learning & Deep Learning
- **CNN** (Convolutional Neural Network)
- **LSTM** (Long Short-Term Memory)
- **Random Forest**
- **ResNet** (planned integration)

### 🗣️ NLP & Translation
- **Text Generation Pipeline**
- **Sign-to-text mappings**

### 🧱 Deployment & DevOps
- **HTTPS** for secure communication
- **Jenkins CI/CD** for automated deployment
- **Gradio** for frontend-hosted interface

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/LakhaniChandresh0130/sign_language_detection.git
cd sign_language_detection
```

### 2️⃣ (Optional) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Required Packages
```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the App
```bash
python app.py
```

### 5️⃣ Access the Interface
Go to the URL printed in terminal, usually:
```
http://127.0.0.1:7860/
```

---

## 🧪 How It Works

1. 📷 **Capture:** Live webcam input detects the user’s hand.
2. 🖐️ **Track:** MediaPipe identifies and maps hand landmarks.
3. 🧠 **Predict:** ML models (CNN, LSTM, RF) predict the sign.
4. 📝 **Translate:** Output is displayed as interpreted text.
5. 🖥️ **Display:** Shown in real-time via Gradio interface.

---

## 📸 Preview

_Add screenshots of the Gradio interface or detection preview here._

---

## 📁 Project Structure

```
sign_language_detection/
├── app.py                # Main app logic
├── model/                # Trained ML models
├── utils/                # Helper functions for prediction
├── requirements.txt      # Dependencies
├── README.md
└── assets/               # Optional: Screenshots, media
```

---

## 🧱 Planned Enhancements

- 🔄 Integration of **ResNet** for improved robustness (+3–4% accuracy)
- 🌐 Multi-language translation via NLP
- 📊 Visualization dashboard of accuracy per sign

---

## 🙌 Contributing

Feel free to fork this repo, open pull requests, and contribute to improving sign language accessibility! 💡

Steps:
1. 🍴 Fork
2. 🛠 Create feature branch
3. ✅ Commit your changes
4. 📤 Push and open PR

---

## 👨‍💻 Authors

Built with ❤️ by [Janak MMakadia](https://github.com/janak-makadia345)

---

## 📬 Contact

📧 makadiask901@gmail.com  
🌐 [LinkedIn](https://www.linkedin.com/in/janak-makadia)

