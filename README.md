# Hand Gesture Recognition Using Python & MediaPipe

A real-time hand gesture recognition project built with **Python, OpenCV, and MediaPipe**. The system uses a webcam to detect hand landmarks and recognize predefined hand gestures, enabling intuitive, vision-based human-computer interaction.

## 🚀 Features

- **Real-Time Detection:** Detects hands through a live webcam feed.
- **21 Hand Landmarks:** Uses MediaPipe to identify key points on the hand.
- **Gesture Recognition:** Recognizes 10 predefined hand gestures.
- **Visual Feedback:** Displays hand landmarks and gesture information on the video feed.
- **Lightweight Implementation:** Uses Python-based computer vision tools.
- **Interactive Experience:** Demonstrates touch-free interaction through hand movements.

## 🤟 Supported Hand Gestures

The project supports the following 10 gestures:

| No. | Gesture |
|---:|---|
| 1 | Fist ✊ |
| 2 | Open Palm ✋ |
| 3 | Point ☝️ |
| 4 | Peace ✌️ |
| 5 | Three Fingers |
| 6 | Four Fingers |
| 7 | Thumbs Up 👍 |
| 8 | Call Me 🤙 |
| 9 | Rock 🤘 |
| 10 | OK 👌 |

## 🛠️ Technologies Used

- **Python** — Core programming language
- **OpenCV** — Webcam access and image processing
- **MediaPipe** — Hand detection and 21-landmark tracking

## ⚙️ How It Works

1. The webcam captures live video frames.
2. OpenCV processes the video stream.
3. MediaPipe detects the hand and identifies its 21 landmarks.
4. The gesture recognition logic analyzes the detected landmarks.
5. The recognized gesture is displayed on the video feed.

## 📁 Project Structure

```text
Hand_gesture/
│
├── hand_gesture.py    # Main hand gesture recognition program
├── test_gesture.py    # Gesture testing script
├── README.md          # Project documentation
└── .gitignore         # Files excluded from Git
```

## 💻 Getting Started

### Prerequisites

- Python 3.9 or a version compatible with your installed MediaPipe release
- A working webcam
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/vaishbyte/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install opencv-python mediapipe
```

### 4. Run the Application

```bash
python hand_gesture.py
```

Allow webcam access if prompted. Keep your hand visible in the camera frame and try the supported gestures.

> **Note:** If your Python version is incompatible with the installed MediaPipe release, use a supported Python version and reinstall the dependencies.

## 🎯 Applications

- Touch-free human-computer interaction
- Computer vision learning and experimentation
- Gesture-controlled interfaces
- Educational AI and robotics demonstrations
- Foundation for accessibility-focused interaction systems

## 🔮 Future Improvements

- Add support for more hand gestures.
- Improve recognition under different lighting conditions.
- Add gesture-based keyboard or mouse controls.
- Integrate gesture recognition with robotics and IoT projects.
- Develop a graphical user interface for easier interaction.

## 👩‍💻 Author

**Vaishnavi Jaiswal**

- GitHub: [@vaishbyte](https://github.com/vaishbyte)
- Project Repository: [Hand Gesture Recognition](https://github.com/vaishbyte/hand-gesture-recognition)

---

⭐ If you find this project useful, consider giving the repository a star!

*Built with Python, computer vision, and a passion for AI.*
