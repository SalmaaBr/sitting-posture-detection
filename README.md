# 🪑 Sitting Posture Detection

## 📌 Project Overview

**Sitting Posture Detection** is a computer vision project.

The main objective of this project is to develop an intelligent system capable of detecting and analyzing a person's sitting posture using computer vision and machine learning techniques.

The system aims to identify incorrect sitting positions and help users become more aware of their posture, contributing to better ergonomic habits and reducing the risks associated with prolonged improper sitting.

## 🎯 Objectives

- Detect human body posture while sitting.
- Analyze body positioning using computer vision techniques.
- Identify incorrect or unhealthy sitting postures.
- Provide a foundation for real-time posture monitoring.
- Explore the use of machine learning for automated posture classification.

## ✨ Key Features

- **Human Pose Detection:** Detect body landmarks and key points.
- **Posture Analysis:** Analyze body alignment and positioning.
- **Posture Classification:** Identify correct and incorrect sitting positions.
- **Real-Time Monitoring:** Process visual input from a camera (if implemented).
- **Ergonomic Awareness:** Help users recognize potentially harmful sitting habits.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing and computer vision |
| MediaPipe | Human pose estimation and landmark detection |
| TensorFlow | Machine learning and posture classification |
| NumPy | Numerical operations |
| Git & GitHub | Version control and project management |

## 🏗️ System Architecture

The proposed system follows a computer vision and machine learning pipeline:

1. **Image Acquisition:** Capture images or video frames from a camera.
2. **Image Processing:** Process frames using OpenCV.
3. **Pose Estimation:** Extract body landmarks using MediaPipe.
4. **Feature Extraction:** Analyze relevant body coordinates and angles.
5. **Posture Classification:** Use a trained machine learning model to classify the detected posture.
6. **Output:** Display the posture classification and corresponding feedback.

## ⚙️ Installation and Setup

### Prerequisites

Make sure you have installed:

- Python 3.9 or compatible version.
- pip (Python package manager).
- Git.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd sitting-posture-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not available, install the required libraries individually according to the project implementation.

## ▶️ How to Run

Run the main Python script:

```bash
python main.py
```

Make sure your camera is connected if the application uses real-time video input.

*Note: Replace the command above with the actual entry point of your project.*

## 📊 Results and Performance

The project focused on improving posture detection accuracy and processing performance.

Reported performance during development:

- **92% detection accuracy**
- **30% improvement in detection performance**
- **Inference time reduced from 1.5 seconds to 0.8 seconds**

These figures should be accompanied by the corresponding evaluation methodology and test dataset when available.

## 📁 Project Structure

```text
sitting-posture-detection/
│
├── dataset/              # Training and testing data
├── models/               # Trained machine learning models
├── src/                  # Source code
│   ├── detection/         # Pose detection
│   ├── processing/        # Image processing
│   └── classification/   # Posture classification
│
├── main.py               # Application entry point
├── requirements.txt      # Project dependencies
├── README.md
└── .gitignore
```

*Adapt this structure to match the actual repository.*

## 🚀 Future Improvements

- Add more sitting posture categories.
- Improve model accuracy and robustness.
- Develop a user-friendly graphical interface.
- Integrate real-time alerts for incorrect posture.
- Add posture history and monitoring statistics.
- Optimize the model for better performance.

## 👩‍💻 Internship Information

**Project:** Sitting Posture Detection  
**Company:** Elastic Solution  
**Internship:** Summer Internship – July to August 2025  
**Domain:** Artificial Intelligence | Computer Vision | Machine Learning

This project provided practical experience in computer vision, human pose estimation, image processing, and machine learning, while addressing a real-world ergonomic challenge.

## 👤 Author

**Salma Ben Romdhane**

- GitHub: [SalmaaBr](https://github.com/SalmaBr)

---

⭐ If you find this project interesting, feel free to explore the repository!
