# 🍂 Handwritten Digit Classification with CNN

A deep learning-based project for classifying handwritten digits using a Convolutional Neural Network (CNN). The model classifies images of handwritten digits into categories (0-9).

## 📌 Features

✅ **Image Classification** – Detects handwritten digits (0-9) from image inputs.  
✅ **Deep Learning Model** – Uses CNN-based architecture (TensorFlow/Keras).  
✅ **Web & Mobile Support** – Deployable as a Flask/FastAPI API, Android app (TensorFlow Lite), or TensorFlow.js for web.  
✅ **Cloud Deployment** – Host on Google Cloud, AWS, or Firebase.

## 🖼️ Dataset

The dataset used for this project is the MNIST dataset, which contains 28x28 grayscale images of handwritten digits.

- **MNIST Digits Dataset**  
- **Classes**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9  

📌 Dataset Source: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow/Keras** (for model training)
- **Flask/FastAPI** (for API deployment)
- **OpenCV/PIL** (for image preprocessing)
- **NumPy, Pandas, Matplotlib** (for data analysis)
- **TensorFlow Lite (TFLite)** (for mobile app deployment)
- **Google Cloud/AWS** (for cloud hosting)

## 🚀 Installation & Setup

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/Handwritten-Digit-Classification-with-CNN.git
cd Handwritten-Digit-Classification-with-CNN
2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Train the Model (Optional)
If you want to retrain the model, run:
python train.py
This will save the model as digit_model.h5.

📡 Deployment Options
🌐 Web API (FastAPI)
Run the API locally:

uvicorn app:app --host 0.0.0.0 --port 8000
Then send a test request:

curl -X POST -F "file=@test_digit.jpg" http://127.0.0.1:8000/predict/
📱 Android App (TensorFlow Lite)
Convert the model to TFLite format:
tflite_convert --saved_model_dir=digit_model/ --output_file=digit_model.tflite
Integrate it into an Android app using ML Kit.

☁️ Cloud Deployment
Google Cloud Run (for scalable API hosting)
AWS Lambda + API Gateway (serverless API)
Firebase Hosting (for web app)
📌 Example Usage
Using Python:

python
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("digit_model.h5")

def predict_image(image_path):
    image = Image.open(image_path).resize((28, 28))
    img_array = np.array(image.convert('L')) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    prediction = model.predict(img_array)
    return prediction

print(predict_image("test_digit.jpg"))
📷 Screenshots
Healthy Digits
Predictions on Various Digits
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repo
Create a new branch (feature-new-idea)
Commit your changes
Submit a Pull Request
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

💡 Acknowledgments
MNIST Dataset
TensorFlow/Keras Community
OpenAI & Deep Learning Research
📩 Contact
For questions or suggestions, reach out:
📧 Email: abenezeralz659@gmail.com
    GitHub: https://github.com/Abena-3565

This should give you a comprehensive and clean README for your project. Let me know if you'd like any modifications!
