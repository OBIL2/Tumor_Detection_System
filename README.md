# 🧠 AI Tumor Detection System

<p align="center">
  <img src="assets/banner.png" alt="Tumor Detection Banner" width="100%">
</p>

<p align="center">
  <b>An AI-powered medical image classification system built using Deep Learning and Streamlit.</b>
</p>

---

# 📌 Overview

The **AI Tumor Detection System** is a deep learning-based web application designed to assist in the detection and classification of tumors from uploaded medical images. The project combines the power of **Artificial Intelligence**, **Computer Vision**, and an interactive **Streamlit** user interface to create a simple yet effective diagnostic support system.

Users can upload medical scan images directly through the application, where the trained AI model processes the image and predicts whether a tumor is detected. The system aims to demonstrate the practical application of machine learning in the healthcare and medical imaging domain.

This project was developed as an educational and research-focused implementation of AI-based medical image analysis.

---

# 🚀 Features

✅ Upload medical images for analysis  
✅ AI-powered tumor classification  
✅ Real-time prediction results  
✅ Streamlit-based interactive web interface  
✅ Simple and user-friendly design  
✅ Fast image preprocessing and prediction  
✅ Lightweight and easy to run locally  
✅ Deep learning model integration  
✅ Clean and modular project structure  

---

# 🧠 AI & Deep Learning

This project uses a **Convolutional Neural Network (CNN)** model trained on medical imaging datasets to perform tumor classification tasks.

The AI model performs the following tasks:

- Image preprocessing
- Feature extraction
- Pattern recognition
- Tumor prediction and classification

Deep learning allows the system to automatically learn important visual patterns from medical images without requiring manual feature engineering.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Web application framework |
| TensorFlow / Keras | Deep learning model development |
| OpenCV | Image processing |
| NumPy | Numerical computations |
| PIL | Image handling |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation and preprocessing |

---

# 📂 Project Structure

```bash
Tumor-Detection-System/
│
├── app.py                     # Main Streamlit application
├── model/                     # Trained deep learning model
├── dataset/                   # Dataset used for training
├── assets/                    # Images, icons, screenshots
├── notebooks/                 # Jupyter notebooks for training
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── utils/                     # Helper functions and preprocessing
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/tumor-detection-system.git
```

---

## 2️⃣ Navigate to Project Directory

```bash
cd tumor-detection-system
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

After running the command, the application will automatically open in your default browser.

---

# 💡 How the System Works

The workflow of the application is simple and efficient:

## Step 1 — Upload Image
The user uploads a medical scan image through the Streamlit interface.

## Step 2 — Image Preprocessing
The uploaded image is resized, normalized, and prepared for the AI model.

## Step 3 — AI Prediction
The trained CNN model analyzes the image and extracts important features.

## Step 4 — Classification
The system predicts whether the image indicates the presence of a tumor.

## Step 5 — Display Results
Prediction results are displayed instantly through the user interface.

---

# 📸 Application Screenshots

## 🖥️ Home Interface

_Add application screenshot here_

```bash
assets/homepage.png
```

---

## 🔍 Prediction Result

_Add prediction screenshot here_

```bash
assets/result.png
```

---

# 📊 Dataset Information

The model was trained using medical imaging datasets containing tumor and non-tumor scan images.

The dataset includes:

- Brain scan images
- Tumor-positive samples
- Tumor-negative samples
- Preprocessed medical image data

---

# 📈 Model Performance

The deep learning model was evaluated using standard machine learning metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score

Performance optimization techniques such as normalization, data augmentation, and dropout layers can also be implemented to improve prediction accuracy.

---

# 🔮 Future Improvements

This project can be expanded further with advanced features such as:

- Multi-class tumor classification
- Tumor segmentation
- Confidence score visualization
- Cloud deployment support
- User authentication system
- Medical report generation
- Database integration
- Real-time webcam/image capture
- Improved AI model accuracy
- Mobile responsiveness

---

# 🌍 Real-World Importance

AI-powered healthcare systems are becoming increasingly important in modern medicine. Projects like this demonstrate how artificial intelligence can assist healthcare professionals by:

- Reducing diagnostic workload
- Improving early detection
- Supporting medical decision-making
- Enhancing accessibility to healthcare technologies

---

# ⚠️ Disclaimer

This project is intended strictly for:

- Educational purposes
- Research purposes
- AI learning demonstrations

It is **NOT** a replacement for professional medical diagnosis, treatment, or healthcare services.

Always consult qualified medical professionals for actual medical concerns.

---



# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub to support the project.

---

# 👨‍💻 Author

## Obil Abid
## Inderias Samson

Computer Science Students | AI & Software Development Enthusiast

---

# 📜 License

This project is licensed under the MIT License.

---

# 📬 Contact

For questions, suggestions, or collaboration opportunities:

📧 271048001@formanite.fccollege.edu.pk
📧 271052262@formanite.fccollege.edu.pk

---

<p align="center">
  <b>Built with Python, Streamlit, and Deep Learning 🚀</b>
</p>
