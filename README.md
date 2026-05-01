# 🎓 Student Grade Prediction

An end-to-end machine learning system that predicts student performance and provides AI-powered tutoring support.

## 📌 Overview

This project combines classical machine learning models with an AI tutor (LLM) to predict student grades and provide personalized support. It features a comprehensive Streamlit web interface for easy interaction and model management.

## ✨ Key Features

- **Grade Prediction**: Predict student performance based on 14+ input features
- **Dashboard**: Visual overview of dataset statistics and model performance
- **Ticket Classifier**: Interactive interface for grade predictions with confidence scores
- **AI Response Generator**: HuggingFace-powered LLM for student support
- **Model Analytics**: Compare models with confusion matrices and feature importance
- **Admin Panel**: Upload datasets and retrain models

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/haneenmohd00/student-grade-prediction.git
   cd student-grade-prediction
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   streamlit run app.py

---

## 📋 Input Features

* **Demographics:** Age, Gender, Ethnicity
* **Academic:** GPA, Study Time Weekly, Absences
* **Family Background:** Parental Education, Parental Support
* **Activities:** Tutoring, Extracurricular, Sports, Music, Volunteering

---

## 🤖 Machine Learning Models

* Logistic Regression
* Random Forest (300 estimators)
* XGBoost (Gradient Boosting)

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Machine Learning:** scikit-learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly
* **AI Integration:** Hugging Face API

---

## 📁 Project Structure

```
student-grade-prediction/
├── app.py              # Main Streamlit app
├── config.py           # Configuration
├── train.py            # Model training
├── predict.py          # Predictions
├── utils.py            # Utilities
├── hf_api.py           # HuggingFace integration
├── requirements.txt    # Dependencies
├── data/               # Datasets
├── models/             # Trained models
└── assets/             # UI assets
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new feature branch
3. Make your changes
4. Push to your branch
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

* **haneenmohd00**

---
