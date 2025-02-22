# Predictor: 10-Year CHD Prediction

## 📌 Project Overview
This project aims to predict whether a person is at risk of developing Coronary Heart Disease (CHD) within the next 10 years based on various health parameters. CHD is a leading cause of heart-related complications, and early prediction can help in preventive measures.

The model is designed to analyze relevant medical and lifestyle factors to classify individuals into high-risk and low-risk categories using machine learning techniques.

## 🚀 Features
- Data ingestion and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Model training and evaluation
- Prediction pipeline for real-world use
- Model monitoring for continuous improvements
- API integration using Flask
- Deployment using Docker

## 📂 Project Structure
```
├── src/
│   ├── predictor/
│   │   ├── __init__.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── data_ingestion.py
│   │   │   ├── data_transformation.py
│   │   │   ├── model_trainer.py
│   │   │   ├── model_monitoring.py
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   ├── training_pipeline.py
│   │   │   ├── prediction_pipeline.py
│   │   ├── exception.py
│   │   ├── logger.py
│   │   ├── utils.py
├── notebooks/
│   ├── data/
│   ├── EDA.ipynb
│   ├── model_training.ipynb
├── artifacts/
├── main.py
├── app.py
├── Dockerfile
├── requirements.txt
├── setup.py
├── .env
└── README.md
```

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Framework**: Flask (for API deployment)
- **Machine Learning**: Scikit-Learn, TensorFlow/PyTorch
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Deployment**: Docker, GitHub Actions (CI/CD)
- **Logging & Monitoring**: Logging module

## 🔧 Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/IAMDSVSSANGRAL/10yrs_chd_predictor.git
   cd predictor
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   python app.py
   ```

## 🏋️‍♂️ Usage
- **Train the Model:**
  ```sh
  python main.py
  ```
- **Predict using API:**
  ```sh
  curl -X POST -H "Content-Type: application/json" -d '{"age": 50, "cholesterol": 220, "smoker": 1}' http://127.0.0.1:5000/predict
  ```

## 📊 Dataset
The dataset contains medical and lifestyle attributes such as:
- Age
- Cholesterol level
- Smoking habits
- Blood pressure
- Diabetes history
- BMI

## 🏆 Model Performance
- **Accuracy:** XX%
- **Precision:** XX%
- **Recall:** XX%
- **F1-score:** XX%

## 📌 Future Improvements
- Add more real-world medical datasets for training.
- Improve model performance with advanced feature selection.
- Deploy as a cloud-based API.

## 🤝 Contributing
If you would like to contribute to this project, feel free to submit a pull request or open an issue.

## 📜 License
This project is licensed under the MIT License.

---
Developed by [Vishal Singh Sangral] | **Havinosh Data Solutions** 🚀

