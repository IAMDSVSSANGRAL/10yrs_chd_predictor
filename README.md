# 10-Year Coronary Heart Disease (CHD) Risk Prediction

A complete Machine Learning pipeline for predicting 10-year Coronary Heart Disease risk with a FastAPI REST API.

## 🚀 Features

- Complete ML pipeline (ingestion, validation, transformation, training, evaluation, monitoring)
- Multiple ML models comparison (Logistic Regression, Random Forest, XGBoost, etc.)
- SMOTE for handling imbalanced data
- Feature engineering for better predictions
- RESTful API with FastAPI
- Comprehensive model monitoring
- Interactive API documentation

## 📋 Requirements

- Python 3.8+
- PostgreSQL database
- Virtual environment (recommended)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd 10yrs_chd_predictor
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup environment variables

Create a `.env` file in the root directory:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=Healthcare
DB_USER=postgres
DB_PASSWORD=your_password
```

### 5. Prepare your database

Make sure your PostgreSQL database has the table `data_cardiovascular_risk` with the required schema.

## 🎯 Usage

### Training the Model

Run the complete ML pipeline:

```bash
python main.py
```

This will:
1. Ingest data from PostgreSQL
2. Validate data quality
3. Transform and engineer features
4. Train multiple models
5. Evaluate and select best model
6. Setup monitoring

### Starting the API Server

```bash
python app.py
```

Or with uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### Testing the API

Run the test script:

```bash
python test_api.py
```

## 📚 API Endpoints

### Health Check
```http
GET /health
```

### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "age": 45,
  "sex": "M",
  "is_smoking": "YES",
  "cigsperday": 20.0,
  "bpmeds": 0.0,
  "prevalentstroke": 0,
  "prevalenthyp": 0,
  "diabetes": 0,
  "totchol": 250.0,
  "sysbp": 140.0,
  "diabp": 90.0,
  "bmi": 28.5,
  "heartrate": 75.0,
  "glucose": 100.0,
  "education": 2
}
```

### Batch Prediction
```http
POST /batch-predict
Content-Type: application/json

[
  {patient_data_1},
  {patient_data_2},
  ...
]
```

### Model Information
```http
GET /model-info
```

## 📊 Interactive Documentation

Once the server is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": "M",
    "is_smoking": "YES",
    "cigsperday": 20.0,
    "bpmeds": 0.0,
    "prevalentstroke": 0,
    "prevalenthyp": 0,
    "diabetes": 0,
    "totchol": 250.0,
    "sysbp": 140.0,
    "diabp": 90.0,
    "bmi": 28.5,
    "heartrate": 75.0,
    "glucose": 100.0,
    "education": 2
  }'
```

## 📁 Project Structure

```
10yrs_chd_predictor/
├── src/
│   └── predictor/
│       ├── components/          # ML pipeline components
│       ├── pipelines/           # Training and prediction pipelines
│       ├── logger.py           # Logging configuration
│       ├── exception.py        # Custom exceptions
│       └── utils.py            # Utility functions
├── artifacts/                  # Model artifacts and reports
├── main.py                    # Training pipeline runner
├── app.py                     # FastAPI application
├── test_api.py               # API test script
└── requirements.txt          # Dependencies
```

## 🔍 Model Features

The model uses the following features:

- **Demographics**: Age, Sex, Education
- **Lifestyle**: Smoking status, Cigarettes per day
- **Medical History**: Previous stroke, Hypertension, Diabetes
- **Vital Signs**: Blood pressure (systolic/diastolic), Heart rate
- **Lab Results**: Total cholesterol, BMI, Glucose
- **Medications**: BP medication usage

## 📈 Model Performance

The pipeline trains multiple models and selects the best based on F1-score. Typical performance:

- Accuracy: ~85%
- F1-Score: ~70%
- ROC-AUC: ~75%

(Note: Actual performance depends on your data)

## 🔧 Configuration

### Adjusting Model Parameters

Edit `src/predictor/components/model_trainer.py` to modify model hyperparameters.

### Changing Resampling Strategy

In `main.py`, change:
```python
data_transformation = DataTransformation(use_smoteenn=True)  # Use SMOTEENN
# or
data_transformation = DataTransformation(use_smoteenn=False)  # Use SMOTE
```

## 📝 Logging

Logs are automatically generated and stored. Check the console output for detailed pipeline execution logs.

## 🐛 Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running
- Check credentials in `.env` file
- Ensure database and table exist

### Model Loading Issues
- Make sure `main.py` has been run successfully
- Check if `artifacts/model.pkl` and `artifacts/preprocessor.pkl` exist

### API Server Issues
- Check if port 8000 is available
- Try running on a different port: `uvicorn app:app --port 8080`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Your Name - [Your Email]

## 🙏 Acknowledgments

- Framingham Heart Study for the CHD risk data concept
- FastAPI for the excellent web framework
- scikit-learn and imbalanced-learn communities