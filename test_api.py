import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("Testing Health Check Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*70)
    print("Testing Single Prediction Endpoint")
    print("="*70)
    
    # Sample patient data
    patient_data = {
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
    
    print("\nSending patient data:")
    print(json.dumps(patient_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_high_risk_patient():
    """Test prediction for high-risk patient"""
    print("\n" + "="*70)
    print("Testing High-Risk Patient Prediction")
    print("="*70)
    
    # High-risk patient profile
    high_risk_patient = {
        "age": 65,
        "sex": "M",
        "is_smoking": "YES",
        "cigsperday": 40.0,
        "bpmeds": 1.0,
        "prevalentstroke": 1,
        "prevalenthyp": 1,
        "diabetes": 1,
        "totchol": 300.0,
        "sysbp": 180.0,
        "diabp": 110.0,
        "bmi": 35.0,
        "heartrate": 95.0,
        "glucose": 150.0,
        "education": 1
    }
    
    print("\nSending high-risk patient data:")
    print(json.dumps(high_risk_patient, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=high_risk_patient)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_low_risk_patient():
    """Test prediction for low-risk patient"""
    print("\n" + "="*70)
    print("Testing Low-Risk Patient Prediction")
    print("="*70)
    
    # Low-risk patient profile
    low_risk_patient = {
        "age": 30,
        "sex": "F",
        "is_smoking": "NO",
        "cigsperday": 0.0,
        "bpmeds": 0.0,
        "prevalentstroke": 0,
        "prevalenthyp": 0,
        "diabetes": 0,
        "totchol": 180.0,
        "sysbp": 110.0,
        "diabp": 70.0,
        "bmi": 22.0,
        "heartrate": 65.0,
        "glucose": 85.0,
        "education": 4
    }
    
    print("\nSending low-risk patient data:")
    print(json.dumps(low_risk_patient, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=low_risk_patient)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*70)
    print("Testing Batch Prediction Endpoint")
    print("="*70)
    
    # Multiple patients
    patients = [
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
        },
        {
            "age": 55,
            "sex": "F",
            "is_smoking": "NO",
            "cigsperday": 0.0,
            "bpmeds": 1.0,
            "prevalentstroke": 0,
            "prevalenthyp": 1,
            "diabetes": 0,
            "totchol": 220.0,
            "sysbp": 135.0,
            "diabp": 85.0,
            "bmi": 26.0,
            "heartrate": 72.0,
            "glucose": 95.0,
            "education": 3
        },
        {
            "age": 35,
            "sex": "M",
            "is_smoking": "NO",
            "cigsperday": 0.0,
            "bpmeds": 0.0,
            "prevalentstroke": 0,
            "prevalenthyp": 0,
            "diabetes": 0,
            "totchol": 190.0,
            "sysbp": 120.0,
            "diabp": 80.0,
            "bmi": 24.0,
            "heartrate": 68.0,
            "glucose": 90.0,
            "education": 4
        }
    ]
    
    print(f"\nSending {len(patients)} patients for batch prediction")
    
    response = requests.post(f"{BASE_URL}/batch-predict", json=patients)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*70)
    print("Testing Model Info Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_invalid_data():
    """Test with invalid data"""
    print("\n" + "="*70)
    print("Testing Invalid Data Handling")
    print("="*70)
    
    # Invalid age
    invalid_patient = {
        "age": 150,  # Invalid age
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
    
    print("\nSending invalid patient data (age=150):")
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_patient)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 422  # Validation error expected


def run_all_tests():
    """Run all API tests"""
    print("\n" + "#"*70)
    print("# STARTING API TESTS")
    print("#"*70)
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("High-Risk Patient", test_high_risk_patient),
        ("Low-Risk Patient", test_low_risk_patient),
        ("Batch Prediction", test_batch_prediction),
        ("Model Info", test_model_info),
        ("Invalid Data", test_invalid_data)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print("#"*70 + "\n")


if __name__ == "__main__":
    print("\n🚀 Starting API Tests...")
    print("Make sure the API server is running on http://localhost:8000\n")
    
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server.")
        print("Please make sure the server is running with: python app.py")
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")