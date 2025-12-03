import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Recommendation functions
def get_risk_level(probability):
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"

def generate_recommendations(patient_data, prediction, probability):
    age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, \
    max_hr, exercise_angina, oldpeak, st_slope = patient_data
    
    recommendations = {
        'lifestyle': [],
        'medications': [],
        'monitoring': [],
        'referrals': [],
        'urgent_actions': []
    }
    
    # Risk-based recommendations
    if prediction == 1:
        if probability > 0.7:
            recommendations['urgent_actions'].extend([
                "Immediate cardiology consultation (within 48 hours)",
                "Daily symptom journal",
                "Fluid restriction (<2L/day)",
                "Daily weight monitoring"
            ])
            recommendations['medications'].extend([
                "Start ACE inhibitor/ARB",
                "Start beta-blocker",
                "Consider diuretic"
            ])
        elif probability > 0.4:
            recommendations['referrals'].append("Cardiology consultation within 2 weeks")
            recommendations['medications'].extend([
                "Consider statin therapy",
                "Antihypertensive if BP > 130/80"
            ])
            recommendations['monitoring'].extend([
                "Bi-weekly BP checks",
                "Monthly lipid panel"
            ])
        
        recommendations['lifestyle'].extend([
            "Sodium restriction (<2g/day)",
            "Cardiac rehabilitation program",
            "Alcohol moderation"
        ])
        recommendations['monitoring'].append("BNP/NT-proBNP testing")
    else:
        recommendations['lifestyle'].extend([
            "Heart-healthy diet (Mediterranean or DASH)",
            "150 minutes moderate exercise weekly",
            "Annual cardiac check-up",
            "Smoking cessation if applicable"
        ])
    
    # Feature-specific recommendations
    if resting_bp >= 140:
        recommendations['medications'].append("Antihypertensive medication")
        recommendations['lifestyle'].append("DASH diet specifically")
    
    if cholesterol >= 240:
        recommendations['medications'].append("High-intensity statin")
        recommendations['monitoring'].append("Lipid panel in 4 weeks")
    
    if fasting_bs == 1:
        recommendations['medications'].append("SGLT2 inhibitor")
        recommendations['monitoring'].append("HbA1c every 3 months")
    
    if exercise_angina == 'Y':
        recommendations['lifestyle'].extend([
            "Avoid strenuous exercise",
            "Cardiac-supervised exercise only"
        ])
        recommendations['referrals'].append("Exercise stress test")
    
    if resting_ecg in ['LVH', 'ST']:
        recommendations['referrals'].append("Echocardiogram")
        recommendations['monitoring'].append("ECG every 6 months")
    
    if age >= 65:
        recommendations['medications'].append("Renal function monitoring")
        recommendations['monitoring'].append("Metabolic panel monthly")
    
    if sex == 'F':
        recommendations['monitoring'].append("Anemia screening")
    else:
        recommendations['monitoring'].append("PSA testing if >50")
    
    if oldpeak > 2.0:
        recommendations['urgent_actions'].append("Urgent coronary angiography")
    
    if st_slope == 'Down':
        recommendations['medications'].append("Antiplatelet therapy")
        recommendations['referrals'].append("Nuclear stress test")
    
    return recommendations

# Try loading the model safely
model = None
model_error = None
model_path = os.path.join(os.path.dirname(__file__), "heart_failure_model.pkl")
try:
    model = joblib.load(model_path)
except Exception as e:
    model_error = str(e)
    model = None

# Define the expected feature order
FEATURE_NAMES = ["age", "sex", "chest_pain", "resting_bp", "cholesterol", 
                 "fasting_bs", "resting_ecg", "max_hr", "exercise_angina", 
                 "oldpeak", "st_slope"]

@app.route('/')
def home():
    if model is None:
        return jsonify({"error": f"Model failed to load: {model_error}"}), 500
    return "Heart Failure Detection API is up and running!"

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is None:
        return jsonify({"error": f"Model failed to load: {model_error}"}), 500
    return jsonify({
        "status": "healthy",
        "service": "Heart Failure Detection API",
        "version": "1.0.0"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Basic prediction endpoint"""
    if model is None:
        return jsonify({"error": f"Model failed to load: {model_error}"}), 500

    try:
        data = request.get_json()
        
        # Validate required fields
        for name in FEATURE_NAMES:
            if name not in data:
                return jsonify({
                    "error": "Missing one or more required features.",
                    "required": FEATURE_NAMES
                }), 400

        # Get features in the correct order
        features = [data.get(name) for name in FEATURE_NAMES]
        
        # Convert to model input format
        patient_data = [
            int(features[0]),    # age
            features[1],        # sex
            features[2],        # chest_pain
            int(features[3]),   # resting_bp
            int(features[4]),   # cholesterol
            int(features[5]),   # fasting_bs
            features[6],        # resting_ecg
            int(features[7]),   # max_hr
            features[8],        # exercise_angina
            float(features[9]), # oldpeak
            features[10]        # st_slope
        ]
        
        # Create DataFrame
        feature_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                          'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                          'Oldpeak', 'ST_Slope']
        patient_df = pd.DataFrame([patient_data], columns=feature_columns)
        
        # Make prediction
        prediction = model.predict(patient_df)[0]
        probability = model.predict_proba(patient_df)[0][1]
        
        # Get risk level
        risk_level = get_risk_level(probability)
        
        return jsonify({
            "prediction": int(prediction),
            "risk_level": risk_level,
            "probability": float(probability),
            "message": "Heart Failure Detected" if prediction == 1 else "Normal"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/assess', methods=['POST'])
def assess():
    """Comprehensive assessment with recommendations"""
    if model is None:
        return jsonify({"error": f"Model failed to load: {model_error}"}), 500

    try:
        data = request.get_json()
        
        # Validate required fields
        for name in FEATURE_NAMES:
            if name not in data:
                return jsonify({
                    "error": "Missing one or more required features.",
                    "required": FEATURE_NAMES
                }), 400

        # Get features in the correct order
        features = [data.get(name) for name in FEATURE_NAMES]
        
        # Convert to model input format
        patient_data = [
            int(features[0]),    # age
            features[1],        # sex
            features[2],        # chest_pain
            int(features[3]),   # resting_bp
            int(features[4]),   # cholesterol
            int(features[5]),   # fasting_bs
            features[6],        # resting_ecg
            int(features[7]),   # max_hr
            features[8],        # exercise_angina
            float(features[9]), # oldpeak
            features[10]        # st_slope
        ]
        
        # Create DataFrame
        feature_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                          'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                          'Oldpeak', 'ST_Slope']
        patient_df = pd.DataFrame([patient_data], columns=feature_columns)
        
        # Make prediction
        prediction = model.predict(patient_df)[0]
        probability = model.predict_proba(patient_df)[0][1]
        
        # Get risk level and recommendations
        risk_level = get_risk_level(probability)
        recommendations = generate_recommendations(patient_data, prediction, probability)
        
        return jsonify({
            "prediction": int(prediction),
            "risk_level": risk_level,
            "probability": float(probability),
            "recommendations": recommendations,
            "patient_data": {
                "age": patient_data[0],
                "sex": patient_data[1],
                "chest_pain": patient_data[2],
                "resting_bp": patient_data[3],
                "cholesterol": patient_data[4],
                "fasting_bs": patient_data[5],
                "resting_ecg": patient_data[6],
                "max_hr": patient_data[7],
                "exercise_angina": patient_data[8],
                "oldpeak": patient_data[9],
                "st_slope": patient_data[10]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    if model is None:
        return jsonify({"error": f"Model failed to load: {model_error}"}), 500

    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({"error": "Expected a list of patient data"}), 400
        
        results = []
        
        for patient in data:
            try:
                # Validate required fields
                for name in FEATURE_NAMES:
                    if name not in patient:
                        results.append({"error": f"Missing required field: {name}"})
                        continue
                
                # Get features in the correct order
                features = [patient.get(name) for name in FEATURE_NAMES]
                
                # Convert to model input format
                patient_data = [
                    int(features[0]),    # age
                    features[1],        # sex
                    features[2],        # chest_pain
                    int(features[3]),   # resting_bp
                    int(features[4]),   # cholesterol
                    int(features[5]),   # fasting_bs
                    features[6],        # resting_ecg
                    int(features[7]),   # max_hr
                    features[8],        # exercise_angina
                    float(features[9]), # oldpeak
                    features[10]        # st_slope
                ]
                
                # Create DataFrame
                feature_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                                  'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                                  'Oldpeak', 'ST_Slope']
                patient_df = pd.DataFrame([patient_data], columns=feature_columns)
                
                # Make prediction
                prediction = model.predict(patient_df)[0]
                probability = model.predict_proba(patient_df)[0][1]
                
                # Get risk level and recommendations
                risk_level = get_risk_level(probability)
                recommendations = generate_recommendations(patient_data, prediction, probability)
                
                results.append({
                    "prediction": int(prediction),
                    "risk_level": risk_level,
                    "probability": float(probability),
                    "recommendations": recommendations
                })
                
            except Exception as e:
                results.append({"error": str(e)})
        
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)