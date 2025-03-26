import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict

########################################
# Database configuration
DB = SqliteDatabase('predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)
########################################

########################################
# Load trained model and metadata
with open(os.path.join('columns.json')) as fh:
    columns = json.load(fh)

with open(os.path.join('dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)

with open(os.path.join('pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)
########################################

########################################
# Define valid categories for categorical columns
valid_categories = {
    "workclass": ['Private', 'Self-emp-inc', 'Local-gov', 'Federal-gov','State-gov', 'Self-emp-not-inc', 'Never-worked', 'Without-pay'],
    "sex": ["Male", "Female"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "education": ['11th', 'Bachelors', 'HS-grad', 'Assoc-voc', 'Some-college','5th-6th', '1st-4th', '10th', 'Masters', 'Doctorate', '7th-8th','12th', '9th', 'Assoc-acdm', 'Prof-school', 'Preschool'],
    "marital-status": ['Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced','Married-spouse-absent', 'Separated', 'Married-AF-spouse']
}

# Define valid ranges for numerical columns (only minimum values)
valid_ranges = {
    "age": (0, 100),
    "capital-gain": (0, 99999), 
    "capital-loss": (0, 4356), 
    "hours-per-week": (1, 99)
}

########################################
# Helper functions for validation
def get_valid_categories(df, column):
    return valid_categories.get(column, []) if column in valid_categories else []

def get_valid_range(df, column):
    return valid_ranges.get(column, (None, None)) if column in valid_ranges else (None, None)

def validate_positive_values(data, column):
    """
    Validates that the value of a column is greater than or equal to zero (only for capital-gain and capital-loss).
    """
    if column in data:
        if data[column] < 0:
            return False, f"Invalid value for '{column}'. It must be greater than or equal to 0. Provided: {data[column]}"
    return True, None

def validate_categories(data, column):
    """
    Validates that the value of a categorical column is within the valid categories.
    """
    if column in data and data[column] not in valid_categories.get(column, []):
        return False, f"Invalid value for '{column}'. Valid values are: {', '.join(valid_categories[column])}. Provided: {data[column]}"
    return True, None

def validate_range(data, column):
    """
    Validates that the value of a numerical column is within the valid range.
    """
    if column in data:
        min_value, max_value = valid_ranges.get(column, (None, None))
        if min_value is not None:
            if data[column] < min_value:
                return False, f"Invalid value for '{column}'. It must be greater than or equal to {min_value}. Provided: {data[column]}"
        # No need to check for max_value, since it's None
    return True, None

def attempt_predict(request_data):
    try:
        if 'data' not in request_data:
            return {"observation_id": request_data.get("observation_id"), "error": "Missing 'data' field in request."}

        data = request_data['data']
        extra_columns = [col for col in data.keys() if col not in columns]
        if extra_columns:
            return {"observation_id": request_data.get("observation_id"), "error": f"Unexpected columns: {', '.join(extra_columns)}"}

        for col in columns:
            if col not in data:
                return {"observation_id": request_data.get("observation_id"), "error": f"Missing required field '{col}'"}

        # Validate categories for categorical columns
        for col in valid_categories:
            is_valid, error_message = validate_categories(data, col)
            if not is_valid:
                return {"observation_id": request_data.get("observation_id"), "error": error_message}

        # Validate ranges for numerical columns
        for col in valid_ranges:
            is_valid, error_message = validate_range(data, col)
            if not is_valid:
                return {"observation_id": request_data.get("observation_id"), "error": error_message}

        # Validate that 'capital-gain' and 'capital-loss' columns have a minimum value of 0
        numeric_columns = ['capital-gain', 'capital-loss']
        for col in numeric_columns:
            if col in data:
                is_valid, error_message = validate_positive_values(data, col)
                if not is_valid:
                    return {
                        "observation_id": request_data.get("observation_id"),
                        "error": error_message
                    }

        input_data = pd.DataFrame([data]).astype(dtypes)
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0, 1]

        return {"observation_id": request_data["observation_id"], "prediction": bool(prediction), "probability": float(probability)}
    except Exception as e:
        return {"observation_id": request_data.get("observation_id"), "error": str(e)}
########################################

########################################
# Flask server configuration
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    response = attempt_predict(request_data)
    
    if "error" not in response:
        try:
            p = Prediction(
                observation_id=request_data["observation_id"],
                proba=response["probability"],
                observation=json.dumps(request_data)
            )
            p.save()
        except IntegrityError:
            response["error"] = f"ERROR: Observation ID {request_data['observation_id']} already exists"
            DB.rollback()
    
    return jsonify(response)

@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        return jsonify({'error': f"Observation ID {obs['id']} does not exist"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5008)
