import joblib
import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify

# Initialize Flask app
superkart_api = Flask("SuperKart Sales Revenue Predictor")

# Load model
model = joblib.load(
    "deployment_files/superkart_prediction_model_v1_0.joblib"
)


# Home route
@superkart_api.route("/", methods=["GET"])
def home():

    return jsonify(
        {
            "message": "Welcome to the SuperKart Sales Revenue Prediction API"
        }
    )


# Prediction route
@superkart_api.route("/predict", methods=["POST"])
def predict():

    try:

        # Get JSON payload
        input_json = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_json])

        # Predict
        prediction = model.predict(input_df)

        return jsonify(
            {
                "predicted_product_store_sales": round(
                    float(prediction[0]), 2
                )
            }
        )

    except Exception as e:

        return jsonify(
            {
                "error": str(e)
            }
        ), 400


if __name__ == "__main__":

    superkart_api.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
