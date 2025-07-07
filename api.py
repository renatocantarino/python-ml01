from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from utils import *


app = Flask(__name__)
CORS(app)

model = load_model("./objects/model.keras")
selector = joblib.load("./objects/selector.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400

    df = pd.DataFrame(input_data)
    df = load_scalers(df, const.scalers)
    df = load_encoders(df, const.encoders)
    df = selector.transform(df)

    return jsonify({"data": model.predict(df).tolist()})


if __name__ == "__main__":
    app.run(debug=True)
