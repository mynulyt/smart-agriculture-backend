import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# =========================
# AUTO TRAIN IF MODEL MISSING
# =========================

if not os.path.exists('models'):

    os.makedirs('models')

if not os.path.exists('models/classifier.pkl'):

    import train_model

classifier = pickle.load(
    open('models/classifier.pkl', 'rb')
)

regressor = pickle.load(
    open('models/regressor.pkl', 'rb')
)

encoders = pickle.load(
    open('models/encoders.pkl', 'rb')
)

X_reg_columns = [
    'crop',
    'year',
    'season',
    'State',
    'area',
    'fertilizer',
    'pesticide',
    'N',
    'P',
    'K',
    'pH',
    'avg_temp_c',
    'total_rainfall_mm',
    'avg_humidity_percent'
]

# =========================
# HOME ROUTE
# =========================

@app.route('/')
def home():

    return jsonify({
        "message": "Smart Agriculture API Running",
        "status": "success"
    })

# =========================
# SMART SUGGESTION FUNCTION
# =========================

def generate_suggestion(temp, humidity, rainfall):

    if rainfall < 50:
        return "Low rainfall detected. Irrigation is recommended."

    elif humidity > 85:
        return "High humidity detected. Monitor fungal diseases carefully."

    elif temp > 35:
        return "High temperature detected. Ensure proper watering schedule."

    else:
        return "Balanced environmental conditions detected."

# =========================
# PREDICTION API
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # =========================
        # GET REQUEST DATA
        # =========================

        data = request.json

        n = float(data['N'])
        p = float(data['P'])
        k = float(data['K'])

        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        state = str(data['state']).lower().strip()
        season = str(data['season']).lower().strip()

        area = float(data['area'])

        # =========================
        # CROP RECOMMENDATION
        # =========================

        input_c = pd.DataFrame(
            [[
                n,
                p,
                k,
                temp,
                humidity,
                ph,
                rainfall
            ]],
            columns=[
                'N',
                'P',
                'K',
                'temperature',
                'humidity',
                'ph',
                'rainfall'
            ]
        )

        crop_prediction = classifier.predict(
            input_c
        )[0]

        # =========================
        # SAFE CROP HANDLING
        # =========================

        available_crops = encoders['crop'].classes_

        if crop_prediction not in available_crops:

            crop_prediction = available_crops[0]

        crop_encoded = encoders['crop'].transform(
            [crop_prediction]
        )[0]

        # =========================
        # SAFE STATE HANDLING
        # =========================

        available_states = encoders['State'].classes_

        if state not in available_states:

            state = available_states[0]

        state_encoded = encoders['State'].transform(
            [state]
        )[0]

        # =========================
        # SAFE SEASON HANDLING
        # =========================

        available_seasons = encoders['season'].classes_

        if season not in available_seasons:

            season = available_seasons[0]

        season_encoded = encoders['season'].transform(
            [season]
        )[0]

        # =========================
        # YIELD PREDICTION
        # =========================

        input_reg = pd.DataFrame([[
            crop_encoded,
            2026,
            season_encoded,
            state_encoded,
            area,
            0,
            0,
            n,
            p,
            k,
            ph,
            temp,
            rainfall,
            humidity
        ]], columns=X_reg_columns)

        yield_prediction = regressor.predict(
            input_reg
        )[0]

        # =========================
        # SMART SUGGESTION
        # =========================

        suggestion = generate_suggestion(
            temp,
            humidity,
            rainfall
        )

        # =========================
        # FINAL RESPONSE
        # =========================

        return jsonify({

            "recommended_crop":
                str(crop_prediction),

            "estimated_yield":
                round(float(yield_prediction), 2),

            "smart_suggestion":
                suggestion,

            "location":
                state.title(),

            "season":
                season.title(),

            "status":
                "success"
        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        })

# =========================
# MAIN
# =========================

if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )