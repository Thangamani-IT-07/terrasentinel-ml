import os
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('landslide_model.pkl')
with open('model_accuracy.txt', 'r') as f:
    model_accuracy = f.read()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    rainfall = data['rainfall']
    soil_moisture = data['soil_moisture']
    slope_angle = data['slope_angle']

    prediction = model.predict([[rainfall, soil_moisture, slope_angle]])

    # Determine the top contributing factor
    importances = model.feature_importances_
    values = [rainfall, soil_moisture, slope_angle]
    feature_names = ['Rainfall', 'Soil Moisture', 'Slope Angle']

    contributions = [importances[i] * values[i] for i in range(3)]
    top_index = contributions.index(max(contributions))
    top_factor = feature_names[top_index]

    return jsonify({
        'risk_level': prediction[0],
        'top_factor': top_factor,
        'model_accuracy': model_accuracy
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=True, port=port)