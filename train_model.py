import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Step 1: Generate a larger, more realistic synthetic dataset
np.random.seed(42)
n_samples = 300

rainfall = np.random.uniform(5, 300, n_samples)
soil_moisture = np.random.uniform(10, 95, n_samples)
slope_angle = np.random.uniform(5, 55, n_samples)

# Risk score combines the three factors with realistic weighting
# (rainfall and soil moisture matter most, slope angle amplifies risk)
risk_score = (
    (rainfall / 300) * 0.45 +
    (soil_moisture / 95) * 0.35 +
    (slope_angle / 55) * 0.20
)
# Add a little natural randomness/noise
risk_score += np.random.normal(0, 0.05, n_samples)

def score_to_label(score):
    if score < 0.35:
        return 'Low'
    elif score < 0.55:
        return 'Moderate'
    elif score < 0.75:
        return 'High'
    else:
        return 'Critical'

risk_level = [score_to_label(s) for s in risk_score]

df = pd.DataFrame({
    'rainfall': rainfall,
    'soil_moisture': soil_moisture,
    'slope_angle': slope_angle,
    'risk_level': risk_level
})

print("Dataset summary:")
print(df['risk_level'].value_counts())
print()

# Step 2: Split into features and target
X = df[['rainfall', 'soil_moisture', 'slope_angle']]
y = df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Save feature importance chart (for your PPT)
importances = model.feature_importances_
feature_names = ['Rainfall', 'Soil Moisture', 'Slope Angle']

plt.figure(figsize=(6, 4))
plt.bar(feature_names, importances, color=['#4285F4', '#34A853', '#FBBC05'])
plt.title('Feature Importance in Landslide Risk Prediction')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Saved feature_importance.png")

# Step 6: Save confusion matrix chart (for your PPT)
from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title('Model Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")

# Step 7: Test with a sample prediction
sample = [[170, 78, 42]]
prediction = model.predict(sample)
print(f"\nSample Prediction (rainfall=170, soil=78%, slope=42°): {prediction[0]}")

# Step 8: Save the trained model
joblib.dump(model, 'landslide_model.pkl')

# Step 9: Save accuracy for the app to display
with open('model_accuracy.txt', 'w') as f:
    f.write(str(round(accuracy * 100, 2)))

print("\nModel saved as landslide_model.pkl")