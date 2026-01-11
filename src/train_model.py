import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/processed/weather_features.csv")

features = [
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "humidity_change",
    "pressure_drop"
]

X = df[features]
y = df["rain_next_30min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "src/rain_model.pkl")
print("Model saved")
