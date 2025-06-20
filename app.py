from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)
model = joblib.load("water_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            inputs = [float(request.form[f]) for f in [
                'pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
            ]]
            scaled = scaler.transform([inputs])
            prediction = model.predict(scaled)[0]
            result = "✅ Water is POTABLE (safe to drink)" if prediction == 1 else "❌ Water is NOT POTABLE"
            return render_template("index.html", result=result)
        except:
            return render_template("index.html", result="⚠️ Invalid input. Please enter numbers only.")
    return render_template("index.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
