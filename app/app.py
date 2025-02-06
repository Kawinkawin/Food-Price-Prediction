from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
#hello
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get input data from user
        rainfall = float(request.form["rainfall"])
        temperature = float(request.form["temperature"])
        inflation_rate = float(request.form["inflation_rate"])
        fuel_cost = float(request.form["fuel_cost"])
        supply = float(request.form["supply"])
        demand = float(request.form["demand"])

        # Prepare input for prediction
        input_data = np.array([[rainfall, temperature, inflation_rate, fuel_cost, supply, demand]])
        scaled_data = scaler.transform(input_data)

        # Predict price
        predicted_price = model.predict(scaled_data)[0]
        return render_template("index.html", predicted_price=round(predicted_price, 2))

    return render_template("index.html", predicted_price=None)

if __name__ == "__main__":
    app.run(debug=True)
