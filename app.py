from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ==========================================
# Load Trained Machine Learning Model
# ==========================================
model = pickle.load(open("models/flood_model.pkl", "rb"))


# ==========================================
# Home Page
# ==========================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction_text=None
    )


# ==========================================
# Prediction
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Read User Inputs
        Temp = float(request.form["Temp"])
        Humidity = float(request.form["Humidity"])
        Cloud_Cover = float(request.form["Cloud Cover"])
        ANNUAL = float(request.form["ANNUAL"])
        Jan_Feb = float(request.form["Jan-Feb"])
        Mar_May = float(request.form["Mar-May"])
        Jun_Sep = float(request.form["Jun-Sep"])
        Oct_Dec = float(request.form["Oct-Dec"])
        avgjune = float(request.form["avgjune"])
        sub = float(request.form["sub"])

        # Create Feature Array
        features = np.array([[
            Temp,
            Humidity,
            Cloud_Cover,
            ANNUAL,
            Jan_Feb,
            Mar_May,
            Jun_Sep,
            Oct_Dec,
            avgjune,
            sub
        ]])

        # Predict
        prediction = model.predict(features)

        # Prediction Message
        if prediction[0] == 1:
            prediction_text = "⚠️ Flood Risk Detected"
        else:
            prediction_text = "✅ No Flood Risk"

        # Return Page with User Inputs Preserved
        return render_template(
            "index.html",
            prediction_text=prediction_text,
            request=request
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text="❌ Error : " + str(e),
            request=request
        )


# ==========================================
# Run Flask Application
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)