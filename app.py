from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        input_data = {
            "Staffed All Beds": float(request.form["staffed_all"]),
            "Staffed ICU Beds": float(request.form["icu_beds"]),
            "Licensed All Beds": float(request.form["licensed"]),
            "ICU Bed Occupancy Rate": float(request.form["icu_rate"]),
            "Population": float(request.form["population"]),
            "Population (20+)": float(request.form["pop_20"]),
            "Population (65+)": float(request.form["pop_65"]),
            "State": request.form["state"],
            "County Name": request.form["county"],
            "ICU Bed Source": request.form["icu_source"]
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
