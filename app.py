from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("placement_model.pkl", "rb"))


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    IQ = float(request.form["IQ"])
    Prev_Sem_Result = float(request.form["Prev_Sem_Result"])
    CGPA = float(request.form["CGPA"])
    Academic_Performance = float(request.form["Academic_Performance"])
    Internship_Experience = float(request.form["Internship_Experience"])
    Extra_Curricular_Score = float(request.form["Extra_Curricular_Score"])
    Communication_Skills = float(request.form["Communication_Skills"])
    Projects_Completed = float(request.form["Projects_Completed"])


    input_data = np.array([[
        IQ,
        Prev_Sem_Result,
        CGPA,
        Academic_Performance,
        Internship_Experience,
        Extra_Curricular_Score,
        Communication_Skills,
        Projects_Completed
    ]])


    prediction = model.predict(input_data)


    if prediction[0] == 1:
        result = "Student will be Placed"
    else:
        result = "Student will not be Placed"


    return render_template(
        "index.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)