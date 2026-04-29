from pathlib import Path

from flask import Flask, render_template, request

from src.dl.predict import integrated_predict

ROOT = Path(__file__).resolve().parents[2]

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        user_text = request.form["ticket"]
        prediction = integrated_predict(user_text)

        urgency = prediction.get("urgency", 0.0)
        if urgency < 0.33:
            priority = "Low"
        elif urgency < 0.66:
            priority = "Medium"
        else:
            priority = "High"

        result = {
            "text": user_text,
            "dl_category": prediction.get("dl_category"),
            "category": prediction.get("category"),
            "urgency": round(urgency, 3),
            "priority": priority,
            "subject": prediction.get("subject"),
            "action": prediction.get("action"),
            "response": prediction.get("response"),
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
