from flask import Flask, render_template, request
import requests

app = Flask(__name__)

AI_URL = "http://ai:5000/predict"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    message = None  # új változó a visszajelzésnek

    if request.method == "POST":
        file = request.files["file"]

        try:
            response = requests.post(
                AI_URL,
                files={"file": (file.filename, file.stream, file.mimetype)},
                timeout=5  # 5 másodperces timeout
            )

            if response.ok:
                data = response.json()
                result = data.get("label")
                confidence = data.get("confidence")
            else:
                message = "Az AI szolgáltatás még nem áll készen, kérjük várjon."

        except requests.exceptions.RequestException:
            message = "Az AI szolgáltatás még nem áll készen, kérjük várjon."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        message=message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
